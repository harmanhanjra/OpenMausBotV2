// Prime Agent driver — embeds the real prime-agent (pi, the self-improving
// agent with /refine) in-process via its SDK (createAgentSession) instead of
// spawning the CLI, whose RPC mode requires a daemon that times out on
// Windows. The NVIDIA NIM endpoint the nvidia driver uses is seeded into a
// per-data-dir models.json the SDK's ModelRegistry reads, so the same key
// that powers the nvidia instance drives prime-agent sessions.
//
// Integrations:
//   - agents (list_bots / ask_bot): mounted as in-process custom tools that
//     call the harness's /api/internal endpoints directly. No subprocess MCP
//     proxy is needed — prime-agent only supports HTTP MCP servers, and this
//     driver is in-process anyway (the harness owns turns/permissions/loops).
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

import { Type } from "typebox";
import type { AgentSession, ToolDefinition } from "prime-agent";

import { DATA_DIR } from "../config.ts";
import type {
  DriverCreateInput,
  ModelCatalog,
  ProviderDriver,
  ProviderInstance,
  ProviderSnapshot,
  RuntimeEvent,
  RuntimeEventListener,
  SendTurnInput,
} from "../contracts.ts";
import { newEventId, newId } from "../contracts.ts";
import { appendNative } from "./native.ts";
import { DEFAULT_KEY_ENV, DEFAULT_URL, discoverModels } from "./nvidia.ts";

const DRIVER_KIND = "primeAgent";
const DEFAULT_PROVIDER = "nvidia";
const DEFAULT_TOOLS = ["bash", "edit"];

// Curated seed for models.json — used immediately on first boot, then
// replaced in the background by the live /v1/models list for this key.
// The flagship first: the "default brain" is the best NVIDIA model reachable.
const DEFAULT_MODEL_PREFERENCE = [
  "nvidia/nemotron-3-ultra-550b-a55b",
  "openai/gpt-oss-120b",
  "z-ai/glm-5.2",
  "meta/llama-3.3-70b-instruct",
];

const FALLBACK_MODELS: ModelCatalog = {
  default: DEFAULT_MODEL_PREFERENCE[0],
  options: [
    { id: "nvidia/nemotron-3-ultra-550b-a55b", label: "Nemotron 3 Ultra 550B" },
    { id: "openai/gpt-oss-120b", label: "GPT-OSS 120B" },
    { id: "openai/gpt-oss-20b", label: "GPT-OSS 20B" },
    { id: "z-ai/glm-5.2", label: "GLM 5.2" },
    { id: "meta/llama-3.3-70b-instruct", label: "Llama 3.3 70B Instruct" },
    { id: "deepseek-ai/deepseek-v3.1", label: "DeepSeek V3.1" },
    { id: "moonshotai/kimi-k2.6", label: "Kimi K2.6" },
    { id: "nvidia/nemotron-3-super-120b-a12b", label: "Nemotron 3 Super 120B" },
    { id: "minimaxai/minimax-m3", label: "MiniMax M3" },
    { id: "thinkingmachines/inkling", label: "Inkling" },
  ],
};

export interface PrimeAgentConfig {
  provider: string;
  baseUrl: string;
  apiKeyEnv: string;
  /** optional override of the seeded model catalog */
  models?: ModelCatalog;
}

function decodeModels(raw: unknown) {
  if (raw && typeof raw === "object") {
    const o = raw as Record<string, unknown>;
    const options = Array.isArray(o.options)
      ? o.options
          .filter((x) => x && typeof x === "object")
          .map((x) => {
            const m = x as Record<string, unknown>;
            return { id: String(m.id ?? ""), label: String(m.label ?? String(m.id ?? "")) };
          })
          .filter((m) => m.id)
      : FALLBACK_MODELS.options;
    if (options.length) {
      const def = typeof o.default === "string" && options.some((m) => m.id === o.default) ? o.default : options[0].id;
      return { default: def, options };
    }
  }
  return undefined;
}

function decodeConfig(raw: unknown): PrimeAgentConfig {
  const o = (raw ?? {}) as Record<string, unknown>;
  return {
    provider: typeof o.provider === "string" && o.provider ? o.provider : DEFAULT_PROVIDER,
    baseUrl: typeof o.baseUrl === "string" && o.baseUrl ? o.baseUrl : DEFAULT_URL,
    apiKeyEnv: typeof o.apiKeyEnv === "string" && o.apiKeyEnv ? o.apiKeyEnv : DEFAULT_KEY_ENV,
    models: decodeModels(o.models),
  };
}

// ── prime-agent SDK (lazy, so a missing/broken package degrades the driver
// to an unavailable snapshot instead of taking the whole server down) ──
type PrimeSdk = typeof import("prime-agent");
let sdkPromise: Promise<PrimeSdk> | null = null;
function loadSdk(): Promise<PrimeSdk> {
  if (!sdkPromise) sdkPromise = import("prime-agent");
  return sdkPromise;
}

const AGENT_DIR = join(DATA_DIR, "prime-agent");
const MODELS_JSON = join(AGENT_DIR, "models.json");
const SESSIONS_DIR = join(AGENT_DIR, "sessions");

function threadAgentDir(threadId: string): string {
  const safe = threadId.replace(/[^a-zA-Z0-9._-]/g, "_").slice(0, 80);
  return join(AGENT_DIR, "threads", safe);
}

function hashString(s: string): string {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) >>> 0;
  return h.toString(36);
}

// Deterministic on-disk session file per thread. SessionManager persists each
// session as a JSONL file; pinning the thread's file lets a restarted server
// resume the conversation instead of starting over.
function threadSessionFile(threadId: string): string {
  return join(SESSIONS_DIR, `${hashString(threadId)}.jsonl`);
}

function resolvePrimeVersion(sdk: PrimeSdk): string | null {
  return typeof sdk.VERSION === "string" && sdk.VERSION ? sdk.VERSION : null;
}

// Shell for prime-agent's bash tool on Windows — it needs a POSIX shell, and
// Git Bash is the standard provider. On POSIX we let prime-agent resolve bash.
function windowsShellPath(): string | undefined {
  if (process.platform !== "win32") return undefined;
  for (const candidate of [
    "C:\\Program Files\\Git\\bin\\bash.exe",
    "C:\\Program Files\\Git\\usr\\bin\\bash.exe",
  ]) {
    if (existsSync(candidate)) return candidate;
  }
  return undefined;
}

export const PrimeAgentDriver: ProviderDriver<PrimeAgentConfig> = {
  driverKind: DRIVER_KIND,
  metadata: { displayName: "Prime Agent", supportsMultipleInstances: true },
  models: FALLBACK_MODELS,
  decodeConfig,
  defaultConfig: () => decodeConfig({}),

  async create(input: DriverCreateInput<PrimeAgentConfig>): Promise<ProviderInstance> {
    const { instanceId, config } = input;
    const apiKey = input.environment[config.apiKeyEnv] ?? process.env[config.apiKeyEnv] ?? "";
    const listeners = new Set<RuntimeEventListener>();

    // Mutable catalog surfaced to the model picker. Seeded with the curated
    // fallback, then refreshed from the endpoint's /v1/models so prime-agent
    // sees every model this key can actually call. Discovery runs in the
    // background — it must never slow boot or the create() contract rejects.
    const catalog: ModelCatalog = {
      default: (config.models ?? FALLBACK_MODELS).default,
      options: (config.models ?? FALLBACK_MODELS).options.map((o) => ({ ...o })),
    };

    mkdirSync(AGENT_DIR, { recursive: true });

    const preferredDefault = (options: Array<{ id: string }>) =>
      DEFAULT_MODEL_PREFERENCE.find((id) => options.some((o) => o.id === id));

    // Write models.json (and pick the default) from the current catalog.
    const writeModelsJson = (catalogToWrite: ModelCatalog) => {
      const models = catalogToWrite.options.map((o) => ({
        id: o.id,
        name: o.label,
        reasoning: /r1|oss|think|reason|ultra/i.test(o.id),
        input: ["text"],
        contextWindow: 128_000,
        maxTokens: 32_768,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      }));
      const payload = {
        providers: {
          [config.provider]: {
            baseUrl: config.baseUrl,
            api: "openai-completions",
            apiKey: config.apiKeyEnv,
            compat: { supportsDeveloperRole: false, supportsReasoningEffort: false },
            models,
          },
        },
      };
      writeFileSync(MODELS_JSON, JSON.stringify(payload, null, 2));
    };

    writeModelsJson(catalog);

    // Shared per-instance infra for every session this instance creates.
    const sdk = await loadSdk();
    const authStorage = sdk.AuthStorage.create(join(AGENT_DIR, "auth.json"));
    authStorage.setRuntimeApiKey(config.provider, apiKey);
    const modelRegistry = sdk.ModelRegistry.create(authStorage, MODELS_JSON);
    const shellPath = windowsShellPath();
    const settingsManager = sdk.SettingsManager.inMemory({
      defaultThinkingLevel: "off",
      ...(shellPath ? { shellPath } : {}),
    });

    // Background refresh AFTER the SDK is loaded. prime-agent's first import
    // blocks the event loop for several seconds; a /v1/models fetch started
    // before that would be aborted on its timeout mid-block and the catalog
    // would stay frozen at the curated seed. Starting here lets the discovery
    // actually complete and replace the seed with the live model list.
    if (apiKey && !config.models) {
      console.log("[primeAgent] Starting background discovery for instance", instanceId, "baseUrl:", config.baseUrl);
      void discoverModels(config.baseUrl, apiKey).then((next) => {
        console.log("[primeAgent] Discovery resolved for", instanceId, "models:", next ? next.options.length : 0);
        if (!next || !next.options.length) {
          console.log("[primeAgent] Discovery returned empty/nil, keeping seed.");
          return;
        }
        const def = preferredDefault(next.options) ?? next.default;
        console.log("[primeAgent] Updating catalog for", instanceId, "to", next.options.length, "models, default:", def);
        catalog.options.splice(0, catalog.options.length, ...next.options.map((o) => ({ ...o })));
        catalog.default = def;
        writeModelsJson(catalog);
        modelRegistry.refresh();
        console.log("[primeAgent] Catalog refreshed and file written for", instanceId);
      }).catch((e: any) => {
        console.error("[primeAgent] Discovery FAILED for", instanceId, e?.message ?? String(e));
      });
    }

    const emit = (event: RuntimeEvent) => {
      for (const l of [...listeners]) l(event);
    };
    const base = (threadId: string, turnId: string) => ({
      eventId: newEventId(),
      provider: DRIVER_KIND,
      providerInstanceId: instanceId,
      threadId,
      turnId,
      createdAt: new Date().toISOString(),
    });

    const resolveModel = (modelId: string) => {
      modelRegistry.refresh();
      return (
        modelRegistry.find(config.provider, modelId) ??
        modelRegistry.find(config.provider, catalog.default) ??
        modelRegistry.getAll()[0]
      );
    };

    interface SessionEntry {
      session: AgentSession;
      unsub: () => void;
      modelId: string;
      sessionId: string;
    }
    interface TurnState {
      turnId: string;
      settled: boolean;
      aborted: boolean;
      buf: string;
      stopReason: string | null;
    }

    const sessions = new Map<string, SessionEntry>();
    const sessionForThread = new Map<string, string>();
    const active = new Map<string, TurnState>();
    // Sessions are keyed by threadId AND persona: a meeting room shares one
    // threadId across many bots, and each speaker must get its own session
    // (own system-prompt.md, own model, own context) — not the first
    // speaker's. The same bot reuses its session across turns.
    const sessionKey = (threadId: string, system: string | undefined) =>
      `${threadId}::${hashString(system ?? "")}`;

    // ── peer-agent tools (list_bots / ask_bot) ────────────────────────────
    // Mirrors server/drivers/agents-proxy.ts but in-process: the harness
    // hands us the spawn contract's env (OMB_*) and we call the same
    // localhost endpoints the proxy would, with the same bearer token.
    const buildPeerTools = (env: Record<string, string>): ToolDefinition[] => {
      const harnessUrl = env.OMB_HARNESS_URL ?? `http://127.0.0.1:8799`;
      const botId = env.OMB_BOT_ID ?? "";
      const token = env.OMB_COMMS_TOKEN ?? "";
      const depth = Number(env.OMB_TURN_DEPTH ?? "0") || 0;
      const call = async (path: string, init?: RequestInit): Promise<Record<string, unknown>> => {
        const res = await fetch(harnessUrl + path, {
          ...init,
          headers: {
            "content-type": "application/json",
            authorization: `Bearer ${token}`,
            ...(init?.headers ?? {}),
          },
        });
        const body = (await res.json().catch(() => ({}))) as Record<string, unknown>;
        if (!res.ok) throw new Error(String(body.error ?? `HTTP ${res.status}`));
        return body;
      };
      return [
        {
          name: "list_bots",
          label: "list_bots",
          description:
            "List the other bots (agents) in this OpenMausBot workspace you can message, with their model and whether they're busy. Call this before ask_bot to discover who's available.",
          parameters: Type.Object({}, { additionalProperties: false }),
          async execute() {
            const r = await call(`/api/internal/agents?self=${encodeURIComponent(botId)}`);
            const bots = Array.isArray(r.bots) ? (r.bots as Array<Record<string, unknown>>) : [];
            if (!bots.length) {
              return { content: [{ type: "text", text: "No other bots in this workspace yet." }], details: null };
            }
            const lines = bots.map((b) => {
              const role = b.title ? ` — ${b.title}` : "";
              const about = b.description ? ` (${String(b.description).slice(0, 120)})` : "";
              return `- ${b.name}${role}${about} [id: ${b.id}, model: ${b.model}${b.busy ? ", busy" : ""}]`;
            });
            return {
              content: [{ type: "text", text: `Other bots you can message with ask_bot:\n${lines.join("\n")}` }],
              details: null,
            };
          },
        },
        {
          name: "ask_bot",
          label: "ask_bot",
          description:
            "Send a message to another bot in this workspace and wait for its reply. Use it to delegate a subtask to a specialist bot or ask a peer a question. The other bot runs a full turn under its own model and permissions; the reply is returned to you as text. Returns promptly with a note if that bot is busy.",
          parameters: Type.Object(
            {
              bot_id: Type.String({ description: "The target bot's id (from list_bots)." }),
              message: Type.String({ description: "What to say / ask the bot." }),
            },
            { additionalProperties: false },
          ),
          async execute(_id, params: { bot_id?: string; message?: string }) {
            const toBotId = String(params.bot_id ?? "").trim();
            const message = String(params.message ?? "").trim();
            if (!toBotId || !message) {
              return { content: [{ type: "text", text: "ask_bot needs bot_id and message." }], details: null };
            }
            const r = await call(`/api/internal/ask-bot`, {
              method: "POST",
              body: JSON.stringify({ fromBotId: botId, toBotId, message, depth }),
            });
            if (r.busy) {
              return {
                content: [{ type: "text", text: "That bot is busy right now — try again after it finishes." }],
                details: null,
              };
            }
            if (r.error) {
              return {
                content: [{ type: "text", text: `Couldn't reach that bot: ${r.error}` }],
                details: null,
              };
            }
            return {
              content: [{ type: "text", text: `${r.botName ?? "Bot"} replied:\n${r.text ?? "(no reply)"}` }],
              details: null,
            };
          },
        },
      ];
    };

    const ensureSession = async (threadId: string, turn: SendTurnInput): Promise<SessionEntry> => {
      const key = sessionKey(threadId, turn.system);
      const existing = sessions.get(key);
      if (existing) {
        sessionForThread.set(threadId, key);
        return existing;
      }

      const agentsEnv = turn.integrations?.agents?.env;
      const peerTools = agentsEnv ? buildPeerTools(agentsEnv) : [];
      const toolNames = [...DEFAULT_TOOLS, ...peerTools.map((t) => t.name)];

      const modelId = turn.model ?? catalog.default;
      const model = resolveModel(modelId);
      if (!model) {
        throw new Error(
          `no model available for provider "${config.provider}" (catalog default: ${catalog.default}) — check models.json at ${MODELS_JSON}`,
        );
      }

      // The persona lives in agentDir/system-prompt.md — the SDK's resource
      // loader reads it once at session start. Sessions are per-thread, so a
      // per-thread agent dir keeps each bot's persona and session files
      // isolated from every other thread sharing this instance.
      const threadDir = threadAgentDir(threadId);
      mkdirSync(threadDir, { recursive: true });
      if (turn.system) {
        writeFileSync(join(threadDir, "system-prompt.md"), turn.system, "utf8");
      }

      // Disk-backed per-thread session (resume the thread's file on restart).
      // prime-agent's self-improvement loop (/refine auto-refine, enabled by
      // default) only runs for persisted sessions — in-memory sessions have no
      // artifact dir, so auto-refine would silently never trigger.
      const cwd = turn.cwd ?? homedir();
      mkdirSync(SESSIONS_DIR, { recursive: true });
      const sessionFile = threadSessionFile(threadId);
      const sessionManager = existsSync(sessionFile)
        ? sdk.SessionManager.open(sessionFile, SESSIONS_DIR, cwd)
        : sdk.SessionManager.create(cwd, SESSIONS_DIR);
      if (!existsSync(sessionFile)) {
        sessionManager.setSessionFile(sessionFile);
      }

      const { session } = await sdk.createAgentSession({
        cwd,
        agentDir: threadDir,
        model,
        thinkingLevel: "off",
        tools: toolNames,
        customTools: peerTools,
        authStorage,
        modelRegistry,
        settingsManager,
        sessionManager,
      });

      const entry: SessionEntry = {
        session,
        modelId,
        sessionId: session.sessionId,
        unsub: () => {},
      };

      entry.unsub = session.subscribe((event: any) => {
        const st = active.get(threadId);
        if (!st) return;
        switch (event.type) {
          case "message_start":
            st.buf = "";
            break;
          case "message_update": {
            const e = event.assistantMessageEvent;
            if (e?.type === "text_delta" && typeof e.delta === "string" && e.delta) {
              st.buf += e.delta;
              emit({ ...base(threadId, st.turnId), type: "content.delta", streamKind: "assistant_text", delta: e.delta });
            } else if (e?.type === "thinking_delta" && typeof e.delta === "string" && e.delta) {
              emit({ ...base(threadId, st.turnId), type: "content.delta", streamKind: "reasoning_text", delta: e.delta });
            }
            break;
          }
          case "message_end": {
            const text = st.buf;
            st.buf = "";
            if (text.trim()) {
              emit({ ...base(threadId, st.turnId), type: "item.completed", itemType: "assistant_text", text });
            }
            break;
          }
          case "tool_execution_start":
            emit({
              ...base(threadId, st.turnId),
              type: "item.started",
              itemType: "tool",
              itemId: event.toolCallId,
              title: event.toolName,
            });
            break;
          case "tool_execution_end":
            emit({
              ...base(threadId, st.turnId),
              type: "item.completed",
              itemType: "tool",
              itemId: event.toolCallId,
              ok: event.isError !== true,
            });
            break;
          case "turn_end":
            st.stopReason = event.message?.stopReason ?? st.stopReason;
            break;
          default:
            break;
        }
      });

      sessionForThread.set(threadId, key);
      sessions.set(key, entry);
      return entry;
    };

    const settle = (threadId: string, ok: boolean, stopReason: string | null, errorMessage?: string) => {
      const st = active.get(threadId);
      if (!st || st.settled) return;
      st.settled = true;
      active.delete(threadId);
      if (errorMessage) {
        emit({ ...base(threadId, st.turnId), type: "runtime.error", message: errorMessage });
      }
      emit({
        ...base(threadId, st.turnId),
        type: "turn.completed",
        ok,
        stopReason,
        cost: null,
      });
    };

    const sendTurn = async (turn: SendTurnInput) => {
      const { threadId } = turn;
      if (active.has(threadId)) throw new Error("a turn is already running on this thread");
      const turnId = newId();
      const st: TurnState = { turnId, settled: false, aborted: false, buf: "", stopReason: null };
      active.set(threadId, st);

      emit({ ...base(threadId, turnId), type: "turn.started" });

      try {
        const entry = await ensureSession(threadId, turn);
        const requested = turn.model ?? catalog.default;
        if (entry.modelId !== requested) {
          const target = resolveModel(requested);
          if (!target) throw new Error(`model not found in prime-agent registry: ${requested}`);
          await entry.session.setModel(target);
          entry.modelId = requested;
        }
        appendNative(threadId, { dir: "out", source: "prime-agent.prompt", msg: { model: requested, text: turn.text } });
        await entry.session.prompt(turn.text);
        const stopReason = st.stopReason ?? "end_turn";
        appendNative(threadId, { dir: "in", source: "prime-agent.turn", msg: { stopReason } });
        settle(threadId, true, stopReason);
      } catch (e) {
        const aborted = st.aborted;
        if (!aborted) {
          appendNative(threadId, { dir: "in", source: "prime-agent.turn", msg: { error: (e as Error).message } });
        }
        settle(threadId, false, aborted ? "interrupted" : "error", aborted ? undefined : (e as Error).message);
      }

      return { turnId };
    };

    const abortThread = (threadId: string) => {
      const st = active.get(threadId);
      const entryKey = sessionForThread.get(threadId);
      const entry = entryKey ? sessions.get(entryKey) : undefined;
      if (!st || !entry) return;
      st.aborted = true;
      entry.session.requestAbort();
      void entry.session.abort().catch(() => {});
    };

    const snapshot = async (): Promise<ProviderSnapshot> => {
      let sdk: PrimeSdk;
      try {
        sdk = await loadSdk();
      } catch (e) {
        return { state: "unavailable", reason: `prime-agent SDK failed to load: ${(e as Error).message}` };
      }
      if (apiKey) {
        return { state: "available", authenticated: true, version: resolvePrimeVersion(sdk) };
      }
      return {
        state: "unavailable",
        reason: `no NVIDIA key — add {"nvidia":{"key":"nvapi-…"}} to ~/.openmausbot/config.json or set ${config.apiKeyEnv}`,
      };
    };

    return {
      instanceId,
      driverKind: DRIVER_KIND,
      displayName: input.displayName,
      enabled: input.enabled,
      models: catalog,
      snapshot,
      adapter: {
        provider: DRIVER_KIND,
        capabilities: { sessionModelSwitch: "in-session", agentsMcp: true },
        sendTurn,
        interruptTurn: async (threadId) => abortThread(threadId),
        respondToRequest: async () => {
          throw new Error("primeAgent driver has no pending asks");
        },
        hasSession: (threadId) => active.has(threadId) || sessionForThread.has(threadId),
        stopAll: async () => {
          for (const threadId of [...active.keys()]) abortThread(threadId);
        },
        onEvent: (listener) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
      generateText: async (prompt: string) => {
        const small =
          catalog.options.find((o) => /flash|8b|mini|nano|20b|lightning/i.test(o.id)) ?? catalog.options[0];
        const model = resolveModel(small.id);
        if (!model) throw new Error("no model available for generateText");
        const { session } = await sdk.createAgentSession({
          cwd: homedir(),
          agentDir: AGENT_DIR,
          model,
          thinkingLevel: "off",
          tools: [],
          authStorage,
          modelRegistry,
          settingsManager,
          sessionManager: sdk.SessionManager.inMemory(),
        });
        let text = "";
        const unsub = session.subscribe((event: any) => {
          if (event.type === "message_update" && event.assistantMessageEvent?.type === "text_delta") {
            text += event.assistantMessageEvent.delta;
          }
        });
        await session.prompt(prompt);
        unsub();
        await session.disposeAsync().catch(() => {});
        return text;
      },
      dispose: async () => {
        for (const threadId of [...active.keys()]) abortThread(threadId);
        for (const entry of [...sessions.values()]) {
          entry.unsub();
          await entry.session.disposeAsync().catch(() => {});
        }
        sessions.clear();
        sessionForThread.clear();
        listeners.clear();
      },
    };
  },
};