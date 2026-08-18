// Prime Agent driver - embeds the real prime-agent (pi, the self-improving
// agent with /refine) in-process via its SDK (createAgentSession) instead of
// spawning the CLI, whose RPC mode requires a daemon that times out on
// Windows. The NVIDIA NIM endpoint the nvidia driver uses is seeded into a
// per-data-dir models.json the SDK's ModelRegistry reads, so the same key
// that powers the nvidia instance drives prime-agent sessions.
//
// Integrations:
//   - agents (list_bots / ask_bot): mounted as in-process custom tools that
//     call the harness's /api/internal endpoints directly. No subprocess MCP
//     proxy is needed - prime-agent only supports HTTP MCP servers, and this
//     driver is in-process anyway (the harness owns turns/permissions/loops).
//   - computer (the bot's cloud computer): mounted as in-process custom
//     tools backed by the same ComputerClient server/computer-proxy.ts uses,
//     so prime-agent bots get the same CUA screen (click/type/scroll), shell
//     exec and Chrome tools the claude bots get - with screenshots returned
//     as image blocks in the tool result.
//   - memory (per-bot long-term facts + the swarm knowledge base): facts
//     are injected into the system prompt each session so accumulated
//     knowledge is in context from turn one, and remember/forget/recall +
//     knowledge_read/write/list tools let the agent update them live.
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

import { Type } from "typebox";
import type { AgentSession } from "prime-agent";

import { DATA_DIR } from "../config.ts";
import { ComputerClient, shellQuote } from "../computer-client.ts";
import type { Frame } from "../computer-client.ts";
import { memoryStore } from "../memory.ts";
import { memoryRAG } from "../memory-rag.ts";
import { toolsRegistry, type ToolDefinition } from "../tools-registry.ts";
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
import { DEFAULT_KEY_ENV, DEFAULT_URL, discoverModels } from "./nvidia-shared.ts";

const DRIVER_KIND = "primeAgent";
const DEFAULT_PROVIDER = "nvidia";
const DEFAULT_TOOLS = ["bash", "edit"];

// Curated seed for models.json - used immediately on first boot, then
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

// Shell for prime-agent's bash tool on Windows - it needs a POSIX shell, and
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


// -- peer-agent tools (list_bots / ask_bot) ----------------------------
// Mirrors server/drivers/agents-proxy.ts but in-process: the harness hands
// us the spawn contract's env (OMB_*) and we call the same localhost
// endpoints the proxy would, with the same bearer token. When OMB_BOT_ROLE
// is "ceo", the bot also gets governance tools (monitor_agents /
// set_bot_model / interrupt_bot) so it can watch the team and switch a
// non-responsive bot to another model.
export function buildPeerTools(env: Record<string, string>): ToolDefinition[] {
  const harnessUrl = env.OMB_HARNESS_URL ?? `http://127.0.0.1:8799`;
  const botId = env.OMB_BOT_ID ?? "";
  const token = env.OMB_COMMS_TOKEN ?? "";
  const role = env.OMB_BOT_ROLE ?? "";
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
  const isCeo = role === "ceo";
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
          const role = b.title ? ` - ${b.title}` : "";
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
            content: [{ type: "text", text: "That bot is busy right now - try again after it finishes." }],
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
    ...(isCeo
      ? ([
          {
            name: "monitor_agents",
            label: "monitor_agents",
            description:
              "As the CEO bot: survey every other bot in the workspace - their model, whether they're busy, how long they've been working, when they last finished a turn, and whether any looks stuck. Use this on a regular watch (a scheduler can wake you every 10 minutes) to find agents that stopped responding.",
            parameters: Type.Object({}, { additionalProperties: false }),
            async execute() {
              const r = await call(`/api/internal/agents?self=${encodeURIComponent(botId)}`);
              const bots = Array.isArray(r.bots) ? (r.bots as Array<Record<string, unknown>>) : [];
              if (!bots.length) {
                return {
                  content: [{ type: "text", text: "No other bots to monitor - the workspace has only you." }],
                  details: null,
                };
              }
              const lines = bots.map((b) => {
                const busy = b.busy
                  ? `busy ${b.busySince ? `${Math.round((Date.now() - Number(b.busySince)) / 1000)}s` : ""}`
                  : "idle";
                const last = b.lastActivityAt
                  ? `${Math.round((Date.now() - Number(b.lastActivityAt)) / 60_000)}m ago`
                  : "never";
                return `- ${b.name} [id: ${b.id}] model=${b.model} role=${b.role ?? "worker"} state=${busy} last_turn=${last}${b.stuck ? " ⚠ STUCK (busy too long)" : ""}`;
              });
              return {
                content: [
                  {
                    type: "text",
                    text: `Team status:\n${lines.join("\n")}\n\nIf an agent is STUCK or busy far past normal, call set_bot_model to switch it to a healthy model, then interrupt_bot to stop its hung turn.`,
                  },
                ],
                details: null,
              };
            },
          },
          {
            name: "set_bot_model",
            label: "set_bot_model",
            description:
              "As the CEO bot: switch another bot to a different model (and provider instance) - e.g. when it stopped responding on its current one. The target's next turn uses the new model. Confirm the instance id + model id exist (list them via the models endpoint or your team status) before calling.",
            parameters: Type.Object(
              {
                bot_id: Type.String({ description: "The target bot's id (from monitor_agents)." }),
                instance_id: Type.String({ description: "The provider instance id, e.g. primeAgent or claude." }),
                model: Type.String({ description: "The model id on that instance, e.g. nvidia/nemotron-3-ultra-550b-a55b." }),
              },
              { additionalProperties: false },
            ),
            async execute(_id, params: { bot_id?: string; instance_id?: string; model?: string }) {
              const r = await call(`/api/internal/set-bot-model`, {
                method: "POST",
                body: JSON.stringify({
                  fromBotId: botId,
                  toBotId: String(params.bot_id ?? ""),
                  instanceId: String(params.instance_id ?? ""),
                  model: String(params.model ?? ""),
                }),
              });
              if (r.error) {
                return { content: [{ type: "text", text: `Couldn't change the model: ${r.error}` }], details: null };
              }
              const sel = (r.modelSelection ?? {}) as Record<string, unknown>;
              return {
                content: [
                  {
                    type: "text",
                    text: `Changed bot ${String(params.bot_id ?? "")} to model ${sel.model} on instance ${sel.instanceId}. Its next turn will use the new model.`,
                  },
                ],
                details: null,
              };
            },
          },
          {
            name: "interrupt_bot",
            label: "interrupt_bot",
            description:
              "As the CEO bot: force-stop a hung turn on another bot (e.g. after set_bot_model, or when it's been stuck for minutes). The bot goes idle and can take a fresh turn on its new model.",
            parameters: Type.Object(
              { bot_id: Type.String({ description: "The target bot's id (from monitor_agents)." }) },
              { additionalProperties: false },
            ),
            async execute(_id, params: { bot_id?: string }) {
              const r = await call(`/api/internal/interrupt-bot`, {
                method: "POST",
                body: JSON.stringify({ fromBotId: botId, toBotId: String(params.bot_id ?? "") }),
              });
              if (r.error) {
                return { content: [{ type: "text", text: `Couldn't interrupt: ${r.error}` }], details: null };
              }
              return {
                content: [{ type: "text", text: `Interrupted bot ${String(params.bot_id ?? "")}. It is now idle.` }],
                details: null,
              };
            },
          },
        ] as ToolDefinition[])
      : []),
  ];
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
    // background - it must never slow boot or the create() contract rejects.
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
    // (own system-prompt.md, own model, own context) - not the first
    // speaker's. The same bot reuses its session across turns.
    const sessionKey = (threadId: string, system: string | undefined) =>
      `${threadId}::${hashString(system ?? "")}`;


    // ── cloud computer tools (screenshot/click/type/exec/Chrome) ─────────
    // Same box logic as server/computer-proxy.ts (shared ComputerClient) but
    // framed as prime-agent custom tools: each returns AgentToolResult with
    // the screenshot as an image block when the screen changed.
    const buildComputerTools = (boxId: string, token: string): ToolDefinition[] => {
      const client = new ComputerClient(boxId, token);
      const observe = Type.Optional(
        Type.Boolean({
          description:
            "default true - return a fresh screenshot with the result. Set false only when chaining mechanical steps you don't need to see.",
        }),
      );
      const settleMs = Type.Optional(
        Type.Number({ description: "wait before the screenshot, default 350, max 3000" }),
      );
      const observeProps = { observe, settle_ms: settleMs };

      // Returns text + (image when the screen actually changed). Mirrors
      // the proxy's observed() - same dedup rules so the model isn't fed
      // byte-identical frames over and over.
      const toResult = (note: string, frame: Frame | null) => {
        if (!frame) {
          return { content: [{ type: "text" as const, text: note }], details: null };
        }
        const obs = client.observedContent(note, frame);
        if (!obs.image) {
          return { content: [{ type: "text" as const, text: obs.text }], details: null };
        }
        return {
          content: [
            { type: "text" as const, text: obs.text },
            { type: "image" as const, data: obs.image.data, mimeType: obs.image.mime },
          ],
          details: null,
        };
      };

      return [
        {
          name: "screenshot",
          label: "screenshot",
          description:
            "See the bot's cloud computer screen (returns an image). The desktop runs Chrome and a full Linux GUI. You usually do NOT need this after acting - click, type_text, press_key, scroll and open_url already return the resulting screen.",
          parameters: Type.Object({}, { additionalProperties: false }),
          async execute() {
            const frame = await client.screenshotFrame();
            if (!frame) {
              return { content: [{ type: "text", text: "screenshot failed: could not capture a frame" }], details: null };
            }
            return { content: [{ type: "image", data: frame.data, mimeType: frame.mime }], details: null };
          },
        },
        {
          name: "click",
          label: "click",
          description:
            "Click on the computer's screen and return the resulting screen. Use pixel coordinates exactly as they appear in the last frame you were given - any scaling to the real display is handled for you.",
          parameters: Type.Object(
            {
              x: Type.Number(),
              y: Type.Number(),
              button: Type.Optional(Type.Union([Type.Literal("left"), Type.Literal("right")], { description: "default left" })),
              double: Type.Optional(Type.Boolean({ description: "double-click" })),
              ...observeProps,
            },
            { additionalProperties: false },
          ),
          async execute(_id, params) {
            const p = params as { x?: number; y?: number; button?: string; double?: boolean };
            const x = Math.round(Number(p.x));
            const y = Math.round(Number(p.y));
            if (!Number.isFinite(x) || !Number.isFinite(y)) {
              return { content: [{ type: "text", text: "click needs numeric x,y" }], details: null };
            }
            const what = `${p.double ? "double-clicked" : p.button === "right" ? "right-clicked" : "clicked"} ${x},${y}`;
            const { acted, frame, stderr } = await client.actAndObserve([{ ...p, action: "click" }], p);
            const note = acted ? what : `${what}\n(the action reported an error: ${stderr.slice(0, 160) || "no detail"})`;
            return toResult(note, frame);
          },
        },
        {
          name: "type_text",
          label: "type_text",
          description: "Type text at the current focus and return the resulting screen.",
          parameters: Type.Object({ text: Type.String(), ...observeProps }, { additionalProperties: false }),
          async execute(_id, params) {
            const t = String((params as { text?: string }).text ?? "");
            if (!t) return { content: [{ type: "text", text: "nothing to type" }], details: null };
            const { acted, frame, stderr } = await client.actAndObserve([{ action: "type_text", text: t }], params, 120_000);
            const note = acted ? `typed ${t.length} chars` : `typing failed: ${stderr.slice(0, 160) || "no detail"}`;
            return toResult(note, frame);
          },
        },
        {
          name: "press_key",
          label: "press_key",
          description:
            'Press a key or chord and return the resulting screen. xdotool syntax: "Return", "Tab", "ctrl+c", "alt+F4", "ctrl+shift+t".',
          parameters: Type.Object({ keys: Type.String(), ...observeProps }, { additionalProperties: false }),
          async execute(_id, params) {
            const keys = String((params as { keys?: string }).keys ?? "").replace(/[^\w+]/g, "");
            if (!keys) return { content: [{ type: "text", text: "press_key needs keys" }], details: null };
            const { acted, frame, stderr } = await client.actAndObserve([{ action: "press_key", keys }], params);
            const note = acted ? `pressed ${keys}` : `keypress failed: ${stderr.slice(0, 160) || "no detail"}`;
            return toResult(note, frame);
          },
        },
        {
          name: "scroll",
          label: "scroll",
          description: "Scroll the screen up or down by N clicks and return the resulting screen.",
          parameters: Type.Object(
            {
              direction: Type.Union([Type.Literal("up"), Type.Literal("down")]),
              clicks: Type.Optional(Type.Number({ description: "default 3" })),
              ...observeProps,
            },
            { additionalProperties: false },
          ),
          async execute(_id, params) {
            const p = params as { direction?: string; clicks?: number };
            const clicks = Math.min(Math.max(Math.round(Number(p.clicks) || 3), 1), 20);
            const direction = p.direction === "up" ? "up" : "down";
            const { acted, frame, stderr } = await client.actAndObserve(
              [{ action: "scroll", direction, clicks }],
              params,
            );
            const note = acted
              ? `scrolled ${direction} ${clicks}`
              : `scroll failed: ${stderr.slice(0, 160) || "no detail"}`;
            return toResult(note, frame);
          },
        },
        {
          name: "computer_batch",
          label: "computer_batch",
          description:
            "Run several UI actions in ONE go and return the screen at the end - much faster than separate calls (one round trip, one screenshot). Use it for mechanical sequences you can predict without looking in between, e.g. click a field, type, Tab, type, press Return. Stop the batch before anything whose outcome you need to see first.",
          parameters: Type.Object(
            {
              actions: Type.Array(
                Type.Object({
                  action: Type.Union([
                    Type.Literal("click"),
                    Type.Literal("type_text"),
                    Type.Literal("press_key"),
                    Type.Literal("scroll"),
                    Type.Literal("wait"),
                  ]),
                  x: Type.Optional(Type.Number()),
                  y: Type.Optional(Type.Number()),
                  button: Type.Optional(Type.Union([Type.Literal("left"), Type.Literal("right")])),
                  double: Type.Optional(Type.Boolean()),
                  text: Type.Optional(Type.String()),
                  keys: Type.Optional(Type.String()),
                  direction: Type.Optional(Type.Union([Type.Literal("up"), Type.Literal("down")])),
                  clicks: Type.Optional(Type.Number()),
                  ms: Type.Optional(Type.Number({ description: "wait: milliseconds, max 5000" })),
                }),
              ),
              ...observeProps,
            },
            { additionalProperties: false },
          ),
          async execute(_id, params) {
            const p = params as { actions?: Array<Record<string, unknown>> };
            const actions = (Array.isArray(p.actions) ? p.actions : []).slice(0, 24);
            if (!actions.length) {
              return { content: [{ type: "text", text: "computer_batch needs a non-empty actions array" }], details: null };
            }
            const summary = actions
              .map((a) =>
                a.action === "click"
                  ? `click ${Math.round(Number(a.x))},${Math.round(Number(a.y))}`
                  : a.action === "type_text"
                    ? `type ${String(a.text ?? "").length} chars`
                    : a.action === "press_key"
                      ? `key ${a.keys}`
                      : a.action === "scroll"
                        ? `scroll ${a.direction ?? "down"}`
                        : `wait ${Math.min(Number(a.ms) || 500, 5000)}ms`,
              )
              .join(" → ");
            const { acted, frame, stderr } = await client.actAndObserve(actions, params, 180_000);
            const note = acted
              ? `ran ${actions.length} actions: ${summary}`
              : `batch reported an error: ${stderr.slice(0, 160) || "no detail"}`;
            return toResult(note, frame);
          },
        },
        {
          name: "computer_exec",
          label: "computer_exec",
          description:
            "Run a shell command on the bot's cloud computer (Linux, passwordless sudo, X11 desktop). Returns stdout/stderr/exit code - and, unlike the UI tools, no screenshot unless you ask for one.",
          parameters: Type.Object(
            {
              command: Type.String(),
              observe: Type.Optional(
                Type.Boolean({
                  description: "default false - set true to also return a screenshot (e.g. after launching a GUI app)",
                }),
              ),
            },
            { additionalProperties: false },
          ),
          async execute(_id, params) {
            const p = params as { command?: string; observe?: boolean };
            const command = String(p.command ?? "").slice(0, 4000);
            const out = await client.runOnBox(command, 120_000);
            const note = `exit ${out.exitCode}\n${out.stdout.slice(-6000)}${out.stderr ? `\n[stderr]\n${out.stderr.slice(-2000)}` : ""}`;
            if (p.observe !== true) return { content: [{ type: "text", text: note }], details: null };
            const frame = await client.screenshotFrame();
            return toResult(note, frame);
          },
        },
        {
          name: "open_url",
          label: "open_url",
          description: "Open a URL in the computer's own Chrome and return the resulting screen.",
          parameters: Type.Object({ url: Type.String(), ...observeProps }, { additionalProperties: false }),
          async execute(_id, params) {
            const url = String((params as { url?: string }).url ?? "");
            if (!/^https?:\/\//.test(url)) {
              return { content: [{ type: "text", text: "only http(s) URLs" }], details: null };
            }
            const q = shellQuote(url.replace(/'/g, "%27"));
            const observe = (params as { observe?: boolean }).observe !== false;
            const command = [
              "export DISPLAY=${DISPLAY:-:0}",
              client.geometryShell,
              `(google-chrome ${q} || chromium ${q} || chromium-browser ${q} || xdg-open ${q}) >/dev/null 2>&1 &`,
              'for i in 1 2 3 4 5 6 7 8 9 10 11 12; do xdotool search --onlyvisible --class "chrom" >/dev/null 2>&1 && break; sleep 0.25; done',
              observe ? client.captureBlock(600) : "true",
            ].join("; ");
            const out = await client.runOnBox(command, 60_000);
            if (!observe) return { content: [{ type: "text", text: `opened ${url}` }], details: null };
            const frame = await client.frameFrom(out);
            return toResult(`opened ${url}`, frame);
          },
        },
      ];
    };

    // ── memory tools (remember / recall / knowledge base) ─────────────────
    const buildMemoryTools = (threadId: string): ToolDefinition[] => {
      const facts = () => memoryStore.text(threadId);
      
      // Local wrappers to help TypeScript infer types correctly
      const addDocuments = async (docs: Array<{ content: string; metadata?: Record<string, unknown> }>) => 
        memoryRAG.addDocuments(threadId, docs);
      const hybridSearch = async (query: string, opts?: { topK?: number; threshold?: number }) => 
        memoryRAG.hybridSearch(threadId, query, opts);
      const ingestFile = async (path: string, content: string, metadata?: Record<string, unknown>) => 
        memoryRAG.ingestFile(threadId, path, content, metadata);

      const baseTools: ToolDefinition[] = [
        {
          name: "remember",
          label: "remember",
          description:
            "Store a fact about the user, a project, or anything you want to remember across conversations and restarts. Facts are injected into your context at the start of every future session, so use it for durable knowledge you'll want later - not for one-off trivia.",
          parameters: Type.Object({ fact: Type.String() }, { additionalProperties: false }),
          async execute(_id, params) {
            const text = String((params as { fact?: string }).fact ?? "").trim();
            if (!text) return { content: [{ type: "text", text: "remember needs a fact." }], details: null };
            memoryStore.remember(threadId, text);
            const count = memoryStore.facts(threadId).length;
            return { content: [{ type: "text", text: `Remembered (${count} facts now).\n${facts()}` }], details: null };
          },
        },
        {
          name: "forget",
          label: "forget",
          description:
            "Remove a fact you previously remembered. Call recall first to see each fact's [id], then forget the ones that are wrong or stale.",
          parameters: Type.Object({ id: Type.String() }, { additionalProperties: false }),
          async execute(_id, params) {
            const id = String((params as { id?: string }).id ?? "").trim();
            if (!id) return { content: [{ type: "text", text: "forget needs a fact id." }], details: null };
            const ok = memoryStore.forget(threadId, id);
            if (!ok) return { content: [{ type: "text", text: `No fact with id ${id} - call recall to list current ids.` }], details: null };
            return { content: [{ type: "text", text: `Forgot ${id}. Remaining:\n${facts() || "(none)"}` }], details: null };
          },
        },
        {
          name: "recall",
          label: "recall",
          description:
            "List everything you have remembered about the user, projects, and past work. Facts carry [id]s you can use with forget to prune wrong or stale entries.",
          parameters: Type.Object({}, { additionalProperties: false }),
          async execute() {
            return { content: [{ type: "text", text: facts() || "Nothing remembered yet." }], details: null };
          },
        },
        {
          name: "knowledge_read",
          label: "knowledge_read",
          description:
            "Read a file from the shared swarm knowledge base (a folder every bot in the workspace can read and write - research notes, playbooks, shared state). Pass the relative path under knowledge/, e.g. notes/alpha.md.",
          parameters: Type.Object({ path: Type.String() }, { additionalProperties: false }),
          async execute(_id, params) {
            const p = String((params as { path?: string }).path ?? "");
            const content = memoryStore.readKnowledge(p);
            if (content === null) {
              return { content: [{ type: "text", text: `No knowledge file at ${p || "(empty path)"}. Call knowledge_list to see what exists.` }], details: null };
            }
            return { content: [{ type: "text", text: content.slice(0, 20_000) }], details: null };
          },
        },
        {
          name: "knowledge_write",
          label: "knowledge_write",
          description:
            "Write (or overwrite) a file in the shared swarm knowledge base. Use it to persist research findings, notes, and state that other bots should be able to read later.",
          parameters: Type.Object(
            { path: Type.String(), content: Type.String() },
            { additionalProperties: false },
          ),
          async execute(_id, params) {
            const p = String((params as { path?: string }).path ?? "");
            const content = String((params as { content?: string }).content ?? "");
            const ok = memoryStore.writeKnowledge(p, content);
            if (!ok) return { content: [{ type: "text", text: `Invalid knowledge path: ${p}` }], details: null };
            return { content: [{ type: "text", text: `Wrote ${p} (${content.length} chars).` }], details: null };
          },
        },
        {
          name: "knowledge_list",
          label: "knowledge_list",
          description: "List the files currently in the shared swarm knowledge base.",
          parameters: Type.Object({}, { additionalProperties: false }),
          async execute() {
            const files = memoryStore.listKnowledge();
            if (!files.length) return { content: [{ type: "text", text: "Knowledge base is empty." }], details: null };
            return { content: [{ type: "text", text: `Knowledge base:\n${files.join("\n")}` }], details: null };
          },
        },
      ];

      // Add RAG tools
      const ragTools: ToolDefinition[] = [
        {
          name: "rag_search",
          label: "rag_search",
          description:
            "Search the vector knowledge base using semantic similarity. Returns relevant chunks with similarity scores.",
          parameters: Type.Object(
            { query: Type.String(), topK: Type.Optional(Type.Number()), threshold: Type.Optional(Type.Number()) },
            { additionalProperties: false },
          ),
          async execute(_id, params) {
            const query = String((params as { query?: string }).query ?? "").trim();
            const topK = Number((params as { topK?: number }).topK ?? 5);
            const threshold = Number((params as { threshold?: number }).threshold ?? 0.7);
            if (!query) return { content: [{ type: "text", text: "rag_search needs a query." }], details: null };
            const results = await hybridSearch(query, { topK, threshold });
            return { content: [{ type: "text", text: JSON.stringify(results, null, 2) }], details: null };
          },
        },
        {
          name: "rag_ingest",
          label: "rag_ingest",
          description:
            "Ingest text into the vector knowledge base for future semantic search. Content is automatically chunked and embedded.",
          parameters: Type.Object(
            { content: Type.String(), metadata: Type.Optional(Type.Record(Type.String(), Type.Unknown())) },
            { additionalProperties: false },
          ),
          async execute(_id, params) {
            const content = String((params as { content?: string }).content ?? "").trim();
            const metadata = (params as { metadata?: Record<string, unknown> }).metadata ?? {};
            if (!content) return { content: [{ type: "text", text: "rag_ingest needs content." }], details: null };
            const documents: Array<{ content: string; metadata?: Record<string, unknown> }> = [{ content, metadata }];
            const ids = await addDocuments(documents);
            return { content: [{ type: "text", text: `Ingested ${ids.length} chunks: ${ids.join(", ")}` }], details: null };
          },
        },
        {
          name: "rag_ingest_file",
          label: "rag_ingest_file",
          description:
            "Ingest a file from the workspace into the vector knowledge base.",
          parameters: Type.Object(
            { path: Type.String(), metadata: Type.Optional(Type.Record(Type.String(), Type.Unknown())) },
            { additionalProperties: false },
          ),
          async execute(_id, params) {
            const fs = await import("node:fs/promises");
            const path = String((params as { path?: string }).path ?? "").trim();
            const metadata = (params as { metadata?: Record<string, unknown> }).metadata ?? {};
            if (!path) return { content: [{ type: "text", text: "rag_ingest_file needs a path." }], details: null };
            try {
              const content = await fs.readFile(path, "utf8");
              const ids = await ingestFile(path, content, metadata);
              return { content: [{ type: "text", text: `Ingested file ${path} (${ids.length} chunks)` }], details: null };
            } catch (e) {
              return { content: [{ type: "text", text: `Error reading file: ${e}` }], details: null };
            }
          },
        },
      ];

      // Add tools from the global registry (builtin tools, MCP tools, etc.)
      const registryTools = toolsRegistry.listTools().map((t) => ({
        name: t.name,
        label: t.label,
        description: t.description,
        parameters: t.parameters,
        async execute(_id: string, params: Record<string, unknown>) {
          const result = await toolsRegistry.executeTool(t.name, params);
          // Convert ToolResult to prime-agent format
          return {
            content: result.content.map((c) =>
              c.type === "text" ? { type: "text" as const, text: c.text } : { type: "image" as const, data: c.data, mimeType: c.mimeType }
            ),
            details: result.details,
          };
        },
      }));

      return [...baseTools, ...ragTools, ...registryTools];
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
      const computer = turn.integrations?.computer;
      const computerTools = computer ? buildComputerTools(computer.boxId, computer.token) : [];
      const memoryTools = buildMemoryTools(threadId);
      
      // Get bot info for conversation engine (available for future use)
      // const _botName = turn.system?.match(/You are ([^,]+)/)?.[1] ?? "Bot";
      // const _botTitle = turn.system?.match(/Role: ([^.]+)/)?.[1];
      
      // Build tool names list
      const toolNames = [
        ...DEFAULT_TOOLS,
        ...peerTools.map((t) => t.name),
        ...computerTools.map((t) => t.name),
        ...memoryTools.map((t) => t.name),
      ];

      const modelId = turn.model ?? catalog.default;
      const model = resolveModel(modelId);
      if (!model) {
        throw new Error(
          `no model available for provider "${config.provider}" (catalog default: ${catalog.default}) - check models.json at ${MODELS_JSON}`,
        );
      }

      // The persona lives in agentDir/system-prompt.md - the SDK's resource
      // loader reads it once at session start. Sessions are per-thread, so a
      // per-thread agent dir keeps each bot's persona and session files
      // isolated from every other thread sharing this instance.
      const threadDir = threadAgentDir(threadId);
      mkdirSync(threadDir, { recursive: true });
      if (turn.system) {
        // Long-term memory rides in the system prompt: the bot's own facts
        // plus the swarm knowledge base index, so accumulated knowledge is
        // in context from the very first turn of any session.
        const memoryFacts = memoryStore.text(threadId);
        const knowledge = memoryStore.knowledgeSummary();
        const memoryBlock = memoryFacts || knowledge
          ? `\n\n## Long-term memory\n${memoryFacts ? `Facts you have remembered:\n${memoryFacts}\n` : ""}${knowledge ? `Shared knowledge base files (read with knowledge_read, write with knowledge_write):\n${knowledge}` : ""}`
          : "";
        writeFileSync(join(threadDir, "system-prompt.md"), turn.system + memoryBlock, "utf8");
      }

      // Disk-backed per-thread session (resume the thread's file on restart).
      // prime-agent's self-improvement loop (/refine auto-refine, enabled by
      // default) only runs for persisted sessions - in-memory sessions have no
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
        customTools: [...peerTools, ...computerTools, ...memoryTools],
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
        reason: `no NVIDIA key - add {"nvidia":{"key":"nvapi-…"}} to ~/.openmausbot/config.json or set ${config.apiKeyEnv}`,
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
        capabilities: { sessionModelSwitch: "in-session", agentsMcp: true, computerMcp: true },
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

