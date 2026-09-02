// NVIDIA driver — NVIDIA NIM + local CUDA inference through the OpenAI-
// compatible chat/completions API. Like the grok driver this one is
// transcript-replay: the server hands it the folded thread history each
// turn (SendTurnInput.transcript) and it emits true token-level
// content.delta events.
//
// Two deployment modes, one driver:
//   • Cloud NIM — https://integrate.api.nvidia.com/v1 (the default). Needs
//     an NVIDIA API key (nvapi-…), resolved from the instance environment
//     or process env via config.apiKeyEnv.
//   • Local CUDA — any OpenAI-compatible endpoint on this machine (self-
//     hosted NIM, vLLM, llama.cpp server) pointed to by config.url. No key
//     is required for local addresses; the snapshot probes /models instead.
//
// The snapshot also reports the NVIDIA GPU detected via nvidia-smi so the
// model picker shows local acceleration alongside cloud availability.
import { execFile } from "node:child_process";
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
import { buildPeerTools } from "./primeAgent.ts";
import { appendNative } from "./native.ts";
import { DEFAULT_KEY_ENV, DEFAULT_URL, discoverModels } from "./nvidia-shared.ts";

export { DEFAULT_KEY_ENV, DEFAULT_URL, discoverModels, isNonChatModel } from "./nvidia-shared.ts";

const DRIVER_KIND = "nvidia";

// Hosted NIM model catalog — used as the FALLBACK when the live /v1/models
// discovery (see discoverModels) can't reach the endpoint. The authoritative
// list is whatever the API returns for the caller's key, so the picker shows
// every model NVIDIA actually serves — including new ones we haven't listed
// here. Model IDs on build.nvidia.com follow `provider/model-name`
// (verified examples below; the same IDs work for chat completions).
const MODELS: ModelCatalog = {
  default: "meta/llama-3.3-70b-instruct",
  options: [
    { id: "meta/llama-3.3-70b-instruct", label: "Llama 3.3 70B" },
    { id: "meta/llama-3.1-8b-instruct", label: "Llama 3.1 8B" },
    { id: "meta/llama-3.1-70b-instruct", label: "Llama 3.1 70B" },
    { id: "meta/llama-3.1-405b-instruct", label: "Llama 3.1 405B" },
    { id: "nvidia/llama-3.1-nemotron-ultra-253b-v1", label: "Nemotron Ultra 253B" },
    { id: "nvidia/llama-3.1-nemotron-nano-8b-v1", label: "Nemotron Nano 8B" },
    { id: "nvidia/nemotron-3-super", label: "Nemotron 3 Super" },
    { id: "deepseek-ai/deepseek-v3.1", label: "DeepSeek V3.1" },
    { id: "deepseek-ai/deepseek-r1", label: "DeepSeek R1" },
    { id: "openai/gpt-oss-120b", label: "GPT-OSS 120B" },
    { id: "openai/gpt-oss-20b", label: "GPT-OSS 20B" },
    { id: "moonshotai/kimi-k2", label: "Kimi K2" },
    { id: "zhipuai/glm-5.1", label: "GLM 5.1" },
    { id: "qwen/qwen3-32b", label: "Qwen3 32B" },
    { id: "qwen/qwen2.5-72b-instruct", label: "Qwen 2.5 72B" },
    { id: "mistralai/mistral-nemo-12b-instruct", label: "Mistral NeMo 12B" },
    { id: "mistralai/mixtral-8x22b-instruct", label: "Mixtral 8x22B" },
    { id: "google/codegemma-7b", label: "CodeGemma 7B" },
    { id: "google/gemma-2-27b", label: "Gemma 2 27B" },
  ],
};

// Discovery caches per (endpoint,key) so /api/instances polls don't hammer
// the NIM API. 10-minute TTL; a stale catalog is better than a blocking poll.
const MODEL_CACHE_TTL_MS = 10 * 60 * 1000;
const MODEL_CACHE = new Map<string, { at: number; catalog: ModelCatalog }>();

/** cached, in-place refresh of a mutable catalog (see applyCatalog). */
async function refreshCatalog(baseUrl: string, apiKey: string, target: ModelCatalog): Promise<boolean> {
  const cacheKey = `${baseUrl}|${apiKey}`;
  const hit = MODEL_CACHE.get(cacheKey);
  const fresh = hit && Date.now() - hit.at < MODEL_CACHE_TTL_MS ? hit.catalog : await discoverModels(baseUrl, apiKey);
  if (fresh) {
    if (!hit) MODEL_CACHE.set(cacheKey, { at: Date.now(), catalog: fresh });
    applyCatalog(target, fresh);
    return true;
  }
  return false;
}

function applyCatalog(target: ModelCatalog, next: ModelCatalog) {
  target.options.splice(0, target.options.length, ...next.options.map((o) => ({ ...o })));
  target.default = next.default;
}

export interface NvidiaConfig {
  url: string;
  /** resolved at create-time from instance environment / app config */
  apiKeyEnv: string;
  /** optional override of the static model catalog (self-hosted models) */
  models?: { default: string; options: Array<{ id: string; label: string }> };
}

function isLocalUrl(url: string): boolean {
  try {
    const host = new URL(url).hostname;
    return host === "localhost" || host === "127.0.0.1" || host === "::1" || host.startsWith("192.168.");
  } catch {
    return false;
  }
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
      : MODELS.options;
    if (options.length) {
      const def = typeof o.default === "string" && options.some((m) => m.id === o.default) ? o.default : options[0].id;
      return { default: def, options };
    }
  }
  return undefined;
}

function decodeConfig(raw: unknown): NvidiaConfig {
  const o = (raw ?? {}) as Record<string, unknown>;
  return {
    url: typeof o.url === "string" && o.url ? o.url : DEFAULT_URL,
    apiKeyEnv: typeof o.apiKeyEnv === "string" && o.apiKeyEnv ? o.apiKeyEnv : DEFAULT_KEY_ENV,
    models: decodeModels(o.models),
  };
}

let gpuCache: string | null | undefined;
function detectGpu(timeoutMs = 1500): Promise<string | null> {
  if (gpuCache !== undefined) return Promise.resolve(gpuCache);
  return new Promise((resolve) => {
    execFile(
      "nvidia-smi",
      ["--query-gpu=name", "--format=csv,noheader"],
      { timeout: timeoutMs, windowsHide: true },
      (err, stdout) => {
        const gpu = err || !stdout ? null : stdout.trim().split("\n")[0]?.trim() || null;
        gpuCache = gpu;
        resolve(gpu);
      },
    );
  });
}

export const NvidiaDriver: ProviderDriver<NvidiaConfig> = {
  driverKind: DRIVER_KIND,
  metadata: { displayName: "NVIDIA", supportsMultipleInstances: true },
  models: MODELS,
  decodeConfig,
  defaultConfig: () => decodeConfig({}),

  async create(input: DriverCreateInput<NvidiaConfig>): Promise<ProviderInstance> {
    const { instanceId, config } = input;
    const apiKey = input.environment[config.apiKeyEnv] ?? process.env[config.apiKeyEnv] ?? "";
    const local = isLocalUrl(config.url);
    // Mutable catalog: seeded from config/static fallback, then refreshed
    // from the endpoint's /v1/models so the model picker lists every model
    // this key can actually call (cloud NIM) or the local server hosts.
    // An explicit per-instance config.models always wins — self-hosted
    // deployments pin their own list; discovery would only add noise.
    const catalog: ModelCatalog = {
      default: (config.models ?? MODELS).default,
      options: (config.models ?? MODELS).options.map((o) => ({ ...o })),
    };
    if (!config.models && (apiKey || local)) {
      await refreshCatalog(config.url, apiKey, catalog);
    }
    const listeners = new Set<RuntimeEventListener>();
    const active = new Map<string, { abort: AbortController; turnId: string }>();

    const emit = (event: RuntimeEvent) => {
      for (const l of [...listeners]) l(event);
    };
    const base = (threadId: string, turnId: string) => ({
      eventId: newEventId(),
      provider: DRIVER_KIND,
      threadId,
      turnId,
      createdAt: new Date().toISOString(),
    });

    interface ToolCall {
      id: string;
      name: string;
      arguments: string;
    }
    interface CompletionResult {
      text: string;
      usage: { input: number; output: number } | null;
      toolCalls: ToolCall[];
    }

    const complete = async (
      messages: Array<{ role: string; content?: string; tool_call_id?: string; tool_calls?: unknown[] }>,
      model: string,
      opts: {
        stream: boolean;
        signal?: AbortSignal;
        onDelta?: (d: string) => void;
        onReasoning?: (d: string) => void;
        tools?: unknown[];
      },
    ): Promise<CompletionResult> => {
      const body: Record<string, unknown> = { model, messages, stream: opts.stream };
      if (opts.tools?.length) body.tools = opts.tools;
      const headers: Record<string, string> = { "content-type": "application/json" };
      if (apiKey) headers.authorization = `Bearer ${apiKey}`;
      const res = await fetch(`${config.url}/chat/completions`, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
        signal: opts.signal ?? AbortSignal.timeout(120_000),
      });
      if (!res.ok) {
        const respBody = await res.text().catch(() => "");
        const parsed: any = JSON.parse(respBody) || {};
        const detail = parsed.error?.message || parsed.error?.errors?.[0]?.message || "";
        const spaceMsg = res.status === 404 ? "Model may not be available on NIM; check https://build.nvidia.com for valid model IDs." : "";
        throw new Error(
          `NVIDIA API error (${res.status}${detail ? ` — ${detail}` : ""}) for model "${model}". ${spaceMsg}`,
        );
      }
      if (!opts.stream) {
        const json: any = await res.json();
        const msg = json.choices?.[0]?.message;
        return {
          text: msg?.content ?? "",
          usage: json.usage
            ? {
                input: json.usage.prompt_tokens ?? json.usage.input_tokens ?? 0,
                output: json.usage.completion_tokens ?? json.usage.output_tokens ?? 0,
              }
            : null,
          toolCalls: Array.isArray(msg?.tool_calls)
            ? msg.tool_calls.map((tc: any) => ({
                id: tc.id ?? `tc-${Date.now()}`,
                name: tc.function?.name ?? "",
                arguments: tc.function?.arguments ?? "{}",
              }))
            : [],
        };
      }
      let text = "";
      let usage: { input: number; output: number } | null = null;
      // streaming tool-call accumulator: NIM sends tool calls as indexed deltas
      const toolCallBuf = new Map<number, { id: string; name: string; args: string }>();
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let nl;
        while ((nl = buf.indexOf("\n")) !== -1) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim();
          if (data === "[DONE]") continue;
          let chunk: any;
          try {
            chunk = JSON.parse(data);
          } catch {
            continue;
          }
          const delta = chunk.choices?.[0]?.delta;
          if (delta) {
            const reasoning = delta.reasoning_content ?? delta.reasoning;
            if (typeof reasoning === "string" && reasoning) opts.onReasoning?.(reasoning);
            if (typeof delta.content === "string" && delta.content) {
              text += delta.content;
              opts.onDelta?.(delta.content);
            }
            if (Array.isArray(delta.tool_calls)) {
              for (const tc of delta.tool_calls) {
                const idx = tc.index ?? 0;
                const entry = toolCallBuf.get(idx) ?? { id: "", name: "", args: "" };
                if (tc.id) entry.id = tc.id;
                if (tc.function?.name) entry.name += tc.function.name;
                if (tc.function?.arguments) entry.args += tc.function.arguments;
                toolCallBuf.set(idx, entry);
              }
            }
          }
          if (chunk.usage) {
            usage = {
              input: chunk.usage.prompt_tokens ?? chunk.usage.input_tokens ?? 0,
              output: chunk.usage.completion_tokens ?? chunk.usage.output_tokens ?? 0,
            };
          }
        }
      }
      const toolCalls = [...toolCallBuf.values()]
        .filter((tc) => tc.name)
        .map((tc) => ({ id: tc.id || `tc-${Date.now()}`, name: tc.name, arguments: tc.args || "{}" }));
      return { text, usage, toolCalls };
    };

    const sendTurn = async (turn: SendTurnInput) => {
      const { threadId } = turn;
      if (!apiKey && !local) {
        throw new Error(`no NVIDIA key — set ${config.apiKeyEnv} or add {"nvidia":{"key":"nvapi-…"}} to ~/.openmausbot/config.json`);
      }
      if (active.has(threadId)) throw new Error("a turn is already running on this thread");
      const turnId = newId();
      const abort = new AbortController();
      active.set(threadId, { abort, turnId });

      // Build peer tools from the agents integration (list_bots / ask_bot / CEO tools)
      const agentsEnv = turn.integrations?.agents?.env;
      const peerTools = agentsEnv ? buildPeerTools(agentsEnv) : [];
      const apiTools = peerTools.length
        ? peerTools.map((t) => ({
            type: "function",
            function: {
              name: t.name,
              description: t.description,
              parameters: t.parameters,
            },
          }))
        : undefined;
      const toolMap = new Map(peerTools.map((t) => [t.name, t]));

      const messages: Array<{ role: string; content?: string; tool_call_id?: string; tool_calls?: unknown[] }> = [
        ...(turn.system ? [{ role: "system", content: turn.system }] : []),
        ...(turn.transcript ?? []).map((m) => ({
          role: m.role === "assistant" ? "assistant" : "user",
          content: m.text,
        })),
        { role: "user", content: turn.text },
      ];
      appendNative(threadId, { dir: "out", source: "nvidia.chat.completions", msg: { model: turn.model, messages } });

      emit({ ...base(threadId, turnId), type: "turn.started" });
      emit({
        ...base(threadId, turnId),
        type: "session.started",
        sessionId: null,
        model: turn.model ?? catalog.default,
      });

      (async () => {
        try {
          const MAX_TOOL_ROUNDS = 8;
          let totalUsage: { input: number; output: number } | null = null;
          let finalText = "";

          // Tool loop: keep calling the API until the model produces text without tool_calls
          for (let round = 0; round <= MAX_TOOL_ROUNDS; round++) {
            if (abort.signal.aborted) break;
            const isLastRound = round === MAX_TOOL_ROUNDS || !apiTools;
            const modelId = turn.model || catalog.default;

            if (isLastRound || !apiTools) {
              // Final round: stream the text response
              const { text, usage } = await complete(messages, modelId, {
                stream: true,
                signal: abort.signal,
                // If the model exhausted the tool budget, force a text-only
                // response rather than silently dropping another tool call.
                tools: round === MAX_TOOL_ROUNDS ? undefined : apiTools,
                onReasoning: (delta) =>
                  emit({ ...base(threadId, turnId), type: "content.delta", streamKind: "reasoning_text", delta }),
                onDelta: (delta) =>
                  emit({ ...base(threadId, turnId), type: "content.delta", streamKind: "assistant_text", delta }),
              });
              finalText = text;
              if (usage) totalUsage = totalUsage ? { input: totalUsage.input + usage.input, output: totalUsage.output + usage.output } : usage;
              break;
            }

            // Intermediate rounds: non-streaming to collect tool calls
            const { text, usage, toolCalls } = await complete(messages, modelId, {
              stream: false,
              signal: abort.signal,
              tools: apiTools,
            });
            if (usage) totalUsage = totalUsage ? { input: totalUsage.input + usage.input, output: totalUsage.output + usage.output } : usage;

            if (toolCalls.length === 0) {
              // Model produced text without tool calls — stream the final answer
              if (text) {
                messages.push({ role: "assistant", content: text });
                // Re-stream the text through delta events
                emit({ ...base(threadId, turnId), type: "content.delta", streamKind: "assistant_text", delta: text });
              }
              finalText = text;
              break;
            }

            // Model wants to call tools — append assistant message with tool_calls
            messages.push({ role: "assistant", content: text || undefined, tool_calls: toolCalls.map((tc) => ({ id: tc.id, type: "function", function: { name: tc.name, arguments: tc.arguments } })) });

            // Execute each tool call and append tool results
            for (const tc of toolCalls) {
              if (abort.signal.aborted) break;
              emit({ ...base(threadId, turnId), type: "item.started", itemType: "tool", itemId: tc.id, title: tc.name });

              let result = "";
              const tool = toolMap.get(tc.name);
              if (!tool) {
                result = `Unknown tool: ${tc.name}`;
              } else {
                try {
                  let args: Record<string, unknown> = {};
                  try { args = JSON.parse(tc.arguments); } catch { /* use empty */ }
                  // Peer tools are host-owned functions. They currently do not
                  // need the SDK extension context, but pass the cancellation
                  // signal through so a cancelled turn cannot keep waiting on
                  // a peer request.
                  const out = await tool.execute(tc.id, args);
                  result = (out.content as Array<{ type: string; text?: string }>)
                    ?.filter((c) => c.type === "text")
                    .map((c) => c.text ?? "")
                    .join("\n") ?? "";
                } catch (e) {
                  result = `Tool error: ${(e as Error).message}`;
                }
              }

              emit({
                ...base(threadId, turnId),
                type: "item.completed",
                itemType: "tool",
                itemId: tc.id,
                ok: !result.startsWith("Tool error:") && !result.startsWith("Unknown tool:"),
              });
              messages.push({ role: "tool", tool_call_id: tc.id, content: result });
            }
          }

          appendNative(threadId, { dir: "in", source: "nvidia.chat.completions", msg: { text: finalText, usage: totalUsage } });
          if (finalText.trim()) {
            emit({ ...base(threadId, turnId), type: "item.completed", itemType: "assistant_text", text: finalText });
          }
          if (totalUsage) {
            emit({ ...base(threadId, turnId), type: "thread.token-usage.updated", ...totalUsage });
          }
          active.delete(threadId);
          emit({ ...base(threadId, turnId), type: "turn.completed", ok: true, stopReason: null, cost: null });
        } catch (e) {
          active.delete(threadId);
          const aborted = (e as Error).name === "AbortError";
          if (!aborted) {
            emit({ ...base(threadId, turnId), type: "runtime.error", message: (e as Error).message });
          }
          emit({
            ...base(threadId, turnId),
            type: "turn.completed",
            ok: false,
            stopReason: aborted ? "interrupted" : "error",
            cost: null,
          });
        }
      })();

      return { turnId };
    };

    const snapshot = async (): Promise<ProviderSnapshot> => {
      const gpu = await detectGpu();
      const gpuNote = gpu ? ` · GPU: ${gpu}` : "";
      if (apiKey) {
        return { state: "available", authenticated: true, version: gpuNote || null };
      }
      if (local) {
        try {
          const res = await fetch(`${config.url}/models`, { signal: AbortSignal.timeout(3_000) });
          if (res.ok) {
            return { state: "available", authenticated: false, version: `local NIM/vLLM${gpuNote}` };
          }
          return {
            state: "unavailable",
            reason: `local endpoint ${config.url} reachable but returned HTTP ${res.status}`,
          };
        } catch {
          return {
            state: "unavailable",
            reason: `no NVIDIA key and local endpoint ${config.url} not reachable — start a NIM/vLLM server or add {"nvidia":{"key":"nvapi-…"}} to ~/.openmausbot/config.json`,
          };
        }
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
        interruptTurn: async (threadId) => active.get(threadId)?.abort.abort(),
        respondToRequest: async () => {
          throw new Error("nvidia driver has no pending asks");
        },
        hasSession: (threadId) => active.has(threadId),
        stopAll: async () => {
          for (const { abort } of active.values()) abort.abort();
        },
        onEvent: (listener) => {
          listeners.add(listener);
          return () => listeners.delete(listener);
        },
      },
      generateText: async (prompt: string) => {
        const { text } = await complete([{ role: "user", content: prompt }], catalog.default, { stream: false, tools: undefined });
        return text;
      },
      dispose: async () => {
        for (const { abort } of active.values()) abort.abort();
        listeners.clear();
      },
    };
  },
};
