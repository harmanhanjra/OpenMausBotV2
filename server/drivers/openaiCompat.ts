// OpenAI-compatible chat/completions plumbing — the transcript-replay half of
// the driver roster (grok's xAI endpoint, NVIDIA NIM and any local vLLM /
// llama.cpp server). Everything these drivers share lives here: the request,
// the SSE parse, the transcript→messages fold, and the turn lifecycle that
// turns a stream of deltas into canonical runtime events. A driver keeps only
// what is actually its own — model catalog, config decoding, snapshot.
import type { SendTurnInput } from "../contracts.ts";
import { newId } from "../contracts.ts";
import type { EventHub } from "./events.ts";
import { appendNative } from "./native.ts";

export interface ChatMessage {
  role: string;
  content: string;
}

export interface ChatUsage {
  input: number;
  output: number;
}

export interface ChatResult {
  text: string;
  usage: ChatUsage | null;
}

export interface CompleteOptions {
  stream: boolean;
  signal?: AbortSignal;
  onDelta?: (delta: string) => void;
  /** Reasoning text, where the endpoint streams it apart from the content. */
  onReasoning?: (delta: string) => void;
}

export type Complete = (messages: ChatMessage[], model: string, opts: CompleteOptions) => Promise<ChatResult>;

export interface ChatCompleterConfig {
  /** Base URL, e.g. https://api.x.ai/v1 — `/chat/completions` is appended. */
  url: string;
  /** Empty for keyless local endpoints: the auth header is then omitted. */
  apiKey: string;
  /** Provider-shaped error for a non-2xx response. */
  errorFor: (status: number, body: string, model: string) => Error;
  timeoutMs?: number;
}

/** prompt_tokens/completion_tokens (OpenAI) or input_tokens/output_tokens
 * (NIM's newer shape) — whichever the endpoint sent. */
function readUsage(raw: any): ChatUsage | null {
  if (!raw) return null;
  return {
    input: raw.prompt_tokens ?? raw.input_tokens ?? 0,
    output: raw.completion_tokens ?? raw.output_tokens ?? 0,
  };
}

/** system + folded thread history + this turn's text. */
export function transcriptMessages(turn: SendTurnInput): ChatMessage[] {
  return [
    ...(turn.system ? [{ role: "system", content: turn.system }] : []),
    ...(turn.transcript ?? []).map((m) => ({
      role: m.role === "assistant" ? "assistant" : "user",
      content: m.text,
    })),
    { role: "user", content: turn.text },
  ];
}

/** Parse one `data:` SSE line, feeding content/reasoning deltas out.
 * Returns the accumulated content so callers can build the whole message. */
function consumeChunk(chunk: any, opts: CompleteOptions): { text: string; usage: ChatUsage | null } {
  let text = "";
  const delta = chunk.choices?.[0]?.delta;
  if (delta) {
    const reasoning = delta.reasoning_content ?? delta.reasoning;
    if (typeof reasoning === "string" && reasoning) opts.onReasoning?.(reasoning);
    if (typeof delta.content === "string" && delta.content) {
      text = delta.content;
      opts.onDelta?.(delta.content);
    }
  }
  return { text, usage: readUsage(chunk.usage) };
}

/** POST /chat/completions, streaming or not. */
export function createChatCompleter(config: ChatCompleterConfig): Complete {
  return async (messages, model, opts) => {
    const headers: Record<string, string> = { "content-type": "application/json" };
    if (config.apiKey) headers.authorization = `Bearer ${config.apiKey}`;
    const res = await fetch(`${config.url}/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify({ model, messages, stream: opts.stream }),
      signal: opts.signal ?? AbortSignal.timeout(config.timeoutMs ?? 120_000),
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw config.errorFor(res.status, body, model);
    }
    if (!opts.stream) {
      const json: any = await res.json();
      return { text: json.choices?.[0]?.message?.content ?? "", usage: readUsage(json.usage) };
    }
    let text = "";
    let usage: ChatUsage | null = null;
    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl: number;
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
        const consumed = consumeChunk(chunk, opts);
        text += consumed.text;
        if (consumed.usage) usage = consumed.usage;
      }
    }
    return { text, usage };
  };
}

export interface ActiveChatTurn {
  abort: AbortController;
  turnId: string;
}

export interface ChatTurnDeps {
  hub: EventHub;
  /** One in-flight turn per thread, keyed by threadId. */
  active: Map<string, ActiveChatTurn>;
  /** Native-protocol log label, e.g. "xai.chat.completions". */
  nativeSource: string;
  /** Read at send time — a discovered catalog's default can change. */
  defaultModel: () => string;
  complete: Complete;
  /** Emit reasoning deltas (endpoints that stream a separate field). */
  reasoning?: boolean;
}

/** ProviderAdapter.sendTurn for a streaming chat-completions driver: emits
 * turn.started/session.started, streams content.delta while the reply is
 * written, then item.completed + token usage + turn.completed. */
export function createChatTurnSender(deps: ChatTurnDeps) {
  const { hub, active } = deps;
  return async (turn: SendTurnInput): Promise<{ turnId: string }> => {
    const { threadId } = turn;
    if (active.has(threadId)) throw new Error("a turn is already running on this thread");
    const turnId = newId();
    const abort = new AbortController();
    active.set(threadId, { abort, turnId });

    const messages = transcriptMessages(turn);
    appendNative(threadId, { dir: "out", source: deps.nativeSource, msg: { model: turn.model, messages } });

    hub.emit({ ...hub.base(threadId, turnId), type: "turn.started" });
    hub.emit({
      ...hub.base(threadId, turnId),
      type: "session.started",
      sessionId: null,
      model: turn.model ?? deps.defaultModel(),
    });

    (async () => {
      try {
        const { text, usage } = await deps.complete(messages, turn.model || deps.defaultModel(), {
          stream: true,
          signal: abort.signal,
          onDelta: (delta) =>
            hub.emit({ ...hub.base(threadId, turnId), type: "content.delta", streamKind: "assistant_text", delta }),
          onReasoning: deps.reasoning
            ? (delta) =>
                hub.emit({ ...hub.base(threadId, turnId), type: "content.delta", streamKind: "reasoning_text", delta })
            : undefined,
        });
        appendNative(threadId, { dir: "in", source: deps.nativeSource, msg: { text, usage } });
        if (text.trim()) {
          hub.emit({ ...hub.base(threadId, turnId), type: "item.completed", itemType: "assistant_text", text });
        }
        if (usage) {
          hub.emit({ ...hub.base(threadId, turnId), type: "thread.token-usage.updated", ...usage });
        }
        active.delete(threadId);
        hub.emit({ ...hub.base(threadId, turnId), type: "turn.completed", ok: true, stopReason: null, cost: null });
      } catch (e) {
        active.delete(threadId);
        const aborted = (e as Error).name === "AbortError";
        if (!aborted) {
          hub.emit({ ...hub.base(threadId, turnId), type: "runtime.error", message: (e as Error).message });
        }
        hub.emit({
          ...hub.base(threadId, turnId),
          type: "turn.completed",
          ok: false,
          stopReason: aborted ? "interrupted" : "error",
          cost: null,
        });
      }
    })();

    return { turnId };
  };
}

/** interrupt/hasSession/stopAll for the abort-controller drivers. */
export function abortControls(active: Map<string, ActiveChatTurn>) {
  const abortAll = () => {
    for (const { abort } of active.values()) abort.abort();
  };
  return {
    abortAll,
    interruptTurn: async (threadId: string) => active.get(threadId)?.abort.abort(),
    hasSession: (threadId: string) => active.has(threadId),
    stopAll: async () => abortAll(),
  };
}
