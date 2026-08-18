// Grok driver — xAI chat-completions API with SSE streaming. Unlike the
// CLI drivers this one is transcript-replay: the server hands it the
// folded thread history each turn (SendTurnInput.transcript) and it emits
// true token-level content.delta events. Also supplies the instance's
// generateText (bot titles, thread names) — upstream's TextGeneration slot.
import type {
  DriverCreateInput,
  ProviderDriver,
  ProviderInstance,
  ProviderSnapshot,
  SendTurnInput,
} from "../contracts.ts";
import { createEventHub } from "./events.ts";
import {
  abortControls,
  createChatCompleter,
  createChatTurnSender,
  type ActiveChatTurn,
} from "./openaiCompat.ts";

const DRIVER_KIND = "grok";
const DEFAULT_URL = "https://api.x.ai/v1";

const MODELS = {
  default: "grok-4",
  options: [
    { id: "grok-4", label: "Grok 4" },
    { id: "grok-4-fast", label: "Grok 4 Fast" },
    { id: "grok-3-mini", label: "Grok 3 Mini" },
  ],
};

export interface GrokConfig {
  url: string;
  /** resolved at create-time from instance environment / app config */
  apiKeyEnv: string;
}

function decodeConfig(raw: unknown): GrokConfig {
  const o = (raw ?? {}) as Record<string, unknown>;
  return {
    url: typeof o.url === "string" ? o.url : DEFAULT_URL,
    apiKeyEnv: typeof o.apiKeyEnv === "string" ? o.apiKeyEnv : "XAI_API_KEY",
  };
}

export const GrokDriver: ProviderDriver<GrokConfig> = {
  driverKind: DRIVER_KIND,
  // "(API)" distinguishes this key-billed driver from grokAgent, the CLI one
  metadata: { displayName: "Grok (API)", supportsMultipleInstances: true },
  models: MODELS,
  decodeConfig,
  defaultConfig: () => decodeConfig({}),

  async create(input: DriverCreateInput<GrokConfig>): Promise<ProviderInstance> {
    const { instanceId, config } = input;
    const apiKey = input.environment[config.apiKeyEnv] ?? process.env[config.apiKeyEnv] ?? "";
    const hub = createEventHub(DRIVER_KIND);
    const active = new Map<string, ActiveChatTurn>();
    const controls = abortControls(active);

    const complete = createChatCompleter({
      url: config.url,
      apiKey,
      errorFor: (status, body) => new Error(`xAI HTTP ${status}${body ? `: ${body.slice(0, 200)}` : ""}`),
    });

    const runTurn = createChatTurnSender({
      hub,
      active,
      nativeSource: "xai.chat.completions",
      defaultModel: () => MODELS.default,
      complete,
    });

    const sendTurn = async (turn: SendTurnInput) => {
      if (!apiKey) throw new Error(`no xAI key — set ${config.apiKeyEnv} or config.json xai.key`);
      return runTurn(turn);
    };

    const snapshot = async (): Promise<ProviderSnapshot> => {
      if (!apiKey) {
        return {
          state: "unavailable",
          reason: `no xAI API key — add {"xai":{"key":"xai-…"}} to ~/.openmausbot/config.json or set ${config.apiKeyEnv}`,
        };
      }
      return { state: "available", authenticated: true, version: null };
    };

    return {
      instanceId,
      driverKind: DRIVER_KIND,
      displayName: input.displayName,
      enabled: input.enabled,
      models: MODELS,
      snapshot,
      adapter: {
        provider: DRIVER_KIND,
        capabilities: { sessionModelSwitch: "in-session" },
        sendTurn,
        interruptTurn: controls.interruptTurn,
        respondToRequest: async () => {
          throw new Error("grok driver has no pending asks");
        },
        hasSession: controls.hasSession,
        stopAll: controls.stopAll,
        onEvent: hub.onEvent,
      },
      generateText: async (prompt: string) => {
        const { text } = await complete([{ role: "user", content: prompt }], "grok-3-mini", { stream: false });
        return text;
      },
      dispose: async () => {
        controls.abortAll();
        hub.clear();
      },
    };
  },
};
