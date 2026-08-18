// DeepSeek Harness (dsh) driver â€” embeds the OFFICIAL DeepSeek agent harness
// (deepseek-ai/deepseek-harness, published as the `@deepseek-ai/dsh` package)
// as a turn driver. Each turn spawns the harness's headless profile:
//
//   node <dsh>/lib/bin.js --profile headless [--patch model.yml] "<task>"
//
// The harness boots a full coding agent (files, shell, web, skills, plan-mode,
// todo, goal, jobs) with no listening port, answers the one task, prints the
// last non-empty assistant text to stdout, and exits (0 = turn completed,
// 1 = error; a terminal error reason writes its code + message to stderr).
//
// The harness reads its model/credentials from its own DSH_HOME tree (an
// isolated copy under DATA_DIR so bots never touch the user's ~/.dsh). The
// selected OpenMausBot model is pushed down as a patch overlay on the
// `agent-default-model` row, and the instance's DEEPSEEK_API_KEY is handed to
// the subprocess env. Since the headless runner materializes its profile +
// installs its bundle dependencies on first boot, the driver lazily warms
// DSH_HOME on the first turn (or at create when a key is present).
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

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
import { DATA_DIR } from "../config.ts";
import { appendNative } from "./native.ts";

const DRIVER_KIND = "deepseek";

// Resolve the installed `dsh` CLI relative to this file's package tree.
const HERE = dirname(fileURLToPath(import.meta.url));
const DSH_BIN = join(HERE, "..", "..", "node_modules", "@deepseek-ai", "dsh", "lib", "bin.js");

// The harness's own model catalog for the `deepseek-official` route (the only
// provider the headless bundle mounts by default). These map 1:1 to the
// `agent-default-model` patch we write per turn.
const MODELS: ModelCatalog = {
  default: "deepseek-v4-flash",
  options: [
    { id: "deepseek-v4-flash", label: "DeepSeek V4 Flash" },
    { id: "deepseek-v4-pro", label: "DeepSeek V4 Pro" },
  ],
};

export interface DeepseekConfig {
  apiKeyEnv: string;
  /** Override where the harness home lives (default: $DATA_DIR/dsh). */
  dshHome?: string;
  /** Milliseconds after which a hanging harness run is killed. */
  timeoutMs: number;
}

function decodeConfig(raw: unknown): DeepseekConfig {
  const o = (raw ?? {}) as Record<string, unknown>;
  return {
    apiKeyEnv: typeof o.apiKeyEnv === "string" && o.apiKeyEnv ? o.apiKeyEnv : "DEEPSEEK_API_KEY",
    dshHome: typeof o.dshHome === "string" && o.dshHome ? o.dshHome : undefined,
    timeoutMs: typeof o.timeoutMs === "number" && o.timeoutMs > 0 ? o.timeoutMs : 10 * 60_000,
  };
}

// â”€â”€ harness invocation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// The headless runner is a one-shot subprocess. `runHeadless` wraps the spawn,
// resolves with { stdout, stderr, code } and rejects on spawn/signal failure.

export interface HeadlessResult {
  code: number | null;
  stdout: string;
  stderr: string;
}

export function runHeadless(opts: {
  nodeExe: string;
  dshBin?: string;
  dshHome: string;
  apiKey: string;
  task: string;
  patchYaml?: string;
  /** Escape hatch for tests: spawn this script instead of the dsh CLI. */
  args?: string[];
  timeoutMs?: number;
  signal?: AbortSignal;
  onStdout?: (chunk: string) => void;
  onStderr?: (chunk: string) => void;
}): Promise<HeadlessResult> {
  const { nodeExe, dshHome, apiKey, task, patchYaml } = opts;
  const dshBin = opts.dshBin ?? DSH_BIN;
  const timeoutMs = opts.timeoutMs ?? 10 * 60_000;
  const args = opts.args ?? (() => {
    const a = ["--profile", "headless"];
    if (patchYaml) a.push("--patch", patchYaml);
    a.push(task);
    return a;
  })();

  return new Promise((resolve, reject) => {
    let child: ReturnType<typeof spawn>;
    try {
      child = spawn(nodeExe, [opts.args ? undefined : dshBin, ...args].filter((x) => x !== undefined) as string[], {
        env: { ...process.env, DSH_HOME: dshHome, DEEPSEEK_API_KEY: apiKey },
        windowsHide: true,
        stdio: ["ignore", "pipe", "pipe"],
      });
    } catch (e) {
      reject(e);
      return;
    }

    let stdout = "";
    let stderr = "";
    let settled = false;
    const stdoutStream = child.stdout;
    const stderrStream = child.stderr;
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      child.kill("SIGKILL");
      reject(new Error(`dsh timed out after ${Math.round(timeoutMs / 1000)}s`));
    }, timeoutMs);
    timer.unref();

    const done = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve({ code: child.exitCode, stdout, stderr });
    };

    stdoutStream?.on("data", (chunk: Buffer) => {
      const s = chunk.toString();
      stdout += s;
      opts.onStdout?.(s);
    });
    stderrStream?.on("data", (chunk: Buffer) => {
      const s = chunk.toString();
      stderr += s;
      opts.onStderr?.(s);
    });
    child.on("error", (e) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(e);
    });
    child.on("close", done);
    child.on("exit", done);

    opts.signal?.addEventListener(
      "abort",
      () => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        child.kill("SIGKILL");
        reject(new Error("dsh run interrupted"));
      },
      { once: true },
    );
  });
}

/** Write a `--patch` overlay that pins the harness default model and,
 *  when agents integration is present, injects the agents proxy MCP server
 *  so the headless agent gets list_bots / ask_bot tools. */
export function writeModelPatch(
  dshHome: string,
  model: string,
  agentsIntegration?: { command: string; args: string[]; env: Record<string, string> },
  nodeExe?: string,
): string {
  const patchDir = join(dshHome, "patches");
  mkdirSync(patchDir, { recursive: true });
  const path = join(patchDir, "model.yml");
  const safeModel = String(model).replace(/^[^a-zA-Z0-9_.-]+/, "");
  let yaml =
    `# generated by OpenMausBot -- pin the harness default model\n` +
    `- id: agent-default-model\n` +
    `  config:\n` +
    `    provider: deepseek-official\n` +
    `    model: ${safeModel}\n`;

  // Inject the agents proxy MCP server so the headless agent can talk to peers
  if (agentsIntegration) {
    const cmd = nodeExe ?? agentsIntegration.command;
    const args = JSON.stringify(agentsIntegration.args);
    const envLines = Object.entries(agentsIntegration.env)
      .map(([k, v]) => `      ${k}: "${String(v).replace(/"/g, '\\"')}"`)
      .join("\n");
    yaml +=
      `\n# agents proxy MCP server -- gives the headless agent list_bots / ask_bot\n` +
      `- id: agents-mcp\n` +
      `  name: '@deepseek-ai/dsh-mcp-client'\n` +
      `  config:\n` +
      `    command: "${cmd}"\n` +
      `    args: ${args}\n` +
      `    env:\n${envLines}\n`;
  }

  writeFileSync(path, yaml, "utf8");
  return path;
}

export const DeepseekDriver: ProviderDriver<DeepseekConfig> = {
  driverKind: DRIVER_KIND,
  metadata: { displayName: "DeepSeek (Harness)", supportsMultipleInstances: true },
  models: MODELS,
  decodeConfig,
  defaultConfig: () => decodeConfig({}),

  async create(input: DriverCreateInput<DeepseekConfig>): Promise<ProviderInstance> {
    const { instanceId, config } = input;
    const apiKey = input.environment[config.apiKeyEnv] ?? process.env[config.apiKeyEnv] ?? "";
    const nodeExe = process.execPath;
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

    const sendTurn = async (turn: SendTurnInput) => {
      const { threadId } = turn;
      if (!apiKey) {
        throw new Error(
          `no DeepSeek key â€” set ${config.apiKeyEnv} or add {"deepseek":{"key":"sk-â€¦"}} to ~/.openmausbot/config.json`,
        );
      }
      if (active.has(threadId)) throw new Error("a turn is already running on this thread");
      const turnId = newId();
      const abort = new AbortController();
      active.set(threadId, { abort, turnId });

      const dshHome = config.dshHome ?? join(DATA_DIR, "dsh");
      const agentsIntegration = turn.integrations?.agents;
      const patchYaml = writeModelPatch(dshHome, turn.model ?? MODELS.default, agentsIntegration, nodeExe);

      const task = [
        ...(turn.system ? [turn.system, ""] : []),
        ...(turn.transcript ?? []).map((m) => `${m.role}: ${m.text}`),
        `user: ${turn.text}`,
      ].join("\n");
      appendNative(threadId, { dir: "out", source: "dsh.headless", msg: { model: turn.model ?? MODELS.default, task } });

      emit({ ...base(threadId, turnId), type: "turn.started" });
      emit({
        ...base(threadId, turnId),
        type: "session.started",
        sessionId: null,
        model: turn.model ?? MODELS.default,
      });

      (async () => {
        try {
          let text = "";
          const res = await runHeadless({
            nodeExe,
            dshHome,
            apiKey,
            task,
            patchYaml,
            timeoutMs: config.timeoutMs,
            signal: abort.signal,
            onStdout: (chunk) => {
              text += chunk;
              emit({ ...base(threadId, turnId), type: "content.delta", streamKind: "assistant_text", delta: chunk });
            },
          });
          appendNative(threadId, { dir: "in", source: "dsh.headless", msg: { text, code: res.code, stderr: res.stderr } });

          const finalText = res.stdout.trim();
          if (res.code !== 0) {
            const errDetail = res.stderr.trim().split("\n")[0] ?? `dsh exited with code ${res.code}`;
            throw new Error(`DeepSeek Harness: ${errDetail}`);
          }
          if (finalText) {
            emit({ ...base(threadId, turnId), type: "item.completed", itemType: "assistant_text", text: finalText });
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
      if (!apiKey) {
        return {
          state: "unavailable",
          reason: `no DeepSeek key â€” add {"deepseek":{"key":"sk-â€¦"}} to ~/.openmausbot/config.json or set ${config.apiKeyEnv}`,
        };
      }
      if (!existsSync(DSH_BIN)) {
        return {
          state: "unavailable",
          reason: "@deepseek-ai/dsh not installed â€” run `pnpm add @deepseek-ai/dsh`",
        };
      }
      const dshHome = config.dshHome ?? join(DATA_DIR, "dsh");
      // Warm the harness home on first sight of a key so the first turn isn't
      // paying the profile-materialization cost. Best-effort; a failed warmup
      // only makes the first turn slower.
      if (!existsSync(join(dshHome, "profiles", "headless", "cordis.yml"))) {
        try {
          await runHeadless({ nodeExe, dshHome, apiKey, task: "--help", timeoutMs: 120_000 });
        } catch {
          /* first-turn warmup will retry */
        }
      }
      return { state: "available", authenticated: true, version: "@deepseek-ai/dsh (headless)" };
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
        capabilities: { sessionModelSwitch: "unsupported", agentsMcp: true },
        sendTurn,
        interruptTurn: async (threadId) => active.get(threadId)?.abort.abort(),
        respondToRequest: async () => {
          throw new Error("deepseek driver has no pending asks");
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
        if (!apiKey) throw new Error(`no DeepSeek key â€” set ${config.apiKeyEnv}`);
        const dshHome = config.dshHome ?? join(DATA_DIR, "dsh");
        const patchYaml = writeModelPatch(dshHome, MODELS.default);
        const res = await runHeadless({ nodeExe, dshHome, apiKey, task: prompt, patchYaml, timeoutMs: config.timeoutMs });
        if (res.code !== 0) throw new Error(`DeepSeek Harness: ${res.stderr.trim().split("\n")[0]}`);
        return res.stdout.trim();
      },
      dispose: async () => {
        for (const { abort } of active.values()) abort.abort();
        listeners.clear();
      },
    };
  },
};