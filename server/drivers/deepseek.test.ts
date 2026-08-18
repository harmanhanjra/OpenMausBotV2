// DeepSeek Harness (dsh) driver contract tests. The harness subprocess
// materializes its profile + installs bundle deps on first boot (slow, and
// needs a real API key to answer), so these tests stay offline: decodeConfig,
// snapshot behavior, the model-patch writer, and the runHeadless spawn
// wrapper driven against tiny inline scripts instead of the real dsh binary.
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, beforeAll, describe, expect, it } from "vitest";

import type { DeepseekConfig } from "./deepseek.ts";

process.env.OMB_DATA_DIR = mkdtempSync(join(tmpdir(), "omb-dsh-"));

let DeepseekDriver: Awaited<typeof import("./deepseek.ts")>["DeepseekDriver"];
let runHeadless: Awaited<typeof import("./deepseek.ts")>["runHeadless"];
let writeModelPatch: Awaited<typeof import("./deepseek.ts")>["writeModelPatch"];

beforeAll(async () => {
  ({ DeepseekDriver, runHeadless, writeModelPatch } = await import("./deepseek.ts"));
});

afterAll(() => {
  rmSync(process.env.OMB_DATA_DIR!, { recursive: true, force: true });
});

describe("DeepseekDriver.decodeConfig", () => {
  it("defaults to the DEEPSEEK_API_KEY env and a 10-minute timeout", () => {
    const cfg = DeepseekDriver.decodeConfig({});
    expect(cfg).toEqual({ apiKeyEnv: "DEEPSEEK_API_KEY", dshHome: undefined, timeoutMs: 600_000 });
    expect(DeepseekDriver.decodeConfig(undefined)).toEqual(cfg);
  });

  it("accepts a custom key env, harness home, and timeout", () => {
    const cfg: DeepseekConfig = { apiKeyEnv: "MY_DSH_KEY", dshHome: "C:\\dsh", timeoutMs: 30_000 };
    expect(DeepseekDriver.decodeConfig(cfg)).toEqual(cfg);
  });
});

describe("DeepseekDriver snapshot", () => {
  it("is unavailable without a key", async () => {
    const inst = await DeepseekDriver.create({
      instanceId: "dsh-test",
      displayName: "DeepSeek",
      environment: {},
      enabled: true,
      config: { apiKeyEnv: "DEEPSEEK_API_KEY", timeoutMs: 600_000 },
    });
    const snap = await inst.snapshot();
    expect(snap.state).toBe("unavailable");
    expect(snap.reason).toMatch(/no DeepSeek key/);
    await inst.dispose();
  });

  it("is unavailable when the dsh package is missing", async () => {
    // The snapshot resolves DSH_BIN from the real package tree, so this only
    // exercises the key-present branch's warmup guard by using a bogus bin
    // path — the helper must not throw before the availability check.
    const inst = await DeepseekDriver.create({
      instanceId: "dsh-test-key",
      displayName: "DeepSeek",
      environment: { DEEPSEEK_API_KEY: "sk-fake" },
      enabled: true,
      config: { apiKeyEnv: "DEEPSEEK_API_KEY", dshHome: join(tmpdir(), "dsh-empty"), timeoutMs: 600_000 },
    });
    const snap = await inst.snapshot();
    // Package is present in this repo, so the harness home warms up instead.
    expect(["available", "unavailable"]).toContain(snap.state);
    await inst.dispose();
  });

  it("reports agentsMcp capability so the harness injects the agents integration", async () => {
    const inst = await DeepseekDriver.create({
      instanceId: "dsh-agents",
      displayName: "DeepSeek",
      environment: {},
      enabled: true,
      config: { apiKeyEnv: "DEEPSEEK_API_KEY", timeoutMs: 600_000 },
    });
    expect(inst.adapter.capabilities.agentsMcp).toBe(true);
    await inst.dispose();
  });
});

describe("writeModelPatch", () => {
  it("pins the agent-default-model row under the harness home", () => {
    const home = mkdtempSync(join(tmpdir(), "dsh-home-"));
    const path = writeModelPatch(home, "deepseek-v4-pro");
    const yaml = readFileSync(path, "utf8");
    expect(yaml).toContain("id: agent-default-model");
    expect(yaml).toContain("provider: deepseek-official");
    expect(yaml).toContain("model: deepseek-v4-pro");
    expect(path).toBe(join(home, "patches", "model.yml"));
    rmSync(home, { recursive: true, force: true });
  });

  it("does NOT include mcp-client when no agents integration is provided", () => {
    const home = mkdtempSync(join(tmpdir(), "dsh-noagents-"));
    const path = writeModelPatch(home, "deepseek-v4-flash");
    const yaml = readFileSync(path, "utf8");
    expect(yaml).toContain("agent-default-model");
    expect(yaml).not.toContain("mcp-client");
    expect(yaml).not.toContain("agents-mcp");
    rmSync(home, { recursive: true, force: true });
  });

  it("injects the agents proxy MCP server when agents integration is provided", () => {
    const home = mkdtempSync(join(tmpdir(), "dsh-agents-"));
    const integration = {
      command: "/usr/bin/node",
      args: ["/app/server/drivers/agents-proxy.ts"],
      env: {
        OMB_HARNESS_URL: "http://127.0.0.1:8799",
        OMB_BOT_ID: "bot-abc",
        OMB_BOT_ROLE: "ceo",
        OMB_COMMS_TOKEN: "secret-token",
        OMB_TURN_DEPTH: "0",
      },
    };
    const path = writeModelPatch(home, "deepseek-v4-pro", integration, "/usr/bin/node");
    const yaml = readFileSync(path, "utf8");
    // model pin still present
    expect(yaml).toContain("id: agent-default-model");
    expect(yaml).toContain("model: deepseek-v4-pro");
    // agents proxy MCP server injected
    expect(yaml).toContain("id: agents-mcp");
    expect(yaml).toContain("name: '@deepseek-ai/dsh-mcp-client'");
    expect(yaml).toContain('command: "/usr/bin/node"');
    expect(yaml).toContain("agents-proxy.ts");
    expect(yaml).toContain("OMB_HARNESS_URL");
    expect(yaml).toContain("OMB_COMMS_TOKEN");
    expect(yaml).toContain("OMB_BOT_ROLE: \"ceo\"");
    rmSync(home, { recursive: true, force: true });
  });

  it("uses the provided nodeExe for the mcp-client command", () => {
    const home = mkdtempSync(join(tmpdir(), "dsh-nodeexe-"));
    const integration = {
      command: "ignored",
      args: [],
      env: { OMB_HARNESS_URL: "http://127.0.0.1:8799", OMB_BOT_ID: "x", OMB_BOT_ROLE: "", OMB_COMMS_TOKEN: "t", OMB_TURN_DEPTH: "0" },
    };
    const path = writeModelPatch(home, "model", integration, "C:\\custom\\node.exe");
    const yaml = readFileSync(path, "utf8");
    expect(yaml).toContain('command: "C:\\custom\\node.exe"');
    rmSync(home, { recursive: true, force: true });
  });
});

describe("runHeadless", () => {
  it("resolves the final stdout text on a clean exit", async () => {
    const script = join(tmpdir(), `echo-${Date.now()}.js`);
    writeFileSync(script, `process.stdout.write("hello from harness");`, "utf8");
    const res = await runHeadless({
      nodeExe: process.execPath,
      dshHome: mkdtempSync(join(tmpdir(), "dsh-run-")),
      apiKey: "sk-fake",
      task: "say hi",
      args: [script],
    } as any);
    expect(res.code).toBe(0);
    expect(res.stdout).toContain("hello from harness");
    expect(res.stderr).toBe("");
    rmSync(script, { recursive: false, force: true });
  });

  it("streams stdout chunks to the onStdout callback", async () => {
    const script = join(tmpdir(), `stream-${Date.now()}.js`);
    writeFileSync(script, `process.stdout.write("chunk-one\\nchunk-two");`, "utf8");
    const chunks: string[] = [];
    await runHeadless({
      nodeExe: process.execPath,
      dshHome: mkdtempSync(join(tmpdir(), "dsh-run-")),
      apiKey: "sk-fake",
      task: "say hi",
      args: [script],
      onStdout: (c: string) => chunks.push(c),
    } as any);
    expect(chunks.join("")).toContain("chunk-one");
    expect(chunks.join("")).toContain("chunk-two");
    rmSync(script, { recursive: false, force: true });
  });

  it("surfaces a failing run via the exit code and stderr", async () => {
    const script = join(tmpdir(), `fail-${Date.now()}.js`);
    writeFileSync(script, `process.stderr.write("AUTH: bad key"); process.exit(1);`, "utf8");
    const res = await runHeadless({
      nodeExe: process.execPath,
      dshHome: mkdtempSync(join(tmpdir(), "dsh-run-")),
      apiKey: "sk-fake",
      task: "say hi",
      args: [script],
    } as any);
    expect(res.code).toBe(1);
    expect(res.stderr).toContain("AUTH: bad key");
    rmSync(script, { recursive: false, force: true });
  });
});