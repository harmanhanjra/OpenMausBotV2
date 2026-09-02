// NVIDIA driver contract tests. The driver is HTTP-only (OpenAI-compatible
// chat/completions), so unlike the CLI-driver tests there's no fake CLI to
// script — a tiny local http server serves the SSE stream. Windows-safe:
// nothing here spawns processes.
import { createServer, type Server } from "node:http";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import type { ProviderInstance, RuntimeEvent } from "../contracts.ts";
import { recordEvents, type EventRecorder } from "../testing/events.ts";
import { NvidiaDriver } from "./nvidia.ts";

let server: Server;
let port = 0;

function sse(payloads: string[]): (req: any, res: any) => void {
  return (_req, res) => {
    res.writeHead(200, { "content-type": "text/event-stream" });
    for (const p of payloads) res.write(`data: ${p}\n\n`);
    res.write("data: [DONE]\n\n");
    res.end();
  };
}

beforeEach(async () => {
  server = createServer((req, res) => {
    const url = new URL(req.url ?? "/", "http://127.0.0.1");
    if (url.pathname === "/v1/models") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ object: "list", data: [] }));
      return;
    }
    // default: a streaming chat completion with a reasoning chunk + text
    const chunks = [
      { choices: [{ delta: { reasoning_content: "hmm" } }] },
      { choices: [{ delta: { content: "hello " } }] },
      { choices: [{ delta: { content: "nvidia" } }] },
      { usage: { prompt_tokens: 10, completion_tokens: 4 } },
    ];
    sse(chunks.map((c) => JSON.stringify(c)))(req, res);
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  port = typeof addr === "object" && addr ? addr.port : 0;
});

afterEach(async () => {
  await new Promise((resolve) => server.close(resolve));
});

describe("NvidiaDriver.decodeConfig", () => {
  it("defaults to the NVIDIA NIM cloud endpoint", () => {
    expect(NvidiaDriver.decodeConfig({})).toEqual({ url: "https://integrate.api.nvidia.com/v1", apiKeyEnv: "NVIDIA_API_KEY", models: undefined });
    expect(NvidiaDriver.decodeConfig(undefined)).toEqual({ url: "https://integrate.api.nvidia.com/v1", apiKeyEnv: "NVIDIA_API_KEY", models: undefined });
  });

  it("accepts a custom url and key env", () => {
    const cfg = NvidiaDriver.decodeConfig({ url: "http://127.0.0.1:8000/v1", apiKeyEnv: "MY_KEY" });
    expect(cfg.url).toBe("http://127.0.0.1:8000/v1");
    expect(cfg.apiKeyEnv).toBe("MY_KEY");
  });

  it("accepts a custom model catalog for self-hosted deployments", () => {
    const cfg = NvidiaDriver.decodeConfig({
      models: { default: "my-model", options: [{ id: "my-model", label: "My Model" }] },
    });
    expect(cfg.models?.default).toBe("my-model");
  });
});

describe("NvidiaDriver snapshot", () => {
  it("is unavailable without a key against the cloud endpoint", async () => {
    const inst = await NvidiaDriver.create({
      instanceId: "nvidia-test",
      displayName: "NVIDIA",
      environment: {},
      enabled: true,
      config: { url: "https://integrate.api.nvidia.com/v1", apiKeyEnv: "NVIDIA_API_KEY" },
    });
    const snap = await inst.snapshot();
    expect(snap.state).toBe("unavailable");
    expect(snap.reason).toMatch(/no NVIDIA key/);
    await inst.dispose();
  });

  it("is available with a key (no network probe)", async () => {
    const inst = await NvidiaDriver.create({
      instanceId: "nvidia-test",
      displayName: "NVIDIA",
      environment: { NVIDIA_API_KEY: "nvapi-test" },
      enabled: true,
      config: { url: "https://integrate.api.nvidia.com/v1", apiKeyEnv: "NVIDIA_API_KEY" },
    });
    const snap = await inst.snapshot();
    expect(snap.state).toBe("available");
    expect(snap.authenticated).toBe(true);
    await inst.dispose();
  });

  it("probes a local endpoint and reports available when reachable", async () => {
    const inst = await NvidiaDriver.create({
      instanceId: "nvidia-local",
      displayName: "NVIDIA",
      environment: {},
      enabled: true,
      config: { url: `http://127.0.0.1:${port}/v1`, apiKeyEnv: "NVIDIA_API_KEY" },
    });
    const snap = await inst.snapshot();
    expect(snap.state).toBe("available");
    expect(snap.authenticated).toBe(false);
    await inst.dispose();
  });
});

describe("NvidiaDriver model discovery", () => {
  it("refreshes the catalog from /v1/models and filters non-chat endpoints", async () => {
    const modelsServer = createServer((_req, res) => {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(
        JSON.stringify({
          object: "list",
          data: [
            { id: "meta/llama-3.3-70b-instruct", owned_by: "meta" },
            { id: "deepseek-ai/deepseek-v3.1", owned_by: "deepseek-ai" },
            { id: "nvidia/nv-embedqa-e5-v5", owned_by: "nvidia" },
            { id: "openai/gpt-oss-120b", owned_by: "openai" },
            { id: "nvidia/parakeet-tdt-0.6b", owned_by: "nvidia" },
            { id: "zhipuai/glm-5.1", owned_by: "zhipuai" },
            { id: "nvidia/nvclip", owned_by: "nvidia" },
            { id: "baai/bge-m3", owned_by: "baai" },
            { id: "nvidia/nemoretriever-parse", owned_by: "nvidia" },
            { id: "nvidia/llama-3.1-nemoguard-8b-topic-control", owned_by: "nvidia" },
          ],
        }),
      );
    });
    await new Promise<void>((resolve) => modelsServer.listen(0, "127.0.0.1", resolve));
    const addr = modelsServer.address();
    const p = typeof addr === "object" && addr ? addr.port : 0;
    try {
      const inst = await NvidiaDriver.create({
        instanceId: "nvidia-discovery",
        displayName: "NVIDIA",
        environment: {},
        enabled: true,
        config: { url: `http://127.0.0.1:${p}/v1`, apiKeyEnv: "NVIDIA_API_KEY" },
      });
      expect(inst.models.options.map((o) => o.id).sort()).toEqual([
        "deepseek-ai/deepseek-v3.1",
        "meta/llama-3.3-70b-instruct",
        "openai/gpt-oss-120b",
        "zhipuai/glm-5.1",
      ]);
      expect(inst.models.default).toBe("meta/llama-3.3-70b-instruct");
      await inst.dispose();
    } finally {
      await new Promise((resolve) => modelsServer.close(resolve));
    }
  });

  it("keeps an explicit config.models catalog verbatim", async () => {
    const inst = await NvidiaDriver.create({
      instanceId: "nvidia-pinned",
      displayName: "NVIDIA",
      environment: {},
      enabled: true,
      config: {
        url: "http://127.0.0.1:1/v1",
        apiKeyEnv: "NVIDIA_API_KEY",
        models: { default: "my-model", options: [{ id: "my-model", label: "My Model" }] },
      },
    });
    expect(inst.models.options).toEqual([{ id: "my-model", label: "My Model" }]);
    await inst.dispose();
  });
});

describe("NvidiaDriver turns (local http server)", () => {
  let instance: ProviderInstance;
  let recorder: EventRecorder;

  beforeEach(async () => {
    instance = await NvidiaDriver.create({
      instanceId: "nvidia-test",
      displayName: "NVIDIA",
      environment: {},
      enabled: true,
      config: { url: `http://127.0.0.1:${port}/v1`, apiKeyEnv: "NVIDIA_API_KEY" },
    });
    recorder = recordEvents(instance.adapter);
  });

  afterEach(async () => {
    recorder?.stop();
    await instance?.dispose();
  });

  it("normalizes a streaming turn into canonical events", async () => {
    const { turnId } = await instance.adapter.sendTurn({
      threadId: "t-happy",
      text: "hi",
      model: "meta/llama-3.3-70b-instruct",
      transcript: [{ role: "user", text: "earlier" }],
    });
    await recorder.until((e) => e.type === "turn.completed");

    const types = recorder.events.map((e) => e.type);
    expect(types).toEqual([
      "turn.started",
      "session.started",
      "content.delta", // reasoning
      "content.delta", // assistant text chunk 1
      "content.delta", // assistant text chunk 2
      "item.completed", // assistant_text
      "thread.token-usage.updated",
      "turn.completed",
    ]);
    expect(recorder.events.every((e) => e.turnId === turnId && e.provider === "nvidia")).toBe(true);

    const reasoning = recorder.events.find((e: any) => e.type === "content.delta" && e.streamKind === "reasoning_text");
    expect(reasoning).toMatchObject({ delta: "hmm" });
    const done = recorder.events.at(-1)!;
    expect(done).toMatchObject({ type: "turn.completed", ok: true });
    const usage = recorder.events.find((e) => e.type === "thread.token-usage.updated")!;
    expect(usage).toMatchObject({ input: 10, output: 4 });
  });

  it("rejects a second turn while one is in flight", async () => {
    await instance.adapter.sendTurn({ threadId: "t-busy", text: "go" });
    await expect(instance.adapter.sendTurn({ threadId: "t-busy", text: "again" })).rejects.toThrow(/already running/);
  });

  it("interrupts the active turn", async () => {
    await instance.adapter.sendTurn({ threadId: "t-stop", text: "go" });
    await instance.adapter.interruptTurn("t-stop");
    const done = await recorder.until((e) => e.type === "turn.completed");
    expect(done).toMatchObject({ ok: false, stopReason: "interrupted" });
  });

  it("reports agentsMcp capability so the harness injects the agents integration", () => {
    expect(instance.adapter.capabilities.agentsMcp).toBe(true);
  });
});

describe("NvidiaDriver tool-calling loop (local http server)", () => {
  let toolServer: Server;
  let toolPort = 0;

  it("executes a tool call and returns the final text answer", async () => {
    // Mock server: first request returns a tool_call, second returns text
    // Also mocks /api/internal/agents for the list_bots tool execution
    let callCount = 0;
    toolServer = createServer((req, res) => {
      const url = new URL(req.url ?? "/", "http://127.0.0.1");
      if (url.pathname.endsWith("/models")) {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ object: "list", data: [] }));
        return;
      }
      // Mock internal agents endpoint for the peer tool
      if (url.pathname === "/api/internal/agents") {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ bots: [{ id: "bot-b", name: "Helper", model: "test-model" }] }));
        return;
      }
      let body = "";
      req.on("data", (c: Buffer) => (body += c));
      req.on("end", () => {
        const parsed = JSON.parse(body);
        const hasTools = Array.isArray(parsed.tools) && parsed.tools.length > 0;
        callCount++;
        if (hasTools && callCount === 1) {
          // First call with tools: return a tool_call
          res.writeHead(200, { "content-type": "application/json" });
          res.end(JSON.stringify({
            choices: [{
              message: {
                role: "assistant",
                content: null,
                tool_calls: [{ id: "tc-1", type: "function", function: { name: "list_bots", arguments: "{}" } }],
              },
              finish_reason: "tool_calls",
            }],
            usage: { prompt_tokens: 20, completion_tokens: 5 },
          }));
        } else {
          // Second call (tool result appended): return final text
          res.writeHead(200, { "content-type": "application/json" });
          res.end(JSON.stringify({
            choices: [{ message: { role: "assistant", content: "I see 2 other bots." }, finish_reason: "stop" }],
            usage: { prompt_tokens: 30, completion_tokens: 8 },
          }));
        }
      });
    });
    await new Promise<void>((resolve) => toolServer.listen(0, "127.0.0.1", resolve));
    const addr = toolServer.address();
    toolPort = typeof addr === "object" && addr ? addr.port : 0;

    const inst = await NvidiaDriver.create({
      instanceId: "nvidia-tools",
      displayName: "NVIDIA",
      environment: {},
      enabled: true,
      config: { url: `http://127.0.0.1:${toolPort}/v1`, apiKeyEnv: "NVIDIA_API_KEY" },
    });
    const rec = recordEvents(inst.adapter);

    // Provide an agents integration with a fake COMMS_TOKEN
    const { turnId } = await inst.adapter.sendTurn({
      threadId: "t-tool",
      text: "list the bots",
      model: "meta/llama-3.3-70b-instruct",
      integrations: {
        agents: {
          command: "fake",
          args: [],
          env: {
            OMB_HARNESS_URL: `http://127.0.0.1:${toolPort}`,
            OMB_BOT_ID: "bot-a",
            OMB_BOT_ROLE: "",
            OMB_COMMS_TOKEN: "test-token",
            OMB_TURN_DEPTH: "0",
          },
        },
      },
    });

    // The turn should complete (tool call executed, then final text streamed)
    const done = await rec.until((e) => e.type === "turn.completed");
    expect(done).toMatchObject({ ok: true, turnId });

    // Should have tool item.started + item.completed for the tool call
    const toolStart = rec.events.find(
      (e): e is Extract<RuntimeEvent, { type: "item.started" }> =>
        e.type === "item.started" && e.itemType === "tool",
    );
    expect(toolStart).toBeTruthy();
    expect(toolStart!.title).toBe("list_bots");
    const toolEnd = rec.events.find((e) => e.type === "item.completed" && e.itemType === "tool");
    expect(toolEnd).toBeTruthy();
    expect(toolEnd!.ok).toBe(true);

    // Final text should be emitted
    const textItem = rec.events.find((e) => e.type === "item.completed" && e.itemType === "assistant_text");
    expect(textItem).toBeTruthy();

    rec.stop();
    await inst.dispose();
    await new Promise((resolve) => toolServer.close(resolve));
  });

  it("sends tools in the API request body when agents integration is present", async () => {
    let capturedBody: any = null;
    toolServer = createServer((req, res) => {
      const url = new URL(req.url ?? "/", "http://127.0.0.1");
      if (url.pathname.endsWith("/models")) {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ object: "list", data: [] }));
        return;
      }
      let body = "";
      req.on("data", (c: Buffer) => (body += c));
      req.on("end", () => {
        capturedBody = JSON.parse(body);
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({
          choices: [{ message: { role: "assistant", content: "No tools needed." }, finish_reason: "stop" }],
          usage: { prompt_tokens: 10, completion_tokens: 3 },
        }));
      });
    });
    await new Promise<void>((resolve) => toolServer.listen(0, "127.0.0.1", resolve));
    const addr = toolServer.address();
    toolPort = typeof addr === "object" && addr ? addr.port : 0;

    const inst = await NvidiaDriver.create({
      instanceId: "nvidia-tools-check",
      displayName: "NVIDIA",
      environment: {},
      enabled: true,
      config: { url: `http://127.0.0.1:${toolPort}/v1`, apiKeyEnv: "NVIDIA_API_KEY" },
    });
    const rec = recordEvents(inst.adapter);

    await inst.adapter.sendTurn({
      threadId: "t-tools-check",
      text: "hi",
      model: "meta/llama-3.3-70b-instruct",
      integrations: {
        agents: {
          command: "fake",
          args: [],
          env: {
            OMB_HARNESS_URL: `http://127.0.0.1:${toolPort}`,
            OMB_BOT_ID: "bot-a",
            OMB_BOT_ROLE: "",
            OMB_COMMS_TOKEN: "test-token",
            OMB_TURN_DEPTH: "0",
          },
        },
      },
    });
    await rec.until((e) => e.type === "turn.completed");

    // The captured request body should include tools
    expect(capturedBody).toBeTruthy();
    expect(Array.isArray(capturedBody.tools)).toBe(true);
    const toolNames = capturedBody.tools.map((t: any) => t.function.name);
    expect(toolNames).toContain("list_bots");
    expect(toolNames).toContain("ask_bot");

    rec.stop();
    await inst.dispose();
    await new Promise((resolve) => toolServer.close(resolve));
  });

  it("does NOT send tools when no agents integration is present", async () => {
    let capturedBody: any = null;
    toolServer = createServer((req, res) => {
      const url = new URL(req.url ?? "/", "http://127.0.0.1");
      if (url.pathname.endsWith("/models")) {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ object: "list", data: [] }));
        return;
      }
      let body = "";
      req.on("data", (c: Buffer) => (body += c));
      req.on("end", () => {
        capturedBody = JSON.parse(body);
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({
          choices: [{ message: { role: "assistant", content: "hello" }, finish_reason: "stop" }],
          usage: { prompt_tokens: 5, completion_tokens: 1 },
        }));
      });
    });
    await new Promise<void>((resolve) => toolServer.listen(0, "127.0.0.1", resolve));
    const addr = toolServer.address();
    toolPort = typeof addr === "object" && addr ? addr.port : 0;

    const inst = await NvidiaDriver.create({
      instanceId: "nvidia-no-tools",
      displayName: "NVIDIA",
      environment: {},
      enabled: true,
      config: { url: `http://127.0.0.1:${toolPort}/v1`, apiKeyEnv: "NVIDIA_API_KEY" },
    });
    const rec = recordEvents(inst.adapter);

    await inst.adapter.sendTurn({
      threadId: "t-no-tools",
      text: "hi",
      model: "meta/llama-3.3-70b-instruct",
    });
    await rec.until((e) => e.type === "turn.completed");

    expect(capturedBody).toBeTruthy();
    expect(capturedBody.tools).toBeUndefined();

    rec.stop();
    await inst.dispose();
    await new Promise((resolve) => toolServer.close(resolve));
  });
});
