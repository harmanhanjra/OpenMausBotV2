// NVIDIA driver contract tests. The driver is HTTP-only (OpenAI-compatible
// chat/completions), so unlike the CLI-driver tests there's no fake CLI to
// script — a tiny local http server serves the SSE stream. Windows-safe:
// nothing here spawns processes.
import { createServer, type Server } from "node:http";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import type { ProviderInstance } from "../contracts.ts";
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
});