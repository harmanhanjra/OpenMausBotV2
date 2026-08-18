// Grok (API) driver contract tests — same shape as the NVIDIA suite:
// the driver is HTTP-only (xAI chat/completions with SSE streaming), so
// a tiny local http server plays the API. Windows-safe: no processes.
import { createServer, type Server } from "node:http";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import type { ProviderInstance } from "../contracts.ts";
import { recordEvents, type EventRecorder } from "../testing/events.ts";
import { GrokDriver } from "./grok.ts";

let server: Server;
let port = 0;
let requests: Array<{ path: string; body: any }> = [];
let hangResponses = false;

beforeEach(async () => {
  requests = [];
  hangResponses = false;
  server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c) => (raw += c));
    req.on("end", () => {
      const body = JSON.parse(raw);
      requests.push({ path: req.url ?? "/", body });
      if (hangResponses) {
        // headers out, then silence — lets the interrupt test abort mid-stream
        res.writeHead(200, { "content-type": "text/event-stream" });
        res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: "partial" } }] })}\n\n`);
        return;
      }
      if (!body.stream) {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(
          JSON.stringify({
            choices: [{ message: { content: "a tidy title" } }],
            usage: { prompt_tokens: 5, completion_tokens: 3 },
          }),
        );
        return;
      }
      res.writeHead(200, { "content-type": "text/event-stream" });
      const chunks = [
        { choices: [{ delta: { content: "hello " } }] },
        { choices: [{ delta: { content: "grok" } }] },
        { usage: { prompt_tokens: 12, completion_tokens: 2 } },
      ];
      for (const c of chunks) res.write(`data: ${JSON.stringify(c)}\n\n`);
      res.write("data: [DONE]\n\n");
      res.end();
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  port = typeof addr === "object" && addr ? addr.port : 0;
});

afterEach(async () => {
  server.closeAllConnections();
  await new Promise((resolve) => server.close(resolve));
});

const create = (overrides: { environment?: Record<string, string> } = {}): Promise<ProviderInstance> =>
  GrokDriver.create({
    instanceId: "grok-test",
    displayName: "Grok",
    environment: { XAI_API_KEY: "xai-test", ...overrides.environment },
    enabled: true,
    config: { url: `http://127.0.0.1:${port}/v1`, apiKeyEnv: "XAI_API_KEY" },
  });

describe("GrokDriver.decodeConfig", () => {
  it("defaults to the xAI endpoint and XAI_API_KEY", () => {
    expect(GrokDriver.decodeConfig({})).toEqual({ url: "https://api.x.ai/v1", apiKeyEnv: "XAI_API_KEY" });
    expect(GrokDriver.decodeConfig(undefined)).toEqual({ url: "https://api.x.ai/v1", apiKeyEnv: "XAI_API_KEY" });
  });

  it("accepts a custom url and key env", () => {
    expect(GrokDriver.decodeConfig({ url: "http://x/v1", apiKeyEnv: "K" })).toEqual({
      url: "http://x/v1",
      apiKeyEnv: "K",
    });
  });
});

describe("GrokDriver snapshot", () => {
  it("is unavailable without a key", async () => {
    const inst = await GrokDriver.create({
      instanceId: "grok-nokey",
      displayName: "Grok",
      environment: {},
      enabled: true,
      config: { url: `http://127.0.0.1:${port}/v1`, apiKeyEnv: "OMB_TEST_MISSING_KEY" },
    });
    const snap = await inst.snapshot();
    expect(snap.state).toBe("unavailable");
    expect(snap.reason).toMatch(/no xAI API key/);
    await inst.dispose();
  });

  it("is available with a key from the instance environment", async () => {
    const inst = await create();
    expect(await inst.snapshot()).toEqual({ state: "available", authenticated: true, version: null });
    await inst.dispose();
  });
});

describe("GrokDriver turns (local http server)", () => {
  let instance: ProviderInstance;
  let recorder: EventRecorder;

  beforeEach(async () => {
    instance = await create();
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
      model: "grok-4",
      system: "be brief",
      transcript: [
        { role: "user", text: "earlier" },
        { role: "assistant", text: "sure" },
      ],
    });
    await recorder.until((e) => e.type === "turn.completed");

    expect(recorder.events.map((e) => e.type)).toEqual([
      "turn.started",
      "session.started",
      "content.delta",
      "content.delta",
      "item.completed",
      "thread.token-usage.updated",
      "turn.completed",
    ]);
    expect(recorder.events.every((e) => e.turnId === turnId && e.provider === "grok")).toBe(true);

    const item: any = recorder.events.find((e) => e.type === "item.completed");
    expect(item).toMatchObject({ itemType: "assistant_text", text: "hello grok" });
    const usage: any = recorder.events.find((e) => e.type === "thread.token-usage.updated");
    expect(usage).toMatchObject({ input: 12, output: 2 });
    const done: any = recorder.events.find((e) => e.type === "turn.completed");
    expect(done).toMatchObject({ ok: true, stopReason: null });

    // the transcript folds into the messages array: system, history, then the turn
    expect(requests[0].body.messages).toEqual([
      { role: "system", content: "be brief" },
      { role: "user", content: "earlier" },
      { role: "assistant", content: "sure" },
      { role: "user", content: "hi" },
    ]);
    expect(requests[0].body.model).toBe("grok-4");
  });

  it("rejects a second turn on a busy thread", async () => {
    hangResponses = true;
    await instance.adapter.sendTurn({ threadId: "t-busy", text: "one" });
    await expect(instance.adapter.sendTurn({ threadId: "t-busy", text: "two" })).rejects.toThrow(
      /already running/,
    );
    expect(instance.adapter.hasSession("t-busy")).toBe(true);
    await instance.adapter.interruptTurn("t-busy");
    await recorder.until((e) => e.type === "turn.completed");
  });

  it("interrupting a turn completes it as interrupted, not an error", async () => {
    hangResponses = true;
    await instance.adapter.sendTurn({ threadId: "t-int", text: "hang" });
    await recorder.until((e) => e.type === "content.delta");
    await instance.adapter.interruptTurn("t-int");
    const done: any = await recorder.until((e) => e.type === "turn.completed");
    expect(done).toMatchObject({ ok: false, stopReason: "interrupted" });
    expect(recorder.events.some((e) => e.type === "runtime.error")).toBe(false);
    expect(instance.adapter.hasSession("t-int")).toBe(false);
  });

  it("surfaces an HTTP failure as runtime.error + a failed turn", async () => {
    const bad = await GrokDriver.create({
      instanceId: "grok-bad",
      displayName: "Grok",
      environment: { XAI_API_KEY: "xai-test" },
      enabled: true,
      // nothing listens here — the fetch itself fails
      config: { url: "http://127.0.0.1:1/v1", apiKeyEnv: "XAI_API_KEY" },
    });
    const rec = recordEvents(bad.adapter);
    await bad.adapter.sendTurn({ threadId: "t-err", text: "hi" });
    const done: any = await rec.until((e) => e.type === "turn.completed");
    expect(done).toMatchObject({ ok: false, stopReason: "error" });
    expect(rec.events.some((e) => e.type === "runtime.error")).toBe(true);
    rec.stop();
    await bad.dispose();
  });

  it("refuses to start a turn without a key", async () => {
    const nokey = await GrokDriver.create({
      instanceId: "grok-nokey2",
      displayName: "Grok",
      environment: {},
      enabled: true,
      config: { url: `http://127.0.0.1:${port}/v1`, apiKeyEnv: "OMB_TEST_MISSING_KEY" },
    });
    await expect(nokey.adapter.sendTurn({ threadId: "t", text: "hi" })).rejects.toThrow(/no xAI key/);
    await nokey.dispose();
  });

  it("generateText uses a non-streaming completion", async () => {
    expect(await instance.generateText!("name this bot")).toBe("a tidy title");
    expect(requests[0].body).toMatchObject({ stream: false, model: "grok-3-mini" });
  });
});
