// Box provider contract: deterministic per-bot resolution (LIST once,
// then the cached id), command execution result shapes, and the
// panel-facing status/sleep/screenshot helpers. The API base URL is
// hardcoded in the module, so the tests stub global fetch with a tiny
// router keyed on method + path.
import { afterEach, describe, expect, it, vi } from "vitest";

import type { AppConfig } from "./config.ts";
import { boxConfigured, boxStatus, execOnBox, findBox, joinBox, provisionBox, runCommand, screenshotBox, sleepBox } from "./box.ts";

const cfg: AppConfig = { box: { token: "tok-test" } };

type Route = (init: RequestInit) => Response | Promise<Response>;

function stubApi(routes: Record<string, Route>) {
  const calls: Array<{ key: string; init: RequestInit }> = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init: RequestInit = {}) => {
      const path = String(url).replace("https://ascii.dev/api/box/v1", "");
      const key = `${init.method ?? "GET"} ${path}`;
      calls.push({ key, init });
      const route = routes[key];
      if (!route) throw new Error(`unrouted: ${key}`);
      return route(init);
    }),
  );
  return calls;
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status });
}

// mirrors boxNameFor — pins the deterministic naming contract
async function nameFor(botId: string) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(botId));
  const hash = [...new Uint8Array(digest)]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("")
    .slice(0, 6);
  return `ogb-${botId.slice(0, 8).toLowerCase().replace(/[^a-z0-9]/g, "")}-${hash}`;
}

// findBox caches box ids per botId at module scope — every test gets a
// fresh bot id so no state leaks between them
let seq = 0;
const freshBotId = () => `bot-${Date.now()}-${seq++}`;

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("boxConfigured", () => {
  it("requires a token", () => {
    expect(boxConfigured({})).toBe(false);
    expect(boxConfigured({ box: {} })).toBe(false);
    expect(boxConfigured(cfg)).toBe(true);
  });
});

describe("runCommand", () => {
  it("posts the command and reports ok on exit 0", async () => {
    const calls = stubApi({
      "POST /boxes/b1/commands": () => json({ exitCode: 0, stdout: "out", stderr: "" }),
    });
    const out = await runCommand(cfg, "b1", "echo hi");
    expect(out).toEqual({ ok: true, exitCode: 0, stdout: "out", stderr: "" });
    expect(JSON.parse(String(calls[0].init.body))).toEqual({ command: "echo hi" });
    expect((calls[0].init.headers as Record<string, string>).authorization).toBe("Bearer tok-test");
  });

  it("is not ok on a nonzero exit and survives a non-JSON body", async () => {
    stubApi({ "POST /boxes/b1/commands": () => json({ exitCode: 3, stdout: "", stderr: "bad" }) });
    expect(await runCommand(cfg, "b1", "false")).toMatchObject({ ok: false, exitCode: 3, stderr: "bad" });

    stubApi({ "POST /boxes/b1/commands": () => new Response("not json", { status: 500 }) });
    expect(await runCommand(cfg, "b1", "x")).toEqual({ ok: false, exitCode: null, stdout: "", stderr: "" });
  });
});

describe("findBox", () => {
  it("resolves by deterministic name from the listing, then goes direct", async () => {
    const botId = freshBotId();
    const name = await nameFor(botId);
    const calls = stubApi({
      "GET /boxes": () =>
        json({ boxes: [{ id: "other", name: "ogb-nope", state: "idle" }, { id: "b-hit", name, state: "idle" }] }),
      "GET /boxes/b-hit": () => json({ box: { id: "b-hit", name, state: "idle" } }),
    });

    const found = await findBox(cfg, botId);
    expect(found?.id).toBe("b-hit");

    const again = await findBox(cfg, botId);
    expect(again?.id).toBe("b-hit");
    expect(calls.map((c) => c.key)).toEqual(["GET /boxes", "GET /boxes/b-hit"]);
  });

  it("skips error-state boxes and returns null when nothing matches", async () => {
    const botId = freshBotId();
    const name = await nameFor(botId);
    stubApi({ "GET /boxes": () => json({ boxes: [{ id: "b-err", name, state: "error" }] }) });
    expect(await findBox(cfg, botId)).toBeNull();
  });

  it("falls back to the listing when the cached box is gone", async () => {
    const botId = freshBotId();
    const name = await nameFor(botId);
    let listed = 0;
    stubApi({
      "GET /boxes": () => json({ boxes: [{ id: `b-${++listed}`, name, state: "idle" }] }),
      "GET /boxes/b-1": () => json({ error: "not found" }, 404),
    });
    await findBox(cfg, botId); // caches b-1
    const second = await findBox(cfg, botId); // direct read 404s → re-list
    expect(second?.id).toBe("b-2");
  });
});

describe("boxStatus", () => {
  it("reports unconfigured without a token", async () => {
    expect(await boxStatus({}, "any")).toEqual({ configured: false, box: null });
  });

  it("maps the found box for the panel", async () => {
    const botId = freshBotId();
    const name = await nameFor(botId);
    stubApi({ "GET /boxes": () => json({ boxes: [{ id: "b9", name, state: "archived", desktopAvailable: true }] }) });
    expect(await boxStatus(cfg, botId)).toEqual({
      configured: true,
      box: { boxId: "b9", state: "archived", desktopAvailable: true },
    });
  });
});

describe("sleepBox / joinBox / provisionBox / execOnBox", () => {
  it("sleepBox stops the box and swallows stop failures", async () => {
    const botId = freshBotId();
    const name = await nameFor(botId);
    const calls = stubApi({
      "GET /boxes": () => json({ boxes: [{ id: "b2", name, state: "idle" }] }),
      "POST /boxes/b2/stop": () => new Response("nope", { status: 500 }),
    });
    expect(await sleepBox(cfg, botId)).toEqual({ ok: true });
    expect(calls.at(-1)?.key).toBe("POST /boxes/b2/stop");
  });

  it("sleepBox and joinBox throw when the bot has no box", async () => {
    stubApi({ "GET /boxes": () => json({ boxes: [] }) });
    await expect(sleepBox(cfg, freshBotId())).rejects.toThrow(/no computer/);
    stubApi({ "GET /boxes": () => json({ boxes: [] }) });
    await expect(joinBox(cfg, freshBotId())).rejects.toThrow(/no computer/);
  });

  it("joinBox waits for ready and mints a fresh desktop URL", async () => {
    const botId = freshBotId();
    const name = await nameFor(botId);
    stubApi({
      "GET /boxes": () => json({ boxes: [{ id: "b3", name, state: "idle" }] }),
      "GET /boxes/b3": () => json({ box: { id: "b3", name, state: "idle" } }),
      "POST /boxes/b3/desktop?vnc=1": () => json({ desktopUrl: "https://desk.example/session" }),
    });
    expect(await joinBox(cfg, botId)).toEqual({ joinUrl: "https://desk.example/session", state: "idle" });
  });

  it("provisionBox refuses without a token", async () => {
    await expect(provisionBox({}, freshBotId(), "Bot")).rejects.toThrow(/box provider not enabled/);
  });

  it("execOnBox runs the command and clips the output", async () => {
    const botId = freshBotId();
    const name = await nameFor(botId);
    stubApi({
      "GET /boxes": () => json({ boxes: [{ id: "b4", name, state: "running" }] }),
      "GET /boxes/b4": () => json({ box: { id: "b4", name, state: "running" } }),
      "POST /boxes/b4/commands": () => json({ exitCode: 0, stdout: "x".repeat(5000), stderr: "" }),
    });
    const out = await execOnBox(cfg, botId, "echo hi");
    expect(out.exitCode).toBe(0);
    expect(out.stdout).toHaveLength(4000);
  });
});

describe("screenshotBox", () => {
  it("captures on the box and reads the frame back as base64", async () => {
    stubApi({
      "POST /boxes/b5/commands": () => json({ exitCode: 0, stdout: "captured\n", stderr: "" }),
      "GET /boxes/b5/artifacts?path=%2Ftmp%2Fogb-panel.jpg": () =>
        new Response(Buffer.from("jpeg-bytes"), { status: 200 }),
    });
    const shot = await screenshotBox(cfg, "unused", "b5");
    expect(shot).toEqual({ png: Buffer.from("jpeg-bytes").toString("base64"), format: "jpeg" });
  });

  it("falls back to the files API when artifacts are unavailable", async () => {
    stubApi({
      "POST /boxes/b6/commands": () => json({ exitCode: 0, stdout: "captured", stderr: "" }),
      "GET /boxes/b6/artifacts?path=%2Ftmp%2Fogb-panel.jpg": () => new Response("no", { status: 404 }),
      "GET /boxes/b6/files?path=%2Ftmp%2Fogb-panel.jpg&encoding=base64": () => json({ content: "YmFzZTY0" }),
    });
    expect(await screenshotBox(cfg, "unused", "b6")).toEqual({ png: "YmFzZTY0", format: "jpeg" });
  });

  it("throws when the capture command fails", async () => {
    stubApi({
      "POST /boxes/b7/commands": () => json({ exitCode: 1, stdout: "", stderr: "no display found" }),
    });
    await expect(screenshotBox(cfg, "unused", "b7")).rejects.toThrow(/no display found/);
  });
});
