// Composio client contract: MCP tool calls over the Connect endpoint
// (JSON and SSE answer shapes), connection status/removal/auth-link
// helpers on top of COMPOSIO_MANAGE_CONNECTIONS, and the marketplace
// catalog with its curated fallback + 10-minute cache.
//
// The v3 catalog URL is hardcoded in the module, so these tests stub
// global fetch instead of standing up a local server. The toolkit cache
// is module state — cache-sensitive tests run in file order (curated
// fallback before the API hit that populates it).
import { afterEach, describe, expect, it, vi } from "vitest";

import type { AppConfig } from "./config.ts";
import { authorizeService, composioTool, connectionStatus, listToolkits, removeService } from "./composio.ts";

const cfg = (composio?: AppConfig["composio"]): AppConfig => ({ composio });

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

function mcpResult(payload: unknown): Response {
  return jsonResponse({ jsonrpc: "2.0", id: 1, result: { content: [{ type: "text", text: JSON.stringify(payload) }] } });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("composioTool", () => {
  it("throws without a configured key", async () => {
    await expect(composioTool(cfg(), "X", {})).rejects.toThrow(/no Composio key configured/);
  });

  it("posts a tools/call and parses a JSON body", async () => {
    const fetchMock = vi.fn(async (_url: string, _init: RequestInit) => mcpResult({ hello: "world" }));
    vi.stubGlobal("fetch", fetchMock);

    const out = await composioTool(cfg({ key: "ck_test" }), "MY_TOOL", { a: 1 });
    expect(out).toEqual({ hello: "world" });

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("https://connect.composio.dev/mcp");
    expect((init.headers as Record<string, string>)["x-consumer-api-key"]).toBe("ck_test");
    const body = JSON.parse(String(init.body));
    expect(body).toMatchObject({ method: "tools/call", params: { name: "MY_TOOL", arguments: { a: 1 } } });
  });

  it("honors a configured url override", async () => {
    const fetchMock = vi.fn(async (_url: string, _init: RequestInit) => mcpResult({}));
    vi.stubGlobal("fetch", fetchMock);
    await composioTool(cfg({ key: "ck_test", url: "https://proxy.example/mcp" }), "T", {});
    expect(fetchMock.mock.calls[0][0]).toBe("https://proxy.example/mcp");
  });

  it("parses an SSE-framed answer", async () => {
    const payload = { jsonrpc: "2.0", id: 1, result: { content: [{ type: "text", text: '{"ok":true}' }] } };
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(`event: message\ndata: ${JSON.stringify(payload)}\n\n`, { status: 200 })),
    );
    expect(await composioTool(cfg({ key: "ck" }), "T", {})).toEqual({ ok: true });
  });

  it("wraps non-JSON tool text as { text }", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => jsonResponse({ jsonrpc: "2.0", id: 1, result: { content: [{ type: "text", text: "plain words" }] } })),
    );
    expect(await composioTool(cfg({ key: "ck" }), "T", {})).toEqual({ text: "plain words" });
  });

  it("surfaces MCP errors and HTTP failures", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ jsonrpc: "2.0", id: 1, error: { message: "boom" } })));
    await expect(composioTool(cfg({ key: "ck" }), "T", {})).rejects.toThrow("boom");

    vi.stubGlobal("fetch", vi.fn(async () => new Response("nope", { status: 502 })));
    await expect(composioTool(cfg({ key: "ck" }), "T", {})).rejects.toThrow("Composio MCP: HTTP 502");
  });
});

describe("connectionStatus", () => {
  it("maps active accounts to connected per slug", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        mcpResult({
          data: {
            results: {
              slack: { status: "INITIATED", accounts: [{ status: "ACTIVE" }] },
              github: { status: "ACTIVE", accounts: [] },
              gmail: { status: "expired", accounts: [{ status: "EXPIRED" }] },
            },
          },
        }),
      ),
    );
    const status = await connectionStatus(cfg({ key: "ck" }), ["slack", "github", "gmail", "notion"]);
    expect(status.slack).toEqual({ connected: true, status: "INITIATED" });
    expect(status.github).toEqual({ connected: true, status: "ACTIVE" });
    expect(status.gmail).toEqual({ connected: false, status: "expired" });
    expect(status.notion).toEqual({ connected: false, status: "unknown" });
  });
});

describe("removeService", () => {
  it("removes every connected account and reports the count", async () => {
    const calls: any[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, init: RequestInit) => {
        const body = JSON.parse(String(init.body));
        calls.push(body.params.arguments.toolkits[0]);
        if (body.params.arguments.toolkits[0].action === "list") {
          return mcpResult({ data: { results: { slack: { accounts: [{ id: "a1" }, { account_id: "a2" }, {}] } } } });
        }
        return mcpResult({ ok: true });
      }),
    );
    const out = await removeService(cfg({ key: "ck" }), "slack");
    expect(out).toEqual({ removed: 2 });
    expect(calls.map((c) => c.action)).toEqual(["list", "remove", "remove"]);
    expect(calls.slice(1).map((c) => c.account_id)).toEqual(["a1", "a2"]);
  });
});

describe("authorizeService", () => {
  it("prefers an auth-looking https URL from the response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        mcpResult({ note: "see https://example.com/docs then https://connect.composio.dev/auth/xyz" }),
      ),
    );
    const { url } = await authorizeService(cfg({ key: "ck" }), "slack");
    expect(url).toBe("https://connect.composio.dev/auth/xyz");
  });

  it("falls back to the first URL and throws when there is none", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => mcpResult({ note: "visit https://example.com/start" })));
    expect((await authorizeService(cfg({ key: "ck" }), "slack")).url).toBe("https://example.com/start");

    vi.stubGlobal("fetch", vi.fn(async () => mcpResult({ note: "no link here" })));
    await expect(authorizeService(cfg({ key: "ck" }), "slack")).rejects.toThrow(/no auth link/);
  });
});

describe("listToolkits", () => {
  it("falls back to the curated catalog without a key", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const { cards, source } = await listToolkits(cfg());
    expect(source).toBe("curated");
    expect(cards.map((c) => c.slug)).toContain("slack");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("falls back to curated when the API errors", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("nope", { status: 401 })));
    expect((await listToolkits(cfg({ apiKey: "ak_bad" }))).source).toBe("curated");
  });

  it("maps the v3 catalog and serves later calls from cache", async () => {
    const fetchMock = vi.fn(async (_url: string, _init: RequestInit) =>
      jsonResponse({
        items: [
          { slug: "SLACK", name: "Slack", meta: { description: "d".repeat(120), logo: "https://logo" } },
          { key: "github", name: "GitHub", description: "code" },
        ],
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const { cards, source } = await listToolkits(cfg({ apiKey: "ak_good" }));
    expect(source).toBe("api");
    expect(cards).toEqual([
      { slug: "slack", label: "Slack", blurb: "d".repeat(90), logo: "https://logo", domain: null },
      { slug: "github", label: "GitHub", blurb: "code", logo: null, domain: null },
    ]);
    expect(fetchMock.mock.calls[0][1].headers).toMatchObject({ "x-api-key": "ak_good" });

    // cached: no key, no fetch, still the API cards
    const again = await listToolkits(cfg());
    expect(again.source).toBe("api");
    expect(again.cards).toEqual(cards);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
