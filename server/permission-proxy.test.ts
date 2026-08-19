// permission-proxy contract: the MCP stdio server claude spawns for
// --permission-prompt-tool. A real proxy process speaks JSON-RPC over
// stdio while a fake broker answers asks over the unix socket / named
// pipe — the exact wiring the harness server uses per turn.
import { spawn, type ChildProcess } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { createServer, type Server } from "node:net";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

const PROXY = join(dirname(fileURLToPath(import.meta.url)), "permission-proxy.ts");

const socketPathFor = (dir: string) =>
  process.platform === "win32" ? `\\\\.\\pipe\\omb-test-perm-${Date.now()}` : join(dir, "perm.sock");

interface Ask {
  t: string;
  id: string;
  kind?: string;
  tool?: string;
  input?: any;
}

class Rig {
  child!: ChildProcess;
  broker!: Server;
  dir!: string;
  asks: Ask[] = [];
  private out = "";
  private responses = new Map<number, any>();
  private waiters: Array<() => void> = [];

  async start(answer?: (ask: Ask) => Record<string, unknown> | null, socketOverride?: string) {
    this.dir = mkdtempSync(join(tmpdir(), "omb-perm-test-"));
    const sock = socketPathFor(this.dir);
    this.broker = createServer((conn) => {
      let buf = "";
      conn.on("data", (chunk) => {
        buf += chunk;
        let nl;
        while ((nl = buf.indexOf("\n")) !== -1) {
          const ask: Ask = JSON.parse(buf.slice(0, nl));
          buf = buf.slice(nl + 1);
          this.asks.push(ask);
          const reply = answer?.(ask);
          if (reply) conn.write(JSON.stringify({ t: "answer", id: ask.id, ...reply }) + "\n");
        }
      });
    });
    await new Promise<void>((resolve) => this.broker.listen(sock, resolve));

    this.child = spawn(process.execPath, [PROXY, socketOverride ?? sock], { stdio: ["pipe", "pipe", "pipe"] });
    this.child.stdout!.on("data", (chunk) => {
      this.out += chunk;
      let nl;
      while ((nl = this.out.indexOf("\n")) !== -1) {
        const line = this.out.slice(0, nl);
        this.out = this.out.slice(nl + 1);
        if (!line.trim()) continue;
        const msg = JSON.parse(line);
        if (msg.id != null) this.responses.set(msg.id, msg);
        for (const w of this.waiters.splice(0)) w();
      }
    });
  }

  async rpc(msg: Record<string, unknown>, timeoutMs = 10_000): Promise<any> {
    this.child.stdin!.write(JSON.stringify(msg) + "\n");
    const id = msg.id as number;
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      const hit = this.responses.get(id);
      if (hit) return hit;
      if (Date.now() > deadline) throw new Error(`no response for id ${id}`);
      await new Promise<void>((resolve) => {
        const timer = setTimeout(resolve, 100);
        this.waiters.push(() => (clearTimeout(timer), resolve()));
      });
    }
  }

  async stop() {
    this.child?.kill();
    await new Promise((resolve) => this.broker?.close(resolve));
    rmSync(this.dir, { recursive: true, force: true });
  }
}

let rig: Rig;

afterEach(async () => {
  await rig?.stop();
});

describe("permission-proxy MCP surface", () => {
  beforeEach(async () => {
    rig = new Rig();
    await rig.start();
  });

  it("answers initialize with its server info", async () => {
    const res = await rig.rpc({ jsonrpc: "2.0", id: 1, method: "initialize", params: { protocolVersion: "2025-01-01" } });
    expect(res.result.serverInfo.name).toBe("openmausbot-permissions");
    expect(res.result.protocolVersion).toBe("2025-01-01");
  });

  it("lists the approve and ask_user tools", async () => {
    const res = await rig.rpc({ jsonrpc: "2.0", id: 2, method: "tools/list" });
    expect(res.result.tools.map((t: any) => t.name)).toEqual(["approve", "ask_user"]);
  });

  it("rejects unknown methods and ignores notifications", async () => {
    rig.child.stdin!.write(JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }) + "\n");
    const res = await rig.rpc({ jsonrpc: "2.0", id: 3, method: "resources/list" });
    expect(res.error.code).toBe(-32601);
  });
});

describe("permission-proxy approve", () => {
  it("forwards the ask and returns the allow contract with suggested rules", async () => {
    rig = new Rig();
    await rig.start(() => ({ behavior: "allow", always: true }));

    const res = await rig.rpc({
      jsonrpc: "2.0",
      id: 10,
      method: "tools/call",
      params: {
        name: "approve",
        arguments: {
          tool_name: "Bash",
          input: { command: "ls" },
          permission_suggestions: [{ type: "addRules" }],
        },
      },
    });
    expect(rig.asks[0]).toMatchObject({ t: "ask", tool: "Bash", input: { command: "ls" } });
    expect(JSON.parse(res.result.content[0].text)).toEqual({
      behavior: "allow",
      updatedInput: { command: "ls" },
      updatedPermissions: [{ type: "addRules" }],
    });
  });

  it("omits updatedPermissions on a plain (not always) allow", async () => {
    rig = new Rig();
    await rig.start(() => ({ behavior: "allow" }));
    const res = await rig.rpc({
      jsonrpc: "2.0",
      id: 11,
      method: "tools/call",
      params: { name: "approve", arguments: { tool_name: "Bash", input: {}, suggestions: [{ type: "addRules" }] } },
    });
    expect(JSON.parse(res.result.content[0].text)).toEqual({ behavior: "allow", updatedInput: {} });
  });

  it("returns a deny with the human's message", async () => {
    rig = new Rig();
    await rig.start(() => ({ behavior: "deny", message: "not on my watch" }));
    const res = await rig.rpc({
      jsonrpc: "2.0",
      id: 12,
      method: "tools/call",
      params: { name: "approve", arguments: { tool_name: "Write", input: { path: "/etc" } } },
    });
    expect(JSON.parse(res.result.content[0].text)).toEqual({ behavior: "deny", message: "not on my watch" });
  });

  it("denies with a broker-unavailable message when the socket is dead", async () => {
    rig = new Rig();
    await rig.start(undefined, join(mkdtempSync(join(tmpdir(), "omb-none-")), "nope.sock"));
    const res = await rig.rpc({
      jsonrpc: "2.0",
      id: 13,
      method: "tools/call",
      params: { name: "approve", arguments: { tool_name: "Bash", input: {} } },
    });
    expect(JSON.parse(res.result.content[0].text)).toMatchObject({
      behavior: "deny",
      message: expect.stringMatching(/broker unavailable/),
    });
  });
});

describe("permission-proxy ask_user", () => {
  it("forwards the question with choices and returns the answer verbatim", async () => {
    rig = new Rig();
    await rig.start(() => ({ message: "the second one" }));
    const res = await rig.rpc({
      jsonrpc: "2.0",
      id: 20,
      method: "tools/call",
      params: { name: "ask_user", arguments: { question: "which?", choices: ["a", "b"] } },
    });
    expect(rig.asks[0]).toMatchObject({
      kind: "question",
      tool: "ask_user",
      input: { question: "which?", choices: ["a", "b"] },
    });
    expect(res.result.content[0].text).toBe("the second one");
  });

  it("falls back to best-judgment text when the answer is empty", async () => {
    rig = new Rig();
    await rig.start(() => ({ message: "" }));
    const res = await rig.rpc({
      jsonrpc: "2.0",
      id: 21,
      method: "tools/call",
      params: { name: "ask_user", arguments: { question: "hm?" } },
    });
    expect(res.result.content[0].text).toMatch(/best judgment/);
  });
});
