// computer-proxy — a minimal MCP stdio server the claude CLI spawns
// (agentcal's permission-proxy pattern, dedicated entry file so there is
// no argv-dispatch fork-bomb hazard). It gives the agent its bot's cloud
// computer (box.ascii.dev) as CUA-grade tools.
//
// The box interaction lives in server/computer-client.ts (shared with the
// prime-agent driver's in-process tools) — this file is only the MCP
// framing: one round trip per action, frame back inside the result.
// stdout is the MCP channel — never console.log here.
import { ComputerClient } from "./computer-client.ts";

const BOX_API = process.env.OGB_BOX_API ?? "https://ascii.dev/api/box/v1";
const boxId = process.env.OGB_BOX_ID ?? "";
const token = process.env.OGB_BOX_TOKEN ?? "";

const client = new ComputerClient(boxId, token, BOX_API);

const send = (obj: unknown): void => {
  process.stdout.write(JSON.stringify(obj) + "\n");
};
const text = (id: unknown, t: string, isError = false): void =>
  send({ jsonrpc: "2.0", id, result: { content: [{ type: "text", text: t }], ...(isError ? { isError: true } : {}) } });

/** An action result: the text plus the frame the action produced. */
function observed(id: unknown, note: string, frame: Frame | null) {
  const obs = client.observedContent(note, frame);
  if (!obs.image) return text(id, obs.text);
  send({
    jsonrpc: "2.0",
    id,
    result: {
      content: [
        { type: "text", text: obs.text },
        { type: "image", data: obs.image.data, mimeType: obs.image.mime },
      ],
    },
  });
}

import type { Frame } from "./computer-client.ts";

const OBSERVE_PROPS = {
  observe: {
    type: "boolean",
    description:
      "default true — return a fresh screenshot with the result. Set false only when chaining mechanical steps you don't need to see.",
  },
  settle_ms: { type: "number", description: "wait before the screenshot, default 350, max 3000" },
};

const TOOLS = [
  {
    name: "screenshot",
    description:
      "See the bot's cloud computer screen (returns an image). The desktop runs Chrome and a full Linux GUI. You usually do NOT need this after acting — click, type_text, press_key, scroll and open_url already return the resulting screen.",
    inputSchema: { type: "object", properties: {} },
  },
  {
    name: "click",
    description:
      "Click on the computer's screen and return the resulting screen. Use pixel coordinates exactly as they appear in the last frame you were given — any scaling to the real display is handled for you.",
    inputSchema: {
      type: "object",
      properties: {
        x: { type: "number" },
        y: { type: "number" },
        button: { type: "string", enum: ["left", "right"], description: "default left" },
        double: { type: "boolean", description: "double-click" },
        ...OBSERVE_PROPS,
      },
      required: ["x", "y"],
    },
  },
  {
    name: "type_text",
    description: "Type text at the current focus and return the resulting screen.",
    inputSchema: {
      type: "object",
      properties: { text: { type: "string" }, ...OBSERVE_PROPS },
      required: ["text"],
    },
  },
  {
    name: "press_key",
    description:
      'Press a key or chord and return the resulting screen. xdotool syntax: "Return", "Tab", "ctrl+c", "alt+F4", "ctrl+shift+t".',
    inputSchema: {
      type: "object",
      properties: { keys: { type: "string" }, ...OBSERVE_PROPS },
      required: ["keys"],
    },
  },
  {
    name: "scroll",
    description: "Scroll the screen up or down by N clicks and return the resulting screen.",
    inputSchema: {
      type: "object",
      properties: {
        direction: { type: "string", enum: ["up", "down"] },
        clicks: { type: "number", description: "default 3" },
        ...OBSERVE_PROPS,
      },
      required: ["direction"],
    },
  },
  {
    name: "computer_batch",
    description:
      "Run several UI actions in ONE go and return the screen at the end — much faster than separate calls (one round trip, one screenshot). Use it for mechanical sequences you can predict without looking in between, e.g. click a field, type, Tab, type, press Return. Stop the batch before anything whose outcome you need to see first.",
    inputSchema: {
      type: "object",
      properties: {
        actions: {
          type: "array",
          description: "in order; each is {action: click|type_text|press_key|scroll|wait, ...its params}",
          items: {
            type: "object",
            properties: {
              action: { type: "string", enum: ["click", "type_text", "press_key", "scroll", "wait"] },
              x: { type: "number" },
              y: { type: "number" },
              button: { type: "string", enum: ["left", "right"] },
              double: { type: "boolean" },
              text: { type: "string" },
              keys: { type: "string" },
              direction: { type: "string", enum: ["up", "down"] },
              clicks: { type: "number" },
              ms: { type: "number", description: "wait: milliseconds, max 5000" },
            },
            required: ["action"],
          },
        },
        ...OBSERVE_PROPS,
      },
      required: ["actions"],
    },
  },
  {
    name: "computer_exec",
    description:
      "Run a shell command on the bot's cloud computer (Linux, passwordless sudo, X11 desktop). Returns stdout/stderr/exit code — and, unlike the UI tools, no screenshot unless you ask for one.",
    inputSchema: {
      type: "object",
      properties: {
        command: { type: "string" },
        observe: {
          type: "boolean",
          description: "default false — set true to also return a screenshot (e.g. after launching a GUI app)",
        },
      },
      required: ["command"],
    },
  },
  {
    name: "open_url",
    description: "Open a URL in the computer's own Chrome and return the resulting screen.",
    inputSchema: {
      type: "object",
      properties: { url: { type: "string" }, ...OBSERVE_PROPS },
      required: ["url"],
    },
  },
];

const wantsFrame = (args: any) => args?.observe !== false;

async function call(id: unknown, name: string, args: any) {
  if (name === "screenshot") {
    const frame = await client.screenshotFrame();
    if (!frame) return text(id, `screenshot failed: could not capture a frame`, true);
    return send({
      jsonrpc: "2.0",
      id,
      result: { content: [{ type: "image", data: frame.data, mimeType: frame.mime }] },
    });
  }
  if (name === "click") {
    const x = Math.round(Number(args.x));
    const y = Math.round(Number(args.y));
    if (!Number.isFinite(x) || !Number.isFinite(y)) return text(id, "click needs numeric x,y", true);
    const what = `${args.double ? "double-clicked" : args.button === "right" ? "right-clicked" : "clicked"} ${x},${y}`;
    return actAndObserve(id, [{ ...args, action: "click" }], what, args);
  }
  if (name === "type_text") {
    const t = String(args.text ?? "");
    if (!t) return text(id, "nothing to type", true);
    return actAndObserve(id, [{ action: "type_text", text: t }], `typed ${t.length} chars`, args, 120_000);
  }
  if (name === "press_key") {
    const keys = String(args.keys ?? "").replace(/[^\w+]/g, "");
    if (!keys) return text(id, "press_key needs keys", true);
    return actAndObserve(id, [{ action: "press_key", keys }], `pressed ${keys}`, args);
  }
  if (name === "scroll") {
    const clicks = Math.min(Math.max(Math.round(Number(args.clicks) || 3), 1), 20);
    const direction = args.direction === "up" ? "up" : "down";
    return actAndObserve(id, [{ action: "scroll", direction, clicks }], `scrolled ${direction} ${clicks}`, args);
  }
  if (name === "computer_batch") {
    const actions = Array.isArray(args.actions) ? args.actions.slice(0, 24) : [];
    if (!actions.length) return text(id, "computer_batch needs a non-empty actions array", true);
    const summary = actions
      .map((a: any) =>
        a.action === "click"
          ? `click ${Math.round(Number(a.x))},${Math.round(Number(a.y))}`
          : a.action === "type_text"
            ? `type ${String(a.text ?? "").length} chars`
            : a.action === "press_key"
              ? `key ${a.keys}`
              : a.action === "scroll"
                ? `scroll ${a.direction ?? "down"}`
                : `wait ${Math.min(Number(a.ms) || 500, 5000)}ms`,
      )
      .join(" → ");
    return actAndObserve(id, actions, `ran ${actions.length} actions: ${summary}`, args, 180_000);
  }
  if (name === "computer_exec") {
    const command = String(args.command ?? "").slice(0, 4000);
    const out = await client.runOnBox(command, 120_000);
    const note = `exit ${out.exitCode}\n${out.stdout.slice(-6000)}${out.stderr ? `\n[stderr]\n${out.stderr.slice(-2000)}` : ""}`;
    if (args.observe !== true) return text(id, note);
    const frame = await client.screenshotFrame();
    return observed(id, note, frame);
  }
  if (name === "open_url") {
    const url = String(args.url ?? "");
    if (!/^https?:\/\//.test(url)) return text(id, "only http(s) URLs", true);
    const q = shellQuote(url.replace(/'/g, "%27"));
    const observe = wantsFrame(args);
    // launch, then poll for a browser window instead of a blind sleep —
    // a fast page returns in a fraction of the old fixed 3s
    const command = [
      "export DISPLAY=${DISPLAY:-:0}",
      client.geometryShell,
      `(google-chrome ${q} || chromium ${q} || chromium-browser ${q} || xdg-open ${q}) >/dev/null 2>&1 &`,
      'for i in 1 2 3 4 5 6 7 8 9 10 11 12; do xdotool search --onlyvisible --class "chrom" >/dev/null 2>&1 && break; sleep 0.25; done',
      observe ? client.captureBlock(600) : "true",
    ].join("; ");
    const out = await client.runOnBox(command, 60_000);
    if (!observe) return text(id, `opened ${url}`);
    return observed(id, `opened ${url}`, await client.frameFrom(out));
  }
  return text(id, `unknown tool ${name}`, true);
}

async function actAndObserve(id: unknown, actions: any[], note: string, args: any, timeoutMs = 60_000): Promise<void> {
  try {
    const { acted, frame, stderr } = await client.actAndObserve(actions, args, timeoutMs);
    if (!acted && !frame) {
      return text(
        id,
        `${note.replace(/^./, (c) => c.toLowerCase())} failed: ${stderr.slice(0, 200) || "exit unknown"}`,
        true,
      );
    }
    const full = acted ? note : `${note}\n(the action reported an error: ${stderr.slice(0, 160) || "no detail"})`;
    return observed(id, full, frame);
  } catch (e) {
    return text(id, (e as Error).message, true);
  }
}

async function handle(msg: any) {
  if (msg.method === "initialize") {
    return send({
      jsonrpc: "2.0",
      id: msg.id,
      result: {
        protocolVersion: msg.params?.protocolVersion ?? "2024-11-05",
        capabilities: { tools: {} },
        serverInfo: { name: "openmausbot-computer", version: "3" },
      },
    });
  }
  if (msg.method === "tools/list") return send({ jsonrpc: "2.0", id: msg.id, result: { tools: TOOLS } });
  if (msg.method === "tools/call") {
    try {
      return await call(msg.id, msg.params?.name, msg.params?.arguments ?? {});
    } catch (e) {
      return text(msg.id, `computer tool failed: ${(e as Error).message}`, true);
    }
  }
  if (String(msg.method ?? "").startsWith("notifications/")) return;
  if (msg.id != null) {
    send({ jsonrpc: "2.0", id: msg.id, error: { code: -32601, message: `method not found: ${msg.method}` } });
  }
}

import { shellQuote } from "./computer-client.ts";

let buf = "";
process.stdin.on("data", (chunk) => {
  buf += chunk;
  let nl;
  while ((nl = buf.indexOf("\n")) !== -1) {
    const line = buf.slice(0, nl);
    buf = buf.slice(nl + 1);
    if (!line.trim()) continue;
    try {
      void handle(JSON.parse(line));
    } catch {
      /* ignore malformed lines */
    }
  }
});
process.stdin.on("end", () => process.exit(0));