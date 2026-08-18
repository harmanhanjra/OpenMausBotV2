// MCP-over-stdio server boilerplate, shared by the proxy entry files the
// agent CLIs spawn (computer-proxy, permission-proxy). Each of them is a
// tools-only MCP server speaking NDJSON JSON-RPC on stdio, and each used to
// carry its own copy of the same four things: the write helper, the
// initialize/tools/list replies, the notifications-are-ignored rule, and the
// stdin framing loop. A proxy now declares only its tools and its call().
//
// stdout is the MCP channel — never console.log from a proxy.
import { onJsonLines } from "./lines.ts";

export const send = (obj: unknown): void => {
  process.stdout.write(JSON.stringify(obj) + "\n");
};

/** A tools/call result carrying one text block. */
export const textResult = (id: unknown, t: string, isError = false): void =>
  send({ jsonrpc: "2.0", id, result: { content: [{ type: "text", text: t }], ...(isError ? { isError: true } : {}) } });

export interface McpStdioServer {
  /** serverInfo.name reported to the client. */
  name: string;
  version: string;
  tools: unknown[];
  /** Handle tools/call; reply with send()/textResult(). */
  call(id: unknown, name: unknown, args: any): void | Promise<void>;
  /** Text for a call() that threw. Default: `tool failed: <message>`. */
  callError?(e: Error): string;
}

/** Serve `server` on stdin/stdout until stdin closes. */
export function serveMcpStdio(server: McpStdioServer): void {
  const handle = async (msg: any): Promise<void> => {
    if (msg.method === "initialize") {
      return send({
        jsonrpc: "2.0",
        id: msg.id,
        result: {
          protocolVersion: msg.params?.protocolVersion ?? "2024-11-05",
          capabilities: { tools: {} },
          serverInfo: { name: server.name, version: server.version },
        },
      });
    }
    if (msg.method === "tools/list") return send({ jsonrpc: "2.0", id: msg.id, result: { tools: server.tools } });
    if (msg.method === "tools/call") {
      try {
        return await server.call(msg.id, msg.params?.name, msg.params?.arguments ?? {});
      } catch (e) {
        return textResult(msg.id, server.callError?.(e as Error) ?? `tool failed: ${(e as Error).message}`, true);
      }
    }
    if (String(msg.method ?? "").startsWith("notifications/")) return;
    if (msg.id != null) {
      send({ jsonrpc: "2.0", id: msg.id, error: { code: -32601, message: `method not found: ${msg.method}` } });
    }
  };

  onJsonLines(process.stdin, (msg) => void handle(msg));
  process.stdin.on("end", () => process.exit(0));
}
