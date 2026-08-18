// permission-proxy — the MCP stdio server the claude CLI spawns for
// --permission-prompt-tool (ported from agentcal's runPermissionProxy;
// dedicated entry file, so there is no argv-dispatch fork-bomb hazard).
// Forwards each ask over a unix socket to the broker living in the
// OpenMausBot server and waits for the human's answer.
//
//   approve   — the CLI calls this for any tool use its permission mode
//               would deny; the answer is the --permission-prompt-tool
//               JSON contract ({behavior:"allow"|"deny", …}).
//   ask_user  — the agent can pose a question mid-run and wait; the
//               human's words come back verbatim.
//
// stdout is the MCP channel — never console.log here.
import { connect } from "node:net";
import { randomUUID } from "node:crypto";

import { onJsonLines } from "./lines.ts";
import { send, serveMcpStdio } from "./mcp-stdio.ts";

const socketPath = process.argv[2] ?? "";

const waiting = new Map<string, (msg: any) => void>();
const conn = connect(socketPath);
const dead = () => {
  for (const resolve of waiting.values()) {
    resolve({ behavior: "deny", message: "OpenMausBot: permission broker unavailable — skip this action" });
  }
  waiting.clear();
};
conn.on("error", dead);
conn.on("close", dead);

onJsonLines(conn, (msg) => {
  if (msg.t !== "answer") return;
  waiting.get(msg.id)?.(msg);
  waiting.delete(msg.id);
});

const TOOLS = [
  {
    name: "approve",
    description: "Ask the OpenMausBot user whether a tool use is allowed",
    inputSchema: {
      type: "object",
      properties: {
        tool_name: { type: "string" },
        input: { type: "object" },
        tool_use_id: { type: "string" },
      },
      required: ["tool_name", "input"],
    },
  },
  {
    name: "ask_user",
    description:
      "Ask the human who owns this bot a question and wait for their answer. Use whenever you need a decision, a preference, missing information, or sign-off before doing something consequential — do not guess on things the owner would want to decide. Returns their answer as text.",
    inputSchema: {
      type: "object",
      properties: {
        question: { type: "string", description: "The question, with enough context to answer at a glance" },
        choices: {
          type: "array",
          items: { type: "string" },
          description: "Optional 2-5 suggested answers, shown as one-tap buttons",
        },
      },
      required: ["question"],
    },
  },
];

serveMcpStdio({
  name: "openmausbot-permissions",
  version: "1",
  tools: TOOLS,
  async call(id, name, args) {
    const askId = randomUUID();
    // the CLI may include its own suggested permission rules; on allow we
    // hand them straight back as updatedPermissions so claude stops asking
    // at its own layer — no invented rule syntax (agentcal)
    const suggestions = Array.isArray(args.permission_suggestions)
      ? args.permission_suggestions
      : Array.isArray(args.suggestions)
        ? args.suggestions
        : null;
    const isQuestion = name === "ask_user";
    const answer: any = await new Promise((resolve) => {
      waiting.set(askId, resolve);
      if (conn.destroyed) return dead();
      const ask = isQuestion
        ? { t: "ask", id: askId, kind: "question", tool: "ask_user", input: { question: args.question, choices: args.choices } }
        : { t: "ask", id: askId, tool: args.tool_name, input: args.input };
      try {
        conn.write(JSON.stringify(ask) + "\n");
      } catch {
        dead();
      }
    });
    const text = isQuestion
      ? answer.message || "No answer was given — use your best judgment."
      : JSON.stringify(
          answer.behavior === "allow"
            ? {
                behavior: "allow",
                updatedInput: args.input ?? {},
                ...(answer.always && suggestions ? { updatedPermissions: suggestions } : {}),
              }
            : { behavior: "deny", message: answer.message || "Denied from OpenMausBot" },
        );
    send({ jsonrpc: "2.0", id, result: { content: [{ type: "text", text }] } });
  },
});
