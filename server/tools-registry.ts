// Tools Registry — plugin system, MCP client, custom tool registry, agent orchestration.
import { EventEmitter } from "node:events";
import { Type, type TSchema } from "typebox";
import { memoryRAG } from "./memory-rag.ts";
import { ComputerClient, shellQuote, type Frame } from "./computer-client.ts";

export interface ToolDefinition {
  name: string;
  label: string;
  description: string;
  parameters: TSchema;
  execute: (id: string, params: Record<string, unknown>) => Promise<ToolResult>;
  permissions?: string[];
  tags?: string[];
  version?: string;
  author?: string;
}

export interface ToolResult {
  content: Array<{ type: "text"; text: string } | { type: "image"; data: string; mimeType: string }>;
  details: unknown;
  isError?: boolean;
}

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  result?: ToolResult;
  error?: string;
  startedAt: number;
  completedAt?: number;
}

export interface MCPServerConfig {
  name: string;
  command: string;
  args: string[];
  env?: Record<string, string>;
  transport?: "stdio" | "sse" | "websocket" | "http" | "streamable-http";
  url?: string; // for SSE/websocket/http
}

export interface MCPTool {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
}

export interface MCPResource {
  uri: string;
  name: string;
  description?: string;
  mimeType?: string;
}

export interface MCPServer {
  config: MCPServerConfig;
  tools: MCPTool[];
  resources: MCPResource[];
  connected: boolean;
  process?: import("child_process").ChildProcess;
  client?: MCPClient;
}

export interface AgentSkill {
  id: string;
  name: string;
  description: string;
  tools: string[]; // tool names this skill provides
  triggers?: string[]; // keywords that suggest this skill
  requiredPermissions?: string[];
}

export interface AgentWorkflow {
  id: string;
  name: string;
  description: string;
  steps: WorkflowStep[];
  triggers: WorkflowTrigger[];
}

export interface WorkflowStep {
  id: string;
  type: "tool" | "llm" | "condition" | "loop" | "parallel";
  tool?: string;
  params?: Record<string, unknown>;
  prompt?: string;
  condition?: string;
  steps?: WorkflowStep[];
  iterations?: number;
}

export interface WorkflowTrigger {
  type: "keyword" | "schedule" | "event" | "webhook";
  pattern?: string;
  cron?: string;
  event?: string;
}

class MCPClient extends EventEmitter {
  private server: MCPServer;
  private requestId = 0;
  private pending = new Map<number, { resolve: (v: any) => void; reject: (e: Error) => void }>();

  constructor(server: MCPServer) {
    super();
    this.server = server;
  }

  async connect(): Promise<void> {
    if (this.server.config.transport === "stdio") {
      await this.connectStdio();
    } else if (this.server.config.transport === "sse") {
      await this.connectSSE();
    } else if (this.server.config.transport === "http" || this.server.config.transport === "streamable-http") {
      await this.connectHTTP();
    } else {
      throw new Error(`Unsupported transport: ${this.server.config.transport}`);
    }
    this.server.connected = true;
    this.emit("connected");
  }

  private async connectStdio(): Promise<void> {
    const { spawn } = await import("child_process");
    const proc = spawn(this.server.config.command, this.server.config.args, {
      env: { ...process.env, ...this.server.config.env },
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.server.process = proc;

    let buffer = "";
    proc.stdout?.on("data", (data) => {
      buffer += data.toString();
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (line.trim()) this.handleMessage(line);
      }
    });
    proc.stderr?.on("data", (data) => this.emit("stderr", data.toString()));
    proc.on("exit", (code) => {
      this.server.connected = false;
      this.emit("exit", code);
    });

    // Initialize
    await this.request("initialize", {
      protocolVersion: "2024-11-05",
      capabilities: { tools: {}, resources: {} },
      clientInfo: { name: "openmausbot", version: "0.1.0" },
    });
    await this.notify("notifications/initialized", {});
  }

  private async connectSSE(): Promise<void> {
    // SSE connection for remote MCP servers
    throw new Error("SSE transport not yet implemented");
  }

  /** Streamable HTTP transport (MCP 2025-06-18): JSON-RPC 2.0 POSTs to a
   * single endpoint. Used by self-hosted servers such as open-computer-use
   * (http://localhost:8081/mcp). Each response may be a plain JSON body or
   * an SSE stream; session id rides the Mcp-Session-Id header. */
  private httpUrl = "";
  private sessionId: string | null = null;
  private httpPending = new Map<number, { resolve: (v: any) => void; reject: (e: Error) => void }>();
  private httpNotify: Array<Record<string, unknown>> = [];

  private async connectHTTP(): Promise<void> {
    this.httpUrl = this.server.config.url ?? "";
    if (!/^https?:\/\//.test(this.httpUrl)) {
      throw new Error(`http transport requires a url, got: ${this.httpUrl}`);
    }
    // Initialize — capture session id from response headers
    const initResult = await this.httpRequest("initialize", {
      protocolVersion: "2024-11-05",
      capabilities: { tools: {}, resources: {} },
      clientInfo: { name: "openmausbot", version: "0.1.0" },
    });
    // flush notifications queued during init
    for (const n of this.httpNotify) {
      await this.httpSend(n);
    }
    this.httpNotify = [];
    void initResult;
  }

  private async httpRequest(method: string, params: Record<string, unknown>): Promise<any> {
    const id = ++this.requestId;
    return new Promise((resolve, reject) => {
      this.httpPending.set(id, { resolve, reject });
      this.httpSend({ jsonrpc: "2.0", id, method, params }).catch(reject);
      setTimeout(() => {
        if (this.httpPending.has(id)) {
          this.httpPending.delete(id);
          reject(new Error(`MCP HTTP request timeout: ${method}`));
        }
      }, 120_000);
    });
  }

  /** POST one JSON-RPC message to the streamable-HTTP endpoint and resolve
   * any pending request from the (JSON or SSE) response. */
  private async httpSend(msg: Record<string, unknown>): Promise<void> {
    const headers: Record<string, string> = {
      "content-type": "application/json",
      accept: "application/json, text/event-stream",
      ...(this.server.config.env ?? {}),
    };
    if (this.sessionId) headers["mcp-session-id"] = this.sessionId;
    let res: Response;
    try {
      res = await fetch(this.httpUrl, {
        method: "POST",
        headers,
        body: JSON.stringify(msg),
        signal: AbortSignal.timeout(120_000),
      });
    } catch (e) {
      this.rejectHttp(msg.id, e instanceof Error ? e : new Error(String(e)));
      return;
    }
    const sid = res.headers.get("mcp-session-id");
    if (sid) this.sessionId = sid;

    const ctype = (res.headers.get("content-type") ?? "").toLowerCase();
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      this.rejectHttp(msg.id, new Error(`MCP HTTP ${res.status}: ${text.slice(0, 500)}`));
      return;
    }
    if (ctype.includes("text/event-stream")) {
      const body = await res.text().catch(() => "");
      this.parseHttpSSE(body);
    } else {
      const body = await res.json().catch(() => null);
      this.handleHttpMessage(body);
    }
  }

  /** Streamable-HTTP servers may push server-initiated messages; keep the
   * line protocol of connectStdio's handler for notifications. */
  private parseHttpSSE(body: string): void {
    let data = "";
    for (const raw of body.split("\n")) {
      const line = raw.replace(/\r$/, "");
      if (line.startsWith("data:")) {
        data += (line.startsWith("data: ") ? line.slice(6) : line.slice(5)) + "\n";
        // `data: [DONE]` terminates a JSON-stream response
        if (data.trim() === "[DONE]") {
          data = "";
          continue;
        }
        continue;
      }
      if (line.trim() === "" && data.trim()) {
        const parsed = data.trim().split("\n").map((p) => {
          try {
            return JSON.parse(p);
          } catch {
            return null;
          }
        });
        for (const p of parsed) if (p) this.handleHttpMessage(p);
        data = "";
      }
    }
  }

  private handleHttpMessage(msg: any): void {
    if (!msg) return;
    if (msg.id !== undefined && this.httpPending.has(msg.id)) {
      const { resolve, reject } = this.httpPending.get(msg.id)!;
      this.httpPending.delete(msg.id);
      if (msg.error) reject(new Error(msg.error.message));
      else resolve(msg.result);
    } else if (msg.method) {
      this.emit("notification", msg);
    }
  }

  private rejectHttp(id: unknown, err: Error): void {
    if (typeof id === "number" && this.httpPending.has(id)) {
      const { reject } = this.httpPending.get(id)!;
      this.httpPending.delete(id);
      reject(err);
    }
  }

  private handleMessage(line: string): void {
    try {
      const msg = JSON.parse(line);
      if (msg.id !== undefined && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id)!;
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(msg.error.message));
        else resolve(msg.result);
      } else if (msg.method) {
        this.emit("notification", msg);
      }
    } catch {
      // Ignore parse errors
    }
  }

  async request(method: string, params: Record<string, unknown>): Promise<any> {
    if (this.isHTTP) return this.httpRequest(method, params);
    const id = ++this.requestId;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.send({ jsonrpc: "2.0", id, method, params });
      // Timeout
      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error(`MCP request timeout: ${method}`));
        }
      }, 120000);
    });
  }

  private get isHTTP(): boolean {
    return this.server.config.transport === "http" || this.server.config.transport === "streamable-http";
  }

  async notify(method: string, params: Record<string, unknown>): Promise<void> {
    if (this.isHTTP) {
      const msg = { jsonrpc: "2.0", method, params };
      if (!this.httpUrl) {
        this.httpNotify.push(msg);
        return;
      }
      await this.httpSend(msg).catch(() => null);
      return;
    }
    this.send({ jsonrpc: "2.0", method, params });
  }

  private send(msg: Record<string, unknown>): void {
    if (this.server.process?.stdin) {
      this.server.process.stdin.write(JSON.stringify(msg) + "\n");
    }
  }

  async listTools(): Promise<MCPTool[]> {
    const result = await this.request("tools/list", {});
    return result.tools ?? [];
  }

  async callTool(name: string, args: Record<string, unknown>): Promise<any> {
    return this.request("tools/call", { name, arguments: args });
  }

  async listResources(): Promise<MCPResource[]> {
    const result = await this.request("resources/list", {});
    return result.resources ?? [];
  }

  async readResource(uri: string): Promise<any> {
    return this.request("resources/read", { uri });
  }

  async disconnect(): Promise<void> {
    if (this.server.process) {
      this.server.process.kill();
      this.server.process = undefined;
    }
    this.server.connected = false;
    this.server.client = undefined;
    this.emit("disconnected");
  }
}

/** Normalize an MCP tool result into a ToolResult. Text content stays as
 * text; image content is inlined as a data: URL so the UI can render it
 * (screenshots from browser-use etc.). */
export function mcpResultToToolResult(result: any): ToolResult {
  const content = result?.content ?? result;
  if (!Array.isArray(content)) {
    return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }], details: result };
  }
  const textParts: string[] = [];
  let dataUrl: string | null = null;
  for (const c of content) {
    if (!c) continue;
    if (c.type === "text") {
      textParts.push(c.text ?? "");
    } else if (c.type === "image" && c.data) {
      dataUrl = `data:${c.mimeType ?? "image/png"};base64,${c.data}`;
    }
  }
  if (dataUrl) {
    textParts.push(`\n![screenshot](${dataUrl})`);
  }
  return {
    content: [{ type: "text", text: textParts.join("\n") || "(no output)" }],
    details: result,
  };
}

export class ToolsRegistry extends EventEmitter {
  private tools = new Map<string, ToolDefinition>();
  private mcpServers = new Map<string, MCPServer>();
  private skills = new Map<string, AgentSkill>();
  private workflows = new Map<string, AgentWorkflow>();
  private toolPermissions = new Map<string, Set<string>>(); // toolName -> allowed roles
  private executionHooks: Array<(call: ToolCall) => Promise<void>> = [];
  /** The bot's cloud computer (box.ascii.dev) the computer/browser tools
   * drive. Bound by the harness at turn dispatch when a box is provisioned. */
  private computerClient: ComputerClient | null = null;

  /** Bind the current bot's cloud computer so the computer_* / browser_*
   * tools can act on it. Call with null/undefined to unbind. */
  bindComputer(boxId: string | null, token: string | null): void {
    this.computerClient = boxId && token ? new ComputerClient(boxId, token) : null;
    this.emit("computerBound", this.computerClient ? { boxId } : null);
  }

  isComputerBound(): boolean {
    return this.computerClient !== null;
  }

  // ── Tool Registration ────────────────────────────────────────────────────

  registerTool(tool: ToolDefinition): void {
    if (this.tools.has(tool.name)) {
      throw new Error(`Tool already registered: ${tool.name}`);
    }
    this.tools.set(tool.name, tool);
    this.emit("toolRegistered", tool);
  }

  unregisterTool(name: string): boolean {
    const tool = this.tools.get(name);
    if (!tool) return false;
    this.tools.delete(name);
    this.emit("toolUnregistered", name);
    return true;
  }

  getTool(name: string): ToolDefinition | undefined {
    return this.tools.get(name);
  }

  listTools(): ToolDefinition[] {
    return [...this.tools.values()];
  }

  getToolsByTag(tag: string): ToolDefinition[] {
    return [...this.tools.values()].filter((t) => t.tags?.includes(tag));
  }

  // ── Tool Permissions ────────────────────────────────────────────────────

  setToolPermissions(toolName: string, roles: string[]): void {
    this.toolPermissions.set(toolName, new Set(roles));
  }

  checkPermission(toolName: string, role: string): boolean {
    const allowed = this.toolPermissions.get(toolName);
    if (!allowed) return true; // no restrictions = allowed
    return allowed.has(role) || allowed.has("*");
  }

  // ── Execution Hooks ──────────────────────────────────────────────────────

  addExecutionHook(hook: (call: ToolCall) => Promise<void>): void {
    this.executionHooks.push(hook);
  }

  private async runHooks(call: ToolCall): Promise<void> {
    for (const hook of this.executionHooks) {
      await hook(call);
    }
  }

  // ── Tool Execution ──────────────────────────────────────────────────────

  async executeTool(name: string, params: Record<string, unknown>, context?: { role?: string; userId?: string }): Promise<ToolResult> {
    const tool = this.tools.get(name);
    if (!tool) {
      return { content: [{ type: "text", text: `Tool not found: ${name}` }], details: null, isError: true };
    }

    if (context?.role && !this.checkPermission(name, context.role)) {
      return { content: [{ type: "text", text: `Permission denied for tool: ${name}` }], details: null, isError: true };
    }

    const call: ToolCall = {
      id: `call-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      name,
      arguments: params,
      startedAt: Date.now(),
    };

    await this.runHooks(call);

    try {
      const result = await tool.execute(call.id, params);
      call.result = result;
      call.completedAt = Date.now();
      this.emit("toolExecuted", call);
      return result;
    } catch (error) {
      call.error = error instanceof Error ? error.message : String(error);
      call.completedAt = Date.now();
      this.emit("toolError", call);
      return { content: [{ type: "text", text: `Tool execution failed: ${call.error}` }], details: null, isError: true };
    }
  }

  // ── MCP Server Management ────────────────────────────────────────────────

  async addMCPServer(config: MCPServerConfig): Promise<MCPServer> {
    if (this.mcpServers.has(config.name)) {
      throw new Error(`MCP server already exists: ${config.name}`);
    }
    const server: MCPServer = { config, tools: [], resources: [], connected: false };
    this.mcpServers.set(config.name, server);

    const client = new MCPClient(server);
    server.client = client;

    client.on("connected", async () => {
      try {
        server.tools = await client.listTools();
      } catch (e) {
        server.tools = [];
        this.emit("mcpError", config.name, `listTools: ${e instanceof Error ? e.message : String(e)}`);
      }
      try {
        server.resources = await client.listResources();
      } catch (e) {
        // Not every MCP server implements resources/list
        server.resources = [];
      }
      // Auto-register MCP tools
      for (const mcpTool of server.tools) {
        this.registerMCPTool(config.name, mcpTool, client);
      }
      this.emit("mcpConnected", config.name);
    });

    client.on("exit", (code) => {
      server.connected = false;
      this.emit("mcpDisconnected", config.name, code);
    });

    await client.connect();
    return server;
  }

  private registerMCPTool(serverName: string, mcpTool: MCPTool, client: MCPClient): void {
    const toolName = `mcp.${serverName}.${mcpTool.name}`;
    if (this.tools.has(toolName)) return;

    this.registerTool({
      name: toolName,
      label: mcpTool.name,
      description: `[MCP:${serverName}] ${mcpTool.description}`,
      parameters: mcpTool.inputSchema as TSchema,
      async execute(_id, params) {
        const result = await client.callTool(mcpTool.name, params);
        return mcpResultToToolResult(result);
      },
      tags: ["mcp", serverName],
    });
  }

  async removeMCPServer(name: string): Promise<boolean> {
    const server = this.mcpServers.get(name);
    if (!server) return false;
    await server.client?.disconnect();
    // Unregister its tools
    for (const toolName of [...this.tools.keys()]) {
      if (toolName.startsWith(`mcp.${name}.`)) this.unregisterTool(toolName);
    }
    this.mcpServers.delete(name);
    return true;
  }

  getMCPServer(name: string): MCPServer | undefined {
    return this.mcpServers.get(name);
  }

  listMCPServers(): MCPServer[] {
    return [...this.mcpServers.values()];
  }

  // ── MCP Presets ─────────────────────────────────────────────────────────
  /** Built-in MCP server recipes shipped with the app. `ensurePresetMCP`
   * wires them in without the user hand-typing a config.
   *
   * - browser-use: real Chrome automation via CDP (browser-use.com). stdio
   *   transport; requires the `browser-use` CLI on PATH (`pip install
   *   browser-use` / `uv tool install browser-use`).
   * - open-computer-use: self-hosted Docker sandbox with bash, browser,
   *   docs and sub-agents (github.com/Wide-Moat/open-computer-use). Streamable
   *   HTTP transport; requires `docker compose up` on port 8081. */
  private static PRESETS: Record<string, MCPServerConfig> = {
    "browser-use": {
      name: "browser-use",
      command: "browser-use",
      args: ["--cli-mcp"],
      transport: "stdio",
      env: { PYTHONIOENCODING: "utf-8" },
    },
    "open-computer-use": {
      name: "open-computer-use",
      command: "",
      args: [],
      transport: "streamable-http",
      url: process.env.OCU_URL ?? "http://localhost:8081/mcp",
      env: process.env.OCU_API_KEY ? { Authorization: `Bearer ${process.env.OCU_API_KEY}` } : undefined,
    },
  };

  listMCPPresets(): { name: string; config: Omit<MCPServerConfig, "env">; installed: boolean }[] {
    return Object.entries(ToolsRegistry.PRESETS).map(([name, config]) => ({
      name,
      config: { name, command: config.command, args: config.args, transport: config.transport, url: config.url },
      installed: true,
    }));
  }

  /** Add a preset MCP server by name. Rejects if the binary is not on PATH
   * (stdio presets) so the app never spawns a phantom process. */
  async ensurePresetMCP(name: string): Promise<MCPServer | null> {
    const preset = ToolsRegistry.PRESETS[name];
    if (!preset) throw new Error(`Unknown MCP preset: ${name}`);
    if (this.mcpServers.has(name)) return this.mcpServers.get(name) ?? null;
    if (preset.transport === "stdio") {
      const { spawnSync } = await import("child_process");
      const probe = spawnSync(preset.command, ["--version"], { timeout: 20_000, windowsHide: true });
      if (probe.error) {
        this.emit("mcpPresetUnavailable", name, `"${preset.command}" not found on PATH`);
        return null;
      }
    }
    return this.addMCPServer(preset);
  }

  // ── Skills ────────────────────────────────────────────────────────────────

  registerSkill(skill: AgentSkill): void {
    this.skills.set(skill.id, skill);
    this.emit("skillRegistered", skill);
  }

  getSkill(id: string): AgentSkill | undefined {
    return this.skills.get(id);
  }

  listSkills(): AgentSkill[] {
    return [...this.skills.values()];
  }

  suggestSkills(query: string): AgentSkill[] {
    const lower = query.toLowerCase();
    return [...this.skills.values()].filter((s) =>
      s.triggers?.some((t) => lower.includes(t.toLowerCase())) ?? false
    );
  }

  // ── Workflows ────────────────────────────────────────────────────────────

  registerWorkflow(workflow: AgentWorkflow): void {
    this.workflows.set(workflow.id, workflow);
    this.emit("workflowRegistered", workflow);
  }

  getWorkflow(id: string): AgentWorkflow | undefined {
    return this.workflows.get(id);
  }

  listWorkflows(): AgentWorkflow[] {
    return [...this.workflows.values()];
  }

  async executeWorkflow(workflowId: string, initialParams: Record<string, unknown>): Promise<Record<string, unknown>> {
    const workflow = this.workflows.get(workflowId);
    if (!workflow) throw new Error(`Workflow not found: ${workflowId}`);

    const context: Record<string, unknown> = { ...initialParams };
    for (const step of workflow.steps) {
      await this.executeStep(step, context);
    }
    return context;
  }

  private async executeStep(step: WorkflowStep, context: Record<string, unknown>): Promise<void> {
    switch (step.type) {
      case "tool": {
        if (!step.tool) throw new Error("Tool step missing tool name");
        const params = this.resolveParams(step.params ?? {}, context);
        const result = await this.executeTool(step.tool, params);
        context[step.id] = result;
        break;
      }
      case "llm": {
        // Would integrate with conversation engine
        context[step.id] = { prompt: step.prompt, note: "LLM step not yet implemented" };
        break;
      }
      case "condition": {
        // Simple condition evaluation
        const shouldRun = this.evaluateCondition(step.condition ?? "true", context);
        if (shouldRun && step.steps) {
          for (const subStep of step.steps) await this.executeStep(subStep, context);
        }
        break;
      }
      case "loop": {
        const iterations = step.iterations ?? 1;
        for (let i = 0; i < iterations; i++) {
          context.$index = i;
          if (step.steps) {
            for (const subStep of step.steps) await this.executeStep(subStep, context);
          }
        }
        break;
      }
      case "parallel": {
        if (step.steps) {
          await Promise.all(step.steps.map((s) => this.executeStep(s, { ...context })));
        }
        break;
      }
    }
  }

  private resolveParams(params: Record<string, unknown>, context: Record<string, unknown>): Record<string, unknown> {
    const resolved: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(params)) {
      if (typeof value === "string" && value.startsWith("$")) {
        // Reference to context variable
        const path = value.slice(1).split(".");
        let val: unknown = context;
        for (const p of path) {
          val = (val as Record<string, unknown>)?.[p];
          if (val === undefined) break;
        }
        resolved[key] = val ?? value;
      } else {
        resolved[key] = value;
      }
    }
    return resolved;
  }

  private evaluateCondition(condition: string, context: Record<string, unknown>): boolean {
    // Very simple condition evaluation - in production use a proper expression evaluator
    try {
      // Replace $vars with context values
      let expr = condition.replace(/\$(\w+)/g, (_, varName) => {
        const val = context[varName];
        return val === undefined ? "undefined" : JSON.stringify(val);
      });
      // eslint-disable-next-line no-eval
      return eval(expr);
    } catch {
      return false;
    }
  }

  // ── Built-in Tools ──────────────────────────────────────────────────────

  // ── Computer / Browser Use Tools ─────────────────────────────────────────
  // Drive the bot's cloud computer (box.ascii.dev) with the same ComputerClient
  // the prime-agent driver and the computer-proxy MCP server use. These act on
  // whatever box the harness bound via bindComputer() — before that they return
  // a clear "no computer" error instead of failing cryptically.
  registerComputerTools(): void {
    const client = () => {
      if (!this.computerClient) {
        throw new Error("no computer bound — provision the bot's box first (bindComputer(boxId, token))");
      }
      return this.computerClient;
    };
    const observe = Type.Optional(
      Type.Boolean({ description: "default true - return a fresh screenshot with the result" }),
    );
    const settleMs = Type.Optional(Type.Number({ description: "wait before the screenshot, default 350, max 3000" }));
    const observeProps = { observe, settle_ms: settleMs };

    const toResult = (note: string, frame: Frame | null) => {
      if (!frame) return { content: [{ type: "text" as const, text: note }], details: null };
      const obs = client().observedContent(note, frame);
      if (!obs.image) return { content: [{ type: "text" as const, text: obs.text }], details: null };
      return {
        content: [
          { type: "text" as const, text: obs.text },
          { type: "image" as const, data: obs.image.data, mimeType: obs.image.mime },
        ],
        details: null,
      };
    };

    this.registerTool({
      name: "computer_screenshot",
      label: "computer_screenshot",
      description:
        "See the bot's cloud computer screen (returns an image). The desktop runs Chrome and a full Linux GUI. You usually do NOT need this after acting - click, type_text, press_key, scroll and browser_open already return the resulting screen.",
      parameters: Type.Object({}, { additionalProperties: false }),
      async execute() {
        const frame = await client().screenshotFrame();
        if (!frame) return { content: [{ type: "text", text: "screenshot failed: could not capture a frame" }], details: null };
        return { content: [{ type: "image", data: frame.data, mimeType: frame.mime }], details: null };
      },
      tags: ["computer", "browser", "builtin"],
    });

    this.registerTool({
      name: "computer_click",
      label: "computer_click",
      description:
        "Click on the computer's screen and return the resulting screen. Use pixel coordinates exactly as they appear in the last frame you were given - any scaling to the real display is handled for you.",
      parameters: Type.Object(
        {
          x: Type.Number(),
          y: Type.Number(),
          button: Type.Optional(Type.Union([Type.Literal("left"), Type.Literal("right")])),
          double: Type.Optional(Type.Boolean()),
          ...observeProps,
        },
        { additionalProperties: false },
      ),
      async execute(_id, params) {
        const p = params as { x?: number; y?: number; button?: string; double?: boolean };
        const x = Math.round(Number(p.x));
        const y = Math.round(Number(p.y));
        if (!Number.isFinite(x) || !Number.isFinite(y)) {
          return { content: [{ type: "text", text: "computer_click needs numeric x,y" }], details: null };
        }
        const what = `${p.double ? "double-clicked" : p.button === "right" ? "right-clicked" : "clicked"} ${x},${y}`;
        const { acted, frame, stderr } = await client().actAndObserve([{ ...p, action: "click" }], p);
        const note = acted ? what : `${what}\n(the action reported an error: ${stderr.slice(0, 160) || "no detail"})`;
        return toResult(note, frame);
      },
      tags: ["computer", "browser", "builtin"],
    });

    this.registerTool({
      name: "computer_type_text",
      label: "computer_type_text",
      description: "Type text at the current focus on the computer and return the resulting screen.",
      parameters: Type.Object({ text: Type.String(), ...observeProps }, { additionalProperties: false }),
      async execute(_id, params) {
        const t = String((params as { text?: string }).text ?? "");
        if (!t) return { content: [{ type: "text", text: "nothing to type" }], details: null };
        const { acted, frame, stderr } = await client().actAndObserve([{ action: "type_text", text: t }], params, 120_000);
        const note = acted ? `typed ${t.length} chars` : `typing failed: ${stderr.slice(0, 160) || "no detail"}`;
        return toResult(note, frame);
      },
      tags: ["computer", "browser", "builtin"],
    });

    this.registerTool({
      name: "computer_press_key",
      label: "computer_press_key",
      description:
        'Press a key or chord on the computer and return the resulting screen. xdotool syntax: "Return", "Tab", "ctrl+c", "alt+F4", "ctrl+shift+t".',
      parameters: Type.Object({ keys: Type.String(), ...observeProps }, { additionalProperties: false }),
      async execute(_id, params) {
        const keys = String((params as { keys?: string }).keys ?? "").replace(/[^\w+]/g, "");
        if (!keys) return { content: [{ type: "text", text: "computer_press_key needs keys" }], details: null };
        const { acted, frame, stderr } = await client().actAndObserve([{ action: "press_key", keys }], params);
        const note = acted ? `pressed ${keys}` : `keypress failed: ${stderr.slice(0, 160) || "no detail"}`;
        return toResult(note, frame);
      },
      tags: ["computer", "browser", "builtin"],
    });

    this.registerTool({
      name: "computer_scroll",
      label: "computer_scroll",
      description: "Scroll the computer screen up or down by N clicks and return the resulting screen.",
      parameters: Type.Object(
        {
          direction: Type.Union([Type.Literal("up"), Type.Literal("down")]),
          clicks: Type.Optional(Type.Number({ description: "default 3" })),
          ...observeProps,
        },
        { additionalProperties: false },
      ),
      async execute(_id, params) {
        const p = params as { direction?: string; clicks?: number };
        const clicks = Math.min(Math.max(Math.round(Number(p.clicks) || 3), 1), 20);
        const direction = p.direction === "up" ? "up" : "down";
        const { acted, frame, stderr } = await client().actAndObserve([{ action: "scroll", direction, clicks }], params);
        const note = acted ? `scrolled ${direction} ${clicks}` : `scroll failed: ${stderr.slice(0, 160) || "no detail"}`;
        return toResult(note, frame);
      },
      tags: ["computer", "browser", "builtin"],
    });

    this.registerTool({
      name: "computer_batch",
      label: "computer_batch",
      description:
        "Run several UI actions in ONE go on the computer and return the screen at the end - much faster than separate calls (one round trip, one screenshot). Use it for mechanical sequences you can predict without looking in between, e.g. click a field, type, Tab, type, press Return. Stop the batch before anything whose outcome you need to see first.",
      parameters: Type.Object(
        {
          actions: Type.Array(
            Type.Object({
              action: Type.Union([
                Type.Literal("click"),
                Type.Literal("type_text"),
                Type.Literal("press_key"),
                Type.Literal("scroll"),
                Type.Literal("wait"),
              ]),
              x: Type.Optional(Type.Number()),
              y: Type.Optional(Type.Number()),
              button: Type.Optional(Type.Union([Type.Literal("left"), Type.Literal("right")])),
              double: Type.Optional(Type.Boolean()),
              text: Type.Optional(Type.String()),
              keys: Type.Optional(Type.String()),
              direction: Type.Optional(Type.Union([Type.Literal("up"), Type.Literal("down")])),
              clicks: Type.Optional(Type.Number()),
              ms: Type.Optional(Type.Number({ description: "wait: milliseconds, max 5000" })),
            }),
          ),
          ...observeProps,
        },
        { additionalProperties: false },
      ),
      async execute(_id, params) {
        const p = params as { actions?: Array<Record<string, unknown>> };
        const actions = (Array.isArray(p.actions) ? p.actions : []).slice(0, 24);
        if (!actions.length) {
          return { content: [{ type: "text", text: "computer_batch needs a non-empty actions array" }], details: null };
        }
        const summary = actions
          .map((a) =>
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
        const { acted, frame, stderr } = await client().actAndObserve(actions, params, 180_000);
        const note = acted ? `ran ${actions.length} actions: ${summary}` : `batch reported an error: ${stderr.slice(0, 160) || "no detail"}`;
        return toResult(note, frame);
      },
      tags: ["computer", "browser", "builtin"],
    });

    this.registerTool({
      name: "computer_exec",
      label: "computer_exec",
      description:
        "Run a shell command on the bot's cloud computer (Linux, passwordless sudo, X11 desktop). Returns stdout/stderr/exit code - and, unlike the UI tools, no screenshot unless you ask for one.",
      parameters: Type.Object(
        {
          command: Type.String(),
          observe: Type.Optional(Type.Boolean({ description: "default false - set true to also return a screenshot" })),
        },
        { additionalProperties: false },
      ),
      async execute(_id, params) {
        const p = params as { command?: string; observe?: boolean };
        const command = String(p.command ?? "").slice(0, 4000);
        const out = await client().runOnBox(command, 120_000);
        const note = `exit ${out.exitCode}\n${out.stdout.slice(-6000)}${out.stderr ? `\n[stderr]\n${out.stderr.slice(-2000)}` : ""}`;
        if (p.observe !== true) return { content: [{ type: "text", text: note }], details: null };
        const frame = await client().screenshotFrame();
        return toResult(note, frame);
      },
      tags: ["computer", "builtin"],
    });

    // ── Browser use ────────────────────────────────────────────────────────
    this.registerTool({
      name: "browser_open",
      label: "browser_open",
      description: "Open a URL in the computer's own Chrome browser and return the resulting screen.",
      parameters: Type.Object({ url: Type.String(), ...observeProps }, { additionalProperties: false }),
      async execute(_id, params) {
        const url = String((params as { url?: string }).url ?? "");
        if (!/^https?:\/\//.test(url)) {
          return { content: [{ type: "text", text: "only http(s) URLs" }], details: null };
        }
        const q = shellQuote(url.replace(/'/g, "%27"));
        const observe = (params as { observe?: boolean }).observe !== false;
        const c = client();
        const command = [
          "export DISPLAY=${DISPLAY:-:0}",
          c.geometryShell,
          `(google-chrome ${q} || chromium ${q} || chromium-browser ${q} || xdg-open ${q}) >/dev/null 2>&1 &`,
          'for i in 1 2 3 4 5 6 7 8 9 10 11 12; do xdotool search --onlyvisible --class "chrom" >/dev/null 2>&1 && break; sleep 0.25; done',
          observe ? c.captureBlock(600) : "true",
        ].join("; ");
        const out = await c.runOnBox(command, 60_000);
        if (!observe) return { content: [{ type: "text", text: `opened ${url}` }], details: null };
        const frame = await c.frameFrom(out);
        return toResult(`opened ${url}`, frame);
      },
      tags: ["browser", "builtin"],
    });

    this.registerTool({
      name: "browser_search",
      label: "browser_search",
      description:
        "Open a web search for the given query in the computer's own Chrome and return the resulting screen. Shortcut for browser_open with a search engine URL.",
      parameters: Type.Object({ query: Type.String(), ...observeProps }, { additionalProperties: false }),
      async execute(_id, params) {
        const query = String((params as { query?: string }).query ?? "").trim();
        if (!query) return { content: [{ type: "text", text: "browser_search needs a query" }], details: null };
        const url = `https://duckduckgo.com/?q=${encodeURIComponent(query)}`;
        const observe = (params as { observe?: boolean }).observe !== false;
        const c = client();
        const q = shellQuote(url.replace(/'/g, "%27"));
        const command = [
          "export DISPLAY=${DISPLAY:-:0}",
          c.geometryShell,
          `(google-chrome ${q} || chromium ${q} || chromium-browser ${q} || xdg-open ${q}) >/dev/null 2>&1 &`,
          'for i in 1 2 3 4 5 6 7 8 9 10 11 12; do xdotool search --onlyvisible --class "chrom" >/dev/null 2>&1 && break; sleep 0.25; done',
          observe ? c.captureBlock(600) : "true",
        ].join("; ");
        const out = await c.runOnBox(command, 60_000);
        if (!observe) return { content: [{ type: "text", text: `searched ${query}` }], details: null };
        const frame = await c.frameFrom(out);
        return toResult(`searched ${query}`, frame);
      },
      tags: ["browser", "builtin"],
    });
  }

  // ── Agent Skills ─────────────────────────────────────────────────────────
  registerComputerSkills(): void {
    this.registerSkill({
      id: "computer-use",
      name: "Computer Use",
      description: "Drive the bot's cloud computer: screenshot, click, type, press keys, scroll, run commands, batch UI sequences.",
      tools: ["computer_screenshot", "computer_click", "computer_type_text", "computer_press_key", "computer_scroll", "computer_batch", "computer_exec"],
      triggers: ["computer", "desktop", "screen", "click", "type", "keyboard", "gui", "click on", "open the computer"],
    });
    this.registerSkill({
      id: "browser-use",
      name: "Browser Use",
      description: "Browse the web in the bot's own Chrome: open URLs and run web searches, seeing the resulting page.",
      tools: ["browser_open", "browser_search", "computer_click", "computer_type_text", "computer_scroll", "computer_batch"],
      triggers: ["browser", "browse", "web page", "website", "open url", "search the web", "search online", "look up", "go to http"],
    });
  }

  registerPresetSkills(): void {
    this.registerSkill({
      id: "browser-use-mcp",
      name: "Browser Use (Chrome)",
      description: "Drive a real Chrome browser via the browser-use MCP server: fill forms, click, extract data, book flights, log into sites with your session.",
      tools: [],
      triggers: ["chrome", "real browser", "fill a form", "book a flight", "logged in", "browser-use"],
    });
    this.registerSkill({
      id: "open-computer-use",
      name: "Open Computer Use",
      description: "Self-hosted Docker sandbox computer for any task: bash, browser, docs, and autonomous sub-agents (open-computer-use MCP).",
      tools: [],
      triggers: ["docker sandbox", "sandbox", "sub agent", "subagent", "computer use", "open-computer-use", "ocu"],
    });
  }

  registerBuiltinTools(): void {
    // File operations
    this.registerTool({
      name: "file_read",
      label: "Read File",
      description: "Read a file from the workspace",
      parameters: Type.Object({ path: Type.String() }),
      async execute(_id, params) {
        const fs = await import("node:fs/promises");
        try {
          const content = await fs.readFile(params.path as string, "utf8");
          return { content: [{ type: "text", text: content }], details: null };
        } catch (e) {
          return { content: [{ type: "text", text: `Error reading file: ${e}` }], details: null, isError: true };
        }
      },
      tags: ["file", "builtin"],
    });

    this.registerTool({
      name: "file_write",
      label: "Write File",
      description: "Write content to a file",
      parameters: Type.Object({ path: Type.String(), content: Type.String() }),
      async execute(_id, params) {
        const fs = await import("node:fs/promises");
        const path = await import("node:path");
        await fs.mkdir(path.dirname(params.path as string), { recursive: true });
        await fs.writeFile(params.path as string, params.content as string, "utf8");
        return { content: [{ type: "text", text: `Wrote ${(params.content as string).length} chars to ${params.path}` }], details: null };
      },
      tags: ["file", "builtin"],
    });

    this.registerTool({
      name: "file_list",
      label: "List Files",
      description: "List files in a directory",
      parameters: Type.Object({ path: Type.String(), recursive: Type.Optional(Type.Boolean()) }),
      async execute(_id, params) {
        const fs = await import("node:fs/promises");
        try {
          const entries = await fs.readdir(params.path as string, { withFileTypes: true });
          const files = entries.map((e) => `${e.isDirectory() ? "📁" : "📄"} ${e.name}`).join("\n");
          return { content: [{ type: "text", text: files || "(empty)" }], details: null };
        } catch (e) {
          return { content: [{ type: "text", text: `Error listing directory: ${e}` }], details: null, isError: true };
        }
      },
      tags: ["file", "builtin"],
    });

    // Web search (placeholder - would integrate with search API)
    this.registerTool({
      name: "web_search",
      label: "Web Search",
      description: "Search the web for information",
      parameters: Type.Object({ query: Type.String(), maxResults: Type.Optional(Type.Number()) }),
      async execute(_id, params) {
        return { content: [{ type: "text", text: `Web search not yet implemented. Query: ${params.query}` }], details: null };
      },
      tags: ["web", "builtin"],
    });

    // HTTP request
    this.registerTool({
      name: "http_request",
      label: "HTTP Request",
      description: "Make an HTTP request",
      parameters: Type.Object({
        url: Type.String(),
        method: Type.Optional(Type.Union([Type.Literal("GET"), Type.Literal("POST"), Type.Literal("PUT"), Type.Literal("DELETE")])),
        headers: Type.Optional(Type.Record(Type.String(), Type.Unknown())),
        body: Type.Optional(Type.String()),
      }),
      async execute(_id, params) {
        const p = params as { url: string; method?: string; headers?: Record<string, string>; body?: string };
        try {
          const res = await fetch(p.url, {
            method: p.method ?? "GET",
            headers: p.headers,
            body: p.body,
          });
          const text = await res.text();
          return { content: [{ type: "text", text: `Status: ${res.status}\n${text.slice(0, 5000)}` }], details: null };
        } catch (e) {
          return { content: [{ type: "text", text: `HTTP error: ${e}` }], details: null, isError: true };
        }
      },
      tags: ["web", "builtin"],
    });

    // Code execution (sandboxed - would need proper sandbox)
    this.registerTool({
      name: "code_exec",
      label: "Execute Code",
      description: "Execute JavaScript/TypeScript code in a sandbox",
      parameters: Type.Object({ code: Type.String(), language: Type.Optional(Type.Union([Type.Literal("js"), Type.Literal("ts")])) }),
      async execute(_id, params) {
        // Placeholder - would use a proper sandbox like vm2, isolated-vm, or e2b
        return { content: [{ type: "text", text: `Code execution not yet implemented (requires sandbox). Code: ${(params.code as string).slice(0, 200)}` }], details: null };
      },
      tags: ["code", "builtin"],
    });

    // Memory/RAG tools
    // Local wrappers for the default thread
    const defaultThreadId = "default";
    const addDocuments = async (docs: Array<{ content: string; metadata?: Record<string, unknown> }>) => 
      memoryRAG.addDocuments(defaultThreadId, docs);
    const hybridSearch = async (query: string, opts?: { topK?: number }) => 
      memoryRAG.hybridSearch(defaultThreadId, query, opts);

    this.registerTool({
      name: "rag_search",
      label: "Semantic Search",
      description: "Search the knowledge base using semantic similarity",
      parameters: Type.Object({ query: Type.String(), topK: Type.Optional(Type.Number()) }),
      async execute(_id, params) {
        const results = await hybridSearch(params.query as string, { topK: params.topK as number });
        return { content: [{ type: "text", text: JSON.stringify(results, null, 2) }], details: null };
      },
      tags: ["memory", "rag", "builtin"],
    });

    this.registerTool({
      name: "rag_ingest",
      label: "Ingest Document",
      description: "Add a document to the knowledge base for future retrieval",
      parameters: Type.Object({ content: Type.String(), metadata: Type.Optional(Type.Record(Type.String(), Type.Unknown())) }),
      async execute(_id, params) {
        const documents: Array<{ content: string; metadata?: Record<string, unknown> }> = [{ 
          content: params.content as string, 
          metadata: params.metadata as Record<string, unknown> 
        }];
        const ids = await addDocuments(documents);
        return { content: [{ type: "text", text: `Ingested ${ids.length} chunks: ${ids.join(", ")}` }], details: null };
      },
      tags: ["memory", "rag", "builtin"],
    });
  }
}

// Singleton
export const toolsRegistry = new ToolsRegistry();
toolsRegistry.registerBuiltinTools();
toolsRegistry.registerComputerTools();
toolsRegistry.registerComputerSkills();
toolsRegistry.registerPresetSkills();

// Auto-wire the browser-use preset when the CLI is on PATH (real Chrome
// automation, no Docker needed). open-computer-use needs `docker compose up`
// first, so it stays opt-in via /api/tools/mcp/preset.
toolsRegistry.ensurePresetMCP("browser-use").catch(() => null);

// Helper to create tool definitions compatible with prime-agent
export function toPrimeAgentTool(tool: ToolDefinition): any {
  return {
    name: tool.name,
    label: tool.label,
    description: tool.description,
    parameters: tool.parameters,
    async execute(_id: string, params: Record<string, unknown>) {
      const result = await toolsRegistry.executeTool(tool.name, params);
      return result;
    },
  };
}