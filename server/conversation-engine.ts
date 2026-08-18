// Conversation Engine — core turn orchestration with streaming, structured output,
// prompt templates, model routing, and multi-modal support.
import { EventEmitter } from "node:events";
import type { ModelSelection, SendTurnInput, ProviderAdapter } from "./contracts.ts";
import { newId } from "./contracts.ts";
import { memoryStore } from "./memory.ts";

export interface ConversationConfig {
  maxTurnDepth: number;
  maxContextMessages: number;
  defaultSystemPrompt: string;
  enableStreaming: boolean;
  enableStructuredOutput: boolean;
  maxTokensPerTurn: number;
  turnTimeoutMs: number;
}

export interface PromptTemplate {
  id: string;
  name: string;
  description: string;
  systemPrompt: string;
  variables: Record<string, { type: string; description: string; required: boolean }>;
  examples?: Array<{ input: string; output: string }>;
}

export interface ModelRoute {
  pattern: RegExp | ((text: string) => boolean);
  modelSelection: ModelSelection;
  priority: number;
  description: string;
}

export interface TurnContext {
  threadId: string;
  botId: string;
  botName: string;
  botTitle?: string;
  botDescription?: string;
  userMessage: string;
  transcript: Array<{ role: "user" | "assistant"; text: string }>;
  systemPrompt: string;
  modelSelection: ModelSelection;
  integrations: SendTurnInput["integrations"];
  metadata: Record<string, unknown>;
  depth: number;
}

export interface TurnResult {
  turnId: string;
  text: string;
  structuredOutput?: unknown;
  toolCalls: ToolCall[];
  stopReason: string;
  tokenUsage?: { input: number; output: number };
  cost?: number;
}

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  result?: unknown;
  error?: string;
  startedAt: number;
  completedAt?: number;
}

export interface StreamingChunk {
  type: "text" | "reasoning" | "tool_call" | "tool_result" | "done" | "error";
  content: string;
  metadata?: Record<string, unknown>;
}

export class ConversationEngine extends EventEmitter {
  private config: ConversationConfig;
  private promptTemplates = new Map<string, PromptTemplate>();
  private modelRoutes: ModelRoute[] = [];
  private adapters = new Map<string, ProviderAdapter>();
  private activeTurns = new Map<string, { abortController: AbortController; context: TurnContext }>();

  constructor(config: Partial<ConversationConfig> = {}) {
    super();
    this.config = {
      maxTurnDepth: config.maxTurnDepth ?? 3,
      maxContextMessages: config.maxContextMessages ?? 40,
      defaultSystemPrompt: config.defaultSystemPrompt ?? "You are a helpful AI assistant.",
      enableStreaming: config.enableStreaming ?? true,
      enableStructuredOutput: config.enableStructuredOutput ?? true,
      maxTokensPerTurn: config.maxTokensPerTurn ?? 32768,
      turnTimeoutMs: config.turnTimeoutMs ?? 5 * 60 * 1000,
    };
    this.registerDefaultTemplates();
  }

  // ── Configuration ────────────────────────────────────────────────────────

  registerAdapter(provider: string, adapter: ProviderAdapter): void {
    this.adapters.set(provider, adapter);
  }

  getAdapter(provider: string): ProviderAdapter | undefined {
    return this.adapters.get(provider);
  }

  addModelRoute(route: ModelRoute): void {
    this.modelRoutes.push(route);
    this.modelRoutes.sort((a, b) => b.priority - a.priority);
  }

  routeModel(text: string, defaultSelection: ModelSelection): ModelSelection {
    for (const route of this.modelRoutes) {
      const matches = route.pattern instanceof RegExp
        ? route.pattern.test(text)
        : route.pattern(text);
      if (matches) return route.modelSelection;
    }
    return defaultSelection;
  }

  // ── Prompt Templates ──────────────────────────────────────────────────────

  private registerDefaultTemplates(): void {
    this.registerTemplate({
      id: "default",
      name: "Default Assistant",
      description: "General-purpose helpful assistant",
      systemPrompt: `You are {{botName}}, a helpful AI assistant.{{#if botTitle}} Role: {{botTitle}}.{{/if}}{{#if botDescription}} About: {{botDescription}}.{{/if}}
{{#if memoryFacts}}
## Long-term memory
Facts you have remembered:
{{memoryFacts}}{{/if}}
{{#if knowledgeSummary}}
Shared knowledge base files (read with knowledge_read, write with knowledge_write):
{{knowledgeSummary}}{{/if}}`,
      variables: {
        botName: { type: "string", description: "Bot's name", required: true },
        botTitle: { type: "string", description: "Bot's role/title", required: false },
        botDescription: { type: "string", description: "Bot's description", required: false },
        memoryFacts: { type: "string", description: "Long-term memory facts", required: false },
        knowledgeSummary: { type: "string", description: "Knowledge base summary", required: false },
      },
    });

    this.registerTemplate({
      id: "code-review",
      name: "Code Reviewer",
      description: "Expert code review with security, performance, and style focus",
      systemPrompt: `You are {{botName}}, an expert code reviewer.{{#if botTitle}} Specialization: {{botTitle}}.{{/if}}
Review code for: security vulnerabilities, performance issues, style consistency, correctness, maintainability.
Provide actionable feedback with specific line references.`,
      variables: {
        botName: { type: "string", description: "Bot's name", required: true },
        botTitle: { type: "string", description: "Specialization area", required: false },
      },
    });

    this.registerTemplate({
      id: "researcher",
      name: "Deep Researcher",
      description: "Thorough research with citations and synthesis",
      systemPrompt: `You are {{botName}}, a deep research agent.{{#if botTitle}} Focus: {{botTitle}}.{{/if}}
Conduct thorough research: search, read, synthesize, cite sources. Use knowledge_read/write to persist findings. Produce comprehensive reports with citations.`,
      variables: {
        botName: { type: "string", description: "Bot's name", required: true },
        botTitle: { type: "string", description: "Research focus", required: false },
      },
    });

    this.registerTemplate({
      id: "ceo",
      name: "CEO / Team Lead",
      description: "Manages other bots, monitors health, delegates work",
      systemPrompt: `You are {{botName}}, the CEO bot managing a team of agents.{{#if botTitle}} {{botTitle}}.{{/if}}
You have governance tools: monitor_agents (survey team health), set_bot_model (switch stuck bots), interrupt_bot (stop hung turns), ask_bot (delegate).
Run regular health checks. Keep the team responsive and on track.`,
      variables: {
        botName: { type: "string", description: "Bot's name", required: true },
        botTitle: { type: "string", description: "Additional context", required: false },
      },
    });
  }

  registerTemplate(template: PromptTemplate): void {
    this.promptTemplates.set(template.id, template);
  }

  getTemplate(id: string): PromptTemplate | undefined {
    return this.promptTemplates.get(id);
  }

  listTemplates(): PromptTemplate[] {
    return [...this.promptTemplates.values()];
  }

  renderTemplate(templateId: string, variables: Record<string, string>): string {
    const template = this.promptTemplates.get(templateId);
    if (!template) throw new Error(`Template not found: ${templateId}`);
    let prompt = template.systemPrompt;
    for (const [key, value] of Object.entries(variables)) {
      prompt = prompt.replace(new RegExp(`\\{\\{#if ${key}\\}\\}([\\s\\S]*?)\\{\\{/if\\}\\}`, "g"), value ? "$1" : "");
      prompt = prompt.replace(new RegExp(`\\{\\{${key}\\}\\}`, "g"), value);
    }
    return prompt;
  }

  // ── Turn Execution ────────────────────────────────────────────────────────

  async executeTurn(context: TurnContext): Promise<TurnResult> {
    const turnId = newId();
    const abortController = new AbortController();
    this.activeTurns.set(turnId, { abortController, context });

    const adapter = this.adapters.get(context.modelSelection.instanceId);
    if (!adapter) {
      throw new Error(`No adapter for provider: ${context.modelSelection.instanceId}`);
    }

    // Apply model routing
    const routedSelection = this.routeModel(context.userMessage, context.modelSelection);
    if (routedSelection.instanceId !== context.modelSelection.instanceId) {
      const routedAdapter = this.adapters.get(routedSelection.instanceId);
      if (routedAdapter) {
        context.modelSelection = routedSelection;
      }
    }

    // Inject long-term memory into system prompt
    const memoryFacts = memoryStore.text(context.threadId);
    const knowledgeSummary = memoryStore.knowledgeSummary();
    let systemPrompt = context.systemPrompt;
    if (memoryFacts || knowledgeSummary) {
      systemPrompt += `\n\n## Long-term memory\n${memoryFacts ? `Facts:\n${memoryFacts}\n` : ""}${knowledgeSummary ? `Knowledge base:\n${knowledgeSummary}` : ""}`;
    }

    // Execute with timeout
    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error("Turn timeout")), this.config.turnTimeoutMs);
    });

    try {
      const result = await Promise.race([
        this.executeTurnWithAdapter(adapter, context, turnId, systemPrompt),
        timeoutPromise,
      ]);
      return result;
    } finally {
      this.activeTurns.delete(turnId);
    }
  }

  private async executeTurnWithAdapter(
    adapter: ProviderAdapter,
    context: TurnContext,
    _turnId: string,
    systemPrompt: string
  ): Promise<TurnResult> {
    const turnInput: SendTurnInput = {
      threadId: context.threadId,
      text: context.userMessage,
      model: context.modelSelection.model,
      system: systemPrompt,
      transcript: context.transcript.slice(-this.config.maxContextMessages),
      integrations: context.integrations,
    };

    const { turnId: providerTurnId } = await adapter.sendTurn(turnInput);
    
    // Wait for completion via events (simplified - in reality would subscribe to events)
    // This is a placeholder - actual implementation would use the event bus
    return {
      turnId: providerTurnId,
      text: "",
      toolCalls: [],
      stopReason: "end_turn",
    };
  }

  async *streamTurn(context: TurnContext): AsyncGenerator<StreamingChunk> {
    const turnId = newId();
    const abortController = new AbortController();
    this.activeTurns.set(turnId, { abortController, context });

    const adapter = this.adapters.get(context.modelSelection.instanceId);
    if (!adapter) {
      yield { type: "error", content: `No adapter for provider: ${context.modelSelection.instanceId}` };
      return;
    }

    // This would need adapter support for streaming
    // For now, yield a placeholder
    yield { type: "text", content: "Streaming not yet implemented for this adapter." };
    yield { type: "done", content: "" };
  }

  abortTurn(turnId: string): boolean {
    const turn = this.activeTurns.get(turnId);
    if (!turn) return false;
    turn.abortController.abort();
    return true;
  }

  abortAllTurns(): void {
    for (const turn of this.activeTurns.values()) {
      turn.abortController.abort();
    }
  }

  getActiveTurns(): TurnContext[] {
    return [...this.activeTurns.values()].map((t) => t.context);
  }

  // ── Structured Output ────────────────────────────────────────────────────

  async generateStructured<T>(
    _context: TurnContext,
    _schema: Record<string, unknown>,
    _prompt: string
  ): Promise<T> {
    // This would use the adapter's structured output capability
    // Placeholder implementation
    throw new Error("Structured output requires adapter support");
  }

  // ── Multi-modal Support ──────────────────────────────────────────────────

  async processImage(_context: TurnContext, _imageData: string, _mimeType: string): Promise<string> {
    // Would delegate to vision-capable model
    throw new Error("Multi-modal requires vision-capable adapter");
  }

  async processAudio(_context: TurnContext, _audioData: string, _mimeType: string): Promise<string> {
    // Would delegate to audio-capable model
    throw new Error("Audio processing requires audio-capable adapter");
  }

  // ── Health & Monitoring ──────────────────────────────────────────────────

  getHealth(): { status: "healthy" | "degraded"; activeTurns: number; adapters: string[] } {
    return {
      status: this.adapters.size > 0 ? "healthy" : "degraded",
      activeTurns: this.activeTurns.size,
      adapters: [...this.adapters.keys()],
    };
  }
}

// Singleton instance
export const conversationEngine = new ConversationEngine();

// Default model routes (can be customized)
conversationEngine.addModelRoute({
  pattern: /code|program|function|class|debug|refactor|test/i,
  modelSelection: { instanceId: "primeAgent", model: "nvidia/nemotron-3-ultra-550b-a55b" },
  priority: 10,
  description: "Code tasks → Nemotron 3 Ultra",
});

conversationEngine.addModelRoute({
  pattern: /research|analyze|investigate|study|survey/i,
  modelSelection: { instanceId: "primeAgent", model: "nvidia/nemotron-3-ultra-550b-a55b" },
  priority: 10,
  description: "Research tasks → Nemotron 3 Ultra",
});

conversationEngine.addModelRoute({
  pattern: /quick|brief|summary|short/i,
  modelSelection: { instanceId: "primeAgent", model: "meta/llama-3.3-70b-instruct" },
  priority: 5,
  description: "Quick tasks → Llama 3.3 70B",
});