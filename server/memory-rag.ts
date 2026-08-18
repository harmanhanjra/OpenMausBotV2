// Memory & RAG — vector embeddings, semantic search, retrieval-augmented generation,
// user profiling, and knowledge graph integration.
import { mkdirSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { DATA_DIR } from "./config.ts";
import { newId } from "./contracts.ts";

export interface EmbeddingConfig {
  provider: "openai" | "local" | "nvidia";
  model: string;
  dimensions: number;
  batchSize: number;
  apiKey?: string;
  baseUrl?: string;
}

export interface VectorDocument {
  id: string;
  content: string;
  embedding: number[];
  metadata: Record<string, unknown>;
  threadId: string;
  createdAt: number;
}

export interface SearchResult {
  document: VectorDocument;
  score: number;
}

export interface RAGConfig {
  topK: number;
  similarityThreshold: number;
  maxContextTokens: number;
  rerankModel?: string;
  enableHybridSearch: boolean;
}

export interface UserProfile {
  id: string;
  name: string;
  email?: string;
  preferences: Record<string, unknown>;
  interests: string[];
  expertise: string[];
  communicationStyle: "concise" | "detailed" | "technical" | "conversational";
  createdAt: number;
  updatedAt: number;
}

export interface KnowledgeGraphNode {
  id: string;
  type: "entity" | "concept" | "document" | "person" | "project";
  label: string;
  properties: Record<string, unknown>;
  threadId: string;
}

export interface KnowledgeGraphEdge {
  id: string;
  source: string;
  target: string;
  relation: string;
  weight: number;
  threadId: string;
}

const VECTORS_DIR = join(DATA_DIR, "vectors");
const PROFILES_DIR = join(DATA_DIR, "profiles");
const GRAPH_DIR = join(DATA_DIR, "graph");
const vectorsFile = (threadId: string) => join(VECTORS_DIR, `${threadId}.json`);
const profileFile = (userId: string) => join(PROFILES_DIR, `${userId}.json`);
const graphNodesFile = (threadId: string) => join(GRAPH_DIR, `${threadId}-nodes.json`);
const graphEdgesFile = (threadId: string) => join(GRAPH_DIR, `${threadId}-edges.json`);

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
}

function chunkText(text: string, maxTokens = 512, overlap = 50): string[] {
  const words = text.split(/\s+/);
  const chunks: string[] = [];
  for (let i = 0; i < words.length; i += maxTokens - overlap) {
    chunks.push(words.slice(i, i + maxTokens).join(" "));
  }
  return chunks;
}

export class MemoryRAG {
  private config: RAGConfig;
  private embeddingConfig: EmbeddingConfig;
  private vectorCache = new Map<string, VectorDocument[]>();
  private profileCache = new Map<string, UserProfile>();
  private graphNodesCache = new Map<string, KnowledgeGraphNode[]>();
  private graphEdgesCache = new Map<string, KnowledgeGraphEdge[]>();

  constructor(ragConfig: Partial<RAGConfig> = {}, embeddingConfig: Partial<EmbeddingConfig> = {}) {
    this.config = {
      topK: ragConfig.topK ?? 5,
      similarityThreshold: ragConfig.similarityThreshold ?? 0.7,
      maxContextTokens: ragConfig.maxContextTokens ?? 4000,
      rerankModel: ragConfig.rerankModel,
      enableHybridSearch: ragConfig.enableHybridSearch ?? true,
    };
    this.embeddingConfig = {
      provider: embeddingConfig.provider ?? "local",
      model: embeddingConfig.model ?? "text-embedding-3-small",
      dimensions: embeddingConfig.dimensions ?? 1536,
      batchSize: embeddingConfig.batchSize ?? 32,
      apiKey: embeddingConfig.apiKey,
      baseUrl: embeddingConfig.baseUrl,
    };
    this.ensureDirs();
  }

  private ensureDirs(): void {
    mkdirSync(VECTORS_DIR, { recursive: true });
    mkdirSync(PROFILES_DIR, { recursive: true });
    mkdirSync(GRAPH_DIR, { recursive: true });
  }

  // ── Embeddings ────────────────────────────────────────────────────────────

  async embed(texts: string[]): Promise<number[][]> {
    if (this.embeddingConfig.provider === "local") {
      return this.localEmbed(texts);
    }
    if (this.embeddingConfig.provider === "openai") {
      return this.openAIEmbed(texts);
    }
    if (this.embeddingConfig.provider === "nvidia") {
      return this.nvidiaEmbed(texts);
    }
    throw new Error(`Unknown embedding provider: ${this.embeddingConfig.provider}`);
  }

  private async localEmbed(texts: string[]): Promise<number[][]> {
    // Simple hash-based embedding for local/development use
    // In production, use a proper local model (e.g., via ollama, sentence-transformers)
    return texts.map((text) => {
      const vec = new Array(this.embeddingConfig.dimensions).fill(0);
      let hash = 0;
      for (let i = 0; i < text.length; i++) {
        hash = ((hash << 5) - hash + text.charCodeAt(i)) | 0;
        vec[Math.abs(hash) % this.embeddingConfig.dimensions] += 1;
      }
      // Normalize
      const norm = Math.sqrt(vec.reduce((a, b) => a + b * b, 0)) || 1;
      return vec.map((v) => v / norm);
    });
  }

  private async openAIEmbed(texts: string[]): Promise<number[][]> {
    if (!this.embeddingConfig.apiKey) throw new Error("OpenAI API key required");
    const url = `${this.embeddingConfig.baseUrl ?? "https://api.openai.com"}/v1/embeddings`;
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${this.embeddingConfig.apiKey}` },
      body: JSON.stringify({ model: this.embeddingConfig.model, input: texts }),
    });
    if (!res.ok) throw new Error(`OpenAI embeddings failed: ${res.statusText}`);
    const data = await res.json() as { data: Array<{ embedding: number[] }> };
    return data.data.map((d) => d.embedding);
  }

  private async nvidiaEmbed(texts: string[]): Promise<number[][]> {
    if (!this.embeddingConfig.apiKey) throw new Error("NVIDIA API key required");
    const url = `${this.embeddingConfig.baseUrl ?? "https://integrate.api.nvidia.com"}/v1/embeddings`;
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${this.embeddingConfig.apiKey}` },
      body: JSON.stringify({ model: this.embeddingConfig.model, input: texts }),
    });
    if (!res.ok) throw new Error(`NVIDIA embeddings failed: ${res.statusText}`);
    const data = await res.json() as { data: Array<{ embedding: number[] }> };
    return data.data.map((d) => d.embedding);
  }

  // ── Vector Store ──────────────────────────────────────────────────────────

  async addDocuments(threadId: string, documents: Array<{ content: string; metadata?: Record<string, unknown> }>): Promise<string[]> {
    const texts = documents.map((d) => d.content);
    const embeddings = await this.embed(texts);
    const ids: string[] = [];

    const vectors = this.getVectors(threadId);
    for (let i = 0; i < documents.length; i++) {
      const id = newId();
      const doc: VectorDocument = {
        id,
        content: documents[i].content,
        embedding: embeddings[i],
        metadata: documents[i].metadata ?? {},
        threadId,
        createdAt: Date.now(),
      };
      vectors.push(doc);
      ids.push(id);
    }
    this.saveVectors(threadId);
    return ids;
  }

  async search(threadId: string, query: string, options?: { topK?: number; threshold?: number; filter?: Record<string, unknown> }): Promise<SearchResult[]> {
    const queryEmbedding = (await this.embed([query]))[0];
    const vectors = this.getVectors(threadId);
    if (!vectors.length) return [];

    const results: SearchResult[] = [];
    for (const doc of vectors) {
      if (options?.filter) {
        let match = true;
        for (const [key, value] of Object.entries(options.filter)) {
          if (doc.metadata[key] !== value) { match = false; break; }
        }
        if (!match) continue;
      }
      const score = cosineSimilarity(queryEmbedding, doc.embedding);
      if (score >= (options?.threshold ?? this.config.similarityThreshold)) {
        results.push({ document: doc, score });
      }
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, options?.topK ?? this.config.topK);
  }

  async hybridSearch(threadId: string, query: string, options?: { topK?: number }): Promise<SearchResult[]> {
    // Combine vector search with keyword search (BM25-like)
    const vectorResults = await this.search(threadId, query, { topK: options?.topK ?? this.config.topK * 2 });
    const keywordResults = this.keywordSearch(threadId, query, options?.topK ?? this.config.topK * 2);

    // Merge and deduplicate by document ID
    const merged = new Map<string, SearchResult>();
    for (const r of [...vectorResults, ...keywordResults]) {
      const existing = merged.get(r.document.id);
      if (!existing || r.score > existing.score) {
        merged.set(r.document.id, r);
      }
    }
    return [...merged.values()].sort((a, b) => b.score - a.score).slice(0, options?.topK ?? this.config.topK);
  }

  private keywordSearch(threadId: string, query: string, topK: number): SearchResult[] {
    const vectors = this.getVectors(threadId);
    const queryTerms = query.toLowerCase().split(/\s+/).filter((t) => t.length > 2);
    if (!queryTerms.length) return [];

    const results: SearchResult[] = [];
    for (const doc of vectors) {
      const content = doc.content.toLowerCase();
      let score = 0;
      for (const term of queryTerms) {
        const matches = (content.match(new RegExp(term, "g")) ?? []).length;
        score += matches;
      }
      if (score > 0) {
        // Normalize by document length
        score = score / Math.log(doc.content.length + 1);
        results.push({ document: doc, score });
      }
    }
    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }

  getVectors(threadId: string): VectorDocument[] {
    let vectors = this.vectorCache.get(threadId) ?? [];
    if (!vectors.length) {
      try {
        vectors = JSON.parse(readFileSync(vectorsFile(threadId), "utf8"));
      } catch {
        vectors = [];
      }
      this.vectorCache.set(threadId, vectors);
    }
    return vectors;
  }

  private saveVectors(threadId: string): void {
    const vectors = this.vectorCache.get(threadId) ?? [];
    writeFileSync(vectorsFile(threadId), JSON.stringify(vectors, null, 2));
  }

  async deleteDocument(threadId: string, docId: string): Promise<boolean> {
    const vectors = this.getVectors(threadId);
    const idx = vectors.findIndex((d) => d.id === docId);
    if (idx === -1) return false;
    vectors.splice(idx, 1);
    this.saveVectors(threadId);
    return true;
  }

  async clearThread(threadId: string): Promise<void> {
    this.vectorCache.delete(threadId);
    try { rmSync(vectorsFile(threadId)); } catch {}
  }

  // ── RAG Pipeline ──────────────────────────────────────────────────────────

  async retrieveContext(threadId: string, query: string): Promise<{ context: string; sources: SearchResult[] }> {
    const results = await this.hybridSearch(threadId, query);
    if (!results.length) return { context: "", sources: [] };

    let totalTokens = 0;
    const contextParts: string[] = [];
    const sources: SearchResult[] = [];

    for (const result of results) {
      const tokens = Math.ceil(result.document.content.length / 4); // rough estimate
      if (totalTokens + tokens > this.config.maxContextTokens) break;
      contextParts.push(`[Source: ${result.document.id}]\n${result.document.content}`);
      sources.push(result);
      totalTokens += tokens;
    }

    return { context: contextParts.join("\n\n---\n\n"), sources };
  }

  async generateRAGResponse(
    threadId: string,
    query: string,
    generateFn: (prompt: string) => Promise<string>
  ): Promise<{ answer: string; sources: SearchResult[] }> {
    const { context, sources } = await this.retrieveContext(threadId, query);
    if (!context) {
      return { answer: "I couldn't find relevant information to answer that.", sources: [] };
    }

    const prompt = `Use the following context to answer the question. Cite sources using [Source: ID] format.

Context:
${context}

Question: ${query}

Answer:`;
    const answer = await generateFn(prompt);
    return { answer, sources };
  }

  // ── Document Ingestion ────────────────────────────────────────────────────

  async ingestFile(threadId: string, filePath: string, content: string, metadata?: Record<string, unknown>): Promise<string[]> {
    const chunks = chunkText(content);
    return this.addDocuments(threadId, chunks.map((chunk, i) => ({
      content: chunk,
      metadata: { ...metadata, source: filePath, chunk: i, totalChunks: chunks.length },
    })));
  }

  async ingestURL(threadId: string, url: string, content: string, metadata?: Record<string, unknown>): Promise<string[]> {
    return this.ingestFile(threadId, url, content, { ...metadata, sourceType: "url" });
  }

  // ── User Profiling ────────────────────────────────────────────────────────

  getProfile(userId: string): UserProfile | null {
    let profile = this.profileCache.get(userId) ?? null;
    if (!profile) {
      try {
        profile = JSON.parse(readFileSync(profileFile(userId), "utf8"));
      } catch {
        return null;
      }
      if (profile) this.profileCache.set(userId, profile);
    }
    return profile;
  }

  upsertProfile(profile: Partial<UserProfile> & { id: string }): UserProfile {
    const existing = this.getProfile(profile.id) ?? {
      id: profile.id,
      name: profile.name ?? "User",
      preferences: {},
      interests: [],
      expertise: [],
      communicationStyle: "conversational",
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    const updated: UserProfile = {
      ...existing,
      ...profile,
      updatedAt: Date.now(),
    };
    this.profileCache.set(profile.id, updated);
    writeFileSync(profileFile(profile.id), JSON.stringify(updated, null, 2));
    return updated;
  }

  updatePreferences(userId: string, preferences: Record<string, unknown>): UserProfile | null {
    const profile = this.getProfile(userId);
    if (!profile) return null;
    return this.upsertProfile({ ...profile, preferences: { ...profile.preferences, ...preferences } });
  }

  // ── Knowledge Graph ───────────────────────────────────────────────────────

  getGraphNodes(threadId: string): KnowledgeGraphNode[] {
    let nodes = this.graphNodesCache.get(threadId);
    if (!nodes) {
      try { nodes = JSON.parse(readFileSync(graphNodesFile(threadId), "utf8")); } catch { nodes = []; }
      this.graphNodesCache.set(threadId, nodes ?? []);
    }
    return nodes ?? [];
  }

  getGraphEdges(threadId: string): KnowledgeGraphEdge[] {
    let edges = this.graphEdgesCache.get(threadId);
    if (!edges) {
      try { edges = JSON.parse(readFileSync(graphEdgesFile(threadId), "utf8")); } catch { edges = []; }
      this.graphEdgesCache.set(threadId, edges ?? []);
    }
    return edges ?? [];
  }

  addGraphNode(threadId: string, node: Omit<KnowledgeGraphNode, "id" | "threadId">): KnowledgeGraphNode {
    const nodes = this.getGraphNodes(threadId);
    const newNode: KnowledgeGraphNode = { id: newId(), threadId, ...node };
    nodes.push(newNode);
    this.saveGraphNodes(threadId);
    return newNode;
  }

  addGraphEdge(threadId: string, edge: Omit<KnowledgeGraphEdge, "id" | "threadId">): KnowledgeGraphEdge {
    const edges = this.getGraphEdges(threadId);
    const newEdge: KnowledgeGraphEdge = { id: newId(), threadId, ...edge };
    edges.push(newEdge);
    this.saveGraphEdges(threadId);
    return newEdge;
  }

  private saveGraphNodes(threadId: string): void {
    const nodes = this.graphNodesCache.get(threadId) ?? [];
    writeFileSync(graphNodesFile(threadId), JSON.stringify(nodes, null, 2));
  }

  private saveGraphEdges(threadId: string): void {
    const edges = this.graphEdgesCache.get(threadId) ?? [];
    writeFileSync(graphEdgesFile(threadId), JSON.stringify(edges, null, 2));
  }

  // ── Entity Extraction (simple) ────────────────────────────────────────────

  async extractEntities(text: string): Promise<Array<{ entity: string; type: string; confidence: number }>> {
    // Simple regex-based extraction; replace with NER model in production
    const entities: Array<{ entity: string; type: string; confidence: number }> = [];
    
    // Email
    for (const match of text.matchAll(/[\w.+-]+@[\w-]+\.[\w.-]+/g)) {
      entities.push({ entity: match[0], type: "email", confidence: 0.9 });
    }
    // URLs
    for (const match of text.matchAll(/https?:\/\/[^\s]+/g)) {
      entities.push({ entity: match[0], type: "url", confidence: 0.95 });
    }
    // Code-like
    for (const match of text.matchAll(/\b[a-z_][a-z0-9_]*\(\)/gi)) {
      entities.push({ entity: match[0], type: "function", confidence: 0.7 });
    }
    // Capitalized phrases (potential proper nouns)
    for (const match of text.matchAll(/\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b/g)) {
      if (match[0].length > 2) {
        entities.push({ entity: match[0], type: "proper_noun", confidence: 0.5 });
      }
    }
    return entities;
  }

  async buildKnowledgeGraphFromText(threadId: string, text: string): Promise<void> {
    const entities = await this.extractEntities(text);
    const seen = new Set<string>();
    for (const e of entities) {
      if (seen.has(e.entity)) continue;
      seen.add(e.entity);
      this.addGraphNode(threadId, { type: e.type as any, label: e.entity, properties: { confidence: e.confidence } });
    }
  }

  // ── Statistics ────────────────────────────────────────────────────────────

  getStats(threadId: string): { vectors: number; graphNodes: number; graphEdges: number } {
    return {
      vectors: this.getVectors(threadId).length,
      graphNodes: this.getGraphNodes(threadId).length,
      graphEdges: this.getGraphEdges(threadId).length,
    };
  }
}

// Singleton
export const memoryRAG = new MemoryRAG(
  { topK: 5, similarityThreshold: 0.7, maxContextTokens: 4000, enableHybridSearch: true },
  { provider: "local", model: "local", dimensions: 1536, batchSize: 32 }
);