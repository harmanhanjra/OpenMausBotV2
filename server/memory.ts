// Long-term memory: per-thread fact store + a shared swarm knowledge base.
// Facts survive restarts (they live on disk, unlike the in-context
// transcript), and the driver injects them into the agent's system prompt
// so accumulated knowledge is always in context from the first turn.
//
// Layout (all under DATA_DIR):
//   memories/<threadId>.json   — per-bot facts: [{ id, text, at }]
//   knowledge/...              — shared files every bot in the swarm can
//                                read/write (research notes, playbooks)
import { mkdirSync, readFileSync, writeFileSync, readdirSync, statSync, rmSync } from "node:fs";
import { dirname, join, relative } from "node:path";

import { DATA_DIR } from "./config.ts";
import { newId } from "./contracts.ts";

export interface MemoryFact {
  id: string;
  text: string;
  at: number;
}

const MEMORIES_DIR = join(DATA_DIR, "memories");
const KNOWLEDGE_DIR = join(DATA_DIR, "knowledge");
const memoriesFile = (threadId: string) => join(MEMORIES_DIR, `${threadId}.json`);

export class MemoryStore {
  private cache = new Map<string, MemoryFact[]>();

  /** The bot's facts, newest first. Never throws — a missing or corrupt
   * file just means an empty memory. */
  facts(threadId: string): MemoryFact[] {
    let list = this.cache.get(threadId);
    if (!list) {
      try {
        const raw = JSON.parse(readFileSync(memoriesFile(threadId), "utf8"));
        list = Array.isArray(raw) ? raw : [];
      } catch {
        list = [];
      }
      this.cache.set(threadId, list);
    }
    return list;
  }

  /** Remember a fact. Returns it. */
  remember(threadId: string, text: string): MemoryFact {
    const fact: MemoryFact = { id: newId(), text: text.trim(), at: Date.now() };
    const list = this.facts(threadId);
    list.unshift(fact);
    this.cache.set(threadId, list);
    this.save(threadId);
    return fact;
  }

  /** Forget a fact by id. Returns true when it existed. */
  forget(threadId: string, id: string): boolean {
    const list = this.facts(threadId);
    const next = list.filter((f) => f.id !== id);
    if (next.length === list.length) return false;
    this.cache.set(threadId, next);
    this.save(threadId);
    return true;
  }

  /** Numbered view for the recall tool / system prompt injection. */
  text(threadId: string): string {
    const facts = this.facts(threadId);
    if (!facts.length) return "";
    return facts
      .map((f, i) => `${i + 1}. [${f.id}] ${f.text}`)
      .join("\n");
  }

  private save(threadId: string) {
    mkdirSync(MEMORIES_DIR, { recursive: true });
    writeFileSync(memoriesFile(threadId), JSON.stringify(this.facts(threadId), null, 2));
  }

  // ── shared swarm knowledge base ────────────────────────────────────────

  /** Files under knowledge/, relative paths (no traversal). */
  listKnowledge(): string[] {
    const walk = (dir: string, prefix: string): string[] => {
      const out: string[] = [];
      let entries: Array<import("node:fs").Dirent> = [];
      try {
        entries = readdirSync(dir, { withFileTypes: true });
      } catch {
        return out;
      }
      for (const e of entries) {
        const rel = prefix ? `${prefix}/${e.name}` : e.name;
        if (e.isDirectory()) out.push(...walk(join(dir, e.name), rel));
        else out.push(rel);
      }
      return out;
    };
    return walk(KNOWLEDGE_DIR, "").sort();
  }

  /** Read a knowledge file. Returns null when missing. Paths are confined
   * to the knowledge dir — `../` and absolute paths are rejected. */
  readKnowledge(relPath: string): string | null {
    const p = this.resolve(relPath);
    if (!p) return null;
    try {
      return readFileSync(p, "utf8");
    } catch {
      return null;
    }
  }

  /** Write a knowledge file (creates parent dirs). Returns false when the
   * path escapes the knowledge dir. */
  writeKnowledge(relPath: string, content: string): boolean {
    const p = this.resolve(relPath);
    if (!p) return false;
    mkdirSync(dirname(p), { recursive: true });
    writeFileSync(p, content, "utf8");
    this.cache.clear(); // nothing to invalidate, but cheap hygiene
    return true;
  }

  /** Delete a knowledge file (or empty dir). Returns false on bad paths. */
  deleteKnowledge(relPath: string): boolean {
    const p = this.resolve(relPath);
    if (!p) return false;
    try {
      rmSync(p, { force: true });
      return true;
    } catch {
      return false;
    }
  }

  /** Summary of the knowledge base for the system prompt: file name +
   * size, not contents (those are read on demand). */
  knowledgeSummary(): string {
    const files = this.listKnowledge();
    if (!files.length) return "";
    return files
      .map((f) => {
        let size = 0;
        try {
          size = statSync(join(KNOWLEDGE_DIR, f)).size;
        } catch {
          /* ignore */
        }
        return `${f} (${size}b)`;
      })
      .join("\n");
  }

  /** Confine a user-supplied path to the knowledge dir. */
  private resolve(relPath: string): string | null {
    const raw = String(relPath ?? "").replace(/\\/g, "/");
    if (!raw || raw.startsWith("/") || /^[a-zA-Z]:/.test(raw) || raw.includes("..")) return null;
    const clean = raw.replace(/^\/+/, "");
    if (!clean) return null;
    const abs = join(KNOWLEDGE_DIR, clean);
    return relative(KNOWLEDGE_DIR, abs).startsWith("..") ? null : abs;
  }
}

export const memoryStore = new MemoryStore();