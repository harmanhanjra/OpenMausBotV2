// Memory store contract tests: per-thread facts persist across instances
// (new MemoryStore over the same DATA_DIR), the knowledge base confines
// paths to its own dir, and the summary indexes files for the prompt.
import { mkdtempSync, rmSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, beforeAll, describe, expect, it } from "vitest";

const DATA_DIR = mkdtempSync(join(tmpdir(), "omb-mem-"));
process.env.OMB_DATA_DIR = DATA_DIR;

let MemoryStore: typeof import("./memory.ts")["MemoryStore"];
let memoryStore: InstanceType<typeof import("./memory.ts")["MemoryStore"]>;

beforeAll(async () => {
  ({ MemoryStore } = await import("./memory.ts"));
  memoryStore = new MemoryStore();
});

afterAll(() => {
  rmSync(DATA_DIR, { recursive: true, force: true });
});

describe("MemoryStore facts", () => {
  it("remembers and recalls facts newest-first", () => {
    const a = memoryStore.remember("t1", "the user likes Node 24");
    const b = memoryStore.remember("t1", "favorite color is teal");
    expect(memoryStore.facts("t1").map((f) => f.text)).toEqual([
      "favorite color is teal",
      "the user likes Node 24",
    ]);
    expect(memoryStore.text("t1")).toContain(`[${a.id}] the user likes Node 24`);
    expect(memoryStore.text("t1")).toContain(`[${b.id}] favorite color is teal`);
  });

  it("forgets by id", () => {
    const c = memoryStore.remember("t1", "ephemeral note");
    expect(memoryStore.forget("t1", c.id)).toBe(true);
    expect(memoryStore.forget("t1", c.id)).toBe(false);
    expect(memoryStore.text("t1")).not.toContain("ephemeral note");
  });

  it("persists across a fresh store instance (same DATA_DIR)", () => {
    const fresh = new MemoryStore();
    expect(fresh.text("t1")).toContain("favorite color is teal");
    expect(fresh.facts("t1")).toHaveLength(2);
  });

  it("isolates threads", () => {
    memoryStore.remember("t2", "only mine");
    expect(memoryStore.text("t1")).not.toContain("only mine");
  });
});

describe("MemoryStore knowledge base", () => {
  it("writes, lists and reads files including nested paths", () => {
    expect(memoryStore.writeKnowledge("notes/alpha.md", "hello")).toBe(true);
    expect(memoryStore.listKnowledge()).toContain("notes/alpha.md");
    expect(memoryStore.readKnowledge("notes/alpha.md")).toBe("hello");
    expect(memoryStore.knowledgeSummary()).toContain("notes/alpha.md");
  });

  it("rejects path traversal and absolute paths", () => {
    expect(memoryStore.writeKnowledge("../escape.md", "x")).toBe(false);
    expect(memoryStore.writeKnowledge("/etc/passwd", "x")).toBe(false);
    expect(memoryStore.writeKnowledge("C:\\windows\\x", "x")).toBe(false);
    expect(existsSync(join(DATA_DIR, "escape.md"))).toBe(false);
    expect(memoryStore.listKnowledge()).not.toContain("escape.md");
  });

  it("deletes knowledge files", () => {
    memoryStore.writeKnowledge("tmp/deleteme.md", "bye");
    expect(memoryStore.deleteKnowledge("tmp/deleteme.md")).toBe(true);
    expect(memoryStore.readKnowledge("tmp/deleteme.md")).toBeNull();
  });
});