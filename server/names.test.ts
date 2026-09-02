// Bot naming contract: fresh names come from the curated pool, taken
// names are skipped case-insensitively, and an exhausted pool falls back
// to "Name 2", "Name 3", … without ever colliding.
import { describe, expect, it } from "vitest";

import { pickBotName } from "./names.ts";

// the full pool, reconstructed by draining pickBotName once
function drainPool(): string[] {
  const taken: string[] = [];
  for (;;) {
    const name = pickBotName(taken);
    if (/ \d+$/.test(name)) return taken;
    taken.push(name);
  }
}

describe("pickBotName", () => {
  it("returns a pool name when nothing is taken", () => {
    const name = pickBotName([]);
    expect(name).toBeTruthy();
    expect(name).not.toMatch(/ \d+$/);
  });

  it("never returns a taken name, ignoring case and whitespace", () => {
    const pool = drainPool();
    const half = pool.slice(0, pool.length / 2).map((n) => `  ${n.toUpperCase()} `);
    for (let i = 0; i < 20; i++) {
      const name = pickBotName(half);
      expect(half.map((n) => n.trim().toLowerCase())).not.toContain(name.toLowerCase());
    }
  });

  it("numbers a base name once the pool is exhausted", () => {
    const pool = drainPool();
    const name = pickBotName(pool);
    expect(name).toMatch(/^[A-Za-z]+ 2$/);
    expect(pool.map((n) => n.toLowerCase())).toContain(name.replace(/ 2$/, "").toLowerCase());
  });

  it("skips numbered names that are taken too", () => {
    const pool = drainPool();
    const withTwos = [...pool, ...pool.map((n) => `${n} 2`)];
    const name = pickBotName(withTwos);
    expect(name).toMatch(/^[A-Za-z]+ 3$/);
  });
});
