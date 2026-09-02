// Scheduler contract tests: cron parsing, next-fire computation, and the
// tick loop firing due schedules exactly once each.
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";

const DATA_DIR = mkdtempSync(join(tmpdir(), "omb-sched-"));
process.env.OMB_DATA_DIR = DATA_DIR;

let Scheduler: typeof import("./scheduler.ts")["Scheduler"];
let scheduler: InstanceType<typeof import("./scheduler.ts")["Scheduler"]>;
let nextAfter: typeof import("./scheduler.ts")["nextAfter"];
let parseCron: typeof import("./scheduler.ts")["parseCron"];

beforeAll(async () => {
  ({ Scheduler, nextAfter, parseCron } = await import("./scheduler.ts"));
  scheduler = new Scheduler().load();
});

afterAll(() => {
  scheduler.stop();
  rmSync(DATA_DIR, { recursive: true, force: true });
});

describe("parseCron", () => {
  it("accepts wildcards, steps, ranges and lists", () => {
    expect(parseCron("* * * * *")).toBeTruthy();
    expect(parseCron("*/30 * * * *")).toBeTruthy();
    expect(parseCron("0 9 * * 1-5")).toBeTruthy();
    expect(parseCron("0,30 8-10 1,15 * *")).toBeTruthy();
    expect(parseCron("0 0 */2 * *")).toBeTruthy();
  });

  it("rejects malformed expressions", () => {
    expect(parseCron("")).toBeNull();
    expect(parseCron("60 * * * *")).toBeNull();
    expect(parseCron("* 24 * * *")).toBeNull();
    expect(parseCron("* * 0 * *")).toBeNull();
    expect(parseCron("* * * 13 *")).toBeNull();
    expect(parseCron("* * * * 8")).toBeNull();
    expect(parseCron("* * * *")).toBeNull();
    expect(parseCron("a * * * *")).toBeNull();
  });
});

describe("nextAfter", () => {
  // cron fires in the server's LOCAL time, so dates are built locally and
  // assertions use local getters — no UTC round-trips in this suite.
  it("computes the next minute for every-minute cron", () => {
    const base = new Date(2026, 7, 16, 10, 30, 0);
    const next = nextAfter(parseCron("* * * * *"), base)!;
    expect(next.getFullYear()).toBe(2026);
    expect(next.getMonth()).toBe(7);
    expect(next.getDate()).toBe(16);
    expect(next.getHours()).toBe(10);
    expect(next.getMinutes()).toBe(31);
  });

  it("computes next half-hour", () => {
    const base = new Date(2026, 7, 16, 10, 45, 0);
    const next = nextAfter(parseCron("*/30 * * * *"), base)!;
    expect(next.getHours()).toBe(11);
    expect(next.getMinutes()).toBe(0);
  });

  it("skips to the next matching day-of-week", () => {
    // 2026-08-16 is a Sunday; weekday-only should land Monday 09:00
    const base = new Date(2026, 7, 16, 10, 0, 0);
    expect(base.getDay()).toBe(0); // Sunday
    const next = nextAfter(parseCron("0 9 * * 1-5"), base)!;
    expect(next.getDay()).toBe(1);
    expect(next.getHours()).toBe(9);
    expect(next.getMinutes()).toBe(0);
  });

  it("never fires for impossible combos (Feb 31)", () => {
    const next = nextAfter(parseCron("0 0 31 2 *"), new Date(2026, 0, 1, 0, 0, 0));
    expect(next).toBeNull();
  });
});

describe("Scheduler tick", () => {
  it("persists schedules and fires due ones once", async () => {
    const dispatch = vi.fn(async () => {});
    scheduler.dispatch = dispatch;
    const s = scheduler.upsert({
      botId: "bot-1",
      cron: "* * * * *",
      prompt: "daily check-in",
      enabled: true,
    })!;
    expect(s).toBeTruthy();
    expect(s.runs).toBe(0);
    expect(s.nextAt).toBeGreaterThan(Date.now());

    // reload from disk to prove persistence
    const reloaded = new Scheduler().load();
    expect(reloaded.list()).toHaveLength(1);
    expect(reloaded.list()[0].botId).toBe("bot-1");

    // force the due time into the past so the tick fires
    reloaded.get(s.id)!.nextAt = Date.now() - 1000;
    reloaded.dispatch = dispatch;
    await (reloaded as any).tick();
    expect(dispatch).toHaveBeenCalledTimes(1);
    expect(dispatch).toHaveBeenCalledWith("bot-1", "daily check-in");
    const after = reloaded.get(s.id)!;
    expect(after.runs).toBe(1);
    expect(after.lastRunAt).toBeGreaterThan(0);
    // and it's rescheduled for the next minute
    expect(after.nextAt).toBeGreaterThan(Date.now());
  });

  it("skips disabled schedules", async () => {
    const dispatch = vi.fn(async () => {});
    scheduler.dispatch = dispatch;
    scheduler.upsert({ id: "disabled-1", botId: "bot-2", cron: "* * * * *", prompt: "x", enabled: false });
    const s = scheduler.get("disabled-1")!;
    s.nextAt = Date.now() - 1000;
    await (scheduler as any).tick();
    expect(dispatch).not.toHaveBeenCalled();
    expect(scheduler.get("disabled-1")!.runs).toBe(0);
  });

  it("upsert rejects invalid cron", () => {
    expect(scheduler.upsert({ botId: "bot-3", cron: "nope", prompt: "x" })).toBeNull();
  });
});