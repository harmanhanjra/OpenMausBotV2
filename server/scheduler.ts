// Scheduler — wakes bots on timers so they run autonomously (24/7 mode).
// A schedule is a persisted record mapping a bot + a cron expression + a
// prompt to the next due time. A 10s tick fires anything that's due and
// reschedules it. The harness injects the turn-dispatch callback so this
// module stays transport-agnostic (no HTTP knowledge here).
//
// Cron format (standard 5 fields, space-separated):
//   minute hour day-of-month month day-of-week
//   *  *  *  *  *     every minute
//   */30 * * * *      every 30 minutes
//   0 9 * * 1-5       weekdays at 09:00
// Supports *, ranges (1-5), lists (1,15), steps (*/10, 0-30/5), and names
// are NOT supported (numeric fields only, as in POSIX cron).
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

import { DATA_DIR } from "./config.ts";

export interface Schedule {
  id: string;
  botId: string;
  /** Standard 5-field cron expression. */
  cron: string;
  /** What to ask the bot when it fires. */
  prompt: string;
  /** Whether the schedule is currently active. */
  enabled: boolean;
  createdAt: number;
  /** Millisecond epoch of the next (or most recent) fire. */
  nextAt?: number | null;
  /** Count of times this schedule has fired. */
  runs: number;
  /** When it last fired (ms epoch), null before the first run. */
  lastRunAt?: number | null;
}

const SCHEDULES_FILE = join(DATA_DIR, "schedules.json");

/** Parse one cron field ("*", step "slash-n", range "1-5", list "1,3",
 * ranged-step "1-5/2", bare "5") into a set of allowed values within
 * [min..max]. Returns null when malformed. */
function parseField(field: string, min: number, max: number): Set<number> | null {
  const out = new Set<number>();
  for (const part of field.split(",")) {
    if (!part) return null;
    const stepMatch = part.match(/^(.+?)\/(\d+)$/);
    const step = stepMatch ? Number(stepMatch[2]) : 1;
    let base = stepMatch ? stepMatch[1] : part;
    if (step < 1 || step > max - min + 1) return null;
    if (base === "*") {
      for (let v = min; v <= max; v++) if ((v - min) % step === 0) out.add(v);
      continue;
    }
    const rangeMatch = base.match(/^(\d+)(?:-(\d+))?$/);
    if (!rangeMatch) return null;
    let lo = Number(rangeMatch[1]);
    let hi = rangeMatch[2] !== undefined ? Number(rangeMatch[2]) : lo;
    if (lo < min || hi > max || lo > hi) return null;
    for (let v = lo; v <= hi; v++) if ((v - lo) % step === 0) out.add(v);
  }
  return out.size ? out : null;
}

export function parseCron(expr: string): {
  minutes: Set<number>;
  hours: Set<number>;
  dom: Set<number>;
  months: Set<number>;
  dow: Set<number>;
} | null {
  const parts = String(expr ?? "").trim().split(/\s+/);
  if (parts.length !== 5) return null;
  const minutes = parseField(parts[0], 0, 59);
  const hours = parseField(parts[1], 0, 23);
  const dom = parseField(parts[2], 1, 31);
  const months = parseField(parts[3], 1, 12);
  const dow = parseField(parts[4], 0, 7); // 0 and 7 are both Sunday
  if (!minutes || !hours || !dom || !months || !dow) return null;
  return { minutes, hours, dom, months, dow };
}

function matches(parsed: ReturnType<typeof parseCron>, d: Date): boolean {
  if (!parsed) return false;
  const dom = d.getDate();
  const dow = d.getDay() === 0 ? 7 : d.getDay();
  const domStar = parsed.dom.size === 31 && parsed.dom.has(0) === false;
  const dowStar = parsed.dow.size === 8;
  // POSIX: if BOTH dom and dow are restricted, a day matches either;
  // if either is *, that * field means "every day" and the other decides.
  const dayOk = domStar && dowStar ? true : domStar ? parsed.dow.has(dow) : dowStar ? parsed.dom.has(dom) : parsed.dom.has(dom) || parsed.dow.has(dow);
  return (
    parsed.minutes.has(d.getMinutes()) &&
    parsed.hours.has(d.getHours()) &&
    parsed.months.has(d.getMonth() + 1) &&
    dayOk
  );
}

/** The next fire time strictly after `after` (defaults to now). */
export function nextAfter(parsed: ReturnType<typeof parseCron>, after: Date = new Date()): Date | null {
  if (!parsed) return null;
  const d = new Date(after);
  d.setSeconds(0, 0);
  // start scanning from the next minute so an already-fired minute doesn't re-fire
  d.setMinutes(d.getMinutes() + 1);
  // cap the scan: never look more than ~5 years out (invalid combos like
  // Feb 31 never fire, and spinning forever is worse than never firing)
  for (let i = 0; i < 5 * 366 * 24 * 60; i++) {
    if (matches(parsed, d)) return d;
    d.setMinutes(d.getMinutes() + 1);
  }
  return null;
}

export class Scheduler {
  private schedules: Schedule[] = [];
  private timer: NodeJS.Timeout | null = null;
  /** Injected by the harness: dispatch a bot turn. */
  dispatch: (botId: string, text: string) => Promise<void> = async () => {};

  load() {
    try {
      const raw = JSON.parse(readFileSync(SCHEDULES_FILE, "utf8"));
      if (Array.isArray(raw)) this.schedules = raw;
    } catch {
      this.schedules = [];
    }
    return this;
  }

  private save() {
    mkdirSync(DATA_DIR, { recursive: true });
    writeFileSync(SCHEDULES_FILE, JSON.stringify(this.schedules, null, 2));
  }

  list(): Schedule[] {
    return this.schedules.map((s) => ({ ...s }));
  }

  get(id: string): Schedule | undefined {
    return this.schedules.find((s) => s.id === id);
  }

  /** Create or replace a schedule. Returns null when the cron is invalid. */
  upsert(schedule: {
    id?: string;
    botId: string;
    cron: string;
    prompt: string;
    enabled?: boolean;
  }): Schedule | null {
    const parsed = parseCron(schedule.cron);
    if (!parsed) return null;
    const existing = schedule.id ? this.schedules.find((s) => s.id === schedule.id) : undefined;
    const now = Date.now();
    const entry: Schedule = {
      id: existing?.id ?? schedule.id ?? `sched-${now.toString(36)}-${Math.floor(Math.random() * 1e6).toString(36)}`,
      botId: schedule.botId,
      cron: schedule.cron,
      prompt: schedule.prompt,
      enabled: schedule.enabled !== false,
      createdAt: existing?.createdAt ?? now,
      nextAt: existing?.nextAt ?? null,
      runs: existing?.runs ?? 0,
      lastRunAt: existing?.lastRunAt ?? null,
    };
    if (entry.enabled) {
      const next = nextAfter(parsed);
      entry.nextAt = next ? next.getTime() : null;
    } else {
      entry.nextAt = null;
    }
    const idx = this.schedules.findIndex((s) => s.id === entry.id);
    if (idx >= 0) this.schedules[idx] = entry;
    else this.schedules.push(entry);
    this.save();
    return { ...entry };
  }

  remove(id: string): boolean {
    const before = this.schedules.length;
    this.schedules = this.schedules.filter((s) => s.id !== id);
    if (this.schedules.length !== before) {
      this.save();
      return true;
    }
    return false;
  }

  /** Fire any schedule whose nextAt is in the past. Busy bots are skipped
   * (their fire reschedules for the next matching minute). */
  private async tick() {
    const now = Date.now();
    const due = this.schedules.filter((s) => s.enabled && s.nextAt != null && s.nextAt <= now);
    for (const s of due) {
      const parsed = parseCron(s.cron);
      const next = parsed ? nextAfter(parsed, new Date(Math.max(now, s.nextAt!))) : null;
      s.nextAt = next ? next.getTime() : null;
      s.runs += 1;
      s.lastRunAt = now;
      this.save();
      try {
        await this.dispatch(s.botId, s.prompt);
      } catch {
        // a busy/failed bot must not stop the loop; it'll fire again next minute
      }
    }
  }

  start(tickMs = 10_000) {
    if (this.timer) return;
    this.timer = setInterval(() => void this.tick(), tickMs);
    this.timer.unref?.();
  }

  stop() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }
}

export const scheduler = new Scheduler();