// The rules these helpers encode: a missing file is a normal state, a
// file we cannot read is an error the caller must see, and a file we are
// about to overwrite is never dropped without a copy left behind.
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { isMissing, readJsonFile, removeFile } from "./fs-safe.ts";

let dir: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "omb-fs-safe-"));
  vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
  vi.restoreAllMocks();
});

describe("readJsonFile", () => {
  it("parses a file that is there", () => {
    const file = join(dir, "bots.json");
    writeFileSync(file, JSON.stringify([{ id: "bot-1" }]));
    expect(readJsonFile<Array<{ id: string }>>(file, "store")).toEqual([{ id: "bot-1" }]);
  });

  it("reports a missing file as null, not an error", () => {
    expect(readJsonFile(join(dir, "nope.json"), "store")).toBeNull();
    expect(console.error).not.toHaveBeenCalled();
  });

  it("throws when the file exists but cannot be read", () => {
    // a directory in the file's place: ENOENT's opposite — something is
    // there, we just cannot use it, and pretending it is empty would let
    // the next write erase whatever it holds
    const file = join(dir, "bots.json");
    mkdirSync(file);
    expect(() => readJsonFile(file, "store")).toThrow(/store: cannot read/);
  });

  it("quarantines invalid JSON so the next write cannot erase it", () => {
    const file = join(dir, "bots.json");
    writeFileSync(file, "{not json");

    expect(readJsonFile(file, "store")).toBeNull();
    expect(existsSync(file)).toBe(false);
    const preserved = readdirSync(dir).filter((f) => f.startsWith("bots.json.corrupt-"));
    expect(preserved).toHaveLength(1);
    expect(readFileSync(join(dir, preserved[0]), "utf8")).toBe("{not json");
    expect(console.error).toHaveBeenCalled();
  });
});

describe("removeFile", () => {
  it("deletes the file", () => {
    const file = join(dir, "messages-t1.json");
    writeFileSync(file, "[]");
    removeFile(file, "store");
    expect(existsSync(file)).toBe(false);
    expect(console.error).not.toHaveBeenCalled();
  });

  it("stays quiet when the file is already gone", () => {
    removeFile(join(dir, "gone.json"), "store");
    expect(console.error).not.toHaveBeenCalled();
  });

  it("reports a delete that failed for any other reason", () => {
    // a non-empty directory cannot be unlinked — stands in for the real
    // cases (permissions, a locked file) that used to vanish
    const path = join(dir, "stubborn");
    mkdirSync(path);
    writeFileSync(join(path, "child"), "x");

    removeFile(path, "store");
    expect(existsSync(path)).toBe(true);
    expect(console.error).toHaveBeenCalledWith(expect.stringContaining("could not delete"), expect.anything());
  });
});

describe("isMissing", () => {
  it("recognizes ENOENT and nothing else", () => {
    expect(isMissing(Object.assign(new Error("nope"), { code: "ENOENT" }))).toBe(true);
    expect(isMissing(Object.assign(new Error("nope"), { code: "EACCES" }))).toBe(false);
    expect(isMissing(new Error("plain"))).toBe(false);
    expect(isMissing("not an error")).toBe(false);
  });
});
