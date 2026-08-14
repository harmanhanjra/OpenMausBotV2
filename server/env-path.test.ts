// PATH augmentation contract (issues #8, #12): a CLI living in a
// well-known install dir — or an nvm bin dir — must be findable even
// when the process itself started with a bare GUI PATH.
import { execFile } from "node:child_process";
import { chmodSync, mkdirSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { delimiter, join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import { augmentedPath, resetPathCacheForTests } from "./env-path.ts";

const posixIt = it.skipIf(process.platform === "win32");

describe("augmentedPath", () => {
  afterEach(() => {
    delete process.env.OMB_EXTRA_PATH;
    resetPathCacheForTests();
  });

  it("keeps the existing PATH entries first", () => {
    resetPathCacheForTests();
    const path = augmentedPath();
    const firstExisting = (process.env.PATH ?? "").split(delimiter).filter(Boolean)[0];
    // OMB_EXTRA_PATH is unset here, so the inherited PATH leads
    expect(path.split(delimiter)[0]).toBe(firstExisting);
  });

  it("prepends OMB_EXTRA_PATH and dedupes", () => {
    process.env.OMB_EXTRA_PATH = ["/tmp/omb-extra", "/tmp/omb-extra"].join(delimiter);
    resetPathCacheForTests();
    const parts = augmentedPath().split(delimiter);
    expect(parts[0]).toBe("/tmp/omb-extra");
    expect(parts.filter((p) => p === "/tmp/omb-extra")).toHaveLength(1);
  });

  posixIt("includes nvm bin dirs from the home dir, newest node first", () => {
    // setup.ts points homedir at a temp dir, so this is hermetic
    const nvm = join(homedir(), ".nvm", "versions", "node");
    mkdirSync(join(nvm, "v9.0.0", "bin"), { recursive: true });
    mkdirSync(join(nvm, "v24.2.0", "bin"), { recursive: true });
    resetPathCacheForTests();

    const parts = augmentedPath().split(delimiter);
    const v24 = parts.indexOf(join(nvm, "v24.2.0", "bin"));
    const v9 = parts.indexOf(join(nvm, "v9.0.0", "bin"));
    expect(v24).toBeGreaterThan(-1);
    expect(v9).toBeGreaterThan(-1);
    // numeric sort: v24 outranks v9 despite lexicographic order
    expect(v24).toBeLessThan(v9);
  });

  posixIt("makes a CLI in a known install dir spawnable despite a bare PATH", async () => {
    const bin = join(homedir(), ".local", "bin");
    mkdirSync(bin, { recursive: true });
    const fake = join(bin, "omb-fake-cli");
    writeFileSync(fake, "#!/bin/sh\necho found-me\n");
    chmodSync(fake, 0o755);
    resetPathCacheForTests();

    const stdout = await new Promise<string>((resolve, reject) => {
      execFile(
        "omb-fake-cli",
        [],
        // bare GUI-style PATH + our augmentation — the augmentation must win
        { env: { PATH: augmentedPath() } },
        (err, out) => (err ? reject(err) : resolve(out)),
      );
    });
    expect(stdout.trim()).toBe("found-me");
  });

  it("skips known dirs that do not exist", () => {
    resetPathCacheForTests();
    const parts = augmentedPath().split(delimiter);
    // temp home: .volta was never created, so it must not appear
    expect(parts).not.toContain(join(homedir(), ".volta", "bin"));
  });
});
