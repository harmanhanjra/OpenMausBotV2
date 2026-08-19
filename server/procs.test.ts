// Cross-platform process helpers: CLI resolution stays a no-op on POSIX,
// spawn/exec go through the resolved command, killCliTree reaps the whole
// process group, and the broker channel is a socket path (POSIX) / named
// pipe (Windows).
import { once } from "node:events";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { brokerSocketPath, execCli, killCliTree, resolveCli, spawnCli } from "./procs.ts";

const posixIt = it.skipIf(process.platform === "win32");
const winIt = it.skipIf(process.platform !== "win32");

describe("resolveCli", () => {
  posixIt("is an identity on POSIX", () => {
    expect(resolveCli("claude")).toEqual({ command: "claude", prefixArgs: [] });
    expect(resolveCli("/usr/local/bin/codex")).toEqual({ command: "/usr/local/bin/codex", prefixArgs: [] });
  });

  winIt("spawns explicit paths and .exe shims directly on Windows", () => {
    expect(resolveCli("C:\\tools\\claude.cmd").prefixArgs).toEqual([]);
    expect(resolveCli("claude.exe")).toEqual({ command: "claude.exe", prefixArgs: [] });
  });
});

describe("brokerSocketPath", () => {
  posixIt("is a socket file under the data dir on POSIX", () => {
    expect(brokerSocketPath("/tmp/omb", "abc")).toBe(join("/tmp/omb", "perm-abc.sock"));
  });

  winIt("is a named pipe on Windows", () => {
    expect(brokerSocketPath("C:\\data", "abc")).toBe("\\\\.\\pipe\\openmausbot-perm-abc");
  });
});

describe("spawnCli / execCli / killCliTree", () => {
  posixIt("spawnCli pipes stdio and runs the CLI", async () => {
    const child = spawnCli("printf", ["spawned-ok"], { stdio: ["pipe", "pipe", "pipe"] });
    let out = "";
    child.stdout.on("data", (c) => (out += c));
    const [code] = await once(child, "exit");
    expect(code).toBe(0);
    expect(out).toBe("spawned-ok");
  });

  posixIt("execCli returns stdout through the callback", async () => {
    const stdout = await new Promise<string>((resolve, reject) => {
      execCli("printf", ["exec-ok"], {}, (err, out) => (err ? reject(err) : resolve(out)));
    });
    expect(stdout).toBe("exec-ok");
  });

  posixIt("execCli surfaces spawn errors", async () => {
    const err = await new Promise<Error | null>((resolve) => {
      execCli("omb-definitely-not-a-cli", [], {}, (e) => resolve(e));
    });
    expect(err).toBeTruthy();
  });

  posixIt("killCliTree terminates the process group", async () => {
    const child = spawnCli("sleep", ["30"], { stdio: ["pipe", "pipe", "pipe"] });
    const exited = once(child, "exit");
    killCliTree(child);
    const [code, signal] = await exited;
    expect(signal ?? code).toBeTruthy();
    expect(signal).toBe("SIGTERM");
  });
});
