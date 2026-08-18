// Shared spawn-adjacent helpers for the CLI-backed drivers (claude, codex,
// antigravity, the ACP harnesses): the `--version` probe every snapshot() runs
// and the bounded stderr tail every turn keeps for its exit diagnostic.
import { augmentedPath } from "../env-path.ts";
import { execCli } from "../procs.ts";

export type CliEnv = Record<string, string | undefined>;

/** process env with the GUI-app PATH gaps patched (see env-path.ts). */
export function cliEnv(extra?: CliEnv): CliEnv {
  return { ...process.env, PATH: augmentedPath(), ...extra };
}

/** Trimmed `<cli> --version` output, or null when the binary isn't there. */
export function probeCliVersion(cli: string, env: CliEnv = cliEnv(), timeout = 8000): Promise<string | null> {
  return new Promise((resolve) => {
    execCli(cli, ["--version"], { timeout, env }, (err, stdout) => resolve(err ? null : stdout.trim()));
  });
}

interface StderrStream {
  stderr: { on(event: "data", listener: (chunk: Buffer | string) => void): unknown } | null;
}

/** Keep the last `max` bytes of a child's stderr; the getter reads the tail.
 * Unbounded accumulation would hold a whole run's noise for one error line. */
export function trackStderr(child: StderrStream, max = 8192): () => string {
  let stderr = "";
  child.stderr?.on("data", (c) => {
    stderr += c;
    if (stderr.length > max) stderr = stderr.slice(-max);
  });
  return () => stderr;
}
