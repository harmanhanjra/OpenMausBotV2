// Filesystem helpers that keep failures visible. A missing file is an
// expected state (first run, already-deleted thread); anything else is an
// error worth reporting rather than swallowing.
import { readFileSync, renameSync, unlinkSync } from "node:fs";

const errno = (e: unknown): string | undefined =>
  typeof e === "object" && e !== null ? (e as { code?: string }).code : undefined;

export const isMissing = (e: unknown): boolean => errno(e) === "ENOENT";

/**
 * Parse a JSON file the app also rewrites wholesale. Returns null when the
 * file does not exist. A file that exists but holds invalid JSON is moved
 * aside as `<name>.corrupt-<timestamp>` and reported: starting empty over a
 * file we still own would erase the only copy of the user's data on the next
 * write. Read failures other than "missing" propagate — an unreadable
 * bots.json must stop the server, not silently reset the fleet.
 */
export function readJsonFile<T>(path: string, label: string): T | null {
  let raw: string;
  try {
    raw = readFileSync(path, "utf8");
  } catch (e) {
    if (isMissing(e)) return null;
    throw new Error(`${label}: cannot read ${path}: ${e instanceof Error ? e.message : String(e)}`, { cause: e });
  }
  try {
    return JSON.parse(raw) as T;
  } catch (e) {
    const quarantine = `${path}.corrupt-${Date.now()}`;
    console.error(
      `${label}: ${path} is not valid JSON (${e instanceof Error ? e.message : String(e)}) — moving it to ${quarantine} and starting empty`,
    );
    try {
      renameSync(path, quarantine);
    } catch (moveError) {
      console.error(`${label}: could not preserve ${path}:`, moveError);
    }
    return null;
  }
}

/** Delete a file, ignoring "already gone" and reporting everything else. */
export function removeFile(path: string, label: string): void {
  try {
    unlinkSync(path);
  } catch (e) {
    if (!isMissing(e)) console.error(`${label}: could not delete ${path}:`, e);
  }
}
