// Native (un-normalized) protocol tee — the debugging trick from upstream's
// EventNdjsonLogger and agentcal's onRaw: every provider-native message is
// written verbatim next to the canonical stream, so protocol drift can be
// diagnosed by diffing the two.
import { appendFileSync } from "node:fs";
import { join } from "node:path";

import { NATIVE_DIR } from "../config.ts";

// the tee fails once per message; one line per process keeps the reason
// visible without drowning the log it is complaining about
let loggedWriteFailure = false;

export function appendNative(threadId: string, entry: { dir: "in" | "out"; source: string; msg: unknown }) {
  try {
    appendFileSync(
      join(NATIVE_DIR, `${threadId}.ndjson`),
      JSON.stringify({ at: new Date().toISOString(), ...entry }) + "\n",
    );
  } catch (e) {
    // never let logging break a run — but a tee that silently stopped
    // recording makes protocol drift undiagnosable
    if (!loggedWriteFailure) {
      loggedWriteFailure = true;
      console.error(`native log: cannot write in ${NATIVE_DIR} (further failures are not repeated) —`, e);
    }
  }
}
