// computer-client — the box-interaction layer behind the computer tools.
// Both consumers use it:
//   - server/computer-proxy.ts  — a stdio MCP server the claude CLI spawns
//   - server/drivers/primeAgent.ts — in-process custom tools for prime-agent
// It has NO MCP or driver framing: it owns the REST round trips to the
// box's run-command endpoint and the shell that turns a UI action into a
// single fused act → settle → capture command. The consumers turn the
// returned Frame/RunOut into their own tool-result shape.
//
// Latency rules that live here (same design as the original proxy):
//   - an action and its screenshot are ONE round trip; the frame rides
//     back in the same tool result as an image block ("act and observe"),
//   - JPEG, not PNG; downscale only when the display is wider than the
//     model's coordinate space,
//   - coordinate scaling happens box-side in shell arithmetic,
//   - big frames fall back to the files API instead of inline stdout,
//   - computer_batch runs a whole mechanical sequence in one round trip.
export const DEFAULT_BOX_API = "https://ascii.dev/api/box/v1";

/** The coordinate space the model sees: frames are downscaled to this
 * width, and clicks are scaled back up to the real display box-side. */
export const SHOT_WIDTH = 1280;
export const JPEG_QUALITY = 75;
export const SHOT_PATH = "/tmp/ogb-shot.jpg";
/** How long the desktop gets to repaint before the fused capture. */
export const SETTLE_MS = 350;
/** Gap between batched actions so focus changes land before typing. */
export const ACTION_GAP_MS = 120;
/** Frames larger than this come back over the files API instead of
 * inline stdout (keeps us clear of the command endpoint's stdout cap). */
export const INLINE_MAX_BYTES = 400_000;

export interface RunOut {
  ok: boolean;
  exitCode: number | null;
  stdout: string;
  stderr: string;
}

export interface Frame {
  data: string;
  mime: string;
  hash: string | null;
  geometry: { width: number; height: number } | null;
}

export class ComputerClient {
  readonly boxId: string;
  readonly token: string;
  readonly baseUrl: string;

  private inlineWorks = true; // flipped off for the client's life on first garbage
  private lastFrameHash: string | null = null;

  constructor(boxId: string, token: string, baseUrl: string = DEFAULT_BOX_API) {
    this.boxId = boxId;
    this.token = token;
    this.baseUrl = baseUrl;
  }

  /** Boxes archive themselves when idle (billing pauses, the disk
   * survives), which can happen mid-conversation — after that every
   * command comes back 409 machine_not_running. Wake it and carry on
   * rather than handing the agent a cryptic failure it can only guess at. */
  async resumeBox(): Promise<boolean> {
    const auth = { authorization: `Bearer ${this.token}`, "content-type": "application/json" };
    await fetch(`${this.baseUrl}/boxes/${this.boxId}/resume`, { method: "POST", headers: auth }).catch(() => null);
    const deadline = Date.now() + 90_000;
    while (Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 2000));
      const res = await fetch(`${this.baseUrl}/boxes/${this.boxId}`, { headers: auth }).catch(() => null);
      const body: any = await res?.json().catch(() => null);
      const state = body?.box?.state;
      if (state && ["idle", "ready", "running"].includes(state)) return true;
      if (state === "error") return false;
    }
    return false;
  }

  async runOnBox(command: string, timeoutMs = 60_000, allowWake = true): Promise<RunOut> {
    const res = await fetch(`${this.baseUrl}/boxes/${this.boxId}/commands`, {
      method: "POST",
      headers: { authorization: `Bearer ${this.token}`, "content-type": "application/json" },
      body: JSON.stringify({ command }),
      signal: AbortSignal.timeout(timeoutMs),
    });
    const body: any = await res.json().catch(() => null);
    if (res.status === 409 && allowWake) {
      const code = body?.code ?? body?.error?.code ?? "";
      if (/machine_not_running|box_starting|not_running|starting/i.test(String(code))) {
        const woke = await this.resumeBox();
        if (woke) return this.runOnBox(command, timeoutMs, false);
        return { ok: false, exitCode: null, stdout: "", stderr: "the computer is asleep and did not wake in time" };
      }
    }
    return {
      ok: res.ok && body?.exitCode === 0,
      exitCode: body?.exitCode ?? null,
      stdout: body?.stdout ?? "",
      stderr: body?.stderr ?? String(body?.message ?? (res.ok ? "" : `HTTP ${res.status}`)),
    };
  }

  /** Big frames (and any inline read that came back malformed) are fetched
   * over HTTP: raw artifact bytes first, the files API's base64-in-JSON
   * envelope second. Both are validated — an error page served with a 200
   * must fall through, not reach the model as an "image". */
  async fetchFrame(expectedBytes?: number): Promise<string | null> {
    const auth = { authorization: `Bearer ${this.token}` };
    try {
      const res = await fetch(
        `${this.baseUrl}/boxes/${this.boxId}/artifacts?path=${encodeURIComponent(SHOT_PATH)}`,
        { headers: auth, signal: AbortSignal.timeout(30_000) },
      );
      if (res.ok) {
        const bytes = Buffer.from(await res.arrayBuffer());
        if (wholeImage(bytes, expectedBytes)) return bytes.toString("base64");
      }
    } catch {
      /* fall through to the files API */
    }
    try {
      const res = await fetch(
        `${this.baseUrl}/boxes/${this.boxId}/files?path=${encodeURIComponent(SHOT_PATH)}&encoding=base64`,
        { headers: auth, signal: AbortSignal.timeout(30_000) },
      );
      const body: any = await res.json().catch(() => null);
      const content = body?.content;
      if (!res.ok || typeof content !== "string" || !content) return null;
      return wholeImage(Buffer.from(content, "base64"), expectedBytes) ? content : null;
    } catch {
      return null;
    }
  }

  async frameFrom(out: RunOut): Promise<Frame | null> {
    if (/SHOT_FAILED/.test(out.stdout)) return null;
    let hash: string | null = null;
    let geometry: Frame["geometry"] = null;
    let inline = "";
    let size = 0;
    for (const line of out.stdout.split("\n")) {
      if (line.startsWith("HASH ")) hash = line.slice(5).trim() || null;
      else if (line.startsWith("SIZE ")) size = Number(line.slice(5).trim()) || 0;
      else if (line.startsWith("GEOM ")) {
        const [w, h] = line.slice(5).trim().split(/\s+/).map(Number);
        if (Number.isFinite(w) && w > 0) geometry = { width: w, height: Number.isFinite(h) ? h : 0 };
      } else if (line.startsWith("B64 ")) inline = line.slice(4).trim();
    }
    if (inline && this.inlineWorks) {
      const bytes = Buffer.from(inline, "base64");
      if (wholeImage(bytes, size || undefined)) return { data: inline, mime: "image/jpeg", hash, geometry };
      // stdout mangled it (the failure this channel is known for) — never
      // hand a partial frame to the model; fetch it and stop trusting stdout
      this.inlineWorks = false;
    }
    const fetched = await this.fetchFrame(size || undefined);
    if (!fetched) return null;
    return { data: fetched, mime: "image/jpeg", hash, geometry };
  }

  /** An action result's content: the text plus the frame the action
   * produced. When the pixels are byte-identical to the frame the model
   * just saw, the image is dropped — it already has it, and it costs
   * ~1.2k tokens. Returns null content when there's no frame at all. */
  observedContent(note: string, frame: Frame | null): { text: string; image: Frame | null } {
    if (!frame) {
      return { text: `${note}\n(couldn't capture the screen — call screenshot to retry)`, image: null };
    }
    const unchanged = frame.hash != null && frame.hash === this.lastFrameHash;
    this.lastFrameHash = frame.hash ?? this.lastFrameHash;
    if (unchanged) {
      // deliberately does NOT suggest repeating the action: the action may
      // well have landed, and re-clicking a button that already submitted
      // is the expensive kind of wrong
      return {
        text: `${note}\n(the screen is identical to the frame you already have, so no new image is attached. Don't repeat the action — if you expected a change, it may still be rendering: call screenshot again in a moment, or re-check your coordinates against that frame.)`,
        image: null,
      };
    }
    return { text: note, image: frame };
  }

  /** Explicit look: always returns pixels, even if nothing moved. */
  async screenshotFrame(): Promise<Frame | null> {
    const out = await this.runOnBox(this.captureCommand(0), 60_000);
    const frame = await this.frameFrom(out);
    if (frame) this.lastFrameHash = frame.hash ?? this.lastFrameHash;
    return frame;
  }

  /** The whole point: one round trip carries geometry, the actions, the
   * settle, the capture and the frame bytes. Returns the acted flag, the
   * captured frame (may be null when observe was false or capture failed)
   * and any stderr from the action itself. */
  async actAndObserve(
    actions: any[],
    args: any,
    timeoutMs = 60_000,
  ): Promise<{ acted: boolean; frame: Frame | null; stderr: string }> {
    const parts: string[] = [];
    for (const a of actions) {
      const shell = this.actionShell(a);
      if (typeof shell !== "string") throw new Error(shell.error);
      // X11 needs a beat between steps — a click that focuses a field and
      // an immediate type will drop leading characters
      if (parts.length) parts.push(`sleep ${(ACTION_GAP_MS / 1000).toFixed(2)}`);
      parts.push(shell);
    }
    const observe = wantsFrame(args);
    // The actions run in a guarded group so a failing xdotool is REPORTED
    // rather than silently swallowed by the capture that follows it — but
    // the capture still runs, so the model always gets to see the state it
    // ended up in. Joining with ";" alone made a failed action look
    // identical to one that did nothing.
    const guarded = `if { ${parts.join("; ")}; }; then ACT=ok; else ACT=failed; fi`;
    const command = [ENV, GEOMETRY, guarded, observe ? captureBlock(settleOf(args)) : "true", 'echo "ACT $ACT"'].join(
      "; ",
    );
    const out = await this.runOnBox(command, timeoutMs);
    const acted = /^ACT ok$/m.test(out.stdout);
    const frame = observe ? await this.frameFrom(out) : null;
    return { acted, frame, stderr: out.stderr };
  }

  /** act → settle → capture → hash → (inline base64 if small). One hop. */
  captureCommand(settleMs = SETTLE_MS): string {
    return [ENV, GEOMETRY, captureBlock(settleMs)].join("; ");
  }

  /** Just the capture block (no ENV/GEOMETRY prefix) for embedding after
   * an inline launch sequence that already set up the display. */
  captureBlock(settleMs = SETTLE_MS): string {
    return captureBlock(settleMs);
  }

  /** Resolve the real display size into $W/$H for box-side click scaling. */
  get geometryShell(): string {
    return GEOMETRY;
  }

  /** Shell that turns a screenshot-space coordinate into a display one. */
  scaled(varName: string, value: number): string {
    return scaled(varName, value);
  }

  /** One action → the shell that performs it (scaling clicks box-side). */
  actionShell(a: any): string | { error: string } {
    return actionShell(a);
  }
}

const ENV = "export DISPLAY=${DISPLAY:-:0}";

/** Resolve the real display size into $W/$H for box-side click scaling. */
const GEOMETRY = [
  "g=$(xdotool getdisplaygeometry 2>/dev/null)",
  "W=${g%% *}",
  "H=${g##* }",
  `case "$W" in ''|*[!0-9]*) W=${SHOT_WIDTH}; H=0;; esac`,
].join("; ");

/** Shell that turns a screenshot-space coordinate into a display one.
 * The capture only downscales when the display is WIDER than the model's
 * space, so scaling must be conditional on exactly the same test — on a
 * 1024-wide desktop the frame is native size and a blind /1280 would put
 * every click at 80% of where the model aimed. */
function scaled(varName: string, value: number): string {
  const v = Math.round(value);
  return `if [ "$W" -gt ${SHOT_WIDTH} ] 2>/dev/null; then ${varName}=$(( ${v} * W / ${SHOT_WIDTH} )); else ${varName}=${v}; fi`;
}

/** act → settle → capture → hash → (inline base64 if small). One hop. */
function captureBlock(settleMs = SETTLE_MS): string {
  return [
    settleMs > 0 ? `sleep ${(settleMs / 1000).toFixed(2)}` : "true",
    `f=${SHOT_PATH}`,
    `rm -f "$f" 2>/dev/null || true`,
    `scrot -o -q ${JPEG_QUALITY} "$f" 2>/dev/null || import -window root -quality ${JPEG_QUALITY} "$f" 2>/dev/null || ffmpeg -y -f x11grab -i "$DISPLAY" -frames:v 1 -q:v 6 "$f" >/dev/null 2>&1`,
    // only re-encode when the display is bigger than the model's space —
    // ImageMagick startup is the most expensive step in the old pipeline
    `if [ "$W" -gt ${SHOT_WIDTH} ] 2>/dev/null && command -v convert >/dev/null 2>&1; then convert "$f" -thumbnail ${SHOT_WIDTH}x -quality ${JPEG_QUALITY} "$f" 2>/dev/null || true; fi`,
    `if [ ! -s "$f" ]; then echo SHOT_FAILED; exit 0; fi`,
    'echo "GEOM $W $H"',
    'echo "HASH $(md5sum "$f" 2>/dev/null | cut -d\' \' -f1)"',
    's=$(stat -c%s "$f" 2>/dev/null || echo 0)',
    // SIZE is what makes the inline path safe: the frame is only trusted
    // when the bytes we decoded match the bytes the box says it wrote
    'echo "SIZE $s"',
    `if [ "$s" -gt 0 ] && [ "$s" -le ${INLINE_MAX_BYTES} ]; then echo "B64 $(base64 -w0 "$f" 2>/dev/null || base64 "$f" | tr -d '\\n')"; fi`,
  ].join("; ");
}

/** A frame is only trusted when the bytes are a WHOLE image. Checking the
 * magic number alone is not enough: the box's command stdout has been
 * observed truncating a payload, and a truncated JPEG still starts with a
 * valid header — it just renders as a grey half-frame for the model. So
 * every frame must also end with its terminator, and (when the box told
 * us how many bytes it wrote) match that length exactly. */
export function wholeImage(bytes: Buffer, expectedBytes?: number): boolean {
  if (bytes.length < 512) return false;
  if (expectedBytes && bytes.length !== expectedBytes) return false;
  const jpeg = bytes[0] === 0xff && bytes[1] === 0xd8;
  const png = bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47;
  if (jpeg) {
    // EOI marker, allowing for trailing padding some encoders append
    const tail = bytes.subarray(Math.max(0, bytes.length - 32));
    return tail.includes(Buffer.from([0xff, 0xd9]));
  }
  if (png) {
    const tail = bytes.subarray(Math.max(0, bytes.length - 12));
    return tail.includes(Buffer.from("IEND", "ascii"));
  }
  return false;
}

export const shellQuote = (s: string) => `'${s.replace(/'/g, "'\\''")}'`;
const settleOf = (args: any) => Math.min(Math.max(Number(args?.settle_ms) || SETTLE_MS, 0), 3000);
export const wantsFrame = (args: any) => args?.observe !== false;

/** One action → the shell that performs it (scaling clicks box-side). */
function actionShell(a: any): string | { error: string } {
  const kind = String(a?.action ?? "");
  if (kind === "click") {
    const x = Math.round(Number(a.x));
    const y = Math.round(Number(a.y));
    if (!Number.isFinite(x) || !Number.isFinite(y)) return { error: "click needs numeric x,y" };
    const btn = a.button === "right" ? 3 : 1;
    const rep = a.double ? "--repeat 2 --delay 60 " : "";
    return `${scaled("CX", x)}; ${scaled("CY", y)}; xdotool mousemove $CX $CY click ${rep}${btn}`;
  }
  if (kind === "type_text") {
    const t = String(a.text ?? "");
    if (!t) return { error: "type_text needs text" };
    return `xdotool type --delay 8 ${shellQuote(t)}`;
  }
  if (kind === "press_key") {
    const keys = String(a.keys ?? "").replace(/[^\w+]/g, "");
    if (!keys) return { error: "press_key needs keys" };
    return `xdotool key ${keys}`;
  }
  if (kind === "scroll") {
    const clicks = Math.min(Math.max(Math.round(Number(a.clicks) || 3), 1), 20);
    const btn = a.direction === "up" ? 4 : 5;
    return `xdotool click --repeat ${clicks} ${btn}`;
  }
  if (kind === "wait") {
    const ms = Math.min(Math.max(Number(a.ms) || 500, 0), 5000);
    return `sleep ${(ms / 1000).toFixed(2)}`;
  }
  return { error: `unknown action ${kind || "(missing)"}` };
}