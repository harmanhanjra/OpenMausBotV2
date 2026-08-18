// Newline-delimited framing — the one place the server splits a byte stream
// into lines. Every stdio/socket protocol here (claude stream-json, the codex
// app-server, ACP, the MCP proxies, the permission broker) is NDJSON, and each
// of them used to carry its own buffer-and-indexOf loop.
export type LineSink = (line: string) => void;

/** Feed chunks in, get whole lines out. Blank lines are dropped. */
export function createLineSplitter(onLine: LineSink): (chunk: string | Uint8Array) => void {
  let buf = "";
  return (chunk) => {
    buf += chunk;
    let nl: number;
    while ((nl = buf.indexOf("\n")) !== -1) {
      const line = buf.slice(0, nl);
      buf = buf.slice(nl + 1);
      if (line.trim()) onLine(line);
    }
  };
}

/** Minimal shape of the streams we frame (child stdout/stderr, sockets, stdin). */
interface ChunkStream {
  on(event: "data", listener: (chunk: Buffer | string) => void): unknown;
}

export function onLines(stream: ChunkStream, onLine: LineSink): void {
  stream.on("data", createLineSplitter(onLine));
}

/** onLines + JSON.parse, silently dropping malformed frames. */
export function onJsonLines(stream: ChunkStream, onMessage: (msg: any) => void): void {
  onLines(stream, (line) => {
    let msg: any;
    try {
      msg = JSON.parse(line);
    } catch {
      return;
    }
    onMessage(msg);
  });
}
