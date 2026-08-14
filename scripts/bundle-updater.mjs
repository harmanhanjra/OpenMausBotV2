// Bundle electron-updater into one self-contained file the packaged app can
// require. This app ships ZERO node_modules at runtime (the harness + UI are
// pre-compiled into Resources), so a main-process dependency has to be
// vendored. esbuild inlines electron-updater + its whole dep tree; `electron`
// stays external (resolved from the runtime). Output ships via files:electron/**.
import { build } from "esbuild";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const require = createRequire(import.meta.url);
const root = join(dirname(fileURLToPath(import.meta.url)), "..");

await build({
  entryPoints: [require.resolve("electron-updater")],
  bundle: true,
  platform: "node",
  target: "node20",
  format: "cjs",
  external: ["electron"],
  outfile: join(root, "electron/vendor/electron-updater.cjs"),
  logLevel: "info",
});
