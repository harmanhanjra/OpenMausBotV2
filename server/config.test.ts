// Config contract: one file (~/.openmausbot/config.json) with env
// fallbacks, merge-on-save (a patch never clobbers sibling keys), the
// legacy ~/.opengrokbot migration, and the default fleet + per-instance
// key injection.
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { DATA_DIR, ensureDirs, instanceConfigs, loadConfig, saveConfig } from "./config.ts";

const CONFIG_PATH = join(DATA_DIR, "config.json");
const ENV_KEYS = ["XAI_API_KEY", "NVIDIA_API_KEY", "COMPOSIO_KEY", "BOX_TOKEN"] as const;
const savedEnv: Record<string, string | undefined> = {};

beforeEach(() => {
  rmSync(DATA_DIR, { recursive: true, force: true });
  for (const k of ENV_KEYS) {
    savedEnv[k] = process.env[k];
    delete process.env[k];
  }
});

afterEach(() => {
  for (const k of ENV_KEYS) {
    if (savedEnv[k] === undefined) delete process.env[k];
    else process.env[k] = savedEnv[k];
  }
});

describe("ensureDirs", () => {
  it("creates the data, events and native dirs", () => {
    ensureDirs();
    expect(existsSync(DATA_DIR)).toBe(true);
    expect(existsSync(join(DATA_DIR, "events"))).toBe(true);
    expect(existsSync(join(DATA_DIR, "native"))).toBe(true);
  });

  it("migrates the legacy .opengrokbot dir when the new one is absent", () => {
    const legacy = join(homedir(), ".opengrokbot");
    rmSync(legacy, { recursive: true, force: true });
    mkdirSync(legacy, { recursive: true });
    writeFileSync(join(legacy, "config.json"), JSON.stringify({ profile: { name: "Legacy" } }));

    ensureDirs();
    expect(existsSync(legacy)).toBe(false);
    expect(JSON.parse(readFileSync(CONFIG_PATH, "utf8"))).toEqual({ profile: { name: "Legacy" } });
  });

  it("leaves an existing data dir alone even when the legacy dir exists", () => {
    mkdirSync(DATA_DIR, { recursive: true });
    writeFileSync(CONFIG_PATH, JSON.stringify({ profile: { name: "Current" } }));
    const legacy = join(homedir(), ".opengrokbot");
    mkdirSync(legacy, { recursive: true });

    ensureDirs();
    expect(JSON.parse(readFileSync(CONFIG_PATH, "utf8"))).toEqual({ profile: { name: "Current" } });
    rmSync(legacy, { recursive: true, force: true });
  });
});

describe("loadConfig", () => {
  it("returns env fallbacks on a first run with no file", () => {
    process.env.XAI_API_KEY = "xai-env";
    process.env.BOX_TOKEN = "box-env";
    const cfg = loadConfig();
    expect(cfg.xai?.key).toBe("xai-env");
    expect(cfg.box?.token).toBe("box-env");
    expect(cfg.nvidia?.key).toBeUndefined();
  });

  it("prefers file values over env fallbacks", () => {
    process.env.XAI_API_KEY = "xai-env";
    mkdirSync(DATA_DIR, { recursive: true });
    writeFileSync(CONFIG_PATH, JSON.stringify({ xai: { key: "xai-file" }, composio: { key: "ck_file" } }));
    const cfg = loadConfig();
    expect(cfg.xai?.key).toBe("xai-file");
    expect(cfg.composio?.key).toBe("ck_file");
  });

  it("survives a corrupt config file", () => {
    mkdirSync(DATA_DIR, { recursive: true });
    writeFileSync(CONFIG_PATH, "{not json");
    expect(() => loadConfig()).not.toThrow();
  });
});

describe("saveConfig", () => {
  it("writes a fresh file on first save", () => {
    saveConfig({ profile: { name: "Ada" } });
    expect(JSON.parse(readFileSync(CONFIG_PATH, "utf8"))).toEqual({ profile: { name: "Ada" } });
  });

  it("merges a patch without clobbering sibling keys", () => {
    saveConfig({ xai: { key: "xai-1", url: "https://xai.example" }, box: { token: "b-1" } });
    saveConfig({ xai: { key: "xai-2" } });
    const disk = JSON.parse(readFileSync(CONFIG_PATH, "utf8"));
    expect(disk.xai).toEqual({ key: "xai-2", url: "https://xai.example" });
    expect(disk.box).toEqual({ token: "b-1" });
  });

  it("ignores non-object and unknown keys", () => {
    saveConfig({ xai: "nope", instances: { x: { driver: "grok" } } } as any);
    const disk = JSON.parse(readFileSync(CONFIG_PATH, "utf8"));
    expect(disk).toEqual({});
  });
});

describe("instanceConfigs", () => {
  it("builds the default fleet when no instances are configured", () => {
    const map = instanceConfigs({});
    expect(Object.keys(map).sort()).toEqual(
      ["antigravity", "claude", "codex", "computer", "grok", "nvidia", "prime"].sort(),
    );
    // the default grok instance rides the CLI driver, not the API-key one
    expect(map.grok.driver).toBe("grokAgent");
  });

  it("uses the configured instances verbatim instead of the defaults", () => {
    const map = instanceConfigs({ instances: { only: { driver: "nvidia" } } });
    expect(Object.keys(map)).toEqual(["only"]);
  });

  it("injects config-file keys as per-instance environment", () => {
    const map = instanceConfigs({ xai: { key: "xai-k" }, nvidia: { key: "nv-k" }, box: { token: "b-k" } });
    expect(map.claude.environment).toEqual({ XAI_API_KEY: "xai-k", NVIDIA_API_KEY: "nv-k", BOX_TOKEN: "b-k" });
  });

  it("lets explicit per-instance environment win over injected keys", () => {
    const map = instanceConfigs({
      xai: { key: "from-config" },
      instances: { g: { driver: "grok", environment: { XAI_API_KEY: "explicit" } } },
    });
    expect(map.g.environment?.XAI_API_KEY).toBe("explicit");
  });

  it("omits env entries for keys that are not configured", () => {
    const map = instanceConfigs({});
    expect(map.claude.environment).toEqual({});
  });
});
