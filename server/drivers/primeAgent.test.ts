// Prime Agent driver contract tests. The driver embeds the prime-agent SDK,
// so these tests stay offline: no createAgentSession/prompt (that would hit
// the NVIDIA endpoint). We pin config.models to suppress the background
// /v1/models discovery and drive everything against a throwaway OMB_DATA_DIR.
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, beforeAll, describe, expect, it } from "vitest";

import type { PrimeAgentConfig } from "./primeAgent.ts";

process.env.OMB_DATA_DIR = mkdtempSync(join(tmpdir(), "omb-prime-"));

let PrimeAgentDriver: Awaited<typeof import("./primeAgent.ts")>["PrimeAgentDriver"];

beforeAll(async () => {
  ({ PrimeAgentDriver } = await import("./primeAgent.ts"));
});

afterAll(() => {
  rmSync(process.env.OMB_DATA_DIR!, { recursive: true, force: true });
});

describe("PrimeAgentDriver.decodeConfig", () => {
  it("defaults to the NVIDIA NIM endpoint and key env", () => {
    const cfg = PrimeAgentDriver.decodeConfig({});
    expect(cfg).toEqual({
      provider: "nvidia",
      baseUrl: "https://integrate.api.nvidia.com/v1",
      apiKeyEnv: "NVIDIA_API_KEY",
      models: undefined,
    });
    expect(PrimeAgentDriver.decodeConfig(undefined)).toEqual(cfg);
  });

  it("accepts a custom provider, baseUrl, key env, and pinned catalog", () => {
    const cfg: PrimeAgentConfig = {
      provider: "mycloud",
      baseUrl: "http://127.0.0.1:9000/v1",
      apiKeyEnv: "MY_KEY",
      models: { default: "my-model", options: [{ id: "my-model", label: "My Model" }] },
    };
    expect(PrimeAgentDriver.decodeConfig(cfg)).toEqual(cfg);
  });
});

describe("PrimeAgentDriver snapshot", () => {
  it("is unavailable without a key", async () => {
    const inst = await PrimeAgentDriver.create({
      instanceId: "prime-test",
      displayName: "Prime Agent",
      environment: {},
      enabled: true,
      config: { provider: "nvidia", baseUrl: "https://integrate.api.nvidia.com/v1", apiKeyEnv: "NVIDIA_API_KEY" },
    });
    const snap = await inst.snapshot();
    expect(snap.state).toBe("unavailable");
    expect(snap.reason).toMatch(/no NVIDIA key/);
    await inst.dispose();
  });

  it("is available with a key (no network probe)", async () => {
    const inst = await PrimeAgentDriver.create({
      instanceId: "prime-test",
      displayName: "Prime Agent",
      environment: { NVIDIA_API_KEY: "nvapi-test" },
      enabled: true,
      config: { provider: "nvidia", baseUrl: "https://integrate.api.nvidia.com/v1", apiKeyEnv: "NVIDIA_API_KEY" },
    });
    const snap = await inst.snapshot();
    expect(snap.state).toBe("available");
    expect(snap.authenticated).toBe(true);
    expect(typeof snap.version).toBe("string");
    await inst.dispose();
  });
});

describe("PrimeAgentDriver models.json seeding", () => {
  it("writes a pinned catalog to the per-data-dir models.json", async () => {
    const inst = await PrimeAgentDriver.create({
      instanceId: "prime-seed",
      displayName: "Prime Agent",
      environment: {},
      enabled: true,
      config: {
        provider: "nvidia",
        baseUrl: "https://integrate.api.nvidia.com/v1",
        apiKeyEnv: "NVIDIA_API_KEY",
        models: { default: "openai/gpt-oss-120b", options: [{ id: "openai/gpt-oss-120b", label: "GPT-OSS 120B" }] },
      },
    });
    const payload = JSON.parse(readFileSync(join(process.env.OMB_DATA_DIR!, "prime-agent", "models.json"), "utf8"));
    const nvidia = payload.providers.nvidia;
    expect(nvidia.baseUrl).toBe("https://integrate.api.nvidia.com/v1");
    expect(nvidia.api).toBe("openai-completions");
    expect(nvidia.apiKey).toBe("NVIDIA_API_KEY");
    expect(nvidia.models.map((m: { id: string }) => m.id)).toEqual(["openai/gpt-oss-120b"]);
    expect(inst.models.default).toBe("openai/gpt-oss-120b");
    await inst.dispose();
  });

  it("seeds the curated fallback catalog with a flagship default", async () => {
    const inst = await PrimeAgentDriver.create({
      instanceId: "prime-fallback",
      displayName: "Prime Agent",
      environment: {},
      enabled: true,
      config: { provider: "nvidia", baseUrl: "https://integrate.api.nvidia.com/v1", apiKeyEnv: "NVIDIA_API_KEY" },
    });
    expect(inst.models.default).toBe("nvidia/nemotron-3-ultra-550b-a55b");
    expect(inst.models.options.length).toBeGreaterThan(5);
    await inst.dispose();
  });
});