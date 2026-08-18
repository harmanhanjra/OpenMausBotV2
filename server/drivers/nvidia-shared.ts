import type { ModelCatalog } from "../contracts.ts";

export const DEFAULT_URL = "https://integrate.api.nvidia.com/v1";
export const DEFAULT_KEY_ENV = "NVIDIA_API_KEY";

export function isNonChatModel(id: string): boolean {
  const t = id.toLowerCase();
  return [
    "embed", "rerank", "retriev", "search",
    "tts", "asr", "whisper", "parakeet", "speech", "vad", "audio", "vocoder",
    "cosmos", "diffus", "sdxl", "flux", "image-gen", "consistency",
    "guard", "shield", "reward", "classif", "moderat", "safety", "judge",
    "alphafold", "esm", "protein", "clip", "bge", "ocr", "img",
  ].some((token) => t.includes(token));
}

function prettyLabel(id: string): string {
  const base = id.split("/").pop() ?? id;
  const words = base.replace(/[-_]+/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  return words.length > 48 ? `${words.slice(0, 48)}…` : words;
}

export async function discoverModels(baseUrl: string, apiKey: string, timeoutMs = 4000): Promise<ModelCatalog | null> {
  try {
    const res = await fetch(`${baseUrl.replace(/\/+$/, "")}/models`, {
      headers: apiKey ? { authorization: `Bearer ${apiKey}` } : {},
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!res.ok) return null;
    const json: any = await res.json();
    if (!Array.isArray(json?.data)) return null;
    const seen = new Set<string>();
    const options = (json.data as Array<{ id?: unknown }>)
      .map((m) => m.id)
      .filter((id): id is string => typeof id === "string" && !!id && !isNonChatModel(id) && !seen.has(id))
      .map((id) => {
        seen.add(id);
        return { id, label: prettyLabel(id) };
      })
      .sort((a, b) => a.id.localeCompare(b.id));
    if (!options.length) return null;
    const def =
      options.find((o) => o.id === "meta/llama-3.3-70b-instruct") ??
      options.find((o) => /llama.*3\.3/i.test(o.id) || /nemotron/i.test(o.id)) ??
      options.find((o) => /instruct|chat/i.test(o.id)) ??
      options[0];
    return { default: def.id, options };
  } catch {
    return null;
  }
}
