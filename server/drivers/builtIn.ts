// Built-in driver registration — upstream builtInDrivers.ts: a static
// array, nothing more. Adding a driver = write drivers/<x>.ts, append.
import type { AnyProviderDriver } from "../contracts.ts";
import { AntigravityDriver } from "./antigravity.ts";
import { BoxAgentDriver } from "./boxagent.ts";
import { ClaudeDriver } from "./claude.ts";
import { CodexDriver } from "./codex.ts";
import { GrokDriver } from "./grok.ts";
import { NvidiaDriver } from "./nvidia.ts";
import { PrimeAgentDriver } from "./primeAgent.ts";
import { DeepseekDriver } from "./deepseek.ts";
import { GrokAgentDriver } from "./acp/grok.ts";
import { GeminiAgentDriver } from "./acp/gemini.ts";

export const BUILT_IN_DRIVERS: readonly AnyProviderDriver[] = [
  GrokDriver,
  NvidiaDriver,
  PrimeAgentDriver,
  GrokAgentDriver,
  GeminiAgentDriver,
  ClaudeDriver,
  CodexDriver,
  AntigravityDriver,
  BoxAgentDriver,
  DeepseekDriver,
];
