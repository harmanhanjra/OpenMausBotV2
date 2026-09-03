import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { safeExternalUrl } from "./security.mjs";

describe("safeExternalUrl", () => {
  for (const [input, expected] of [
    ["https://example.com/docs", "https://example.com/docs"],
    ["http://127.0.0.1:8799/health", "http://127.0.0.1:8799/health"],
  ]) {
    it(`allows web URL ${input}`, () => {
      assert.equal(safeExternalUrl(input), expected);
    });
  }

  for (const input of [
    "file:///etc/passwd",
    "smb://fileserver/share",
    "javascript:alert(1)",
    "data:text/html,unsafe",
    "openmausbot://action",
    "not a url",
  ]) {
    it(`rejects non-web target ${input}`, () => {
      assert.equal(safeExternalUrl(input), null);
    });
  }
});
