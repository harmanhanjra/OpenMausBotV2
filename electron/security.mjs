const EXTERNAL_PROTOCOLS = new Set(["https:", "http:"]);

/**
 * Return a normalized URL only when it is safe to hand to the operating
 * system. Electron must not forward arbitrary renderer-controlled schemes
 * (for example file:, smb:, or custom application protocols).
 */
export function safeExternalUrl(value) {
  try {
    const url = new URL(value);
    return EXTERNAL_PROTOCOLS.has(url.protocol) ? url.href : null;
  } catch {
    return null;
  }
}
