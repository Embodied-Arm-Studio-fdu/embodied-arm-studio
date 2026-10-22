export const BROWSER_ID_COMPATIBILITY_ERROR =
  "当前浏览器无法安全生成请求 ID。请使用支持 Web Crypto 的浏览器或 HTTPS 访问后重试。";

export class BrowserIdCompatibilityError extends Error {
  constructor() {
    super(BROWSER_ID_COMPATIBILITY_ERROR);
    this.name = "BrowserIdCompatibilityError";
  }
}

export interface BrowserCrypto {
  randomUUID?: () => string;
  getRandomValues?: (array: Uint8Array) => Uint8Array;
}

export function createBrowserId(
  cryptoSource: BrowserCrypto | undefined = globalThis.crypto,
): string {
  if (typeof cryptoSource?.randomUUID === "function") {
    return cryptoSource.randomUUID();
  }
  if (typeof cryptoSource?.getRandomValues !== "function") {
    throw new BrowserIdCompatibilityError();
  }

  const bytes = cryptoSource.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (value) => value.toString(16).padStart(2, "0"));
  return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex
    .slice(6, 8)
    .join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10).join("")}`;
}
