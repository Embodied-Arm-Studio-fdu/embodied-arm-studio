import { describe, expect, it, vi } from "vitest";
import {
  BROWSER_ID_COMPATIBILITY_ERROR,
  BrowserIdCompatibilityError,
  createBrowserId,
  type BrowserCrypto,
} from "./browserId";

const UUID_V4_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

describe("createBrowserId", () => {
  it("prefers crypto.randomUUID when available", () => {
    const expected = "123e4567-e89b-42d3-a456-426614174000";
    const randomUUID = vi.fn(() => expected);
    const getRandomValues = vi.fn((bytes: Uint8Array) => bytes);

    expect(createBrowserId({ randomUUID, getRandomValues })).toBe(expected);
    expect(randomUUID).toHaveBeenCalledOnce();
    expect(getRandomValues).not.toHaveBeenCalled();
  });

  it("uses getRandomValues to generate unique UUID v4 IDs", () => {
    let counter = 0;
    const cryptoSource: BrowserCrypto = {
      getRandomValues(bytes) {
        counter += 1;
        bytes.fill(counter);
        return bytes;
      },
    };

    const first = createBrowserId(cryptoSource);
    const second = createBrowserId(cryptoSource);

    expect(first).toMatch(UUID_V4_PATTERN);
    expect(second).toMatch(UUID_V4_PATTERN);
    expect(first).not.toBe(second);
    expect(first).toHaveLength(36);
  });

  it("throws a clear compatibility error without Web Crypto entropy", () => {
    expect(() => createBrowserId({})).toThrow(BrowserIdCompatibilityError);
    expect(() => createBrowserId({})).toThrow(BROWSER_ID_COMPATIBILITY_ERROR);
  });
});
