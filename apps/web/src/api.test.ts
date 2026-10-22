import { afterEach, describe, expect, it, vi } from "vitest";

function sessionStorageStub(initialValue: string | null = null) {
  let value = initialValue;
  return {
    getItem: vi.fn(() => value),
    setItem: vi.fn((_key: string, next: string) => { value = next; }),
    removeItem: vi.fn(() => { value = null; }),
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.resetModules();
});

describe("HTTP browser request IDs", () => {
  it("generates distinct UUID v4 idempotency keys without native randomUUID", async () => {
    const getRandomValues = globalThis.crypto.getRandomValues.bind(globalThis.crypto);
    vi.stubGlobal("crypto", { getRandomValues });
    const { idempotencyKey } = await import("./api");
    const keys = Array.from({ length: 20 }, () => idempotencyKey());

    expect(new Set(keys).size).toBe(keys.length);
    for (const key of keys) {
      expect(key).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
    }
  });
});

describe("access token persistence", () => {
  it("restores the token from the current tab session", async () => {
    const sessionStorage = sessionStorageStub("persisted-token");
    vi.stubGlobal("window", { sessionStorage });

    const { getAccessToken } = await import("./api");

    expect(getAccessToken()).toBe("persisted-token");
  });

  it("updates and clears the current tab session", async () => {
    const sessionStorage = sessionStorageStub();
    vi.stubGlobal("window", { sessionStorage });
    const { getAccessToken, setAccessToken } = await import("./api");

    setAccessToken("new-token");
    expect(getAccessToken()).toBe("new-token");
    expect(sessionStorage.setItem).toHaveBeenCalledWith(
      "embodied-arm-access-token",
      "new-token",
    );

    setAccessToken(null);
    expect(getAccessToken()).toBeNull();
    expect(sessionStorage.removeItem).toHaveBeenCalledWith(
      "embodied-arm-access-token",
    );
  });
});

describe("API error messages", () => {
  it("uses a structured FastAPI detail message without stringifying the object", async () => {
    const sessionStorage = sessionStorageStub();
    vi.stubGlobal("window", { sessionStorage, dispatchEvent: vi.fn() });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ detail: { code: "run_conflict", message: "运行状态冲突" } }),
      { status: 409, headers: { "Content-Type": "application/json", "x-request-id": "req-1" } },
    )));
    const { api } = await import("./api");

    await expect(api("/api/v1/runs/test")).rejects.toMatchObject({
      message: "运行状态冲突（request_id: req-1）",
      status: 409,
      requestId: "req-1",
    });
  });

  it("aborts a request that never completes", async () => {
    vi.useFakeTimers();
    const sessionStorage = sessionStorageStub();
    vi.stubGlobal("window", { sessionStorage, dispatchEvent: vi.fn(), setTimeout, clearTimeout });
    vi.stubGlobal("fetch", vi.fn((_path: string, options?: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        options?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
      }),
    ));
    const { api } = await import("./api");
    const outcome = api("/api/agent/sessions/test/messages").then(
      () => ({ ok: true as const }),
      (error) => ({ ok: false as const, error }),
    );

    await vi.advanceTimersByTimeAsync(35_000);
    await expect(outcome).resolves.toMatchObject({ ok: false, error: { message: "请求超时，请检查 Agent Provider 的网络连接后重试" } });
    vi.useRealTimers();
  });

  it("preserves string details and falls back for unsupported detail shapes", async () => {
    const sessionStorage = sessionStorageStub();
    vi.stubGlobal("window", { sessionStorage, dispatchEvent: vi.fn() });
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(
        JSON.stringify({ detail: "运行不存在" }),
        { status: 404, headers: { "Content-Type": "application/json" } },
      ))
      .mockResolvedValueOnce(new Response(
        JSON.stringify({ detail: [{ msg: "invalid" }] }),
        { status: 422, statusText: "Unprocessable Content", headers: { "Content-Type": "application/json" } },
      ));
    vi.stubGlobal("fetch", fetchMock);
    const { api } = await import("./api");

    await expect(api("/missing")).rejects.toThrow("运行不存在");
    await expect(api("/invalid")).rejects.toThrow("Unprocessable Content");
  });
});
