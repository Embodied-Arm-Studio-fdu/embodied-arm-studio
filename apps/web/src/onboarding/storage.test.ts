import { describe, expect, it } from "vitest";
import { createTourStorageAdapter, onboardingStorageKey } from "./storage";

describe("first-time onboarding storage", () => {
  it("isolates users and keeps terminal states across adapter instances", () => {
    const data = new Map<string, string>();
    const storage = {
      getItem: (key: string) => data.get(key) ?? null,
      setItem: (key: string, value: string) => { data.set(key, value); },
      removeItem: (key: string) => { data.delete(key); },
    };
    const first = createTourStorageAdapter(storage);
    expect(first.hasSeenFirstTimeOnboarding("a")).toBe(false);
    first.markFirstTimeOnboardingCompleted("a");
    first.markFirstTimeOnboardingSkipped("b");
    const reloaded = createTourStorageAdapter(storage);
    expect(reloaded.hasSeenFirstTimeOnboarding("a")).toBe(true);
    expect(reloaded.hasSeenFirstTimeOnboarding("b")).toBe(true);
    expect(reloaded.hasSeenFirstTimeOnboarding("c")).toBe(false);
    expect(JSON.parse(data.get(onboardingStorageKey("a"))!).firstTimeOnboarding.status).toBe("completed");
    reloaded.resetOnboardingForDebug("a");
    expect(reloaded.hasSeenFirstTimeOnboarding("a")).toBe(false);
    expect(reloaded.hasSeenFirstTimeOnboarding("b")).toBe(true);
  });

  it("does not replay an older completed version", () => {
    const adapter = createTourStorageAdapter({
      getItem: () => JSON.stringify({ firstTimeOnboarding: { version: 0, status: "completed" } }),
      setItem: () => {}, removeItem: () => {},
    });
    expect(adapter.hasSeenFirstTimeOnboarding()).toBe(true);
  });
});
