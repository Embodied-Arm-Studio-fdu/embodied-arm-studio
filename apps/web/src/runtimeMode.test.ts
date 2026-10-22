import { afterEach, describe, expect, it, vi } from "vitest";
import {
  CANONICAL_PLATFORM_RUNTIME_CONTEXT,
  LEGACY_QUICK_COURSE_RUNTIME_CONTEXT,
  runtimeModeLabels,
} from "./runtimeMode";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("authenticated runtime mode labels", () => {
  it("uses simulation labels for the canonical authenticated shared shell", () => {
    const labels = runtimeModeLabels(
      CANONICAL_PLATFORM_RUNTIME_CONTEXT,
    );

    expect(labels).toEqual({
      mode: "仿真训练模式",
      capability: "支持 GPU 训练",
      safety: "真实机械臂已禁用",
    });
    expect(Object.values(labels)).not.toContain("MOCK MODE");
  });

  it("does not claim that a Run is currently training", () => {
    const labels = runtimeModeLabels(
      CANONICAL_PLATFORM_RUNTIME_CONTEXT,
    );

    expect(Object.values(labels).join(" ")).not.toMatch(
      /TRAINING (RUNNING|IN PROGRESS)/,
    );
  });

  it("keeps the explicitly isolated legacy Quick Course labeled as mock", () => {
    expect(
      runtimeModeLabels(LEGACY_QUICK_COURSE_RUNTIME_CONTEXT),
    ).toEqual({
      mode: "演示练习模式",
      capability: "仅用于教学演示",
      safety: "真实机械臂已禁用",
    });
  });

  it("returns English labels for both runtime contexts", () => {
    expect(runtimeModeLabels(CANONICAL_PLATFORM_RUNTIME_CONTEXT, "en")).toEqual({
      mode: "Simulation training mode",
      capability: "GPU training supported",
      safety: "Real robot arm disabled",
    });
    expect(runtimeModeLabels(LEGACY_QUICK_COURSE_RUNTIME_CONTEXT, "en")).toEqual({
      mode: "Demo practice mode",
      capability: "For teaching demos only",
      safety: "Real robot arm disabled",
    });
  });

  it("does not issue API, stop, Run creation, WebSocket, or media requests", () => {
    const fetchRequest = vi.fn();
    const openWebSocket = vi.fn();
    const openPeerConnection = vi.fn();
    vi.stubGlobal("fetch", fetchRequest);
    vi.stubGlobal("WebSocket", openWebSocket);
    vi.stubGlobal("RTCPeerConnection", openPeerConnection);

    runtimeModeLabels(CANONICAL_PLATFORM_RUNTIME_CONTEXT);
    runtimeModeLabels(LEGACY_QUICK_COURSE_RUNTIME_CONTEXT);

    expect(fetchRequest).not.toHaveBeenCalled();
    expect(openWebSocket).not.toHaveBeenCalled();
    expect(openPeerConnection).not.toHaveBeenCalled();
  });
});
