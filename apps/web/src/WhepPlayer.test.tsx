import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { I18nProvider } from "./i18n";
import {
  descriptorPollingEnabled,
  descriptorPresentation,
  descriptorRefetchInterval,
  descriptorRetryAllowed,
  streamDescriptorPath,
  WhepPlayer,
} from "./WhepPlayer";
import type { StreamStatus } from "./api";
import type { StreamDescriptor } from "./api";

describe("Run video lifecycle", () => {
  it.each([
    ["CREATED", "starting"],
    ["STARTING", "starting"],
    ["READY", "connecting"],
    ["ERROR", "unavailable"],
    ["STOPPING", "closing"],
    ["STOPPED", "unavailable"],
  ] as Array<[StreamStatus, string]>) (
    "maps descriptor %s to %s",
    (status, expected) => {
      expect(descriptorPresentation("running", status)).toBe(expected);
    },
  );

  it("polls only while the Run is running", () => {
    expect(descriptorPollingEnabled("running")).toBe(true);
    for (const status of ["queued", "initializing", "finalizing", "stopping", "completed", "failed", "stopped"] as const) {
      expect(descriptorPollingEnabled(status)).toBe(false);
    }
  });

  it("requests the canonical overview camera descriptor path", () => {
    expect(streamDescriptorPath("run/id 1"))
      .toBe("/api/v1/runs/run%2Fid%201/stream?camera_id=overview");
  });

  it("retries the first descriptor failure only for a running Run", () => {
    expect(descriptorRetryAllowed("running", 0)).toBe(true);
    expect(descriptorRetryAllowed("running", 4)).toBe(false);
    expect(descriptorRetryAllowed("completed", 0)).toBe(false);
  });

  it("polls CREATED and STARTING with capped backoff until READY", () => {
    expect(descriptorRefetchInterval("CREATED", 1)).toBe(1_000);
    expect(descriptorRefetchInterval("STARTING", 2)).toBe(2_000);
    expect(descriptorRefetchInterval("STARTING", 9)).toBe(8_000);
    expect(descriptorRefetchInterval("READY", 3)).toBe(false);
  });

  it.each(["completed", "failed", "stopped"] as const)(
    "renders terminal Run %s as normally closed even after a descriptor failure",
    (status) => {
      expect(descriptorPresentation(status, undefined, true)).toBe("closed");
    },
  );

  it("does not present terminal HTTP 409 as reconnecting", () => {
    expect(descriptorPresentation("completed", undefined, true)).not.toBe("reconnecting");
  });

  it("renders the formal video attributes", () => {
    const markup = renderToStaticMarkup(
      <I18nProvider>
        <WhepPlayer url="http://media.example.invalid/stream/whep" active={false} />
      </I18nProvider>,
    );
    expect(markup).toContain("autoplay=\"\"");
    expect(markup).toContain("muted=\"\"");
    expect(markup).toContain("playsinline=\"\"");
  });

  it("preserves run_id in the exact ten-field Frontend descriptor type", () => {
    const descriptor: StreamDescriptor = {
      stream_id: "stream-1",
      run_id: "run-1",
      task_id: "task-1",
      camera_id: "overview",
      status: "READY",
      webrtc_url: "http://media.example.invalid/stream-1",
      whep_url: "http://media.example.invalid/stream-1/whep",
      width: 640,
      height: 480,
      fps: 15,
    };
    expect(Object.keys(descriptor).sort()).toEqual([
      "camera_id", "fps", "height", "run_id", "status",
      "stream_id", "task_id", "webrtc_url", "whep_url", "width",
    ]);
    expect(descriptor.run_id).toBe("run-1");
  });
});
