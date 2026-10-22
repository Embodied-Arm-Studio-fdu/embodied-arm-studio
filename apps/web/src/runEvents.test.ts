import { describe, expect, it } from "vitest";
import type { MetricEvent } from "./api";
import {
  isTerminalRunStatus,
  latestRunEventSequence,
  mergeRunEvents,
  normalizeRunEventsResponse,
  reconcileRunStatus,
  runConnectionLabel,
} from "./runEvents";

const stoppedEvent: MetricEvent = {
  event_id: "event-1",
  sequence: 1,
  type: "state",
  occurred_at: "2026-08-31T00:00:00Z",
  payload: { status: "stopped" },
};

describe("run event recovery", () => {
  it("treats completed as a normal terminal connection close", () => {
    expect(isTerminalRunStatus("completed")).toBe(true);
    expect(runConnectionLabel("completed")).toBe("训练已完成，结果已保存");
    expect(runConnectionLabel("running")).toBeNull();
  });

  it("localizes terminal connection labels in English", () => {
    expect(runConnectionLabel("completed", "en")).toBe("Training completed; results saved");
    expect(runConnectionLabel("stopped", "en")).toBe("Training stopped; results saved");
    expect(runConnectionLabel("failed", "en")).toBe("Training failed; live connection closed");
  });

  it("normalizes a canonical REST state event and advances sequence", () => {
    const events = normalizeRunEventsResponse([stoppedEvent]);

    expect(events).toHaveLength(1);
    expect(events[0].payload.status).toBe("stopped");
    expect(latestRunEventSequence(events)).toBe(1);
  });

  it("does not add a duplicate event_id", () => {
    expect(mergeRunEvents([stoppedEvent], [stoppedEvent])).toHaveLength(1);
  });

  it("does not let an older sequence replace the latest sequence", () => {
    const older = { ...stoppedEvent, event_id: "event-0", sequence: 0 };

    expect(latestRunEventSequence([older], 1)).toBe(1);
  });

  it.each(["stopped", "completed", "failed"] as const)(
    "does not regress terminal status %s with an old running event",
    (status) => {
      const running = {
        ...stoppedEvent,
        event_id: "event-0",
        sequence: 0,
        payload: { status: "running" as const },
      };

      expect(reconcileRunStatus(status, 1, running)).toBe(status);
    },
  );
});