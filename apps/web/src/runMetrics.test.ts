import { describe, expect, it } from "vitest";
import type { MetricEvent } from "./api";
import { metricSeriesData, runMetricHistory } from "./runMetrics";

function metricEvent(
  eventId: string,
  sequence: number,
  step: number,
  metrics: Record<string, number>,
): MetricEvent {
  return {
    event_id: eventId,
    sequence,
    type: "metric",
    payload: { step, metrics },
  };
}

describe("canonical run metric history", () => {
  it("sorts real metric events by one-based update_index", () => {
    const points = runMetricHistory([
      metricEvent("second", 4, 32, { update_index: 2, average_reward: 0.4 }),
      metricEvent("first", 2, 16, { update_index: 1, average_reward: 0.1 }),
    ]);

    expect(points.map((point) => [point.updateIndex, point.trainingSteps])).toEqual([
      [1, 16],
      [2, 32],
    ]);
    expect(metricSeriesData(points, "average_reward")).toEqual([[1, 0.1], [2, 0.4]]);
  });

  it("keeps the highest-sequence event for a duplicate update", () => {
    const points = runMetricHistory([
      metricEvent("new", 9, 16, { update_index: 1, entropy: 7.5 }),
      metricEvent("old", 3, 16, { update_index: 1, entropy: 7.1 }),
    ]);

    expect(metricSeriesData(points, "entropy")).toEqual([[1, 7.5]]);
  });

  it("does not invent points from missing, invalid, or legacy metrics", () => {
    const points = runMetricHistory([
      metricEvent("missing-index", 1, 16, { average_reward: 0.1 }),
      metricEvent("zero-index", 2, 16, { update_index: 0, average_reward: 0.2 }),
      metricEvent("fractional-index", 3, 16, { update_index: 1.5, entropy: 7.2 }),
      metricEvent("legacy", 3, 16, { update_index: 1, success_rate: 0.8 }),
      metricEvent("non-finite", 4, 32, { update_index: 2, average_reward: Number.NaN }),
      metricEvent("sparse", 5, 48, { update_index: 3, policy_loss: 0.3 }),
    ]);

    expect(points).toHaveLength(1);
    expect(metricSeriesData(points, "average_reward")).toEqual([[3, null]]);
    expect(metricSeriesData(points, "policy_loss")).toEqual([[3, 0.3]]);
  });

  it("preserves a missing metric as a chart gap instead of zero", () => {
    const points = runMetricHistory([
      metricEvent("first", 1, 16, { update_index: 1, value_loss: 0.4 }),
      metricEvent("gap", 2, 32, { update_index: 2, entropy: 7.2 }),
      metricEvent("third", 3, 48, { update_index: 3, value_loss: 0.2 }),
    ]);

    expect(metricSeriesData(points, "value_loss")).toEqual([
      [1, 0.4],
      [2, null],
      [3, 0.2],
    ]);
  });
});