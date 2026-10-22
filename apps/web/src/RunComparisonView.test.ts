import { describe, expect, it } from "vitest";
import type { MetricEvent, RunRecord } from "./api";
import {
  displayEvidenceSummary,
  runComparisonMetricRows,
  summarizeDiagnosticChannel,
  summarizeFailureChannel,
} from "./RunComparisonView";

function run(evaluation: RunRecord["metrics_summary"]["evaluation"]): RunRecord {
  return {
    run_id: "run-compare", user_id: "user", project_id: "project", class_id: null,
    execution_mode: "simulation", run_kind: "train", project_config_snapshot: {},
    code_sha256: "0".repeat(64), simulation_task_id: "task", progress: 1,
    metrics_summary: evaluation ? { evaluation } : {}, error: null,
    last_sequence: 1, training_steps: 20000, update_steps: 100, stop_reason: null,
    created_at: "2026-09-14T00:00:00Z", started_at: "2026-09-14T00:00:10Z",
    ended_at: "2026-09-14T00:10:15Z", updated_at: "2026-09-14T00:10:15Z",
    status: "completed",
  };
}

function event(payload: MetricEvent["payload"], type: MetricEvent["type"] = "metric"): MetricEvent {
  return { event_id: crypto.randomUUID(), type, payload };
}

describe("Run comparison evidence channels", () => {
  it("renders an explicit zero differently from an unavailable channel", () => {
    expect(displayEvidenceSummary({ available: true, count: 0 }, " 次触发")).toBe("0 次触发");
    expect(displayEvidenceSummary({ available: false, count: 0 }, " 次触发")).toBe("未记录");
    expect(displayEvidenceSummary({ available: false, count: 0 }, " triggers", "en")).toBe("Not recorded");
    expect(displayEvidenceSummary({ available: true, count: 2 }, { zh: " 次触发", en: " triggers" }, "en")).toBe("2 triggers");
  });

  it("treats an explicit zero failure rate as available evidence", () => {
    expect(summarizeFailureChannel([event({ metrics: { failure_rate: 0 } })])).toEqual({
      available: true,
      count: 0,
    });
  });

  it("keeps a missing failure channel unavailable", () => {
    expect(summarizeFailureChannel([event({ metrics: { average_reward: 1 } })])).toEqual({
      available: false,
      count: 0,
    });
  });

  it("counts explicit error events even without a failure-rate metric", () => {
    expect(summarizeFailureChannel([event({}, "error")])).toEqual({
      available: true,
      count: 1,
    });
  });

  it("distinguishes zero collision triggers from a missing collision channel", () => {
    expect(summarizeDiagnosticChannel([event({ collision: { detected: false } })], "collision")).toEqual({
      available: true,
      count: 0,
    });
    expect(summarizeDiagnosticChannel([event({})], "collision")).toEqual({
      available: false,
      count: 0,
    });
  });

  it("counts only explicit joint-limit violations", () => {
    const events = [
      event({ joint_limits: { violated: false } }),
      event({ joint_limits: { violated: true } }),
    ];
    expect(summarizeDiagnosticChannel(events, "joint_limits")).toEqual({
      available: true,
      count: 1,
    });
  });

  it("compares formal policy metrics without turning missing evaluation into zero", () => {
    const evaluated = run({
      status: "failed", deterministic: true, policy_passed: false,
      episode_count: 100, success_count: 25, success_rate: 0.25,
      wilson_95_lower_bound: 0.175, wilson_95_upper_bound: 0.343,
      mean_final_distance: 0.2, mean_object_displacement: 0.12,
    });
    const rows = runComparisonMetricRows([evaluated, run(undefined)]);
    expect(rows.find((row) => row.key === "success_rate")?.values).toEqual(["25.00%", "未记录"]);
    expect(rows.find((row) => row.key === "training_duration")?.values).toEqual(["0h 10m 5s", "0h 10m 5s"]);
    expect(runComparisonMetricRows([evaluated, run(undefined)], "en").find((row) => row.key === "success_rate")).toEqual({
      key: "success_rate",
      label: "Formal evaluation success rate",
      values: ["25.00%", "Not recorded"],
    });
  });
});
