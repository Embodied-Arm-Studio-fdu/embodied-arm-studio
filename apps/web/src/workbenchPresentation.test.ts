import { describe, expect, it } from "vitest";
import type { MetricEvent, RunRecord } from "./api";
import { experimentConditions, experimentMetrics, formatExperimentMetric } from "./workbenchPresentation";
import { PUSH_BASELINE, pushSceneDocument } from "./pushScene";
import { clampAgentPosition } from "./useFloatingAgentPosition";

function run(overrides: Partial<RunRecord> = {}): RunRecord {
  return { run_id: "run", project_id: "project", status: "completed", code_sha256: "reward-sha", parameters: { seed: 7, video_enabled: true }, metrics_summary: {}, project_config_snapshot: { environment_type: "s1_push", expected_training_steps: 20000, scene: { document: pushSceneDocument(PUSH_BASELINE, "scene-a") } }, ...overrides } as RunRecord;
}

describe("beginner experiment evidence", () => {
  it("does not invent success from completion or flattened metrics", () => {
    const rows = experimentMetrics(run({ metrics_summary: { success_rate: 1, average_reward: 0 } }));
    expect(formatExperimentMetric(rows[0], "zh-CN")).toBe("未记录");
    expect(formatExperimentMetric(rows[2], "en")).toBe("0.000");
  });
  it("uses valid formal evaluation and converts meters into centimeters", () => {
    const rows = experimentMetrics(run({ metrics_summary: { evaluation: { status: "passed", deterministic: true, policy_passed: true, success_rate: .99, mean_final_distance: .015 } } }));
    expect(formatExperimentMetric(rows[0], "en")).toBe("99.0 %");
    expect(formatExperimentMetric(rows[1], "en")).toBe("1.5 cm");
  });
  it("treats invalid evaluation quantities as missing evidence", () => {
    const rows = experimentMetrics(run({ metrics_summary: { evaluation: { status: "passed", deterministic: true, policy_passed: true, success_rate: 2, mean_final_distance: -.01 } } }));
    expect(rows[0].value).toBeNull();
    expect(rows[1].value).toBeNull();
  });
  it("uses recorded rollout events if a summary has no average reward", () => {
    const events = [{ event_id: "metric", sequence: 2, type: "metric", payload: { metrics: { update_index: 1, average_reward: -.2 } } }] as MetricEvent[];
    expect(experimentMetrics(run(), events)[2].value).toBe(-.2);
  });
  it("compares actual geometry across independently saved scene IDs", () => {
    const other = run({ training_steps: 5, parameters: { video_enabled: false, seed: 7 }, project_config_snapshot: { environment_type: "s1_push", expected_training_steps: 20000, scene: { document: pushSceneDocument(PUSH_BASELINE, "scene-b") } } });
    const conditions = experimentConditions(run(), other);
    expect(conditions.find(item => item.key === "scene")?.state).toBe("same");
    expect(conditions.find(item => item.key === "budget")?.state).toBe("same");
    expect(conditions.find(item => item.key === "parameters")?.state).toBe("same");
  });
  it("reports missing frozen evidence separately from differing seeds and geometry", () => {
    const other = run({ code_sha256: "other", parameters: { seed: 11 }, project_config_snapshot: { environment_type: "s1_push", scene: { document: pushSceneDocument({ ...PUSH_BASELINE, goalX: .64 }, "scene-a") } } });
    const conditions = experimentConditions(run(), other);
    expect(conditions.find(item => item.key === "scene")?.state).toBe("different");
    expect(conditions.find(item => item.key === "seed")?.state).toBe("different");
    expect(conditions.find(item => item.key === "budget")?.state).toBe("unknown");
  });
  it("keeps a restored floating position above the persistent launch bar", () => {
    expect(clampAgentPosition({ x: 320, y: 800 }, 390, 844, 190).y).toBe(590);
  });
});
