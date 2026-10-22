import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { RunRecord } from "./api";
import { RunEvaluationCard } from "./App";

function run(
  metrics_summary: RunRecord["metrics_summary"],
  overrides: Partial<RunRecord> = {},
): RunRecord {
  return {
    run_id: "run-evaluation-test",
    user_id: "user-test",
    project_id: "project-test",
    class_id: null,
    execution_mode: "simulation",
    run_kind: "train",
    project_config_snapshot: {},
    code_sha256: "0".repeat(64),
    simulation_task_id: "task-test",
    progress: 1,
    metrics_summary,
    error: null,
    last_sequence: 1,
    training_steps: 20000,
    update_steps: 100,
    stop_reason: null,
    created_at: "2026-09-14T00:00:00Z",
    started_at: "2026-09-14T00:00:01Z",
    ended_at: "2026-09-14T00:10:00Z",
    updated_at: "2026-09-14T00:10:00Z",
    status: "completed",
    ...overrides,
  };
}

describe("Run detail Production evaluation card", () => {
  it("shows completed and evaluation unavailable separately", () => {
    const markup = renderToStaticMarkup(
      <RunEvaluationCard record={run({ success_rate: 1 })} />,
    );
    expect(markup).toContain("<small>Run</small><strong>已完成</strong>");
    expect(markup).toContain("<small>策略</small><strong>不可用</strong>");
    expect(markup).toContain("该运行未生成正式评估");
    expect(markup).toContain("不会显示为 0%");
    expect(markup).not.toContain("<strong>failed</strong>");
  });

  it("renders persisted deterministic metrics and preserves explicit zero", () => {
    const markup = renderToStaticMarkup(
      <RunEvaluationCard record={run({ evaluation: {
        status: "passed",
        deterministic: true,
        policy_passed: true,
        episode_count: 100,
        success_count: 80,
        success_rate: 0.8,
        wilson_95_lower_bound: 0.7111708344068411,
        wilson_95_upper_bound: 0.8666330666689676,
        mean_episode_length: 240,
        mean_episode_return: 1.25,
        mean_final_distance: 0.2,
        mean_minimum_distance: 0.1,
        mean_object_displacement: 0,
        timeout_rate: 0.2,
        failure_rate: 0,
      } })} />,
    );
    expect(markup).toContain("80 / 100");
    expect(markup).toContain("71.12% – 86.66%");
    expect(markup).toContain("0 m");
    expect(markup).toContain("<small>策略</small><strong>通过</strong>");
    expect(markup).toContain("仅使用 metrics_summary.evaluation");
  });

  it("renders evaluation status labels in English when requested", () => {
    const markup = renderToStaticMarkup(
      <RunEvaluationCard record={run({ success_rate: 1 })} locale="en" />,
    );
    expect(markup).toContain("PRODUCTION EVALUATION");
    expect(markup).toContain("<small>Run</small><strong>Completed</strong>");
    expect(markup).toContain("<small>Policy</small><strong>Unavailable</strong>");
    expect(markup).not.toMatch(/[\u3400-\u9fff]/u);
  });

  it.each(["completed", "stopped"] as const)("treats a %s Demo with null model/evaluation and zero counters as a normal result", (status) => {
    const markup = renderToStaticMarkup(
      <RunEvaluationCard record={run(
        { final_distance_m: 0.012, commands: 120 },
        { run_kind: "play", status, training_steps: 0, update_steps: 0 },
      )} />,
    );
    expect(markup).toContain("Demo 结果");
    expect(markup).toContain(`<small>Run</small><strong>${status === "completed" ? "已完成" : "已停止"}</strong>`);
    expect(markup).toContain("不代表失败");
    expect(markup).not.toContain("Policy</small><strong>failed");
  });
});
