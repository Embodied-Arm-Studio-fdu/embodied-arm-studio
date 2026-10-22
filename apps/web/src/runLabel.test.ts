import { describe, expect, it } from "vitest";
import type { RunRecord } from "./api";
import { runDisplayName } from "./runLabel";

const record = (overrides: Partial<RunRecord>): RunRecord => ({
  run_id: "00000000-0000-0000-0000-000000000001",
  user_id: "user-1",
  project_id: "project-1",
  class_id: "class-1",
  execution_mode: "simulation",
  run_kind: "train",
  parameters: {},
  project_config_snapshot: {},
  code_sha256: "sha256:test",
  simulation_task_id: null,
  progress: 0,
  metrics_summary: {},
  error: null,
  last_sequence: 0,
  training_steps: 0,
  update_steps: 0,
  stop_reason: null,
  created_at: "2026-10-01T00:00:00Z",
  started_at: null,
  ended_at: null,
  updated_at: "2026-10-01T00:00:00Z",
  status: "queued",
  ...overrides,
});

describe("runDisplayName", () => {
  it("shows the training profile instead of the raw UUID", () => {
    expect(runDisplayName(record({ project_config_snapshot: { environment_type: "s1_push", training_profile: "standard" } }))).toBe("Standard Push #2");
  });

  it("shows Reach and Demo task names with a stable numeric suffix", () => {
    expect(runDisplayName(record({ project_config_snapshot: { environment_type: "s1_reach" } }))).toBe("Reach #2");
    expect(runDisplayName(record({ run_kind: "play", project_config_snapshot: { environment_type: "demo_s1_circle_tracking" } }))).toBe("圆轨迹 Demo #2");
    expect(runDisplayName(record({ run_kind: "play", project_config_snapshot: { environment_type: "demo_s1_scripted_pick_place" } }))).toBe("规则抓取放下 Demo #2");
  });
});
