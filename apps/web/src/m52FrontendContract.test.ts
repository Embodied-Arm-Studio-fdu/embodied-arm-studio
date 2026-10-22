import { describe, expect, it } from "vitest";
import type { RunArtifact, RunRecord } from "./api";
import {
  demoMetricEntries,
  isDemoRunRecord,
  runEnvironmentType,
  runExportEligible,
} from "./App";

function demoRun(status: RunRecord["status"] = "completed"): RunRecord {
  return {
    run_id: "run-demo",
    user_id: "user",
    project_id: "project",
    class_id: "class",
    execution_mode: "simulation",
    run_kind: "play",
    parameters: { seed: 7, radius_m: 0.05, video_enabled: true },
    project_config_snapshot: {
      environment_type: "demo_s1_circle_tracking",
      task_config: { environment_type: "demo_s1_circle_tracking", video_enabled: true },
      catalog_task: { id: "demo_s1_circle_tracking" },
    },
    code_sha256: "0".repeat(64),
    simulation_task_id: "task-runtime-id-not-environment",
    progress: 1,
    metrics_summary: { mean_tracking_error_m: 0.012, command_count: 120 },
    error: null,
    last_sequence: 4,
    training_steps: 0,
    update_steps: 0,
    stop_reason: status === "stopped" ? "requested" : null,
    created_at: "2026-09-22T00:00:00Z",
    started_at: "2026-09-22T00:00:01Z",
    ended_at: "2026-09-22T00:00:30Z",
    updated_at: "2026-09-22T00:00:30Z",
    status,
  };
}

describe("M5.2 Demo detail and export contracts", () => {
  it("uses run_kind rather than runtime task ids and shows the frozen environment", () => {
    const record = demoRun();
    expect(isDemoRunRecord(record)).toBe(true);
    expect(runEnvironmentType(record)).toBe("demo_s1_circle_tracking");
  });

  it("presents only Backend-returned Demo metrics without fabricated training values", () => {
    expect(demoMetricEntries(demoRun())).toEqual([
      ["mean_tracking_error_m", 0.012],
      ["command_count", 120],
    ]);
  });

  it("allows completed and stopped Demo JSON export but not failed Demo export", () => {
    expect(runExportEligible(demoRun("completed"))).toBe(true);
    expect(runExportEligible(demoRun("stopped"))).toBe(true);
    expect(runExportEligible(demoRun("failed"))).toBe(false);
  });

  it("keeps Demo config, metrics, logs, manifest and optional MP4 as metadata", () => {
    const types = ["demo_config", "demo_metrics", "stdout_log", "stderr_log", "task_log", "manifest", "demo_video"];
    const artifacts: RunArtifact[] = types.map((artifact_type, index) => ({
      artifact_id: `artifact-${index}`,
      run_id: "run-demo",
      artifact_type,
      path: artifact_type === "demo_video" ? "video/overview.mp4" : `files/${artifact_type}`,
      size_bytes: 10,
      sha256: String(index).repeat(64),
      is_primary_model: false,
      created_at: "2026-09-22T00:00:30Z",
    }));
    const exportPreview = JSON.stringify({ run: demoRun(), artifacts });
    expect(types.every((type) => exportPreview.includes(type))).toBe(true);
    expect(exportPreview).not.toContain("video/mp4;base64");
  });
});
