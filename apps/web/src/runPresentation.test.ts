import { describe, expect, it } from "vitest";
import type { RunRecord } from "./api";
import {
  expectedTrainingSteps,
  formatElapsedDuration,
  parameterFields,
  runElapsedMilliseconds,
  runWaitingMilliseconds,
  runProgressPresentation,
  runStatusPresentation,
} from "./runPresentation";

const record = {
  progress: 0.5,
  training_steps: 16,
  update_steps: 1,
  project_config_snapshot: { expected_training_steps: 32 },
  created_at: "2026-09-01T00:00:00Z",
  started_at: "2026-09-01T00:00:05Z",
  ended_at: null,
} satisfies Pick<
  RunRecord,
  | "progress"
  | "training_steps"
  | "update_steps"
  | "project_config_snapshot"
  | "created_at"
  | "started_at"
  | "ended_at"
>;

describe("canonical run presentation", () => {
  it("uses the eight canonical statuses without inventing another lifecycle state", () => {
    expect(runStatusPresentation("queued").label).toBe("排队中");
    expect(runStatusPresentation("finalizing").label).toBe("收尾中");
    expect(runStatusPresentation("failed").description).toContain("错误");
    expect(runStatusPresentation("running", "en").label).toBe("Training");
  });

  it("reads progress targets only from the frozen project snapshot", () => {
    expect(expectedTrainingSteps(record.project_config_snapshot)).toBe(32);
    expect(expectedTrainingSteps({ expected_training_steps: "32" })).toBeNull();
    expect(runProgressPresentation(record, 32, 2)).toEqual({
      expectedSteps: 32,
      trainingSteps: 32,
      updateIndex: 2,
      progress: 0.5,
      percentage: 50,
    });
  });

  it("uses started_at and freezes elapsed time at ended_at", () => {
    expect(runElapsedMilliseconds(record, Date.parse("2026-09-01T00:01:10Z"))).toBe(65_000);
    expect(formatElapsedDuration(3_725_000)).toBe("1时 2分 5秒");
    expect(formatElapsedDuration(3_725_000, "en")).toBe("1h 2m 5s");
    expect(runElapsedMilliseconds({
      ...record,
      ended_at: "2026-09-01T00:00:35Z",
    }, Date.parse("2026-09-01T01:00:00Z"))).toBe(30_000);
  });

  it("keeps a 63-hour queue wait separate from training time", () => {
    const queued = { ...record, started_at: null, ended_at: null };
    const now = Date.parse(queued.created_at) + 63 * 3_600_000;
    expect(runElapsedMilliseconds(queued, now)).toBeNull();
    expect(runWaitingMilliseconds(queued, now)).toBe(63 * 3_600_000);
    expect(runWaitingMilliseconds({ ...queued, ended_at: "2026-09-01T00:00:30Z" }, now)).toBe(30_000);
    expect(runWaitingMilliseconds(record, now)).toBe(5_000);
  });

  it("derives integer and enum controls from parameter_schema", () => {
    expect(parameterFields({
      type: "object",
      required: ["seed"],
      properties: {
        seed: { type: "integer", minimum: 0, maximum: 10, default: 4 },
        profile: { type: "string", enum: ["short", "standard"], default: "short" },
      },
    }, { seed: 7 })).toEqual([
      {
        name: "seed",
        label: "seed",
        description: null,
        type: "integer",
        required: true,
        value: 7,
        defaultValue: 4,
        minimum: 0,
        maximum: 10,
        step: null,
        options: [],
      },
      expect.objectContaining({
        name: "profile",
        value: "short",
        options: ["short", "standard"],
      }),
    ]);
  });
});