import { afterEach, describe, expect, it, vi } from "vitest";
import { setAccessToken, type RunRecord } from "./api";
import { BROWSER_ID_COMPATIBILITY_ERROR } from "./browserId";
import { createCanonicalRun, createCanonicalRunSubmission } from "./canonicalRunRequest";

const base = {
  project_id: "project-1",
  class_id: "class-1",
  project_config_version: 3,
  code: "",
};

afterEach(() => {
  setAccessToken(null);
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("catalog-driven canonical Run creation", () => {
  it("coalesces concurrent submissions into exactly one POST", async () => {
    let resolveRequest!: (run: RunRecord) => void;
    const request = new Promise<RunRecord>((resolve) => { resolveRequest = resolve; });
    const createRun = vi.fn(() => request);
    const submission = createCanonicalRunSubmission(createRun);
    const payload = { ...base, run_kind: "train" as const, environment_type: "s1_push", parameters: { seed: 42 } };
    const first = submission.submit(payload);
    const second = submission.submit(payload);
    expect(second).toBe(first);
    expect(createRun).toHaveBeenCalledOnce();
    expect(submission.isPending()).toBe(true);
    resolveRequest({ run_id: "run-1" } as RunRecord);
    await first;
    await Promise.resolve();
    expect(submission.isPending()).toBe(false);
  });

  it("serializes both Training and all six Demo task payloads", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({ run_id: "run" }) });
    vi.stubGlobal("fetch", fetchMock);
    const cases = [
      ["s1_push", "train"],
      ["s1_reach", "train"],
      ["demo_s1_target_pose", "play"],
      ["demo_s1_line_tracking", "play"],
      ["demo_s1_circle_tracking", "play"],
      ["demo_s1_joint_position", "play"],
      ["demo_s1_joint_sweep", "play"],
      ["demo_s1_scripted_pick_place", "play"],
    ] as const;
    for (const [environment_type, run_kind] of cases) {
      await createCanonicalRun({ ...base, environment_type, run_kind, parameters: { seed: 7, video_enabled: true } });
    }
    expect(fetchMock).toHaveBeenCalledTimes(8);
    fetchMock.mock.calls.forEach(([, options], index) => {
      const body = JSON.parse(options?.body as string);
      expect(body).toMatchObject({
        execution_mode: "simulation",
        environment_type: cases[index][0],
        run_kind: cases[index][1],
        parameters: { seed: 7, video_enabled: true },
      });
      for (const forbidden of ["task_type", "run_mode", "device", "algorithm", "checkpoint", "object_size", "object_start", "goal_position"]) {
        expect(body).not.toHaveProperty(forbidden);
        expect(body.parameters).not.toHaveProperty(forbidden);
      }
    });
  });

  it("keeps scene_ref only for s1_push Training", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({ run_id: "run-scene" }) });
    vi.stubGlobal("fetch", fetchMock);
    await createCanonicalRun({
      ...base,
      run_kind: "train",
      environment_type: "s1_push",
      parameters: { seed: 42 },
      scene_ref: { scene_id: "scene-s1", revision: 1 },
    });
    const body = JSON.parse(fetchMock.mock.calls[0][1]?.body as string);
    expect(body.scene_ref).toEqual({ scene_id: "scene-s1", revision: 1 });
    expect(body.parameters.scene_ref).toBeUndefined();
    await createCanonicalRun({
      ...base,
      run_kind: "train",
      environment_type: "s1_reach",
      parameters: { seed: 42 },
      scene_ref: { scene_id: "scene-s1", revision: 1 },
    });
    const reachBody = JSON.parse(fetchMock.mock.calls[1][1]?.body as string);
    expect(reachBody.scene_ref).toBeUndefined();
    expect(() => createCanonicalRun({
      ...base,
      run_kind: "play",
      environment_type: "demo_s1_target_pose",
      parameters: {},
      scene_ref: { scene_id: "scene-s1", revision: 1 },
    })).toThrow("must not include scene_ref");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("sends one stable bounded Idempotency-Key and the authenticated API route", async () => {
    const key = "123e4567-e89b-42d3-a456-426614174000";
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({ run_id: "run-1" }) });
    vi.stubGlobal("crypto", { randomUUID: vi.fn(() => key) });
    vi.stubGlobal("fetch", fetchMock);
    setAccessToken("test-token");
    await createCanonicalRun({ ...base, run_kind: "train", environment_type: "s1_push", parameters: { seed: 42 } });
    const [path, options] = fetchMock.mock.calls[0];
    expect(path).toBe("/api/v1/runs");
    expect(options?.headers).toMatchObject({ "Idempotency-Key": key, Authorization: "Bearer test-token" });
  });

  it("does not POST when secure ID generation is unavailable", () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("crypto", {});
    vi.stubGlobal("fetch", fetchMock);
    expect(() => createCanonicalRun({ ...base, run_kind: "train", environment_type: "s1_push", parameters: { seed: 42 } })).toThrow(BROWSER_ID_COMPATIBILITY_ERROR);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("fails closed on a Backend error without retrying or calling Simulation create", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: () => Promise.resolve({ detail: "Backend unavailable" }),
    });
    vi.stubGlobal("crypto", { randomUUID: vi.fn(() => "123e4567-e89b-42d3-a456-426614174000") });
    vi.stubGlobal("fetch", fetchMock);

    await expect(createCanonicalRun({
      ...base,
      run_kind: "train",
      environment_type: "s1_push",
      parameters: { seed: 7, video_enabled: false },
    })).rejects.toThrow();

    expect(fetchMock).toHaveBeenCalledOnce();
    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/runs");
    expect(fetchMock.mock.calls.some(([path]) => String(path).includes("/simulation/tasks"))).toBe(false);
  });
});
