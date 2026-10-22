import { api, type RunRecord } from "./api";
import { createBrowserId } from "./browserId";

export interface CanonicalRunPayload {
  project_id: string;
  class_id: string;
  project_config_version: number;
  run_kind: "train" | "play";
  environment_type: string;
  parameters: Record<string, unknown>;
  code: string;
  scene_ref?: { scene_id: string; revision: number };
  reward_extension?: { schema: string; source: string; sha256: string };
}

type CanonicalRunCreator = (
  payload: CanonicalRunPayload,
) => Promise<RunRecord>;

export function createCanonicalRun(
  payload: CanonicalRunPayload,
  idempotencyKey = createBrowserId(),
): Promise<RunRecord> {
  if (payload.run_kind === "play" && payload.scene_ref) {
    throw new Error("Demo Run payload must not include scene_ref");
  }
  const sceneRef = payload.environment_type === "s1_push"
    ? payload.scene_ref
    : undefined;
  return api<RunRecord>("/api/v1/runs", {
    method: "POST",
    headers: { "Idempotency-Key": idempotencyKey },
    body: JSON.stringify({
      project_id: payload.project_id,
      class_id: payload.class_id,
      project_config_version: payload.project_config_version,
      execution_mode: "simulation",
      run_kind: payload.run_kind,
      environment_type: payload.environment_type,
      parameters: payload.parameters,
      ...(sceneRef ? { scene_ref: sceneRef } : {}),
      ...(payload.reward_extension ? { reward_extension: payload.reward_extension } : {}),
      code: payload.code,
    }),
  });
}

export function createCanonicalTrainingRun(
  payload: Omit<CanonicalRunPayload, "run_kind">,
  idempotencyKey?: string,
): Promise<RunRecord> {
  return createCanonicalRun({ ...payload, run_kind: "train" }, idempotencyKey);
}

export function createCanonicalRunSubmission(
  createRun: CanonicalRunCreator = createCanonicalRun,
) {
  let pending: Promise<RunRecord> | null = null;

  return {
    submit(payload: CanonicalRunPayload) {
      if (pending) return pending;

      const request = createRun(payload);
      pending = request;
      void request
        .finally(() => {
          if (pending === request) pending = null;
        })
        .catch(() => undefined);
      return request;
    },
    isPending() {
      return pending !== null;
    },
  };
}
