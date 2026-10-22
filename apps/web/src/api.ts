import { createBrowserId } from "./browserId";

export type RunStatus =
  | "queued"
  | "initializing"
  | "running"
  | "finalizing"
  | "stopping"
  | "completed"
  | "stopped"
  | "failed";

export interface RunEvaluation {
  status: "passed" | "failed" | "unavailable";
  deterministic: true;
  policy_passed: boolean;
  episode_count?: number;
  success_count?: number;
  success_rate?: number;
  wilson_95_lower_bound?: number;
  wilson_95_upper_bound?: number;
  mean_episode_length?: number;
  mean_episode_return?: number;
  mean_final_distance?: number;
  mean_minimum_distance?: number;
  mean_object_displacement?: number;
  timeout_rate?: number;
  failure_rate?: number;
}

export interface RunMetricsSummary extends Record<string, unknown> {
  evaluation?: RunEvaluation;
}

export interface CourseStep {
  id: string;
  title: string;
  objective: string;
  theory: string;
  prediction: string;
  success: string;
  allowed_files: string[];
}

export interface Course {
  id: string;
  title: string;
  summary: string;
  estimated_minutes: number;
  steps: CourseStep[];
}

export interface TeachingCourse {
  course_id: string;
  organization_id: string;
  code: string;
  title: string;
  description: string;
  owner_instructor_user_id: string;
  enabled: boolean;
}

export interface TeachingClass {
  class_id: string;
  course_id: string;
  code: string;
  name: string;
  term: string;
  instructor_user_id: string;
  enabled: boolean;
}

export interface TeachingProject {
  project_id: string;
  course_id: string;
  title: string;
  description: string;
  instructions: string;
  starter_code: string;
  task_type: string;
  environment_type: string;
  algorithm: string;
  success_condition: Record<string, unknown>;
  training_profile: string;
  expected_training_steps: number;
  video_enabled: boolean;
  parameter_schema: Record<string, unknown>;
  evaluation_rubric: Record<string, unknown>;
  config_version: number;
  enabled: boolean;
  permissions: { can_view: boolean; can_use: boolean; can_modify: boolean };
}

export interface WorkspaceNode {
  name: string;
  path: string;
  type: "file" | "directory";
  size?: number;
  read_only?: boolean;
  children?: WorkspaceNode[];
}

export interface WorkspaceProject {
  project_id: string;
  file_count: number;
  workspace_status: string;
}

export interface RunRecord {
  run_id: string;
  user_id: string;
  project_id: string;
  class_id: string | null;
  execution_mode: "simulation";
  run_kind: "train" | "play";
  parameters?: Record<string, unknown>;
  project_config_snapshot: Record<string, unknown>;
  code_sha256: string;
  simulation_task_id: string | null;
  progress: number;
  metrics_summary: RunMetricsSummary;
  error: { code: string; message: string; retryable: boolean } | null;
  last_sequence: number;
  training_steps: number;
  update_steps: number;
  stop_reason: string | null;
  created_at: string;
  started_at: string | null;
  ended_at: string | null;
  updated_at: string;
  status: RunStatus;
}

export interface RunArtifact {
  artifact_id: string;
  run_id: string;
  artifact_type: string;
  path: string;
  size_bytes: number | null;
  sha256: string | null;
  is_primary_model: boolean;
  created_at: string;
}

export interface RunComparisonResponse {
  schema_version: "1.0";
  run_ids: string[];
  cross_project: boolean;
  project_ids: string[];
  runs: Array<{
    run_id: string;
    project_id: string;
    status: RunStatus;
    parameters: Record<string, unknown>;
    project_config_snapshot_summary: {
      config_version: string | number | null;
      sha256: string;
      keys: string[];
    };
    metrics_summary: Record<string, unknown>;
    training_steps: number;
    update_steps: number;
    artifacts: {
      total_count: number;
      counts_by_type: Record<string, number>;
      missing_size_count: number;
      missing_sha256_count: number;
      primary_model_count: number;
    };
    evaluation: {
      status: string;
      score: number | null;
      grade: string | null;
      rubric_version: string;
    } | null;
  }>;
  differences: Record<string, unknown>;
  limitations: string[];
}

export interface AuthenticatedBlob {
  blob: Blob;
  filename: string | null;
}

export type StreamStatus =
  | "CREATED"
  | "STARTING"
  | "READY"
  | "STOPPING"
  | "STOPPED"
  | "ERROR";

export interface StreamDescriptor {
  stream_id: string;
  run_id: string;
  task_id: string;
  camera_id: string;
  status: StreamStatus;
  webrtc_url: string;
  whep_url: string;
  width: number;
  height: number;
  fps: number;
}

export interface LegacyRunRecord {
  run_id: string;
  project_id?: string;
  environment_id: string;
  strategy_id: string;
  seed: number;
  max_steps: number;
  config: Record<string, number>;
  status: RunStatus;
  started_at?: string;
  ended_at?: string;
  error?: string | null;
}

export interface AuthUser {
  user_id: string;
  username: string;
  display_name: string;
  enabled: boolean;
}

export interface OrganizationContext {
  organization_id: string;
  slug: string;
  name: string;
  enabled: boolean;
  role: string;
  permissions: string[];
}

export interface CurrentUserContext extends AuthUser {
  organizations: OrganizationContext[];
}

export interface StrategyInfo {
  id: string;
  name: string;
  family: string;
  description: string;
  compatible_environments: string[];
  requires_gpu: boolean;
}

export interface SandboxTemplateInfo {
  template_id: string;
  requires_gpu: boolean;
  timeout_seconds: number;
  cpu_limit: number;
  memory_mb: number;
}

export interface SandboxJobInfo {
  job_id: string;
  template_id: string;
  status: RunStatus;
  created_at: string;
  started_at?: string | null;
  ended_at?: string | null;
  exit_code?: number | null;
  error?: string | null;
}

export interface SandboxArtifactInfo {
  name: string;
  media_type: string;
  size: number;
  sha256: string;
}

export interface MetricEvent {
  event_id: string;
  sequence?: number;
  run_id?: string;
  occurred_at?: string;
  schema_version?: string;
  type: "metric" | "state" | "status" | "error" | "heartbeat";
  payload: {
    step?: number;
    metrics?: Record<string, number>;
    arm?: { joint_positions: number[]; target: number[] };
    trajectory?: {
      end_effector_xyz?: number[];
      object_xyz?: number[];
      goal_xyz?: number[];
      grasped?: boolean;
    };
    collision?: {
      model?: string;
      detected?: boolean;
      minimum_margin?: number;
    };
    joint_limits?: {
      violated?: boolean;
      minimum_margin?: number;
    };
    status?: RunStatus;
  };
}

export interface PipelinePort {
  name: string;
  type: string;
  required?: boolean;
}

export interface PipelineNode {
  node_id: string;
  kind: string;
  implementation: string;
  capability: "mock" | "external" | "planned" | "unavailable";
  enabled: boolean;
  config: Record<string, unknown>;
  inputs: PipelinePort[];
  outputs: PipelinePort[];
  metadata: Record<string, unknown>;
}

export interface PipelineEdge {
  edge_id: string;
  source: { node_id: string; port: string };
  target: { node_id: string; port: string };
  type: string;
  feedback?: boolean;
}

export interface PipelineDocument {
  schema_version: string;
  pipeline_id: string;
  revision: number;
  name: string;
  capability: "mock" | "external" | "planned" | "unavailable";
  environment_ref: { id: string; revision: number };
  nodes: PipelineNode[];
  edges: PipelineEdge[];
  reproducibility: { seed_policy: string; deterministic: boolean };
  metadata: Record<string, unknown>;
}

export interface PipelineValidation {
  valid: boolean;
  errors: Array<{ path: string; message: string }>;
  warnings: Array<{ path: string; message: string }>;
}

export interface EnvironmentField {
  type: string;
  shape?: number[];
  unit?: string;
  range?: number[];
}

export interface EnvironmentInfo {
  id: string;
  name: string;
  description: string;
  simulator: string;
  robot: string;
  capabilities: string[];
  available: boolean;
  revision: number;
  capability: "mock" | "external" | "planned" | "unavailable";
  observation_space: Record<string, EnvironmentField>;
  action_space: Record<string, EnvironmentField>;
  reward_terms: Array<{ id: string; description: string }>;
  terminations: Array<{ id: string; description: string }>;
  evidence_channels: string[];
}

export interface EnvironmentSchema {
  type: "object";
  properties: Record<string, {
    type: string;
    minimum: number;
    maximum: number;
    default: number;
    title: string;
  }>;
}

export interface SceneObject {
  object_id: string;
  type: "plane" | "box" | "cylinder" | "sphere" | "target" | "wall";
  label: string;
  position: number[];
  rotation: number[];
  scale: number[];
  color: string;
  static: boolean;
}

export interface SceneDocument {
  schema_version: string;
  scene_id: string;
  revision: number;
  name: string;
  capability: string;
  robot: {
    id: string;
    dof: number;
    gripper_dof?: number;
    end_effector?: string;
    control_mode: string;
    asset?: string;
  };
  objects: SceneObject[];
  observation: string[];
  action: { type: string; shape: number[]; range: number[] };
  reward: Array<{ id: string; weight: number }>;
  termination: string[];
  safety: { real_robot_control_enabled: boolean; sim2real: string };
  evidence?: Record<string, string>;
}

export interface AgentProviderInfo {
  id: string;
  name: string;
  status: "healthy" | "degraded" | "unavailable" | "unknown";
  configured: boolean;
  models: string[];
  capabilities: { chat: boolean; structured_output: boolean; tool_calls: boolean };
  data_scope: string;
  connection?: { status: string; message?: string; checked_at?: string };
}

export interface AssistanceLevelInfo {
  level: number;
  id: string;
  capabilities: string[];
  requires_teacher_authorization: boolean;
}

export interface AgentMessageResponse {
  content: string;
  tools_used: string[];
}

export interface EvaluationCriterion {
  criterion_id: string;
  name: string;
  weight: number;
  mandatory: boolean;
  status: "assessed" | "not_assessed";
  score: number | null;
  evidence_refs: string[];
  finding: string;
  missing_evidence: string;
}

export interface EvaluationReport {
  report_id: string;
  rubric_version: string;
  run_ids: string[];
  capability: "mock";
  criteria: EvaluationCriterion[];
  evidence_completeness: number;
  overall: { status: "assessed" | "not_assessed"; score: number | null };
  missing_evidence: string[];
  limitations: string[];
  sim2real: { status: "blocked"; real_robot_control_enabled: false };
}

export interface CanonicalEvaluationReport {
  evaluation_id: string;
  run_id: string;
  requested_by_user_id: string;
  status: "assessed" | "insufficient_evidence";
  score: number | null;
  grade: string;
  rubric_version: string;
  rubric_snapshot: Record<string, unknown>;
  evidence: Record<string, unknown>;
  evidence_sha256: string;
  summary: Record<string, unknown>;
  limitations: string[];
  evaluator_version: string;
  created_at: string;
}

export interface EvaluationComparison {
  before_report_id: string;
  after_report_id: string;
  rubric_version: string;
  overall: {
    before: EvaluationReport["overall"];
    after: EvaluationReport["overall"];
    delta: number | null;
  };
  completeness_delta: number;
  criteria: Array<{
    criterion_id: string;
    name: string;
    before: { status: EvaluationCriterion["status"]; score: number | null };
    after: { status: EvaluationCriterion["status"]; score: number | null };
    delta: number | null;
    change: "newly_assessed" | "lost_evidence" | "improved" | "regressed" | "unchanged";
    evidence_refs: string[];
  }>;
}

export class ApiError extends Error {
  status: number;
  requestId?: string;

  constructor(message: string, status: number, requestId?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.requestId = requestId;
  }
}

const ACCESS_TOKEN_STORAGE_KEY = "embodied-arm-access-token";

function storedAccessToken() {
  if (typeof window === "undefined") return null;
  try {
    return window.sessionStorage.getItem(ACCESS_TOKEN_STORAGE_KEY);
  } catch {
    return null;
  }
}

let accessToken: string | null = storedAccessToken();

export function setAccessToken(token: string | null) {
  accessToken = token;
  if (typeof window === "undefined") return;
  try {
    if (token) window.sessionStorage.setItem(ACCESS_TOKEN_STORAGE_KEY, token);
    else window.sessionStorage.removeItem(ACCESS_TOKEN_STORAGE_KEY);
  } catch {
    // Keep the in-memory token when browser storage is unavailable.
  }
}

export function getAccessToken() {
  return accessToken;
}

function apiErrorMessage(body: unknown, fallback: string) {
  if (!body || typeof body !== "object") return fallback || "请求失败";
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === "string" && detail.trim()) return detail;
  if (detail && typeof detail === "object" && !Array.isArray(detail)) {
    const structured = detail as { code?: unknown; message?: unknown };
    if (typeof structured.message === "string" && structured.message.trim()) {
      return structured.message;
    }
    if (typeof structured.code === "string" && structured.code.trim()) {
      return structured.code;
    }
  }
  return fallback || "请求失败";
}

function responseRequestId(body: unknown) {
  if (!body || typeof body !== "object" || Array.isArray(body)) return undefined;
  const requestId = (body as { request_id?: unknown }).request_id;
  return typeof requestId === "string" ? requestId : undefined;
}

const API_REQUEST_TIMEOUT_MS = 35_000;

export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const token = accessToken;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), API_REQUEST_TIMEOUT_MS);
  const abortFromCaller = () => controller.abort();
  options?.signal?.addEventListener("abort", abortFromCaller, { once: true });
  try {
    const response = await fetch(path, {
      ...options,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...options?.headers,
      },
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({ detail: response.statusText }));
      const requestId = response.headers.get("x-request-id") ?? responseRequestId(body);
      const suffix = requestId ? `（request_id: ${requestId}）` : "";
      if (response.status === 401) {
        window.dispatchEvent(new Event("studio-auth-expired"));
      }
      throw new ApiError(`${apiErrorMessage(body, response.statusText)}${suffix}`, response.status, requestId);
    }
    return await response.json();
  } catch (caught) {
    if (caught instanceof DOMException && caught.name === "AbortError") {
      throw new Error("请求超时，请检查 Agent Provider 的网络连接后重试");
    }
    throw caught;
  } finally {
    clearTimeout(timeout);
    options?.signal?.removeEventListener("abort", abortFromCaller);
  }
}

export async function apiBlob(path: string, options?: RequestInit): Promise<Response> {
  const token = accessToken;
  const response = await fetch(path, {
    ...options,
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options?.headers,
    },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    const requestId = response.headers.get("x-request-id") ?? responseRequestId(body);
    if (response.status === 401) window.dispatchEvent(new Event("studio-auth-expired"));
    throw new ApiError(`${apiErrorMessage(body, response.statusText)}${requestId ? `（request_id: ${requestId}）` : ""}`, response.status, requestId);
  }
  return response;
}

export async function apiBlobData(
  path: string,
  options?: RequestInit,
): Promise<AuthenticatedBlob> {
  const response = await apiBlob(path, options);
  const disposition = response.headers.get("content-disposition") ?? "";
  const reader = response.body?.getReader();
  let blob: Blob;
  if (reader) {
    // Some browsers fail Response.blob() while the byte stream remains readable.
    const parts: ArrayBuffer[] = [];
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const part = new Uint8Array(value.byteLength);
        part.set(value);
        parts.push(part.buffer);
      }
    } finally {
      reader.releaseLock();
    }
    blob = new Blob(parts, {
      type: response.headers.get("content-type") ?? "",
    });
  } else {
    blob = await response.blob();
  }
  return {
    blob,
    filename: disposition.match(/filename="?([^";]+)"?/)?.[1] ?? null,
  };
}

export async function apiWithFallback<T>(
  primaryPath: string,
  fallbackPath: string,
  options?: RequestInit,
  fallbackOptions?: RequestInit,
): Promise<T> {
  try {
    return await api<T>(primaryPath, options);
  } catch (primaryError) {
    if (!(primaryError instanceof ApiError) || ![404, 405].includes(primaryError.status)) {
      throw primaryError;
    }
    try {
      return await api<T>(fallbackPath, fallbackOptions ?? options);
    } catch {
      throw primaryError;
    }
  }
}

export function idempotencyKey(): string {
  return createBrowserId();
}
