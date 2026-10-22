import type { RunRecord, RunStatus } from "./api";
import type { Locale } from "./i18n";

export interface RunStatusPresentation {
  status: RunStatus;
  label: string;
  description: string;
}

export const RUN_STATUS_PRESENTATIONS: RunStatusPresentation[] = [
  { status: "queued", label: "排队中", description: "等待仿真训练资源" },
  { status: "initializing", label: "初始化", description: "正在准备仿真环境" },
  { status: "running", label: "训练中", description: "PPO 正在更新策略" },
  { status: "finalizing", label: "收尾中", description: "正在保存指标和制品" },
  { status: "stopping", label: "停止中", description: "正在处理停止请求" },
  { status: "completed", label: "已完成", description: "训练完成，结果已保存" },
  { status: "stopped", label: "已停止", description: "训练已停止，已有结果已保存" },
  { status: "failed", label: "失败", description: "训练失败，请查看错误信息" },
];

const EN_RUN_STATUS_PRESENTATIONS: RunStatusPresentation[] = [
  { status: "queued", label: "Queued", description: "Waiting for simulation training resources" },
  { status: "initializing", label: "Initializing", description: "Preparing the simulation environment" },
  { status: "running", label: "Training", description: "PPO is updating the policy" },
  { status: "finalizing", label: "Finalizing", description: "Saving metrics and artifacts" },
  { status: "stopping", label: "Stopping", description: "Processing the stop request" },
  { status: "completed", label: "Completed", description: "Training completed and results were saved" },
  { status: "stopped", label: "Stopped", description: "Training stopped and available results were saved" },
  { status: "failed", label: "Failed", description: "Training failed; review the error details" },
];

export interface ParameterField {
  name: string;
  label: string;
  description: string | null;
  type: "integer" | "number" | "string" | "boolean";
  required: boolean;
  value: unknown;
  defaultValue: unknown;
  minimum: number | null;
  maximum: number | null;
  step: number | null;
  options: Array<string | number | boolean>;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function timestamp(value: string | null | undefined): number | null {
  if (!value) return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function runStatusPresentation(
  status: RunStatus,
  locale: Locale = "zh-CN",
): RunStatusPresentation {
  const presentations = locale === "en"
    ? EN_RUN_STATUS_PRESENTATIONS
    : RUN_STATUS_PRESENTATIONS;
  return presentations.find((item) => item.status === status) ?? presentations[0];
}

export function runStatusLabel(status: RunStatus, locale: Locale = "zh-CN"): string {
  return runStatusPresentation(status, locale).label;
}

export function expectedTrainingSteps(snapshot: Record<string, unknown>): number | null {
  const value = finiteNumber(snapshot.expected_training_steps);
  return value !== null && value > 0 ? Math.floor(value) : null;
}

export function runProgressPresentation(
  record: Pick<RunRecord, "progress" | "training_steps" | "update_steps" | "project_config_snapshot">,
  latestStep?: number,
  latestUpdateIndex?: number,
) {
  const expectedSteps = expectedTrainingSteps(record.project_config_snapshot);
  const trainingSteps = Math.max(record.training_steps, finiteNumber(latestStep) ?? 0);
  const updateIndex = Math.max(record.update_steps, finiteNumber(latestUpdateIndex) ?? 0);
  const progress = Math.min(1, Math.max(0, finiteNumber(record.progress) ?? 0));
  return {
    expectedSteps,
    trainingSteps,
    updateIndex,
    progress,
    percentage: Math.round(progress * 100),
  };
}

export function runElapsedMilliseconds(
  record: Pick<RunRecord, "created_at" | "started_at" | "ended_at">,
  now = Date.now(),
): number | null {
  const start = timestamp(record.started_at);
  if (start === null) return null;
  const end = timestamp(record.ended_at) ?? now;
  return Math.max(0, end - start);
}

export function runWaitingMilliseconds(
  record: Pick<RunRecord, "created_at" | "started_at" | "ended_at">,
  now = Date.now(),
): number | null {
  const created = timestamp(record.created_at);
  if (created === null) return null;
  return Math.max(0, (timestamp(record.started_at) ?? timestamp(record.ended_at) ?? now) - created);
}

export function formatElapsedDuration(
  milliseconds: number | null,
  locale: Locale = "zh-CN",
): string {
  if (milliseconds === null || !Number.isFinite(milliseconds)) return "--";
  const totalSeconds = Math.floor(Math.max(0, milliseconds) / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (locale === "en") {
    if (hours > 0) return `${hours}h ${minutes}m ${seconds}s`;
    if (minutes > 0) return `${minutes}m ${seconds}s`;
    return `${seconds}s`;
  }
  if (hours > 0) return `${hours}时 ${minutes}分 ${seconds}秒`;
  if (minutes > 0) return `${minutes}分 ${seconds}秒`;
  return `${seconds}秒`;
}

export function parameterFields(
  schema: unknown,
  parameters: Record<string, unknown> = {},
): ParameterField[] {
  if (!isRecord(schema) || !isRecord(schema.properties)) return [];
  const required = new Set(
    Array.isArray(schema.required)
      ? schema.required.filter((item): item is string => typeof item === "string")
      : [],
  );
  return Object.entries(schema.properties).flatMap(([name, rawDefinition]) => {
    if (!isRecord(rawDefinition)) return [];
    const type = rawDefinition.type;
    if (type !== "integer" && type !== "number" && type !== "string" && type !== "boolean") {
      return [];
    }
    const options = Array.isArray(rawDefinition.enum)
      ? rawDefinition.enum.filter(
          (value): value is string | number | boolean =>
            typeof value === "string" || typeof value === "number" || typeof value === "boolean",
        )
      : [];
    return [{
      name,
      label: typeof rawDefinition.title === "string" ? rawDefinition.title : name,
      description: typeof rawDefinition.description === "string" ? rawDefinition.description : null,
      type,
      required: required.has(name),
      value: parameters[name] ?? rawDefinition.default,
      defaultValue: rawDefinition.default,
      minimum: finiteNumber(rawDefinition.minimum),
      maximum: finiteNumber(rawDefinition.maximum),
      step: finiteNumber(rawDefinition.multipleOf),
      options,
    }];
  });
}