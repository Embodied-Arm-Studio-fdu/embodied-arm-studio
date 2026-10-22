import type { MetricEvent, RunRecord } from "./api";
import type { Locale } from "./i18n";
import { productionEvaluation, runMetricHistory } from "./runMetrics";
import { readPushGeometry } from "./pushScene";
import type { SceneDocument } from "./api";

export const WORKBENCH_ASSISTANT_TOGGLE = "studio-workbench-assistant-toggle";
export const WORKBENCH_ASSISTANT_STATE = "studio-workbench-assistant-state";

export function experimentLabel(run: RunRecord, locale: Locale) {
  const names: Record<string, [string, string]> = { s1_push: ["推物", "Push"], s1_reach: ["平面到达", "Reach"], s1_place: ["搬运放置", "Place"] };
  const task = names[String(run.project_config_snapshot.environment_type)];
  const date = new Date(run.created_at);
  const time = Number.isNaN(date.getTime()) ? (locale === "en" ? "Time not recorded" : "时间未记录") : new Intl.DateTimeFormat(locale === "en" ? "en-GB" : "zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false }).format(date);
  return `${task?.[locale === "en" ? 1 : 0] ?? (locale === "en" ? "Experiment" : "实验")} · ${time}`;
}

const finite = (value: unknown) => typeof value === "number" && Number.isFinite(value) ? value : null;
export function experimentMetrics(run: RunRecord, events: MetricEvent[] = []) {
  const evaluation = productionEvaluation(run.metrics_summary);
  const history = runMetricHistory(events);
  const latest = history[history.length - 1];
  const rate = finite(evaluation?.success_rate);
  const distance = finite(evaluation?.mean_final_distance);
  return [
    { key: "success", label: ["评估成功率", "Evaluation success rate"], value: evaluation?.status !== "unavailable" && rate !== null && rate >= 0 && rate <= 1 ? rate : null, scale: 100, unit: "%", hint: ["来自正式评估；越高通常越好。", "From formal evaluation; higher is usually better."] },
    { key: "distance", label: ["最终目标距离", "Final goal distance"], value: evaluation?.status !== "unavailable" && distance !== null && distance >= 0 ? distance : null, scale: 100, unit: "cm", hint: ["评估回合结束时的平均距离；越低通常越好。", "Mean end-of-episode evaluation distance; lower is usually better."] },
    { key: "reward", label: ["末次平均训练奖励", "Final average training reward"], value: finite(run.metrics_summary.average_reward) ?? latest?.values.average_reward ?? null, scale: 1, unit: "", hint: ["每训练步的平均奖励；奖励定义改变后，数值不能直接判定优劣。", "Mean reward per training step; changed reward definitions prevent a direct quality ranking."] },
  ];
}

export function formatExperimentMetric(row: ReturnType<typeof experimentMetrics>[number], locale: Locale) {
  return row.value === null ? (locale === "en" ? "Not recorded" : "未记录") : `${(row.value * row.scale).toFixed(row.key === "reward" ? 3 : 1)}${row.unit ? ` ${row.unit}` : ""}`;
}

function stable(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stable).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([key, item]) => `${key}:${stable(item)}`).join(",")}}`;
  return JSON.stringify(value) ?? "undefined";
}

/** Compare frozen values, never scene IDs or achieved training steps. */
export function experimentConditions(left: RunRecord, right: RunRecord) {
  const scene = (run: RunRecord): unknown => {
    const frozen = run.project_config_snapshot.scene as { document?: SceneDocument } | undefined;
    if (run.project_config_snapshot.environment_type === "s1_push") return frozen?.document ? readPushGeometry(frozen.document) : null;
    return frozen?.document ?? null;
  };
  const parameters = (run: RunRecord) => run.parameters ? Object.fromEntries(Object.entries(run.parameters).filter(([key]) => key !== "seed" && !key.startsWith("video_") && key !== "camera_id")) : null;
  const budget = (run: RunRecord) => finite(run.project_config_snapshot.expected_training_steps);
  const rows = [
    { key: "task", label: ["任务", "Task"], left: left.project_config_snapshot.environment_type, right: right.project_config_snapshot.environment_type },
    { key: "budget", label: ["训练预算", "Training budget"], left: budget(left), right: budget(right) },
    { key: "seed", label: ["随机种子", "Random seed"], left: left.parameters?.seed, right: right.parameters?.seed },
    { key: "scene", label: ["冻结场景", "Frozen scene"], left: scene(left), right: scene(right) },
    { key: "parameters", label: ["任务参数", "Task parameters"], left: parameters(left), right: parameters(right) },
    { key: "reward", label: ["奖励版本", "Reward version"], left: left.code_sha256 || null, right: right.code_sha256 || null },
  ];
  return rows.map(row => ({ ...row, state: row.left == null || row.right == null ? "unknown" : stable(row.left) === stable(row.right) ? "same" : "different" }));
}
