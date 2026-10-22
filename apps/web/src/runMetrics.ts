import type { MetricEvent, RunEvaluation, RunMetricsSummary, RunRecord } from "./api";
import type { Locale } from "./i18n";

export const RUN_METRIC_DEFINITIONS = [
  {
    key: "average_reward",
    label: "平均奖励",
    labelEn: "Average reward",
    description: "本次 rollout 中每个训练步的平均奖励",
    descriptionEn: "Average reward per training step in this rollout",
    unit: "奖励/步",
    unitEn: "reward/step",
  },
  {
    key: "episode_length",
    label: "回合长度",
    labelEn: "Episode length",
    description: "本次 rollout 已完成回合的平均动作步数",
    descriptionEn: "Average action steps in completed episodes in this rollout",
    unit: "动作步",
    unitEn: "action steps",
  },
  {
    key: "policy_loss",
    label: "策略损失",
    labelEn: "Policy loss",
    description: "PPO 策略目标在本次更新中的平均损失",
    descriptionEn: "Average PPO policy objective loss in this update",
    unit: "无量纲",
    unitEn: "dimensionless",
  },
  {
    key: "value_loss",
    label: "价值损失",
    labelEn: "Value loss",
    description: "价值函数在本次更新中的平均损失",
    descriptionEn: "Average value-function loss in this update",
    unit: "无量纲",
    unitEn: "dimensionless",
  },
  {
    key: "entropy",
    label: "策略熵",
    labelEn: "Policy entropy",
    description: "策略动作分布在本次更新中的平均熵",
    descriptionEn: "Average entropy of the policy action distribution in this update",
    unit: "nat",
    unitEn: "nat",
  },
  {
    key: "approx_kl",
    label: "近似 KL 散度",
    labelEn: "Approximate KL divergence",
    description: "Simulation 在 PPO 更新后输出的新旧策略近似 KL 散度",
    descriptionEn: "Approximate KL divergence between the old and new policies after the PPO update",
    unit: "无量纲",
    unitEn: "dimensionless",
    optional: true,
  },
  {
    key: "clip_fraction",
    label: "裁剪比例",
    labelEn: "Clip fraction",
    description: "Simulation 输出的 PPO 比率被裁剪样本比例",
    descriptionEn: "Fraction of PPO ratios clipped by Simulation",
    unit: "%",
    unitEn: "%",
    optional: true,
  },
  {
    key: "explained_variance",
    label: "解释方差",
    labelEn: "Explained variance",
    description: "Simulation 输出的价值函数对回报方差的解释程度",
    descriptionEn: "How much of the return variance is explained by the value function",
    unit: "无量纲",
    unitEn: "dimensionless",
    optional: true,
  },
  {
    key: "learning_rate",
    label: "学习率",
    labelEn: "Learning rate",
    description: "当前 Simulation 尚未在训练记录或 Event 中输出该字段",
    descriptionEn: "The current Simulation does not emit this field in training records or events",
    unit: "每次更新",
    unitEn: "per update",
    optional: true,
  },
  {
    key: "steps_per_second",
    label: "训练吞吐",
    labelEn: "Training throughput",
    description: "当前 Simulation 尚未在训练记录或 Event 中输出该字段",
    descriptionEn: "The current Simulation does not emit this field in training records or events",
    unit: "步/秒",
    unitEn: "steps/second",
    optional: true,
  },
  {
    key: "candidate_rate", label: "候选抓取率", labelEn: "Grasp candidate rate",
    description: "满足接触、闭合与稳定窗口条件的抓取候选比例", descriptionEn: "Share of episodes meeting contact, closure and stability-window conditions",
    unit: "比例", unitEn: "ratio", optional: true,
  },
  {
    key: "verified_success_rate", label: "验证抓取成功率", labelEn: "Verified grasp success",
    description: "候选抓取通过抬升验证器的比例", descriptionEn: "Share of episodes whose grasp passes the lift verifier",
    unit: "比例", unitEn: "ratio", optional: true,
  },
  {
    key: "mean_maximum_cube_rise", label: "平均最大抬升", labelEn: "Mean maximum cube rise",
    description: "每个回合方块相对初始高度的最大升高平均值", descriptionEn: "Mean per-episode maximum cube rise from its initial height",
    unit: "m", unitEn: "m", optional: true,
  },
  {
    key: "mean_hold_duration", label: "平均稳定保持", labelEn: "Mean stable hold",
    description: "方块处于目标高度带且抓取稳定的连续保持时长", descriptionEn: "Continuous duration in the target band with a stable grasp",
    unit: "s", unitEn: "s", optional: true,
  },
  {
    key: "descended_rate", label: "下降到位率", labelEn: "Descended-to-height rate",
    description: "到达合法释放高度的回合比例", descriptionEn: "Share of episodes reaching the legal release height",
    unit: "比例", unitEn: "ratio", optional: true,
  },
  {
    key: "detached_rate", label: "物体脱离率", labelEn: "Object detachment rate",
    description: "松爪后物体已从夹爪脱离的回合比例", descriptionEn: "Share of episodes where the object detached after release",
    unit: "比例", unitEn: "ratio", optional: true,
  },
  {
    key: "settled_1s_rate", label: "稳定放置率", labelEn: "Stable placement rate",
    description: "物体在目标区域稳定保持至少一秒的回合比例", descriptionEn: "Share of episodes settled in the target zone for at least one second",
    unit: "比例", unitEn: "ratio", optional: true,
  },
  {
    key: "mean_final_xy_error", label: "平均最终平面误差", labelEn: "Mean final XY error",
    description: "放置结束时方块中心与目标中心的平均平面距离", descriptionEn: "Mean planar distance from the cube center to the target at episode end",
    unit: "m", unitEn: "m", optional: true,
  },
] as const;

export type RunMetricKey = typeof RUN_METRIC_DEFINITIONS[number]["key"];

export function runMetricText(
  definition: typeof RUN_METRIC_DEFINITIONS[number],
  field: "label" | "description" | "unit",
  locale: Locale = "zh-CN",
): string {
  if (locale !== "en") return definition[field];
  return definition[`${field}En` as "labelEn" | "descriptionEn" | "unitEn"];
}

export interface RunMetricPoint {
  updateIndex: number;
  trainingSteps: number | null;
  sequence: number | null;
  values: Partial<Record<RunMetricKey, number>>;
}

export function visibleRunMetricDefinitions(points: RunMetricPoint[]) {
  return RUN_METRIC_DEFINITIONS.filter(
    (definition) => !("optional" in definition && definition.optional)
      || points.some((point) => point.values[definition.key] !== undefined),
  );
}

export function missingSimulationMetricDefinitions(points: RunMetricPoint[]) {
  return RUN_METRIC_DEFINITIONS.filter(
    (definition) => "optional" in definition && definition.optional
      && !points.some((point) => point.values[definition.key] !== undefined),
  );
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export const PRODUCTION_EVALUATION_DEFINITIONS = [
  { key: "success_rate", label: "成功率", labelEn: "Success rate", format: "percent", unit: "%", unitEn: "%", description: "100 个固定 seed 中达到正式成功条件的比例", descriptionEn: "Share of 100 fixed-seed episodes that meet the formal success condition" },
  { key: "mean_episode_length", label: "平均回合长度", labelEn: "Mean episode length", format: "number", unit: "动作步", unitEn: "action steps", description: "确定性评估回合实际动作步数的平均值", descriptionEn: "Mean action steps in deterministic evaluation episodes" },
  { key: "mean_episode_return", label: "平均回报", labelEn: "Mean return", format: "number", unit: "累计奖励", unitEn: "cumulative reward", description: "确定性评估每回合累计奖励的平均值", descriptionEn: "Mean cumulative reward per deterministic evaluation episode" },
  { key: "mean_final_distance", label: "平均最终距离", labelEn: "Mean final distance", format: "distance", unit: "m", unitEn: "m", description: "评估回合结束时物体到目标距离的平均值", descriptionEn: "Mean object-to-target distance at the end of evaluation episodes" },
  { key: "mean_minimum_distance", label: "平均最小距离", labelEn: "Mean minimum distance", format: "distance", unit: "m", unitEn: "m", description: "每个评估回合达到的最小目标距离平均值", descriptionEn: "Mean minimum object-to-target distance reached per evaluation episode" },
  { key: "mean_object_displacement", label: "平均物体位移", labelEn: "Mean object displacement", format: "distance", unit: "m", unitEn: "m", description: "评估回合内箱体相对初始位置的位移平均值", descriptionEn: "Mean displacement of the box from its initial position during evaluation" },
  { key: "timeout_rate", label: "超时率", labelEn: "Timeout rate", format: "percent", unit: "%", unitEn: "%", description: "达到回合步数上限但未成功的评估回合比例", descriptionEn: "Share of evaluation episodes that hit the step limit without succeeding" },
  { key: "failure_rate", label: "失败率", labelEn: "Failure rate", format: "percent", unit: "%", unitEn: "%", description: "Simulation 明确标记失败的评估回合比例", descriptionEn: "Share of evaluation episodes explicitly marked as failed by Simulation" },
] as const;

export function evaluationMetricText(
  definition: typeof PRODUCTION_EVALUATION_DEFINITIONS[number],
  field: "label" | "description" | "unit",
  locale: Locale = "zh-CN",
): string {
  if (locale !== "en") return definition[field];
  return definition[`${field}En` as "labelEn" | "descriptionEn" | "unitEn"];
}

export type ProductionEvaluationMetricKey =
  typeof PRODUCTION_EVALUATION_DEFINITIONS[number]["key"];

/** Read final policy evidence only from the formal nested callback contract. */
export function productionEvaluation(
  metrics: RunMetricsSummary | Record<string, unknown>,
): RunEvaluation | null {
  const candidate = metrics.evaluation;
  if (!candidate || typeof candidate !== "object" || Array.isArray(candidate)) return null;
  const value = candidate as Partial<RunEvaluation>;
  if (
    !["passed", "failed", "unavailable"].includes(value.status ?? "")
    || value.deterministic !== true
    || typeof value.policy_passed !== "boolean"
    || value.policy_passed !== (value.status === "passed")
  ) {
    return null;
  }
  return value as RunEvaluation;
}

export function evaluationMetric(
  evaluation: RunEvaluation | null,
  key: ProductionEvaluationMetricKey,
): number | null {
  return evaluation ? finiteNumber(evaluation[key]) : null;
}

export function runMetricHistory(events: MetricEvent[]): RunMetricPoint[] {
  const points = new Map<number, RunMetricPoint>();

  for (const event of events) {
    if (event.type !== "metric" || !event.payload.metrics) continue;
    const updateIndex = finiteNumber(event.payload.metrics.update_index);
    if (updateIndex === null || updateIndex < 1 || !Number.isInteger(updateIndex)) continue;

    const values: Partial<Record<RunMetricKey, number>> = {};
    for (const definition of RUN_METRIC_DEFINITIONS) {
      const value = finiteNumber(event.payload.metrics[definition.key]);
      if (value !== null) values[definition.key] = value;
    }
    if (!Object.keys(values).length) continue;

    const sequence = finiteNumber(event.sequence);
    const existing = points.get(updateIndex);
    if (
      existing
      && existing.sequence !== null
      && (sequence === null || sequence < existing.sequence)
    ) {
      continue;
    }
    points.set(updateIndex, {
      updateIndex,
      trainingSteps: finiteNumber(event.payload.step),
      sequence,
      values,
    });
  }

  return [...points.values()].sort((left, right) => left.updateIndex - right.updateIndex);
}

export function metricSeriesData(
  points: RunMetricPoint[],
  key: RunMetricKey,
): Array<[number, number | null]> {
  return points.map((point) => [point.updateIndex, point.values[key] ?? null]);
}

function csvCell(value: unknown): string {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

/** Export only persisted Event/evaluation values; absent evidence stays blank. */
export function buildRunMetricsCsv(record: RunRecord, points: RunMetricPoint[]): string {
  const evaluation = productionEvaluation(record.metrics_summary);
  const trainingKeys = RUN_METRIC_DEFINITIONS.map((definition) => definition.key);
  const evaluationKeys = [
    "status", "success_count", "episode_count", "success_rate",
    "wilson_95_lower_bound", "wilson_95_upper_bound",
    ...PRODUCTION_EVALUATION_DEFINITIONS
      .map((definition) => definition.key)
      .filter((key) => key !== "success_rate"),
  ] as const;
  const headers = [
    "source", "run_id", "update_index", "training_steps",
    ...trainingKeys,
    ...evaluationKeys.map((key) => `evaluation_${key}`),
  ];
  const rows: unknown[][] = points.map((point) => [
    "training_event", record.run_id, point.updateIndex, point.trainingSteps,
    ...trainingKeys.map((key) => point.values[key]),
    ...evaluationKeys.map(() => undefined),
  ]);
  if (evaluation) {
    rows.push([
      "formal_evaluation", record.run_id, undefined, undefined,
      ...trainingKeys.map(() => undefined),
      ...evaluationKeys.map((key) => evaluation[key]),
    ]);
  }
  return [headers, ...rows]
    .map((row) => row.map(csvCell).join(","))
    .join("\r\n") + "\r\n";
}
