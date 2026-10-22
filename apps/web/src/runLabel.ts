import type { RunRecord } from "./api";

const TASK_NAMES: Record<string, { "zh-CN": string; en: string }> = {
  s1_reach: { "zh-CN": "Reach", en: "Reach" },
  s1_grasp: { "zh-CN": "Grasp 抓取", en: "Grasp" },
  s1_lift: { "zh-CN": "Grasp 抓取抬起", en: "Grasp & Lift" },
  s1_lower_release: { "zh-CN": "Grasp 抓取放下", en: "Lower & Release" },
  s1_place: { "zh-CN": "Place 搬运放置", en: "Transport & Place" },
  demo_s1_target_pose: { "zh-CN": "目标位姿 Demo", en: "Target Pose Demo" },
  demo_s1_line_tracking: { "zh-CN": "直线跟踪 Demo", en: "Line Tracking Demo" },
  demo_s1_circle_tracking: { "zh-CN": "圆轨迹 Demo", en: "Circle Tracking Demo" },
  demo_s1_joint_position: { "zh-CN": "关节位置 Demo", en: "Joint Position Demo" },
  demo_s1_joint_sweep: { "zh-CN": "关节扫描 Demo", en: "Joint Sweep Demo" },
  demo_s1_scripted_pick_place: { "zh-CN": "规则抓取放下 Demo", en: "Scripted Pick & Place Demo" },
};

export function runTaskId(record: RunRecord): string {
  const snapshot = record.project_config_snapshot;
  const catalogTask = snapshot.catalog_task as { id?: unknown } | undefined;
  const taskConfig = snapshot.task_config as { environment_type?: unknown } | undefined;
  if (typeof catalogTask?.id === "string") return catalogTask.id;
  if (typeof snapshot.environment_type === "string") return snapshot.environment_type;
  if (typeof taskConfig?.environment_type === "string") return taskConfig.environment_type;
  return record.simulation_task_id ?? "run";
}

export function runDisplayName(record: RunRecord, locale: "zh-CN" | "en" = "zh-CN"): string {
  const taskId = runTaskId(record);
  let title = TASK_NAMES[taskId]?.[locale];
  if (!title && taskId === "s1_push") {
    const profile = record.project_config_snapshot.training_profile;
    title = profile === "standard" ? "Standard Push" : "Short Push";
  }
  title ??= taskId === "run" ? (locale === "en" ? "Experiment" : "实验") : taskId;
  const hex = record.run_id.replace(/[^0-9a-f]/gi, "").slice(-8);
  const numericId = hex ? (Number.parseInt(hex, 16) % 10000) + 1 : 1;
  return `${title} #${numericId}`;
}
