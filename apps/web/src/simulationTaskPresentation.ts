import { translateUiText, type Locale } from "./i18n";
import type { SimulationTaskDefinition } from "./simulationCatalog";

const ENGLISH_TASK_TITLES: Readonly<Record<string, string>> = {
  s1_push: "S1 Push",
  s1_reach: "S1 Reach",
  s1_grasp: "S1 Grasp",
  s1_lift: "S1 Grasp and Lift",
  s1_lower_release: "S1 Lower and Release",
  s1_place: "S1 Transport and Place",
  demo_s1_target_pose: "S1 Target Pose",
  demo_s1_line_tracking: "S1 Line Tracking",
  demo_s1_circle_tracking: "S1 Circle Tracking",
  demo_s1_joint_position: "S1 Joint Position",
  demo_s1_joint_sweep: "S1 Joint Sweep",
  demo_s1_scripted_pick_place: "S1 Scripted Pick and Place",
};

export function simulationTaskTitle(task: Pick<SimulationTaskDefinition, "id" | "title">, locale: Locale): string {
  return locale === "en" ? ENGLISH_TASK_TITLES[task.id] ?? translateUiText(task.title) : task.title;
}
