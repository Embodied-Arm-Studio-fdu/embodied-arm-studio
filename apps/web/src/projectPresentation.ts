import type { TeachingProject } from "./api";

export function teachingProjectLabel(
  project: Pick<TeachingProject, "title" | "environment_type" | "training_profile">,
  locale: "zh-CN" | "en",
): string {
  if (project.environment_type === "s1_push") {
    if (project.training_profile === "short") return locale === "en" ? "S1 Push · introductory short run" : "S1 推动 · 入门短程";
    if (project.training_profile === "standard") return locale === "en" ? "S1 Push · standard training" : "S1 推动 · 标准训练";
  }
  return project.title;
}