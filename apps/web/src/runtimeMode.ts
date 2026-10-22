export type RuntimeModeContext =
  | "canonical-platform"
  | "legacy-quick-course";

export interface RuntimeModeLabels {
  readonly mode: string;
  readonly capability: string;
  readonly safety: string;
}

export type RuntimeLocale = "zh-CN" | "en";

export const CANONICAL_PLATFORM_RUNTIME_CONTEXT =
  "canonical-platform" as const;

export const LEGACY_QUICK_COURSE_RUNTIME_CONTEXT =
  "legacy-quick-course" as const;

const CANONICAL_SIMULATION_LABELS = Object.freeze({
  mode: "仿真训练模式",
  capability: "支持 GPU 训练",
  safety: "真实机械臂已禁用",
}) satisfies RuntimeModeLabels;

const LEGACY_MOCK_LABELS = Object.freeze({
  mode: "演示练习模式",
  capability: "仅用于教学演示",
  safety: "真实机械臂已禁用",
}) satisfies RuntimeModeLabels;

const CANONICAL_SIMULATION_LABELS_EN = Object.freeze({
  mode: "Simulation training mode",
  capability: "GPU training supported",
  safety: "Real robot arm disabled",
}) satisfies RuntimeModeLabels;

const LEGACY_MOCK_LABELS_EN = Object.freeze({
  mode: "Demo practice mode",
  capability: "For teaching demos only",
  safety: "Real robot arm disabled",
}) satisfies RuntimeModeLabels;

export function runtimeModeLabels(
  context: RuntimeModeContext,
  locale: RuntimeLocale = "zh-CN",
): RuntimeModeLabels {
  if (locale === "en") {
    return context === LEGACY_QUICK_COURSE_RUNTIME_CONTEXT
      ? LEGACY_MOCK_LABELS_EN
      : CANONICAL_SIMULATION_LABELS_EN;
  }
  return context === LEGACY_QUICK_COURSE_RUNTIME_CONTEXT
    ? LEGACY_MOCK_LABELS
    : CANONICAL_SIMULATION_LABELS;
}
