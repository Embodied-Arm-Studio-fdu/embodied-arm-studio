export interface S1JointDefinition {
  name: string;
  label: string;
  labelEn: string;
  lower: number;
  upper: number;
  effort: number;
  velocity: number;
  origin: [number, number, number];
  rpy: [number, number, number];
  axis: [number, number, number];
}

// Values are transcribed from the S1 v2 URDF loaded by the Isaac Gym adapter.
export const S1_ARM_JOINTS: S1JointDefinition[] = [
  { name: "1joint", label: "基座旋转", labelEn: "Base rotation", lower: -2.967, upper: 2.967, effort: 9, velocity: 10, origin: [0, 0, 0], rpy: [0, 0, 0], axis: [0, 0, 1] },
  { name: "2joint", label: "肩部", labelEn: "Shoulder", lower: 0, upper: 3.14, effort: 9, velocity: 10, origin: [0, 0, 0.065224], rpy: [1.5708, 0, 0], axis: [0, 0, 1] },
  { name: "3joint", label: "肘部", labelEn: "Elbow", lower: 0, upper: 3.14, effort: 9, velocity: 10, origin: [0.3, 0, 0.0005], rpy: [0, 0, 1.5708], axis: [0, 0, -1] },
  { name: "4joint", label: "腕部旋转", labelEn: "Wrist rotation", lower: -1.57, upper: 1.518, effort: 3, velocity: 10, origin: [0.057369, 0.22028, -0.000495], rpy: [0, 0, 1.5708], axis: [0, 0, 1] },
  { name: "5joint", label: "腕部俯仰", labelEn: "Wrist pitch", lower: -1.57, upper: 1.57, effort: 3, velocity: 10, origin: [0.080666, -0.05857, 0.00047555], rpy: [-1.5708, 0, 0], axis: [0, 0, 1] },
  { name: "6joint", label: "末端旋转", labelEn: "End-effector rotation", lower: -1.57, upper: 1.57, effort: 3, velocity: 10, origin: [0, 0, 0.059088], rpy: [1.5708, 0, 1.5708], axis: [0, 0, 1] },
];

export function s1JointLabel(joint: S1JointDefinition, locale: "zh-CN" | "en" = "zh-CN") {
  return locale === "en" ? joint.labelEn : joint.label;
}

export const S1_INITIAL_ARM_POSITIONS = [0, 0.785, 1.57, 0, 0, 0];
export const S1_GRIPPER_RANGE: [number, number] = [0, 0.05];

export function clampS1Joint(index: number, value: number): number {
  const joint = S1_ARM_JOINTS[index];
  return Math.min(joint.upper, Math.max(joint.lower, value));
}