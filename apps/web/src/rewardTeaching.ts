export type RewardTermLesson = {
  title: [string, string];
  explanation: [string, string];
  sample: number;
};

// Samples illustrate a single step; they are deliberately not measured Run data.
export const rewardLessons: Record<string, RewardTermLesson> = {
  progress_reward: { title: ["接近目标", "Approach the target"], explanation: ["比上一步更接近目标得正分，远离目标得负分。", "Moving closer than the previous step earns positive reward; moving away earns negative reward."], sample: .04 },
  action_penalty: { title: ["动作幅度惩罚", "Action magnitude penalty"], explanation: ["动作幅度越大，扣分越多。这项已经是负数，增大权重会加重扣分；它不直接衡量动作平滑度。", "Larger actions cost more. This term is already negative, so a larger weight increases the cost. It does not directly measure smoothness."], sample: -.01 },
  precontact_progress: { title: ["接触前接近方块", "Approach before contact"], explanation: ["奖励从合适位置接近方块；退远也会影响这项分数。", "Rewards approaching the cube from a suitable position; moving away also affects this term."], sample: .03 },
  wrong_side_or_alignment_term: { title: ["接触方向惩罚", "Approach alignment penalty"], explanation: ["从不适合推动的方向接近时扣分；增大权重会更重视接近方向。", "Penalizes approaching from an unsuitable pushing direction. A larger weight emphasizes approach alignment."], sample: -.02 },
  directional_push_progress: { title: ["沿目标方向推动", "Push toward the goal"], explanation: ["方块沿目标方向移动得正分，反方向移动得负分。", "Cube motion toward the goal earns positive reward; opposite motion earns negative reward."], sample: .04 },
  lateral_push_penalty: { title: ["侧向偏移惩罚", "Lateral motion penalty"], explanation: ["方块偏离推动方向时扣分；增大权重会加重这项扣分。", "Penalizes cube motion sideways from the pushing direction; a larger weight increases this cost."], sample: -.01 },
  contact_sustain_bonus: { title: ["保持接触", "Maintain contact"], explanation: ["末端与方块保持接触时得分。接触本身不代表任务成功。", "Earns reward while the end effector contacts the cube. Contact alone does not imply success."], sample: .01 },
  xy_progress: { title: ["搬运接近目标", "Transport toward the target"], explanation: ["刷新最佳目标距离时得分；在已到过的位置来回移动不会重复获得这项进展分。", "Rewards a new best target distance; revisiting an already reached position does not repeatedly earn progress reward."], sample: .04 },
  target_zone: { title: ["进入目标区域", "Enter the target zone"], explanation: ["方块在目标区域内时得分。正式成功还需要稳定保持一秒。", "Earns reward inside the target zone. Formal success also requires a stable one-second hold."], sample: .02 },
  height_stability: { title: ["高度偏差惩罚", "Height error penalty"], explanation: ["偏离搬运高度时扣分，偏差越大扣分越多。", "Penalizes deviation from the transport height; larger errors cost more."], sample: -.01 },
  grasp_stability: { title: ["抓持漂移惩罚", "Grasp drift penalty"], explanation: ["方块相对夹爪漂移或转动时扣分。", "Penalizes cube drift relative to the gripper and angular motion."], sample: -.01 },
  action_failure: { title: ["动作与失败惩罚", "Action and failure penalty"], explanation: ["合并动作幅度成本与终止失败扣分；修改权重不会取消安全终止。", "Combines action magnitude cost and terminal failure penalty. Changing its weight does not disable safety termination."], sample: -.01 },
};

/** Recognize only an exact nonnegative weighted sum, never approximate custom Python. */
export function readRewardWeights(source: string, terms: readonly string[]): Record<string, number> | null {
  if (!terms.length || new Set(terms).size !== terms.length) return null;
  const clean = source.split("\n").filter((line) => !line.trimStart().startsWith("#")).join("\n");
  const match = /^\s*def combine_reward\(terms\):\s*\n[ \t]+return ([^\n]+)\s*$/.exec(clean);
  if (!match) return null;
  const weights = Object.fromEntries(terms.map((term) => [term, 0]));
  const seen = new Set<string>();
  for (const part of match[1].split("+")) {
    const item = /^\s*(?:(\d+(?:\.\d+)?|\.\d+)\s*\*\s*)?terms\[(?:"([a-z_]+)"|'([a-z_]+)')\]\s*$/.exec(part);
    if (!item) return null;
    const term = item[2] ?? item[3];
    const weight = item[1] === undefined ? 1 : Number(item[1]);
    if (!terms.includes(term) || seen.has(term) || !Number.isFinite(weight) || weight < 0 || weight > 5) return null;
    seen.add(term);
    weights[term] = weight;
  }
  return weights;
}

export function writeRewardWeights(terms: readonly string[], weights: Record<string, number>): string {
  if (!terms.length || terms.some((term) => !/^[a-z_]+$/.test(term) || !Number.isFinite(weights[term]) || weights[term] < 0 || weights[term] > 5)) throw new Error("Invalid teaching reward weights");
  return 'def combine_reward(terms):\n    return ' + terms.map((term) => `${weights[term] === 1 ? "" : `${weights[term]} * `}terms["${term}"]`).join(" + ") + "\n";
}

export function rewardPreset(terms: readonly string[], preset: "baseline" | "progress" | "restraint") {
  return Object.fromEntries(terms.map((term) => [term,
    preset === "progress" && ["progress_reward", "directional_push_progress", "xy_progress"].includes(term) ? 2
      : preset === "restraint" && ["action_penalty", "action_failure"].includes(term) ? 2 : 1,
  ]));
}
