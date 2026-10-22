import { describe, expect, it } from "vitest";
import { readRewardWeights, rewardPreset, writeRewardWeights } from "./rewardTeaching";

const terms = ["progress_reward", "action_penalty"];
describe("teaching weights preserve the executable expression", () => {
  it("reads the Starter and a partially omitted weighted sum without inventing contributions", () => {
    expect(readRewardWeights('def combine_reward(terms):\n    return terms["progress_reward"] + terms["action_penalty"]\n', terms)).toEqual({ progress_reward: 1, action_penalty: 1 });
    expect(readRewardWeights("def combine_reward(terms):\n    return 2 * terms['action_penalty']\n", terms)).toEqual({ progress_reward: 0, action_penalty: 2 });
  });
  it.each([
    'terms["progress_reward"] - terms["action_penalty"]',
    'terms["progress_reward"] / terms["action_penalty"]',
    '-2 * terms["action_penalty"]',
    'terms["success_bonus"]',
    'terms["progress_reward"] + terms["progress_reward"]',
    'terms["progress_reward"] + 3',
    '6 * terms["action_penalty"]',
    'float(terms["progress_reward"])',
  ])("leaves custom Python unchanged rather than approximating it: %s", (expression) => {
    expect(readRewardWeights(`def combine_reward(terms):\n    return ${expression}\n`, terms)).toBeNull();
  });
  it("rejects additional statements and an unavailable allowlist", () => {
    expect(readRewardWeights('def combine_reward(terms):\n    return terms["progress_reward"]\nprint(1)', terms)).toBeNull();
    expect(readRewardWeights('def combine_reward(terms):\n    return terms["progress_reward"]', [])).toBeNull();
  });
  it("doubles the negative action contribution using addition and roundtrips zero weights", () => {
    const weights = rewardPreset(terms, "restraint");
    const source = writeRewardWeights(terms, weights);
    expect(source).toContain('+ 2 * terms["action_penalty"]');
    expect(.04 * weights.progress_reward + -.01 * weights.action_penalty).toBeCloseTo(.02);
    expect(readRewardWeights(writeRewardWeights(terms, { progress_reward: 0, action_penalty: .5 }), terms)).toEqual({ progress_reward: 0, action_penalty: .5 });
  });
  it("maps progress and restraint presets to the actual Place and Push terms", () => {
    expect(rewardPreset(["xy_progress", "target_zone", "action_failure"], "progress")).toEqual({ xy_progress: 2, target_zone: 1, action_failure: 1 });
    expect(rewardPreset(["directional_push_progress", "action_penalty"], "restraint")).toEqual({ directional_push_progress: 1, action_penalty: 2 });
  });
  it("rejects invalid generator inputs", () => {
    expect(() => writeRewardWeights(terms, { progress_reward: Infinity, action_penalty: 1 })).toThrow();
    expect(() => writeRewardWeights(terms, { progress_reward: -1, action_penalty: 1 })).toThrow();
  });
});
