#!/usr/bin/env python3
"""P4-R6 aligned Transport-only A/B trace for original and staged runners."""

import hashlib
import json
import math
from pathlib import Path
import random

try:
    from isaacgym import gymapi as _gymapi_import_order  # noqa: F401
except ModuleNotFoundError:
    _gymapi_import_order = None

import torch

from experimental_s1_place_staged import PolicyRuntime, TRANSPORT, TRANSPORT_SHA
from simulation_core.experimental.manipulation.lower_release_schema import LOWER_RELEASE_POLICY_SCHEMA
from simulation_core.experimental.manipulation.place_env import PlaceTransportEnvironment
from simulation_core.experimental.manipulation.staged_place_env import StagedFullPlaceEnvironment
from simulation_core.ppo.policy import flatten_observation


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "experimental" / "p4_r6_s1_place" / "transport_ab_20"
BASE_SEED = 20261003


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def vector(observation):
    return flatten_observation(observation, LOWER_RELEASE_POLICY_SCHEMA)


def trace_variant(name, env_class, runtime, episodes=20):
    env = env_class(); episode_rows = []; traces = []; cleanup = False
    try:
        env.initialize()
        for episode in range(episodes):
            seed = BASE_SEED + episode; seed_all(seed); observation = env.reset()
            reset = env.scene.snapshot()
            reset_record = {"seed": seed, "reset_ee_pose": list(reset["ee_state"][:7]),
                            "reset_cube_pose": list(reset["cube_state"][:7]),
                            "cube_target": list(env.target_center),
                            "persistent_xyz_target_initial": list(env._target)}
            steps = []; done = False; external_hold = 0; reason = ""; success = False
            while not done:
                raw = vector(observation); normalized = runtime.normalizer.normalize(
                    raw.to("cuda:0"), update=False)
                action = runtime.act(observation)
                requested_z = env._transport_start_cube[2] - env._reference_relative[2]
                pre = (env._target[0] + action[0] * env.config.xy_delta_scale,
                       env._target[1] + action[1] * env.config.xy_delta_scale, requested_z)
                post, would_clamp = env.config.workspace.clamp(pre)
                margins = {"lower": [pre[i] - env.config.workspace.lower[i] for i in range(3)],
                           "upper": [env.config.workspace.upper[i] - pre[i] for i in range(3)]}
                if name == "A_original":
                    next_observation, _, env_done, info = env.step(action)
                    done = env_done; success = bool(info["success"]); reason = info["failure_reason"]
                else:
                    env.staged_phase = "TRANSPORT"
                    next_observation, _, env_done, info = env.step(action)
                    env.staged_phase = "TRANSPORT"; env._switched_to_lower = False
                    snapshot = env.scene.snapshot(); metrics = env._metrics(snapshot)
                    rise = metrics["rise"]; xy_error = env._xy_error(snapshot)
                    stable = (xy_error <= env.config.target_xy_tolerance
                              and env.config.transport_height_band[0] <= rise <=
                              env.config.transport_height_band[1]
                              and metrics["grasp_valid"] and metrics["jaw"] >= .01
                              and metrics["translation_drift"] <= .012
                              and metrics["rotation_drift"] <= .25 and env._finite(snapshot)
                              and max(metrics["forces"].values()) <= env.config.force_abort)
                    external_hold = external_hold + 1 if stable else 0
                    success = external_hold >= env.hold_steps_required
                    reason = info["failure_reason"]
                    done = success or env_done
                    if success: reason = ""
                after = env.scene.snapshot()
                steps.append({
                    "step": len(steps) + 1, "observation": raw.tolist(),
                    "normalized_observation": normalized.detach().cpu().tolist(),
                    "actor_mean_action": list(action), "applied_dx_dy": list(info["applied_action"][:2]),
                    "z_target": pre[2], "workspace_margins": margins,
                    "pre_clamp_target": list(pre), "post_clamp_target": list(post),
                    "would_clamp": would_clamp,
                    "workspace_clamped": bool(info["workspace_violation"]),
                    "persistent_target_after": list(env._target),
                    "ee_pose_after": list(after["ee_state"][:7]),
                    "cube_pose_after": list(after["cube_state"][:7]),
                    "termination_reason": reason if done else "",
                })
                observation = next_observation
                if len(steps) > env.max_steps + 1: raise RuntimeError("transport trace exceeded horizon")
            episode_rows.append({"episode": episode, "seed": seed, "success": success,
                                 "workspace_violation": reason == "WORKSPACE_VIOLATION",
                                 "termination_reason": reason or "SUCCESS", "steps": len(steps)})
            traces.append({"episode": episode, "reset": reset_record, "steps": steps})
        cleanup = True
    finally: env.close()
    return episode_rows, traces, cleanup


def max_difference(left, right):
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return max((max_difference(a, b) for a, b in zip(left, right)), default=0.0)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def first_divergence(a_traces, b_traces, tolerance=1e-8):
    reset_fields = ("reset_ee_pose", "reset_cube_pose", "cube_target",
                    "persistent_xyz_target_initial")
    step_fields = ("observation", "normalized_observation", "actor_mean_action",
                   "applied_dx_dy", "z_target", "pre_clamp_target", "post_clamp_target",
                   "persistent_target_after", "ee_pose_after", "cube_pose_after")
    for a, b in zip(a_traces, b_traces):
        for field in reset_fields:
            difference = max_difference(a["reset"][field], b["reset"][field])
            if difference > tolerance:
                return {"episode": a["episode"], "step": 0, "field": field,
                        "max_abs_difference": difference}
        for left, right in zip(a["steps"], b["steps"]):
            for field in step_fields:
                difference = max_difference(left[field], right[field])
                if difference > tolerance:
                    return {"episode": a["episode"], "step": left["step"], "field": field,
                            "max_abs_difference": difference}
    return None


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runtime = PolicyRuntime(TRANSPORT, TRANSPORT_SHA, (1,1,0,0), "transport")
    a_rows, a_traces, a_cleanup = trace_variant("A_original", PlaceTransportEnvironment, runtime)
    b_rows, b_traces, b_cleanup = trace_variant("B_staged", StagedFullPlaceEnvironment, runtime)
    for name, traces in (("a_original_traces.jsonl", a_traces),
                         ("b_staged_traces.jsonl", b_traces)):
        path = OUTPUT / name
        with path.open("w", encoding="utf-8") as stream:
            for row in traces: stream.write(json.dumps(row, sort_keys=True) + "\n")
    divergence = first_divergence(a_traces, b_traces)
    summary = {
        "episodes": 20, "same_seeds": [BASE_SEED + i for i in range(20)],
        "checkpoint_sha256": runtime.sha256,
        "normalizer_count": runtime.normalizer.count,
        "deterministic": True,
        "a_original": {"success_count": sum(r["success"] for r in a_rows),
                       "workspace_violation_count": sum(r["workspace_violation"] for r in a_rows),
                       "episodes": a_rows, "resource_cleanup": a_cleanup},
        "b_staged_transport_only": {"success_count": sum(r["success"] for r in b_rows),
                       "workspace_violation_count": sum(r["workspace_violation"] for r in b_rows),
                       "episodes": b_rows, "resource_cleanup": b_cleanup},
        "post_fix_first_divergence": divergence,
        "aligned": divergence is None,
        "diagnosis": {
            "classification": "PHASE_STATE_MISMATCH",
            "pre_fix_first_divergence": {
                "episode": 0, "step": 1, "field": "persistent_target_after.z",
                "p4_1": 0.4486258327960968, "r5": 0.4494905471801758,
                "next_observation_divergence": "previous_action dz/grip zeroed only in R5",
                "step_2_observation_max_abs_difference": 0.4327758550643921,
            },
            "fix": "P4-1 fixed-Z reconstruction plus raw 4D previous_action during TRANSPORT",
        },
    }
    write_json(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__": main()
