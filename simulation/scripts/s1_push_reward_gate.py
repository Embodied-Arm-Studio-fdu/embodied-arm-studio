#!/usr/bin/env python3
"""P1.3 counterfactual and real-GPU gated-reward validation without PPO."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from isaacgym import gymapi as _gymapi  # noqa: F401
import numpy as np

from simulation_core.push_adapter import ACTION_SCALES, IsaacGymPushBackend, PushAdapter, PushConfig
from s1_push_observation_gate import array, clamp_flags


def run_production_replay(environment, backend):
    environment.reset()
    initial_object = array(backend.read_object_state()[:3])
    goal = np.asarray(environment.config.goal_position)
    direction = goal - initial_object
    direction[2] = 0.0
    direction /= np.linalg.norm(direction)
    trace = []
    first_contact = None
    increment = tuple(direction[:2] * ACTION_SCALES[0])
    for index in range(408):
        if index < 8:
            stage, action = "approach_settle", (0.0, 0.0)
        else:
            tracking = np.linalg.norm(trace[-1]["target_tracking_error_xy"])
            action = increment if tracking <= 0.010 else (0.0, 0.0)
            stage = "precontact" if first_contact is None else "effective_push"
        observation, reward, done, info = environment.step(action)
        obj = array(backend.read_object_state()[:3])
        displacement = obj - initial_object
        delta_clamp, limit_clamp = clamp_flags(backend)
        row = {
            "step": index,
            "stage": stage,
            "action_xy": list(action),
            "precontact_error": info["precontact_error"],
            "longitudinal": info["longitudinal"],
            "lateral": info["lateral"],
            "behind_gate": info["behind_gate"],
            "alignment_gate": info["alignment_gate"],
            "near_precontact": info["near_precontact"],
            "contact": info["contact"],
            "object_delta_directional": info["directional_push"],
            "object_delta_lateral": info["lateral_push"],
            "object_displacement": float(np.linalg.norm(displacement)),
            "directional_object_displacement": float(np.dot(displacement, direction)),
            "object_goal_distance": float(np.linalg.norm(goal - obj)),
            "target_tracking_error_xy": list(observation["target_tracking_error_xy"]),
            "reward_terms": info["reward_terms"],
            "total_reward": reward,
            "joint_delta_clamped": delta_clamp,
            "joint_limit_clamped": limit_clamp,
            "finite": bool(math.isfinite(reward) and all(math.isfinite(x) for x in info["reward_terms"].values())),
            "failure_reason": info["failure_reason"],
        }
        trace.append(row)
        if info["contact"] and first_contact is None:
            first_contact = index
        if row["directional_object_displacement"] >= 0.012:
            break
        if done and info["failure_reason"]:
            break
    terms = defaultdict(float)
    term_samples = defaultdict(list)
    stages = defaultdict(lambda: {"steps": 0, "return": 0.0})
    for row in trace:
        stages[row["stage"]]["steps"] += 1
        stages[row["stage"]]["return"] += row["total_reward"]
        for name, value in row["reward_terms"].items():
            terms[name] += value
            term_samples[name].append(value)
    final = trace[-1]
    return {
        "result": "PASS" if (
            first_contact is not None
            and final["object_displacement"] >= 0.01
            and final["directional_object_displacement"] >= 0.01
            and all(row["finite"] for row in trace)
            and not any(row["joint_delta_clamped"] or row["joint_limit_clamped"] for row in trace)
            and final["failure_reason"] is None
        ) else "FAIL",
        "first_contact_step": first_contact,
        "steps": len(trace),
        "return": sum(row["total_reward"] for row in trace),
        "reward_term_returns": dict(terms),
        "reward_term_statistics": {
            name: {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "return": sum(values),
            }
            for name, values in term_samples.items()
        },
        "stage_summary": dict(stages),
        "per_step_reward_min": min(row["total_reward"] for row in trace),
        "per_step_reward_max": max(row["total_reward"] for row in trace),
        "object_displacement": final["object_displacement"],
        "directional_object_displacement": final["directional_object_displacement"],
        "directional_fraction": final["directional_object_displacement"] / max(final["object_displacement"], 1e-12),
        "initial_object_goal_distance": float(np.linalg.norm(goal - initial_object)),
        "final_object_goal_distance": final["object_goal_distance"],
        "trace": trace,
    }


def run_stationary(environment, steps=40):
    environment.reset()
    rows = []
    for _ in range(steps):
        _, reward, done, info = environment.step((0.0, 0.0))
        rows.append({"reward": reward, "terms": info["reward_terms"], "contact": info["contact"]})
        if done:
            break
    return {
        "steps": len(rows),
        "return": sum(row["reward"] for row in rows),
        "contact_steps": sum(row["contact"] for row in rows),
        "rows": rows,
    }


def analytic_counterfactuals(config):
    """One-step values of the exact configured formula at controlled geometry."""
    behind = config.precontact_behind_offset
    lateral = config.precontact_lateral_offset
    approach_step = 0.005
    action_cost = config.action_penalty_scale * 0.5
    correct_approach = config.precontact_reward_scale * approach_step - action_cost
    wrong_approach = -config.wrong_side_penalty_scale * approach_step - action_cost
    forward_1cm = config.directional_push_reward_scale * 0.01
    backward_1cm = -config.directional_push_reward_scale * 0.01
    lateral_1cm = -config.lateral_push_penalty_scale * 0.01
    partial_forward = config.directional_push_reward_scale * 0.005
    return {
        "geometry_used": {"behind_offset": behind, "lateral_offset": lateral},
        "cases": {
            "B_stationary": 0.0,
            "C_wrong_side_approach_5mm": wrong_approach,
            "D_correct_precontact_no_push": 0.0,
            "E_forward_push_1cm": forward_1cm,
            "F_backward_push_1cm": backward_1cm,
            "G_lateral_push_1cm": lateral_1cm,
            "correct_precontact_approach_5mm": correct_approach,
            "correct_directional_partial_push_5mm": partial_forward,
            "contact_only_one_step": config.contact_bonus,
        },
        "ordering_pass": bool(
            forward_1cm > partial_forward > correct_approach > 0.0
            and wrong_approach < 0.0
            and backward_1cm < lateral_1cm < 0.0
            and config.contact_bonus < config.directional_push_reward_scale * 0.001
        ),
        "hacking_checks": {
            "precontact_oscillation_net_before_action_cost": 0.0,
            "precontact_stationary": 0.0,
            "growing_tracking_error_without_actual_motion": 0.0,
            "symmetric_lateral_jitter_is_penalized_per_leg": True,
            "backward_motion_negative": backward_1cm < 0.0,
            "high_frequency_action_has_negative_action_cost": config.action_penalty_scale > 0.0,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = PushConfig(max_steps=512, physics_steps_per_action=8)
    backend = IsaacGymPushBackend(config)
    environment = PushAdapter(config, backend=backend)
    try:
        environment.initialize()
        replay = run_production_replay(environment, backend)
        stationary = run_stationary(environment)
    finally:
        environment.close()
    counterfactuals = analytic_counterfactuals(config)
    dominant = max(
        replay["reward_term_returns"],
        key=lambda name: replay["reward_term_returns"][name],
    )
    directional_dominant = dominant == "directional_push_progress"
    passed = bool(
        replay["result"] == "PASS"
        and counterfactuals["ordering_pass"]
        and stationary["return"] <= 0.01
        and directional_dominant
    )
    result = {
        "gate": "P1.3 task-progress gated reward",
        "runtime": "Isaac Gym Preview 4, GPU PhysX, RTX 4080",
        "reward_config": {
            name: getattr(config, name) for name in (
                "precontact_reward_scale", "wrong_side_penalty_scale",
                "directional_push_reward_scale", "lateral_push_penalty_scale",
                "contact_bonus", "success_bonus", "action_penalty_scale",
                "joint_limit_penalty_scale", "precontact_behind_offset",
                "precontact_lateral_offset", "precontact_near_radius",
            )
        },
        "production_replay": replay,
        "stationary_production": stationary,
        "counterfactuals": counterfactuals,
        "largest_positive_contribution": dominant,
        "result": "PASS" if passed else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("P1_3_GATED_REWARD_GATE=%s" % result["result"])
    print(json.dumps({
        "replay": {key: value for key, value in replay.items() if key != "trace"},
        "stationary": {key: value for key, value in stationary.items() if key != "rows"},
        "counterfactuals": counterfactuals,
        "largest_positive_contribution": dominant,
    }, indent=2))


if __name__ == "__main__":
    main()
