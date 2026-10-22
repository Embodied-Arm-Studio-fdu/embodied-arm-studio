#!/usr/bin/env python3
"""P1.2 real-GPU observation, normalization, and frozen-evaluation gate."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random

from isaacgym import gymapi as _gymapi  # noqa: F401
import numpy as np
import torch

from simulation_core.cartesian_controller import S1_LOWER_LIMITS, S1_UPPER_LIMITS
from simulation_core.ppo import ObservationNormalizer, flatten_observation
from simulation_core.push_adapter import ACTION_SCALES, IsaacGymPushBackend, PushAdapter, PushConfig


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value, dtype=float)


def clamp_flags(backend):
    command = backend.last_command
    if command is None:
        return False, False
    delta = array(command.joint_delta)
    target = array(command.positions[:6])
    lower = np.asarray(S1_LOWER_LIMITS[:6])
    upper = np.asarray(S1_UPPER_LIMITS[:6])
    return (
        bool(np.any(np.abs(delta) >= backend.controller.config.max_joint_delta * (1.0 - 1.0e-5))),
        bool(np.any((target <= lower + 1.0e-6) | (target >= upper - 1.0e-6))),
    )


def state_equal(left, right):
    return (
        left["count"] == right["count"]
        and torch.equal(left["mean"], right["mean"])
        and torch.equal(left["m2"], right["m2"])
        and torch.equal(left["variance"], right["variance"])
    )


def sample(environment, backend, normalizer, action, update, stage, index, initial_object, goal, direction):
    observation, reward, done, info = environment.step(action)
    raw = flatten_observation(observation)
    normalized = normalizer.normalize(raw, update=update)
    obj = array(backend.read_object_state()[:3])
    state = backend.read_robot_state()
    ee_xy = array(state.ee_link_state[:2])
    expected_tracking = np.asarray(backend.last_target.position[:2]) - ee_xy
    displacement = obj - initial_object
    delta_clamp, limit_clamp = clamp_flags(backend)
    return observation, {
        "step": index,
        "stage": stage,
        "action_xy": list(action),
        "raw_observation": {name: list(values) for name, values in observation.items()},
        "normalized_observation": normalized.tolist(),
        "normalizer_count": normalizer.count,
        "ee_to_object_xy": list(observation["ee_to_object_xy"]),
        "object_to_goal_xy": list(observation["object_to_goal_xy"]),
        "target_tracking_error_xy": list(observation["target_tracking_error_xy"]),
        "previous_action_xy": list(observation["previous_action_xy"]),
        "relative_consistency_error_m": float(max(
            np.linalg.norm(np.asarray(observation["ee_to_object_xy"]) - (obj[:2] - ee_xy)),
            np.linalg.norm(np.asarray(observation["object_to_goal_xy"]) - (goal[:2] - obj[:2])),
            np.linalg.norm(np.asarray(observation["target_tracking_error_xy"]) - expected_tracking),
        )),
        "object_position": obj.tolist(),
        "object_displacement_m": float(np.linalg.norm(displacement)),
        "directional_displacement_m": float(np.dot(displacement, direction)),
        "object_goal_distance_m": float(np.linalg.norm(goal - obj)),
        "contact": bool(info["contact"]),
        "reward": float(reward),
        "joint_delta_clamped": delta_clamp,
        "joint_limit_clamped": limit_clamp,
        "finite_raw": bool(torch.isfinite(raw).all()),
        "finite_normalized": bool(torch.isfinite(normalized).all()),
        "done": bool(done),
        "failure_reason": info.get("failure_reason"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = PushConfig(max_steps=512, physics_steps_per_action=8)
    backend = IsaacGymPushBackend(config)
    environment = PushAdapter(config, backend=backend)
    training_normalizer = ObservationNormalizer()
    replay_trace = []
    random_trace = []
    try:
        environment.initialize()
        reset_observation = environment.reset()
        reset_raw = flatten_observation(reset_observation)
        reset_normalized = training_normalizer.normalize(reset_raw, update=True)
        initial_object = array(backend.read_object_state()[:3])
        goal = np.asarray(config.goal_position)
        direction = goal - initial_object
        direction[2] = 0.0
        direction /= np.linalg.norm(direction)
        initial_goal_distance = float(np.linalg.norm(goal - initial_object))
        first_contact = None
        increment = tuple(direction[:2] * ACTION_SCALES[0])
        # Same future production path as P1.1; no direct backend controller call.
        for index in range(408):
            if index < 8:
                stage, action = "settle", (0.0, 0.0)
            else:
                previous = replay_trace[-1]
                tracking = np.linalg.norm(previous["target_tracking_error_xy"])
                action = increment if tracking <= 0.010 else (0.0, 0.0)
                stage = "contact" if first_contact is None else "push"
            observation, row = sample(
                environment, backend, training_normalizer, action, True, stage,
                index, initial_object, goal, direction,
            )
            replay_trace.append(row)
            if row["contact"] and first_contact is None:
                first_contact = index
            if row["directional_displacement_m"] >= 0.012:
                break
            if row["done"] and row["failure_reason"]:
                break

        saved_state = training_normalizer.state_dict()
        evaluation_normalizer = ObservationNormalizer(frozen=True)
        evaluation_normalizer.load_state_dict(saved_state, frozen=True)
        freeze_before = evaluation_normalizer.state_dict()

        random.seed(12)
        environment.reset()
        random_initial_object = array(backend.read_object_state()[:3])
        for index in range(100):
            action = tuple(random.uniform(-bound, bound) for bound in ACTION_SCALES)
            observation, row = sample(
                environment, backend, evaluation_normalizer, action, False,
                "evaluation_random", index, random_initial_object, goal, direction,
            )
            random_trace.append(row)
            if row["done"]:
                break
        freeze_after = evaluation_normalizer.state_dict()
    finally:
        environment.close()

    final = replay_trace[-1]
    displacement = final["object_displacement_m"]
    directional = final["directional_displacement_m"]
    directional_fraction = directional / max(displacement, 1.0e-12)
    replay_clamps = {
        "joint_delta_clamp_ratio": sum(row["joint_delta_clamped"] for row in replay_trace) / len(replay_trace),
        "joint_limit_clamp_ratio": sum(row["joint_limit_clamped"] for row in replay_trace) / len(replay_trace),
    }
    random_clamps = {
        "joint_delta_clamp_ratio": sum(row["joint_delta_clamped"] for row in random_trace) / len(random_trace),
        "joint_limit_clamp_ratio": sum(row["joint_limit_clamped"] for row in random_trace) / len(random_trace),
    }
    replay_pass = bool(
        first_contact is not None
        and displacement >= 0.01
        and directional >= 0.01
        and directional_fraction >= 0.8
        and final["object_goal_distance_m"] < initial_goal_distance
        and max(replay_clamps.values()) <= 0.1
        and all(row["finite_raw"] and row["finite_normalized"] for row in replay_trace)
        and max(row["relative_consistency_error_m"] for row in replay_trace) < 1.0e-6
        and final["failure_reason"] is None
    )
    random_pass = bool(
        len(random_trace) == 100
        and max(random_clamps.values()) <= 0.1
        and all(row["finite_raw"] and row["finite_normalized"] for row in random_trace)
        and max(row["relative_consistency_error_m"] for row in random_trace) < 1.0e-6
        and all(row["failure_reason"] is None for row in random_trace)
    )
    freeze_pass = state_equal(freeze_before, freeze_after)
    result = {
        "gate": "P1.2 task-aligned observation and frozen evaluation normalization",
        "runtime": "Isaac Gym Preview 4, GPU PhysX, RTX 4080",
        "observation_dim": len(reset_raw),
        "normalization": {
            "epsilon": training_normalizer.epsilon,
            "variance_floor": training_normalizer.variance_floor,
            "clip": training_normalizer.clip,
            "training_final_count": training_normalizer.count,
            "evaluation_count_before": freeze_before["count"],
            "evaluation_count_after": freeze_after["count"],
            "evaluation_statistics_bitwise_identical": freeze_pass,
            "max_abs_normalized_replay": max(abs(value) for row in replay_trace for value in row["normalized_observation"]),
            "max_abs_normalized_random": max(abs(value) for row in random_trace for value in row["normalized_observation"]),
        },
        "reset": {
            "raw_observation": {name: list(values) for name, values in reset_observation.items()},
            "normalized_observation": reset_normalized.tolist(),
            "previous_action_reset_zero": tuple(reset_observation["previous_action_xy"]) == (0.0, 0.0),
            "tracking_error_reset_zero": max(abs(x) for x in reset_observation["target_tracking_error_xy"]) < 1.0e-6,
        },
        "production_action_replay": {
            "result": "PASS" if replay_pass else "FAIL",
            "first_contact_step": first_contact,
            "steps": len(replay_trace),
            "initial_object_goal_distance_m": initial_goal_distance,
            "final_object_goal_distance_m": final["object_goal_distance_m"],
            "object_displacement_m": displacement,
            "directional_displacement_m": directional,
            "directional_fraction": directional_fraction,
            "max_tracking_error_m": max(np.linalg.norm(row["target_tracking_error_xy"]) for row in replay_trace),
            "max_relative_consistency_error_m": max(row["relative_consistency_error_m"] for row in replay_trace),
            **replay_clamps,
            "trace": replay_trace,
        },
        "random_evaluation_sanity": {
            "result": "PASS" if random_pass else "FAIL",
            "steps": len(random_trace),
            "max_relative_consistency_error_m": max(row["relative_consistency_error_m"] for row in random_trace),
            **random_clamps,
            "trace": random_trace,
        },
        "freeze_test": {"result": "PASS" if freeze_pass else "FAIL"},
        "result": "PASS" if replay_pass and random_pass and freeze_pass else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("P1_2_OBSERVATION_NORMALIZATION_GATE=%s" % result["result"])
    print(json.dumps({
        "replay": {key: value for key, value in result["production_action_replay"].items() if key != "trace"},
        "random": {key: value for key, value in result["random_evaluation_sanity"].items() if key != "trace"},
        "normalization": result["normalization"],
        "freeze": result["freeze_test"],
    }, indent=2))


if __name__ == "__main__":
    main()
