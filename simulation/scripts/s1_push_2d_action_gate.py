#!/usr/bin/env python3
"""Real-GPU P1.1 gate through the exact production two-dimensional action path."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random

# Isaac Gym Preview 4 must load before modules which import torch.
from isaacgym import gymapi as _gymapi  # noqa: F401
import numpy as np

from simulation_core.cartesian_controller import S1_LOWER_LIMITS, S1_UPPER_LIMITS
from simulation_core.push_adapter import (
    ACTION_SCALES,
    PUSH_FIXED_ORIENTATION_XYZW,
    PUSH_FIXED_TARGET_Z,
    IsaacGymPushBackend,
    PushAdapter,
    PushConfig,
)


def quaternion_angle(left, right):
    dot = min(1.0, abs(float(np.dot(array(left), array(right)))))
    return 2.0 * math.acos(dot)


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value, dtype=float)


def clamp_flags(backend):
    command = backend.last_command
    if command is None:
        return False, False
    delta = np.asarray(command.joint_delta.detach().cpu(), dtype=float)
    targets = np.asarray(command.positions[:6].detach().cpu(), dtype=float)
    lower = np.asarray(S1_LOWER_LIMITS[:6])
    upper = np.asarray(S1_UPPER_LIMITS[:6])
    delta_clamp = bool(np.any(np.abs(delta) >= backend.controller.config.max_joint_delta * (1.0 - 1.0e-5)))
    limit_clamp = bool(np.any((targets <= lower + 1.0e-6) | (targets >= upper - 1.0e-6)))
    return delta_clamp, limit_clamp


def run_replay(environment, backend):
    observation = environment.reset()
    # Let the reset drives settle, while still going through PushAdapter.step.
    initial_object = np.asarray(backend.read_object_state()[:3], dtype=float)
    goal = np.asarray(environment.config.goal_position, dtype=float)
    direction = goal - initial_object
    direction[2] = 0.0
    direction /= np.linalg.norm(direction)
    initial_goal_distance = float(np.linalg.norm(goal - initial_object))
    trace = []
    stats = defaultdict(lambda: {"steps": 0, "joint_delta_clamps": 0, "joint_limit_clamps": 0})
    first_contact = None
    contact_samples = 0

    def take(stage, action):
        nonlocal first_contact, contact_samples
        obs, reward, done, info = environment.step(tuple(float(x) for x in action))
        state = backend.read_robot_state()
        ee = array(state.ee_link_state[:3])
        orientation = array(state.ee_link_state[3:7])
        obj = np.asarray(backend.read_object_state()[:3], dtype=float)
        displacement = obj - initial_object
        delta_clamp, limit_clamp = clamp_flags(backend)
        target = backend.last_target
        item = {
            "step": len(trace),
            "stage": stage,
            "action_xy": list(action),
            "target_position": list(target.position),
            "target_orientation_xyzw": list(target.orientation_xyzw),
            "ee_position": ee.tolist(),
            "ee_orientation_xyzw": orientation.tolist(),
            "target_position_error_m": float(np.linalg.norm(np.asarray(target.position) - ee)),
            "actual_z_error_m": float(ee[2] - PUSH_FIXED_TARGET_Z),
            "actual_orientation_error_rad": quaternion_angle(orientation, PUSH_FIXED_ORIENTATION_XYZW),
            "object_position": obj.tolist(),
            "object_displacement_m": float(np.linalg.norm(displacement)),
            "directional_displacement_m": float(np.dot(displacement, direction)),
            "object_goal_distance_m": float(np.linalg.norm(goal - obj)),
            "contact": bool(info["contact"]),
            "reward": float(reward),
            "finite": bool(np.all(np.isfinite(np.r_[ee, orientation, obj, reward]))),
            "joint_delta_clamped": delta_clamp,
            "joint_limit_clamped": limit_clamp,
            "done": bool(done),
            "failure_reason": info.get("failure_reason"),
        }
        stats[stage]["steps"] += 1
        stats[stage]["joint_delta_clamps"] += int(delta_clamp)
        stats[stage]["joint_limit_clamps"] += int(limit_clamp)
        if item["contact"]:
            contact_samples += 1
            if first_contact is None:
                first_contact = item["step"]
        trace.append(item)
        return item

    for _ in range(8):
        take("settle", (0.0, 0.0))

    # Increment the persistent production target by a bounded 5 mm vector.
    # Zero actions let the drive catch up; they do not bypass or rewrite target state.
    increment = direction[:2] * ACTION_SCALES[0]
    for _ in range(400):
        current = trace[-1]
        action = increment if current["target_position_error_m"] <= 0.010 else (0.0, 0.0)
        item = take("contact" if first_contact is None else "push", action)
        if item["directional_displacement_m"] >= 0.012:
            break
        if item["done"] and item["failure_reason"]:
            break

    for values in stats.values():
        values["joint_delta_clamp_ratio"] = values["joint_delta_clamps"] / values["steps"]
        values["joint_limit_clamp_ratio"] = values["joint_limit_clamps"] / values["steps"]
    final = trace[-1]
    directional_fraction = final["directional_displacement_m"] / max(final["object_displacement_m"], 1.0e-12)
    max_clamp_ratio = max(
        max(value["joint_delta_clamp_ratio"], value["joint_limit_clamp_ratio"])
        for name, value in stats.items() if name in ("contact", "push")
    )
    target_constraints_exact = all(
        item["target_position"][2] == PUSH_FIXED_TARGET_Z
        and tuple(item["target_orientation_xyzw"]) == PUSH_FIXED_ORIENTATION_XYZW
        for item in trace
    )
    passed = bool(
        first_contact is not None
        and final["object_displacement_m"] >= 0.01
        and final["directional_displacement_m"] >= 0.01
        and directional_fraction >= 0.8
        and final["object_goal_distance_m"] < initial_goal_distance
        and max_clamp_ratio <= 0.1
        and target_constraints_exact
        and all(item["finite"] for item in trace)
        and final["failure_reason"] is None
    )
    return {
        "result": "PASS" if passed else "FAIL",
        "action_path": "PushAdapter.step -> PushActionMapper -> IsaacGymPushBackend.set_ee_target -> DLS -> S1",
        "initial_object_position": initial_object.tolist(),
        "goal_position": goal.tolist(),
        "push_direction": direction.tolist(),
        "initial_object_goal_distance_m": initial_goal_distance,
        "final_object_goal_distance_m": final["object_goal_distance_m"],
        "object_displacement_m": final["object_displacement_m"],
        "directional_displacement_m": final["directional_displacement_m"],
        "directional_fraction": directional_fraction,
        "first_contact_step": first_contact,
        "contact_samples": contact_samples,
        "target_constraints_exact": target_constraints_exact,
        "max_actual_z_error_m": max(abs(item["actual_z_error_m"]) for item in trace),
        "max_actual_orientation_error_rad": max(item["actual_orientation_error_rad"] for item in trace),
        "stage_stats": dict(stats),
        "all_finite": all(item["finite"] for item in trace),
        "failure_reason": final["failure_reason"],
        "trace": trace,
    }


def run_random_sanity(environment, backend, steps=100):
    random.seed(11)
    environment.reset()
    records = []
    for index in range(steps):
        action = tuple(random.uniform(-bound, bound) for bound in ACTION_SCALES)
        observation, reward, done, info = environment.step(action)
        state = backend.read_robot_state()
        delta_clamp, limit_clamp = clamp_flags(backend)
        records.append({
            "step": index,
            "action_xy": list(action),
            "target_z": backend.last_target.position[2],
            "target_orientation_xyzw": list(backend.last_target.orientation_xyzw),
            "actual_z_error_m": float(state.ee_link_state[2] - PUSH_FIXED_TARGET_Z),
            "actual_orientation_error_rad": quaternion_angle(state.ee_link_state[3:7], PUSH_FIXED_ORIENTATION_XYZW),
            "joint_delta_clamped": delta_clamp,
            "joint_limit_clamped": limit_clamp,
            "finite": bool(
                math.isfinite(float(reward))
                and all(math.isfinite(float(value)) for field in observation.values() for value in field)
            ),
            "done": bool(done),
            "failure_reason": info.get("failure_reason"),
        })
        if done:
            break
    count = len(records)
    target_constraints_exact = all(
        row["target_z"] == PUSH_FIXED_TARGET_Z
        and tuple(row["target_orientation_xyzw"]) == PUSH_FIXED_ORIENTATION_XYZW
        for row in records
    )
    delta_ratio = sum(row["joint_delta_clamped"] for row in records) / count
    limit_ratio = sum(row["joint_limit_clamped"] for row in records) / count
    passed = bool(
        count == steps
        and target_constraints_exact
        and all(row["finite"] for row in records)
        and delta_ratio <= 0.1
        and limit_ratio <= 0.1
        and all(row["failure_reason"] is None for row in records)
    )
    return {
        "result": "PASS" if passed else "FAIL",
        "requested_steps": steps,
        "completed_steps": count,
        "target_constraints_exact": target_constraints_exact,
        "joint_delta_clamp_ratio": delta_ratio,
        "joint_limit_clamp_ratio": limit_ratio,
        "max_actual_z_error_m": max(abs(row["actual_z_error_m"]) for row in records),
        "max_actual_orientation_error_rad": max(row["actual_orientation_error_rad"] for row in records),
        "all_finite": all(row["finite"] for row in records),
        "records": records,
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
        replay = run_replay(environment, backend)
        random_sanity = run_random_sanity(environment, backend)
    finally:
        environment.close()
    result = {
        "gate": "P1.1 constrained 2D Push action",
        "action_dim": 2,
        "action_bounds_m": [list(-np.asarray(ACTION_SCALES)), list(ACTION_SCALES)],
        "fixed_policy_target_z_m": PUSH_FIXED_TARGET_Z,
        "fixed_orientation_xyzw": list(PUSH_FIXED_ORIENTATION_XYZW),
        "physics_steps_per_action": 8,
        "production_action_replay": replay,
        "random_action_sanity": random_sanity,
        "result": "PASS" if replay["result"] == random_sanity["result"] == "PASS" else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("P1_1_CONSTRAINED_ACTION_GATE=%s" % result["result"])
    print(json.dumps({
        "replay": {key: value for key, value in replay.items() if key != "trace"},
        "random": {key: value for key, value in random_sanity.items() if key != "records"},
    }, indent=2))


if __name__ == "__main__":
    main()
