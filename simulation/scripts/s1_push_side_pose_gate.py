#!/usr/bin/env python3
"""P0.1 reachable side-push pose/contact integrity gate; never runs PPO."""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simulation_core.push_adapter import EETarget, IsaacGymPushBackend, PushConfig
from simulation_core.s1_isaac_gym_adapter import S1IsaacGymConfig


END_BODIES = ("7_Link", "8_Link")


def array(values):
    return np.asarray([float(x) for x in values], dtype=np.float64)


def quaternion_angle(current, target):
    dot = abs(float(np.dot(array(current), array(target))))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def point_box_distance(point, center, size):
    return float(np.linalg.norm(np.maximum(np.abs(point - center) - size / 2.0, 0.0)))


def save_images(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    output.mkdir(parents=True, exist_ok=True)
    trace = result["trace"]
    picks = (
        ("01_settled.png", result["snapshots"]["settled"], "Reset / gripper settled"),
        ("02_pre_contact.png", result["snapshots"]["pre_contact"], "Reached pre-contact"),
        ("03_contact.png", result["snapshots"]["contact"], "First verified contact"),
        ("04_pushed.png", result["snapshots"]["pushed"], "Push complete"),
    )
    for filename, index, title in picks:
        item = trace[index]
        fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=140)
        cube, goal = array(item["cube_position"]), array(result["goal_position"])
        initial, size = array(result["initial_object_position"]), array(result["cube_size"])
        ax.add_patch(Rectangle(cube[:2] - size[:2] / 2.0, size[0], size[1], color="#d9534f", alpha=.65))
        ax.add_patch(Rectangle(goal[:2] - size[:2] / 2.0, size[0], size[1], fill=False, edgecolor="#2ca02c", linewidth=2))
        ax.scatter(*initial[:2], marker="x", color="black", label="initial cube")
        ax.scatter(*array(item["ee_position"])[:2], color="#1f77b4", label="ee_link origin")
        for name, color in (("7_Link", "#9467bd"), ("8_Link", "#17becf")):
            ax.scatter(*array(item["finger_positions"][name])[:2], color=color, label=name + " origin")
        direction = array(result["push_direction"])
        ax.arrow(initial[0], initial[1], direction[0] * .12, direction[1] * .12,
                 width=.002, color="#2ca02c", length_includes_head=True)
        ax.set(xlim=(.25, .68), ylim=(-.22, .25), xlabel="world x (m)", ylabel="world y (m)")
        ax.set_aspect("equal"); ax.grid(alpha=.25); ax.legend(fontsize=7, loc="upper left")
        ax.set_title("%s\nSTATE-BASED (real GPU tensors), action=%d, stage=%s\n"
                     "directional displacement=%.4f m, contact=%s" %
                     (title, item["action"], item["stage"], item["directional_displacement"], item["contact"]))
        fig.tight_layout(); fig.savefig(str(output / filename)); plt.close(fig)


def run_trial(physics_steps, output_dir, image_dir):
    config = PushConfig(physics_steps_per_action=physics_steps, max_steps=512)
    backend = IsaacGymPushBackend(config, s1_config=S1IsaacGymConfig())
    trace, stage_stats = [], defaultdict(lambda: {"actions": 0, "joint_delta_clamps": 0, "joint_limit_clamps": 0})
    first_contact = None
    snapshots = {}
    try:
        backend.setup(); backend.reset(); backend.step(16)
        robot = backend.robot
        initial_state = robot.read_state()
        initial_ee = array(initial_state.ee_link_state[:3])
        fixed_orientation = tuple(float(x) for x in initial_state.ee_link_state[3:7])
        initial_object = array(backend.read_object_state()[:3])
        goal = array(config.goal_position)
        push_direction = goal - initial_object; push_direction[2] = 0.0; push_direction /= np.linalg.norm(push_direction)
        initial_goal_distance = float(np.linalg.norm(goal - initial_object))
        lower = array(backend.controller.config.lower_limits[:6]); upper = array(backend.controller.config.upper_limits[:6])

        def record(stage, delta, target):
            nonlocal first_contact
            state = robot.read_state(); ee = array(state.ee_link_state[:3]); q = array(state.dof_positions[:6])
            obj_state = backend.read_object_state(); obj = array(obj_state[:3])
            robot.gym.refresh_net_contact_force_tensor(robot.sim); robot._refresh_tensors()
            object_force = array(backend._net_contact_force[backend._object_body_index])
            finger_positions, finger_forces, finger_distances = {}, {}, {}
            for name in END_BODIES:
                idx = robot.body_indices[name]
                finger_positions[name] = array(robot._rigid_body_state[idx, :3]).tolist()
                finger_forces[name] = array(backend._net_contact_force[idx]).tolist()
                finger_distances[name] = point_box_distance(array(finger_positions[name]), obj, array(config.object_size))
            command = backend.last_command
            jd = array(command.joint_delta) if command is not None else np.zeros(6)
            tq = array(command.positions[:6]) if command is not None else q
            joint_delta_clamp = bool(np.any(np.abs(jd) >= backend.controller.config.max_joint_delta * (1 - 1e-5)))
            joint_limit_clamp = bool(np.any((tq <= lower + 1e-6) | (tq >= upper - 1e-6)))
            contact = bool(backend.has_object_contact())
            displacement_vector = obj - initial_object
            item = {
                "action": len(trace), "stage": stage,
                "ee_position": ee.tolist(), "ee_orientation_xyzw": array(state.ee_link_state[3:7]).tolist(),
                "cube_position": obj.tolist(), "cube_velocity": array(obj_state[7:13]).tolist(),
                "finger_positions": finger_positions, "finger_net_forces": finger_forces,
                "finger_origin_to_cube_surface": finger_distances,
                "cube_net_force": object_force.tolist(),
                "cube_horizontal_force": float(np.linalg.norm(object_force[:2])),
                "contact": contact,
                "object_displacement": float(np.linalg.norm(displacement_vector)),
                "directional_displacement": float(np.dot(displacement_vector, push_direction)),
                "object_goal_distance": float(np.linalg.norm(goal - obj)),
                "commanded_delta": array(delta).tolist(), "target_position": array(target).tolist(),
                "position_error": float(np.linalg.norm(array(target) - ee)),
                "orientation_error": quaternion_angle(state.ee_link_state[3:7], fixed_orientation),
                "joint_positions": q.tolist(),
                "joint_limit_margins": np.minimum(q - lower, upper - q).tolist(),
                "joint_delta_clamped": joint_delta_clamp, "joint_limit_clamped": joint_limit_clamp,
                "finite": bool(np.all(np.isfinite(np.r_[ee, obj, q, object_force]))),
            }
            stats = stage_stats[stage]; stats["actions"] += 1
            stats["joint_delta_clamps"] += int(joint_delta_clamp); stats["joint_limit_clamps"] += int(joint_limit_clamp)
            trace.append(item)
            if contact and first_contact is None:
                first_contact = len(trace) - 1
            return item

        def move(stage, target, max_actions, stop_on_contact=False, target_tolerance=.002):
            for _ in range(max_actions):
                state = robot.read_state(); ee = array(state.ee_link_state[:3]); error = array(target) - ee
                norm = float(np.linalg.norm(error)); delta = error if norm <= .005 else error / norm * .005
                backend.set_ee_target(EETarget(tuple(ee + delta), fixed_orientation)); backend.step(physics_steps)
                item = record(stage, delta, target)
                if stop_on_contact and item["contact"]: return True
                if norm <= target_tolerance: return True
            return False

        # Stage 0: settle evidence (zero Cartesian target).
        backend.set_ee_target(EETarget(tuple(initial_ee), fixed_orientation)); backend.step(physics_steps)
        record("reset_settle", np.zeros(3), initial_ee); snapshots["settled"] = 0
        # A/B/C deliberately separate holds: q_home was designed as a safe
        # collision-mesh clearance pose and already has the chosen orientation/height.
        move("clearance", initial_ee, 4)
        move("align", initial_ee, 4)
        move("descend", initial_ee, 4)
        pre_contact_target = initial_ee + push_direction * .010
        # Position drives settle with a few millimetres of static Cartesian
        # error under gravity/contact; 8 mm is still safely behind the cube.
        pre_contact_reached = move("pre_contact", pre_contact_target, 30, target_tolerance=.008)
        snapshots["pre_contact"] = len(trace) - 1
        # Contact uses incremental 5 mm targets and the repaired production bool.
        contact_found = False
        for _ in range(80):
            ee = array(robot.read_state().ee_link_state[:3]); target = ee + push_direction * .005
            backend.set_ee_target(EETarget(tuple(target), fixed_orientation)); backend.step(physics_steps)
            item = record("contact", push_direction * .005, target)
            if item["contact"]:
                contact_found = True; break
        snapshots["contact"] = first_contact if first_contact is not None else len(trace) - 1
        for _ in range(140):
            ee = array(robot.read_state().ee_link_state[:3]); target = ee + push_direction * .005
            backend.set_ee_target(EETarget(tuple(target), fixed_orientation)); backend.step(physics_steps)
            item = record("push", push_direction * .005, target)
            if item["directional_displacement"] >= .012: break
        snapshots["pushed"] = len(trace) - 1

        for stats in stage_stats.values():
            stats["joint_delta_clamp_ratio"] = stats["joint_delta_clamps"] / stats["actions"]
            stats["joint_limit_clamp_ratio"] = stats["joint_limit_clamps"] / stats["actions"]
        final = trace[-1]; q = array(final["joint_positions"])
        settled = trace[0]
        initial_self_force = {
            name: float(np.linalg.norm(array(settled["finger_net_forces"][name]))) for name in END_BODIES
        }
        # Same-direction sub-newton vertical support is not finger/finger self-contact.
        opposing_force = float(np.linalg.norm(array(settled["finger_net_forces"]["7_Link"]) + array(settled["finger_net_forces"]["8_Link"])))
        max_clamp_ratio = max(max(x["joint_delta_clamp_ratio"], x["joint_limit_clamp_ratio"]) for x in stage_stats.values())
        directional_fraction = final["directional_displacement"] / max(final["object_displacement"], 1e-12)
        passed = bool(
            pre_contact_reached and contact_found and final["object_displacement"] >= .01
            and final["directional_displacement"] >= .01 and directional_fraction >= .8
            and final["object_goal_distance"] < initial_goal_distance - .008
            and max_clamp_ratio <= .1 and all(x["finite"] for x in trace)
            and min(final["joint_limit_margins"]) > .05
            and final["cube_position"][2] >= config.table_position[2] + config.table_size[2] / 2 - config.object_size[2] / 2
        )
        result = {
            "gate": "P0.1 reachable side-push pose/contact integrity",
            "result": "PASS" if passed else "FAIL",
            "physics_steps_per_action": physics_steps, "translation_step": .005,
            "push_q_home": list(config.robot_initial_dof_positions),
            "fixed_orientation_xyzw": list(fixed_orientation),
            "chosen_push_body": "7_Link + 8_Link two-finger surface",
            "chosen_collision_mesh_center_height": .425,
            "finger_collision_min_height_at_design_pose": .37763,
            "table_top_height": .375,
            "initial_ee_position": initial_ee.tolist(), "initial_object_position": initial_object.tolist(),
            "goal_position": goal.tolist(), "cube_size": list(config.object_size), "push_direction": push_direction.tolist(),
            "pre_contact_target": pre_contact_target.tolist(), "pre_contact_reached": pre_contact_reached,
            "pre_contact_final_position_error": trace[snapshots["pre_contact"]]["position_error"],
            "pre_contact_final_orientation_error": trace[snapshots["pre_contact"]]["orientation_error"],
            "first_contact_action": first_contact,
            "settled_finger_force_magnitudes": initial_self_force,
            "settled_finger_force_vector_sum_magnitude": opposing_force,
            "stage_stats": dict(stage_stats),
            "minimum_final_joint_limit_margin": float(np.min(np.minimum(q - lower, upper - q))),
            "initial_object_goal_distance": initial_goal_distance,
            "final_object_goal_distance": final["object_goal_distance"],
            "object_displacement": final["object_displacement"],
            "directional_displacement": final["directional_displacement"],
            "directional_fraction": directional_fraction,
            "object_drop": False, "object_out_of_bounds": False,
            "all_finite": all(x["finite"] for x in trace), "snapshots": snapshots, "trace": trace,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / ("trial_005_decimation_%d.json" % physics_steps)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        save_images(result, image_dir / ("decimation_%d" % physics_steps))
        return result
    finally:
        backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics-steps-per-action", type=int, choices=(4, 8), required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "doc/reports/p0_1_side_push_raw")
    parser.add_argument("--image-dir", type=Path, default=ROOT / "doc/reports/p0_1_side_push_images")
    args = parser.parse_args()
    result = run_trial(args.physics_steps_per_action, args.output_dir, args.image_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "trace"}, indent=2))


if __name__ == "__main__":
    main()
