#!/usr/bin/env python3
"""Scripted S1 contact/push reachability gate (no PPO or platform task)."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simulation_core.push_adapter import EETarget, IsaacGymPushBackend, PushConfig
from simulation_core.s1_isaac_gym_adapter import S1IsaacGymConfig


END_BODIES = ("ee_link", "7_Link", "8_Link")


def vec(values):
    return np.asarray([float(x) for x in values], dtype=np.float64)


def point_box_distance(point, center, size):
    outside = np.maximum(np.abs(point - center) - size / 2.0, 0.0)
    return float(np.linalg.norm(outside))


def save_state_images(result, image_dir):
    """Save explicitly labelled state-based evidence; no graphics context needed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    image_dir.mkdir(parents=True, exist_ok=True)
    trace = result["trace"]
    geometry = result["geometry"]
    indices = result["snapshots"]
    selections = (
        ("01_initial.png", indices["initial"], "Initial state"),
        ("02_pre_contact.png", indices["pre_contact"], "End of scripted approach"),
        ("03_contact.png", indices.get("real_contact", indices["final"]), "First horizontal cube-force event"),
        ("04_final.png", indices["final"], "Final state"),
    )
    for filename, index, title in selections:
        item = trace[index]
        fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=140)
        cube = vec(item["cube_position"])
        initial = vec(result["initial_object_position"])
        goal = vec(geometry["goal_position"])
        ee = vec(item["ee_position"])
        size = vec(geometry["cube_size"])
        ax.add_patch(Rectangle(cube[:2] - size[:2] / 2, size[0], size[1], color="#d9534f", alpha=.65))
        ax.add_patch(Rectangle(goal[:2] - size[:2] / 2, size[0], size[1], fill=False, edgecolor="#2ca02c", linewidth=2))
        ax.scatter(*initial[:2], marker="x", color="black", label="initial cube")
        ax.scatter(*ee[:2], s=55, color="#1f77b4", label="ee_link origin")
        for name, color in (("7_Link", "#9467bd"), ("8_Link", "#17becf")):
            pos = vec(item["body_diagnostics"][name]["position"])
            ax.scatter(*pos[:2], s=38, color=color, label=name + " origin")
        direction = vec(geometry["push_direction"])
        ax.arrow(initial[0], initial[1], direction[0] * .12, direction[1] * .12,
                 width=.002, color="#2ca02c", length_includes_head=True)
        ax.set(xlim=(.20, .70), ylim=(-.25, .30), xlabel="world x (m)", ylabel="world y (m)")
        ax.set_aspect("equal")
        ax.grid(alpha=.25)
        ax.legend(fontsize=7, loc="upper left")
        ax.set_title("%s\nSTATE-BASED (not Isaac Gym RGB), step=%d, stage=%s\n"
                     "cube displacement=%.4f m, physical_contact=%s" %
                     (title, item["step"], item["stage"], item["object_displacement"], item["physical_contact"]))
        fig.tight_layout()
        fig.savefig(str(image_dir / filename))
        plt.close(fig)


def body_snapshot(backend):
    robot = backend.robot
    robot._refresh_tensors()
    backend.robot.gym.refresh_net_contact_force_tensor(robot.sim)
    result = {}
    for name in END_BODIES:
        index = robot.body_indices[name]
        state = robot._rigid_body_state[index].detach().cpu().tolist()
        force = backend._net_contact_force[index].detach().cpu().tolist()
        result[name] = {
            "position": [float(x) for x in state[:3]],
            "orientation_xyzw": [float(x) for x in state[3:7]],
            "net_contact_force": [float(x) for x in force],
            "net_contact_force_magnitude": float(np.linalg.norm(force)),
        }
    obj_force = backend._net_contact_force[backend._object_body_index].detach().cpu().tolist()
    result["push_object"] = {
        "net_contact_force": [float(x) for x in obj_force],
        "net_contact_force_magnitude": float(np.linalg.norm(obj_force)),
    }
    return result


def command_target(backend, position, orientation, physics_steps):
    applied = backend.set_ee_target(EETarget(tuple(position), tuple(orientation)))
    command = backend.last_command
    raw_delta = vec(command.joint_delta.detach().cpu().tolist())
    q = vec(backend.robot.read_state().dof_positions.detach().cpu().tolist())
    target_q = vec(command.positions.detach().cpu().tolist())
    clamp = backend.controller.config.max_joint_delta
    delta_clamped = bool(np.any(np.abs(raw_delta) >= clamp * (1.0 - 1.0e-5)))
    lower = vec(backend.controller.config.lower_limits)
    upper = vec(backend.controller.config.upper_limits)
    limit_clamped = bool(np.any((target_q <= lower + 1.0e-6) | (target_q >= upper - 1.0e-6)))
    finite = bool(np.all(np.isfinite(raw_delta)) and np.all(np.isfinite(target_q)))
    backend.step(physics_steps)
    return bool(applied), delta_clamped, limit_clamped, finite, q


def run_trial(translation_step, physics_steps, output_dir):
    config = PushConfig(physics_steps_per_action=physics_steps, max_steps=2000)
    s1_config = S1IsaacGymConfig(graphics_device_id=-1, headless=True)
    backend = IsaacGymPushBackend(config, s1_config=s1_config)
    trace = []
    snapshots = {}
    first_contact = None
    contact_bodies = set()
    max_force = {name: 0.0 for name in END_BODIES}
    persistent_joint_clamps = 0
    limit_clamps = 0
    all_finite = True
    try:
        backend.setup()
        backend.reset()
        backend.step(8)
        initial_robot = backend.read_robot_state()
        initial_object = vec(backend.read_object_state()[:3])
        goal = vec(config.goal_position)
        initial_ee = vec(initial_robot.ee_link_state[:3])
        orientation = tuple(float(x) for x in initial_robot.ee_link_state[3:7])
        push_direction = goal - initial_object
        push_direction[2] = 0.0
        push_direction /= np.linalg.norm(push_direction)
        horizontal_support = float(
            abs(push_direction[0]) * config.object_size[0] / 2.0
            + abs(push_direction[1]) * config.object_size[1] / 2.0
        )
        # With the reset orientation, 7/8_Link collision meshes extend roughly
        # 8 cm along local -Z.  Their initial world origins show ee_link is the
        # midpoint between the two finger link frames.  Keep this orientation
        # fixed and place ee_link above the cube so the finger meshes sweep its side.
        push_height = float(config.object_start[2] + 0.080)
        pre_contact = initial_object.copy()
        pre_contact[:2] -= push_direction[:2] * (horizontal_support + 0.080)
        pre_contact[2] = push_height
        contact_target = initial_object.copy()
        contact_target[:2] -= push_direction[:2] * max(horizontal_support - 0.012, 0.0)
        contact_target[2] = push_height
        push_target = contact_target + push_direction * 0.16
        push_target[2] = push_height

        geometry = {
            "robot_root_position": list(config.robot_position),
            "robot_root_yaw_rad": config.robot_yaw,
            "initial_arm_joint_positions": [float(x) for x in initial_robot.dof_positions[:6]],
            "initial_ee_position": initial_ee.tolist(),
            "initial_ee_orientation_xyzw": list(orientation),
            "cube_center": initial_object.tolist(),
            "cube_size": list(config.object_size),
            "goal_position": goal.tolist(),
            "table_top_height": config.table_position[2] + config.table_size[2] / 2.0,
            "initial_ee_to_object": (initial_object - initial_ee).tolist(),
            "push_direction": push_direction.tolist(),
            "cube_support_distance_along_push": horizontal_support,
            "pre_contact_position": pre_contact.tolist(),
            "initial_to_pre_contact_distance": float(np.linalg.norm(pre_contact - initial_ee)),
            "fixed_orientation_xyzw": list(orientation),
            "fixed_orientation_euler_xyz_rad": [-0.785788, -0.002382, -1.571110],
            "push_height_ee_link_z": push_height,
        }

        global_step = 0
        initial_goal_distance = float(np.linalg.norm(goal - initial_object))

        def record(stage, commanded_delta, applied, joint_clamp, limit_clamp, finite):
            nonlocal global_step, first_contact, persistent_joint_clamps, limit_clamps, all_finite
            state = backend.read_robot_state()
            ee = vec(state.ee_link_state[:3])
            obj = vec(backend.read_object_state()[:3])
            bodies = body_snapshot(backend)
            distances = {
                name: point_box_distance(vec(bodies[name]["position"]), obj, vec(config.object_size))
                for name in END_BODIES
            }
            near_and_forced = []
            for name in END_BODIES:
                force = bodies[name]["net_contact_force_magnitude"]
                max_force[name] = max(max_force[name], force)
                # Body origins are not collision surfaces.  A generous 10 cm
                # proximity gate covers the URDF finger mesh's 8 cm offset,
                # while object net force and displacement provide independent proof.
                if force >= config.contact_force_threshold and distances[name] <= 0.10:
                    near_and_forced.append(name)
                    contact_bodies.add(name)
            object_force = vec(bodies["push_object"]["net_contact_force"])
            # The cube always has a vertical table-support force.  A nonzero
            # horizontal cube force plus nearby finger geometry is the useful
            # net-tensor evidence of a manipulator/object contact.  This also
            # rejects the large equal-and-opposite 7/8_Link self-contact forces.
            physical_contact = bool(
                near_and_forced and np.linalg.norm(object_force[:2])
                >= config.contact_force_threshold
            )
            if physical_contact and first_contact is None:
                first_contact = global_step
                snapshots["real_contact"] = len(trace)
            persistent_joint_clamps += int(joint_clamp)
            limit_clamps += int(limit_clamp)
            all_finite = bool(
                all_finite and finite
                and np.all(np.isfinite(ee)) and np.all(np.isfinite(obj))
            )
            item = {
                "step": global_step,
                "stage": stage,
                "ee_position": ee.tolist(),
                "cube_position": obj.tolist(),
                "ee_to_cube_center_distance": float(np.linalg.norm(ee - obj)),
                "estimated_surface_distances": distances,
                "minimum_estimated_surface_distance": min(distances.values()),
                "production_contact_bool": bool(backend.has_object_contact()),
                "physical_contact": physical_contact,
                "contact_bodies": near_and_forced,
                "body_diagnostics": bodies,
                "object_displacement": float(np.linalg.norm(obj - initial_object)),
                "object_to_goal_distance": float(np.linalg.norm(goal - obj)),
                "commanded_cartesian_delta": list(commanded_delta),
                "controller_applied": applied,
                "joint_delta_clamped": joint_clamp,
                "joint_limit_clamped": limit_clamp,
                "finite": finite,
            }
            trace.append(item)
            global_step += 1

        snapshots["initial"] = 0
        approach_clearance = pre_contact.copy()
        approach_clearance[2] = max(pre_contact[2] + 0.16, 0.65)
        stages = (
            ("approach_clearance", approach_clearance, 260),
            ("approach", pre_contact, 260),
            ("contact", contact_target, 140),
            ("push", push_target, 180),
        )
        for stage, target, max_actions in stages:
            if stage == "contact":
                snapshots["pre_contact"] = max(0, len(trace) - 1)
            for _ in range(max_actions):
                ee = vec(backend.read_robot_state().ee_link_state[:3])
                error = target - ee
                distance = float(np.linalg.norm(error))
                if distance > translation_step:
                    delta = error / distance * translation_step
                else:
                    delta = error
                next_position = ee + delta
                applied, joint_clamp, limit_clamp, finite, _ = command_target(
                    backend, next_position, orientation, physics_steps
                )
                record(stage, delta, applied, joint_clamp, limit_clamp, finite)
                if stage.startswith("approach") and distance <= 0.004:
                    break
                if stage == "contact" and trace[-1]["physical_contact"]:
                    break
                if stage == "push" and trace[-1]["object_displacement"] >= 0.012:
                    break
                if not all_finite:
                    break
            if not all_finite:
                break

        snapshots["final"] = len(trace) - 1
        final_object = vec(backend.read_object_state()[:3])
        displacement_vector = final_object - initial_object
        displacement = float(np.linalg.norm(displacement_vector))
        directional_displacement = float(np.dot(displacement_vector, push_direction))
        final_goal_distance = float(np.linalg.norm(goal - final_object))
        dropped = bool(final_object[2] < config.table_position[2] + config.table_size[2] / 2.0 - config.object_size[2] / 2.0)
        oob = bool(any(abs(final_object[i] - config.table_position[i]) > config.table_size[i] / 2.0 for i in (0, 1)))
        reached_precontact = any(x["stage"] == "approach" and np.linalg.norm(vec(x["ee_position"]) - pre_contact) <= 0.015 for x in trace)
        passed = bool(reached_precontact and first_contact is not None and displacement >= 0.01 and directional_displacement > 0 and not dropped and not oob and all_finite and persistent_joint_clamps < max(10, len(trace) // 2))
        result = {
            "translation_step": translation_step,
            "physics_steps_per_action": physics_steps,
            "geometry": geometry,
            "first_contact_step": first_contact,
            "max_contact_force_by_body": max_force,
            "contact_bodies": sorted(contact_bodies),
            "initial_object_position": initial_object.tolist(),
            "final_object_position": final_object.tolist(),
            "object_displacement": displacement,
            "directional_displacement": directional_displacement,
            "initial_object_goal_distance": initial_goal_distance,
            "final_object_goal_distance": final_goal_distance,
            "minimum_surface_distance": min(x["minimum_estimated_surface_distance"] for x in trace),
            "reached_pre_contact": bool(reached_precontact),
            "joint_delta_clamp_steps": persistent_joint_clamps,
            "joint_limit_clamp_steps": limit_clamps,
            "all_finite": all_finite,
            "object_drop": dropped,
            "object_out_of_bounds": oob,
            "pass": passed,
            "snapshots": snapshots,
            "trace": trace,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        trial_path = output_dir / ("trial_step_%03d_decimation_%d.json" % (round(translation_step * 1000), physics_steps))
        trial_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        save_state_images(result, ROOT / "doc/reports/p0_push_gate_images")
        return result
    finally:
        backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translation-step", type=float, required=True)
    parser.add_argument("--physics-steps-per-action", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "doc/reports/p0_push_gate_raw")
    args = parser.parse_args()
    if args.translation_step not in (0.005, 0.010, 0.020):
        raise SystemExit("translation step must be 0.005, 0.010, or 0.020 m")
    if args.physics_steps_per_action not in (2, 4, 8):
        raise SystemExit("physics steps/action must be 2, 4, or 8")
    result = run_trial(args.translation_step, args.physics_steps_per_action, args.output_dir)
    summary = {key: value for key, value in result.items() if key != "trace"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
