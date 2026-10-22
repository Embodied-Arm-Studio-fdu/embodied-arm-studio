#!/usr/bin/env python3
"""Independent S1 asset/DOF motion check with fixed-view camera evidence.

This diagnostic intentionally uses only the public S1 adapter lifecycle.  It
does not import or modify Push, PPO, queue, streaming, or service code.
"""

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from simulation_core.s1_isaac_gym_adapter import (  # noqa: E402
    DEFAULT_INITIAL_DOF_POSITIONS,
    S1IsaacGymAdapter,
    S1IsaacGymConfig,
)


def parse_urdf(path):
    root = ET.parse(str(path)).getroot()
    rows = []
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        axis = joint.find("axis")
        limit = joint.find("limit")
        origin = joint.find("origin")
        rows.append(
            {
                "name": joint.get("name"),
                "type": joint.get("type"),
                "parent": parent.get("link") if parent is not None else None,
                "child": child.get("link") if child is not None else None,
                "axis": axis.get("xyz") if axis is not None else None,
                "lower": float(limit.get("lower")) if limit is not None and limit.get("lower") else None,
                "upper": float(limit.get("upper")) if limit is not None and limit.get("upper") else None,
                "effort": float(limit.get("effort")) if limit is not None and limit.get("effort") else None,
                "velocity": float(limit.get("velocity")) if limit is not None and limit.get("velocity") else None,
                "origin_xyz": origin.get("xyz", "0 0 0") if origin is not None else "0 0 0",
                "origin_rpy": origin.get("rpy", "0 0 0") if origin is not None else "0 0 0",
            }
        )
    return rows


def label(image, title, details):
    canvas = Image.new("RGB", (image.width, image.height + 68), "white")
    canvas.paste(image, (0, 68))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, fill="black")
    draw.text((12, 34), details, fill="black")
    return canvas


def grid(images):
    w = max(im.width for im in images)
    h = max(im.height for im in images)
    canvas = Image.new("RGB", (2 * w, 2 * h), "white")
    for i, im in enumerate(images):
        canvas.paste(im, ((i % 2) * w, (i // 2) * h))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "doc" / "arm_check_images")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cfg = S1IsaacGymConfig(graphics_device_id=0, headless=True)
    urdf_rows = parse_urdf(cfg.urdf_path)
    result = {
        "asset_root": str(cfg.asset_root.resolve()),
        "robot_urdf": str(cfg.robot_urdf),
        "urdf_path": str(cfg.urdf_path.resolve()),
        "fix_base_link": True,
        "urdf_joints": urdf_rows,
        "initial_dof_positions": list(DEFAULT_INITIAL_DOF_POSITIONS),
        "tests": [],
    }

    robot = S1IsaacGymAdapter(cfg)
    try:
        robot.setup()
        gym, sim, env, actor, api = robot.gym, robot.sim, robot.env, robot.actor, robot._gymapi
        result["isaac_asset_dof_names"] = list(gym.get_asset_dof_names(robot.asset))
        props = gym.get_actor_dof_properties(env, actor)
        result["isaac_dof_properties"] = [
            {
                "name": result["isaac_asset_dof_names"][i],
                "type": int(gym.get_asset_dof_type(robot.asset, i)),
                "lower": float(props["lower"][i]),
                "upper": float(props["upper"][i]),
                "effort": float(props["effort"][i]),
                "velocity": float(props["velocity"][i]),
            }
            for i in range(len(props))
        ]
        result["rigid_body_names"] = list(gym.get_asset_rigid_body_names(robot.asset))

        camera_props = api.CameraProperties()
        camera_props.width = 800
        camera_props.height = 650
        camera_props.enable_tensors = False
        camera = gym.create_camera_sensor(env, camera_props)
        gym.set_camera_location(camera, env, api.Vec3(0.95, -1.15, 0.72), api.Vec3(0.12, 0.05, 0.30))

        child_body = {row["name"]: row["child"] for row in urdf_rows}

        def settle(target, steps=150):
            robot.set_joint_position_targets(target)
            robot.step(steps)
            state = robot.read_state()
            return [float(x) for x in state.dof_positions.cpu().tolist()]

        def body_pose(name):
            robot._refresh_tensors()
            idx = gym.find_actor_rigid_body_index(env, actor, name, api.IndexDomain.DOMAIN_SIM)
            return [float(value) for value in robot._rigid_body_state[idx, :7].detach().cpu().tolist()]

        def shot(title, details):
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
            rgba = gym.get_camera_image(sim, env, camera, api.IMAGE_COLOR)
            rgba = np.asarray(rgba).reshape(camera_props.height, camera_props.width, 4)
            return label(Image.fromarray(rgba[:, :, :3].astype(np.uint8)), title, details)

        home = list(DEFAULT_INITIAL_DOF_POSITIONS)
        achieved = settle(home)
        home_image = shot("S1 safe initial pose", "q=[0, 45, 90, 0, 0, 0] deg; fixed base")
        home_image.save(args.output / "01_base_and_initial_pose.png")

        gym.set_camera_location(camera, env, api.Vec3(0.32, -0.36, 0.20), api.Vec3(0.0, 0.0, 0.035))
        shot("Base close-up", "black base_link is fixed; gray 1_Link rotates at 1joint about local +Z").save(
            args.output / "06_base_closeup.png"
        )
        gym.set_camera_location(camera, env, api.Vec3(0.95, -1.15, 0.72), api.Vec3(0.12, 0.05, 0.30))

        group_images = {}
        signs = [1, 1, -1, 1, 1, 1]
        amplitudes = [30, 25, 25, 25, 25, 25]
        for i in range(6):
            robot.reset_dofs()
            robot.step(2)
            target = home.copy()
            target[i] += signs[i] * math.radians(amplitudes[i])
            before = body_pose(child_body[f"{i + 1}joint"])
            actual = settle(target)
            after = body_pose(child_body[f"{i + 1}joint"])
            delta = actual[i] - achieved[i]
            pose_delta = float(np.linalg.norm(np.asarray(after) - np.asarray(before)))
            result["tests"].append(
                {
                    "joint": f"{i + 1}joint",
                    "commanded_rad": target[i],
                    "achieved_rad": actual[i],
                    "delta_from_home_rad": delta,
                    "child_link": child_body[f"{i + 1}joint"],
                    "child_pose_vector_delta": pose_delta,
                    "other_joint_max_error_rad": max(abs(actual[k] - home[k]) for k in range(6) if k != i),
                }
            )
            moved = shot(
                f"J{i + 1} only: {math.degrees(delta):+.1f} deg achieved",
                f"child={child_body[f'{i + 1}joint']}; all other arm targets held at home",
            )
            group = (i // 2) + 1
            group_images.setdefault(group, []).extend([home_image.copy(), moved])
            robot.reset_dofs()

        names = {1: "02_j1_j2_motion.png", 2: "03_j3_j4_motion.png", 3: "04_j5_j6_motion.png"}
        for group, images in group_images.items():
            grid(images).save(args.output / names[group])

        robot.reset_dofs()
        closed = home.copy()
        closed[6:] = [0.0, 0.0]
        actual_closed = settle(closed)
        closed_image = shot("Gripper closed target", "7joint=0 mm, 8joint=0 mm")
        opened = home.copy()
        opened[6:] = [0.04, 0.04]
        actual_open = settle(opened)
        open_image = shot("Gripper open target", "7joint=40 mm, 8joint=40 mm")
        # Also place the two prismatic coordinates directly at 40 mm.  This
        # distinguishes geometry/kinematic existence from position-drive motion.
        robot._dof_state[0, :, 0] = robot._dof_state.new_tensor(opened)
        robot._dof_state[0, :, 1] = 0.0
        gym.set_dof_state_tensor(sim, robot._gymtorch.unwrap_tensor(robot._dof_state))
        robot._refresh_tensors()
        direct_open = robot.read_state().dof_positions.cpu().tolist()
        direct_open_image = shot("Gripper direct-state open check", "both prismatic DOFs placed at 40 mm")
        result["gripper_test"] = {
            "closed_achieved": actual_closed[6:],
            "position_drive_open_achieved": actual_open[6:],
            "direct_state_open_achieved": [float(x) for x in direct_open[6:]],
        }
        side = Image.new("RGB", (closed_image.width * 3, closed_image.height), "white")
        side.paste(closed_image, (0, 0))
        side.paste(open_image, (closed_image.width, 0))
        side.paste(direct_open_image, (closed_image.width * 2, 0))
        side.save(args.output / "05_gripper_open_close.png")

        robot.reset_dofs()
        result["restored_dof_positions"] = [
            float(x) for x in robot.read_state().dof_positions.cpu().tolist()
        ]
        with (args.output / "runtime_results.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    finally:
        robot.close()


if __name__ == "__main__":
    main()
