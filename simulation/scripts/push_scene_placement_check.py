#!/usr/bin/env python3
"""GPU Push placement diagnostic and fixed evidence capture."""

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simulation_core.isaac_gym_camera import IsaacGymCameraConfig  # noqa: E402
from simulation_core.push_adapter import EETarget, IsaacGymPushBackend, PushConfig  # noqa: E402
from simulation_core.s1_isaac_gym_adapter import S1IsaacGymConfig  # noqa: E402


def quat_matrix(q):
    return trimesh.transformations.quaternion_matrix([q[3], q[0], q[1], q[2]])


def main():
    os.environ.pop("DISPLAY", None)
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    output = ROOT / "doc" / "arm_check_images"
    output.mkdir(parents=True, exist_ok=True)
    config = PushConfig()
    camera_config = IsaacGymCameraConfig()
    backend = IsaacGymPushBackend(
        config,
        s1_config=S1IsaacGymConfig(graphics_device_id=0, headless=True),
        camera_config=camera_config,
    )
    try:
        backend.setup()
        backend.reset()
        backend.step(5)
        robot = backend.robot
        robot._refresh_tensors()
        gym, api, env = robot.gym, robot._gymapi, robot.env
        robot_actor_index = gym.get_actor_index(env, robot.actor, api.IndexDomain.DOMAIN_SIM)
        gym.refresh_actor_root_state_tensor(robot.sim)
        root = backend._root_state[robot_actor_index, :7].detach().cpu().tolist()
        names = list(gym.get_asset_rigid_body_names(robot.asset))
        poses = {}
        mesh_checks = {}
        table_min = np.asarray(config.table_position) - np.asarray(config.table_size) / 2
        table_max = np.asarray(config.table_position) + np.asarray(config.table_size) / 2
        asset_root = robot.config.asset_root
        for name in names:
            index = gym.find_actor_rigid_body_index(env, robot.actor, name, api.IndexDomain.DOMAIN_SIM)
            pose = robot._rigid_body_state[index, :7].detach().cpu().numpy()
            poses[name] = pose.tolist()
            mesh_name = "base_link.STL" if name == "base_link" else f"{name}.STL"
            mesh_path = asset_root / "meshes" / mesh_name
            if name in ("base_link", "1_Link", "2_Link", "3_Link") and mesh_path.exists():
                mesh = trimesh.load_mesh(mesh_path, process=False)
                transform = quat_matrix(pose[3:7])
                transform[:3, 3] = pose[:3]
                vertices = trimesh.transform_points(mesh.vertices, transform)
                bounds = np.asarray([vertices.min(axis=0), vertices.max(axis=0)])
                # Exclude the intended base/table contact plane and numerical
                # noise; report only vertices strictly inside the box volume.
                epsilon = 1.0e-6
                inside = np.all(
                    (vertices > table_min + epsilon) & (vertices < table_max - epsilon),
                    axis=1,
                )
                overlap = np.maximum(0.0, np.minimum(bounds[1], table_max) - np.maximum(bounds[0], table_min))
                mesh_checks[name] = {
                    "world_aabb": bounds.tolist(),
                    "table_aabb_overlap_xyz": overlap.tolist(),
                    "mesh_vertices_inside_table": int(inside.sum()),
                }

        result = {
            "robot_root_pose_xyzw": root,
            "robot_yaw_rad": config.robot_yaw,
            "table_pose_xyz": list(config.table_position),
            "table_size_xyz": list(config.table_size),
            "table_world_aabb": [table_min.tolist(), table_max.tolist()],
            "table_top_z": float(table_max[2]),
            "object_start_xyz": list(config.object_start),
            "goal_xyz": list(config.goal_position),
            "camera_position": list(camera_config.position),
            "camera_target": list(camera_config.target),
            "rigid_body_pose_xyzw": poses,
            "lower_mesh_table_checks": mesh_checks,
        }

        frame = backend.capture_camera_frame()
        Image.fromarray(frame.data, mode="RGBA").save(output / "push_scene_yaw_aligned.png")

        initial_ee = np.asarray(robot.read_state().ee_link_state[:3].detach().cpu().tolist())
        work_target = np.asarray(
            [config.object_start[0], config.object_start[1], config.object_start[2] + 0.12]
        )
        initial_orientation = tuple(
            float(x) for x in robot.read_state().ee_link_state[3:7].detach().cpu().tolist()
        )
        initial_distance = float(np.linalg.norm(initial_ee - work_target))
        target = EETarget(tuple(work_target), initial_orientation)
        for _ in range(120):
            backend.set_ee_target(target)
            backend.step(4)
        final_ee = np.asarray(robot.read_state().ee_link_state[:3].detach().cpu().tolist())
        result["cube_work_target_xyz"] = work_target.tolist()
        result["initial_ee_xyz"] = initial_ee.tolist()
        result["initial_ee_to_work_target_distance"] = initial_distance
        result["final_ee_xyz"] = final_ee.tolist()
        result["final_ee_to_work_target_distance"] = float(np.linalg.norm(final_ee - work_target))
        result["initial_horizontal_ee_vector_from_root"] = (initial_ee[:2] - np.asarray(root[:2])).tolist()
        result["cube_horizontal_vector_from_root"] = (
            np.asarray(config.object_start[:2]) - np.asarray(root[:2])
        ).tolist()
        backend.reset()
        (output / "push_scene_placement.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

        properties = api.CameraProperties()
        properties.width, properties.height = 800, 600
        side = gym.create_camera_sensor(env, properties)
        gym.set_camera_location(side, env, api.Vec3(-0.25, -1.15, 0.50), api.Vec3(0.25, 0.0, 0.32))
        gym.step_graphics(robot.sim)
        gym.render_all_camera_sensors(robot.sim)
        rgba = gym.get_camera_image(robot.sim, env, side, api.IMAGE_COLOR)
        rgba = np.asarray(rgba).reshape(properties.height, properties.width, 4).astype(np.uint8)
        Image.fromarray(rgba, mode="RGBA").save(output / "push_scene_side.png")
        print(json.dumps(result, indent=2))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
