"""Experimental one-S1/one-table/one-cube scene for P1 only."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional, Tuple

from datetime import datetime, timezone

import numpy as np

from ...isaac_gym_camera import (
    CameraFrame, IsaacGymCameraConfig, RGBA8,
)
from ...push_adapter import PUSH_FIXED_ORIENTATION_XYZW, PUSH_INITIAL_DOF_POSITIONS
from ...s1_isaac_gym_adapter import S1IsaacGymAdapter, S1IsaacGymConfig


@dataclass(frozen=True)
class ManipulationSceneConfig:
    table_size: Tuple[float, float, float] = (0.8, 0.8, 0.05)
    table_position: Tuple[float, float, float] = (0.45, 0.0, 0.35)
    robot_position: Tuple[float, float, float] = (0.10, 0.0, 0.40925)
    robot_yaw: float = math.pi
    cube_size: float = 0.045
    cube_density: float = 300.0
    cube_friction: float = 1.0
    cube_restitution: float = 0.0
    table_friction: float = 1.0
    table_restitution: float = 0.0
    finger_friction: float = 1.5
    robot_body_friction: float = 0.1
    robot_restitution: float = 0.0
    gripper_stiffness: float = 300.0
    gripper_damping: float = 20.0
    cube_xy: Tuple[float, float] = (0.45, 0.0)
    initial_open: float = 0.05
    graphics_device_id: int = -1
    with_cube: bool = True
    with_camera: bool = False
    fixed_orientation_xyzw: Tuple[float, float, float, float] = PUSH_FIXED_ORIENTATION_XYZW

    def __post_init__(self):
        for name in ("table_size", "table_position", "robot_position"):
            value = tuple(float(v) for v in getattr(self, name))
            if len(value) != 3 or not all(math.isfinite(v) for v in value):
                raise ValueError("%s must contain three finite values" % name)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "cube_xy", tuple(float(v) for v in self.cube_xy))
        object.__setattr__(self, "fixed_orientation_xyzw",
                           tuple(float(v) for v in self.fixed_orientation_xyzw))
        if len(self.cube_xy) != 2 or len(self.fixed_orientation_xyzw) != 4:
            raise ValueError("cube_xy/orientation dimensions are invalid")
        positive = ("cube_size", "cube_density", "cube_friction", "table_friction",
                    "finger_friction")
        if any(not math.isfinite(getattr(self, n)) or getattr(self, n) <= 0 for n in positive):
            raise ValueError("cube geometry/density and friction values must be positive")
        for name in ("cube_restitution", "table_restitution", "robot_restitution",
                     "robot_body_friction"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError("material values must be finite and non-negative")
        if not 0.01 <= self.initial_open <= 0.05:
            raise ValueError("initial_open must be within [0.01, 0.05]")
        if self.gripper_stiffness <= 0 or self.gripper_damping <= 0:
            raise ValueError("experimental gripper gains must be positive")

    @property
    def table_top(self):
        return self.table_position[2] + self.table_size[2] / 2.0

    @property
    def cube_start(self):
        return (self.cube_xy[0], self.cube_xy[1],
                self.table_top + self.cube_size / 2.0 + 0.0005)


def _float_list(value):
    if hasattr(value, "tolist"):
        return [float(v) for v in value.tolist()]
    return [float(v) for v in value]


def _vec3(value):
    return [float(value.x), float(value.y), float(value.z)]


def _mat33(value):
    return [_vec3(value.x), _vec3(value.y), _vec3(value.z)]


class _ExperimentalCpuCamera:
    """Host-readback camera that deliberately does not allocate CUDA tensors."""

    def __init__(self, gym, sim, env, gymapi, config):
        self.gym, self.sim, self.env, self.gymapi = gym, sim, env, gymapi
        self.config = config
        self.handle = None

    def initialize(self):
        properties = self.gymapi.CameraProperties()
        properties.width = self.config.width
        properties.height = self.config.height
        properties.horizontal_fov = self.config.horizontal_fov
        properties.enable_tensors = False
        self.handle = self.gym.create_camera_sensor(self.env, properties)
        if self.handle is None or self.handle < 0:
            raise RuntimeError("Isaac Gym failed to create experimental CPU camera")
        self.gym.set_camera_location(
            self.handle, self.env,
            self.gymapi.Vec3(*self.config.position),
            self.gymapi.Vec3(*self.config.target),
        )
        return self

    def close(self):
        if self.handle is not None:
            self.gym.destroy_camera_sensor(self.sim, self.env, self.handle)
            self.handle = None


class ManipulationScene:
    ROBOT_GROUP = 0
    ROBOT_FILTER = 0
    TABLE_GROUP = 0
    TABLE_FILTER = 0
    CUBE_GROUP = 0
    CUBE_FILTER = 0

    def __init__(self, config: ManipulationSceneConfig):
        self.config = config
        initial = tuple(PUSH_INITIAL_DOF_POSITIONS[:6]) + (config.initial_open, config.initial_open)
        self.robot = S1IsaacGymAdapter(S1IsaacGymConfig(
            graphics_device_id=config.graphics_device_id,
            headless=True,
            initial_dof_positions=initial,
        ))
        self.table_actor = None
        self.cube_actor = None
        self.camera = None
        self._camera_frame_index = 0
        self._root_state = None
        self._contact_force = None
        self._cube_actor_index = None
        self._cube_actor_index_tensor = None
        self.body_indices = {}
        self.shape_indices = {}
        self._property_dump = None
        self._closed = False

    def initialize(self):
        try:
            self.robot.initialize()
            self.robot.load_asset()
            self.robot.create_env(self.config.robot_position, self.config.robot_yaw)
            self._configure_gripper_drive()
            self._create_scene()
            self._configure_shape_properties()
            if self.config.with_camera:
                self._create_camera()
            self._property_dump = self._dump_properties_before_prepare()
            self.robot.prepare_and_acquire()
            self._acquire_tensors()
            self.reset()
            return self
        except Exception:
            self.close()
            raise

    def _configure_gripper_drive(self):
        props = self.robot.gym.get_actor_dof_properties(self.robot.env, self.robot.actor)
        props["stiffness"][6:] = self.config.gripper_stiffness
        props["damping"][6:] = self.config.gripper_damping
        self.robot.gym.set_actor_dof_properties(self.robot.env, self.robot.actor, props)

    def _create_scene(self):
        api, gym = self.robot._gymapi, self.robot.gym
        table_options = api.AssetOptions(); table_options.fix_base_link = True
        table_asset = gym.create_box(self.robot.sim, *self.config.table_size, table_options)
        table_pose = api.Transform(); table_pose.p = api.Vec3(*self.config.table_position)
        self.table_actor = gym.create_actor(
            self.robot.env, table_asset, table_pose, "p1_table",
            self.TABLE_GROUP, self.TABLE_FILTER,
        )
        if self.config.with_cube:
            cube_options = api.AssetOptions(); cube_options.density = self.config.cube_density
            cube_asset = gym.create_box(
                self.robot.sim, self.config.cube_size, self.config.cube_size,
                self.config.cube_size, cube_options,
            )
            cube_pose = api.Transform(); cube_pose.p = api.Vec3(*self.config.cube_start)
            self.cube_actor = gym.create_actor(
                self.robot.env, cube_asset, cube_pose, "p1_cube",
                self.CUBE_GROUP, self.CUBE_FILTER,
            )
            gym.set_rigid_body_color(
                self.robot.env, self.cube_actor, 0, api.MESH_VISUAL_AND_COLLISION,
                api.Vec3(0.85, 0.12, 0.08),
            )
        if self.table_actor is None or self.table_actor < 0:
            raise RuntimeError("failed to create experimental table")
        if self.config.with_cube and (self.cube_actor is None or self.cube_actor < 0):
            raise RuntimeError("failed to create experimental cube")

    def _actor_body_index(self, actor, name):
        return self.robot.gym.find_actor_rigid_body_index(
            self.robot.env, actor, name, self.robot._gymapi.IndexDomain.DOMAIN_ACTOR
        )

    def _configure_shape_properties(self):
        gym, env = self.robot.gym, self.robot.env
        robot_props = gym.get_actor_rigid_shape_properties(env, self.robot.actor)
        body_ranges = gym.get_actor_rigid_body_shape_indices(env, self.robot.actor)
        finger_shape_indices = []
        for body_name in ("8_Link", "7_Link"):
            body = self._actor_body_index(self.robot.actor, body_name)
            shape_range = body_ranges[body]
            indices = list(range(shape_range.start, shape_range.start + shape_range.count))
            self.shape_indices[body_name] = indices
            finger_shape_indices.extend(indices)
        for index, prop in enumerate(robot_props):
            prop.friction = (self.config.finger_friction if index in finger_shape_indices
                             else self.config.robot_body_friction)
            prop.restitution = self.config.robot_restitution
        if not gym.set_actor_rigid_shape_properties(env, self.robot.actor, robot_props):
            raise RuntimeError("failed to set experimental robot shape properties")
        table_props = gym.get_actor_rigid_shape_properties(env, self.table_actor)
        for prop in table_props:
            prop.friction = self.config.table_friction
            prop.restitution = self.config.table_restitution
        if not gym.set_actor_rigid_shape_properties(env, self.table_actor, table_props):
            raise RuntimeError("failed to set experimental table shape properties")
        if self.cube_actor is not None:
            cube_props = gym.get_actor_rigid_shape_properties(env, self.cube_actor)
            for prop in cube_props:
                prop.friction = self.config.cube_friction
                prop.restitution = self.config.cube_restitution
            if not gym.set_actor_rigid_shape_properties(env, self.cube_actor, cube_props):
                raise RuntimeError("failed to set experimental cube shape properties")

    def _shape_dump(self, actor):
        return [{
            "friction": float(p.friction), "restitution": float(p.restitution),
            "rolling_friction": float(p.rolling_friction),
            "torsion_friction": float(p.torsion_friction),
            "contact_offset": float(p.contact_offset), "rest_offset": float(p.rest_offset),
            "filter": int(p.filter),
        } for p in self.robot.gym.get_actor_rigid_shape_properties(self.robot.env, actor)]

    def _body_dump(self, actor):
        result = []
        for p in self.robot.gym.get_actor_rigid_body_properties(self.robot.env, actor):
            result.append({"mass": float(p.mass), "com": _vec3(p.com),
                           "inertia": _mat33(p.inertia)})
        return result

    def _dump_properties_before_prepare(self):
        gym = self.robot.gym
        dof = gym.get_actor_dof_properties(self.robot.env, self.robot.actor)
        params = gym.get_sim_params(self.robot.sim)
        dump = {
            "dof_names": list(gym.get_actor_dof_names(self.robot.env, self.robot.actor)),
            "dof_lower": _float_list(dof["lower"]),
            "dof_upper": _float_list(dof["upper"]),
            "dof_effort": _float_list(dof["effort"]),
            "dof_velocity": _float_list(dof["velocity"]),
            "dof_drive_mode": [int(v) for v in dof["driveMode"]],
            "dof_stiffness": _float_list(dof["stiffness"]),
            "dof_damping": _float_list(dof["damping"]),
            "finger_actor_body_indices": {
                name: int(self._actor_body_index(self.robot.actor, name))
                for name in ("8_Link", "7_Link")
            },
            "finger_shape_indices": dict(self.shape_indices),
            "robot_shapes": self._shape_dump(self.robot.actor),
            "table_shapes": self._shape_dump(self.table_actor),
            "cube_shapes": self._shape_dump(self.cube_actor) if self.cube_actor is not None else [],
            "robot_bodies": self._body_dump(self.robot.actor),
            "table_bodies": self._body_dump(self.table_actor),
            "cube_bodies": self._body_dump(self.cube_actor) if self.cube_actor is not None else [],
            "physx": {
                "dt": float(params.dt), "substeps": int(params.substeps),
                "contact_collection": str(params.physx.contact_collection),
                "contact_offset": float(params.physx.contact_offset),
                "rest_offset": float(params.physx.rest_offset),
                "position_iterations": int(params.physx.num_position_iterations),
                "velocity_iterations": int(params.physx.num_velocity_iterations),
            },
            "collision_actor_configuration": {
                "robot": {"group": self.ROBOT_GROUP, "filter": self.ROBOT_FILTER},
                "table": {"group": self.TABLE_GROUP, "filter": self.TABLE_FILTER},
                "cube": {"group": self.CUBE_GROUP, "filter": self.CUBE_FILTER},
            },
            "configured": {
                "cube_size_m": self.config.cube_size,
                "cube_density_kg_m3": self.config.cube_density,
                "cube_friction": self.config.cube_friction,
                "cube_restitution": self.config.cube_restitution,
                "table_friction": self.config.table_friction,
                "table_restitution": self.config.table_restitution,
                "finger_friction": self.config.finger_friction,
                "robot_body_friction": self.config.robot_body_friction,
                "robot_restitution": self.config.robot_restitution,
                "gripper_stiffness": self.config.gripper_stiffness,
                "gripper_damping": self.config.gripper_damping,
            },
        }
        return dump

    def _create_camera(self):
        cfg = IsaacGymCameraConfig(
            camera_id="p1_overview", width=640, height=480, horizontal_fov=52.0,
            position=(0.92, -0.92, 0.92), target=(0.44, 0.0, 0.45),
        )
        self.camera = _ExperimentalCpuCamera(
            self.robot.gym, self.robot.sim, self.robot.env,
            self.robot._gymapi, cfg,
        ).initialize()

    def _acquire_tensors(self):
        import torch
        gym = self.robot.gym
        self._root_state = self.robot._gymtorch.wrap_tensor(
            gym.acquire_actor_root_state_tensor(self.robot.sim)).view(-1, 13)
        self._contact_force = self.robot._gymtorch.wrap_tensor(
            gym.acquire_net_contact_force_tensor(self.robot.sim)).view(-1, 3)
        self.body_indices = dict(self.robot.body_indices)
        self.body_indices["table"] = gym.get_actor_rigid_body_index(
            self.robot.env, self.table_actor, 0, self.robot._gymapi.IndexDomain.DOMAIN_SIM)
        if self.cube_actor is not None:
            self.body_indices["cube"] = gym.get_actor_rigid_body_index(
                self.robot.env, self.cube_actor, 0, self.robot._gymapi.IndexDomain.DOMAIN_SIM)
            self._cube_actor_index = gym.get_actor_index(
                self.robot.env, self.cube_actor, self.robot._gymapi.IndexDomain.DOMAIN_SIM)
            self._cube_actor_index_tensor = torch.tensor(
                [self._cube_actor_index], dtype=torch.int32,
                device=self._root_state.device,
            )
        for name, index in self.body_indices.items():
            if index < 0 or index >= self._contact_force.shape[0]:
                raise RuntimeError("invalid body index for %s" % name)

    @property
    def property_dump(self):
        if self._property_dump is None:
            raise RuntimeError("scene is not initialized")
        result = dict(self._property_dump)
        result["sim_body_indices"] = dict(self.body_indices)
        return result

    def reset(self, cube_xy: Optional[Tuple[float, float]] = None):
        self.robot.reset_dofs()
        if self.cube_actor is not None:
            xy = tuple(cube_xy or self.config.cube_xy)
            self.robot.gym.refresh_actor_root_state_tensor(self.robot.sim)
            row = self._root_state[self._cube_actor_index]
            row[:3] = row.new_tensor((xy[0], xy[1], self.config.cube_start[2]))
            row[3:7] = row.new_tensor((0.0, 0.0, 0.0, 1.0))
            row[7:13] = 0.0
            self.robot.gym.set_actor_root_state_tensor_indexed(
                self.robot.sim, self.robot._gymtorch.unwrap_tensor(self._root_state),
                self.robot._gymtorch.unwrap_tensor(self._cube_actor_index_tensor), 1,
            )
        self.robot.step(8)

    def refresh(self):
        self.robot._refresh_tensors()
        self.robot.gym.refresh_net_contact_force_tensor(self.robot.sim)
        self.robot.gym.refresh_actor_root_state_tensor(self.robot.sim)

    def snapshot(self):
        self.refresh()
        state = self.robot.read_state()
        bodies = {}
        for name in ("ee_link", "8_Link", "7_Link"):
            index = self.body_indices[name]
            bodies[name] = {
                "state": _float_list(self.robot._rigid_body_state[index].detach().cpu()),
                "force": _float_list(self._contact_force[index].detach().cpu()),
            }
        table_index = self.body_indices["table"]
        bodies["table"] = {
            "state": _float_list(self.robot._rigid_body_state[table_index].detach().cpu()),
            "force": _float_list(self._contact_force[table_index].detach().cpu()),
        }
        cube_state = None
        if self.cube_actor is not None:
            index = self.body_indices["cube"]
            cube_state = _float_list(self._root_state[self._cube_actor_index].detach().cpu())
            bodies["cube"] = {
                "state": _float_list(self.robot._rigid_body_state[index].detach().cpu()),
                "force": _float_list(self._contact_force[index].detach().cpu()),
            }
        q = _float_list(state.dof_positions.detach().cpu())
        return {
            "q": q, "dq": _float_list(state.dof_velocities.detach().cpu()),
            "ee_state": _float_list(state.ee_link_state.detach().cpu()),
            "cube_state": cube_state, "bodies": bodies,
            "estimated_aperture": max(0.0, 2.0 * (q[6] + q[7]) / 2.0 - 0.0099),
        }

    def capture(self):
        if self.camera is None:
            raise RuntimeError("camera is disabled")
        # On this Isaac Gym Preview 4 / driver combination the CUDA camera
        # tensor contains alternating corrupt scanlines, while the documented
        # host readback is correct.  Keep this workaround experimental-only;
        # physics remains on the GPU and Production camera behavior is not
        # changed.
        gym = self.robot.gym
        gym.step_graphics(self.robot.sim)
        gym.render_all_camera_sensors(self.robot.sim)
        raw = gym.get_camera_image(
            self.robot.sim, self.robot.env, self.camera.handle,
            self.robot._gymapi.IMAGE_COLOR,
        )
        config = self.camera.config
        pixels = np.asarray(raw, dtype=np.uint8).reshape(
            config.height, config.width, 4,
        ).copy(order="C")
        frame = CameraFrame(
            camera_id=config.camera_id, width=config.width,
            height=config.height, pixel_format=RGBA8,
            frame_index=self._camera_frame_index,
            captured_at=datetime.now(timezone.utc), data=pixels,
        )
        self._camera_frame_index += 1
        return frame

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.camera is not None:
                self.camera.close()
        finally:
            self.camera = None
            self.robot.close()
