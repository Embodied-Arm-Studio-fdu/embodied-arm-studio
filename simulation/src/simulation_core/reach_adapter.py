"""S1 planar Reach task with a fixed height and fixed tool orientation."""

from dataclasses import dataclass, replace
import math
import random
from typing import Any, Callable, Dict, Optional, Protocol, Sequence, Tuple

from .cartesian_controller import (
    ControllerAdapter, JointPositionCommand, S1_LOWER_LIMITS, S1_UPPER_LIMITS,
)
from .isaac_gym_camera import CameraFrame, IsaacGymCamera, IsaacGymCameraConfig
from .push_adapter import (
    EETarget, PUSH_FIXED_ORIENTATION_XYZW, PUSH_FIXED_TARGET_Z,
    PUSH_INITIAL_DOF_POSITIONS,
)
from .s1_isaac_gym_adapter import S1IsaacGymAdapter, S1IsaacGymConfig, S1State
from .video_frame_pipeline import FramePipeline


REACH_TASK_ID = "s1_reach"
REACH_OBSERVATION_SCHEMA_VERSION = 1
REACH_OBSERVATION_FIELDS = (
    "arm_q", "arm_dq", "ee_to_goal_xy", "ee_velocity_xy", "previous_action_xy",
)
REACH_OBSERVATION_SIZES = (6, 6, 2, 2, 2)
REACH_OBSERVATION_DIM = 18
REACH_OBSERVATION_SEMANTICS = (
    "arm_q_rad[6]", "arm_dq_rad_per_s[6]", "ee_to_goal_xy_m[2]",
    "ee_velocity_xy_m_per_s[2]", "previous_action_xy_normalized[2]",
)
REACH_ACTION_DIM = 2
REACH_ACTION_SEMANTICS = "normalized_xy_delta_fixed_z_fixed_orientation"
REACH_REWARD_SCHEMA_VERSION = 1
REACH_REWARD_SEMANTICS = (
    "xy_distance_progress", "one_time_held_success_bonus",
    "normalized_2d_action_penalty", "arm_joint_limit_penalty",
)
REACH_REWARD_TERM_NAMES = (
    "progress_reward", "success_bonus", "action_penalty", "joint_limit_penalty",
)


class ReachAdapterError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReachConfig:
    table_size: Tuple[float, float, float] = (0.8, 0.8, 0.05)
    table_position: Tuple[float, float, float] = (0.45, 0.0, 0.35)
    robot_position: Tuple[float, float, float] = (0.10, 0.0, 0.40925)
    robot_yaw: float = math.pi
    robot_initial_dof_positions: Tuple[float, ...] = PUSH_INITIAL_DOF_POSITIONS
    fixed_target_z_m: float = PUSH_FIXED_TARGET_Z
    fixed_orientation_xyzw: Tuple[float, ...] = PUSH_FIXED_ORIENTATION_XYZW
    target_center_xy_m: Tuple[float, float] = (0.36, 0.0)
    target_half_extent_xy_m: Tuple[float, float] = (0.03, 0.08)
    workspace_min_xy_m: Tuple[float, float] = (0.30, -0.13)
    workspace_max_xy_m: Tuple[float, float] = (0.50, 0.13)
    action_delta_m: float = 0.005
    success_distance_m: float = 0.02
    success_hold_steps: int = 3
    max_steps: int = 128
    physics_steps_per_action: int = 8
    seed: int = 7
    progress_scale: float = 100.0
    success_bonus: float = 2.0
    action_penalty_scale: float = 0.002
    joint_limit_penalty_scale: float = 0.01
    joint_limit_margin_fraction: float = 0.1
    reward_extension: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.reward_extension is not None:
            from .reward_extension import validate_extension
            validate_extension("s1_reach", self.reward_extension)
        tuple_sizes = {
            "table_size": 3, "table_position": 3, "robot_position": 3,
            "robot_initial_dof_positions": 8, "fixed_orientation_xyzw": 4,
            "target_center_xy_m": 2, "target_half_extent_xy_m": 2,
            "workspace_min_xy_m": 2, "workspace_max_xy_m": 2,
        }
        for name, size in tuple_sizes.items():
            value = tuple(float(item) for item in getattr(self, name))
            if len(value) != size or not all(math.isfinite(item) for item in value):
                raise ValueError("%s must contain %d finite values" % (name, size))
            object.__setattr__(self, name, value)
        if not math.isclose(
            math.sqrt(sum(x * x for x in self.fixed_orientation_xyzw)),
            1.0, abs_tol=1.0e-5,
        ):
            raise ValueError("fixed_orientation_xyzw must be a unit quaternion")
        for name in (
            "fixed_target_z_m", "action_delta_m", "success_distance_m",
            "progress_scale", "success_bonus", "action_penalty_scale",
            "joint_limit_penalty_scale", "joint_limit_margin_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError("%s must be finite and non-negative" % name)
        if self.action_delta_m <= 0 or self.success_distance_m <= 0:
            raise ValueError("action_delta_m and success_distance_m must be positive")
        for name in ("success_hold_steps", "max_steps", "physics_steps_per_action"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("%s must be a positive integer" % name)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for low, high in zip(self.workspace_min_xy_m, self.workspace_max_xy_m):
            if low >= high:
                raise ValueError("workspace minimum must be below maximum")
        for center, half, low, high in zip(
            self.target_center_xy_m, self.target_half_extent_xy_m,
            self.workspace_min_xy_m, self.workspace_max_xy_m,
        ):
            if half <= 0 or center - half < low or center + half > high:
                raise ValueError("target sampling rectangle must fit the workspace")


def reward_schema_metadata(config: ReachConfig) -> Dict[str, Any]:
    metadata = {
        "reward_schema_version": REACH_REWARD_SCHEMA_VERSION,
        "reward_semantics": REACH_REWARD_SEMANTICS,
        "reward_weights": {
            "progress_scale": config.progress_scale,
            "success_bonus": config.success_bonus,
            "action_penalty_scale": config.action_penalty_scale,
            "joint_limit_penalty_scale": config.joint_limit_penalty_scale,
            "success_distance_m": config.success_distance_m,
            "success_hold_steps": config.success_hold_steps,
        },
    }
    if config.reward_extension is not None:
        metadata["reward_weights"]["extension_sha256"] = config.reward_extension["sha256"]
    return metadata


class ReachActionMapper:
    """Persistent normalized XY target with fixed Z/orientation and clamp."""

    def __init__(self, config: ReachConfig) -> None:
        self.config = config
        self._target_xy = None

    def reset(self, ee_state: Sequence[float]) -> EETarget:
        xy = (float(ee_state[0]), float(ee_state[1]))
        self._target_xy = self._clamp(xy)
        return self.current_target()

    def map(self, action: Sequence[float]) -> EETarget:
        if self._target_xy is None:
            raise ReachAdapterError("ReachActionMapper.reset() must run before map()")
        values = self.validate_action(action)
        self._target_xy = self._clamp(tuple(
            self._target_xy[i] + values[i] * self.config.action_delta_m
            for i in range(2)
        ))
        return self.current_target()

    def _clamp(self, xy):
        return tuple(max(low, min(high, value)) for value, low, high in zip(
            xy, self.config.workspace_min_xy_m, self.config.workspace_max_xy_m
        ))

    def current_target(self) -> EETarget:
        if self._target_xy is None:
            raise ReachAdapterError("ReachActionMapper has no target before reset()")
        return EETarget(
            (self._target_xy[0], self._target_xy[1], self.config.fixed_target_z_m),
            self.config.fixed_orientation_xyzw,
        )

    @staticmethod
    def validate_action(action: Sequence[float]) -> Tuple[float, float]:
        try:
            values = tuple(float(value) for value in action)
        except (TypeError, ValueError) as exc:
            raise ValueError("Reach action must contain 2 finite normalized values") from exc
        if len(values) != 2 or not all(math.isfinite(value) for value in values):
            raise ValueError("Reach action must contain 2 finite normalized values")
        if any(abs(value) > 1.0 + 1.0e-12 for value in values):
            raise ValueError("Reach action must stay in [-1, 1]")
        return values


class ReachBackend(Protocol):
    def setup(self) -> None: ...
    def reset(self) -> None: ...
    def read_robot_state(self) -> S1State: ...
    def set_ee_target(self, target: EETarget) -> bool: ...
    def set_goal_overlay(self, target_xy, radius: float) -> None: ...
    def step(self, n: int) -> None: ...
    def close(self) -> None: ...


def _tuple(value) -> Tuple[float, ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return tuple(float(item) for item in value)


class ReachAdapter:
    def __init__(self, config=None, backend=None, backend_factory=None) -> None:
        self.config = config or ReachConfig()
        from .reward_extension import validate_extension
        self._reward_expression = validate_extension("s1_reach", self.config.reward_extension) if self.config.reward_extension is not None else None
        if backend is not None and backend_factory is not None:
            raise ValueError("provide backend or backend_factory, not both")
        self._backend = backend or (backend_factory or IsaacGymReachBackend)(self.config)
        self._mapper = ReachActionMapper(self.config)
        self._rng = random.Random(self.config.seed)
        self._target_xy = None
        self._previous_distance = None
        self._previous_action = (0.0, 0.0)
        self._initial_distance = None
        self._minimum_distance = None
        self._success_hold = 0
        self._step_count = 0
        self._setup = False
        self._closed = False

    @property
    def backend(self): return self._backend
    @property
    def target_xy(self): return self._target_xy

    def initialize(self):
        if self._closed: raise ReachAdapterError("adapter is closed")
        if not self._setup:
            try:
                self._backend.setup(); self._setup = True
            except Exception:
                self.close(); raise
        return self

    def reset(self):
        self._require_setup()
        self._backend.reset()
        state = self._backend.read_robot_state()
        self._mapper.reset(state.ee_link_state)
        ee_xy = _tuple(state.ee_link_state)[:2]
        self._target_xy = self._sample_target(ee_xy)
        self._backend.set_goal_overlay(self._target_xy, self.config.success_distance_m)
        self._previous_action = (0.0, 0.0)
        self._step_count = 0
        self._success_hold = 0
        self._previous_distance = math.hypot(
            self._target_xy[0] - ee_xy[0], self._target_xy[1] - ee_xy[1]
        )
        self._initial_distance = self._previous_distance
        self._minimum_distance = self._previous_distance
        return self.get_observation()

    def _sample_target(self, ee_xy):
        cx, cy = self.config.target_center_xy_m
        hx, hy = self.config.target_half_extent_xy_m
        for _ in range(1024):
            target = (self._rng.uniform(cx - hx, cx + hx), self._rng.uniform(cy - hy, cy + hy))
            if math.hypot(target[0] - ee_xy[0], target[1] - ee_xy[1]) > self.config.success_distance_m:
                return target
        raise ReachAdapterError("could not sample a target outside success radius")

    def get_observation(self):
        self._require_setup()
        state = self._backend.read_robot_state()
        q, dq, ee = _tuple(state.dof_positions), _tuple(state.dof_velocities), _tuple(state.ee_link_state)
        if len(q) != 8 or len(dq) != 8 or len(ee) != 13:
            raise ReachAdapterError("invalid S1 state shape")
        return {
            "arm_q": q[:6], "arm_dq": dq[:6],
            "ee_to_goal_xy": (self._target_xy[0] - ee[0], self._target_xy[1] - ee[1]),
            "ee_velocity_xy": ee[7:9], "previous_action_xy": self._previous_action,
        }

    def step(self, action):
        self._require_setup()
        values = self._mapper.validate_action(action)
        target = self._mapper.map(values)
        controller_applied = self._backend.set_ee_target(target)
        self._backend.step(self.config.physics_steps_per_action)
        capture = getattr(self._backend, "maybe_capture_frame", None)
        if capture is not None: capture()
        self._step_count += 1
        self._previous_action = values
        observation = self.get_observation()
        distance = math.hypot(*observation["ee_to_goal_xy"])
        self._minimum_distance = min(self._minimum_distance, distance)
        within = distance <= self.config.success_distance_m
        self._success_hold = self._success_hold + 1 if within else 0
        success = self._success_hold >= self.config.success_hold_steps
        state = self._backend.read_robot_state()
        failure_reason = self._failure_reason(state)
        failure = failure_reason is not None
        timeout = self._step_count >= self.config.max_steps and not success and not failure
        reward_terms = self.compute_reward_terms(distance, values, success, state)
        action_norm = math.sqrt(sum(value * value for value in values))
        info = {
            "success": success and not failure, "timeout": timeout, "failure": failure,
            "failure_reason": failure_reason, "distance_to_goal": distance,
            "initial_distance_to_goal": self._initial_distance,
            "minimum_distance": self._minimum_distance,
            "time_to_success": self._step_count if success else None,
            "action_norm": action_norm, "saturation_rate": sum(abs(v) >= 1 - 1e-6 for v in values) / 2,
            "final_action_norm": action_norm,
            "joint_limit_proximity": self._joint_limit_proximity(state),
            "reward_terms": reward_terms, "step_count": self._step_count,
            "success_hold_steps": self._success_hold, "target_xy": self._target_xy,
            "controller_applied": bool(controller_applied), "ee_target": target,
        }
        self._previous_distance = distance
        reward = self._reward_expression.evaluate(reward_terms) if self._reward_expression is not None else sum(reward_terms.values())
        return observation, reward, success or timeout or failure, info

    def compute_reward_terms(self, distance, action=(0.0, 0.0), success=False, state=None):
        values = self._mapper.validate_action(action)
        previous = distance if self._previous_distance is None else self._previous_distance
        proximity = 0.0 if state is None else self._joint_limit_proximity(state)
        return {
            "progress_reward": self.config.progress_scale * (previous - float(distance)),
            "success_bonus": self.config.success_bonus if success else 0.0,
            "action_penalty": -self.config.action_penalty_scale * sum(v * v for v in values),
            "joint_limit_penalty": -self.config.joint_limit_penalty_scale * proximity,
        }

    def _failure_reason(self, state):
        values = _tuple(state.dof_positions) + _tuple(state.dof_velocities) + _tuple(state.ee_link_state)
        if not all(math.isfinite(value) for value in values): return "non_finite_state"
        ee = _tuple(state.ee_link_state)
        margin = 0.08
        if any(ee[i] < self.config.workspace_min_xy_m[i] - margin or ee[i] > self.config.workspace_max_xy_m[i] + margin for i in range(2)):
            return "workspace_violation"
        return None

    def _joint_limit_proximity(self, state):
        penalties = []
        for value, low, high in zip(_tuple(state.dof_positions)[:6], S1_LOWER_LIMITS, S1_UPPER_LIMITS):
            margin = (high - low) * self.config.joint_limit_margin_fraction
            distance = min(value - low, high - value)
            penalties.append(max(0.0, min(1.0, (margin - distance) / margin)))
        return sum(penalties) / len(penalties)

    def close(self):
        if self._closed: return
        try: self._backend.close()
        finally:
            self._setup = False; self._closed = True

    def _require_setup(self):
        if self._closed: raise ReachAdapterError("adapter is closed")
        if not self._setup: raise ReachAdapterError("initialize() must be called first")


class IsaacGymReachBackend:
    """Table + S1 runtime; the goal is a non-physical Camera Sensor overlay."""

    def __init__(self, config, robot=None, s1_config=None, controller=None,
                 camera_config=None, frame_pipeline: Optional[FramePipeline]=None,
                 stage_callback=None):
        self.config = config
        base = s1_config or S1IsaacGymConfig()
        self.robot = robot or S1IsaacGymAdapter(replace(
            base, initial_dof_positions=config.robot_initial_dof_positions
        ))
        self.controller = controller or ControllerAdapter()
        self.camera_config = camera_config
        self.frame_pipeline = frame_pipeline
        self.camera = None
        self.table_actor = None
        self.last_target = None
        self.last_command = None
        self.last_video_error = None
        self._overlay_xy = None
        self._overlay_radius = config.success_distance_m
        self._stage_callback = stage_callback or (lambda stage: None)

    def setup(self):
        try:
            self._stage_callback("sim"); self.robot.initialize()
            self._stage_callback("asset"); self.robot.load_asset()
            self._stage_callback("env"); self.robot.create_env(self.config.robot_position, self.config.robot_yaw)
            self._stage_callback("scene"); self._create_table()
            self._stage_callback("camera"); self._create_camera()
            self._stage_callback("prepare"); self.robot.prepare_and_acquire()
            self._stage_callback("ready")
        except Exception:
            self.close(); raise

    def _create_table(self):
        api, gym = self.robot._gymapi, self.robot.gym
        options = api.AssetOptions(); options.fix_base_link = True
        asset = gym.create_box(self.robot.sim, *self.config.table_size, options)
        pose = api.Transform(); pose.p = api.Vec3(*self.config.table_position)
        self.table_actor = gym.create_actor(self.robot.env, asset, pose, "table", 0, 0)
        if self.table_actor is None or self.table_actor < 0:
            raise ReachAdapterError("Isaac Gym failed to create Reach table")

    def _create_camera(self):
        if self.camera_config is None: return
        self.camera = IsaacGymCamera(
            self.robot.gym, self.robot.sim, self.robot.env, self.robot._gymapi,
            self.robot._gymtorch, self.camera_config,
        ).initialize()

    def reset(self):
        self.robot.reset_dofs(); self.robot.step(1)
        self.last_target = None; self.last_command = None

    def read_robot_state(self): return self.robot.read_state()
    def set_goal_overlay(self, target_xy, radius):
        self._overlay_xy = tuple(float(v) for v in target_xy); self._overlay_radius = float(radius)

    def set_ee_target(self, target):
        state = self.robot.read_state(); command = self.controller.compute(target, state)
        self.robot.set_joint_position_targets(command.positions)
        self.last_target, self.last_command = target, command
        return True

    def step(self, n): self.robot.step(n)

    def capture_camera_frame(self):
        if self.camera is None: raise ReachAdapterError("camera is disabled")
        frame = self.camera.capture()
        if self._overlay_xy is None: return frame
        pixels = frame.data.copy()
        projected = self._project((self._overlay_xy[0], self._overlay_xy[1], self.config.fixed_target_z_m))
        edge = self._project((self._overlay_xy[0] + self._overlay_radius, self._overlay_xy[1], self.config.fixed_target_z_m))
        if projected is not None and edge is not None:
            u, v = projected; radius = max(4, int(math.hypot(edge[0]-u, edge[1]-v)))
            yy, xx = __import__("numpy").ogrid[:frame.height, :frame.width]
            distance2 = (xx-u)**2 + (yy-v)**2
            pixels[distance2 <= 16, :3] = (255, 20, 20)
            ring = (distance2 >= (radius-2)**2) & (distance2 <= (radius+2)**2)
            pixels[ring, :3] = (255, 20, 20)
            pixels[(distance2 <= 16) | ring, 3] = 255
        return CameraFrame(frame.camera_id, frame.width, frame.height, frame.pixel_format,
                           frame.frame_index, frame.captured_at, pixels)

    def _project(self, point):
        import numpy as np
        camera = np.asarray(self.camera_config.position, dtype=float)
        target = np.asarray(self.camera_config.target, dtype=float)
        forward = target-camera; forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray((0.,0.,1.))); right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        relative = np.asarray(point)-camera; depth = float(relative @ forward)
        if depth <= 1e-6: return None
        focal = self.camera_config.width/(2*math.tan(math.radians(self.camera_config.horizontal_fov)/2))
        return (int(round(self.camera_config.width/2 + focal*float(relative@right)/depth)),
                int(round(self.camera_config.height/2 - focal*float(relative@up)/depth)))

    def maybe_capture_frame(self, now=None):
        if self.camera is None or self.frame_pipeline is None: return None
        try: return self.frame_pipeline.maybe_produce(self.capture_camera_frame, now)
        except Exception as exc: self.last_video_error = exc; return None

    def close(self):
        try:
            if self.camera is not None: self.camera.close()
        finally:
            self.camera = None; self.robot.close()
