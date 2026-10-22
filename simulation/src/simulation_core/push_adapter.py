"""Minimal S1 Push episode adapter independent of DyWA and PPO.

The task lifecycle is deliberately separated from the Isaac Gym scene backend,
which lets ordinary unit tests run without importing Isaac Gym or initializing
CUDA.  The first backend creates one table and one cube beside the existing S1.
"""

from dataclasses import dataclass, replace
import math
from typing import Any, Callable, Dict, Optional, Protocol, Sequence, Tuple

from .s1_isaac_gym_adapter import S1IsaacGymAdapter, S1IsaacGymConfig, S1State
from .cartesian_controller import (
    ControllerAdapter,
    JointPositionCommand,
    S1_LOWER_LIMITS,
    S1_UPPER_LIMITS,
)
from .isaac_gym_camera import IsaacGymCamera, IsaacGymCameraConfig
from .video_frame_pipeline import FramePipeline


ACTION_DIM = 2
ACTION_SCALES = (0.005, 0.005)
PUSH_REWARD_SCHEMA_VERSION = 2
PUSH_REWARD_SEMANTICS = (
    "gated_precontact_progress",
    "wrong_side_or_alignment_penalty",
    "signed_directional_object_motion",
    "absolute_lateral_object_motion_penalty",
    "dynamic_contact_sustain_bonus",
    "one_time_terminal_success_bonus",
    "normalized_2d_action_penalty",
    "arm_joint_limit_penalty",
)
PUSH_REWARD_TERM_NAMES = (
    "precontact_progress",
    "wrong_side_or_alignment_term",
    "directional_push_progress",
    "lateral_push_penalty",
    "contact_sustain_bonus",
    "success_bonus",
    "action_penalty",
    "joint_limit_penalty",
)
# P1.1 real-GPU trace, final sample before first dynamic contact (step 11).
VALIDATED_PRECONTACT_BEHIND_OFFSET = 0.09665584447072773
VALIDATED_PRECONTACT_LATERAL_OFFSET = 0.016129788026494007
VALIDATED_PRECONTACT_NEAR_RADIUS = 0.005
# Push-only reset found by bounded URDF FK and verified in the real GPU scene.
# The final two prismatic joints open each finger by 10 mm, removing the
# q7=q8=0 finger/finger self-contact without changing the shared S1 default.
PUSH_INITIAL_DOF_POSITIONS = (
    -0.2351417354,
    1.7607090316,
    0.8574648293,
    0.4087952421,
    0.0926116363,
    0.0120945175,
    0.01,
    0.01,
)
PUSH_FIXED_TARGET_Z = 0.3873559236526489
PUSH_FIXED_ORIENTATION_XYZW = (
    0.6040455102920532,
    -0.7851132154464722,
    0.12926433980464935,
    -0.044911183416843414,
)


class PushAdapterError(RuntimeError):
    """Raised for invalid lifecycle use or an invalid backend state."""


@dataclass(frozen=True)
class PushConfig:
    """Configuration for one deterministic table-and-cube Push episode."""

    table_size: Tuple[float, float, float] = (0.8, 0.8, 0.05)
    table_position: Tuple[float, float, float] = (0.45, 0.0, 0.35)
    # Mount the S1 base on the rear edge of the table.  The z value is table
    # top (0.375 m) plus the base mesh's 34.25 mm extent below its root frame.
    robot_position: Tuple[float, float, float] = (0.10, 0.0, 0.40925)
    robot_yaw: float = math.pi
    robot_initial_dof_positions: Tuple[float, ...] = PUSH_INITIAL_DOF_POSITIONS
    fixed_target_z: float = PUSH_FIXED_TARGET_Z
    fixed_orientation_xyzw: Tuple[float, ...] = PUSH_FIXED_ORIENTATION_XYZW
    object_size: Tuple[float, float, float] = (0.08, 0.08, 0.08)
    object_start: Tuple[float, float, float] = (0.45, 0.0, 0.415)
    goal_position: Tuple[float, float, float] = (0.60, 0.15, 0.415)
    success_distance: float = 0.05
    max_steps: int = 256
    physics_steps_per_action: int = 1
    precontact_reward_scale: float = 2.0
    wrong_side_penalty_scale: float = 1.0
    directional_push_reward_scale: float = 20.0
    lateral_push_penalty_scale: float = 5.0
    contact_bonus: float = 0.002
    success_bonus: float = 1.0
    action_penalty_scale: float = 0.001
    joint_limit_penalty_scale: float = 0.01
    joint_limit_margin_fraction: float = 0.1
    contact_force_threshold: float = 0.1
    contact_distance_margin: float = 0.03
    drop_height_margin: float = 0.0
    out_of_bounds_margin: float = 0.0
    precontact_behind_offset: float = VALIDATED_PRECONTACT_BEHIND_OFFSET
    precontact_lateral_offset: float = VALIDATED_PRECONTACT_LATERAL_OFFSET
    precontact_near_radius: float = VALIDATED_PRECONTACT_NEAR_RADIUS
    reward_extension: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.reward_extension is not None:
            from .reward_extension import validate_extension
            validate_extension("s1_push", self.reward_extension)
        for name in ("table_size", "table_position", "robot_position", "robot_initial_dof_positions", "fixed_orientation_xyzw", "object_size", "object_start", "goal_position"):
            value = tuple(getattr(self, name))
            expected = 8 if name == "robot_initial_dof_positions" else (4 if name == "fixed_orientation_xyzw" else 3)
            if len(value) != expected or not all(math.isfinite(float(item)) for item in value):
                raise ValueError("%s must contain %d finite values" % (name, expected))
            object.__setattr__(self, name, value)
        if any(value <= 0 for value in self.table_size + self.object_size):
            raise ValueError("table_size and object_size values must be positive")
        if not math.isfinite(self.robot_yaw):
            raise ValueError("robot_yaw must be finite")
        if not math.isfinite(self.fixed_target_z):
            raise ValueError("fixed_target_z must be finite")
        orientation_norm = math.sqrt(sum(value * value for value in self.fixed_orientation_xyzw))
        if not math.isclose(orientation_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-5):
            raise ValueError("fixed_orientation_xyzw must be a unit quaternion")
        if self.success_distance <= 0:
            raise ValueError("success_distance must be positive")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or self.max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if (
            isinstance(self.physics_steps_per_action, bool)
            or not isinstance(self.physics_steps_per_action, int)
            or self.physics_steps_per_action <= 0
        ):
            raise ValueError("physics_steps_per_action must be a positive integer")
        non_negative = (
            "precontact_reward_scale", "wrong_side_penalty_scale",
            "directional_push_reward_scale", "lateral_push_penalty_scale",
            "contact_bonus",
            "success_bonus", "action_penalty_scale", "joint_limit_penalty_scale",
            "joint_limit_margin_fraction", "contact_force_threshold",
            "contact_distance_margin", "drop_height_margin",
            "out_of_bounds_margin",
            "precontact_behind_offset", "precontact_near_radius",
        )
        for name in non_negative:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError("%s must be finite and non-negative" % name)
        if not math.isfinite(self.precontact_lateral_offset):
            raise ValueError("precontact_lateral_offset must be finite")


@dataclass(frozen=True)
class EETarget:
    """Decoded Cartesian pose target consumed by differential IK."""

    position: Tuple[float, float, float]
    orientation_xyzw: Tuple[float, float, float, float]


@dataclass(frozen=True)
class PushRewardGeometry:
    push_direction: Tuple[float, float]
    precontact_xy: Tuple[float, float]
    precontact_error: float
    longitudinal: float
    lateral: float
    behind_gate: float
    alignment_gate: float


def reward_schema_metadata(config: PushConfig) -> Dict[str, Any]:
    metadata = {
        "reward_schema_version": PUSH_REWARD_SCHEMA_VERSION,
        "reward_semantics": PUSH_REWARD_SEMANTICS,
        "reward_weights": {
            "precontact_reward_scale": config.precontact_reward_scale,
            "wrong_side_penalty_scale": config.wrong_side_penalty_scale,
            "directional_push_reward_scale": config.directional_push_reward_scale,
            "lateral_push_penalty_scale": config.lateral_push_penalty_scale,
            "contact_bonus": config.contact_bonus,
            "success_bonus": config.success_bonus,
            "action_penalty_scale": config.action_penalty_scale,
            "joint_limit_penalty_scale": config.joint_limit_penalty_scale,
            "precontact_behind_offset": config.precontact_behind_offset,
            "precontact_lateral_offset": config.precontact_lateral_offset,
            "precontact_near_radius": config.precontact_near_radius,
        },
    }
    if config.reward_extension is not None:
        metadata["reward_weights"]["extension_sha256"] = config.reward_extension["sha256"]
    return metadata


class PushActionMapper:
    """Map exactly two physical XY deltas onto the validated Push manifold."""

    def __init__(self, fixed_target_z: float, fixed_orientation_xyzw: Sequence[float]):
        self.fixed_target_z = float(fixed_target_z)
        self.fixed_orientation_xyzw = tuple(float(x) for x in fixed_orientation_xyzw)
        self._target_xy: Optional[Tuple[float, float]] = None

    def reset(self, ee_state: Sequence[float]) -> EETarget:
        self._target_xy = (float(ee_state[0]), float(ee_state[1]))
        return self.current_target()

    def map(self, action: Sequence[float]) -> EETarget:
        if self._target_xy is None:
            raise PushAdapterError("PushActionMapper.reset() must run before map()")
        values = PushAdapter._validate_action(action)
        self._target_xy = tuple(
            value + delta for value, delta in zip(self._target_xy, values)
        )
        return self.current_target()

    def current_target(self) -> EETarget:
        if self._target_xy is None:
            raise PushAdapterError("PushActionMapper has no target before reset()")
        return EETarget(
            (self._target_xy[0], self._target_xy[1], self.fixed_target_z),
            self.fixed_orientation_xyzw,
        )


class PushBackend(Protocol):
    """Small boundary required by :class:`PushAdapter`."""

    def setup(self) -> None: ...
    def reset(self) -> None: ...
    def read_robot_state(self) -> S1State: ...
    def read_object_state(self) -> Sequence[float]: ...
    def has_object_contact(self) -> bool: ...
    def set_ee_target(self, target: EETarget) -> bool: ...
    def step(self, n: int) -> None: ...
    def close(self) -> None: ...


def _tuple(value: Any) -> Tuple[float, ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return tuple(float(item) for item in value)


class PushAdapter:
    """Own the minimal reset/observation/action/reward/done Push lifecycle."""

    def __init__(
        self,
        config: Optional[PushConfig] = None,
        backend: Optional[PushBackend] = None,
        backend_factory: Optional[Callable[[PushConfig], PushBackend]] = None,
    ) -> None:
        self.config = config or PushConfig()
        from .reward_extension import validate_extension
        self._reward_expression = validate_extension("s1_push", self.config.reward_extension) if self.config.reward_extension is not None else None
        if backend is not None and backend_factory is not None:
            raise ValueError("provide backend or backend_factory, not both")
        self._backend = backend or (backend_factory or IsaacGymPushBackend)(self.config)
        self._action_mapper = PushActionMapper(
            self.config.fixed_target_z, self.config.fixed_orientation_xyzw
        )
        self._step_count = 0
        self._previous_distance: Optional[float] = None
        self._previous_precontact_error: Optional[float] = None
        self._previous_object_position: Optional[Tuple[float, ...]] = None
        self._previous_push_direction: Optional[Tuple[float, float]] = None
        self._initial_object_position: Optional[Tuple[float, ...]] = None
        self._initial_distance: Optional[float] = None
        self._success_rewarded = False
        self._last_observation: Optional[Dict[str, Tuple[float, ...]]] = None
        self._last_scene_state = None
        self._previous_action = (0.0, 0.0)
        self._setup = False
        self._closed = False

    def __enter__(self) -> "PushAdapter":
        self.initialize()
        self.reset()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    @property
    def backend(self):
        """Runtime collaborator access; ownership remains with this adapter."""
        return self._backend

    def initialize(self) -> "PushAdapter":
        if self._closed:
            raise PushAdapterError("adapter is closed")
        if not self._setup:
            try:
                self._backend.setup()
                self._setup = True
            except Exception:
                self.close()
                raise
        return self

    def reset(self) -> Dict[str, Tuple[float, ...]]:
        self._require_setup()
        self._backend.reset()
        self._step_count = 0
        reset_robot = self._backend.read_robot_state()
        self._action_mapper.reset(reset_robot.ee_link_state)
        self._previous_action = (0.0, 0.0)
        observation = self.get_observation()
        self._previous_distance = self._distance(self._last_scene_state)
        reset_geometry = self.reward_geometry(self._last_scene_state)
        self._previous_precontact_error = reset_geometry.precontact_error
        self._previous_push_direction = reset_geometry.push_direction
        self._initial_object_position = self._last_scene_state["object_state"][:3]
        self._previous_object_position = self._initial_object_position
        self._initial_distance = self._previous_distance
        self._success_rewarded = False
        return observation

    def get_observation(self) -> Dict[str, Tuple[float, ...]]:
        self._require_setup()
        robot = self._backend.read_robot_state()
        object_state = _tuple(self._backend.read_object_state())
        ee_state = _tuple(robot.ee_link_state)
        if len(object_state) != 13 or len(ee_state) != 13:
            raise PushAdapterError("ee_link_state and object_state must each have 13 values")
        positions = _tuple(robot.dof_positions)
        velocities = _tuple(robot.dof_velocities)
        if len(positions) != 8 or len(velocities) != 8:
            raise PushAdapterError("S1 robot state must contain 8 positions and 8 velocities")
        target = self._action_mapper.current_target()
        observation = {
            "arm_q": positions[:6],
            "arm_dq": velocities[:6],
            "ee_to_object_xy": tuple(
                object_state[index] - ee_state[index] for index in range(2)
            ),
            "object_to_goal_xy": tuple(
                self.config.goal_position[index] - object_state[index]
                for index in range(2)
            ),
            "ee_velocity_xy": ee_state[7:9],
            "object_velocity_xy": object_state[7:9],
            "target_tracking_error_xy": tuple(
                target.position[index] - ee_state[index] for index in range(2)
            ),
            "previous_action_xy": tuple(
                value / scale
                for value, scale in zip(self._previous_action, ACTION_SCALES)
            ),
        }
        self._last_scene_state = {
            "robot_state": positions + velocities,
            "ee_link_state": ee_state,
            "object_state": object_state,
            "goal_state": self.config.goal_position,
            "relative_position": tuple(
                self.config.goal_position[index] - object_state[index]
                for index in range(3)
            ),
        }
        self._last_observation = observation
        return dict(observation)

    def step(self, action: Sequence[float]):
        self._require_setup()
        values = self._validate_action(action)
        target = self._action_mapper.map(values)
        controller_applied = self._backend.set_ee_target(target)
        self._backend.step(self.config.physics_steps_per_action)
        # Video is an optional observation side effect. The backend isolates
        # capture/pipeline failures and this never changes the PPO action.
        maybe_capture = getattr(self._backend, "maybe_capture_frame", None)
        if maybe_capture is not None:
            maybe_capture()
        self._step_count += 1
        self._previous_action = values
        observation = self.get_observation()
        scene_state = self._last_scene_state
        distance = self._distance(scene_state)
        failure_reason = self._failure_reason(scene_state)
        failure = failure_reason is not None
        success = distance <= self.config.success_distance and not failure
        contact = bool(self._backend.has_object_contact())
        reward_terms = self.compute_reward_terms(
            scene_state, values, contact=contact, success=success
        )
        reward = self._reward_expression.evaluate(reward_terms) if self._reward_expression is not None else sum(reward_terms.values())
        done = self.is_done(scene_state)
        timeout = (
            self._step_count >= self.config.max_steps and not success and not failure
        )
        action_norm = math.sqrt(sum(
            (value / scale) ** 2 for value, scale in zip(values, ACTION_SCALES)
        ))
        saturation_rate = sum(
            abs(value) >= scale * (1.0 - 1.0e-6)
            for value, scale in zip(values, ACTION_SCALES)
        ) / ACTION_DIM
        object_displacement = self._object_displacement(scene_state)
        info = {
            "success": success,
            "timeout": timeout,
            "failure": failure,
            "failure_reason": failure_reason,
            "distance_to_goal": distance,
            "initial_distance_to_goal": self._initial_distance,
            "object_displacement": object_displacement,
            "contact": contact,
            "action_norm": action_norm,
            "saturation_rate": saturation_rate,
            "reward_terms": reward_terms,
            **self.reward_diagnostics(scene_state),
            "step_count": self._step_count,
            "controller_applied": bool(controller_applied),
            "ee_target": target,
        }
        self._previous_distance = distance
        self._previous_precontact_error = self.reward_geometry(
            scene_state
        ).precontact_error
        self._previous_object_position = scene_state["object_state"][:3]
        self._previous_push_direction = self.reward_geometry(scene_state).push_direction
        return observation, reward, done, info

    def compute_reward_terms(
        self, observation=None, action=None, contact=False, success=None
    ) -> Dict[str, float]:
        if observation is None or "relative_position" not in observation:
            observation = self._last_scene_state
        if observation is None:
            raise PushAdapterError("reset() must be called before computing reward")
        values = self._validate_action(
            (0.0,) * ACTION_DIM if action is None else action
        )
        distance = self._distance(observation)
        geometry = self.reward_geometry(observation)
        previous_precontact = (
            geometry.precontact_error
            if self._previous_precontact_error is None
            else self._previous_precontact_error
        )
        precontact_progress = previous_precontact - geometry.precontact_error
        geometry_gate = geometry.behind_gate * geometry.alignment_gate
        approach_weight = min(
            1.0, geometry.precontact_error / self.config.precontact_near_radius
        )
        positive_precontact = max(precontact_progress, 0.0)
        negative_precontact = min(precontact_progress, 0.0)
        object_delta = self._object_delta_xy(observation)
        reference_direction = self._previous_push_direction or geometry.push_direction
        directional_push = sum(
            object_delta[index] * reference_direction[index]
            for index in range(2)
        )
        lateral_direction = (reference_direction[1], -reference_direction[0])
        lateral_push = sum(
            object_delta[index] * lateral_direction[index] for index in range(2)
        )
        is_success = (
            distance <= self.config.success_distance if success is None else bool(success)
        )
        success_reward = is_success and not self._success_rewarded
        if success_reward:
            self._success_rewarded = True
        normalized_action_cost = sum(
            (value / scale) ** 2 for value, scale in zip(values, ACTION_SCALES)
        ) / ACTION_DIM
        terms = {
            "precontact_progress": self.config.precontact_reward_scale * (
                positive_precontact * geometry_gate * approach_weight
                + negative_precontact
            ),
            "wrong_side_or_alignment_term": -self.config.wrong_side_penalty_scale
            * positive_precontact * (1.0 - geometry_gate),
            "directional_push_progress": self.config.directional_push_reward_scale
            * directional_push,
            "lateral_push_penalty": -self.config.lateral_push_penalty_scale
            * abs(lateral_push),
            "contact_sustain_bonus": self.config.contact_bonus if contact else 0.0,
            "success_bonus": self.config.success_bonus if success_reward else 0.0,
            "action_penalty": -self.config.action_penalty_scale
            * normalized_action_cost,
            "joint_limit_penalty": -self.config.joint_limit_penalty_scale
            * self._joint_limit_proximity(observation),
        }
        return {name: float(value) for name, value in terms.items()}

    def reward_geometry(self, observation=None) -> PushRewardGeometry:
        if observation is None or "object_state" not in observation:
            observation = self._last_scene_state
        if observation is None:
            raise PushAdapterError("reset() must be called before reward geometry")
        obj = observation["object_state"]
        ee = observation["ee_link_state"]
        goal = observation["goal_state"]
        goal_vector = (goal[0] - obj[0], goal[1] - obj[1])
        goal_norm = math.hypot(*goal_vector)
        if goal_norm <= 1.0e-12:
            push_direction = (1.0, 0.0)
        else:
            push_direction = tuple(value / goal_norm for value in goal_vector)
        lateral_direction = (push_direction[1], -push_direction[0])
        precontact = tuple(
            obj[index]
            - push_direction[index] * self.config.precontact_behind_offset
            + lateral_direction[index] * self.config.precontact_lateral_offset
            for index in range(2)
        )
        object_to_ee = (ee[0] - obj[0], ee[1] - obj[1])
        longitudinal = sum(object_to_ee[i] * push_direction[i] for i in range(2))
        lateral = sum(object_to_ee[i] * lateral_direction[i] for i in range(2))
        error = math.hypot(ee[0] - precontact[0], ee[1] - precontact[1])
        behind_ratio = max(0.0, min(1.0, -longitudinal / self.config.precontact_behind_offset))
        behind_gate = behind_ratio * behind_ratio * (3.0 - 2.0 * behind_ratio)
        lateral_error = abs(lateral - self.config.precontact_lateral_offset)
        alignment_gate = max(
            0.0, 1.0 - lateral_error / self.config.precontact_behind_offset
        )
        return PushRewardGeometry(
            push_direction, precontact, error, longitudinal, lateral,
            behind_gate, alignment_gate,
        )

    def reward_diagnostics(self, observation=None) -> Dict[str, Any]:
        geometry = self.reward_geometry(observation)
        delta = self._object_delta_xy(observation or self._last_scene_state)
        reference_direction = self._previous_push_direction or geometry.push_direction
        lateral_direction = (reference_direction[1], -reference_direction[0])
        return {
            "precontact_error": geometry.precontact_error,
            "precontact_xy": geometry.precontact_xy,
            "push_direction_xy": geometry.push_direction,
            "longitudinal": geometry.longitudinal,
            "lateral": geometry.lateral,
            "behind_gate": geometry.behind_gate,
            "alignment_gate": geometry.alignment_gate,
            "near_precontact": geometry.precontact_error <= self.config.precontact_near_radius,
            "directional_push": sum(delta[i] * reference_direction[i] for i in range(2)),
            "lateral_push": sum(delta[i] * lateral_direction[i] for i in range(2)),
        }

    def _object_delta_xy(self, observation) -> Tuple[float, float]:
        current = observation["object_state"]
        previous = self._previous_object_position or current
        return (current[0] - previous[0], current[1] - previous[1])

    def compute_reward(self, observation=None, action=None, contact=False, success=None) -> float:
        terms = self.compute_reward_terms(observation, action, contact, success)
        return self._reward_expression.evaluate(terms) if self._reward_expression is not None else sum(terms.values())

    def is_done(self, observation=None) -> bool:
        if observation is None or "relative_position" not in observation:
            observation = self._last_scene_state
        if observation is None:
            raise PushAdapterError("reset() must be called before is_done()")
        return (
            self._failure_reason(observation) is not None
            or self._distance(observation) <= self.config.success_distance
            or self._step_count >= self.config.max_steps
        )

    def _failure_reason(self, observation) -> Optional[str]:
        object_position = observation["object_state"][:3]
        table_top = self.config.table_position[2] + self.config.table_size[2] / 2.0
        object_half_height = self.config.object_size[2] / 2.0
        drop_threshold = (
            table_top - object_half_height - self.config.drop_height_margin
        )
        if object_position[2] < drop_threshold:
            return "object_drop"
        for index in (0, 1):
            half_extent = self.config.table_size[index] / 2.0
            lower = (
                self.config.table_position[index] - half_extent
                - self.config.out_of_bounds_margin
            )
            upper = (
                self.config.table_position[index] + half_extent
                + self.config.out_of_bounds_margin
            )
            if object_position[index] < lower or object_position[index] > upper:
                return "object_out_of_bounds"
        return None

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._backend.close()
        finally:
            self._last_observation = None
            self._last_scene_state = None
            self._previous_action = (0.0, 0.0)
            self._previous_distance = None
            self._previous_precontact_error = None
            self._previous_object_position = None
            self._previous_push_direction = None
            self._initial_object_position = None
            self._initial_distance = None
            self._success_rewarded = False
            self._setup = False
            self._closed = True

    def _require_setup(self) -> None:
        if self._closed:
            raise PushAdapterError("adapter is closed")
        if not self._setup:
            raise PushAdapterError("initialize() must be called first")

    @staticmethod
    def _distance(observation) -> float:
        return math.sqrt(sum(value * value for value in observation["relative_position"]))

    def _object_displacement(self, observation) -> float:
        if self._initial_object_position is None:
            return 0.0
        return math.sqrt(sum(
            (observation["object_state"][index] - self._initial_object_position[index]) ** 2
            for index in range(3)
        ))

    def _joint_limit_proximity(self, observation) -> float:
        values = observation["robot_state"][:6]
        penalties = []
        margin_fraction = self.config.joint_limit_margin_fraction
        for value, lower, upper in zip(values, S1_LOWER_LIMITS, S1_UPPER_LIMITS):
            margin = (upper - lower) * margin_fraction
            if margin <= 0:
                penalties.append(0.0)
                continue
            distance = min(value - lower, upper - value)
            penalties.append(max(0.0, min(1.0, (margin - distance) / margin)))
        return sum(penalties) / len(penalties)

    @staticmethod
    def _validate_action(action: Sequence[float]) -> Tuple[float, ...]:
        try:
            values = tuple(float(value) for value in action)
        except (TypeError, ValueError) as exc:
            raise ValueError("action must be a sequence of 2 finite numbers") from exc
        if len(values) != ACTION_DIM or not all(math.isfinite(value) for value in values):
            raise ValueError("action must contain exactly 2 finite values")
        if any(abs(value) > scale + 1.0e-12 for value, scale in zip(values, ACTION_SCALES)):
            raise ValueError("action must stay within physical XY bounds +/-0.005 m")
        return values


class IsaacGymPushBackend:
    """One-table/one-cube backend built around :class:`S1IsaacGymAdapter`.

    Cartesian targets are converted by the injected controller into S1 joint
    position targets. This remains simpler than DyWA's impedance torque path.
    """

    TABLE_NAME = "table"
    OBJECT_NAME = "push_object"
    GOAL_NAME = "push_goal"
    PHYSICS_COLLISION_GROUP = 0
    GOAL_VISUAL_COLLISION_GROUP = 1

    def __init__(
        self,
        config: PushConfig,
        robot: Optional[S1IsaacGymAdapter] = None,
        s1_config: Optional[S1IsaacGymConfig] = None,
        controller: Optional[ControllerAdapter] = None,
        camera_config: Optional[IsaacGymCameraConfig] = None,
        stage_callback: Optional[Callable[[str], None]] = None,
        frame_pipeline: Optional[FramePipeline] = None,
    ) -> None:
        self.config = config
        if robot is None:
            base_s1_config = s1_config or S1IsaacGymConfig()
            # IsaacGymPushBackend is the task boundary: override only this
            # task's reset pose while preserving device/physics/camera fields.
            push_s1_config = replace(
                base_s1_config,
                initial_dof_positions=config.robot_initial_dof_positions,
            )
            robot = S1IsaacGymAdapter(push_s1_config)
        self.robot = robot
        self.controller = controller or ControllerAdapter()
        self.camera_config = camera_config
        self.camera = None
        self._stage_callback = stage_callback or (lambda stage: None)
        self.frame_pipeline = frame_pipeline
        self.last_video_error = None
        self.table_actor = None
        self.object_actor = None
        self.goal_actor = None
        self.last_target: Optional[EETarget] = None
        self.last_command: Optional[JointPositionCommand] = None
        self._root_state = None
        self._object_actor_index = None
        self._object_actor_index_tensor = None
        self._net_contact_force = None
        self._contact_body_indices = ()
        self._object_body_index = None

    def setup(self) -> None:
        try:
            self._stage_callback("sim")
            self.robot.initialize()
            self._stage_callback("asset")
            self.robot.load_asset()
            self._stage_callback("env")
            self.robot.create_env(self.config.robot_position, self.config.robot_yaw)
            self._stage_callback("scene")
            self._create_scene()
            self._stage_callback("camera")
            self._create_camera()
            self._stage_callback("prepare")
            self.robot.prepare_and_acquire()
            self._stage_callback("scene_tensors")
            self._acquire_scene_tensors()
            self._stage_callback("ready")
        except Exception:
            self.close()
            raise

    def _create_scene(self) -> None:
        gymapi = self.robot._gymapi
        gym = self.robot.gym
        table_options = gymapi.AssetOptions()
        table_options.fix_base_link = True
        table_asset = gym.create_box(self.robot.sim, *self.config.table_size, table_options)
        object_options = gymapi.AssetOptions()
        object_asset = gym.create_box(self.robot.sim, *self.config.object_size, object_options)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(*self.config.table_position)
        object_pose = gymapi.Transform()
        object_pose.p = gymapi.Vec3(*self.config.object_start)
        self.table_actor = gym.create_actor(
            self.robot.env, table_asset, table_pose, self.TABLE_NAME,
            self.PHYSICS_COLLISION_GROUP, 0
        )
        self.object_actor = gym.create_actor(
            self.robot.env, object_asset, object_pose, self.OBJECT_NAME,
            self.PHYSICS_COLLISION_GROUP, 0
        )
        if self.camera_config is not None:
            goal_options = gymapi.AssetOptions()
            goal_options.fix_base_link = True
            goal_asset = gym.create_box(self.robot.sim, 0.10, 0.10, 0.006, goal_options)
            goal_pose = gymapi.Transform()
            goal_pose.p = gymapi.Vec3(
                self.config.goal_position[0], self.config.goal_position[1],
                self.config.table_position[2] + self.config.table_size[2] / 2 + 0.004,
            )
            self.goal_actor = gym.create_actor(
                self.robot.env, goal_asset, goal_pose, self.GOAL_NAME,
                self.GOAL_VISUAL_COLLISION_GROUP, 0
            )
            gym.set_rigid_body_color(
                self.robot.env, self.goal_actor, 0,
                gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.15, 0.8, 0.2),
            )
            gym.set_rigid_body_color(
                self.robot.env, self.object_actor, 0,
                gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.85, 0.15, 0.1),
            )
        if (
            self.table_actor is None
            or self.table_actor < 0
            or self.object_actor is None
            or self.object_actor < 0
        ):
            raise PushAdapterError("Isaac Gym failed to create the Push scene actors")

    def _create_camera(self) -> None:
        if self.camera_config is None:
            return
        if self.robot.config.graphics_device_id < 0:
            raise PushAdapterError(
                "Camera Sensor requires a non-negative graphics_device_id"
            )
        self.camera = IsaacGymCamera(
            self.robot.gym,
            self.robot.sim,
            self.robot.env,
            self.robot._gymapi,
            self.robot._gymtorch,
            self.camera_config,
        ).initialize()

    def capture_camera_frame(self):
        if self.camera is None:
            raise PushAdapterError("camera is disabled")
        return self.camera.capture()

    def maybe_capture_frame(self, now: Optional[float] = None):
        """Best-effort video production; video errors never alter Push state."""
        if self.camera is None or self.frame_pipeline is None:
            return None
        try:
            return self.frame_pipeline.maybe_produce(self.capture_camera_frame, now)
        except Exception as exc:
            self.last_video_error = exc
            return None

    def _acquire_scene_tensors(self) -> None:
        import torch

        descriptor = self.robot.gym.acquire_actor_root_state_tensor(self.robot.sim)
        self._root_state = self.robot._gymtorch.wrap_tensor(descriptor).view(-1, 13)
        contact_descriptor = self.robot.gym.acquire_net_contact_force_tensor(self.robot.sim)
        self._net_contact_force = self.robot._gymtorch.wrap_tensor(
            contact_descriptor
        ).view(-1, 3)
        self._object_actor_index = self.robot.gym.get_actor_index(
            self.robot.env,
            self.object_actor,
            self.robot._gymapi.IndexDomain.DOMAIN_SIM,
        )
        if (
            self._object_actor_index < 0
            or self._object_actor_index >= self._root_state.shape[0]
        ):
            raise PushAdapterError("object actor index is outside root-state tensor")
        self._object_actor_index_tensor = self._root_state.new_tensor(
            [self._object_actor_index], dtype=torch.int32
        )
        body_indices = self.robot.body_indices
        self._contact_body_indices = tuple(
            body_indices[name] for name in ("ee_link", "7_Link", "8_Link")
        )
        self._object_body_index = self.robot.gym.get_actor_rigid_body_index(
            self.robot.env, self.object_actor, 0,
            self.robot._gymapi.IndexDomain.DOMAIN_SIM,
        )
        indexed_bodies = tuple(zip(
            ("ee_link", "7_Link", "8_Link"), self._contact_body_indices
        )) + (("push_object", self._object_body_index),)
        for name, index in indexed_bodies:
            if index < 0 or index >= self._net_contact_force.shape[0]:
                raise PushAdapterError("%s body index is outside contact tensor" % name)

    def reset(self) -> None:
        self.robot.reset_dofs()
        self.robot.gym.refresh_actor_root_state_tensor(self.robot.sim)
        self._root_state[self._object_actor_index, :3] = self._root_state.new_tensor(
            self.config.object_start
        )
        self._root_state[self._object_actor_index, 3:7] = self._root_state.new_tensor(
            (0.0, 0.0, 0.0, 1.0)
        )
        self._root_state[self._object_actor_index, 7:13] = 0.0
        self.robot.gym.set_actor_root_state_tensor_indexed(
            self.robot.sim,
            self.robot._gymtorch.unwrap_tensor(self._root_state),
            self.robot._gymtorch.unwrap_tensor(self._object_actor_index_tensor),
            1,
        )
        self.last_target = None
        self.last_command = None
        self.robot.step(1)

    def read_robot_state(self) -> S1State:
        return self.robot.read_state()

    def read_object_state(self) -> Tuple[float, ...]:
        self.robot.gym.refresh_actor_root_state_tensor(self.robot.sim)
        return _tuple(self._root_state[self._object_actor_index].detach().clone())

    def has_object_contact(self) -> bool:
        """Detect finger/object contact without treating finger self-contact as object contact.

        Preview 4's rigid-contact-pair getter is unavailable after a GPU
        pipeline starts.  The fallback therefore combines finger/cube spatial
        proximity with cube horizontal force and motion.  Finger net-force is
        deliberately not evidence: q7=q8=0 demonstrated multi-kN self-contact.
        """
        self.robot.gym.refresh_net_contact_force_tensor(self.robot.sim)
        object_state = self.read_object_state()
        object_position = object_state[:3]
        finger_distances = []
        self.robot._refresh_tensors()
        for index in self._contact_body_indices[1:]:  # 7_Link and 8_Link only
            position = self.robot._rigid_body_state[index, :3]
            outside_sq = 0.0
            for axis in range(3):
                outside = max(
                    abs(float(position[axis]) - object_position[axis])
                    - self.config.object_size[axis] / 2.0,
                    0.0,
                )
                outside_sq += outside * outside
            finger_distances.append(math.sqrt(outside_sq))
        # Each finger collision mesh extends about 90 mm from its body origin.
        finger_near_cube = min(finger_distances) <= 0.10
        object_force = self._net_contact_force[self._object_body_index]
        horizontal_force = math.sqrt(
            float(object_force[0]) ** 2 + float(object_force[1]) ** 2
        )
        horizontal_speed = math.sqrt(
            object_state[7] ** 2 + object_state[8] ** 2
        )
        return bool(
            finger_near_cube
            and horizontal_force >= self.config.contact_force_threshold
            and horizontal_speed >= 1.0e-3
        )

    def set_ee_target(self, target: EETarget) -> bool:
        self.last_target = target
        self.last_command = self.controller.compute(target, self.robot.read_state())
        self.robot.set_joint_position_targets(self.last_command.positions)
        return True

    def step(self, n: int) -> None:
        self.robot.step(n)

    def close(self) -> None:
        if self.camera is not None:
            self.camera.close()
            self.camera = None
        self.last_target = None
        self.last_command = None
        self.object_actor = None
        self.table_actor = None
        self.goal_actor = None
        self.last_video_error = None
        self._root_state = None
        self._object_actor_index = None
        self._object_actor_index_tensor = None
        self._net_contact_force = None
        self._contact_body_indices = ()
        self._object_body_index = None
        self.robot.close()
