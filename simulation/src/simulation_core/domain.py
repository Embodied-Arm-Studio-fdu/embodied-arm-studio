"""Simulation task domain model and centralized state-transition rules."""

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple


SMOKE_RUN_MODE = "smoke"
TRAINING_RUN_MODE = "training"
SMOKE_EPISODE_STEPS = 64
TRAINING_EPISODE_STEPS = 256
TRAINING_PHYSICS_STEPS_PER_ACTION = 8
REACH_TRAINING_EPISODE_STEPS = 128
MANIPULATION_TASK_IDS = frozenset({"s1_grasp", "s1_lift", "s1_lower_release", "s1_place"})


class TaskStatus(str, Enum):
    QUEUED = "queued"
    INITIALIZING = "initializing"
    RUNNING = "running"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


TERMINAL_STATUSES: FrozenSet[TaskStatus] = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.STOPPED}
)

ALLOWED_TRANSITIONS = {
    TaskStatus.QUEUED: frozenset({TaskStatus.INITIALIZING, TaskStatus.STOPPED}),
    TaskStatus.INITIALIZING: frozenset({TaskStatus.RUNNING, TaskStatus.FAILED}),
    TaskStatus.RUNNING: frozenset(
        {TaskStatus.FINALIZING, TaskStatus.STOPPING, TaskStatus.FAILED}
    ),
    TaskStatus.FINALIZING: frozenset({TaskStatus.COMPLETED, TaskStatus.FAILED}),
    TaskStatus.STOPPING: frozenset({TaskStatus.STOPPED}),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.STOPPED: frozenset(),
    TaskStatus.FAILED: frozenset(),
}


class InvalidStateTransition(ValueError):
    """Raised when a task is moved through a transition outside the specification."""

    def __init__(self, current: TaskStatus, target: TaskStatus) -> None:
        super().__init__(
            "invalid task status transition: %s -> %s" % (current.value, target.value)
        )
        self.current = current
        self.target = target


class TaskNotFound(LookupError):
    """Raised when a task identifier does not exist in the repository."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class TaskConfig:
    """Opaque validated configuration snapshot for later task-specific adapters."""

    values: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TaskConfig":
        # Recursively copying prevents a caller from changing an accepted snapshot.
        from copy import deepcopy

        return cls(values=deepcopy(dict(values)))


@dataclass(frozen=True)
class TrainingTaskConfig:
    """Serializable configuration needed to execute the current Push PPO task."""

    task_type: str
    environment_type: str
    seed: int
    rollout_steps: int
    num_updates: int
    checkpoint_every: int
    episode_steps: int
    physics_steps_per_action: int
    update_epochs: int = 1
    batch_size: int = 16
    device: str = "cuda:0"
    video_enabled: bool = False
    camera_id: str = "overview"
    video_fps: float = 15.0
    video_width: int = 640
    video_height: int = 480
    run_mode: str = SMOKE_RUN_MODE
    object_size: Optional[Tuple[float, float, float]] = None
    object_start: Optional[Tuple[float, float, float]] = None
    goal_position: Optional[Tuple[float, float, float]] = None
    reward_extension: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.task_type != "training":
            raise ValueError("task_type must be 'training'")
        if self.environment_type != "s1_push":
            raise ValueError("environment_type must be 's1_push'")
        if self.reward_extension is not None:
            from .reward_extension import validate_extension
            validate_extension(self.environment_type, self.reward_extension)
        if self.run_mode not in (SMOKE_RUN_MODE, TRAINING_RUN_MODE):
            raise ValueError("run_mode must be 'smoke' or 'training'")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in (
            "rollout_steps",
            "num_updates",
            "checkpoint_every",
            "episode_steps",
            "physics_steps_per_action",
            "update_epochs",
            "batch_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("%s must be a positive integer" % name)
        if self.device != "cuda:0":
            raise ValueError("Phase 4.1 only supports device 'cuda:0'")
        if not isinstance(self.video_enabled, bool):
            raise ValueError("video_enabled must be boolean")
        if not isinstance(self.camera_id, str) or not self.camera_id:
            raise ValueError("camera_id is required")
        for name in ("video_width", "video_height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("%s must be a positive integer" % name)
        if isinstance(self.video_fps, bool) or self.video_fps <= 0:
            raise ValueError("video_fps must be positive")
        for name in ("object_size", "object_start", "goal_position"):
            value = getattr(self, name)
            if value is None:
                continue
            normalized = tuple(float(item) for item in value)
            if len(normalized) != 3 or not all(math.isfinite(item) for item in normalized):
                raise ValueError("%s must contain 3 finite values" % name)
            if name == "object_size" and not all(item > 0 for item in normalized):
                raise ValueError("object_size values must be positive")
            object.__setattr__(self, name, normalized)
        if (
            self.run_mode == TRAINING_RUN_MODE
            and self.episode_steps != TRAINING_EPISODE_STEPS
        ):
            raise ValueError(
                "training run_mode requires episode_steps=%d"
                % TRAINING_EPISODE_STEPS
            )
        if (
            self.run_mode == TRAINING_RUN_MODE
            and self.physics_steps_per_action != TRAINING_PHYSICS_STEPS_PER_ACTION
        ):
            raise ValueError(
                "training run_mode requires physics_steps_per_action=%d"
                % TRAINING_PHYSICS_STEPS_PER_ACTION
            )

    @classmethod
    def from_task(cls, task: "SimulationTask") -> "TrainingTaskConfig":
        """Validate and translate a task's JSON-compatible config snapshot."""
        try:
            return cls(**dict(task.config.values))
        except TypeError as exc:
            raise ValueError("invalid training task configuration: %s" % exc) from exc

    def to_dict(self) -> Dict[str, Any]:
        """Return a backend-JSON-compatible representation."""
        values = {
            "task_type": self.task_type,
            "environment_type": self.environment_type,
            "seed": self.seed,
            "rollout_steps": self.rollout_steps,
            "num_updates": self.num_updates,
            "checkpoint_every": self.checkpoint_every,
            "episode_steps": self.episode_steps,
            "physics_steps_per_action": self.physics_steps_per_action,
            "update_epochs": self.update_epochs,
            "batch_size": self.batch_size,
            "device": self.device,
            "video_enabled": self.video_enabled,
            "camera_id": self.camera_id,
            "video_fps": self.video_fps,
            "video_width": self.video_width,
            "video_height": self.video_height,
            "run_mode": self.run_mode,
        }
        for name in ("object_size", "object_start", "goal_position"):
            value = getattr(self, name)
            if value is not None:
                values[name] = list(value)
        if self.reward_extension is not None:
            values["reward_extension"] = dict(self.reward_extension)
        return values


@dataclass(frozen=True)
class ReachTrainingTaskConfig:
    """Validated execution configuration for the S1 planar Reach task."""

    task_type: str
    environment_type: str
    seed: int
    rollout_steps: int
    num_updates: int
    checkpoint_every: int
    episode_steps: int
    physics_steps_per_action: int
    update_epochs: int = 1
    batch_size: int = 16
    device: str = "cuda:0"
    video_enabled: bool = False
    camera_id: str = "overview"
    video_fps: float = 15.0
    video_width: int = 640
    video_height: int = 480
    run_mode: str = SMOKE_RUN_MODE
    action_delta_m: float = 0.005
    target_center_xy_m: Tuple[float, float] = (0.36, 0.0)
    target_half_extent_xy_m: Tuple[float, float] = (0.03, 0.08)
    success_distance_m: float = 0.02
    success_hold_steps: int = 3
    reward_extension: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.task_type != "training":
            raise ValueError("task_type must be 'training'")
        if self.environment_type != "s1_reach":
            raise ValueError("environment_type must be 's1_reach'")
        if self.reward_extension is not None:
            from .reward_extension import validate_extension
            validate_extension(self.environment_type, self.reward_extension)
        if self.run_mode not in (SMOKE_RUN_MODE, TRAINING_RUN_MODE):
            raise ValueError("run_mode must be 'smoke' or 'training'")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in (
            "rollout_steps", "num_updates", "checkpoint_every", "episode_steps",
            "physics_steps_per_action", "update_epochs", "batch_size",
            "success_hold_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("%s must be a positive integer" % name)
        if self.device != "cuda:0":
            raise ValueError("Reach production only supports device 'cuda:0'")
        if not isinstance(self.video_enabled, bool):
            raise ValueError("video_enabled must be boolean")
        if not isinstance(self.camera_id, str) or not self.camera_id:
            raise ValueError("camera_id is required")
        for name in ("video_width", "video_height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("%s must be a positive integer" % name)
        for name in ("video_fps", "action_delta_m", "success_distance_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("%s must be numeric" % name)
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError("%s must be positive and finite" % name)
        if not 0.001 <= float(self.action_delta_m) <= 0.01:
            raise ValueError("action_delta_m must be in [0.001, 0.01]")
        if not 0.01 <= float(self.success_distance_m) <= 0.05:
            raise ValueError("success_distance_m must be in [0.01, 0.05]")
        for name in ("target_center_xy_m", "target_half_extent_xy_m"):
            value = tuple(float(item) for item in getattr(self, name))
            if len(value) != 2 or not all(math.isfinite(item) for item in value):
                raise ValueError("%s must contain 2 finite values" % name)
            object.__setattr__(self, name, value)
        if not all(0.0 < item <= 0.15 for item in self.target_half_extent_xy_m):
            raise ValueError("target_half_extent_xy_m values must be in (0, 0.15]")
        if self.success_hold_steps >= self.episode_steps:
            raise ValueError("success_hold_steps must be smaller than episode_steps")
        if self.run_mode == TRAINING_RUN_MODE:
            if self.episode_steps != REACH_TRAINING_EPISODE_STEPS:
                raise ValueError(
                    "Reach training requires episode_steps=%d"
                    % REACH_TRAINING_EPISODE_STEPS
                )
            if self.physics_steps_per_action != TRAINING_PHYSICS_STEPS_PER_ACTION:
                raise ValueError(
                    "Reach training requires physics_steps_per_action=%d"
                    % TRAINING_PHYSICS_STEPS_PER_ACTION
                )

    def to_dict(self) -> Dict[str, Any]:
        values = {
            "task_type": self.task_type,
            "environment_type": self.environment_type,
            "seed": self.seed,
            "rollout_steps": self.rollout_steps,
            "num_updates": self.num_updates,
            "checkpoint_every": self.checkpoint_every,
            "episode_steps": self.episode_steps,
            "physics_steps_per_action": self.physics_steps_per_action,
            "update_epochs": self.update_epochs,
            "batch_size": self.batch_size,
            "device": self.device,
            "video_enabled": self.video_enabled,
            "camera_id": self.camera_id,
            "video_fps": float(self.video_fps),
            "video_width": self.video_width,
            "video_height": self.video_height,
            "run_mode": self.run_mode,
            "action_delta_m": float(self.action_delta_m),
            "target_center_xy_m": list(self.target_center_xy_m),
            "target_half_extent_xy_m": list(self.target_half_extent_xy_m),
            "success_distance_m": float(self.success_distance_m),
            "success_hold_steps": self.success_hold_steps,
        }
        if self.reward_extension is not None:
            values["reward_extension"] = dict(self.reward_extension)
        return values


@dataclass(frozen=True)
class ManipulationTrainingTaskConfig:
    """Validated platform wrapper for the three trained manipulation policies."""

    task_type: str
    environment_type: str
    seed: int
    rollout_steps: int
    num_updates: int
    checkpoint_every: int
    episode_steps: int
    physics_steps_per_action: int
    update_epochs: int = 1
    batch_size: int = 16
    device: str = "cuda:0"
    video_enabled: bool = True
    camera_id: str = "overview"
    video_fps: float = 15.0
    video_width: int = 640
    video_height: int = 480
    run_mode: str = SMOKE_RUN_MODE
    randomize_xy_m: float = 0.0
    action_delta_m: float = 0.005
    hold_seconds: float = 1.0
    release_trigger_threshold: float = 0.0
    reward_extension: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.task_type != "training":
            raise ValueError("task_type must be 'training'")
        if self.environment_type not in MANIPULATION_TASK_IDS:
            raise ValueError("unsupported manipulation environment_type")
        if self.run_mode not in (SMOKE_RUN_MODE, TRAINING_RUN_MODE):
            raise ValueError("run_mode must be 'smoke' or 'training'")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in (
            "rollout_steps", "num_updates", "checkpoint_every", "episode_steps",
            "physics_steps_per_action", "update_epochs", "batch_size",
            "video_width", "video_height",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("%s must be a positive integer" % name)
        if self.physics_steps_per_action != 4:
            raise ValueError("manipulation tasks require physics_steps_per_action=4")
        if self.device != "cuda:0":
            raise ValueError("manipulation training only supports device 'cuda:0'")
        if not isinstance(self.video_enabled, bool):
            raise ValueError("video_enabled must be boolean")
        if self.camera_id != "overview":
            raise ValueError("manipulation tasks only support the overview camera")
        if (self.video_width, self.video_height) != (640, 480):
            raise ValueError("manipulation video is fixed at 640x480")
        for name in (
            "video_fps", "randomize_xy_m", "action_delta_m", "hold_seconds",
            "release_trigger_threshold",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("%s must be numeric" % name)
            if not math.isfinite(float(value)):
                raise ValueError("%s must be finite" % name)
        if self.video_fps <= 0:
            raise ValueError("video_fps must be positive")
        if not 0.0 <= self.randomize_xy_m <= 0.006:
            raise ValueError("randomize_xy_m must be in [0, 0.006]")
        if not 0.0025 <= self.action_delta_m <= 0.005:
            raise ValueError("action_delta_m must be in [0.0025, 0.005]")
        if self.hold_seconds not in (1.0, 5.0):
            raise ValueError("hold_seconds must be 1 or 5")
        if not -1.0 < self.release_trigger_threshold < 1.0:
            raise ValueError("release_trigger_threshold must be inside (-1, 1)")
        if self.reward_extension is not None:
            from .reward_extension import validate_extension
            validate_extension(self.environment_type, self.reward_extension)

    def to_dict(self) -> Dict[str, Any]:
        values = {
            "task_type": self.task_type,
            "environment_type": self.environment_type,
            "seed": self.seed,
            "rollout_steps": self.rollout_steps,
            "num_updates": self.num_updates,
            "checkpoint_every": self.checkpoint_every,
            "episode_steps": self.episode_steps,
            "physics_steps_per_action": self.physics_steps_per_action,
            "update_epochs": self.update_epochs,
            "batch_size": self.batch_size,
            "device": self.device,
            "video_enabled": self.video_enabled,
            "camera_id": self.camera_id,
            "video_fps": float(self.video_fps),
            "video_width": self.video_width,
            "video_height": self.video_height,
            "run_mode": self.run_mode,
            "randomize_xy_m": float(self.randomize_xy_m),
            "action_delta_m": float(self.action_delta_m),
            "hold_seconds": float(self.hold_seconds),
            "release_trigger_threshold": float(self.release_trigger_threshold),
        }
        if self.reward_extension is not None:
            values["reward_extension"] = dict(self.reward_extension)
        return values


@dataclass(frozen=True)
class DemoTaskConfig:
    """Serializable configuration shared by future non-training demos.

    M1 deliberately registers no production demo.  This contract exists so a
    demo can use the normal task/worker lifecycle without inheriting PPO fields.
    """

    task_type: str
    environment_type: str
    seed: int = 7
    duration_seconds: float = 30.0
    tick_interval_seconds: float = 0.1
    video_enabled: bool = False
    camera_id: str = "overview"
    video_fps: float = 15.0
    video_width: int = 640
    video_height: int = 480

    def __post_init__(self) -> None:
        if self.task_type != "demo":
            raise ValueError("task_type must be 'demo'")
        if not isinstance(self.environment_type, str) or not self.environment_type:
            raise ValueError("environment_type is required")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in ("duration_seconds", "tick_interval_seconds", "video_fps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("%s must be numeric" % name)
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError("%s must be positive and finite" % name)
        if not isinstance(self.video_enabled, bool):
            raise ValueError("video_enabled must be boolean")
        if not isinstance(self.camera_id, str) or not self.camera_id:
            raise ValueError("camera_id is required")
        for name in ("video_width", "video_height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("%s must be a positive integer" % name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "environment_type": self.environment_type,
            "seed": self.seed,
            "duration_seconds": float(self.duration_seconds),
            "tick_interval_seconds": float(self.tick_interval_seconds),
            "video_enabled": self.video_enabled,
            "camera_id": self.camera_id,
            "video_fps": float(self.video_fps),
            "video_width": self.video_width,
            "video_height": self.video_height,
        }


@dataclass(frozen=True)
class SimulationTask:
    task_id: str
    run_id: str
    project_id: str
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    config: TaskConfig

    def transition_to(
        self, target: TaskStatus, updated_at: Optional[datetime] = None
    ) -> "SimulationTask":
        if target not in ALLOWED_TRANSITIONS[self.status]:
            raise InvalidStateTransition(self.status, target)
        return replace(self, status=target, updated_at=updated_at or utc_now())

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES
