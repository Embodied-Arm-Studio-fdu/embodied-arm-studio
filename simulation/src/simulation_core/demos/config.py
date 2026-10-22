"""Strict typed configuration for the four M3 S1 demos."""

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Tuple


@dataclass(frozen=True)
class S1DemoConfig:
    task_type: str
    environment_type: str
    duration_seconds: float = 30.0
    seed: int = 7
    physics_steps_per_command: int = 1
    video_enabled: bool = True
    camera_id: str = "overview"
    width: int = 640
    height: int = 480
    fps: float = 15.0

    def __post_init__(self):
        if self.task_type != "demo": raise ValueError("task_type must be 'demo'")
        if not isinstance(self.environment_type, str) or not self.environment_type: raise ValueError("environment_type is required")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.duration_seconds, (int, float)) or isinstance(self.duration_seconds, bool) or not 5 <= float(self.duration_seconds) <= 300:
            raise ValueError("duration_seconds must be within [5, 300]")
        if isinstance(self.physics_steps_per_command, bool) or not isinstance(self.physics_steps_per_command, int) or not 1 <= self.physics_steps_per_command <= 16:
            raise ValueError("physics_steps_per_command must be within [1, 16]")
        if not isinstance(self.video_enabled, bool): raise ValueError("video_enabled must be boolean")
        if not isinstance(self.camera_id, str) or not self.camera_id: raise ValueError("camera_id is required")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (self.width, self.height)):
            raise ValueError("video dimensions must be integers")
        if self.width < 160 or self.width > 1920 or self.height < 120 or self.height > 1080:
            raise ValueError("video dimensions are outside supported range")
        if not isinstance(self.fps, (int, float)) or isinstance(self.fps, bool) or not 1 <= float(self.fps) <= 60:
            raise ValueError("fps must be within [1, 60]")

    @property
    def tick_interval_seconds(self): return 1.0 / 60.0
    @property
    def video_width(self): return self.width
    @property
    def video_height(self): return self.height
    @property
    def video_fps(self): return self.fps
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class TargetPoseDemoConfig(S1DemoConfig):
    target_offset_xy_m: Tuple[float, float] = (0.08, 0.05)
    def __post_init__(self):
        super().__post_init__(); _pair("target_offset_xy_m", self.target_offset_xy_m, -0.15, 0.15)


@dataclass(frozen=True)
class LineTrackingDemoConfig(S1DemoConfig):
    line_length_m: float = 0.15
    period_seconds: float = 6.0
    direction: str = "x"
    def __post_init__(self):
        super().__post_init__(); _finite_range("line_length_m", self.line_length_m, 0.02, 0.18); _finite_range("period_seconds", self.period_seconds, 2, 30)
        if self.direction not in ("x", "y"): raise ValueError("direction must be 'x' or 'y'")


@dataclass(frozen=True)
class CircleTrackingDemoConfig(S1DemoConfig):
    radius_m: float = 0.05
    period_seconds: float = 8.0
    def __post_init__(self):
        super().__post_init__(); _finite_range("radius_m", self.radius_m, 0.01, 0.08); _finite_range("period_seconds", self.period_seconds, 2, 30)


SAFE_JOINT_POSES = {
    # Demo-local rest pose, based on the S1 SDK's documented q_home.  It does
    # not alter Push/Reach reset constants or S1IsaacGymConfig defaults.
    "home": (0.0, 0.24, 0.16, -0.33, 0.0, 0.0, 0.01, 0.01),
    "ready": (-0.2351, 1.7607, 0.8575, 0.4088, 0.0926, 0.0121, 0.01, 0.01),
    "side": (0.30, 1.45, 1.05, 0.20, 0.0, 0.0, 0.01, 0.01),
}


@dataclass(frozen=True)
class JointPositionDemoConfig(S1DemoConfig):
    pose: str = "ready"
    def __post_init__(self):
        super().__post_init__()
        if self.pose not in SAFE_JOINT_POSES: raise ValueError("pose must be one of: home, ready, side")


@dataclass(frozen=True)
class JointSweepDemoConfig(S1DemoConfig):
    joint_index: object = "all_arm"
    speed_scale: float = 0.2
    def __post_init__(self):
        super().__post_init__()
        if self.joint_index != "all_arm" and (
            isinstance(self.joint_index, bool)
            or not isinstance(self.joint_index, int)
            or not 0 <= self.joint_index <= 5
        ):
            raise ValueError("joint_index must be 'all_arm' or an integer within [0, 5]")
        _finite_range("speed_scale", self.speed_scale, 0.05, 0.5)


@dataclass(frozen=True)
class ScriptedPickPlaceDemoConfig(S1DemoConfig):
    def __post_init__(self):
        super().__post_init__()
        if self.physics_steps_per_command != 4:
            raise ValueError("scripted pick-and-place requires physics_steps_per_command=4")
        if self.width != 640 or self.height != 480:
            raise ValueError("scripted pick-and-place video is fixed at 640x480")


def _finite_range(name, value, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not low <= float(value) <= high:
        raise ValueError("%s must be within [%s, %s]" % (name, low, high))


def _pair(name, value, low, high):
    if not isinstance(value, (tuple, list)) or len(value) != 2: raise ValueError("%s must contain two values" % name)
    for item in value: _finite_range(name, item, low, high)


def _parse(cls, task_id, values: Mapping):
    data = dict(values)
    if data.get("environment_type") != task_id: raise ValueError("environment_type must be %s" % task_id)
    try: return cls(**data)
    except TypeError as exc: raise ValueError("invalid %s configuration: %s" % (task_id, exc)) from exc


def parse_target_pose(v): return _parse(TargetPoseDemoConfig, "demo_s1_target_pose", v)
def parse_line_tracking(v): return _parse(LineTrackingDemoConfig, "demo_s1_line_tracking", v)
def parse_circle_tracking(v): return _parse(CircleTrackingDemoConfig, "demo_s1_circle_tracking", v)
def parse_joint_position(v): return _parse(JointPositionDemoConfig, "demo_s1_joint_position", v)
def parse_joint_sweep(v): return _parse(JointSweepDemoConfig, "demo_s1_joint_sweep", v)
def parse_scripted_pick_place(v): return _parse(
    ScriptedPickPlaceDemoConfig, "demo_s1_scripted_pick_place", v
)
