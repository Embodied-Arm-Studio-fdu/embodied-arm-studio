"""Single-writer composition of S1 arm-DLS and symmetric jaw targets."""

from dataclasses import dataclass
import math
from typing import Any, Sequence, Tuple

from ...cartesian_controller import ControllerAdapter
from ...push_adapter import EETarget


@dataclass(frozen=True)
class Workspace3D:
    lower: Tuple[float, float, float] = (0.32, -0.12, 0.34)
    upper: Tuple[float, float, float] = (0.50, 0.12, 0.62)

    def __post_init__(self):
        object.__setattr__(self, "lower", tuple(float(v) for v in self.lower))
        object.__setattr__(self, "upper", tuple(float(v) for v in self.upper))
        if len(self.lower) != 3 or len(self.upper) != 3:
            raise ValueError("workspace bounds must be 3D")
        if not all(math.isfinite(v) for v in self.lower + self.upper):
            raise ValueError("workspace bounds must be finite")
        if any(lo >= hi for lo, hi in zip(self.lower, self.upper)):
            raise ValueError("workspace lower bounds must be below upper bounds")

    def clamp(self, xyz: Sequence[float]):
        values = tuple(float(v) for v in xyz)
        if len(values) != 3 or not all(math.isfinite(v) for v in values):
            raise ValueError("XYZ target must contain three finite values")
        clamped = tuple(max(lo, min(hi, value)) for value, lo, hi in zip(values, self.lower, self.upper))
        return clamped, clamped != values

    def contains(self, xyz: Sequence[float], tolerance: float = 0.0) -> bool:
        return all(lo - tolerance <= float(v) <= hi + tolerance
                   for v, lo, hi in zip(xyz, self.lower, self.upper))


@dataclass(frozen=True)
class CompositeCommand:
    positions: Any
    cartesian_error: Any
    arm_joint_delta: Any
    jaw_target: float
    workspace_clamped: bool
    joint_delta_clamped: bool
    joint_limit_clamped: bool


class ExperimentalTargetCompositor:
    """Own the only 8D position-target write for experimental manipulation."""

    def __init__(self, controller=None, workspace=None, jaw_rate_limit=0.002):
        self.controller = controller or ControllerAdapter()
        self.workspace = workspace or Workspace3D()
        self.jaw_rate_limit = float(jaw_rate_limit)
        if not math.isfinite(self.jaw_rate_limit) or self.jaw_rate_limit <= 0:
            raise ValueError("jaw_rate_limit must be finite and positive")
        self._commanded_jaw = None

    def reset(self, actual_q: Sequence[float]):
        values = tuple(float(v) for v in actual_q)
        if len(values) != 8 or not all(math.isfinite(v) for v in values):
            raise ValueError("actual_q must contain eight finite values")
        self._commanded_jaw = 0.5 * (values[6] + values[7])

    def compose(self, xyz, orientation_xyzw, desired_jaw, state):
        import torch

        if self._commanded_jaw is None:
            self.reset(state.dof_positions)
        target_xyz, workspace_clamped = self.workspace.clamp(xyz)
        desired_jaw = float(desired_jaw)
        if not math.isfinite(desired_jaw) or not 0.0 <= desired_jaw <= 0.05:
            raise ValueError("desired jaw target must be within [0, 0.05] m")
        change = max(-self.jaw_rate_limit, min(self.jaw_rate_limit,
                                               desired_jaw - self._commanded_jaw))
        self._commanded_jaw += change
        arm = self.controller.compute(EETarget(target_xyz, tuple(orientation_xyzw)), state)
        positions = torch.as_tensor(arm.positions).detach().clone()
        positions[6:] = self._commanded_jaw
        delta_limit = self.controller.config.max_joint_delta
        joint_delta_clamped = bool(torch.any(torch.abs(arm.joint_delta) >= delta_limit * (1.0 - 1e-5)).item())
        lower = positions.new_tensor(self.controller.config.lower_limits[:6])
        upper = positions.new_tensor(self.controller.config.upper_limits[:6])
        joint_limit_clamped = bool(torch.any(
            (positions[:6] <= lower + 1e-6) | (positions[:6] >= upper - 1e-6)
        ).item())
        return CompositeCommand(
            positions=positions,
            cartesian_error=arm.cartesian_error,
            arm_joint_delta=arm.joint_delta,
            jaw_target=self._commanded_jaw,
            workspace_clamped=workspace_clamped,
            joint_delta_clamped=joint_delta_clamped,
            joint_limit_clamped=joint_limit_clamped,
        )
