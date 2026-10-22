"""Small Cartesian-to-joint controller for the virtual S1.

This first controller uses damped least-squares differential IK and emits an
eight-DOF position target.  It is intentionally independent of PushAdapter and
does not implement DyWA's mass-matrix Cartesian impedance controller.
"""

from dataclasses import dataclass
import math
from typing import Any, Sequence, Tuple

from .s1_isaac_gym_adapter import ARM_DOF_COUNT, S1State


S1_LOWER_LIMITS = (-2.967, 0.0, 0.0, -1.57, -1.57, -1.57, 0.0, 0.0)
S1_UPPER_LIMITS = (2.967, 3.14, 3.14, 1.518, 1.57, 1.57, 0.05, 0.05)


@dataclass(frozen=True)
class CartesianControllerConfig:
    damping: float = 0.08
    position_gain: float = 1.0
    orientation_gain: float = 0.35
    max_joint_delta: float = 0.08
    lower_limits: Tuple[float, ...] = S1_LOWER_LIMITS
    upper_limits: Tuple[float, ...] = S1_UPPER_LIMITS

    def __post_init__(self) -> None:
        object.__setattr__(self, "lower_limits", tuple(self.lower_limits))
        object.__setattr__(self, "upper_limits", tuple(self.upper_limits))
        if self.damping <= 0 or not math.isfinite(self.damping):
            raise ValueError("damping must be positive and finite")
        if self.position_gain <= 0 or self.orientation_gain < 0:
            raise ValueError("controller gains must be non-negative and position_gain positive")
        if self.max_joint_delta <= 0 or not math.isfinite(self.max_joint_delta):
            raise ValueError("max_joint_delta must be positive and finite")
        if len(self.lower_limits) != 8 or len(self.upper_limits) != 8:
            raise ValueError("joint limits must contain exactly 8 values")
        if any(lower > upper for lower, upper in zip(self.lower_limits, self.upper_limits)):
            raise ValueError("each lower joint limit must not exceed its upper limit")


@dataclass(frozen=True)
class JointPositionCommand:
    """Eight S1 joint position targets; final two preserve the gripper."""

    positions: Any
    cartesian_error: Any
    joint_delta: Any


def _quaternion_conjugate(quaternion):
    x, y, z, w = quaternion
    return (-x, -y, -z, w)


def _quaternion_multiply(left, right):
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _orientation_error(current, target):
    delta = _quaternion_multiply(target, _quaternion_conjugate(current))
    vector = delta[:3]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm < 1.0e-12:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(norm, delta[3])
    if angle > math.pi:
        angle -= 2.0 * math.pi
    return tuple(value * angle / norm for value in vector)


class ControllerAdapter:
    """Convert a Cartesian pose target to a bounded S1 position command."""

    def __init__(self, config: CartesianControllerConfig = None) -> None:
        self.config = config or CartesianControllerConfig()

    def compute(self, target: Any, state: S1State) -> JointPositionCommand:
        import torch

        q = torch.as_tensor(state.dof_positions)
        jacobian = torch.as_tensor(state.arm_ee_jacobian, device=q.device, dtype=q.dtype)
        ee = torch.as_tensor(state.ee_link_state, device=q.device, dtype=q.dtype)
        if tuple(q.shape) != (8,):
            raise ValueError("S1 DOF positions must have shape (8,)")
        if tuple(jacobian.shape) != (6, ARM_DOF_COUNT):
            raise ValueError("S1 arm EE Jacobian must have shape (6, 6)")
        if tuple(ee.shape) != (13,):
            raise ValueError("ee_link state must have shape (13,)")

        position_error = tuple(
            float(target.position[index]) - float(ee[index]) for index in range(3)
        )
        orientation_error = _orientation_error(
            tuple(float(value) for value in ee[3:7]),
            tuple(float(value) for value in target.orientation_xyzw),
        )
        error = q.new_tensor(
            tuple(value * self.config.position_gain for value in position_error)
            + tuple(value * self.config.orientation_gain for value in orientation_error)
        )
        identity = torch.eye(6, device=q.device, dtype=q.dtype)
        system = jacobian @ jacobian.transpose(-1, -2)
        system = system + (self.config.damping ** 2) * identity
        delta = jacobian.transpose(-1, -2) @ torch.linalg.solve(system, error)
        delta = delta.clamp(-self.config.max_joint_delta, self.config.max_joint_delta)

        positions = q.detach().clone()
        positions[:ARM_DOF_COUNT] += delta
        lower = q.new_tensor(self.config.lower_limits)
        upper = q.new_tensor(self.config.upper_limits)
        positions = torch.maximum(torch.minimum(positions, upper), lower)
        return JointPositionCommand(
            positions=positions,
            cartesian_error=error,
            joint_delta=delta,
        )
