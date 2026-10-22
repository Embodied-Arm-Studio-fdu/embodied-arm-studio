"""Smooth, demo-local safe-range scan for the six revolute S1 arm joints."""

import math

from .base import S1DemoRuntime
from .config import SAFE_JOINT_POSES


ARM_JOINT_INDICES = (0, 1, 2, 3, 4, 5)
# Conservative teaching ranges, strictly inside the S1 URDF limits.  They are
# intentionally local to this demo and preserve the M3.5 home for other DOFs.
SAFE_SWEEP_RANGES = {
    0: (-0.75, 0.75),
    1: (0.15, 0.90),
    2: (0.12, 1.00),
    3: (-0.75, 0.45),
    4: (-0.60, 0.60),
    5: (-0.60, 0.60),
}
NOMINAL_SPEED_RAD_PER_SECOND = 1.5
MIN_SEGMENT_SECONDS = 0.5


def selected_joint_indices(joint_index):
    return ARM_JOINT_INDICES if joint_index == "all_arm" else (int(joint_index),)


def smoothstep(fraction):
    value = min(1.0, max(0.0, float(fraction)))
    return value * value * (3.0 - 2.0 * value)


def build_sweep_targets(joint_index):
    home = tuple(SAFE_JOINT_POSES["home"])
    targets = [("enter_home", None, home)]
    for index in selected_joint_indices(joint_index):
        lower, upper = SAFE_SWEEP_RANGES[index]
        low_target = list(home); low_target[index] = lower
        high_target = list(home); high_target[index] = upper
        targets.extend((("lower", index, tuple(low_target)),
                        ("hold_lower", index, tuple(low_target)),
                        ("upper", index, tuple(high_target)),
                        ("hold_upper", index, tuple(high_target)),
                        ("return_home", index, home)))
    return tuple(targets)


class JointSweepRuntime(S1DemoRuntime):
    def on_initialize(self):
        properties = self.robot.gym.get_actor_dof_properties(self.robot.env, self.robot.actor)
        self.physical_lower = tuple(float(value) for value in properties["lower"])
        self.physical_upper = tuple(float(value) for value in properties["upper"])
        for index, (lower, upper) in SAFE_SWEEP_RANGES.items():
            if not self.physical_lower[index] < lower < upper < self.physical_upper[index]:
                raise ValueError("joint %d safe range is not strictly inside physical limits" % index)
        self.indices = selected_joint_indices(self.config.joint_index)
        self.plan = build_sweep_targets(self.config.joint_index)
        self.segment_index = 0; self.segment_started_at = 0.0
        self.segment_start = tuple(float(value) for value in self.robot.read_state().dof_positions)
        self.sweep_complete = False; self.non_finite_count = 0; self.limit_clamp_count = 0
        self.max_tracking_error = 0.0
        self.actual_min = {index: float("inf") for index in self.indices}
        self.actual_max = {index: float("-inf") for index in self.indices}

    def _segment_duration(self, target):
        distance = max(abs(end-start) for start,end in zip(self.segment_start, target))
        speed = NOMINAL_SPEED_RAD_PER_SECOND * float(self.config.speed_scale)
        return max(MIN_SEGMENT_SECONDS, distance / speed)

    def command(self, control_time):
        state = self.robot.read_state()
        actual = tuple(float(value) for value in state.dof_positions)
        if not all(math.isfinite(value) for value in actual): self.non_finite_count += 1
        if self.segment_index < len(self.plan):
            phase, active_joint, endpoint = self.plan[self.segment_index]
            duration = self._segment_duration(endpoint)
            fraction = (control_time - self.segment_started_at) / duration
            alpha = smoothstep(fraction)
            target = tuple(start + (end-start)*alpha for start,end in zip(self.segment_start, endpoint))
            if fraction >= 1.0:
                target = endpoint
                self.segment_index += 1
                self.segment_start = endpoint
                self.segment_started_at = control_time
                if self.segment_index >= len(self.plan): self.sweep_complete = True
        else:
            phase, active_joint, alpha = "complete_home", None, 1.0
            target = tuple(SAFE_JOINT_POSES["home"])

        if active_joint is not None:
            self.actual_min[active_joint] = min(self.actual_min[active_joint], actual[active_joint])
            self.actual_max[active_joint] = max(self.actual_max[active_joint], actual[active_joint])

        for index, value in enumerate(target):
            if value < self.physical_lower[index] or value > self.physical_upper[index]:
                self.limit_clamp_count += 1
        self.robot.set_joint_position_targets(target)
        tracking = max(abs(current-commanded) for current,commanded in zip(actual,target))
        self.max_tracking_error = max(self.max_tracking_error, tracking)
        home = SAFE_JOINT_POSES["home"]
        home_error = max(abs(current-goal) for current,goal in zip(actual,home))
        per_joint = {
            str(index): {
                "actual_min": (self.actual_min[index] if math.isfinite(self.actual_min[index]) else None),
                "actual_max": (self.actual_max[index] if math.isfinite(self.actual_max[index]) else None),
                "target_min": SAFE_SWEEP_RANGES[index][0], "target_max": SAFE_SWEEP_RANGES[index][1],
            } for index in self.indices
        }
        return {
            "scanned_joint_indices": list(self.indices), "active_joint_index": active_joint,
            "sweep_phase": phase, "segment_index": self.segment_index,
            "segment_count": len(self.plan), "segment_fraction": alpha,
            "per_joint_results": per_joint, "max_tracking_error_rad": self.max_tracking_error,
            "final_home_error_rad": home_error, "sweep_completed": self.sweep_complete,
            "non_finite_count": self.non_finite_count, "limit_clamp_count": self.limit_clamp_count,
            "physical_lower_limits": list(self.physical_lower),
            "physical_upper_limits": list(self.physical_upper),
        }


def create_joint_sweep_runtime(config, task=None):
    return JointSweepRuntime(config, task)
