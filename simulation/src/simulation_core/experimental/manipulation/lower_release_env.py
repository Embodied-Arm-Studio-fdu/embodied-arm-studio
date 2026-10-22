"""P4-2 target-above reset and policy-controlled lower-and-release environment."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Tuple

from .evidence import quaternion_distance
from .lift_env import _norm
from .lower_release_schema import (
    LOWER_RELEASE_ACTIVE_ACTION_MASK, LowerReleaseRewardConfig,
    compute_lower_release_reward,
)
from .place_env import (
    PLACE_BOOTSTRAP_VERSION, PlaceEnvConfig, PlaceTransportEnvironment,
)
from .scene import ManipulationSceneConfig


LOWER_RELEASE_BOOTSTRAP_VERSION = (
    PLACE_BOOTSTRAP_VERSION + "+scripted-target-above-transport/v1"
)
LOWER_RELEASE_TARGET = (0.450, 0.080, 0.458)


@dataclass(frozen=True)
class LowerReleaseEnvConfig(PlaceEnvConfig):
    max_episode_seconds: float = 8.0
    dz_scale: float = 0.003
    legal_release_clearance: float = 0.012
    high_release_clearance: float = 0.018
    early_release_clearance: float = 0.030
    release_started_jaw: float = 0.020
    released_jaw: float = 0.032
    release_trigger_threshold: float = 0.0
    release_trigger_steps: int = 2
    table_contact_clearance: float = 0.004
    bottom_height_tolerance: float = 0.004
    penetration_tolerance: float = 0.002
    settled_linear_speed: float = 0.035
    settled_angular_speed: float = 0.60
    maximum_tilt: float = 0.25
    reset_target_xy_tolerance: float = 0.010
    reset_settle_steps: int = 10
    reward: LowerReleaseRewardConfig = field(default_factory=LowerReleaseRewardConfig)

    def __post_init__(self):
        super().__post_init__()
        if self.dz_scale != 0.003:
            raise ValueError("P4-2 keeps the existing 3 mm Cartesian Z scale")
        if not (self.legal_release_clearance < self.high_release_clearance
                < self.early_release_clearance):
            raise ValueError("release clearance thresholds must be ordered")
        if self.hold_seconds != 1.0:
            raise ValueError("P4-2 stable placement hold is exactly 1 second")
        if not -1.0 < self.release_trigger_threshold < 1.0:
            raise ValueError("release trigger threshold must be inside action bounds")
        if self.release_trigger_steps != 2:
            raise ValueError("release trigger requires exactly two control steps")


def lower_release_config_digest(config):
    payload = {"lower_release_env": asdict(config),
               "scene": asdict(ManipulationSceneConfig())}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


class LowerReleaseEnvironment(PlaceTransportEnvironment):
    """Place lower/release with a policy-triggered, physically executed jaw latch."""

    active_action_mask = LOWER_RELEASE_ACTIVE_ACTION_MASK

    def __init__(self, config=None, **kwargs):
        super().__init__(config or LowerReleaseEnvConfig(), **kwargs)
        self.target_center = LOWER_RELEASE_TARGET
        self._policy_xy = None
        self._start_clearance = None
        self._minimum_clearance = float("inf")
        self._maximum_open_fraction = 0.0
        self._release_started = False
        self._release_latched = False
        self._release_trigger_streak = 0
        self._actual_jaw_opened = False
        self._detached = False
        self._table_contact = False
        self._inside_target = False
        self._transport_reached = False
        self._descended_to_release_height = False
        self._settled_half = False
        self._settle_steps = 0
        self._maximum_settle_steps = 0
        self._release_height = None
        self._detach_height = None
        self._minimum_bottom_clearance = float("inf")

    def _bottom_clearance(self, snapshot):
        return (snapshot["cube_state"][2] - self.scene.config.cube_size / 2.0
                - self.scene.config.table_top)

    def _transport_above_target(self):
        start = self.scene.snapshot()
        cube, ee = start["cube_state"], start["ee_state"]
        delta = (self.target_center[0] - cube[0], self.target_center[1] - cube[1])
        target = (ee[0] + delta[0], ee[1] + delta[1], ee[2])
        stable_steps = 0
        for _ in range(180):
            snapshot, command = self._control(target, self.config.stable_grasp_jaw)
            metrics = self._metrics(snapshot, command.jaw_target)
            xy_error = self._xy_error(snapshot)
            valid = (
                xy_error <= self.config.reset_target_xy_tolerance
                and self.config.target_band[0] <= metrics["rise"] <= self.config.target_band[1]
                and metrics["grasp_valid"]
                and metrics["translation_drift"] <= self.config.maximum_translation_drift
                and metrics["rotation_drift"] <= self.config.maximum_rotation_drift
                and self._finite(snapshot)
                and max(metrics["forces"].values()) <= self.config.force_abort
                and not command.workspace_clamped and not command.joint_limit_clamped)
            stable_steps = stable_steps + 1 if valid else 0
            if not self._finite(snapshot):
                raise RuntimeError("RESET_TARGET_ABOVE_NON_FINITE")
            if command.workspace_clamped:
                raise RuntimeError("RESET_TARGET_ABOVE_WORKSPACE")
            if command.joint_limit_clamped:
                raise RuntimeError("RESET_TARGET_ABOVE_JOINT_LIMIT")
            if max(metrics["forces"].values()) > self.config.force_abort:
                raise RuntimeError("RESET_TARGET_ABOVE_UNSAFE_FORCE")
            if stable_steps >= self.config.reset_settle_steps:
                self._target = target
                return snapshot
        raise RuntimeError("RESET_TARGET_ABOVE_SETTLE_TIMEOUT")

    def reset(self):
        # The inherited reset constructs grasp + lift.  Its episode state is
        # discarded; target-above transport remains bootstrap and earns no reward.
        super().reset()
        snapshot = self._transport_above_target()
        metrics = self._metrics(snapshot)
        clearance = self._bottom_clearance(snapshot)
        xy_error = self._xy_error(snapshot)
        valid = (
            0.055 <= clearance <= 0.065
            and xy_error <= self.config.reset_target_xy_tolerance
            and metrics["grasp_valid"] and metrics["jaw"] >= 0.01
            and metrics["translation_drift"] <= self.config.maximum_translation_drift
            and metrics["rotation_drift"] <= self.config.maximum_rotation_drift
            and self._finite(snapshot)
            and max(metrics["forces"].values()) <= self.config.force_abort)
        if not valid:
            raise RuntimeError("TARGET_ABOVE_RESET_VALIDATION_FAILED")
        self._policy_xy = tuple(snapshot["ee_state"][:2])
        self._target = tuple(snapshot["ee_state"][:3])
        self._place_previous_action = (0.0,) * 4
        self._steps = self._settle_steps = self._maximum_settle_steps = 0
        self._start_clearance = clearance
        self._minimum_clearance = clearance
        self._minimum_bottom_clearance = clearance
        self._maximum_open_fraction = 0.0
        self._release_started = self._release_latched = False
        self._release_trigger_streak = 0
        self._actual_jaw_opened = False
        self._detached = self._table_contact = False
        self._inside_target = self._descended_to_release_height = False
        self._transport_reached = True
        self._settled_half = False
        self._release_height = self._detach_height = None
        self.phase = "LOWER_RELEASE_POLICY"
        self._bootstrap_last = dict(self._bootstrap_last)
        self._bootstrap_last.update({
            "lower_release_bootstrap_version": LOWER_RELEASE_BOOTSTRAP_VERSION,
            "target_above_valid": True,
            "target_above_xy_error": xy_error,
            "target_above_bottom_clearance": clearance,
        })
        return self._place_observation(snapshot)

    def _open_fraction(self, commanded_jaw):
        span = self.config.open_safe - self.config.stable_grasp_jaw
        return max(0.0, min(1.0,
            (float(commanded_jaw) - self.config.stable_grasp_jaw) / span))

    def _apply_release_trigger(self, release, clearance):
        """Advance the two-sample trigger and return whether it fired now."""
        if self._release_started:
            return False
        self._release_trigger_streak = (
            self._release_trigger_streak + 1
            if float(release) > self.config.release_trigger_threshold else 0)
        if self._release_trigger_streak < self.config.release_trigger_steps:
            return False
        self._release_started = True
        self._release_height = float(clearance)
        if clearance <= self.config.legal_release_clearance:
            self._release_latched = True
        return True

    def step(self, action):
        if self.phase != "LOWER_RELEASE_POLICY":
            raise RuntimeError("episode is not in LOWER_RELEASE_POLICY phase")
        values = tuple(float(value) for value in action)
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            raise ValueError("ACTION_NON_FINITE_OR_DIMENSION")
        values = tuple(max(-1.0, min(1.0, value)) for value in values)
        mask = self.active_action_mask
        dx = values[0] if mask[0] else 0.0
        dy = values[1] if mask[1] else 0.0
        dz = values[2] if mask[2] else 0.0
        release = values[3]

        # The policy owns WHEN to release.  Two consecutive trigger samples are
        # required.  Illegal-height triggers terminate without latching/opening.
        before = self.scene.snapshot()
        clearance_before = self._bottom_clearance(before)
        trigger_event = self._apply_release_trigger(release, clearance_before)

        requested = (self._target[0] + dx * self.config.xy_delta_scale,
                     self._target[1] + dy * self.config.xy_delta_scale,
                     self._target[2] + dz * self.config.dz_scale)
        desired_jaw = (self.config.open_safe if self._release_latched
                       else self.config.stable_grasp_jaw)
        try:
            snapshot, command = self._control(requested, desired_jaw)
        except Exception as exc:
            self.phase = "FAILURE"
            raise RuntimeError("CONTROLLER_FAILURE: %s" % exc) from exc
        self._target = self.config.workspace.clamp(requested)[0]
        applied = (dx, dy, dz, release)
        self._place_previous_action = applied
        self._steps += 1

        metrics = self._metrics(snapshot, command.jaw_target)
        cube = snapshot["cube_state"]
        clearance = self._bottom_clearance(snapshot)
        xy_error = self._xy_error(snapshot)
        linear_speed = _norm(cube[7:10])
        angular_speed = _norm(cube[10:13])
        tilt = quaternion_distance(cube[3:7], (0.0, 0.0, 0.0, 1.0))
        previous_minimum = self._minimum_clearance
        self._minimum_clearance = min(self._minimum_clearance, max(0.0, clearance))
        downward_progress = max(0.0, previous_minimum - self._minimum_clearance)
        self._minimum_bottom_clearance = min(self._minimum_bottom_clearance, clearance)
        self._descended_to_release_height |= clearance <= self.config.legal_release_clearance

        # Credit the one policy decision, not subsequent actuator motion.  Once
        # latched PPO has no jaw-aperture authority; physics and the rate limiter
        # alone determine actual opening and detachment.
        latch_event = self._release_latched and trigger_event
        legal_open_progress = 1.0 if latch_event else 0.0
        premature_open_progress = (1.0 if trigger_event and not self._release_latched
                                   else 0.0)

        no_blockage = metrics["blockage_gap"] <= .001
        self._actual_jaw_opened |= metrics["jaw"] >= self.config.released_jaw
        detached_now = (self._release_latched and self._actual_jaw_opened
                        and no_blockage)
        if detached_now and not self._detached:
            self._detached = True
            self._detach_height = clearance
        table_contact_now = clearance <= self.config.table_contact_clearance
        self._table_contact |= table_contact_now
        in_target = xy_error <= self.config.target_xy_tolerance
        self._transport_reached |= in_target
        self._inside_target |= self._table_contact and in_target
        settle_valid = (
            self._detached and table_contact_now and in_target
            and abs(clearance) <= self.config.bottom_height_tolerance
            and linear_speed <= self.config.settled_linear_speed
            and angular_speed <= self.config.settled_angular_speed
            and tilt <= self.config.maximum_tilt
            and metrics["jaw"] >= self.config.released_jaw
            and no_blockage and self._finite(snapshot))
        self._settle_steps = self._settle_steps + 1 if settle_valid else 0
        self._maximum_settle_steps = max(self._maximum_settle_steps, self._settle_steps)
        self._settled_half |= self._settle_steps * self.control_dt >= 0.5

        reason = ""
        if not self._finite(snapshot): reason = "NON_FINITE"
        elif command.workspace_clamped: reason = "WORKSPACE_VIOLATION"
        elif command.joint_limit_clamped: reason = "JOINT_LIMIT"
        elif max(metrics["forces"].values()) > self.config.force_abort: reason = "UNSAFE_FORCE"
        elif clearance < -self.config.penetration_tolerance: reason = "TABLE_PENETRATION"
        elif self._release_started and not self._release_latched \
                and self._release_height > self.config.early_release_clearance:
            reason = "EARLY_RELEASE"
        elif self._release_started and not self._release_latched:
            reason = "RELEASE_TOO_HIGH"
        elif self._table_contact and xy_error > self.config.target_xy_tolerance:
            reason = "DROP_OUTSIDE_TARGET"
        elif self._detached and tilt > 0.60: reason = "OBJECT_UNSTABLE"
        elif self._target[2] < self.config.workspace.lower[2] + .005: reason = "TARGET_OVERSHOOT"
        success = self._settle_steps >= self.hold_steps_required
        timeout = not success and not reason and self._steps >= self.max_steps
        if timeout: reason = "TIMEOUT"
        done = success or bool(reason)
        failure = bool(reason) and reason != "TIMEOUT"
        reward, terms = compute_lower_release_reward(
            config=self.config.reward, downward_progress=downward_progress,
            legal_open_progress=legal_open_progress,
            premature_open_progress=premature_open_progress,
            in_target_zone=in_target, table_contact=self._detached and table_contact_now,
            settle_valid=settle_valid, success_event=success,
            action=applied, terminal_failure=failure)
        if done:
            self.phase = "SUCCESS" if success else "FAILURE"
        info = {
            "success": success, "failure": failure, "timeout": timeout,
            "failure_reason": reason, "distance_to_goal": max(0.0, clearance),
            "initial_distance_to_goal": self._start_clearance,
            "action_norm": math.sqrt(dx * dx + dy * dy + dz * dz + release * release),
            "saturation_rate": float(max(abs(dx), abs(dy), abs(dz), abs(release)) >= .999),
            "object_displacement": self._start_clearance - max(0.0, clearance),
            "reward_terms": terms, "xy_error": xy_error,
            "descended_to_release_height": self._descended_to_release_height,
            "transport_reached": self._transport_reached,
            "legal_release_height": self._descended_to_release_height,
            "release_triggered": self._release_started,
            "release_started": self._release_started,
            "release_latched": self._release_latched,
            "actual_jaw_opened": self._actual_jaw_opened,
            "detached": self._detached,
            "table_contact": self._table_contact, "inside_target_zone": self._inside_target,
            "settled_0p5s": self._settled_half,
            "settled_1p0s": success,
            "settle_duration": self._maximum_settle_steps * self.control_dt,
            "release_height": self._release_height,
            "detach_height": self._detach_height,
            "bottom_clearance": clearance,
            "minimum_bottom_clearance": self._minimum_bottom_clearance,
            "cube_linear_speed": linear_speed, "cube_angular_speed": angular_speed,
            "cube_tilt": tilt, "jaw_actual": metrics["jaw"],
            "raw_policy_release_action": release,
            "raw_policy_grip_action": release,
            # Legacy diagnostic key retained as an observation of the fixed
            # actuator request; it is no longer derived from policy action.
            "absolute_grip_request": desired_jaw,
            "latched_jaw_request": desired_jaw,
            "rate_limited_jaw_command": command.jaw_target,
            "blockage_gap": metrics["blockage_gap"],
            "grasp_valid": metrics["grasp_valid"],
            "early_release": reason == "EARLY_RELEASE",
            "release_too_high": reason == "RELEASE_TOO_HIGH",
            "drop_outside_target": reason == "DROP_OUTSIDE_TARGET",
            "object_unstable": reason == "OBJECT_UNSTABLE",
            "workspace_violation": reason == "WORKSPACE_VIOLATION",
            "table_penetration": reason == "TABLE_PENETRATION",
            "target_overshoot": reason == "TARGET_OVERSHOOT",
            "joint_limit": reason == "JOINT_LIMIT",
            "unsafe": reason == "UNSAFE_FORCE", "non_finite": reason == "NON_FINITE",
            "active_action_mask": self.active_action_mask,
            "applied_action": applied, "bootstrap": dict(self._bootstrap_last),
        }
        return self._place_observation(snapshot), reward, done, info
