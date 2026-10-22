"""P4-R5 staged composition of verified Transport and Lower policies."""

from dataclasses import dataclass
import math

from .auto_release_env import AutoReleaseEnvConfig
from .evidence import quaternion_distance
from .place_env import (
    advance_transport_success_event, transport_success_predicate,
)
from .place_completion import FullPlaceEnvironment


STAGED_ACTIVE_ACTION_MASK = (1, 1, 1, 0)


@dataclass(frozen=True)
class StagedPlaceEnvConfig(AutoReleaseEnvConfig):
    max_episode_seconds: float = 10.0

    def __post_init__(self):
        super().__post_init__()
        if self.max_episode_seconds != 10.0:
            raise ValueError("P4-R6 staged Full Place horizon is exactly 10 s")


class StagedFullPlaceEnvironment(FullPlaceEnvironment):
    """Phase-gated Cartesian control with environment-owned physical release."""

    active_action_mask = STAGED_ACTIVE_ACTION_MASK

    def __init__(self, config=None, **kwargs):
        super().__init__(config or StagedPlaceEnvConfig(), **kwargs)
        self.staged_phase = "UNINITIALIZED"
        self._transport_ready_steps = 0
        self._transport_target_reached = False
        self._transport_success = False
        self._switched_to_lower = False
        self._frozen_xy = None
        self._auto_release_streak = 0
        self._auto_release_ready = False
        self._release_condition = {}

    def _reset_once(self):
        observation = super().reset()
        self.staged_phase = "TRANSPORT"
        self._transport_ready_steps = 0
        self._transport_target_reached = False
        self._transport_success = False
        self._switched_to_lower = False
        self._frozen_xy = None
        self._auto_release_streak = 0
        self._auto_release_ready = False
        self._release_condition = {}
        return observation

    def reset(self):
        # GPU PhysX retains solver/contact history across tensor state resets.
        # A discarded pass brings the reused simulator through the same full
        # grasped+lifted reset history as a newly created evaluator before the
        # episode-visible reset.  This changes no policy target, gate, action,
        # workspace, checkpoint, or physical parameter.
        self._reset_once()
        return self._reset_once()

    def _apply_release_trigger(self, release, clearance):
        del release
        if self.staged_phase != "LOWER" or self._release_started:
            return False
        snapshot = self.scene.snapshot(); metrics = self._metrics(snapshot)
        cube = snapshot["cube_state"]
        self._release_condition = {
            "target_zone": self._xy_error(snapshot) <= self.config.target_xy_tolerance,
            "legal_release_height": clearance <= self.config.legal_release_clearance,
            "vertical_speed_small": abs(float(cube[9])) <= self.config.release_vertical_speed,
            "stable_grasp": bool(metrics["grasp_valid"]),
            "translation_stable": (
                metrics["translation_drift"] <= self.config.maximum_translation_drift),
            "rotation_stable": (
                metrics["rotation_drift"] <= self.config.maximum_rotation_drift
                and quaternion_distance(cube[3:7], (0., 0., 0., 1.)) <=
                self.config.maximum_tilt),
            "finite": self._finite(snapshot),
            "safe_force": max(metrics["forces"].values()) <= self.config.force_abort,
            "no_penetration": clearance >= -self.config.penetration_tolerance,
        }
        ready = all(self._release_condition.values())
        self._auto_release_streak = self._auto_release_streak + 1 if ready else 0
        self._release_trigger_streak = self._auto_release_streak
        if self._auto_release_streak < self.config.auto_release_ready_steps:
            return False
        self._auto_release_ready = True
        self._release_started = self._release_latched = True
        self._release_height = float(clearance)
        self.staged_phase = "AUTO_RELEASE"
        return True

    def _transport_condition(self, snapshot, info):
        metrics = self._metrics(snapshot)
        return transport_success_predicate(
            self.config, xy_error=self._xy_error(snapshot), cube_rise=metrics["rise"],
            grasp_valid=metrics["grasp_valid"], jaw_actual=metrics["jaw"],
            translation_drift=metrics["translation_drift"],
            rotation_drift=metrics["rotation_drift"], finite=self._finite(snapshot),
            maximum_force=max(metrics["forces"].values()),
            workspace_violation=info["workspace_violation"],
            joint_limit=info.get("joint_limit", False))

    def step(self, action):
        values = tuple(float(value) for value in action)
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            raise ValueError("ACTION_NON_FINITE_OR_DIMENSION")
        phase_before = self.staged_phase
        if phase_before == "TRANSPORT":
            # Match the original P4-1 runner exactly.  Its physical mask applies
            # only dx/dy, while its persistent Z target is reconstructed from
            # the lifted cube/reference grasp transform on every control step.
            transport_z = self._transport_start_cube[2] - self._reference_relative[2]
            self._target = (self._target[0], self._target[1], transport_z)
            selected = (values[0], values[1], 0.0, 0.0)
        elif phase_before == "LOWER": selected = (0.0, 0.0, values[2], 0.0)
        else: selected = (0.0, 0.0, 0.0, 0.0)
        observation, reward, done, info = super().step(selected)
        if phase_before == "TRANSPORT":
            # P4-1 stores the complete raw 4D actor output in previous_action,
            # even though dz/grip are masked at the physics boundary.
            raw_action = tuple(max(-1.0, min(1.0, value)) for value in values)
            self._place_previous_action = raw_action
            observation = dict(observation)
            observation["previous_action"] = raw_action
        snapshot = self.scene.snapshot()
        transport_condition = self._transport_condition(snapshot, info)
        transport_success = False
        if phase_before == "TRANSPORT" and not done:
            self._transport_target_reached |= bool(
                transport_condition["xy_within_15mm"])
            self._transport_ready_steps, transport_success = (
                advance_transport_success_event(
                    self._transport_ready_steps, transport_condition,
                    self.hold_steps_required))
            if transport_success:
                self._transport_success = True
                self._switched_to_lower = True
                self._frozen_xy = tuple(self._target[:2])
                self._target = (self._frozen_xy[0], self._frozen_xy[1], self._target[2])
                self.staged_phase = "LOWER"
        elif self.staged_phase in ("LOWER", "AUTO_RELEASE") and self._frozen_xy is not None:
            self._target = (self._frozen_xy[0], self._frozen_xy[1], self._target[2])
        if self._detached: self.staged_phase = "SETTLE"
        if info["success"]: self.staged_phase = "SUCCESS"
        info.update({
            "staged_phase": self.staged_phase,
            "transport_target_reached": self._transport_target_reached,
            "transport_ready_steps": self._transport_ready_steps,
            "transport_ready_duration": self._transport_ready_steps * self.control_dt,
            "transport_condition": transport_condition,
            "transport_success_event": transport_success,
            "transport_success": self._transport_success,
            "policy_switched_to_lower": self._switched_to_lower,
            "frozen_xy_target": self._frozen_xy,
            "auto_release_ready": self._auto_release_ready,
            "auto_release_ready_streak": self._auto_release_streak,
            "auto_release_condition": dict(self._release_condition),
            "active_action_mask": STAGED_ACTIVE_ACTION_MASK,
        })
        return observation, reward, done, info
