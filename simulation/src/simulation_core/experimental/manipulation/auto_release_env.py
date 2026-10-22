"""P4-R4 Full Place with a deterministic, safety-gated release primitive."""

from dataclasses import dataclass
import math

from .evidence import quaternion_distance
from .lift_env import _norm
from .lower_release_env import LowerReleaseEnvConfig
from .place_completion import FullPlaceEnvironment


AUTO_RELEASE_ACTIVE_ACTION_MASK = (1, 1, 1, 0)
AUTO_RELEASE_SEMANTICS = "deterministic_safe_auto_release"


@dataclass(frozen=True)
class AutoReleaseEnvConfig(LowerReleaseEnvConfig):
    auto_release_ready_steps: int = 3
    release_vertical_speed: float = 0.035

    def __post_init__(self):
        super().__post_init__()
        if self.auto_release_ready_steps != 3:
            raise ValueError("auto release requires exactly three ready control steps")
        if self.release_vertical_speed <= 0:
            raise ValueError("release vertical speed threshold must be positive")


class AutoReleaseFullPlaceEnvironment(FullPlaceEnvironment):
    """PPO transports/lowers; the environment releases only from a safe state."""

    active_action_mask = AUTO_RELEASE_ACTIVE_ACTION_MASK

    def __init__(self, config=None, **kwargs):
        super().__init__(config or AutoReleaseEnvConfig(), **kwargs)
        self._auto_release_streak = 0
        self._auto_release_ready = False
        self._last_workspace_violation = False
        self._release_condition = {}
        self._auto_minimum_xy_error = float("inf")

    def reset(self):
        observation = super().reset()
        self._auto_release_streak = 0
        self._auto_release_ready = False
        self._last_workspace_violation = False
        self._release_condition = {}
        self._auto_minimum_xy_error = self._xy_error(self.scene.snapshot())
        return observation

    def _release_ready_condition(self, snapshot, clearance):
        metrics = self._metrics(snapshot)
        cube = snapshot["cube_state"]
        xy_error = self._xy_error(snapshot)
        finite = self._finite(snapshot)
        maximum_force = max(metrics["forces"].values())
        condition = {
            "target_zone": xy_error <= self.config.target_xy_tolerance,
            "legal_release_height": clearance <= self.config.legal_release_clearance,
            "stable_grasp": bool(metrics["grasp_valid"]),
            "vertical_speed_small": abs(float(cube[9])) <= self.config.release_vertical_speed,
            "translation_stable": (
                metrics["translation_drift"] <= self.config.maximum_translation_drift),
            "rotation_stable": (
                metrics["rotation_drift"] <= self.config.maximum_rotation_drift
                and quaternion_distance(cube[3:7], (0., 0., 0., 1.)) <=
                self.config.maximum_tilt),
            "finite": finite,
            "safe_force": maximum_force <= self.config.force_abort,
            "no_penetration": clearance >= -self.config.penetration_tolerance,
            "workspace_valid": (
                not self._last_workspace_violation
                and self.config.workspace.contains(self._target)),
        }
        condition.update({
            "xy_error": xy_error, "bottom_clearance": clearance,
            "cube_vertical_speed": float(cube[9]),
            "cube_linear_speed": _norm(cube[7:10]),
            "cube_angular_speed": _norm(cube[10:13]),
            "maximum_force": maximum_force,
        })
        return condition

    def _apply_release_trigger(self, release, clearance):
        del release  # The fourth PPO action is deliberately ignored.
        if self._release_started:
            return False
        snapshot = self.scene.snapshot()
        self._release_condition = self._release_ready_condition(snapshot, clearance)
        required = (
            "target_zone", "legal_release_height", "stable_grasp",
            "vertical_speed_small", "translation_stable", "rotation_stable",
            "finite", "safe_force", "no_penetration", "workspace_valid",
        )
        ready = all(bool(self._release_condition[name]) for name in required)
        self._auto_release_streak = self._auto_release_streak + 1 if ready else 0
        self._release_trigger_streak = self._auto_release_streak
        if self._auto_release_streak < self.config.auto_release_ready_steps:
            return False
        self._auto_release_ready = True
        self._release_started = True
        self._release_latched = True
        self._release_height = float(clearance)
        return True

    def step(self, action):
        values = tuple(float(value) for value in action)
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            raise ValueError("ACTION_NON_FINITE_OR_DIMENSION")
        # Zeroing the ignored component also removes it from reward/action
        # diagnostics; StagePlaceTrainer excludes it from PPO likelihood.
        previous_best = self._auto_minimum_xy_error
        observation, reward, done, info = super().step(
            (values[0], values[1], values[2], 0.0))
        self._auto_minimum_xy_error = min(self._auto_minimum_xy_error, info["xy_error"])
        transport_progress = max(0.0, previous_best - self._auto_minimum_xy_error)
        transport_reward = 120.0 * transport_progress
        # Full Place must transport before descending.  Remove the inherited
        # descent credit while outside the target zone, but never script either
        # Cartesian action; PPO remains responsible for both decisions.
        illegal_descent_credit = (info["reward_terms"]["downward_progress"]
                                  if info["xy_error"] > self.config.target_xy_tolerance
                                  else 0.0)
        reward += transport_reward - illegal_descent_credit
        info["reward_terms"] = dict(info["reward_terms"])
        info["reward_terms"].update({
            "transport_progress": transport_reward,
            "outside_target_descent_credit_removed": -illegal_descent_credit,
        })
        self._last_workspace_violation = bool(info["workspace_violation"])
        info.update({
            "auto_release_ready": self._auto_release_ready,
            "auto_release_ready_streak": self._auto_release_streak,
            "auto_release_condition": dict(self._release_condition),
            "release_semantics": AUTO_RELEASE_SEMANTICS,
            "active_action_mask": AUTO_RELEASE_ACTIVE_ACTION_MASK,
            "applied_action": (info["applied_action"][0], info["applied_action"][1],
                               info["applied_action"][2], 0.0),
        })
        return observation, reward, done, info
