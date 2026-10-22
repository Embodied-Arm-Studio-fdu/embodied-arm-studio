"""P4-1 grasped-and-lifted reset plus horizontal transport environment."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Tuple, Optional, Dict, Any

from .evidence import quaternion_distance
from .lift_env import LIFT_BOOTSTRAP_VERSION, LiftEnvConfig, LiftPPOEnvironment, _clip, _norm, _sub
from .place_schema import PlaceRewardConfig, compute_place_reward
from .scene import ManipulationSceneConfig
from .target_compositor import Workspace3D


PLACE_BOOTSTRAP_VERSION = LIFT_BOOTSTRAP_VERSION + "+scripted-lift-60mm/v1"
PLACE_TARGET_SELECTION_VERSION = "collision-free-candidate-probe/v1"


def transport_success_predicate(config, *, xy_error, cube_rise, grasp_valid,
                                jaw_actual, translation_drift, rotation_drift,
                                finite, maximum_force, workspace_violation,
                                joint_limit=False):
    """Return the canonical per-step P4-1 Transport success predicate."""
    return {
        "xy_within_15mm": float(xy_error) <= config.target_xy_tolerance,
        "valid_lift_height": (
            config.transport_height_band[0] <= float(cube_rise)
            <= config.transport_height_band[1]),
        "stable_grasp": bool(grasp_valid),
        "jaw_stable": float(jaw_actual) >= .01,
        "translation_stable": (
            float(translation_drift) <= config.maximum_translation_drift),
        "rotation_stable": (
            float(rotation_drift) <= config.maximum_rotation_drift),
        "finite": bool(finite),
        "safe_force": float(maximum_force) <= config.force_abort,
        "workspace_valid": not bool(workspace_violation),
        "joint_limit_valid": not bool(joint_limit),
    }


def advance_transport_success_event(hold_steps, condition, required_steps):
    """Advance the canonical consecutive-hold counter and emit its event."""
    next_steps = int(hold_steps) + 1 if all(condition.values()) else 0
    return next_steps, next_steps >= int(required_steps)


@dataclass(frozen=True)
class PlaceEnvConfig(LiftEnvConfig):
    max_episode_seconds: float = 8.0
    hold_seconds: float = 1.0
    xy_delta_scale: float = 0.0045
    transport_distance: float = 0.080
    target_xy_tolerance: float = 0.015
    transport_height_band: Tuple[float, float] = (0.050, 0.070)
    lifted_settle_steps: int = 10
    target_selection_version: str = PLACE_TARGET_SELECTION_VERSION
    reward: PlaceRewardConfig = field(default_factory=PlaceRewardConfig)
    reward_extension: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.reward_extension is not None:
            from ...reward_extension import validate_extension
            validate_extension("s1_place", self.reward_extension)
        if not 0.060 <= self.transport_distance <= 0.100:
            raise ValueError("P4-1 transport distance must be 60-100 mm")
        if not 0.004 <= self.xy_delta_scale <= 0.005:
            raise ValueError("P4-1 XY delta scale must be 4-5 mm")
        if self.target_xy_tolerance != 0.015:
            raise ValueError("P4-1 target tolerance is locked to 15 mm")


def place_config_from_snapshot(snapshot):
    values = dict(snapshot or {})
    if isinstance(values.get("reward"), dict):
        values["reward"] = PlaceRewardConfig(**values["reward"])
    if isinstance(values.get("workspace"), dict):
        values["workspace"] = Workspace3D(**values["workspace"])
    for name in ("target_band", "transport_height_band", "fixed_orientation_xyzw", "grasp_frame_object_offset"):
        if name in values:
            values[name] = tuple(values[name])
    return PlaceEnvConfig(**values)


def place_config_digest(config):
    environment_config = asdict(config)
    if environment_config.get("reward_extension") is None:
        environment_config.pop("reward_extension", None)  # Preserve legacy checkpoint digests.
    payload = {"place_env": environment_config, "scene": asdict(ManipulationSceneConfig())}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


class PlaceTransportEnvironment(LiftPPOEnvironment):
    """Four-dimensional Place contract with P4-1 mask [1,1,0,0]."""

    def __init__(self, config=None, **kwargs):
        super().__init__(config or PlaceEnvConfig(), **kwargs)
        from ...reward_extension import validate_extension
        self._reward_expression = validate_extension("s1_place", self.config.reward_extension) if self.config.reward_extension else None
        self.target_center = None
        self.target_selection = None
        self._transport_start_cube = None
        self._transport_height = None
        self._place_previous_action = (0.0,) * 4
        self._minimum_xy_error = float("inf")
        self._target_reached = False
        self._reached_50mm_error = False
        self._reached_30mm_error = False
        self._hold_half = False
        self._hold_steps = 0
        self._maximum_hold_steps = 0
        self._rise_min = float("inf")
        self._rise_max = float("-inf")
        self._transport_time = None

    def _select_target(self, lifted_snapshot):
        cube = lifted_snapshot["cube_state"]
        relative = _sub(cube[:3], lifted_snapshot["ee_state"][:3])
        base_xy = self.scene.config.cube_xy
        directions = ((0.0, 1.0), (0.0, -1.0), (-1.0, 0.0), (1.0, 0.0))
        candidates = []
        table = self.scene.config
        half_x = table.table_size[0] / 2.0 - table.cube_size
        half_y = table.table_size[1] / 2.0 - table.cube_size
        for index, direction in enumerate(directions):
            target = (base_xy[0] + direction[0] * self.config.transport_distance,
                      base_xy[1] + direction[1] * self.config.transport_distance,
                      self.scene.config.cube_start[2] + self.config.target_lift)
            implied_ee = tuple(target[i] - relative[i] for i in range(3))
            table_safe = (abs(target[0] - table.table_position[0]) <= half_x
                          and abs(target[1] - table.table_position[1]) <= half_y)
            workspace_safe = self.config.workspace.contains(implied_ee, tolerance=-.005)
            base_clearance = math.hypot(target[0] - table.robot_position[0],
                                        target[1] - table.robot_position[1])
            base_safe = base_clearance >= .25
            candidates.append({"index": index, "direction": list(direction),
                               "target_center": list(target), "implied_ee": list(implied_ee),
                               "table_safe": table_safe, "workspace_safe": workspace_safe,
                               "base_clearance": base_clearance, "base_safe": base_safe,
                               "geometry_passed": table_safe and workspace_safe and base_safe})
        valid = [row for row in candidates if row["geometry_passed"]]
        if not valid:
            raise RuntimeError("PLACE_TARGET_NO_GEOMETRIC_CANDIDATE")
        # Prefer large robot-base clearance, then the declared deterministic
        # order. This derives a fixed target from the admitted lifted state.
        selected = dict(max(valid, key=lambda row: (row["base_clearance"], -row["index"])))
        selected["selection_reason"] = (
            "maximum base clearance among table/workspace-safe 80mm candidates")
        selected["all_candidates"] = [dict(row) for row in candidates]
        self.target_center = tuple(selected["target_center"])
        self.target_selection = selected

    def _scripted_lift(self):
        self.phase = "GRASPED_LIFTED_RESET"
        lift_target = (self._target[0], self._target[1],
                       self._target[2] + self.config.target_lift)
        stable_steps = 0; last = None
        for _ in range(180):
            snapshot, command = self._control(lift_target, self.config.stable_grasp_jaw)
            last = snapshot; metrics = self._metrics(snapshot, command.jaw_target)
            rise = metrics["rise"]
            stable = (self.config.target_band[0] <= rise <= self.config.target_band[1]
                      and metrics["grasp_valid"]
                      and metrics["translation_drift"] <= self.config.maximum_translation_drift
                      and metrics["rotation_drift"] <= self.config.maximum_rotation_drift)
            stable_steps = stable_steps + 1 if stable else 0
            if not self._finite(snapshot): raise RuntimeError("RESET_LIFT_NON_FINITE")
            if command.workspace_clamped: raise RuntimeError("RESET_LIFT_WORKSPACE")
            if command.joint_limit_clamped: raise RuntimeError("RESET_LIFT_JOINT_LIMIT")
            if max(metrics["forces"].values()) > self.config.force_abort:
                raise RuntimeError("RESET_LIFT_UNSAFE_FORCE")
            if metrics["rise"] < -.005: raise RuntimeError("RESET_LIFT_DROP")
            if stable_steps >= self.config.lifted_settle_steps:
                self._target = lift_target
                return snapshot
        final_metrics = self._metrics(last) if last is not None else {}
        raise RuntimeError("RESET_LIFT_SETTLE_TIMEOUT rise=%r drift=%r rotation=%r "
                           "grasp=%r vz=%r angular=%r" % (
            last["cube_state"][2] - self._initial_cube[2] if last else None,
            final_metrics.get("translation_drift"), final_metrics.get("rotation_drift"),
            final_metrics.get("grasp_valid"), last["cube_state"][9] if last else None,
            _norm(last["cube_state"][10:13]) if last else None))

    def reset(self):
        last_error = None
        for attempt in range(1, self.config.bootstrap_attempts + 1):
            try:
                super().reset()
                snapshot = self._scripted_lift()
                break
            except RuntimeError as exc:
                last_error = str(exc)
        else:
            raise RuntimeError("GRASPED_LIFTED_RESET_EXHAUSTED: %s" % last_error)
        if self.target_center is None:
            self._select_target(snapshot)
        metrics = self._metrics(snapshot)
        if not (self.config.target_band[0] <= metrics["rise"] <= self.config.target_band[1]
                and metrics["grasp_valid"] and metrics["jaw"] >= .01
                and metrics["translation_drift"] <= .012
                and metrics["rotation_drift"] <= .25 and self._finite(snapshot)
                and max(metrics["forces"].values()) <= self.config.force_abort):
            raise RuntimeError("GRASPED_LIFTED_RESET_VALIDATION_FAILED")
        self._transport_start_cube = tuple(snapshot["cube_state"][:3])
        self._transport_height = metrics["rise"]
        ee = snapshot["ee_state"]
        self._target = tuple(ee[:3])
        self._place_previous_action = (0.0,) * 4
        self._steps = 0; self._hold_steps = 0; self._maximum_hold_steps = 0
        self._minimum_xy_error = self._xy_error(snapshot)
        self._target_reached = self._hold_half = False
        self._reached_50mm_error = self._minimum_xy_error < .050
        self._reached_30mm_error = self._minimum_xy_error < .030
        self._rise_min = self._rise_max = metrics["rise"]
        self._transport_time = None; self._invalid_grasp_steps = 0
        self.phase = "PLACE_POLICY"
        self._bootstrap_last = dict(self._bootstrap_last)
        self._bootstrap_last.update({"place_reset_attempts": attempt,
                                     "scripted_lift_rise": metrics["rise"],
                                     "place_bootstrap_version": PLACE_BOOTSTRAP_VERSION})
        return self._place_observation(snapshot)

    def _xy_error(self, snapshot):
        cube = snapshot["cube_state"]
        return math.hypot(cube[0] - self.target_center[0], cube[1] - self.target_center[1])

    def _place_observation(self, snapshot):
        cube, ee = snapshot["cube_state"], snapshot["ee_state"]
        metrics = self._metrics(snapshot)
        workspace = self.config.workspace
        nominal = self.scene.config.cube_start
        def workspace_norm(values):
            return tuple(_clip((values[i] - (workspace.lower[i] + workspace.upper[i]) / 2.0) /
                               ((workspace.upper[i] - workspace.lower[i]) / 2.0)) for i in range(3))
        return {
            "ee_position": workspace_norm(ee[:3]),
            "ee_velocity": tuple(_clip(v) for v in ee[7:10]),
            "cube_position": tuple(_clip((cube[i] - nominal[i]) / .15) for i in range(3)),
            "cube_linear_velocity": tuple(_clip(v) for v in cube[7:10]),
            "cube_angular_velocity": tuple(_clip(v / 5.0) for v in cube[10:13]),
            "target_position": tuple(_clip((self.target_center[i] - nominal[i]) / .15)
                                     for i in range(3)),
            "cube_target_relative": tuple(_clip((cube[i] - self.target_center[i]) / .15)
                                          for i in range(3)),
            "ee_cube_relative": tuple(_clip(v / .15) for v in _sub(cube[:3], ee[:3])),
            "relative_rotation_error": (_clip(metrics["rotation_drift"] / .25),),
            "jaw_actual": (_clip(metrics["jaw"] / .05, 0.0, 1.5),),
            "blockage_gap": (_clip(metrics["blockage_gap"] / self.config.minimum_blockage),),
            "grasp_stability": (1.0 if metrics["grasp_valid"] else 0.0,),
            "previous_action": self._place_previous_action,
        }

    def step(self, action):
        if self.phase != "PLACE_POLICY":
            raise RuntimeError("episode is not in PLACE_POLICY phase")
        values = tuple(float(v) for v in action)
        if len(values) != 4 or not all(math.isfinite(v) for v in values):
            raise ValueError("ACTION_NON_FINITE_OR_DIMENSION")
        values = tuple(max(-1.0, min(1.0, v)) for v in values)
        requested = (self._target[0] + values[0] * self.config.xy_delta_scale,
                     self._target[1] + values[1] * self.config.xy_delta_scale,
                     self._transport_start_cube[2] - self._reference_relative[2])
        try:
            snapshot, command = self._control(requested, self.config.stable_grasp_jaw)
        except Exception as exc:
            self.phase = "FAILURE"
            raise RuntimeError("CONTROLLER_FAILURE: %s" % exc) from exc
        self._target = self.config.workspace.clamp(requested)[0]
        self._place_previous_action = values; self._steps += 1
        metrics = self._metrics(snapshot, command.jaw_target)
        rise = metrics["rise"]; xy_error = self._xy_error(snapshot)
        previous_best = self._minimum_xy_error
        self._minimum_xy_error = min(self._minimum_xy_error, xy_error)
        progress = max(0.0, previous_best - self._minimum_xy_error)
        transport_condition = transport_success_predicate(
            self.config, xy_error=xy_error, cube_rise=rise,
            grasp_valid=metrics["grasp_valid"], jaw_actual=metrics["jaw"],
            translation_drift=metrics["translation_drift"],
            rotation_drift=metrics["rotation_drift"],
            finite=self._finite(snapshot),
            maximum_force=max(metrics["forces"].values()),
            workspace_violation=command.workspace_clamped,
            joint_limit=command.joint_limit_clamped)
        in_zone = transport_condition["xy_within_15mm"]
        self._hold_steps, success = advance_transport_success_event(
            self._hold_steps, transport_condition, self.hold_steps_required)
        self._maximum_hold_steps = max(self._maximum_hold_steps, self._hold_steps)
        self._target_reached |= in_zone
        self._reached_50mm_error |= xy_error < .050
        self._reached_30mm_error |= xy_error < .030
        self._hold_half |= self._hold_steps * self.control_dt >= .5
        self._rise_min = min(self._rise_min, rise); self._rise_max = max(self._rise_max, rise)
        if in_zone and self._transport_time is None:
            self._transport_time = self._steps * self.control_dt
        self._invalid_grasp_steps = 0 if metrics["grasp_valid"] else self._invalid_grasp_steps + 1
        clearance = snapshot["cube_state"][2] - self.scene.config.cube_size / 2.0 - self.scene.config.table_top
        reason = ""
        if not self._finite(snapshot): reason = "NON_FINITE"
        elif command.workspace_clamped: reason = "WORKSPACE_VIOLATION"
        elif command.joint_limit_clamped: reason = "JOINT_LIMIT"
        elif max(metrics["forces"].values()) > self.config.force_abort: reason = "UNSAFE_FORCE"
        elif clearance < .015: reason = "TABLE_COLLISION" if rise >= .020 else "DROP"
        elif rise < .030: reason = "DROP"
        elif self._invalid_grasp_steps >= 3 or metrics["translation_drift"] > .024 \
                or metrics["rotation_drift"] > .50: reason = "GRASP_LOST"
        elif self._target_reached and xy_error > .030: reason = "TARGET_OVERSHOOT"
        timeout = not success and not reason and self._steps >= self.max_steps
        if timeout: reason = "TIMEOUT"
        done = success or bool(reason); failure = bool(reason) and reason != "TIMEOUT"
        reward, terms = compute_place_reward(
            config=self.config.reward, new_best_progress=progress,
            in_target_zone=in_zone, height_error=rise - self._transport_height,
            translation_drift=metrics["translation_drift"],
            angular_speed=_norm(snapshot["cube_state"][10:13]), action=values,
            terminal_failure=failure)
        if self._reward_expression is not None:
            # Place's existing reward has no separate success/joint bonus. Keep
            # both platform-owned terms at their baseline zero values.
            reward = self._reward_expression.evaluate({**terms, "success_bonus": 0.0, "joint_limit_penalty": 0.0})
        if done: self.phase = "SUCCESS" if success else "FAILURE"
        info = {"success": success, "failure": failure, "timeout": timeout,
                "failure_reason": reason, "distance_to_goal": xy_error,
                "initial_distance_to_goal": self.config.transport_distance,
                "action_norm": math.hypot(values[0], values[1]),
                "saturation_rate": float(max(abs(values[0]), abs(values[1])) >= .999),
                "object_displacement": self.config.transport_distance - xy_error,
                "reward_terms": terms, "xy_error": xy_error,
                "minimum_xy_error": self._minimum_xy_error,
                "target_reached": self._target_reached,
                "hold_duration": self._maximum_hold_steps * self.control_dt,
                "hold_half_second": self._hold_half,
                "reached_50mm_error": self._reached_50mm_error,
                "reached_30mm_error": self._reached_30mm_error,
                "transport_time": self._transport_time,
                "cube_rise": rise, "cube_height_variation": self._rise_max - self._rise_min,
                "relative_translation_drift": metrics["translation_drift"],
                "relative_rotation_drift": metrics["rotation_drift"],
                "drop": reason == "DROP", "grasp_lost": reason == "GRASP_LOST",
                "workspace_violation": reason == "WORKSPACE_VIOLATION",
                "table_collision": reason == "TABLE_COLLISION",
                "target_overshoot": reason == "TARGET_OVERSHOOT",
                "unsafe": reason == "UNSAFE_FORCE", "non_finite": reason == "NON_FINITE",
                "transport_success_event": success,
                "transport_success_condition": transport_condition,
                "active_action_mask": (1, 1, 0, 0),
                "applied_action": (values[0], values[1], 0.0, 0.0),
                "bootstrap": dict(self._bootstrap_last)}
        return self._place_observation(snapshot), reward, done, info

    def scripted_path_probe(self):
        snapshot_obs = self.reset()
        del snapshot_obs
        start = self.scene.snapshot(); cube = start["cube_state"]
        delta = (self.target_center[0] - cube[0], self.target_center[1] - cube[1])
        target = (start["ee_state"][0] + delta[0], start["ee_state"][1] + delta[1],
                  self._target[2])
        max_force = 0.0; max_drift = 0.0; max_rotation = 0.0; stable_steps = 0
        rows = 0
        for _ in range(180):
            snapshot, command = self._control(target, self.config.stable_grasp_jaw); rows += 1
            metrics = self._metrics(snapshot, command.jaw_target)
            max_force = max(max_force, max(metrics["forces"].values()))
            max_drift = max(max_drift, metrics["translation_drift"])
            max_rotation = max(max_rotation, metrics["rotation_drift"])
            rise = metrics["rise"]; error = self._xy_error(snapshot)
            valid = (error <= .015 and self.config.transport_height_band[0] <= rise <=
                     self.config.transport_height_band[1] and metrics["grasp_valid"]
                     and metrics["translation_drift"] <= .012
                     and metrics["rotation_drift"] <= .25)
            stable_steps = stable_steps + 1 if valid else 0
            if (not self._finite(snapshot) or command.workspace_clamped
                    or command.joint_limit_clamped or max_force > self.config.force_abort):
                break
            if stable_steps >= self.hold_steps_required:
                return {"passed": True, "target_selection": self.target_selection,
                        "target_center": list(self.target_center), "steps": rows,
                        "final_xy_error": error, "cube_rise": rise,
                        "max_force": max_force, "max_translation_drift": max_drift,
                        "max_rotation_drift": max_rotation,
                        "table_clearance": (snapshot["cube_state"][2]
                                            - self.scene.config.cube_size / 2
                                            - self.scene.config.table_top),
                        "stable_hold_seconds": stable_steps * self.control_dt}
        return {"passed": False, "target_selection": self.target_selection,
                "target_center": list(self.target_center), "steps": rows,
                "max_force": max_force, "max_translation_drift": max_drift,
                "max_rotation_drift": max_rotation}
