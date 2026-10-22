"""Experimental stable-grasp-reset S1 Lift environment."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Callable, Optional, Tuple

from .evidence import quaternion_distance
from .lift_schema import LiftRewardConfig, compute_lift_reward
from .scene import ManipulationScene, ManipulationSceneConfig
from .target_compositor import ExperimentalTargetCompositor, Workspace3D


LIFT_BOOTSTRAP_VERSION = "p1-p2-deterministic-stable-grasp/v1"


def _norm(values):
    return math.sqrt(sum(float(v) ** 2 for v in values))


def _sub(left, right):
    return tuple(float(a) - float(b) for a, b in zip(left, right))


def _clip(value, low=-5.0, high=5.0):
    return max(low, min(high, float(value)))


@dataclass(frozen=True)
class LiftEnvConfig:
    open_safe: float = 0.05
    stable_grasp_jaw: float = 0.012
    jaw_rate_limit: float = 0.0015
    force_abort: float = 150.0
    physics_steps_per_action: int = 4
    max_episode_seconds: float = 6.0
    dz_scale: float = 0.003
    target_lift: float = 0.060
    target_band: Tuple[float, float] = (0.055, 0.065)
    overshoot_lift: float = 0.075
    hold_seconds: float = 1.0
    maximum_translation_drift: float = 0.012
    maximum_rotation_drift: float = 0.25
    minimum_blockage: float = 0.002
    contact_threshold: float = 0.2
    bootstrap_attempts: int = 5
    bootstrap_stable_steps: int = 10
    bootstrap_cube_displacement: float = 0.008
    workspace: Workspace3D = field(default_factory=Workspace3D)
    fixed_orientation_xyzw: Tuple[float, float, float, float] = (
        0.6040455102920532, -0.7851132154464722,
        0.12926433980464935, -0.044911183416843414,
    )
    grasp_frame_object_offset: Tuple[float, float, float] = (-0.008, 0.009, 0.012)
    pregrasp_clearance: float = 0.080
    reward: LiftRewardConfig = field(default_factory=LiftRewardConfig)

    def __post_init__(self):
        if not 0.01 <= self.stable_grasp_jaw < self.open_safe <= 0.05:
            raise ValueError("invalid stable grasp jaw command")
        if self.physics_steps_per_action != 4 or self.dz_scale <= 0:
            raise ValueError("Lift requires four physics steps and positive dz scale")
        if self.target_band != (0.055, 0.065) or self.target_lift != 0.060:
            raise ValueError("P3-1 target and band are locked")
        if self.hold_seconds not in (1.0, 5.0):
            raise ValueError("Lift supports only the P3-1 1 s or P3-2 5 s hold")
        minimum_horizon = 6.0 if self.hold_seconds == 1.0 else 9.0
        if self.max_episode_seconds < minimum_horizon:
            raise ValueError("episode horizon is too short for the configured hold")


def lift_config_digest(config: LiftEnvConfig) -> str:
    payload = {"lift_env": asdict(config), "scene": asdict(ManipulationSceneConfig())}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


class LiftPPOEnvironment:
    """Reset through deterministic grasp, then expose only a one-dimensional lift action."""

    def __init__(self, config: LiftEnvConfig = None, *, with_camera=False,
                 frame_callback: Optional[Callable] = None):
        self.config = config or LiftEnvConfig()
        self.scene = ManipulationScene(ManipulationSceneConfig(
            graphics_device_id=0 if with_camera else -1, with_camera=with_camera,
            initial_open=self.config.open_safe,
            fixed_orientation_xyzw=self.config.fixed_orientation_xyzw,
        ))
        self.compositor = ExperimentalTargetCompositor(
            workspace=self.config.workspace, jaw_rate_limit=self.config.jaw_rate_limit)
        self.frame_callback = frame_callback
        self.phase = "UNINITIALIZED"
        self._initialized = False
        self._closed = False
        self._previous_action = (0.0,)
        self._target = None
        self._initial_cube = None
        self._reference_relative = None
        self._reference_rotation = None
        self._steps = 0
        self._hold_steps = 0
        self._max_rise = 0.0
        self._reached_20 = False
        self._reached_40 = False
        self._target_reached = False
        self._hold_half = False
        self._hold_one = False
        self._hold_three = False
        self._hold_five = False
        self._maximum_hold_steps = 0
        self._invalid_grasp_steps = 0
        self._bootstrap_attempts_total = 0
        self._bootstrap_last = {}

    @property
    def control_dt(self):
        return self.scene.robot.config.physics_dt * self.config.physics_steps_per_action

    @property
    def hold_steps_required(self):
        return int(math.ceil(self.config.hold_seconds / self.control_dt))

    @property
    def max_steps(self):
        return int(math.ceil(self.config.max_episode_seconds / self.control_dt))

    def initialize(self):
        if not self._initialized:
            self.scene.initialize(); self._initialized = True
        return self

    def _capture(self):
        if self.frame_callback is not None:
            self.frame_callback(self.scene.capture())

    def _forces(self, snapshot):
        return {name: _norm(body["force"]) for name, body in snapshot["bodies"].items()}

    def _finite(self, snapshot):
        values = snapshot["q"] + snapshot["dq"] + snapshot["ee_state"] + snapshot["cube_state"]
        return all(math.isfinite(float(v)) for v in values)

    def _control(self, xyz, jaw):
        state = self.scene.robot.read_state()
        command = self.compositor.compose(
            xyz, self.config.fixed_orientation_xyzw, jaw, state)
        self.scene.robot.set_joint_position_targets(command.positions)
        self.scene.robot.step(self.config.physics_steps_per_action)
        self._capture()
        return self.scene.snapshot(), command

    def _move_until(self, xyz, jaw, limit, predicate):
        rows = []
        for _ in range(limit):
            snapshot, command = self._control(xyz, jaw); rows.append(snapshot)
            if not self._finite(snapshot):
                raise RuntimeError("BOOTSTRAP_NON_FINITE")
            if command.workspace_clamped:
                raise RuntimeError("BOOTSTRAP_WORKSPACE_VIOLATION")
            if max(self._forces(snapshot).values()) > self.config.force_abort:
                raise RuntimeError("BOOTSTRAP_UNSAFE_FORCE")
            if predicate(snapshot, command, rows):
                return snapshot, command, rows
        raise RuntimeError("BOOTSTRAP_CONTROLLER_TIMEOUT")

    def _stable_grasp_flags(self, snapshot, command, rows, reset_cube):
        forces = self._forces(snapshot)
        jaw = 0.5 * (snapshot["q"][6] + snapshot["q"][7])
        blockage = jaw - command.jaw_target
        stable_rows = rows[-self.config.bootstrap_stable_steps:]
        jaw_stable = len(stable_rows) == self.config.bootstrap_stable_steps and all(
            max(row["q"][index] for row in stable_rows)
            - min(row["q"][index] for row in stable_rows) < 0.0003 for index in (6, 7))
        cube = snapshot["cube_state"]
        cube_displacement = _norm(_sub(cube[:3], reset_cube))
        return {
            "finite": self._finite(snapshot),
            "jaw_safe": jaw >= 0.01,
            "blockage": blockage >= self.config.minimum_blockage,
            "left_contact": forces.get("8_Link", 0.0) >= self.config.contact_threshold,
            "right_contact": forces.get("7_Link", 0.0) >= self.config.contact_threshold,
            "cube_contact": forces.get("cube", 0.0) >= self.config.contact_threshold,
            "jaw_stable": jaw_stable,
            "cube_not_abnormally_displaced": cube_displacement <= self.config.bootstrap_cube_displacement,
            "force_safe": max(forces.values()) <= self.config.force_abort,
        }, jaw, blockage, forces, cube_displacement

    def _bootstrap_once(self):
        self.phase = "GRASP_BOOTSTRAP"
        self.scene.reset(self.scene.config.cube_xy)
        initial = self.scene.snapshot(); reset_cube = tuple(initial["cube_state"][:3])
        self.compositor.reset(initial["q"])
        grasp = tuple(reset_cube[i] - self.config.grasp_frame_object_offset[i] for i in range(3))
        pre = (grasp[0], grasp[1], grasp[2] + self.config.pregrasp_clearance)
        current = tuple(initial["ee_state"][:3])
        self._move_until(current, self.config.open_safe, 45,
                         lambda s, c, r: max(abs(v) for v in s["dq"]) < 0.05
                         and _norm(s["cube_state"][7:13]) < 0.05)
        clearance = (current[0], current[1], pre[2])
        self._move_until(clearance, self.config.open_safe, 180,
                         lambda s, c, r: _norm(_sub(s["ee_state"][:3], clearance)) <= 0.006)
        self._move_until(pre, self.config.open_safe, 240,
                         lambda s, c, r: _norm(_sub(s["ee_state"][:3], pre)) <= 0.006)
        self._move_until(grasp, self.config.open_safe, 180,
                         lambda s, c, r: _norm(_sub(s["ee_state"][:3], grasp)) <= 0.006)

        def stable(snapshot, command, rows):
            flags, _, _, _, _ = self._stable_grasp_flags(
                snapshot, command, rows, reset_cube)
            return abs(command.jaw_target - self.config.stable_grasp_jaw) < 1e-9 and all(flags.values())

        snapshot, command, rows = self._move_until(
            grasp, self.config.stable_grasp_jaw, 180, stable)
        flags, jaw, blockage, forces, displacement = self._stable_grasp_flags(
            snapshot, command, rows, reset_cube)
        if not all(flags.values()):
            raise RuntimeError("BOOTSTRAP_VALIDATION_FAILED")
        return snapshot, {
            "version": LIFT_BOOTSTRAP_VERSION, "flags": flags,
            "actual_jaw": jaw, "commanded_jaw": command.jaw_target,
            "blockage_gap": blockage, "forces": forces,
            "cube_displacement": displacement,
        }

    def reset(self):
        self.initialize(); last_error = None
        for attempt in range(1, self.config.bootstrap_attempts + 1):
            self._bootstrap_attempts_total += 1
            try:
                snapshot, bootstrap = self._bootstrap_once(); break
            except RuntimeError as exc:
                last_error = str(exc)
        else:
            raise RuntimeError("STABLE_GRASP_RESET_EXHAUSTED: %s" % last_error)
        bootstrap["attempts_for_reset"] = attempt
        self._bootstrap_last = bootstrap
        self._initial_cube = tuple(snapshot["cube_state"][:3])
        self._reference_relative = _sub(snapshot["cube_state"][:3], snapshot["ee_state"][:3])
        self._reference_rotation = tuple(snapshot["cube_state"][3:7])
        self._target = tuple(snapshot["ee_state"][:3])
        target_ee = (self._target[0], self._target[1], self._target[2] + self.config.target_lift)
        if not self.config.workspace.contains(self._target) or not self.config.workspace.contains(target_ee):
            raise RuntimeError("WORKSPACE_PROBE_FAILED")
        self._previous_action = (0.0,); self._steps = 0; self._hold_steps = 0
        self._max_rise = 0.0; self._invalid_grasp_steps = 0
        self._reached_20 = self._reached_40 = self._target_reached = self._hold_half = False
        self._hold_one = self._hold_three = self._hold_five = False
        self._maximum_hold_steps = 0
        self.phase = "LIFT_POLICY"
        return self._observation(snapshot)

    def _metrics(self, snapshot, commanded_jaw=None):
        cube, ee = snapshot["cube_state"], snapshot["ee_state"]
        relative = _sub(cube[:3], ee[:3])
        drift = _norm(_sub(relative, self._reference_relative))
        rotation = quaternion_distance(cube[3:7], self._reference_rotation)
        jaw = 0.5 * (snapshot["q"][6] + snapshot["q"][7])
        command = (self.compositor._commanded_jaw if commanded_jaw is None else commanded_jaw)
        blockage = jaw - float(command)
        forces = self._forces(snapshot)
        rise = cube[2] - self._initial_cube[2]
        contact = (min(forces.get("8_Link", 0.0), forces.get("7_Link", 0.0))
                   >= self.config.contact_threshold and
                   forces.get("cube", 0.0) >= self.config.contact_threshold)
        equivalent = blockage >= self.config.minimum_blockage and drift <= self.config.maximum_translation_drift
        return {"rise": rise, "translation_drift": drift, "rotation_drift": rotation,
                "jaw": jaw, "blockage_gap": blockage, "forces": forces,
                "contact": contact, "grasp_valid": jaw >= 0.01 and (contact or equivalent)}

    def _observation(self, snapshot):
        metrics = self._metrics(snapshot)
        ee, cube = snapshot["ee_state"], snapshot["cube_state"]
        workspace = self.config.workspace
        relative = _sub(cube[:3], ee[:3])
        return {
            "ee_z": (_clip((ee[2] - (workspace.lower[2] + workspace.upper[2]) / 2.0) /
                            ((workspace.upper[2] - workspace.lower[2]) / 2.0)),),
            "ee_linear_z_velocity": (_clip(ee[9] / 1.0),),
            "cube_z": (_clip((cube[2] - self.scene.config.cube_start[2]) / 0.10),),
            "cube_linear_velocity": tuple(_clip(v / 1.0) for v in cube[7:10]),
            "cube_angular_velocity": tuple(_clip(v / 5.0) for v in cube[10:13]),
            "cube_rise": (_clip(metrics["rise"] / self.config.target_lift),),
            "ee_cube_relative_position": tuple(_clip(v / 0.15) for v in relative),
            "relative_rotation_error": (_clip(metrics["rotation_drift"] / 0.25),),
            "jaw_actual": (_clip(metrics["jaw"] / 0.05, 0.0, 1.5),),
            "blockage_gap": (_clip(metrics["blockage_gap"] / self.config.minimum_blockage),),
            "target_height_error": (_clip((self.config.target_lift - metrics["rise"]) /
                                            self.config.target_lift),),
            "previous_dz_action": self._previous_action,
        }

    def step(self, action):
        if self.phase != "LIFT_POLICY":
            raise RuntimeError("episode is not in LIFT_POLICY phase")
        values = tuple(float(v) for v in action)
        if len(values) != 1 or not all(math.isfinite(v) for v in values):
            raise ValueError("ACTION_NON_FINITE_OR_DIMENSION")
        dz = max(-1.0, min(1.0, values[0]))
        requested = (self._target[0], self._target[1], self._target[2] + dz * self.config.dz_scale)
        try:
            snapshot, command = self._control(requested, self.config.stable_grasp_jaw)
        except Exception as exc:
            self.phase = "FAILURE"
            raise RuntimeError("CONTROLLER_FAILURE: %s" % exc) from exc
        self._target = self.config.workspace.clamp(requested)[0]
        self._previous_action = (dz,); self._steps += 1
        metrics = self._metrics(snapshot, command.jaw_target)
        rise = metrics["rise"]
        previous_max = self._max_rise
        self._max_rise = max(self._max_rise, rise)
        new_progress = max(0.0, min(self._max_rise, self.config.target_lift)
                           - min(previous_max, self.config.target_lift))
        in_band = self.config.target_band[0] <= rise <= self.config.target_band[1]
        stable = (metrics["translation_drift"] <= self.config.maximum_translation_drift
                  and metrics["rotation_drift"] <= self.config.maximum_rotation_drift
                  and metrics["jaw"] >= 0.01 and metrics["grasp_valid"] and self._finite(snapshot)
                  and max(metrics["forces"].values()) <= self.config.force_abort)
        if in_band and stable:
            self._hold_steps += 1
        else:
            self._hold_steps = 0
        self._maximum_hold_steps = max(self._maximum_hold_steps, self._hold_steps)
        self._reached_20 |= rise >= 0.020; self._reached_40 |= rise >= 0.040
        self._target_reached |= in_band
        self._hold_half |= self._hold_steps * self.control_dt >= 0.5
        self._hold_one |= self._hold_steps * self.control_dt >= 1.0
        self._hold_three |= self._hold_steps * self.control_dt >= 3.0
        self._hold_five |= self._hold_steps * self.control_dt >= 5.0
        self._invalid_grasp_steps = 0 if metrics["grasp_valid"] else self._invalid_grasp_steps + 1

        reason = ""
        if not self._finite(snapshot): reason = "NON_FINITE"
        elif command.workspace_clamped: reason = "WORKSPACE_VIOLATION"
        elif command.joint_limit_clamped: reason = "JOINT_LIMIT"
        elif max(metrics["forces"].values()) > self.config.force_abort: reason = "UNSAFE_FORCE"
        elif rise > self.config.overshoot_lift: reason = "TARGET_OVERSHOOT"
        elif self._reached_20 and rise < 0.008: reason = "DROP"
        elif self._invalid_grasp_steps >= 3: reason = "GRASP_LOST"
        elif metrics["translation_drift"] > 0.024 or metrics["rotation_drift"] > 0.50:
            reason = "GRASP_LOST"
        success = self._hold_steps >= self.hold_steps_required
        timeout = not success and not reason and self._steps >= self.max_steps
        if timeout: reason = "TIMEOUT"
        done = success or bool(reason)
        failure = bool(reason) and reason != "TIMEOUT"
        reward, terms = compute_lift_reward(
            config=self.config.reward, new_max_progress=new_progress,
            in_target_band=in_band, hold_valid=in_band and stable,
            translation_drift=metrics["translation_drift"],
            cube_angular_speed=_norm(snapshot["cube_state"][10:13]),
            cube_vertical_speed=snapshot["cube_state"][9], action=(dz,),
            terminal_failure=failure,
        )
        if done: self.phase = "SUCCESS" if success else "FAILURE"
        info = {
            "success": success, "failure": failure, "timeout": timeout,
            "failure_reason": reason, "distance_to_goal": abs(self.config.target_lift - rise),
            "initial_distance_to_goal": self.config.target_lift,
            "action_norm": abs(dz), "saturation_rate": float(abs(dz) >= 0.999),
            "object_displacement": rise, "reward_terms": terms,
            "cube_rise": rise, "maximum_cube_rise": self._max_rise,
            "target_reached": self._target_reached,
            "hold_duration": self._maximum_hold_steps * self.control_dt,
            "hold_half_second": self._hold_half,
            "hold_1s": self._hold_one, "hold_3s": self._hold_three,
            "hold_5s": self._hold_five,
            "relative_translation_drift": metrics["translation_drift"],
            "relative_rotation_drift": metrics["rotation_drift"],
            "blockage_gap": metrics["blockage_gap"], "jaw_actual": metrics["jaw"],
            "jaw_safe": metrics["jaw"] >= 0.01,
            "stable_grasp_valid": metrics["grasp_valid"],
            "reached_20mm": self._reached_20, "reached_40mm": self._reached_40,
            "unsafe": reason == "UNSAFE_FORCE", "non_finite": reason == "NON_FINITE",
            "drop": reason == "DROP", "grasp_lost": reason == "GRASP_LOST",
            "target_overshoot": reason == "TARGET_OVERSHOOT",
            "workspace_violation": reason == "WORKSPACE_VIOLATION",
            "bootstrap": dict(self._bootstrap_last),
        }
        return self._observation(snapshot), reward, done, info

    def workspace_probe(self):
        if self._initial_cube is None or self._target is None:
            raise RuntimeError("reset must complete before workspace probe")
        target_ee = (self._target[0], self._target[1], self._target[2] + self.config.target_lift)
        target_cube = (self._initial_cube[0], self._initial_cube[1],
                       self._initial_cube[2] + self.config.target_lift)
        return {"passed": self.config.workspace.contains(target_ee),
                "reset_ee": list(self._target), "target_ee": list(target_ee),
                "reset_cube": list(self._initial_cube), "target_cube": list(target_cube),
                "workspace_lower": list(self.config.workspace.lower),
                "workspace_upper": list(self.config.workspace.upper)}

    def close(self):
        if not self._closed:
            self._closed = True; self.scene.close()
