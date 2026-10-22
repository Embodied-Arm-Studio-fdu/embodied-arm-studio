"""Experimental single-environment S1 Grasp PPO task built on the P1 scene."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import random
from typing import Callable, Optional, Sequence, Tuple

from .evidence import GraspEvidenceConfig, evaluate_grasp, quaternion_distance
from .grasp_schema import (
    GraspRewardConfig, closure_reward_potentials, compute_grasp_reward,
)
from .scene import ManipulationScene, ManipulationSceneConfig
from .target_compositor import ExperimentalTargetCompositor, Workspace3D


CURRICULUM_STAGE_A = "A_descend_close"
CURRICULUM_STAGE_B = "B_align_descend_close"
CURRICULUM_STAGE_C = "C_full"
CURRICULUM_STAGES = (CURRICULUM_STAGE_A, CURRICULUM_STAGE_B, CURRICULUM_STAGE_C)
STAGE_A_DESCENT_DISTANCE = 0.045
STAGE_B_GRASP_XY_OFFSET = (-0.032, -0.036)
GRASP_VERIFIER_IMPLEMENTATION_REVISION = "grasp-verifier/v1.1"
PROGRESSIVE_RESET_B_FRACTIONS = {
    1: (0.5, 1.0),
    2: (0.0, 1.0),
    3: (0.0, 2.0 / 3.0),
    4: (0.0, 1.0 / 3.0),
    5: (0.0, 0.0),
}


def progressive_reset_pose(stage, stage_b_pose, full_pose):
    """Interpolate reset EE pose only; no stage value enters policy data."""
    if stage not in PROGRESSIVE_RESET_B_FRACTIONS:
        raise ValueError("progressive reset stage must be 1..5")
    xy_b_fraction, z_b_fraction = PROGRESSIVE_RESET_B_FRACTIONS[stage]
    return (
        full_pose[0] + xy_b_fraction * (stage_b_pose[0] - full_pose[0]),
        full_pose[1] + xy_b_fraction * (stage_b_pose[1] - full_pose[1]),
        full_pose[2] + z_b_fraction * (stage_b_pose[2] - full_pose[2]),
    )


def _norm(values):
    return math.sqrt(sum(float(v) ** 2 for v in values))


def _sub(left, right):
    return tuple(float(a) - float(b) for a, b in zip(left, right))


def _clip(value, low=-5.0, high=5.0):
    return max(low, min(high, float(value)))


def candidate_local_ee_rise(current_ee_z, verification_start_ee_z):
    """Return verifier-local EE rise; never fall back to an episode anchor."""
    if verification_start_ee_z is None:
        raise RuntimeError("VERIFIER_ANCHOR_MISSING")
    values = (float(current_ee_z), float(verification_start_ee_z))
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("VERIFIER_ANCHOR_NON_FINITE")
    return values[0] - values[1]


@dataclass(frozen=True)
class GraspEnvConfig:
    open_safe: float = 0.05
    close_safe: float = 0.025
    jaw_rate_limit: float = 0.001
    force_abort: float = 50.0
    physics_steps_per_action: int = 4
    learning_horizon: int = 120
    candidate_window: int = 6
    verification_steps: int = 30
    verification_lift: float = 0.025
    xyz_delta_scale: float = 0.005
    near_table_z_scale: float = 0.0025
    near_table_threshold: float = 0.42
    pregrasp_clearance: float = 0.080
    corridor_tolerance: float = 0.032
    premature_push_distance: float = 0.018
    randomize_xy: float = 0.0
    random_seed: int = 20261002
    curriculum_stage: str = CURRICULUM_STAGE_C
    workspace: Workspace3D = field(default_factory=Workspace3D)
    fixed_orientation_xyzw: Tuple[float, float, float, float] = (
        0.6040455102920532, -0.7851132154464722,
        0.12926433980464935, -0.044911183416843414,
    )
    grasp_frame_object_offset: Tuple[float, float, float] = (-0.008, 0.009, 0.012)
    reward: GraspRewardConfig = field(default_factory=GraspRewardConfig)

    def __post_init__(self):
        if not 0.01 <= self.close_safe < self.open_safe <= 0.05:
            raise ValueError("grasp jaw values are invalid")
        if self.physics_steps_per_action != 4:
            raise ValueError("P2 baseline requires four physics steps per action")
        if self.learning_horizon <= 0 or self.candidate_window < 2:
            raise ValueError("horizon/window must be positive")
        if not 0.0 <= self.randomize_xy <= 0.006:
            raise ValueError("P2 randomization must remain within P1 ±6 mm")
        if self.curriculum_stage not in CURRICULUM_STAGES:
            raise ValueError("unknown experimental grasp curriculum stage")
        if self.curriculum_stage != CURRICULUM_STAGE_C and self.randomize_xy != 0.0:
            raise ValueError("P2-R1 staged curriculum requires the fixed cube")


def physics_config_digest(config: GraspEnvConfig) -> str:
    grasp_config = asdict(config)
    # Curriculum stage changes only the reset distribution.  It is recorded
    # separately in checkpoints and must not masquerade as a physics change.
    grasp_config.pop("curriculum_stage", None)
    payload = {
        "grasp_env": grasp_config,
        "p1_scene": asdict(ManipulationSceneConfig()),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


class GraspPPOEnvironment:
    """LEARNING -> inline scripted VERIFYING -> terminal environment."""

    ARM_LOWER = (-2.967, 0.0, 0.0, -1.57, -1.57, -1.57)
    ARM_UPPER = (2.967, 3.14, 3.14, 1.518, 1.57, 1.57)

    def __init__(self, config: GraspEnvConfig = None, *, with_camera=False,
                 frame_callback: Optional[Callable] = None):
        self.config = config or GraspEnvConfig()
        scene_config = ManipulationSceneConfig(
            graphics_device_id=0 if with_camera else -1,
            with_camera=with_camera, initial_open=self.config.open_safe,
            fixed_orientation_xyzw=self.config.fixed_orientation_xyzw,
        )
        self.scene = ManipulationScene(scene_config)
        self.compositor = ExperimentalTargetCompositor(
            workspace=self.config.workspace, jaw_rate_limit=self.config.jaw_rate_limit)
        self.frame_callback = frame_callback
        self.rng = random.Random(self.config.random_seed)
        self._initialized = False
        self._closed = False
        self._episode_index = 0
        self.phase = "UNINITIALIZED"
        self._target = None
        self._previous_action = (0.0,) * 4
        self._initial_cube = None
        self._initial_ee = None
        self._verification_start_ee_pose = None
        self._verification_metrics = {}
        self._initial_distance = 0.0
        self._candidate_rows = []
        self._candidate_awarded = False
        self._learning_steps = 0
        self._reward_totals = {}
        self._last_info = {}
        self._last_candidate_flags = {}
        self._candidate_stabilizing = False
        self._stabilizing_target = None
        self._stabilizing_jaw_target = None
        self._stabilizing_steps = 0
        self._stabilizing_entered_once = False
        self._stabilizing_completed_once = False
        self._stabilizing_evidence_lost_count = 0

    def initialize(self):
        if not self._initialized:
            self.scene.initialize()
            self._initialized = True
        return self

    @property
    def property_dump(self):
        return self.scene.property_dump

    def _cube_xy(self):
        radius = self.config.randomize_xy
        base = self.scene.config.cube_xy
        return (base[0] + self.rng.uniform(-radius, radius),
                base[1] + self.rng.uniform(-radius, radius))

    def _curriculum_reset_target(self, snapshot):
        """Return the final reset EE pose; no stage data enters observation."""
        if self.config.curriculum_stage == CURRICULUM_STAGE_C:
            return None
        grasp = self._grasp_point(snapshot)
        if self.config.curriculum_stage == CURRICULUM_STAGE_A:
            return (grasp[0], grasp[1], grasp[2] + STAGE_A_DESCENT_DISTANCE)
        return (
            grasp[0] + STAGE_B_GRASP_XY_OFFSET[0],
            grasp[1] + STAGE_B_GRASP_XY_OFFSET[1],
            grasp[2] + self.config.pregrasp_clearance,
        )

    def _move_to_curriculum_reset(self, snapshot, expected_xy):
        final_target = self._curriculum_reset_target(snapshot)
        if final_target is None:
            return snapshot
        grasp = self._grasp_point(snapshot)
        clearance_z = grasp[2] + self.config.pregrasp_clearance
        current = tuple(snapshot["ee_state"][:3])
        waypoints = [
            (current[0], current[1], clearance_z),
            (final_target[0], final_target[1], clearance_z),
        ]
        if abs(final_target[2] - clearance_z) > 1.0e-6:
            waypoints.append(final_target)
        self.compositor.reset(snapshot["q"])
        for target in waypoints:
            reached = False
            for _ in range(240):
                state = self.scene.robot.read_state()
                command = self.compositor.compose(
                    target, self.config.fixed_orientation_xyzw,
                    self.config.open_safe, state,
                )
                self.scene.robot.set_joint_position_targets(command.positions)
                self.scene.robot.step(self.config.physics_steps_per_action)
                snapshot = self.scene.snapshot()
                values = snapshot["q"] + snapshot["dq"] + snapshot["ee_state"] + snapshot["cube_state"]
                if not all(math.isfinite(float(value)) for value in values):
                    raise RuntimeError("RESET_CURRICULUM_NON_FINITE")
                if math.hypot(snapshot["cube_state"][0] - expected_xy[0],
                              snapshot["cube_state"][1] - expected_xy[1]) > 0.004:
                    raise RuntimeError("RESET_CURRICULUM_MOVED_CUBE")
                if max(self._forces(snapshot).values()) > self.config.force_abort:
                    raise RuntimeError("RESET_CURRICULUM_UNSAFE_FORCE")
                if _norm(_sub(snapshot["ee_state"][:3], target)) <= 0.006:
                    reached = True
                    break
            if not reached:
                raise RuntimeError("RESET_CURRICULUM_TARGET_TIMEOUT")
        # Hold the open-jaw target briefly so the policy never receives a
        # moving reset state.  This is reset initialization, not LEARNING.
        for _ in range(12):
            state = self.scene.robot.read_state()
            command = self.compositor.compose(
                final_target, self.config.fixed_orientation_xyzw,
                self.config.open_safe, state,
            )
            self.scene.robot.set_joint_position_targets(command.positions)
            self.scene.robot.step(self.config.physics_steps_per_action)
        snapshot = self.scene.snapshot()
        self._validate_reset(snapshot, expected_xy)
        if _norm(_sub(snapshot["ee_state"][:3], final_target)) > 0.008:
            raise RuntimeError("RESET_CURRICULUM_FINAL_POSE")
        return snapshot

    def reset(self):
        self.initialize()
        xy = self._cube_xy()
        last_error = None
        placement_xy = xy
        for reset_attempt in range(3):
            self.scene.reset(placement_xy)
            snapshot = self.scene.snapshot()
            try:
                self._validate_reset(snapshot, xy)
                snapshot = self._move_to_curriculum_reset(snapshot, xy)
                break
            except RuntimeError as exc:
                last_error = exc
                cube = snapshot["cube_state"]
                correction = (cube[0] - xy[0], cube[1] - xy[1])
                if math.hypot(*correction) > 0.006:
                    break
                # Contact warm-start from the previous terminal pose can move
                # the cube during the settle steps.  Compensate only inside the
                # P1-verified ±6 mm region, then validate against the original
                # requested reset (never against the compensated placement).
                placement_xy = (placement_xy[0] - correction[0],
                                placement_xy[1] - correction[1])
        else:
            raise RuntimeError("RESET_RETRY_EXHAUSTED: %s" % last_error)
        self._reset_attempts = reset_attempt + 1
        self._episode_index += 1
        self.phase = "LEARNING"
        self._target = tuple(snapshot["ee_state"][:3])
        self._previous_action = (0.0,) * 4
        self._initial_cube = tuple(snapshot["cube_state"][:3])
        self._initial_ee = tuple(snapshot["ee_state"][:3])
        self._clear_verification_anchor()
        self._candidate_rows = []
        self._candidate_awarded = False
        self._learning_steps = 0
        self._reward_totals = {}
        self._last_info = {}
        self._last_candidate_flags = {}
        self._candidate_stabilizing = False
        self._stabilizing_target = None
        self._stabilizing_jaw_target = None
        self._stabilizing_steps = 0
        self._stabilizing_entered_once = False
        self._stabilizing_completed_once = False
        self._stabilizing_evidence_lost_count = 0
        self.compositor.reset(snapshot["q"])
        self._initial_distance = self._grasp_distance(snapshot)
        initial_corridor = self._corridor_error(snapshot)
        self._minimum_distance = self._initial_distance
        self._minimum_xy_error = math.hypot(*initial_corridor[:2])
        self._minimum_z_error = abs(initial_corridor[2])
        self._aligned_once = self._minimum_xy_error <= 0.020
        self._descended_once = self._minimum_z_error <= 0.015
        # Diagnostics only.  Unlike ``descended``, this metric cannot be
        # satisfied by an episode that merely starts near grasp height: the
        # policy must first visit the pre-grasp clearance during LEARNING.
        self._clearance_reached_once = False
        self._post_clearance_descended_once = False
        self._closure_started_once = False
        self._candidate_once = False
        self._verification_entered_once = False
        self._previous_close_potential = 0.0
        self._previous_blockage_potential = 0.0
        self._minimum_commanded_jaw = self.config.open_safe
        self._actual_jaw_sum = 0.0
        self._actual_jaw_count = 0
        self._maximum_blockage_gap = 0.0
        self._time_to_contact = None
        self._time_to_candidate = None
        self._jaw_below_040 = False
        self._jaw_below_035 = False
        self._jaw_below_030 = False
        self._blockage_ge_0005 = False
        self._blockage_ge_0010 = False
        self._blockage_ge_0015 = False
        self._blockage_ge_0020 = False
        return self._observation(snapshot)

    def _validate_reset(self, snapshot, expected_xy):
        values = snapshot["q"] + snapshot["dq"] + snapshot["ee_state"] + snapshot["cube_state"]
        if not all(math.isfinite(float(v)) for v in values):
            raise RuntimeError("RESET_NON_FINITE")
        cube = snapshot["cube_state"]
        if math.hypot(cube[0] - expected_xy[0], cube[1] - expected_xy[1]) > 0.004:
            raise RuntimeError("RESET_CUBE_POSITION expected=%r actual=%r" %
                               (expected_xy, tuple(cube[:3])))
        if _norm(cube[7:13]) > 0.06:
            raise RuntimeError("RESET_CUBE_NOT_SETTLED")
        bottom = cube[2] - self.scene.config.cube_size / 2.0
        if bottom < self.scene.config.table_top - 0.004:
            raise RuntimeError("RESET_PENETRATION")
        if snapshot["estimated_aperture"] < 0.07:
            raise RuntimeError("RESET_GRIPPER_NOT_OPEN q=%r aperture=%r" %
                               (tuple(snapshot["q"][6:8]), snapshot["estimated_aperture"]))
        forces = self._forces(snapshot)
        if max(forces.get("8_Link", 0.0), forces.get("7_Link", 0.0),
               forces.get("cube", 0.0)) > self.config.force_abort:
            raise RuntimeError("RESET_UNSAFE_FORCE")

    def _forces(self, snapshot):
        return {name: _norm(body["force"]) for name, body in snapshot["bodies"].items()}

    def _grasp_point(self, snapshot):
        cube = snapshot["cube_state"][:3]
        return tuple(cube[i] - self.config.grasp_frame_object_offset[i] for i in range(3))

    def _corridor_error(self, snapshot):
        cube = snapshot["cube_state"][:3]
        ee = snapshot["ee_state"][:3]
        expected = tuple(ee[i] + self.config.grasp_frame_object_offset[i] for i in range(3))
        return _sub(cube, expected)

    def _grasp_distance(self, snapshot):
        return _norm(self._corridor_error(snapshot))

    def _stage_target(self, snapshot):
        grasp = self._grasp_point(snapshot)
        pre = (grasp[0], grasp[1], grasp[2] + self.config.pregrasp_clearance)
        ee = snapshot["ee_state"][:3]
        xy_error = math.hypot(ee[0] - grasp[0], ee[1] - grasp[1])
        if ee[2] < pre[2] - 0.015 and xy_error > 0.030:
            return (self._initial_ee[0], self._initial_ee[1], pre[2])
        if xy_error > 0.018:
            return pre
        return grasp

    def _observation(self, snapshot):
        q, dq = snapshot["q"], snapshot["dq"]
        ee, cube = snapshot["ee_state"], snapshot["cube_state"]
        arm_q = []
        for value, low, high in zip(q[:6], self.ARM_LOWER, self.ARM_UPPER):
            arm_q.append(_clip((value - (low + high) / 2.0) / ((high - low) / 2.0)))
        workspace = self.config.workspace
        ee_norm = tuple(_clip((ee[i] - (workspace.lower[i] + workspace.upper[i]) / 2.0) /
                              ((workspace.upper[i] - workspace.lower[i]) / 2.0)) for i in range(3))
        nominal = self.scene.config.cube_start
        relative = _sub(cube[:3], ee[:3])
        corridor = self._corridor_error(snapshot)
        clearance = cube[2] - self.scene.config.cube_size / 2.0 - self.scene.config.table_top
        return {
            "arm_q": tuple(arm_q),
            "arm_dq": tuple(_clip(v / 10.0) for v in dq[:6]),
            "gripper_q": tuple(_clip(2.0 * v / 0.05 - 1.0) for v in q[6:8]),
            "gripper_dq": tuple(_clip(v / 0.2) for v in dq[6:8]),
            "ee_position": ee_norm,
            "object_position": tuple(_clip((cube[i] - nominal[i]) / 0.10) for i in range(3)),
            "object_velocity": tuple(_clip(v / (1.0 if i < 3 else 5.0))
                                     for i, v in enumerate(cube[7:13])),
            "ee_object_relative_position": tuple(_clip(v / 0.15) for v in relative),
            "gripper_width": (_clip(snapshot["estimated_aperture"] / 0.0901, 0.0, 1.5),),
            "previous_action": self._previous_action,
            "grasp_corridor_error": tuple(_clip(v / 0.10) for v in corridor),
            "object_table_clearance": (_clip(clearance / 0.05, -1.0, 3.0),),
        }

    def _map_action(self, action):
        values = tuple(max(-1.0, min(1.0, float(v))) for v in action)
        if len(values) != 4 or not all(math.isfinite(v) for v in values):
            raise ValueError("ACTION_NON_FINITE_OR_DIMENSION")
        z_scale = (self.config.near_table_z_scale
                   if self._target[2] < self.config.near_table_threshold
                   else self.config.xyz_delta_scale)
        scales = (self.config.xyz_delta_scale, self.config.xyz_delta_scale, z_scale)
        requested = tuple(self._target[i] + values[i] * scales[i] for i in range(3))
        jaw = self.config.close_safe + ((values[3] + 1.0) * 0.5) * (
            self.config.open_safe - self.config.close_safe)
        return values, requested, jaw

    def _candidate_base_flags(self, snapshot, jaw_target):
        forces = self._forces(snapshot)
        actual = 0.5 * (snapshot["q"][6] + snapshot["q"][7])
        return {
            "finite": all(math.isfinite(float(v)) for v in (
                snapshot["q"] + snapshot["dq"] + snapshot["ee_state"] + snapshot["cube_state"])),
            "jaw_closed": actual <= self.config.open_safe - 0.004,
            "jaw_safe": actual >= 0.01,
            "in_corridor": self._grasp_distance(snapshot) <= self.config.corridor_tolerance,
            "jaw_blocked": actual >= float(jaw_target) + 0.002,
            "bilateral_force": min(forces.get("8_Link", 0.0),
                                     forces.get("7_Link", 0.0)) >= 0.2,
            "cube_force": forces.get("cube", 0.0) >= 0.2,
        }

    def _candidate(self, snapshot, command):
        flags = self._candidate_base_flags(snapshot, command.jaw_target)
        base = all(flags.values())
        self._candidate_rows.append((snapshot["q"][6], snapshot["q"][7], base))
        self._candidate_rows = self._candidate_rows[-self.config.candidate_window:]
        flags["window_full"] = len(self._candidate_rows) >= self.config.candidate_window
        if not flags["window_full"]:
            flags["jaw_window_stable"] = False
            self._last_candidate_flags = flags
            return False
        stable = all(row[2] for row in self._candidate_rows)
        window_stable = all(
            max(row[index] for row in self._candidate_rows)
            - min(row[index] for row in self._candidate_rows) < 0.0003
            for index in (0, 1)
        )
        flags["jaw_window_stable"] = window_stable
        self._last_candidate_flags = flags
        return stable and window_stable

    def _evidence_config(self):
        return GraspEvidenceConfig(
            open_safe=self.config.open_safe, self_contact_floor=0.01,
            corridor_tolerance=self.config.corridor_tolerance,
        )

    def _abort_reason(self, snapshot, command, before_candidate=True):
        values = snapshot["q"] + snapshot["dq"] + snapshot["ee_state"] + snapshot["cube_state"]
        if not all(math.isfinite(float(v)) for v in values): return "NON_FINITE"
        if command.workspace_clamped: return "WORKSPACE_VIOLATION"
        if command.joint_limit_clamped: return "JOINT_LIMIT"
        cube = snapshot["cube_state"]
        if abs(cube[0] - self.scene.config.table_position[0]) > self.scene.config.table_size[0] / 2:
            return "OBJECT_OUT_OF_TABLE"
        if abs(cube[1] - self.scene.config.table_position[1]) > self.scene.config.table_size[1] / 2:
            return "OBJECT_OUT_OF_TABLE"
        if cube[2] - self.scene.config.cube_size / 2 < self.scene.config.table_top - 0.004:
            return "UNSAFE_FORCE"  # penetration is a safety failure
        forces = self._forces(snapshot)
        if max(forces.get("8_Link", 0.0), forces.get("7_Link", 0.0),
               forces.get("cube", 0.0)) > self.config.force_abort:
            return "UNSAFE_FORCE"
        if before_candidate and math.hypot(cube[0] - self._initial_cube[0],
                                           cube[1] - self._initial_cube[1]) > self.config.premature_push_distance:
            return "PREMATURE_OBJECT_PUSH"
        return ""

    def _capture(self):
        if self.frame_callback is not None:
            self.frame_callback(self.scene.capture())

    def _clear_verification_anchor(self):
        self._verification_start_ee_pose = None
        self._verification_metrics = {}

    def _set_verification_anchor(self, candidate_snapshot):
        if self._verification_start_ee_pose is not None:
            raise RuntimeError("VERIFIER_ANCHOR_ALREADY_SET")
        pose = tuple(float(value) for value in candidate_snapshot["ee_state"][:7])
        if len(pose) != 7 or not all(math.isfinite(value) for value in pose):
            raise RuntimeError("VERIFIER_ANCHOR_NON_FINITE")
        self._verification_start_ee_pose = pose

    def _verify(self, candidate_snapshot, command):
        try:
            self._set_verification_anchor(candidate_snapshot)
        except RuntimeError as exc:
            return False, str(exc), candidate_snapshot, 0, 0.0, 0.0, None
        self.phase = "VERIFYING"
        reference_translation = _sub(candidate_snapshot["cube_state"][:3],
                                     candidate_snapshot["ee_state"][:3])
        reference_rotation = tuple(candidate_snapshot["cube_state"][3:7])
        lift_target = (self._target[0], self._target[1],
                       min(self.config.workspace.upper[2], self._target[2] + self.config.verification_lift))
        max_drift = 0.0
        last_evidence = None
        candidate_cube_z = float(candidate_snapshot["cube_state"][2])
        for step in range(self.config.verification_steps):
            state = self.scene.robot.read_state()
            verify_command = self.compositor.compose(
                lift_target, self.config.fixed_orientation_xyzw,
                command.jaw_target, state)
            self.scene.robot.set_joint_position_targets(verify_command.positions)
            self.scene.robot.step(self.config.physics_steps_per_action)
            self._capture()
            snapshot = self.scene.snapshot()
            abort = self._abort_reason(snapshot, verify_command, before_candidate=False)
            if abort:
                return False, abort, snapshot, step + 1, 0.0, max_drift, None
            cube, ee = snapshot["cube_state"], snapshot["ee_state"]
            relative = _sub(cube[:3], ee[:3])
            drift = _norm(_sub(relative, reference_translation)); max_drift = max(max_drift, drift)
            rotation_drift = quaternion_distance(cube[3:7], reference_rotation)
            clearance = cube[2] - self.scene.config.cube_size / 2 - self.scene.config.table_top
            forces = self._forces(snapshot)
            try:
                local_ee_rise = candidate_local_ee_rise(
                    ee[2], self._verification_start_ee_pose[2]
                    if self._verification_start_ee_pose is not None else None)
            except RuntimeError as exc:
                return False, str(exc), snapshot, step + 1, 0.0, max_drift, None
            reset_relative_ee_rise = ee[2] - self._initial_ee[2]
            reset_relative_cube_rise = cube[2] - self._initial_cube[2]
            candidate_local_cube_rise = cube[2] - candidate_cube_z
            self._verification_metrics = {
                "verifier_revision": GRASP_VERIFIER_IMPLEMENTATION_REVISION,
                "verification_start_ee_pose": self._verification_start_ee_pose,
                "candidate_local_ee_rise": local_ee_rise,
                "reset_relative_ee_rise": reset_relative_ee_rise,
                "cube_rise": reset_relative_cube_rise,
                "candidate_local_cube_rise": candidate_local_cube_rise,
                "lift_ratio": (reset_relative_cube_rise / local_ee_rise
                               if local_ee_rise > 0.0 else 0.0),
                "candidate_local_lift_ratio": (
                    candidate_local_cube_rise / local_ee_rise
                    if local_ee_rise > 0.0 else 0.0),
                "translation_drift": drift,
                "rotation_drift": rotation_drift,
                "clearance": clearance,
                "verification_steps": step + 1,
            }
            last_evidence = evaluate_grasp(
                config=self._evidence_config(),
                actual_jaw=0.5 * (snapshot["q"][6] + snapshot["q"][7]),
                object_in_corridor=self._grasp_distance(snapshot) <= self.config.corridor_tolerance,
                object_table_clearance=clearance,
                object_rise=reset_relative_cube_rise,
                ee_rise=local_ee_rise,
                relative_translation_drift=drift,
                relative_rotation_drift=rotation_drift,
                state_values=values_for_evidence(snapshot),
                left_force=forces.get("8_Link", 0.0), right_force=forces.get("7_Link", 0.0),
                object_force=forces.get("cube", 0.0),
            )
            if last_evidence.grasp_verified:
                self._target = lift_target
                return True, "", snapshot, step + 1, clearance, max_drift, last_evidence
        return (False, "GRASP_VERIFICATION_FAILED", snapshot,
                self.config.verification_steps, 0.0, max_drift, last_evidence)

    def step(self, action):
        if self.phase not in ("LEARNING", "CANDIDATE_STABILIZING"):
            raise RuntimeError("episode is not in LEARNING phase")
        before = self.scene.snapshot()
        previous_stage_target = self._stage_target(before)
        previous_stage_distance = _norm(_sub(before["ee_state"][:3], previous_stage_target))
        previous_alignment = math.hypot(*self._corridor_error(before)[:2])
        grasp = self._grasp_point(before)
        previous_z_error = abs(before["ee_state"][2] - grasp[2])
        stabilizing_entered_event = False
        current_jaw_target = self.compositor._commanded_jaw
        if current_jaw_target is None:
            current_jaw_target = 0.5 * (before["q"][6] + before["q"][7])
        base_before = self._candidate_base_flags(before, current_jaw_target)
        if not self._candidate_stabilizing and all(base_before.values()):
            self._candidate_stabilizing = True
            self._stabilizing_target = tuple(self._target)
            self._stabilizing_jaw_target = float(current_jaw_target)
            self._stabilizing_entered_once = True
            self.phase = "CANDIDATE_STABILIZING"
            stabilizing_entered_event = True
        values, requested, desired_jaw = self._map_action(action)
        if self._candidate_stabilizing:
            requested = self._stabilizing_target
            desired_jaw = self._stabilizing_jaw_target
        self._target = requested
        requested_target_preclamp = tuple(requested)
        persistent_target_postclamp, _ = self.config.workspace.clamp(
            requested_target_preclamp)
        try:
            state = self.scene.robot.read_state()
            command = self.compositor.compose(
                self._target, self.config.fixed_orientation_xyzw, desired_jaw, state)
            self._target = self.config.workspace.clamp(self._target)[0]
            self.scene.robot.set_joint_position_targets(command.positions)
            self.scene.robot.step(self.config.physics_steps_per_action)
            self._capture()
        except Exception as exc:
            self.phase = "FAILURE"
            raise RuntimeError("CONTROLLER_FAILURE: %s" % exc) from exc
        after = self.scene.snapshot()
        self._learning_steps += 1
        stage_distance = _norm(_sub(after["ee_state"][:3], previous_stage_target))
        alignment = math.hypot(*self._corridor_error(after)[:2])
        current_grasp = self._grasp_point(after)
        z_error = abs(after["ee_state"][2] - current_grasp[2])
        candidate = self._candidate(after, command)
        candidate_window_progress = 0
        for width in range(1, min(len(self._candidate_rows), self.config.candidate_window) + 1):
            rows = self._candidate_rows[-width:]
            if not all(row[2] for row in rows):
                break
            if not all(max(row[index] for row in rows) - min(row[index] for row in rows) < 0.0003
                       for index in (0, 1)):
                break
            candidate_window_progress = width
        stabilizing_evidence_lost_event = False
        stabilizing_completed_event = False
        if self._candidate_stabilizing:
            self._stabilizing_steps += 1
            base_after = all(self._last_candidate_flags.get(name, False) for name in (
                "finite", "jaw_closed", "jaw_safe", "in_corridor",
                "jaw_blocked", "bilateral_force", "cube_force"))
            if candidate:
                self._candidate_stabilizing = False
                self._stabilizing_completed_once = True
                stabilizing_completed_event = True
            elif not base_after:
                self._candidate_stabilizing = False
                self._stabilizing_target = None
                self._stabilizing_jaw_target = None
                self._stabilizing_evidence_lost_count += 1
                stabilizing_evidence_lost_event = True
                self.phase = "LEARNING"
        corridor = self._corridor_error(after)
        current_distance = _norm(corridor)
        current_xy_error = math.hypot(*corridor[:2])
        current_z_error = abs(corridor[2])
        actual_jaw = 0.5 * (after["q"][6] + after["q"][7])
        forces = self._forces(after)
        blockage_gap = max(0.0, actual_jaw - command.jaw_target)
        finite = all(math.isfinite(float(v)) for v in (
            after["q"] + after["dq"] + after["ee_state"] + after["cube_state"]))
        cube_xy_displacement = math.hypot(
            after["cube_state"][0] - self._initial_cube[0],
            after["cube_state"][1] - self._initial_cube[1],
        )
        cube_bottom = after["cube_state"][2] - self.scene.config.cube_size / 2.0
        maximum_force = max(forces.values())
        close_potential, blockage_potential, closure_audit = closure_reward_potentials(
            open_safe=self.config.open_safe, close_safe=self.config.close_safe,
            commanded_jaw=command.jaw_target, actual_jaw=actual_jaw,
            finite=finite, in_corridor=current_distance <= self.config.corridor_tolerance,
            xy_aligned=current_xy_error <= 0.020,
            at_grasp_height=current_z_error <= 0.015,
            no_premature_push=(cube_xy_displacement <= self.config.premature_push_distance),
            safe_contact=(cube_bottom >= self.scene.config.table_top - 0.004
                          and maximum_force <= self.config.force_abort),
            bilateral_contact=(min(forces.get("8_Link", 0.0),
                                   forces.get("7_Link", 0.0)) >= 0.2),
            cube_contact=forces.get("cube", 0.0) >= 0.2,
        )
        closure_gate = closure_audit["closure_gate"]
        contact_gate = closure_audit["contact_gate"]
        close_fraction = closure_audit["close_fraction"]
        closure_potential_progress = close_potential - self._previous_close_potential
        blockage_potential_progress = (
            blockage_potential - self._previous_blockage_potential)
        self._previous_close_potential = close_potential
        self._previous_blockage_potential = blockage_potential
        self._minimum_commanded_jaw = min(self._minimum_commanded_jaw, command.jaw_target)
        self._actual_jaw_sum += actual_jaw
        self._actual_jaw_count += 1
        self._maximum_blockage_gap = max(self._maximum_blockage_gap, blockage_gap)
        self._jaw_below_040 = self._jaw_below_040 or command.jaw_target < 0.040
        self._jaw_below_035 = self._jaw_below_035 or command.jaw_target < 0.035
        self._jaw_below_030 = self._jaw_below_030 or command.jaw_target < 0.030
        self._blockage_ge_0005 = self._blockage_ge_0005 or blockage_gap >= 0.0005
        self._blockage_ge_0010 = self._blockage_ge_0010 or blockage_gap >= 0.0010
        self._blockage_ge_0015 = self._blockage_ge_0015 or blockage_gap >= 0.0015
        self._blockage_ge_0020 = self._blockage_ge_0020 or blockage_gap >= 0.0020
        if contact_gate and self._time_to_contact is None:
            self._time_to_contact = (
                self._learning_steps * self.scene.robot.config.physics_dt
                * self.config.physics_steps_per_action)
        self._minimum_distance = min(self._minimum_distance, current_distance)
        self._minimum_xy_error = min(self._minimum_xy_error, current_xy_error)
        self._minimum_z_error = min(self._minimum_z_error, current_z_error)
        self._aligned_once = self._aligned_once or current_xy_error <= 0.020
        self._descended_once = self._descended_once or current_z_error <= 0.015
        clearance_z = current_grasp[2] + self.config.pregrasp_clearance
        at_clearance = abs(after["ee_state"][2] - clearance_z) <= 0.015
        self._clearance_reached_once = self._clearance_reached_once or at_clearance
        self._post_clearance_descended_once = (
            self._post_clearance_descended_once
            or (self._clearance_reached_once and current_z_error <= 0.015)
        )
        self._closure_started_once = (
            self._closure_started_once or actual_jaw <= self.config.open_safe - 0.004)
        self._candidate_once = self._candidate_once or candidate
        candidate_event = candidate and not self._candidate_awarded
        if candidate_event: self._candidate_awarded = True
        if candidate and self._time_to_candidate is None:
            self._time_to_candidate = (
                self._learning_steps * self.scene.robot.config.physics_dt
                * self.config.physics_steps_per_action)
        failure_reason = self._abort_reason(after, command)
        verified = False; verification_steps = 0; verification_lift = 0.0
        max_drift = 0.0; evidence = None
        terminal_snapshot = after
        if not failure_reason and candidate:
            self._verification_entered_once = True
            verified, failure_reason, terminal_snapshot, verification_steps, verification_lift, max_drift, evidence = self._verify(after, command)
        timeout = not failure_reason and not verified and self._learning_steps >= self.config.learning_horizon
        if timeout: failure_reason = "TIMEOUT"
        done = bool(verified or failure_reason)
        if done: self.phase = "SUCCESS" if verified else "FAILURE"
        saturated = sum(abs(v) >= 0.98 for v in values) / 4.0
        reward, terms = compute_grasp_reward(
            config=self.config.reward,
            approach_progress=previous_stage_distance - stage_distance,
            alignment_progress=previous_alignment - alignment,
            descend_progress=previous_z_error - z_error,
            xy_aligned=alignment <= 0.020,
            closure_potential_progress=closure_potential_progress,
            blockage_potential_progress=blockage_potential_progress,
            candidate_event=candidate_event, verified_event=verified,
            action=values, saturated=saturated,
            clamped=command.workspace_clamped or command.joint_limit_clamped,
            # TIMEOUT is an ordinary learning boundary and is bootstrapped by
            # the shared PPO runner.  Penalizing it as a -5 safety failure made
            # that action-independent terminal dominate metre-scale progress.
            failure=bool(failure_reason and failure_reason != "TIMEOUT"),
        )
        for name, value in terms.items(): self._reward_totals[name] = self._reward_totals.get(name, 0.0) + value
        self._previous_action = values
        cube = terminal_snapshot["cube_state"]
        info = {
            "success": verified, "timeout": timeout, "failure": bool(failure_reason),
            "failure_reason": failure_reason, "phase": self.phase,
            "distance_to_goal": self._grasp_distance(terminal_snapshot),
            "initial_distance_to_goal": self._initial_distance,
            "object_displacement": _norm(_sub(cube[:3], self._initial_cube)),
            "action_norm": _norm(values), "saturation_rate": saturated,
            "reward_terms": terms, "reward_term_totals": dict(self._reward_totals),
            "grasp_candidate": bool(candidate), "candidate_event": candidate_event,
            "verified_success": verified, "premature_push": failure_reason == "PREMATURE_OBJECT_PUSH",
            "unsafe": failure_reason in ("UNSAFE_FORCE", "NON_FINITE"),
            "non_finite": failure_reason == "NON_FINITE",
            "learning_steps": self._learning_steps,
            "time_to_grasp": self._learning_steps * self.scene.robot.config.physics_dt * self.config.physics_steps_per_action,
            "verification_steps": verification_steps,
            "verification_lift": verification_lift,
            "relative_drift": max_drift,
            "verifier_revision": GRASP_VERIFIER_IMPLEMENTATION_REVISION,
            "verification_start_ee_pose": self._verification_start_ee_pose,
            "candidate_local_ee_rise": float(self._verification_metrics.get(
                "candidate_local_ee_rise", 0.0)),
            "reset_relative_ee_rise": float(self._verification_metrics.get(
                "reset_relative_ee_rise", 0.0)),
            "cube_rise": float(self._verification_metrics.get("cube_rise", 0.0)),
            "candidate_local_cube_rise": float(self._verification_metrics.get(
                "candidate_local_cube_rise", 0.0)),
            "lift_ratio": float(self._verification_metrics.get("lift_ratio", 0.0)),
            "candidate_local_lift_ratio": float(self._verification_metrics.get(
                "candidate_local_lift_ratio", 0.0)),
            "rotation_drift": float(self._verification_metrics.get("rotation_drift", 0.0)),
            "verification_clearance": float(self._verification_metrics.get("clearance", 0.0)),
            "time_to_verified": (
                (self._learning_steps + verification_steps)
                * self.scene.robot.config.physics_dt * self.config.physics_steps_per_action
                if verified else None),
            "grasp_confidence": evidence.confidence if evidence else 0.0,
            "force_max": max(self._forces(terminal_snapshot).values()),
            "reset_attempts": self._reset_attempts,
            "curriculum_stage": self.config.curriculum_stage,
            "minimum_distance": self._minimum_distance,
            "minimum_xy_error": self._minimum_xy_error,
            "final_xy_error": current_xy_error,
            "minimum_z_error": self._minimum_z_error,
            "final_z_error": current_z_error,
            "aligned": self._aligned_once,
            "descended": self._descended_once,
            "clearance_reached": self._clearance_reached_once,
            "post_clearance_descended": self._post_clearance_descended_once,
            "closure_started": self._closure_started_once,
            "candidate_once": self._candidate_once,
            "verification_entered": self._verification_entered_once,
            "candidate_evidence_flags": dict(self._last_candidate_flags),
            "candidate_window_progress": candidate_window_progress,
            "candidate_window_required": self.config.candidate_window,
            "candidate_only_missing_stability_window": bool(
                not candidate
                and all(self._last_candidate_flags.get(name, False) for name in (
                    "finite", "jaw_closed", "jaw_safe", "in_corridor",
                    "jaw_blocked", "bilateral_force", "cube_force"))
                and candidate_window_progress < self.config.candidate_window),
            "stabilizing_active": self._candidate_stabilizing,
            "stabilizing_entered": stabilizing_entered_event,
            "stabilizing_entered_once": self._stabilizing_entered_once,
            "stabilizing_completed": stabilizing_completed_event,
            "stabilizing_completed_once": self._stabilizing_completed_once,
            "stabilizing_evidence_lost": stabilizing_evidence_lost_event,
            "stabilizing_evidence_lost_count": self._stabilizing_evidence_lost_count,
            "stabilization_steps": self._stabilizing_steps,
            "stabilization_hold_target": self._stabilizing_target,
            "stabilization_hold_jaw_target": self._stabilizing_jaw_target,
            "stabilization_z_target_margin": (
                min(
                    (self._stabilizing_target or self._target)[2]
                    - self.config.workspace.lower[2],
                    self.config.workspace.upper[2]
                    - (self._stabilizing_target or self._target)[2],
                )),
            "closure_gate": closure_gate,
            "contact_gate": contact_gate,
            "close_fraction": close_fraction,
            "blockage_gap": blockage_gap,
            "minimum_commanded_jaw": self._minimum_commanded_jaw,
            "mean_actual_jaw": self._actual_jaw_sum / max(1, self._actual_jaw_count),
            "maximum_blockage_gap": self._maximum_blockage_gap,
            "time_to_contact": self._time_to_contact,
            "time_to_candidate": self._time_to_candidate,
            "jaw_below_040": self._jaw_below_040,
            "jaw_below_035": self._jaw_below_035,
            "jaw_below_030": self._jaw_below_030,
            "blockage_ge_0005": self._blockage_ge_0005,
            "blockage_ge_0010": self._blockage_ge_0010,
            "blockage_ge_0015": self._blockage_ge_0015,
            "blockage_ge_0020": self._blockage_ge_0020,
            # Compact diagnostic state.  These are metrics only; they are not
            # part of the policy observation or the success contract.
            "ee_position": tuple(terminal_snapshot["ee_state"][:3]),
            "cube_position": tuple(terminal_snapshot["cube_state"][:3]),
            "persistent_target": tuple(self._target),
            "learning_ee_position": tuple(after["ee_state"][:3]),
            "learning_cube_position": tuple(after["cube_state"][:3]),
            "learning_corridor_error": tuple(self._corridor_error(after)),
            "action": tuple(values),
            "requested_target_preclamp": requested_target_preclamp,
            "persistent_target_postclamp": persistent_target_postclamp,
            "workspace_lower": tuple(self.config.workspace.lower),
            "workspace_upper": tuple(self.config.workspace.upper),
            "workspace_trigger_axes": tuple(
                "xyz"[index] for index, (value, low, high) in enumerate(zip(
                    requested_target_preclamp, self.config.workspace.lower,
                    self.config.workspace.upper)) if value < low or value > high),
            "workspace_overshoot_mm": {
                "xyz"[index]: 1000.0 * (low - value if value < low else value - high)
                for index, (value, low, high) in enumerate(zip(
                    requested_target_preclamp, self.config.workspace.lower,
                    self.config.workspace.upper)) if value < low or value > high},
            "left_finger_force": float(forces.get("8_Link", 0.0)),
            "right_finger_force": float(forces.get("7_Link", 0.0)),
            "cube_force": float(forces.get("cube", 0.0)),
            "stage_target": tuple(self._stage_target(terminal_snapshot)),
            "jaw_actual": tuple(terminal_snapshot["q"][6:8]),
            "jaw_command": float(command.jaw_target),
            "corridor_error": tuple(self._corridor_error(terminal_snapshot)),
        }
        self._last_info = info
        return self._observation(terminal_snapshot), reward, done, info

    def close(self):
        if not self._closed:
            self._closed = True
            self.scene.close()


def values_for_evidence(snapshot):
    return snapshot["q"] + snapshot["dq"] + snapshot["ee_state"] + snapshot["cube_state"]
