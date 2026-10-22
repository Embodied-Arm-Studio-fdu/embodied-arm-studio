#!/usr/bin/env python3
"""Record a post-gate P2 grasp success plus display-only lift and hold."""

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

try:
    from isaacgym import gymapi as _gymapi_import_order  # noqa: F401
except ModuleNotFoundError:
    _gymapi_import_order = None

import cv2
import torch

from simulation_core.experimental.manipulation.grasp_env import (
    CURRICULUM_STAGE_C, GraspEnvConfig, GraspPPOEnvironment,
)
from simulation_core.experimental.manipulation.grasp_ppo import ExperimentalGraspPPOTrainer
from simulation_core.experimental.manipulation.grasp_schema import GRASP_POLICY_SCHEMA
from simulation_core.experimental.manipulation.scripted_runtime import ExperimentalMp4Recorder
from simulation_core.ppo.policy import flatten_observation
from simulation_core.ppo.trainer import PPOConfig


ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIRECTORY = ROOT / "artifacts" / "experimental" / "p2_r8_s1_grasp" / "video"
VIDEO_PATH = VIDEO_DIRECTORY / "grasp_success_visualization.mp4"
METADATA_PATH = VIDEO_DIRECTORY / "grasp_success_visualization.json"
CHECKPOINT = (ROOT / "artifacts" / "experimental" / "p2_r6_s1_grasp" /
              "random_xy" / "checkpoints" / "model_20000.pt")
FORMAL_SUMMARY = (ROOT / "artifacts" / "experimental" / "p2_r8_s1_grasp" /
                  "randomized_100" / "summary.json")
FORMAL_EPISODES = (ROOT / "artifacts" / "experimental" / "p2_r8_s1_grasp" /
                   "randomized_100" / "episodes.json")
FPS = 15
EXTRA_LIFT_M = .030
HOLD_SECONDS = 2.0
LIFT_CONTROL_STEPS = 30
HOLD_CONTROL_STEPS = int(HOLD_SECONDS * FPS)
MAX_ATTEMPTS = 10


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def norm(values):
    return math.sqrt(sum(float(value) ** 2 for value in values))


def finite_snapshot(snapshot):
    values = snapshot["q"] + snapshot["dq"] + snapshot["ee_state"] + snapshot["cube_state"]
    return all(math.isfinite(float(value)) for value in values)


def cube_clearance(env, snapshot):
    return (float(snapshot["cube_state"][2]) - env.scene.config.cube_size / 2.0
            - env.scene.config.table_top)


class VerifiedOverlay:
    def __init__(self, recorder):
        self.recorder = recorder
        self.verified = False
        self.clearance_m = 0.0

    def write(self, frame):
        if self.verified:
            image = frame.data
            first = "GRASP VERIFIED"
            second = "cube clearance: %.1f mm" % (self.clearance_m * 1000.0)
            cv2.putText(image, first, (18, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        .72, (0, 0, 0, 255), 4, cv2.LINE_AA)
            cv2.putText(image, first, (18, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        .72, (70, 255, 90, 255), 2, cv2.LINE_AA)
            cv2.putText(image, second, (18, 58), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, (0, 0, 0, 255), 4, cv2.LINE_AA)
            cv2.putText(image, second, (18, 58), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, (255, 255, 255, 255), 2, cv2.LINE_AA)
        self.recorder.write(frame)


def make_env():
    return GraspPPOEnvironment(
        GraspEnvConfig(curriculum_stage=CURRICULUM_STAGE_C, randomize_xy=.006),
        with_camera=True,
    )


def load_trainer(env):
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    trainer = ExperimentalGraspPPOTrainer(
        env, config=PPOConfig(**payload["ppo_config"]), device="cuda:0")
    trainer.load_checkpoint(CHECKPOINT)
    trainer.policy.eval()
    return trainer


def deterministic_action(observation, trainer):
    raw = flatten_observation(observation, GRASP_POLICY_SCHEMA).to(trainer.device)
    normalized = trainer.normalizer.normalize(raw, update=False)
    with torch.no_grad():
        action, _, _ = trainer.policy.act(normalized, deterministic=True)
    return tuple(float(value) for value in action.detach().cpu())


def run_formal_episode(env, trainer):
    observation = env.reset()
    cube_xy = tuple(float(value) for value in env._initial_cube[:2])
    done = False
    steps = 0
    info = {}
    while not done:
        observation, _, done, info = env.step(deterministic_action(observation, trainer))
        steps += 1
        if steps > env.config.learning_horizon + 1:
            raise RuntimeError("formal episode exceeded environment horizon")
    return cube_xy, steps, dict(info)


def validate_display_step(env, snapshot, command):
    if not finite_snapshot(snapshot):
        return "NON_FINITE"
    if command.workspace_clamped:
        return "WORKSPACE_VIOLATION"
    if command.joint_limit_clamped:
        return "JOINT_LIMIT"
    abort = env._abort_reason(snapshot, command, before_candidate=False)
    if abort:
        return abort
    if cube_clearance(env, snapshot) <= .005:
        return "DROP"
    return ""


def run_display_motion(env, overlay, jaw_target):
    start_snapshot = env.scene.snapshot()
    start_target = tuple(float(value) for value in env._target)
    final_target = (start_target[0], start_target[1], start_target[2] + EXTRA_LIFT_M)
    lower, upper = env.config.workspace.lower, env.config.workspace.upper
    if not all(lower[index] <= final_target[index] <= upper[index] for index in range(3)):
        return {"passed": False, "failure_reason": "WORKSPACE_VIOLATION"}

    start_cube_z = float(start_snapshot["cube_state"][2])
    start_ee_z = float(start_snapshot["ee_state"][2])
    minimum_clearance = cube_clearance(env, start_snapshot)
    overlay.verified = True
    overlay.clearance_m = minimum_clearance
    start_frame = overlay.recorder.frames
    start_wall_time = now_iso()
    last_snapshot = start_snapshot
    last_command = None

    for step in range(1, LIFT_CONTROL_STEPS + 1):
        alpha = step / LIFT_CONTROL_STEPS
        target = tuple(start_target[index] + alpha * (final_target[index] - start_target[index])
                       for index in range(3))
        state = env.scene.robot.read_state()
        command = env.compositor.compose(
            target, env.config.fixed_orientation_xyzw, jaw_target, state)
        env.scene.robot.set_joint_position_targets(command.positions)
        env.scene.robot.step(env.config.physics_steps_per_action)
        snapshot = env.scene.snapshot()
        failure = validate_display_step(env, snapshot, command)
        if failure:
            return {"passed": False, "failure_reason": failure}
        minimum_clearance = min(minimum_clearance, cube_clearance(env, snapshot))
        overlay.clearance_m = cube_clearance(env, snapshot)
        env._capture()
        last_snapshot, last_command = snapshot, command

    lift_end_frame = overlay.recorder.frames
    for _ in range(HOLD_CONTROL_STEPS):
        state = env.scene.robot.read_state()
        command = env.compositor.compose(
            final_target, env.config.fixed_orientation_xyzw, jaw_target, state)
        env.scene.robot.set_joint_position_targets(command.positions)
        env.scene.robot.step(env.config.physics_steps_per_action)
        snapshot = env.scene.snapshot()
        failure = validate_display_step(env, snapshot, command)
        if failure:
            return {"passed": False, "failure_reason": failure}
        minimum_clearance = min(minimum_clearance, cube_clearance(env, snapshot))
        overlay.clearance_m = cube_clearance(env, snapshot)
        env._capture()
        last_snapshot, last_command = snapshot, command

    final_clearance = cube_clearance(env, last_snapshot)
    cube_rise = float(last_snapshot["cube_state"][2]) - start_cube_z
    ee_rise = float(last_snapshot["ee_state"][2]) - start_ee_z
    final_cube_speed = norm(last_snapshot["cube_state"][7:10])
    jaw_command_unchanged = abs(float(last_command.jaw_target) - jaw_target) < 1.0e-12
    passed = (cube_rise >= .020 and final_clearance > .010
              and minimum_clearance > .005 and final_cube_speed < .10
              and jaw_command_unchanged)
    return {
        "passed": passed,
        "failure_reason": "" if passed else "DISPLAY_HOLD_VALIDATION_FAILED",
        "start_target_xyz": start_target,
        "final_target_xyz": final_target,
        "target_lift_distance_m": EXTRA_LIFT_M,
        "actual_ee_rise_m": ee_rise,
        "actual_cube_rise_m": cube_rise,
        "minimum_cube_clearance_m": minimum_clearance,
        "final_cube_clearance_m": final_clearance,
        "final_cube_linear_speed_mps": final_cube_speed,
        "jaw_target_m": jaw_target,
        "jaw_command_unchanged": jaw_command_unchanged,
        "lift_control_steps": LIFT_CONTROL_STEPS,
        "hold_control_steps": HOLD_CONTROL_STEPS,
        "hold_duration_s": HOLD_SECONDS,
        "demonstration_start_frame": start_frame,
        "demonstration_lift_end_frame": lift_end_frame,
        "demonstration_start_video_time_s": start_frame / FPS,
        "demonstration_lift_end_video_time_s": lift_end_frame / FPS,
        "demonstration_start_wall_time": start_wall_time,
    }


def main():
    os.environ.pop("DISPLAY", None)
    checkpoint_sha_before = sha256(CHECKPOINT)
    formal_hashes_before = {
        "randomized_100_summary_sha256": sha256(FORMAL_SUMMARY),
        "randomized_100_episodes_sha256": sha256(FORMAL_EPISODES),
    }
    formal_summary = json.loads(FORMAL_SUMMARY.read_text(encoding="utf-8"))
    if formal_summary.get("success_count") != 98 or formal_summary.get("episodes") != 100:
        raise RuntimeError("P2-R8 formal evaluation baseline is not the accepted 98/100 result")

    VIDEO_DIRECTORY.mkdir(parents=True, exist_ok=True)
    env = make_env()
    attempts = []
    accepted = None
    try:
        env.initialize()
        trainer = load_trainer(env)
        for attempt_index in range(1, MAX_ATTEMPTS + 1):
            temporary_video = VIDEO_DIRECTORY / (".grasp_visualization_attempt_%02d.mp4" % attempt_index)
            recorder = ExperimentalMp4Recorder(temporary_video, fps=FPS).start()
            overlay = VerifiedOverlay(recorder)
            env.frame_callback = overlay.write
            attempt_started = time.monotonic()
            try:
                cube_xy, formal_steps, info = run_formal_episode(env, trainer)
                formal_end_frame = recorder.frames
                formal_end_wall_time = now_iso()
                formal_valid = bool(info.get("success") and info.get("verified_success"))
                attempt = {
                    "attempt": attempt_index,
                    "cube_reset_xy": cube_xy,
                    "formal_success": bool(info.get("success")),
                    "verified": bool(info.get("verified_success")),
                    "formal_failure_reason": str(info.get("failure_reason", "")),
                    "formal_learning_steps": formal_steps,
                    "stabilizing_entered": bool(info.get("stabilizing_entered_once")),
                    "stabilizing_completed": bool(info.get("stabilizing_completed_once")),
                    "formal_verification_end_frame": formal_end_frame,
                    "formal_verification_end_video_time_s": formal_end_frame / FPS,
                    "formal_verification_end_wall_time": formal_end_wall_time,
                }
                if formal_valid:
                    display = run_display_motion(env, overlay, float(info["jaw_command"]))
                    attempt["display"] = display
                else:
                    attempt["display"] = {"passed": False, "failure_reason": "FORMAL_FAILURE"}
                env.frame_callback = None
                encoded = recorder.close()
                attempt["encoded"] = encoded
                attempt["elapsed_wall_seconds"] = time.monotonic() - attempt_started
                attempts.append(attempt)
                if formal_valid and display["passed"] and encoded:
                    temporary_video.replace(VIDEO_PATH)
                    accepted = attempt
                    break
                if temporary_video.is_file():
                    temporary_video.unlink()
            except Exception:
                env.frame_callback = None
                recorder.close()
                if temporary_video.is_file():
                    temporary_video.unlink()
                raise
    finally:
        env.close()

    if accepted is None:
        raise RuntimeError("no safe verified visualization obtained in %d attempts" % MAX_ATTEMPTS)

    checkpoint_sha_after = sha256(CHECKPOINT)
    formal_hashes_after = {
        "randomized_100_summary_sha256": sha256(FORMAL_SUMMARY),
        "randomized_100_episodes_sha256": sha256(FORMAL_EPISODES),
    }
    if checkpoint_sha_after != checkpoint_sha_before:
        raise RuntimeError("policy checkpoint changed during visualization")
    if formal_hashes_after != formal_hashes_before:
        raise RuntimeError("formal P2-R8 evaluation artifacts changed during visualization")

    display = accepted["display"]
    metadata = {
        "video": str(VIDEO_PATH),
        "video_sha256": sha256(VIDEO_PATH),
        "video_bytes": VIDEO_PATH.stat().st_size,
        "video_width": 640,
        "video_height": 480,
        "video_fps": FPS,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": checkpoint_sha_after,
        "checkpoint_unchanged": checkpoint_sha_after == checkpoint_sha_before,
        "training_performed": False,
        "training_transitions": 0,
        "randomized_reset_xy_radius_m": .006,
        "cube_reset_xy": accepted["cube_reset_xy"],
        "formal_success": True,
        "verified": True,
        "formal_verification_end_time": accepted["formal_verification_end_wall_time"],
        "formal_verification_end_video_time_s": accepted["formal_verification_end_video_time_s"],
        "demonstration_lift_start_time": display["demonstration_start_wall_time"],
        "demonstration_lift_start_video_time_s": display["demonstration_start_video_time_s"],
        "extra_lift_distance_m": EXTRA_LIFT_M,
        "actual_cube_rise_during_demonstration_m": display["actual_cube_rise_m"],
        "actual_ee_rise_during_demonstration_m": display["actual_ee_rise_m"],
        "hold_duration_s": HOLD_SECONDS,
        "jaw_command_unchanged": display["jaw_command_unchanged"],
        "final_cube_clearance_m": display["final_cube_clearance_m"],
        "demonstration_motion_after_success": True,
        "demonstration_motion_used_for_success": False,
        "formal_p2_r8_result": "98/100",
        "formal_p2_r8_artifacts_unchanged": True,
        "formal_artifact_hashes_before": formal_hashes_before,
        "formal_artifact_hashes_after": formal_hashes_after,
        "attempts": attempts,
    }
    write_json(METADATA_PATH, metadata)
    print("POLICY_CHECKPOINT_UNMODIFIED=true")
    print("TRAINING_PERFORMED=false")
    print("FORMAL_GRASP_SUCCESS_BEFORE_DEMONSTRATION=true")
    print("DEMONSTRATION_LIFT_HOLD_ONLY=true")
    print("P2_R8_FORMAL_98_OF_100_UNMODIFIED=true")
    print("GRASP_SUCCESS_VISUALIZATION=%s" % json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
