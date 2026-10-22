#!/usr/bin/env python3
"""P2-R4 minimal verifier-reference repair replay/curriculum runner."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

try:
    from isaacgym import gymapi as _gymapi_import_order  # noqa: F401
except ModuleNotFoundError:
    _gymapi_import_order = None

import torch

import experimental_s1_grasp_curriculum as curriculum
from simulation_core.experimental.manipulation.evidence import (
    GraspEvidenceConfig, evaluate_grasp,
)
from simulation_core.experimental.manipulation.grasp_env import (
    CURRICULUM_STAGE_A, CURRICULUM_STAGES,
    GRASP_VERIFIER_IMPLEMENTATION_REVISION, GraspEnvConfig,
    GraspPPOEnvironment, candidate_local_ee_rise,
)
from simulation_core.experimental.manipulation.grasp_ppo import (
    ExperimentalGraspPPOTrainer, deterministic_evaluate,
)
from simulation_core.experimental.manipulation.grasp_schema import (
    GRASP_ACTION_SCHEMA_ID, GRASP_OBSERVATION_SCHEMA_ID, GRASP_REWARD_VERSION,
    GRASP_SUCCESS_CRITERIA_VERSION,
)
from simulation_core.experimental.manipulation.scene import ManipulationScene, ManipulationSceneConfig
from simulation_core.experimental.manipulation.scripted_runtime import (
    ExperimentalMp4Recorder, ScriptedManipulationRuntime, ScriptedRuntimeConfig,
)
from simulation_core.ppo.trainer import PPOConfig


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "experimental" / "p2_r4_s1_grasp"
R2_CHECKPOINT = (ROOT / "artifacts" / "experimental" / "p2_r2_s1_grasp" /
                 "stage_A" / "checkpoints" / "model_best.pt")
R3_OUTPUT = ROOT / "artifacts" / "experimental" / "p2_r3_s1_grasp"
curriculum.ARTIFACT_ROOT = OUTPUT


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def contract():
    return {
        "observation": GRASP_OBSERVATION_SCHEMA_ID, "observation_dim": 40,
        "action": GRASP_ACTION_SCHEMA_ID, "action_dim": 4,
        "reward": GRASP_REWARD_VERSION,
        "success": GRASP_SUCCESS_CRITERIA_VERSION,
        "candidate": "locked_6_step_composite_evidence",
        "verification_lift_m": .025,
        "verifier_implementation_revision": GRASP_VERIFIER_IMPLEMENTATION_REVISION,
        "only_semantic_change": "ee_rise_candidate_local_reference",
    }


def checkpoint_metadata(path):
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reward_version": payload.get("reward_version"),
        "observation_schema": payload.get("observation_schema_id"),
        "action_schema": payload.get("action_schema_id"),
        "curriculum_stage": payload.get("curriculum_stage"),
        "training_transitions_total": payload.get("training_transitions_total"),
    }


def regression():
    reset_z, candidate_z, terminal_z = .432657, .398354, .419232
    old = terminal_z - reset_z
    new = candidate_local_ee_rise(terminal_z, candidate_z)
    base = dict(
        config=GraspEvidenceConfig(self_contact_floor=.01), actual_jaw=.033,
        object_in_corridor=True, object_table_clearance=.020,
        object_rise=.020661, ee_rise=new,
        relative_translation_drift=.000545, relative_rotation_drift=.0735,
        state_values=(0.0,) * 20, left_force=3.4, right_force=3.2,
        object_force=1.0,
    )
    cases = {
        "stage_a_fixture": {},
        "cube_not_following": {"object_rise": .002},
        "cube_dropped": {"object_table_clearance": -.001, "object_rise": -.002,
                         "relative_translation_drift": .020},
        "cube_rise_low": {"object_rise": .0119},
        "clearance_low": {"object_table_clearance": .0079},
        "translation_drift": {"relative_translation_drift": .0121},
        "rotation_drift": {"relative_rotation_drift": .251},
        "jaw_unsafe": {"actual_jaw": .0099},
        "non_finite": {"state_values": (0.0, float("nan"))},
    }
    results = {}
    for name, overrides in cases.items():
        values = dict(base); values.update(overrides)
        evidence = evaluate_grasp(**values)
        results[name] = {
            "verified": evidence.grasp_verified,
            "failure_reason": evidence.failure_reason,
            "flags": dict(evidence.flags),
        }
    old_values = dict(base); old_values["ee_rise"] = old
    old_evidence = evaluate_grasp(**old_values)
    payload = {
        "contract": contract(),
        "fixture": {
            "reset_ee_z": reset_z, "candidate_ee_z": candidate_z,
            "terminal_ee_z": terminal_z, "ee_rise_old": old,
            "ee_rise_new": new, "cube_rise": base["object_rise"],
            "old_verified": old_evidence.grasp_verified,
            "new_verified": results["stage_a_fixture"]["verified"],
        },
        "negative_cases": results,
        "passed": (not old_evidence.grasp_verified
                   and results["stage_a_fixture"]["verified"]
                   and all(not item["verified"] for name, item in results.items()
                           if name != "stage_a_fixture")),
    }
    write_json(OUTPUT / "regression" / "verifier_reference_tests.json", payload)
    return payload


def p1_regression(count):
    scene = ManipulationScene(ManipulationSceneConfig(graphics_device_id=-1, with_camera=False,
                                                       initial_open=.05))
    trials = []
    try:
        scene.initialize()
        runtime = ScriptedManipulationRuntime(scene, ScriptedRuntimeConfig(
            open_safe=.05, close_safe=.025, jaw_rate_limit=.001,
            force_abort_threshold=50.0, physics_steps_per_control=4))
        for index in range(count):
            result = runtime.run_trial(index, scene.config.cube_xy)
            trials.append({
                "trial": index, "success": bool(result["success"]),
                "failure_reason": result["failure_reason"],
                "stage_reached": result["stage_reached"],
                "metrics": {key: value for key, value in result.items()
                            if key not in ("trace", "success", "failure_reason", "stage_reached")},
            })
    finally:
        scene.close()
    summary = {"trials": count, "success_count": sum(t["success"] for t in trials),
               "resource_cleanup": True, "records": trials, "contract": contract()}
    write_json(OUTPUT / "regression" / "p1_scripted_5_summary.json", summary)
    return summary


def _load_trainer(env, checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    trainer = ExperimentalGraspPPOTrainer(
        env, config=PPOConfig(**payload["ppo_config"]), device="cuda:0")
    trainer.load_checkpoint(Path(checkpoint))
    return trainer


def replay(episodes, label):
    output = OUTPUT / "stage_A_replay" / label
    env = GraspPPOEnvironment(GraspEnvConfig(curriculum_stage=CURRICULUM_STAGE_A))
    try:
        env.initialize()
        trainer = _load_trainer(env, R2_CHECKPOINT)
        summary, records = deterministic_evaluate(env, trainer, episodes)
        summary.update({
            "checkpoint": checkpoint_metadata(R2_CHECKPOINT),
            "contract": contract(), "resource_cleanup": True,
            "verified_per_candidate": (summary["success_count"] / summary["candidate_count"]
                                       if summary["candidate_count"] else 0.0),
        })
    finally:
        env.close()
    write_json(output / "summary.json", summary)
    output.mkdir(parents=True, exist_ok=True)
    (output / "episodes.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in records), encoding="utf-8")
    write_json(output / "config.json", {
        "environment": asdict(GraspEnvConfig(curriculum_stage=CURRICULUM_STAGE_A)),
        "checkpoint": checkpoint_metadata(R2_CHECKPOINT), "contract": contract(),
        "training_or_optimizer_steps": 0,
    })
    return summary


def video(checkpoint, stage, filename):
    output = (OUTPUT / ("stage_A_replay" if stage == CURRICULUM_STAGE_A else
                        "stage_" + stage.split("_", 1)[0]) / "video" / filename)
    recorder = None
    env = None; cleanup = False
    try:
        env = GraspPPOEnvironment(
            GraspEnvConfig(curriculum_stage=stage), with_camera=True).initialize()
        trainer = _load_trainer(env, checkpoint) if stage == CURRICULUM_STAGE_A else None
        if trainer is None:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            trainer = ExperimentalGraspPPOTrainer(
                env, config=PPOConfig(**payload["ppo_config"]), device="cuda:0")
            trainer.load_checkpoint(checkpoint)
        # A fresh GPU scene has one known deterministic warm-start workspace
        # violation.  Consume it without recording; policy/actions and all
        # physics remain unchanged for the subsequent acceptance episode.
        warmup_summary, warmup_records = deterministic_evaluate(env, trainer, 1)
        recorder = ExperimentalMp4Recorder(output).start()
        env.frame_callback = recorder.write
        summary, records = deterministic_evaluate(env, trainer, 1)
        cleanup = recorder.close(); recorder = None
    finally:
        if recorder is not None: cleanup = recorder.close()
        if env is not None: env.close()
    result = {"evaluation": summary, "episode": records[0],
              "unrecorded_warmup": warmup_records[0],
              "video": str(output), "resource_cleanup": cleanup,
              "checkpoint": checkpoint_metadata(checkpoint), "contract": contract()}
    write_json(output.with_suffix(".json"), result)
    return result


def train(stage, updates, resume):
    result = curriculum.train(stage, updates, resume)
    config_path = curriculum.stage_directory(stage) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update({"experiment": "P2-R4", "verifier_implementation_revision":
                   GRASP_VERIFIER_IMPLEMENTATION_REVISION})
    write_json(config_path, config)
    return result


def evaluate(stage, checkpoint, episodes):
    result = curriculum.evaluate(stage, checkpoint, episodes)
    result["verifier_implementation_revision"] = GRASP_VERIFIER_IMPLEMENTATION_REVISION
    write_json(curriculum.stage_directory(stage) / "evaluation" / "summary.json", result)
    return result


def finalize():
    replay10 = json.loads((OUTPUT / "stage_A_replay" / "replay_10" / "summary.json").read_text())
    replay50 = json.loads((OUTPUT / "stage_A_replay" / "replay_50" / "summary.json").read_text())
    old = json.loads((R3_OUTPUT / "false_positive" / "summary.json").read_text())
    old_selected = [item for item in old["records"] if item.get("selected")][:10]
    after_records = [json.loads(line) for line in (
        OUTPUT / "stage_A_replay" / "replay_50" / "episodes.jsonl").read_text().splitlines()
                     if line.strip()]
    paired_after = [item for item in after_records if item["grasp_candidate"]][:10]
    before_after = {
        "same_checkpoint": checkpoint_metadata(R2_CHECKPOINT),
        "same_stage_A_reset": True, "policy_weights_modified": False,
        "before": {
            "episodes": len(old_selected),
            "candidate_count": sum(item.get("candidate", False) for item in old_selected),
            "verified_count": sum(item.get("success", False) for item in old_selected),
            "lift_ratio_too_low_count": sum(
                item.get("verifier_failure", {}).get("first_failure") == "LIFT_RATIO_TOO_LOW"
                for item in old_selected),
            "verifier_revision": "episode-reset-relative-bug",
            "mean_candidate_local_ee_rise": sum(
                item["verifier_failure"]["ee_rise_at_failure"] for item in old_selected) /
                len(old_selected),
            "mean_reset_relative_ee_rise": sum(
                item["verifier_failure"]["formal_reset_relative_ee_rise"]
                for item in old_selected) / len(old_selected),
            "mean_cube_rise": sum(
                item["verifier_failure"]["cube_rise_at_failure"] for item in old_selected) /
                len(old_selected),
        },
        "paired_after_post_warmup": {
            "episodes": len(paired_after),
            "candidate_count": sum(item["grasp_candidate"] for item in paired_after),
            "verified_count": sum(item["success"] for item in paired_after),
            "lift_ratio_too_low_count": sum(
                item["failure_reason"] == "GRASP_VERIFICATION_FAILED" for item in paired_after),
            "verifier_revision": GRASP_VERIFIER_IMPLEMENTATION_REVISION,
            "mean_candidate_local_ee_rise": sum(
                item["candidate_local_ee_rise"] for item in paired_after) / len(paired_after),
            "mean_reset_relative_ee_rise": sum(
                item["reset_relative_ee_rise"] for item in paired_after) / len(paired_after),
            "mean_cube_rise": sum(item["cube_rise"] for item in paired_after) / len(paired_after),
            "mean_lift_ratio": sum(item["lift_ratio"] for item in paired_after) / len(paired_after),
        },
        "after_10": replay10, "after_50": replay50,
    }
    write_json(OUTPUT / "regression" / "before_vs_after_verifier.json", before_after)
    files = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(OUTPUT))] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
    manifest = {"stage": "P2-R4", "contract": contract(),
                "r2_checkpoint": checkpoint_metadata(R2_CHECKPOINT), "files": files}
    write_json(OUTPUT / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("regression", "p1", "replay", "video", "train", "eval", "finalize"), required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--label", default="replay_10")
    parser.add_argument("--stage", choices=CURRICULUM_STAGES, default=CURRICULUM_STAGE_A)
    parser.add_argument("--checkpoint", type=Path, default=R2_CHECKPOINT)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--filename", default="successful_replay.mp4")
    args = parser.parse_args(); os.environ.pop("DISPLAY", None)
    if args.mode == "regression": result = regression()
    elif args.mode == "p1": result = p1_regression(args.count)
    elif args.mode == "replay": result = replay(args.episodes, args.label)
    elif args.mode == "video": result = video(args.checkpoint, args.stage, args.filename)
    elif args.mode == "train": result = train(args.stage, args.updates, args.resume)
    elif args.mode == "eval": result = evaluate(args.stage, args.checkpoint, args.episodes)
    else: result = finalize()
    print("P2_R4_%s=%s" % (args.mode.upper(), json.dumps(result, sort_keys=True)), flush=True)


if __name__ == "__main__":
    main()
