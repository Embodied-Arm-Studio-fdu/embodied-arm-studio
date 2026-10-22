#!/usr/bin/env python3
"""Read-only P3 Final 100-episode evaluation; no training entry point."""

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

try:
    from isaacgym import gymapi as _gymapi_import_order  # noqa: F401
except ModuleNotFoundError:
    _gymapi_import_order = None

import cv2

from experimental_s1_lift_stable_hold import (
    P3_1_CHECKPOINT, P3_1_SHA256, contract, environment_config, make_trainer,
)
from simulation_core.experimental.manipulation.lift_env import LiftPPOEnvironment
from simulation_core.experimental.manipulation.lift_ppo import deterministic_evaluate
from simulation_core.experimental.manipulation.scripted_runtime import ExperimentalMp4Recorder
from simulation_core.task_catalog import task_catalog_document


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "experimental" / "p3_final_s1_lift"
CATALOG_DIGEST = "sha256:c0f3feb0771f4ab636c61fa53ba0d49d5f2a489a224b6890ae38ed798423f791"


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def checkpoint_lock():
    actual = hashlib.sha256(P3_1_CHECKPOINT.read_bytes()).hexdigest()
    if actual != P3_1_SHA256:
        raise RuntimeError("locked P3-1 checkpoint SHA-256 mismatch")
    return {"path": str(P3_1_CHECKPOINT), "expected_sha256": P3_1_SHA256,
            "actual_sha256": actual, "passed": True}


def truth_audit(records, config):
    violations = []
    for row in records:
        if not row["success"]:
            continue
        checks = {
            "cube_in_band": config.target_band[0] <= row["cube_rise"] <= config.target_band[1],
            "hold_5s": row["hold_duration"] >= 5.0,
            "translation_drift": row["relative_translation_drift"] <= .012,
            "rotation_drift": row["relative_rotation_drift"] <= .25,
            "stable_grasp": bool(row["stable_grasp_valid"]),
            "jaw_safe": bool(row["jaw_safe"]),
            "no_drop": not row["drop"], "finite": not row["non_finite"],
            "safe": not row["unsafe"],
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed: violations.append({"episode": row["episode"], "failed": failed})
    return {"successful_episodes_checked": sum(x["success"] for x in records),
            "violation_count": len(violations), "violations": violations,
            "passed": not violations}


def evaluate(episodes=100):
    if episodes != 100:
        raise ValueError("P3 Final requires exactly 100 episodes")
    lock = checkpoint_lock(); config = environment_config()
    env = LiftPPOEnvironment(config); cleanup = False
    try:
        env.initialize(); trainer = make_trainer(env, P3_1_CHECKPOINT)
        summary, records = deterministic_evaluate(env, trainer, episodes)
        audit = truth_audit(records, config); cleanup = True
    finally:
        env.close()
    gate = (summary["success_rate"] >= .90 and summary["wilson_95_lower"] >= .80
            and summary["unsafe_count"] == 0 and summary["non_finite_count"] == 0
            and audit["passed"])
    summary.update({"gate": "PASS" if gate else "FAIL", "checkpoint_lock": lock,
                    "contract": contract(), "formal_success_truth_audit": audit,
                    "training_performed": False, "new_checkpoint_generated": False,
                    "resource_cleanup": cleanup})
    output = OUTPUT / "evaluation_100"
    write_json(output / "summary.json", summary)
    for row in records: append_jsonl(output / "episodes.jsonl", row)
    return summary


class FinalRecorder:
    def __init__(self, recorder, env_getter):
        self.recorder, self.env_getter = recorder, env_getter
    def write(self, frame):
        pixels = frame.data.copy(); bootstrap = self.env_getter().phase == "GRASP_BOOTSTRAP"
        label = "STABLE GRASP RESET" if bootstrap else "LIFT POLICY START / 5S HOLD"
        color = (255, 220, 30, 255) if bootstrap else (50, 255, 70, 255)
        cv2.rectangle(pixels, (12, 12), (510, 54), (0, 0, 0, 255), -1)
        cv2.putText(pixels, label, (22, 42), cv2.FONT_HERSHEY_SIMPLEX,
                    .68, color, 2, cv2.LINE_AA)
        self.recorder.write(replace(frame, data=pixels))


def video():
    checkpoint_lock(); path = OUTPUT / "video" / "formal_success_5s_hold.mp4"
    recorder = ExperimentalMp4Recorder(path).start(); holder = {}; env = None; cleanup = False
    labeled = FinalRecorder(recorder, lambda: holder["env"])
    try:
        env = LiftPPOEnvironment(environment_config(), with_camera=True,
                                 frame_callback=labeled.write)
        holder["env"] = env; env.initialize(); trainer = make_trainer(env, P3_1_CHECKPOINT)
        summary, records = deterministic_evaluate(env, trainer, 1)
        audit = truth_audit(records, environment_config())
        cleanup = recorder.close(); recorder = None
    finally:
        if recorder is not None: cleanup = recorder.close()
        if env is not None: env.close()
    result = {"passed": summary["success_count"] == 1 and audit["passed"] and cleanup,
              "evaluation": summary, "episode": records[0], "truth_audit": audit,
              "video": str(path), "resource_cleanup": cleanup,
              "labels": ["STABLE GRASP RESET", "LIFT POLICY START / 5S HOLD"],
              "checkpoint_sha256": P3_1_SHA256}
    write_json(path.with_suffix(".json"), result); return result


def finalize():
    summary = json.loads((OUTPUT / "evaluation_100" / "summary.json").read_text())
    catalog = task_catalog_document()
    isolation = {"catalog_version": catalog["catalog_version"],
                 "catalog_digest": catalog["catalog_digest"],
                 "production_task_count": len(catalog["tasks"]),
                 "experimental_registered": [x["id"] for x in catalog["tasks"]
                                             if x["id"] in ("s1_grasp", "s1_lift", "s1_place")]}
    isolation["passed"] = (isolation["catalog_version"] == "task-catalog/v1"
                           and isolation["catalog_digest"] == CATALOG_DIGEST
                           and isolation["production_task_count"] == 7
                           and not isolation["experimental_registered"])
    write_json(OUTPUT / "production_isolation.json", isolation)
    files = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(OUTPUT))] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size}
    gate = summary["gate"] == "PASS" and isolation["passed"]
    result = {"stage": "P3 Final", "gate": "PASS" if gate else "FAIL",
              "p3_s1_lift_ppo_gate": "PASS" if gate else "FAIL",
              "evaluation": summary, "contract": contract(), "isolation": isolation,
              "training_performed": False, "new_checkpoint_generated": False,
              "files": files}
    write_json(OUTPUT / "manifest.json", result); return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("evaluate", "video", "finalize"), required=True)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args(); os.environ.pop("DISPLAY", None)
    if args.mode == "evaluate": result = evaluate(args.episodes)
    elif args.mode == "video": result = video()
    else: result = finalize()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
