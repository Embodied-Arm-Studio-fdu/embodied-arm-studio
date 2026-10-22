#!/usr/bin/env python3
"""P4-R3 policy-triggered release curriculum and Full Place gate."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

try:
    from isaacgym import gymapi as _gymapi_import_order  # noqa: F401
except ModuleNotFoundError:
    _gymapi_import_order = None

import cv2
import numpy as np
import torch

from simulation_core.experimental.manipulation.lower_release_env import (
    LOWER_RELEASE_TARGET, LowerReleaseEnvironment,
)
from simulation_core.experimental.manipulation.lower_release_ppo import deterministic_evaluate
from simulation_core.experimental.manipulation.place_completion import (
    FULL_PLACE_ACTIVE_ACTION_MASK, P4_2_20K_SHA256,
    RELEASE_ONLY_ACTIVE_ACTION_MASK, FullPlaceEnvironment,
    ReleaseOnlyEnvironment, StagePlaceTrainer,
)
from simulation_core.experimental.manipulation.lower_release_schema import (
    LOWER_RELEASE_ACTIVE_ACTION_MASK,
)
from simulation_core.experimental.manipulation.scripted_runtime import ExperimentalMp4Recorder
from simulation_core.ppo.trainer import PPOConfig
from simulation_core.task_catalog import task_catalog_document


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "experimental" / "p4_r3_s1_place"
SOURCE = (ROOT / "artifacts" / "experimental" / "p4_2_s1_place" /
          "train_20k" / "checkpoints" / "model_latest.pt")
RELEASE_SEMANTICS = "policy_trigger_latched_open"


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def seed_everything(seed=20261003):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def ppo_config():
    return PPOConfig(rollout_steps=200, update_epochs=2, batch_size=32,
                     learning_rate=3e-4, gamma=.99, gae_lambda=.95,
                     clip_ratio=.2, value_coefficient=.5,
                     entropy_coefficient=.01, max_grad_norm=.5)


STAGES = {
    "stage_a": (ReleaseOnlyEnvironment, RELEASE_ONLY_ACTIVE_ACTION_MASK, "release_only"),
    "stage_b": (LowerReleaseEnvironment, LOWER_RELEASE_ACTIVE_ACTION_MASK, "lower_release"),
    "stage_c": (FullPlaceEnvironment, FULL_PLACE_ACTIVE_ACTION_MASK, "full_place"),
}


def new_trainer(env, stage_key):
    _, mask, name = STAGES[stage_key]
    return StagePlaceTrainer(env, active_action_mask=mask, stage_name=name,
                             config=ppo_config(), device="cuda:0")


def checkpoint_info(path):
    path = Path(path); payload = torch.load(path, map_location="cpu", weights_only=False)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "task_id": payload.get("task_id"),
            "active_action_mask": list(payload.get("active_action_mask") or ()),
            "stage_name": payload.get("stage_name"),
            "release_semantics": payload.get("release_semantics"),
            "stage_training_transitions": payload.get("stage_training_transitions"),
            "source_checkpoint": payload.get("source_checkpoint")}


def initialize_stage(trainer, source, *, reset_release=False):
    return trainer.initialize_from_checkpoint(
        source,
        P4_2_20K_SHA256 if Path(source) == SOURCE else None,
        reinitialize_release_row=reset_release)


def materialize(stage_key, source, run_name, reset_release=False):
    env_cls, _, _ = STAGES[stage_key]; env = env_cls()
    try:
        env.initialize(); env.reset(); trainer = new_trainer(env, stage_key)
        initialize_stage(trainer, source, reset_release=reset_release)
        return trainer.save_checkpoint(OUTPUT.parent, OUTPUT.name, run_name)
    finally:
        env.close()


def train(stage_key, run_name, transitions, source, *, reset_release=False, resume=False):
    if transitions <= 0 or transitions % 200:
        raise ValueError("transitions must be a positive multiple of 200")
    seed_everything(); output = OUTPUT / run_name; output.mkdir(parents=True, exist_ok=True)
    env_cls, mask, stage_name = STAGES[stage_key]
    env = env_cls(); records = []; best_rank = None; start = time.time(); cleanup = False
    try:
        env.initialize(); env.reset(); trainer = new_trainer(env, stage_key)
        if resume:
            trainer.load_checkpoint(Path(source))
        else:
            initialize_stage(trainer, source, reset_release=reset_release)
        write_json(output / "config.json", {
            "task_id": "s1_place", "stage": stage_name,
            "active_action_mask": list(mask), "transitions_this_call": transitions,
            "release_semantics": RELEASE_SEMANTICS,
            "release_trigger": {"threshold": env.config.release_trigger_threshold,
                                "consecutive_control_steps": env.config.release_trigger_steps,
                                "legal_clearance_max_m": env.config.legal_release_clearance},
            "latched_jaw_target_m": env.config.open_safe,
            "jaw_rate_limit_m_per_control_step": env.config.jaw_rate_limit,
            "source": str(source), "ppo": asdict(ppo_config()),
        })
        for local in range(1, transitions // trainer.config.rollout_steps + 1):
            metrics = trainer.train_one_update()
            row = {"update_index": trainer.update_steps,
                   "training_steps": trainer.training_steps, **metrics}
            records.append(row); append_jsonl(output / "metrics.jsonl", row)
            for episode in trainer._rollout_episode_records:
                append_jsonl(output / "episodes.jsonl", episode)
            rank = (row.get("success_rate", 0.0), row.get("detached_rate", 0.0),
                    row.get("release_latched_rate", 0.0), row["average_reward"])
            if best_rank is None or rank > best_rank:
                trainer.save_checkpoint(OUTPUT.parent, OUTPUT.name, run_name, "model_best.pt")
                best_rank = rank
            if local % 5 == 0 or local == transitions // trainer.config.rollout_steps:
                trainer.save_checkpoint(OUTPUT.parent, OUTPUT.name, run_name, "model_latest.pt")
                print("%s steps=%d success=%.3f trigger=%.3f latch=%.3f detach=%.3f" % (
                    stage_name, trainer.training_steps, row.get("success_rate", 0.0),
                    row.get("release_started_rate", 0.0),
                    row.get("release_latched_rate", 0.0),
                    row.get("detached_rate", 0.0)), flush=True)
        latest = output / "checkpoints" / "model_latest.pt"
        cleanup = True
    finally:
        env.close()
    result = {"stage": stage_name, "transitions_this_call": transitions,
              "stage_training_transitions": checkpoint_info(latest)["stage_training_transitions"],
              "elapsed_seconds": time.time() - start, "resource_cleanup": cleanup,
              "last_metrics": records[-1], "checkpoint": checkpoint_info(latest)}
    write_json(output / "training_summary.json", result)
    return result


def evaluate(stage_key, run_name, checkpoint, episodes, *, migrate=False):
    seed_everything(); env_cls, mask, stage_name = STAGES[stage_key]
    env = env_cls(); cleanup = False
    try:
        env.initialize(); env.reset(); trainer = new_trainer(env, stage_key)
        if migrate: initialize_stage(trainer, checkpoint)
        else: trainer.load_checkpoint(Path(checkpoint))
        summary, records = deterministic_evaluate(env, trainer, episodes)
        cleanup = True
    finally:
        env.close()
    summary.update({"stage": stage_name, "active_action_mask": list(mask),
                    "release_semantics": RELEASE_SEMANTICS,
                    "checkpoint": checkpoint_info(checkpoint),
                    "resource_cleanup": cleanup})
    directory = OUTPUT / run_name
    write_json(directory / "summary.json", summary)
    for row in records: append_jsonl(directory / "episodes.jsonl", row)
    return summary


def stage_gate(summary, stage_key):
    safe = summary["unsafe_count"] == 0 and summary["non_finite_count"] == 0
    if stage_key == "stage_a":
        return summary["detached_count"] >= math.ceil(.8 * summary["episodes"]) and safe
    systematic = any(summary[name] >= math.ceil(.1 * summary["episodes"]) for name in (
        "early_release_count", "drop_outside_target_count",
        "table_penetration_count", "unstable_placement_count"))
    return summary["success_rate"] >= .9 and safe and not systematic


def funnel_score(summary):
    funnel = summary["behavior_funnel"]; episodes = max(1, summary["episodes"])
    keys = ("transport_reached", "legal_release_height", "release_triggered",
            "release_latched", "cube_detached", "cube_contacts_table",
            "settled_0p5s", "settled_1p0s", "success")
    return sum(float(funnel.get(key, 0)) / episodes for key in keys)


class TargetZoneRecorder:
    """Camera-only target marker; it has no physics/observation/reward presence."""
    def __init__(self, recorder, env_getter):
        self.recorder, self.env_getter = recorder, env_getter

    @staticmethod
    def project(config, point):
        camera = np.asarray(config.position, dtype=float)
        target = np.asarray(config.target, dtype=float)
        forward = target - camera; forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray((0., 0., 1.))); right /= np.linalg.norm(right)
        up = np.cross(right, forward); relative = np.asarray(point) - camera
        depth = float(relative @ forward)
        if depth <= 1e-6: return None
        focal = config.width / (2 * math.tan(math.radians(config.horizontal_fov) / 2))
        return (int(round(config.width / 2 + focal * float(relative @ right) / depth)),
                int(round(config.height / 2 - focal * float(relative @ up) / depth)))

    def write(self, frame):
        env = self.env_getter(); pixels = frame.data.copy(); cfg = env.scene.camera.config
        z = env.scene.config.table_top + .001
        center = self.project(cfg, (LOWER_RELEASE_TARGET[0], LOWER_RELEASE_TARGET[1], z))
        edge = self.project(cfg, (LOWER_RELEASE_TARGET[0] + .015, LOWER_RELEASE_TARGET[1], z))
        if center is not None and edge is not None:
            radius = max(5, int(math.hypot(edge[0] - center[0], edge[1] - center[1])))
            cv2.circle(pixels, center, radius, (40, 255, 255, 255), 3, cv2.LINE_AA)
            cv2.drawMarker(pixels, center, (40, 255, 255, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(pixels, "TARGET ZONE", (center[0] - 70, center[1] + radius + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (40, 255, 255, 255), 2, cv2.LINE_AA)
        if env._maximum_settle_steps > 0: label = "DETACHED / SETTLING 1.0 S"
        elif env._release_latched: label = "RELEASE LATCHED / PHYSICAL JAW OPENING"
        elif env._transport_reached: label = "TARGET REACHED / LOWER"
        else: label = "GRASPED + LIFTED RESET / TRANSPORT"
        cv2.rectangle(pixels, (8, 8), (632, 48), (0, 0, 0, 255), -1)
        cv2.putText(pixels, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    .58, (70, 255, 90, 255), 2, cv2.LINE_AA)
        self.recorder.write(replace(frame, data=pixels))


def video(checkpoint):
    path = OUTPUT / "video" / "full_place_success.mp4"
    holder = {}; env = recorder = None; summary = records = None; cleanup = False
    try:
        for attempt in range(1, 11):
            recorder = ExperimentalMp4Recorder(path).start()
            overlay = TargetZoneRecorder(recorder, lambda: holder["env"])
            env = FullPlaceEnvironment(with_camera=True, frame_callback=overlay.write)
            holder["env"] = env; env.initialize(); env.reset(); trainer = new_trainer(env, "stage_c")
            trainer.load_checkpoint(Path(checkpoint))
            summary, records = deterministic_evaluate(env, trainer, 1)
            cleanup = recorder.close(); recorder = None; env.close(); env = None
            if summary["success_count"] == 1: break
    finally:
        if recorder is not None: cleanup = recorder.close()
        if env is not None: env.close()
    result = {"passed": bool(summary and summary["success_count"] == 1 and cleanup),
              "evaluation": summary, "episode": records[0] if records else None,
              "video": str(path), "resource_cleanup": cleanup,
              "target_marker_camera_overlay_only": True}
    write_json(path.with_suffix(".json"), result); return result


def production_isolation():
    catalog = task_catalog_document()
    registered = [row["id"] for row in catalog["tasks"]
                  if row["id"] in ("s1_grasp", "s1_lift", "s1_place")]
    result = {"catalog_version": catalog["catalog_version"],
              "catalog_digest": catalog["catalog_digest"],
              "production_task_count": len(catalog["tasks"]),
              "experimental_registered": registered,
              "passed": len(catalog["tasks"]) == 7 and not registered}
    write_json(OUTPUT / "production_isolation.json", result); return result


def continue_after_stage_a(result, a_checkpoint, a_eval):
    if not stage_gate(a_eval, "stage_a"):
        a2_metrics = result["stage_a_2k"]["training"]["last_metrics"]
        trend = (a_eval["release_triggered_count"] >= math.ceil(.5 * a_eval["episodes"])
                 or a2_metrics.get("release_started_rate", 0.0) >= .5)
        if a_eval["detached_count"] == 0 and not trend:
            result.update({"stopped_after": "stage_a_2k",
                           "blocker": "NO_DETACH_AND_NO_RELEASE_TRIGGER_LEARNING"})
            return finalize(result)
        a4 = train("stage_a", "stage_a_2k", 2000, a_checkpoint, resume=True)
        a_checkpoint = Path(a4["checkpoint"]["path"])
        a_eval = evaluate("stage_a", "stage_a_4k/evaluation_30", a_checkpoint, 30)
        result["stage_a_4k"] = {"training": a4, "evaluation": a_eval}
    if not stage_gate(a_eval, "stage_a"):
        result.update({"stopped_after": "stage_a", "blocker": "STAGE_A_GATE_FAILED"})
        return finalize(result)
    result["stage_a_gate"] = "PASS"

    b_checkpoint = materialize("stage_b", a_checkpoint, "stage_b_replay_checkpoint")
    b_replay = evaluate("stage_b", "stage_b_replay_30", b_checkpoint, 30)
    result["stage_b_replay"] = b_replay
    if not stage_gate(b_replay, "stage_b"):
        btrain = train("stage_b", "stage_b_train_5k", 5000, a_checkpoint)
        b_checkpoint = Path(btrain["checkpoint"]["path"])
        beval = evaluate("stage_b", "stage_b_eval_50", b_checkpoint, 50)
        result["stage_b_training"] = btrain; result["stage_b_evaluation"] = beval
    else:
        beval = b_replay
    if not stage_gate(beval, "stage_b"):
        result.update({"stopped_after": "stage_b", "blocker": "STAGE_B_GATE_FAILED"})
        return finalize(result)
    result["stage_b_gate"] = "PASS"

    c_checkpoint = materialize("stage_c", b_checkpoint, "stage_c_replay_checkpoint")
    c_replay = evaluate("stage_c", "stage_c_replay_20", c_checkpoint, 20)
    result["stage_c_replay"] = c_replay; stage_c_transitions = 0
    if c_replay["success_rate"] < .9:
        ctrain = train("stage_c", "stage_c_train_10k", 10000, b_checkpoint)
        stage_c_transitions = 10000; c_checkpoint = Path(ctrain["checkpoint"]["path"])
        ceval = evaluate("stage_c", "stage_c_eval_10k_50", c_checkpoint, 50)
        result["stage_c_train_10k"] = ctrain; result["stage_c_eval_10k"] = ceval
        improvement = (ceval["success_rate"] > c_replay["success_rate"]
                       or funnel_score(ceval) >= funnel_score(c_replay) + .25)
        if ceval["success_rate"] < .9 and improvement:
            ctrain20 = train("stage_c", "stage_c_train_10k", 10000, c_checkpoint, resume=True)
            stage_c_transitions = 20000
            c_checkpoint = Path(ctrain20["checkpoint"]["path"])
            ceval = evaluate("stage_c", "stage_c_eval_20k_50", c_checkpoint, 50)
            result["stage_c_train_20k"] = ctrain20; result["stage_c_eval_20k"] = ceval
        elif ceval["success_rate"] < .9:
            result.update({"stage_c_training_transitions": stage_c_transitions,
                           "stopped_after": "stage_c_10k",
                           "blocker": "STAGE_C_FUNNEL_NO_MATERIAL_IMPROVEMENT"})
            return finalize(result)
    result["stage_c_training_transitions"] = stage_c_transitions
    final = evaluate("stage_c", "final_100", c_checkpoint, 100)
    result["final_100"] = final
    final_gate = (final["success_rate"] >= .9 and final["wilson_95_lower"] >= .8
                  and final["unsafe_count"] == 0 and final["non_finite_count"] == 0)
    result["p4_full_place_gate"] = "PASS" if final_gate else "FAIL"
    result["p4_s1_place_ppo_gate"] = "PASS" if final_gate else "FAIL"
    if final_gate:
        result["video"] = video(c_checkpoint)
    return finalize(result)


def run_all():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if source_digest != P4_2_20K_SHA256:
        raise RuntimeError("P4-2 source checkpoint digest mismatch")
    result = {"task_id": "s1_place", "release_semantics": RELEASE_SEMANTICS,
              "source_checkpoint": str(SOURCE), "source_sha256": source_digest}

    a2 = train("stage_a", "stage_a_2k", 2000, SOURCE, reset_release=True)
    a_checkpoint = Path(a2["checkpoint"]["path"])
    a_eval = evaluate("stage_a", "stage_a_2k/evaluation_30", a_checkpoint, 30)
    result["stage_a_2k"] = {"training": a2, "evaluation": a_eval}
    return continue_after_stage_a(result, a_checkpoint, a_eval)


def continue_from_a2():
    result = json.loads((OUTPUT / "manifest.json").read_text())
    result.pop("files", None); result.pop("isolation", None)
    result.pop("stopped_after", None); result.pop("blocker", None)
    a_checkpoint = Path(result["stage_a_2k"]["training"]["checkpoint"]["path"])
    a_eval = result["stage_a_2k"]["evaluation"]
    return continue_after_stage_a(result, a_checkpoint, a_eval)


def finalize(result):
    result["isolation"] = production_isolation()
    if not result["isolation"]["passed"]:
        result["p4_full_place_gate"] = result["p4_s1_place_ppo_gate"] = "FAIL"
        result["blocker"] = "PRODUCTION_ISOLATION_FAILED"
    result.setdefault("p4_full_place_gate", "NOT_REACHED")
    result.setdefault("p4_s1_place_ppo_gate", "FAIL")
    files = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(OUTPUT))] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size}
    result["files"] = files
    write_json(OUTPUT / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("run", "continue-a2"), default="run")
    args = parser.parse_args(); os.environ.pop("DISPLAY", None)
    result = run_all() if args.mode == "run" else continue_from_a2()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
