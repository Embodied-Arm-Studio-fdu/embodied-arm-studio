#!/usr/bin/env python3
"""P4-R4 replay-first Full Place completion with safe automatic release."""

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

from simulation_core.experimental.manipulation.auto_release_env import (
    AUTO_RELEASE_ACTIVE_ACTION_MASK, AUTO_RELEASE_SEMANTICS,
    AutoReleaseFullPlaceEnvironment,
)
from simulation_core.experimental.manipulation.lower_release_env import LOWER_RELEASE_TARGET
from simulation_core.experimental.manipulation.lower_release_ppo import deterministic_evaluate
from simulation_core.experimental.manipulation.place_completion import (
    P4_2_20K_SHA256, StagePlaceTrainer,
)
from simulation_core.experimental.manipulation.scripted_runtime import ExperimentalMp4Recorder
from simulation_core.ppo.trainer import PPOConfig
from simulation_core.task_catalog import task_catalog_document


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "experimental" / "p4_r4_s1_place"
SOURCE = (ROOT / "artifacts" / "experimental" / "p4_2_s1_place" /
          "train_20k" / "checkpoints" / "model_latest.pt")


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


def trainer(env):
    return StagePlaceTrainer(
        env, active_action_mask=AUTO_RELEASE_ACTIVE_ACTION_MASK,
        stage_name="full_place_auto_release", release_semantics=AUTO_RELEASE_SEMANTICS,
        config=ppo_config(), device="cuda:0")


def checkpoint_info(path):
    path = Path(path); payload = torch.load(path, map_location="cpu", weights_only=False)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "task_id": payload.get("task_id"),
            "active_action_mask": list(payload.get("active_action_mask") or ()),
            "release_semantics": payload.get("release_semantics"),
            "stage_training_transitions": payload.get("stage_training_transitions"),
            "source_checkpoint": payload.get("source_checkpoint")}


def materialize_source():
    env = AutoReleaseFullPlaceEnvironment()
    try:
        env.initialize(); env.reset(); agent = trainer(env)
        agent.initialize_from_checkpoint(SOURCE, P4_2_20K_SHA256)
        return agent.save_checkpoint(OUTPUT.parent, OUTPUT.name, "replay_checkpoint")
    finally:
        env.close()


def evaluate(run_name, checkpoint, episodes):
    seed_everything(); env = AutoReleaseFullPlaceEnvironment(); cleanup = False
    try:
        env.initialize(); env.reset(); agent = trainer(env); agent.load_checkpoint(checkpoint)
        summary, records = deterministic_evaluate(env, agent, episodes); cleanup = True
    finally:
        env.close()
    summary.update({"stage": "full_place_auto_release",
                    "active_action_mask": list(AUTO_RELEASE_ACTIVE_ACTION_MASK),
                    "release_semantics": AUTO_RELEASE_SEMANTICS,
                    "checkpoint": checkpoint_info(checkpoint),
                    "resource_cleanup": cleanup})
    directory = OUTPUT / run_name; write_json(directory / "summary.json", summary)
    for row in records: append_jsonl(directory / "episodes.jsonl", row)
    return summary


def train(run_name, source, transitions=5000):
    seed_everything(); output = OUTPUT / run_name; output.mkdir(parents=True, exist_ok=True)
    env = AutoReleaseFullPlaceEnvironment(); records = []; best = None; cleanup = False
    start = time.time()
    try:
        env.initialize(); env.reset(); agent = trainer(env)
        agent.initialize_from_checkpoint(source)
        write_json(output / "config.json", {
            "task_id": "s1_place", "stage": "full_place_auto_release",
            "active_action_mask": list(AUTO_RELEASE_ACTIVE_ACTION_MASK),
            "release_semantics": AUTO_RELEASE_SEMANTICS,
            "source": str(source), "transitions": transitions,
            "environment": asdict(env.config), "ppo": asdict(ppo_config()),
            "release_action_trained": False,
        })
        for local in range(1, transitions // agent.config.rollout_steps + 1):
            metrics = agent.train_one_update()
            row = {"update_index": agent.update_steps,
                   "training_steps": agent.training_steps, **metrics}
            records.append(row); append_jsonl(output / "metrics.jsonl", row)
            for episode in agent._rollout_episode_records:
                append_jsonl(output / "episodes.jsonl", episode)
            rank = (row.get("success_rate", 0.0), row.get("detached_rate", 0.0),
                    row.get("descended_rate", 0.0), row["average_reward"])
            if best is None or rank > best:
                agent.save_checkpoint(OUTPUT.parent, OUTPUT.name, run_name, "model_best.pt")
                best = rank
            if local % 5 == 0 or local == transitions // agent.config.rollout_steps:
                agent.save_checkpoint(OUTPUT.parent, OUTPUT.name, run_name, "model_latest.pt")
                print("steps=%d success=%.3f target=%.3f legal=%.3f detach=%.3f" % (
                    agent.training_steps, row.get("success_rate", 0.0),
                    row.get("transport_reached_rate", 0.0),
                    row.get("descended_rate", 0.0), row.get("detached_rate", 0.0)),
                    flush=True)
        latest = output / "checkpoints" / "model_latest.pt"; cleanup = True
    finally:
        env.close()
    result = {"run": run_name, "transitions": transitions,
              "elapsed_seconds": time.time() - start, "resource_cleanup": cleanup,
              "last_metrics": records[-1], "checkpoint": checkpoint_info(latest)}
    write_json(output / "training_summary.json", result); return result


def gate(summary):
    systematic = any(summary[name] >= math.ceil(.1 * summary["episodes"]) for name in (
        "drop_outside_target_count", "workspace_violation_count",
        "table_penetration_count", "unstable_placement_count"))
    return (summary["success_rate"] >= .9 and summary["unsafe_count"] == 0
            and summary["non_finite_count"] == 0 and not systematic)


def score(summary):
    n = max(1, summary["episodes"]); funnel = summary["behavior_funnel"]
    keys = ("transport_reached", "legal_release_height", "auto_release_ready",
            "release_latched", "actual_jaw_opened", "cube_detached",
            "cube_contacts_table", "settled_0p5s", "settled_1p0s", "success")
    return sum(float(funnel.get(key, 0)) / n for key in keys)


class Overlay:
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
        if env._maximum_settle_steps >= env.hold_steps_required: label = "PLACED"
        elif env._detached: label = "DETACHED"
        elif env._release_latched: label = "AUTO RELEASE"
        else: label = "PLACE POLICY"
        cv2.rectangle(pixels, (8, 8), (632, 48), (0, 0, 0, 255), -1)
        cv2.putText(pixels, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    .62, (70, 255, 90, 255), 2, cv2.LINE_AA)
        self.recorder.write(replace(frame, data=pixels))


def video(checkpoint):
    path = OUTPUT / "video" / "full_place_success.mp4"
    holder = {}; env = recorder = None; summary = records = None; cleanup = False
    try:
        for attempt in range(1, 6):
            recorder = ExperimentalMp4Recorder(path).start()
            overlay = Overlay(recorder, lambda: holder["env"])
            env = AutoReleaseFullPlaceEnvironment(with_camera=True, frame_callback=overlay.write)
            holder["env"] = env; env.initialize(); env.reset(); agent = trainer(env)
            agent.load_checkpoint(checkpoint)
            summary, records = deterministic_evaluate(env, agent, 1)
            cleanup = recorder.close(); recorder = None; env.close(); env = None
            if summary["success_count"] == 1: break
    finally:
        if recorder is not None: cleanup = recorder.close()
        if env is not None: env.close()
    result = {"passed": bool(summary and summary["success_count"] == 1 and cleanup),
              "evaluation": summary, "episode": records[0] if records else None,
              "video": str(path), "resource_cleanup": cleanup,
              "target_marker_camera_only": True}
    write_json(path.with_suffix(".json"), result); return result


def isolation():
    catalog = task_catalog_document()
    registered = [row["id"] for row in catalog["tasks"]
                  if row["id"] in ("s1_grasp", "s1_lift", "s1_place")]
    value = {"catalog_version": catalog["catalog_version"],
             "catalog_digest": catalog["catalog_digest"],
             "production_task_count": len(catalog["tasks"]),
             "experimental_registered": registered,
             "passed": len(catalog["tasks"]) == 7 and not registered}
    write_json(OUTPUT / "production_isolation.json", value); return value


def finalize(result):
    result["isolation"] = isolation()
    passed = (result.get("final_gate") == "PASS" and result["isolation"]["passed"])
    result["p4_r4_full_place_gate"] = "PASS" if passed else "FAIL"
    result["p4_s1_place_ppo_gate"] = "PASS" if passed else "FAIL"
    files = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(OUTPUT))] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size}
    result["files"] = files; write_json(OUTPUT / "manifest.json", result); return result


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if digest != P4_2_20K_SHA256: raise RuntimeError("P4-2 checkpoint digest mismatch")
    result = {"task_id": "s1_place", "release_semantics": AUTO_RELEASE_SEMANTICS,
              "source_checkpoint": str(SOURCE), "source_sha256": digest,
              "training_transitions": 0}
    candidate = materialize_source()
    replay = evaluate("replay_20", candidate, 20); result["replay_20"] = replay
    if not gate(replay):
        policy_problem = (replay["transport_reached_count"] < math.ceil(.9 * replay["episodes"])
                          or replay["legal_release_height_count"] <
                          math.ceil(.9 * replay["episodes"]))
        if not policy_problem:
            result["blocker"] = "AUTO_RELEASE_OR_PHYSICAL_PLACEMENT_NOT_POLICY"
            return finalize(result)
        trained5 = train("train_5k", candidate); result["train_5k"] = trained5
        result["training_transitions"] = 5000
        eval5 = evaluate("eval_5k_30", Path(trained5["checkpoint"]["path"]), 30)
        result["eval_5k_30"] = eval5
        if gate(eval5):
            candidate = Path(trained5["checkpoint"]["path"])
        else:
            improved = score(eval5) >= score(replay) + .25
            if not improved:
                result["blocker"] = "NO_MATERIAL_TRANSPORT_LOWER_FUNNEL_IMPROVEMENT_AT_5K"
                return finalize(result)
            trained10 = train("train_10k", Path(trained5["checkpoint"]["path"]))
            result["train_10k"] = trained10; result["training_transitions"] = 10000
            eval10 = evaluate("eval_10k_30", Path(trained10["checkpoint"]["path"]), 30)
            result["eval_10k_30"] = eval10
            candidates = ((eval5, Path(trained5["checkpoint"]["path"])),
                          (eval10, Path(trained10["checkpoint"]["path"])))
            best_summary, candidate = max(candidates, key=lambda pair:
                                           (pair[0]["success_rate"], score(pair[0])))
            result["best_pre_final"] = best_summary
    final = evaluate("final_100", candidate, 100); result["final_100"] = final
    final_pass = (gate(final) and final["wilson_95_lower"] >= .8)
    result["final_gate"] = "PASS" if final_pass else "FAIL"
    if final_pass: result["video"] = video(candidate)
    if not final_pass: result["blocker"] = "FINAL_100_GATE_FAILED"
    return finalize(result)


def finalize_stopped():
    replay = json.loads((OUTPUT / "replay_20" / "summary.json").read_text())
    metrics_path = OUTPUT / "train_5k" / "metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    best = OUTPUT / "train_5k" / "checkpoints" / "model_best.pt"
    latest = OUTPUT / "train_5k" / "checkpoints" / "model_latest.pt"
    result = {
        "task_id": "s1_place", "release_semantics": AUTO_RELEASE_SEMANTICS,
        "source_checkpoint": str(SOURCE),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "status": "stopped", "stop_requested_by_user": True,
        "blocker": "NO_MEANINGFUL_TRANSPORT_LOWER_IMPROVEMENT_USER_STOPPED",
        "replay_20": replay,
        "training_transitions_collected": int(metrics[-1]["training_steps"]),
        "last_training_metrics": metrics[-1],
        "recoverable_best_checkpoint": checkpoint_info(best),
        "periodic_latest_checkpoint": checkpoint_info(latest),
        "final_100_run": False, "video_run": False,
        "final_gate": "FAIL",
    }
    return finalize(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize-stopped", action="store_true")
    args = parser.parse_args(); os.environ.pop("DISPLAY", None)
    result = finalize_stopped() if args.finalize_stopped else run()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
