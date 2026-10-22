#!/usr/bin/env python3
"""P4-R5 zero-training staged Full Place replay and gate."""

import hashlib
import json
import math
import os
from dataclasses import replace
from pathlib import Path
from collections import Counter

try:
    from isaacgym import gymapi as _gymapi_import_order  # noqa: F401
except ModuleNotFoundError:
    _gymapi_import_order = None

import cv2
import numpy as np
import torch

from simulation_core.evaluation import wilson_confidence_interval
from simulation_core.experimental.manipulation.lower_release_env import LOWER_RELEASE_TARGET
from simulation_core.experimental.manipulation.lower_release_schema import LOWER_RELEASE_POLICY_SCHEMA
from simulation_core.experimental.manipulation.scripted_runtime import ExperimentalMp4Recorder
from simulation_core.experimental.manipulation.staged_place_env import StagedFullPlaceEnvironment
from simulation_core.ppo.normalizer import ObservationNormalizer
from simulation_core.ppo.policy import MLPActorCritic, flatten_observation
from simulation_core.task_catalog import task_catalog_document


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "experimental" / "p4_r5_s1_place"
TRANSPORT = (ROOT / "artifacts" / "experimental" / "p4_1_s1_place" /
             "train_20k" / "checkpoints" / "model_latest.pt")
LOWER = (ROOT / "artifacts" / "experimental" / "p4_2_s1_place" /
         "train_20k" / "checkpoints" / "model_latest.pt")
TRANSPORT_SHA = "eb9fab2287ffd6a735dc27a1ccccd8eb3c71678617b4aef2bac1561d556cf291"
LOWER_SHA = "58ddf5e91a29ac087d37dac9094dad77a3596f5e32693fe5ad3f8e78b9dcba33"


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


class PolicyRuntime:
    def __init__(self, path, expected_sha, expected_mask, name):
        self.path = Path(path); self.name = str(name)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.sha256 != expected_sha: raise ValueError("%s checkpoint SHA mismatch" % name)
        payload = torch.load(self.path, map_location="cuda:0", weights_only=False)
        expected = {"task_id": "s1_place", "observation_dim": 32, "action_dim": 4,
                    "active_action_mask": tuple(expected_mask)}
        for key, value in expected.items():
            actual = payload.get(key)
            if key == "active_action_mask": actual = tuple(actual or ())
            if actual != value: raise ValueError("%s checkpoint %s mismatch" % (name, key))
        self.policy = MLPActorCritic(32, 4, action_low=(-1.,) * 4,
                                     action_high=(1.,) * 4).to("cuda:0")
        self.policy.load_state_dict(payload["policy"]); self.policy.eval()
        self.normalizer = ObservationNormalizer(dimension=32,
                                                previous_action_slice=slice(0, 32))
        self.normalizer.load_state_dict(payload["normalizer_state"], frozen=True)
        self.metadata = {"role": self.name, "path": str(self.path), "sha256": self.sha256,
                         "active_action_mask": list(expected_mask),
                         "normalizer_count": self.normalizer.count,
                         "normalizer_object_id": id(self.normalizer),
                         "training_steps": int(payload["training_steps"])}

    def act(self, observation):
        raw = flatten_observation(observation, LOWER_RELEASE_POLICY_SCHEMA).to("cuda:0")
        normalized = self.normalizer.normalize(raw, update=False)
        with torch.no_grad(): action, _, _ = self.policy.act(normalized, deterministic=True)
        return tuple(float(value) for value in action.detach().cpu())


def episode(env, transport, lower, index):
    observation = env.reset(); done = False; reward_sum = 0.0; steps = 0
    while not done:
        if env.staged_phase == "TRANSPORT": action = transport.act(observation)
        elif env.staged_phase == "LOWER": action = lower.act(observation)
        else: action = (0.0, 0.0, 0.0, 0.0)
        observation, reward, done, info = env.step(action)
        reward_sum += float(reward); steps += 1
        if steps > env.max_steps + 1: raise RuntimeError("staged evaluation exceeded horizon")
    return {"episode": index, "steps": steps, "reward": reward_sum,
            **{name: info.get(name) for name in (
                "success", "failure_reason", "transport_target_reached",
                "transport_success",
                "policy_switched_to_lower", "transport_ready_duration",
                "legal_release_height", "auto_release_ready", "release_latched",
                "actual_jaw_opened", "detached", "table_contact", "inside_target_zone",
                "settled_0p5s", "settled_1p0s", "xy_error", "bottom_clearance",
                "minimum_bottom_clearance", "release_height", "detach_height",
                "cube_linear_speed", "cube_angular_speed", "cube_tilt", "grasp_valid",
                "workspace_violation", "drop_outside_target", "table_penetration",
                "object_unstable", "unsafe", "non_finite", "staged_phase",
                "transport_condition", "auto_release_condition")}}


def summarize(records):
    n = len(records); successes = sum(bool(row["success"]) for row in records)
    lower, upper = wilson_confidence_interval(successes, n)
    tally = lambda name: sum(bool(row.get(name)) for row in records)
    histogram = Counter(row["failure_reason"] or "SUCCESS" for row in records)
    funnel = {
        "valid_reset": n,
        "transport_target_reached": tally("transport_target_reached"),
        "transport_success": tally("transport_success"),
        "policy_switched_to_lower": tally("policy_switched_to_lower"),
        "legal_release_height": tally("legal_release_height"),
        "auto_release_ready": tally("auto_release_ready"),
        "release_latched": tally("release_latched"),
        "actual_jaw_opened": tally("actual_jaw_opened"),
        "detached": tally("detached"), "table_contact": tally("table_contact"),
        "settled_0p5s": tally("settled_0p5s"),
        "settled_1p0s": tally("settled_1p0s"), "success": successes,
    }
    failure_stages = Counter()
    for row in records:
        if row["success"]:
            continue
        if not row.get("transport_success"):
            stage = "transport_timeout"
        elif not row.get("legal_release_height"):
            stage = "lower_failure"
        elif not row.get("detached"):
            stage = "release_failure"
        else:
            stage = "settle_failure"
        failure_stages[stage] += 1
    return {"deterministic": True, "episodes": n, "success_count": successes,
            "success_rate": successes / n, "wilson_95_lower": lower,
            "wilson_95_upper": upper, "behavior_funnel": funnel,
            "workspace_violation_count": tally("workspace_violation"),
            "drop_outside_target_count": tally("drop_outside_target"),
            "table_penetration_count": tally("table_penetration"),
            "unstable_placement_count": tally("object_unstable"),
            "unsafe_count": tally("unsafe"), "non_finite_count": tally("non_finite"),
            "failure_reason_histogram": dict(histogram),
            "failure_stage_histogram": dict(failure_stages),
            "training_transitions": 0}


def evaluate(name, episodes, transport, lower):
    env = StagedFullPlaceEnvironment(); records = []; cleanup = False
    try:
        env.initialize()
        for index in range(episodes): records.append(episode(env, transport, lower, index))
        cleanup = True
    finally: env.close()
    summary = summarize(records)
    summary.update({"resource_cleanup": cleanup,
                    "transport_runtime": transport.metadata,
                    "lower_runtime": lower.metadata,
                    "separate_policy_weights": transport.policy is not lower.policy,
                    "separate_normalizers": transport.normalizer is not lower.normalizer})
    directory = OUTPUT / name; write_json(directory / "summary.json", summary)
    for row in records: append_jsonl(directory / "episodes.jsonl", row)
    return summary


def passed(summary, final=False):
    systematic = any(summary[name] >= math.ceil(.1 * summary["episodes"]) for name in (
        "workspace_violation_count", "drop_outside_target_count",
        "table_penetration_count", "unstable_placement_count"))
    gate = (summary["success_rate"] >= .9 and summary["unsafe_count"] == 0
            and summary["non_finite_count"] == 0 and not systematic)
    return gate and (not final or summary["wilson_95_lower"] >= .8)


class Overlay:
    TARGET_RING_ALPHA = .42
    TARGET_RING_THICKNESS = 1
    TARGET_CROSS_HALF_SIZE = 4
    TARGET_HUD_FONT_SCALE = .36

    def __init__(self, recorder, env_getter):
        self.recorder, self.env_getter = recorder, env_getter
    @staticmethod
    def project(config, point):
        camera = np.asarray(config.position, dtype=float); target = np.asarray(config.target, dtype=float)
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
        if center and edge:
            radius = max(5, int(math.hypot(edge[0]-center[0], edge[1]-center[1])))
            marker = pixels.copy()
            cyan = (40,255,255,255)
            cv2.circle(marker, center, radius, cyan, self.TARGET_RING_THICKNESS, cv2.LINE_AA)
            cv2.line(marker, (center[0]-self.TARGET_CROSS_HALF_SIZE, center[1]),
                     (center[0]+self.TARGET_CROSS_HALF_SIZE, center[1]),
                     cyan, self.TARGET_RING_THICKNESS, cv2.LINE_AA)
            cv2.line(marker, (center[0], center[1]-self.TARGET_CROSS_HALF_SIZE),
                     (center[0], center[1]+self.TARGET_CROSS_HALF_SIZE),
                     cyan, self.TARGET_RING_THICKNESS, cv2.LINE_AA)
            cv2.addWeighted(marker, self.TARGET_RING_ALPHA, pixels,
                            1. - self.TARGET_RING_ALPHA, 0., pixels)
        labels = {"TRANSPORT":"TRANSPORT POLICY", "LOWER":"LOWER POLICY",
                  "AUTO_RELEASE":"AUTO RELEASE", "SETTLE":"DETACHED",
                  "SUCCESS":"PLACE SUCCESS"}
        label = labels.get(env.staged_phase, env.staged_phase)
        cv2.rectangle(pixels, (8,8), (632,48), (0,0,0,255), -1)
        cv2.putText(pixels, label, (18,36), cv2.FONT_HERSHEY_SIMPLEX,
                    .62, (70,255,90,255), 2, cv2.LINE_AA)
        milestones = (
            ("GRASPED + LIFTED RESET", True),
            ("TRANSPORT SUCCESS", bool(getattr(env, "_transport_success", False))),
            ("LOWER", bool(getattr(env, "_switched_to_lower", False))),
            ("AUTO RELEASE", bool(getattr(env, "_release_started", False))),
            ("DETACHED", bool(getattr(env, "_detached", False))),
            ("SETTLED", bool(getattr(env, "_settled_half", False))),
            ("PLACE SUCCESS", env.staged_phase == "SUCCESS"),
        )
        cv2.rectangle(pixels, (8,58), (250,252), (0,0,0,190), -1)
        for index, (name, reached) in enumerate(milestones):
            color = (70,255,90,255) if reached else (150,150,150,255)
            prefix = "[OK]" if reached else "[  ]"
            cv2.putText(pixels, "%s %s" % (prefix, name), (18,82 + index * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, .40, color, 1, cv2.LINE_AA)
        hud_text = "TARGET ZONE"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), _ = cv2.getTextSize(
            hud_text, font, self.TARGET_HUD_FONT_SCALE, 1)
        text_origin = (pixels.shape[1] - text_width - 14, pixels.shape[0] - 12)
        hud = pixels.copy()
        cv2.rectangle(hud,
                      (text_origin[0] - 6, text_origin[1] - text_height - 5),
                      (pixels.shape[1] - 8, text_origin[1] + 5),
                      (0,0,0,255), -1)
        cv2.addWeighted(hud, .45, pixels, .55, 0., pixels)
        cv2.putText(pixels, hud_text, text_origin, font,
                    self.TARGET_HUD_FONT_SCALE, (40,255,255,255), 1, cv2.LINE_AA)
        self.recorder.write(replace(frame, data=pixels))


def video(transport, lower):
    path = OUTPUT / "video" / "staged_full_place_success.mp4"
    holder = {}; env = recorder = None; row = None; cleanup = False
    try:
        for attempt in range(1, 6):
            recorder = ExperimentalMp4Recorder(path).start()
            overlay = Overlay(recorder, lambda: holder["env"])
            env = StagedFullPlaceEnvironment(with_camera=True, frame_callback=overlay.write)
            holder["env"] = env; env.initialize(); row = episode(env, transport, lower, attempt-1)
            if row["success"]:
                # The terminal success latch is updated after the final control
                # step's camera callback.  Append one second of the terminal
                # camera frame so PLACE SUCCESS is visible in the evidence.
                terminal_frame = env.scene.capture()
                for _ in range(30):
                    overlay.write(terminal_frame)
            cleanup = recorder.close(); recorder = None; env.close(); env = None
            if row["success"]: break
    finally:
        if recorder is not None: cleanup = recorder.close()
        if env is not None: env.close()
    result = {"passed": bool(row and row["success"] and cleanup), "episode": row,
              "video": str(path), "resource_cleanup": cleanup,
              "target_marker_camera_only": True,
              "target_overlay": {
                  "ring_fill": "transparent",
                  "ring_alpha": Overlay.TARGET_RING_ALPHA,
                  "ring_thickness_px": Overlay.TARGET_RING_THICKNESS,
                  "center_cross_half_size_px": Overlay.TARGET_CROSS_HALF_SIZE,
                  "hud_anchor": "bottom_right",
                  "hud_font_scale": Overlay.TARGET_HUD_FONT_SCALE,
              }}
    write_json(path.with_suffix(".json"), result); return result


def isolation():
    catalog = task_catalog_document(); registered = [row["id"] for row in catalog["tasks"]
        if row["id"] in ("s1_grasp", "s1_lift", "s1_place")]
    return {"catalog_version": catalog["catalog_version"],
            "catalog_digest": catalog["catalog_digest"],
            "production_task_count": len(catalog["tasks"]),
            "experimental_registered": registered,
            "passed": len(catalog["tasks"]) == 7 and not registered}


def finalize(result):
    result["isolation"] = isolation(); write_json(OUTPUT / "production_isolation.json", result["isolation"])
    gate = result.get("final_gate") == "PASS" and result["isolation"]["passed"]
    result["p4_r5_staged_place_gate"] = "PASS" if gate else "FAIL"
    result["p4_s1_place_ppo_gate"] = "PASS" if gate else "FAIL"
    files = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(OUTPUT))] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    result["files"] = files; write_json(OUTPUT / "manifest.json", result); return result


def run():
    os.environ.pop("DISPLAY", None); OUTPUT.mkdir(parents=True, exist_ok=True)
    transport = PolicyRuntime(TRANSPORT, TRANSPORT_SHA, (1,1,0,0), "transport")
    lower = PolicyRuntime(LOWER, LOWER_SHA, (0,0,1,1), "lower")
    result = {"task_id":"s1_place", "training_transitions":0,
              "separate_weights":True, "separate_normalizers":True,
              "transport_checkpoint":transport.metadata, "lower_checkpoint":lower.metadata}
    replay = evaluate("replay_20", 20, transport, lower); result["replay_20"] = replay
    if not passed(replay):
        funnel = replay["behavior_funnel"]
        ordered = ("transport_target_reached", "policy_switched_to_lower",
                   "legal_release_height", "auto_release_ready", "release_latched",
                   "actual_jaw_opened", "detached", "table_contact", "settled_1p0s", "success")
        result["minimal_blocker_stage"] = next((name for name in ordered if funnel[name] < 18), "gate")
        result["final_100_run"] = False; result["video_run"] = False
        return finalize(result)
    final = evaluate("final_100", 100, transport, lower); result["final_100"] = final
    result["final_gate"] = "PASS" if passed(final, final=True) else "FAIL"
    if result["final_gate"] == "PASS": result["video"] = video(transport, lower)
    return finalize(result)


if __name__ == "__main__": print(json.dumps(run(), indent=2, sort_keys=True))
