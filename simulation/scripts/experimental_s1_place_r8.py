#!/usr/bin/env python3
"""P4-R8 zero-training Full Place gate with unified Transport success."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import experimental_s1_place_staged as staged


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "experimental" / "p4_r8_s1_place"
TRANSPORT = (ROOT / "artifacts" / "experimental" / "p4_r7_s1_place" /
             "train_3k" / "checkpoints" / "model_latest.pt")
LOWER = (ROOT / "artifacts" / "experimental" / "p4_2_s1_place" /
         "train_20k" / "checkpoints" / "model_latest.pt")
TRANSPORT_SHA = "d2f7f662efe7d7c978ac241050664dfd8d94e3b192bd68806ffbb69bef7ea587"
LOWER_SHA = "58ddf5e91a29ac087d37dac9094dad77a3596f5e32693fe5ad3f8e78b9dcba33"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtimes():
    if sha256(TRANSPORT) != TRANSPORT_SHA:
        raise ValueError("R7 validated Transport checkpoint SHA mismatch")
    if sha256(LOWER) != LOWER_SHA:
        raise ValueError("locked P4-2 Lower checkpoint SHA mismatch")
    return (
        staged.PolicyRuntime(TRANSPORT, TRANSPORT_SHA, (1, 1, 0, 0), "transport"),
        staged.PolicyRuntime(LOWER, LOWER_SHA, (0, 0, 1, 1), "lower"),
    )


def first_failure_stage(summary):
    funnel = summary["behavior_funnel"]
    ordered = (
        "transport_success", "policy_switched_to_lower", "legal_release_height",
        "auto_release_ready", "actual_jaw_opened", "detached", "table_contact",
        "settled_0p5s", "settled_1p0s", "success",
    )
    threshold = int(.9 * summary["episodes"])
    return next((name for name in ordered if funnel[name] < threshold), None)


def run():
    os.environ.pop("DISPLAY", None)
    staged.OUTPUT = OUTPUT
    transport, lower = runtimes()
    result = {
        "task_id": "s1_place",
        "phase": "P4-R8",
        "training_transitions": 0,
        "transport_checkpoint": transport.metadata,
        "lower_checkpoint": lower.metadata,
        "transport_gate": {
            "predicate": (
                "simulation_core.experimental.manipulation.place_env."
                "transport_success_predicate"),
            "event": (
                "simulation_core.experimental.manipulation.place_env."
                "advance_transport_success_event"),
            "xy_tolerance_m": .015,
            "hold_seconds": 1.0,
            "shared_by_standalone_and_staged": True,
        },
        "fixed_z_reconstruction": True,
        "raw_4d_previous_action_observation": True,
        "transport_physics_action_mask": [1, 1, 0, 0],
    }

    replay = staged.evaluate("replay_20", 20, transport, lower)
    result["replay_20"] = replay
    replay_passed = staged.passed(replay)
    result["replay_gate"] = "PASS" if replay_passed else "FAIL"
    if not replay_passed:
        result["earliest_failure_stage"] = first_failure_stage(replay)
        result["final_100_run"] = False
        result["video_run"] = False
    else:
        final = staged.evaluate("final_100", 100, transport, lower)
        result["final_100"] = final
        result["final_100_run"] = True
        final_passed = staged.passed(final, final=True)
        result["final_gate"] = "PASS" if final_passed else "FAIL"
        if final_passed:
            result["video"] = staged.video(transport, lower)
            result["video_run"] = True
        else:
            result["earliest_failure_stage"] = first_failure_stage(final)
            result["video_run"] = False

    result["isolation"] = staged.isolation()
    passed = bool(result.get("final_gate") == "PASS"
                  and result["isolation"]["passed"]
                  and result.get("video", {}).get("passed"))
    result["p4_r8_full_place_gate"] = "PASS" if passed else "FAIL"
    result["p4_s1_place_ppo_gate"] = "PASS" if passed else "FAIL"
    write_json(OUTPUT / "production_isolation.json", result["isolation"])
    files = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(OUTPUT))] = {
                "sha256": sha256(path), "bytes": path.stat().st_size}
    result["files"] = files
    write_json(OUTPUT / "manifest.json", result)
    return result


def attach_video():
    staged.OUTPUT = OUTPUT
    transport, lower = runtimes()
    result = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    result["video"] = staged.video(transport, lower)
    result["video_run"] = True
    files = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(OUTPUT))] = {
                "sha256": sha256(path), "bytes": path.stat().st_size}
    result["files"] = files
    write_json(OUTPUT / "manifest.json", result)
    return result["video"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("run", "video"), default="run")
    args = parser.parse_args()
    value = run() if args.mode == "run" else attach_video()
    print(json.dumps(value, indent=2, sort_keys=True))
