#!/usr/bin/env python3
"""P4-R10 clean-reset robustness and conditional final gate (zero training)."""

import hashlib
import json
import os
from pathlib import Path
import random

try:
    from isaacgym import gymapi as _gymapi_import_order  # noqa: F401
except ModuleNotFoundError:
    _gymapi_import_order = None

import torch

import experimental_s1_place_staged as staged
from simulation_core.experimental.manipulation.staged_place_env import (
    StagedFullPlaceEnvironment,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "experimental" / "p4_r10_s1_place"
TRANSPORT = (ROOT / "artifacts" / "experimental" / "p4_r7_s1_place" /
             "train_3k" / "checkpoints" / "model_latest.pt")
LOWER = (ROOT / "artifacts" / "experimental" / "p4_2_s1_place" /
         "train_20k" / "checkpoints" / "model_latest.pt")
TRANSPORT_SHA = "d2f7f662efe7d7c978ac241050664dfd8d94e3b192bd68806ffbb69bef7ea587"
LOWER_SHA = "58ddf5e91a29ac087d37dac9094dad77a3596f5e32693fe5ad3f8e78b9dcba33"
STAGE_A_SEEDS = tuple(range(2026101000, 2026101030))
FINAL_SEEDS = tuple(range(2026102000, 2026102100))


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path, records):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_everything(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def runtimes():
    if sha256(TRANSPORT) != TRANSPORT_SHA:
        raise ValueError("locked R7 Transport checkpoint SHA mismatch")
    if sha256(LOWER) != LOWER_SHA:
        raise ValueError("locked P4-2 Lower checkpoint SHA mismatch")
    return (
        staged.PolicyRuntime(TRANSPORT, TRANSPORT_SHA, (1, 1, 0, 0), "transport"),
        staged.PolicyRuntime(LOWER, LOWER_SHA, (0, 0, 1, 1), "lower"),
    )


def evaluate_seeded(name, seeds, transport, lower):
    if len(seeds) != len(set(seeds)):
        raise ValueError("evaluation seeds must be unique")
    env = StagedFullPlaceEnvironment(); records = []; cleanup = False
    try:
        env.initialize()
        for index, seed in enumerate(seeds):
            seed_everything(seed)
            row = staged.episode(env, transport, lower, index)
            row["deterministic_seed"] = seed
            records.append(row)
        cleanup = True
    finally:
        env.close()
    summary = staged.summarize(records)
    summary.update({
        "seed_count": len(seeds), "unique_seed_count": len(set(seeds)),
        "seed_first": seeds[0], "seed_last": seeds[-1],
        "same_environment_simulator": True,
        "r9_discarded_priming_reset": True,
        "r9_visible_reset": True,
        "formal_place_reset_distribution": True,
        "resource_cleanup": cleanup,
        "transport_runtime": transport.metadata,
        "lower_runtime": lower.metadata,
        "target_capture_enabled": False,
    })
    directory = OUTPUT / name
    write_json(directory / "summary.json", summary)
    write_jsonl(directory / "episodes.jsonl", records)
    return summary


def replay_gate(summary):
    return (summary["success_rate"] >= .90
            and summary["unsafe_count"] == 0
            and summary["non_finite_count"] == 0)


def final_gate(summary):
    return (replay_gate(summary) and summary["wilson_95_lower"] >= .80)


def run():
    os.environ.pop("DISPLAY", None)
    staged.OUTPUT = OUTPUT
    transport, lower = runtimes()
    result = {
        "task_id": "s1_place", "phase": "P4-R10",
        "training_transitions": 0,
        "transport_checkpoint": transport.metadata,
        "lower_checkpoint": lower.metadata,
        "r9_reset_fix_locked": True,
        "target_capture_required": None,
        "target_capture_implemented": False,
        "final_100_run": False,
    }
    baseline = evaluate_seeded(
        "stage_a_baseline_30", STAGE_A_SEEDS, transport, lower)
    result["stage_a_baseline_30"] = baseline
    baseline_passed = replay_gate(baseline)
    result["p4_r10_baseline_robustness_gate"] = (
        "PASS" if baseline_passed else "FAIL")
    if baseline_passed:
        result["target_capture_required"] = False
        final = evaluate_seeded("final_100", FINAL_SEEDS, transport, lower)
        result["final_100"] = final
        result["final_100_run"] = True
        result["final_gate"] = "PASS" if final_gate(final) else "FAIL"
        if result["final_gate"] == "PASS":
            result["video"] = staged.video(transport, lower)
    else:
        result["target_capture_required"] = True
        result["stage_b_required"] = True
        result["stop_reason"] = (
            "Stage A below 90%; implement and A/B validate Target Capture before Final 100")
    result["isolation"] = staged.isolation()
    complete = bool(result.get("final_gate") == "PASS"
                    and result.get("video", {}).get("passed")
                    and result["isolation"]["passed"])
    result["p4_r10_full_place_gate"] = "PASS" if complete else "FAIL"
    result["p4_s1_place_ppo_gate"] = "PASS" if complete else "FAIL"
    write_json(OUTPUT / "production_isolation.json", result["isolation"])
    files = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(OUTPUT))] = {
                "sha256": sha256(path), "bytes": path.stat().st_size}
    result["files"] = files
    write_json(OUTPUT / "manifest.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
