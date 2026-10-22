#!/usr/bin/env python3
"""P4-R6 post-repair Full Place replay, final gate, and manifest."""

import hashlib
import json
import os
from pathlib import Path

import experimental_s1_place_staged as staged


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "experimental" / "p4_r6_s1_place"


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def finalize(result):
    result["isolation"] = staged.isolation()
    write_json(OUTPUT / "production_isolation.json", result["isolation"])
    passed = result.get("final_gate") == "PASS" and result["isolation"]["passed"]
    result["p4_r6_integration_gate"] = "PASS" if passed else "FAIL"
    result["p4_s1_place_ppo_gate"] = "PASS" if passed else "FAIL"
    files = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(OUTPUT))] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size}
    result["files"] = files; write_json(OUTPUT / "manifest.json", result); return result


def run():
    os.environ.pop("DISPLAY", None); OUTPUT.mkdir(parents=True, exist_ok=True)
    staged.OUTPUT = OUTPUT
    transport = staged.PolicyRuntime(staged.TRANSPORT, staged.TRANSPORT_SHA,
                                     (1,1,0,0), "transport")
    lower = staged.PolicyRuntime(staged.LOWER, staged.LOWER_SHA, (0,0,1,1), "lower")
    ab = json.loads((OUTPUT / "transport_ab_20" / "summary.json").read_text())
    result = {"task_id": "s1_place", "training_transitions": 0,
              "diagnosis_classification": "PHASE_STATE_MISMATCH",
              "integration_bug_found": True, "transport_ab_20": ab,
              "transport_checkpoint": transport.metadata,
              "lower_checkpoint": lower.metadata,
              "episode_horizon_seconds": 10.0}
    replay = staged.evaluate("full_place_replay_20", 20, transport, lower)
    result["full_place_replay_20"] = replay
    if not staged.passed(replay):
        funnel = replay["behavior_funnel"]
        ordered = ("transport_target_reached", "policy_switched_to_lower",
                   "legal_release_height", "auto_release_ready", "release_latched",
                   "actual_jaw_opened", "detached", "table_contact", "settled_1p0s", "success")
        result["minimal_blocker_stage"] = next((name for name in ordered
                                                if funnel[name] < 18), "replay_gate")
        result["final_100_run"] = False; result["video_run"] = False
        return finalize(result)
    final = staged.evaluate("final_100", 100, transport, lower)
    result["final_100"] = final
    result["final_gate"] = "PASS" if staged.passed(final, final=True) else "FAIL"
    if result["final_gate"] == "PASS": result["video"] = staged.video(transport, lower)
    return finalize(result)


if __name__ == "__main__": print(json.dumps(run(), indent=2, sort_keys=True))
