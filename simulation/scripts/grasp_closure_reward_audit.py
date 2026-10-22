#!/usr/bin/env python3
"""Reproducible offline closure-reward audit for P2-R2."""

import argparse
import json
from pathlib import Path

from simulation_core.experimental.manipulation.grasp_schema import (
    GraspRewardConfig, compute_grasp_reward, compute_grasp_reward_v2,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "experimental" / "p2_r2_s1_grasp" / "reward_audit"
COMMANDS = (.050, .048, .046, .044, .042, .041, .040, .038,
            .036, .034, .032, .030, .028, .026, .025)
CONTROLLED_GAPS = (0., 0., 0., 0., 0., 0., .0005, .001, .0015,
                   .002, .002, .002, .002, .002, .002)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def grip_action(command):
    return 2.0 * (command - .025) / (.05 - .025) - 1.0


def v2_row(command, previous, gap, candidate_event=False):
    grip = grip_action(command)
    saturation = .25 if abs(grip) >= .98 else 0.0
    total, terms = compute_grasp_reward_v2(
        approach_progress=0., alignment_progress=0.,
        descend_progress=0., xy_aligned=True,
        closure_credit=max(0., previous - command), closure_gated=True,
        candidate_event=candidate_event, verified_event=False,
        action=(0., 0., 0., grip), saturated=saturation,
        clamped=False, failure=False,
    )
    return {
        "commanded_jaw": command, "actual_jaw": command + gap,
        "close_fraction": (.05 - command) / (.05 - .025),
        "blockage_gap": gap, "normalized_grip_action": grip,
        "candidate_event": candidate_event, "total_reward": total,
        **terms,
    }


def comparison_audit():
    rows = []
    previous_command = .05
    previous_close = 0.0
    previous_blockage = 0.0
    candidate_crossed = False
    for command, gap in zip(COMMANDS, CONTROLLED_GAPS):
        grip = grip_action(command)
        saturation = .25 if abs(grip) >= .98 else 0.0
        close_fraction = (.05 - command) / (.05 - .025)
        blockage = min(1.0, max(0.0, gap / .002))
        candidate = gap >= .002 and not candidate_crossed
        v2_total, v2_terms = compute_grasp_reward_v2(
            approach_progress=0., alignment_progress=0., descend_progress=0.,
            xy_aligned=True, closure_credit=max(0., previous_command - command),
            closure_gated=True, candidate_event=candidate, verified_event=False,
            action=(0., 0., 0., grip), saturated=saturation,
            clamped=False, failure=False,
        )
        v3_total, v3_terms = compute_grasp_reward(
            config=GraspRewardConfig(), approach_progress=0., alignment_progress=0.,
            descend_progress=0., xy_aligned=True,
            closure_potential_progress=close_fraction - previous_close,
            blockage_potential_progress=blockage - previous_blockage,
            candidate_event=candidate, verified_event=False,
            action=(0., 0., 0., grip), saturated=saturation,
            clamped=False, failure=False,
        )
        rows.append({
            "commanded_jaw": command, "actual_jaw": command + gap,
            "close_fraction": close_fraction, "blockage_gap": gap,
            "candidate_event": candidate,
            "v2_closure_reward": v2_terms["valid_closure"],
            "v3_closure_progress": v3_terms["closure_progress"],
            "v3_blockage_progress": v3_terms["blockage_progress"],
            "total_v2": v2_total, "total_v3": v3_total,
        })
        candidate_crossed = candidate_crossed or candidate
        previous_command = min(previous_command, command)
        previous_close = close_fraction
        previous_blockage = blockage
    return {
        "audit_schema": "p2-r2-reward-v2-v3-comparison/v1",
        "K_close": GraspRewardConfig().closure_progress,
        "K_block": GraspRewardConfig().blockage_progress,
        "rows": rows,
        "summary": {
            "v2_full_close_shaping": sum(row["v2_closure_reward"] for row in rows),
            "v3_full_close_shaping": sum(row["v3_closure_progress"] for row in rows),
            "v3_full_blockage_shaping": sum(row["v3_blockage_progress"] for row in rows),
            "v3_total_directional_shaping": sum(
                row["v3_closure_progress"] + row["v3_blockage_progress"] for row in rows),
        },
    }


def v2_audit():
    no_contact = []
    contact = []
    previous = .05
    candidate_crossed = False
    for command, gap in zip(COMMANDS, CONTROLLED_GAPS):
        candidate = gap >= .002 and not candidate_crossed
        no_contact.append(v2_row(command, previous, 0., False))
        contact.append(v2_row(command, previous, gap, candidate))
        candidate_crossed = candidate_crossed or candidate
        previous = min(previous, command)
    return {
        "audit_schema": "p2-r2-reward-v2-closure-audit/v1",
        "controlled_conditions": {
            "finite": True, "in_corridor": True, "xy_aligned": True,
            "at_grasp_height": True, "premature_push": False,
            "unsafe": False, "other_progress_terms": 0.0,
        },
        "commands_m": list(COMMANDS),
        "case_A_no_contact": no_contact,
        "case_B_controlled_contact": contact,
        "recorded_r1": {
            "episodes": 50,
            "mean_commanded_jaw": .04096736182272434,
            "mean_actual_jaw": .04013918291777372,
            "mean_actual_minus_command": -.0008281789049506202,
            "jaw_blocked_count": 0,
        },
        "summary": {
            "full_close_cumulative_v2_closure": sum(r["valid_closure"] for r in no_contact),
            "closure_to_0p041": 40. * (.05 - .041),
            "closure_remaining_after_0p041": 40. * (.041 - .025),
            "blockage_precursor_reward_before_candidate": 0.0,
            "hover_closure_reward": 0.0,
            "finding": (
                "No mathematical positive hover reward, but a progress plateau with no "
                "blockage precursor: after commanded motion stops, closure reward is zero; "
                "the next physical evidence signal is the discontinuous candidate event."
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("v2", "compare"), required=True)
    args = parser.parse_args()
    result = v2_audit() if args.mode == "v2" else comparison_audit()
    path = OUTPUT / ("reward_v2_closure_audit.json" if args.mode == "v2"
                     else "reward_v2_vs_v3_closure_audit.json")
    write_json(path, result)
    print("P2_R2_REWARD_AUDIT=%s" % json.dumps(
        {"path": str(path), **result["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
