#!/usr/bin/env python3
"""P4-R9 Full Place evaluation/reset isolation audit (zero training)."""

import hashlib
import json
import math
import os
from pathlib import Path
import random

try:
    from isaacgym import gymapi as _gymapi_import_order  # noqa: F401
except ModuleNotFoundError:
    _gymapi_import_order = None

import torch

import experimental_s1_place_staged as staged
from simulation_core.experimental.manipulation.lower_release_schema import (
    LOWER_RELEASE_POLICY_SCHEMA,
)
from simulation_core.experimental.manipulation.staged_place_env import (
    StagedFullPlaceEnvironment,
)
from simulation_core.ppo.policy import flatten_observation
from simulation_core.task_catalog import task_catalog_document


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "experimental" / "p4_r9_s1_place"
TRANSPORT = (ROOT / "artifacts" / "experimental" / "p4_r7_s1_place" /
             "train_3k" / "checkpoints" / "model_latest.pt")
LOWER = (ROOT / "artifacts" / "experimental" / "p4_2_s1_place" /
         "train_20k" / "checkpoints" / "model_latest.pt")
TRANSPORT_SHA = "d2f7f662efe7d7c978ac241050664dfd8d94e3b192bd68806ffbb69bef7ea587"
LOWER_SHA = "58ddf5e91a29ac087d37dac9094dad77a3596f5e32693fe5ad3f8e78b9dcba33"
RESET_SEED = 20261004
FLOAT_TOLERANCE = 1.0e-6


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


def seed_everything():
    random.seed(RESET_SEED)
    torch.manual_seed(RESET_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RESET_SEED)


def as_list(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (tuple, list)):
        return [float(item) for item in value]
    return value


def runtime_pair():
    if sha256(TRANSPORT) != TRANSPORT_SHA:
        raise ValueError("locked Transport SHA mismatch")
    if sha256(LOWER) != LOWER_SHA:
        raise ValueError("locked Lower SHA mismatch")
    return (
        staged.PolicyRuntime(TRANSPORT, TRANSPORT_SHA, (1, 1, 0, 0), "transport"),
        staged.PolicyRuntime(LOWER, LOWER_SHA, (0, 0, 1, 1), "lower"),
    )


def policy_action(runtime, observation):
    raw = flatten_observation(observation, LOWER_RELEASE_POLICY_SCHEMA).to("cuda:0")
    normalized = runtime.normalizer.normalize(raw, update=False)
    with torch.no_grad():
        action, _, _ = runtime.policy.act(normalized, deterministic=True)
    return (raw.detach().cpu(), normalized.detach().cpu(), action.detach().cpu())


def relative_xyz(snapshot):
    return [float(snapshot["cube_state"][i] - snapshot["ee_state"][i])
            for i in range(3)]


def state_fields(env, snapshot):
    robot = env.scene.robot
    position_targets = getattr(robot, "_position_targets", None)
    return {
        "robot_dof_position": as_list(snapshot["q"]),
        "robot_dof_velocity": as_list(snapshot["dq"]),
        "ee_pose": as_list(snapshot["ee_state"][:7]),
        "ee_linear_angular_velocity": as_list(snapshot["ee_state"][7:13]),
        "cube_pose": as_list(snapshot["cube_state"][:7]),
        "cube_linear_velocity": as_list(snapshot["cube_state"][7:10]),
        "cube_angular_velocity": as_list(snapshot["cube_state"][10:13]),
        "grasp_transform": {
            "actual_cube_minus_ee": relative_xyz(snapshot),
            "reference_cube_minus_ee": as_list(env._reference_relative),
            "reference_cube_rotation": as_list(env._reference_rotation),
        },
        "persistent_cartesian_target": as_list(env._target),
        "target_center": as_list(env.target_center),
        "previous_action": as_list(env._place_previous_action),
        "phase": env.staged_phase,
        "base_phase": env.phase,
        "transport_hold_timer_steps": int(env._transport_ready_steps),
        "transport_hold_timer_seconds": float(
            env._transport_ready_steps * env.control_dt),
        "settle_timer_steps": int(env._settle_steps),
        "maximum_settle_timer_steps": int(env._maximum_settle_steps),
        "release_started": bool(env._release_started),
        "release_latch": bool(env._release_latched),
        "release_trigger_streak": int(env._release_trigger_streak),
        "auto_release_state": {
            "ready": bool(env._auto_release_ready),
            "streak": int(env._auto_release_streak),
            "condition": dict(env._release_condition),
        },
        "workspace_violation_state": getattr(env, "_last_workspace_violation", None),
        "controller_cached_state": {
            "controller_mutable_fields": sorted(
                name for name in vars(env.compositor.controller) if name != "config"),
            "compositor_commanded_jaw": env.compositor._commanded_jaw,
            "robot_position_targets": as_list(
                position_targets.reshape(-1) if position_targets is not None else None),
        },
        "episode_steps": int(env._steps),
        "transport_success_latched": bool(env._transport_success),
        "switched_to_lower_latched": bool(env._switched_to_lower),
        "detached_latched": bool(env._detached),
        "table_contact_latched": bool(env._table_contact),
        "actual_jaw_opened_latched": bool(env._actual_jaw_opened),
        "frozen_xy_target": as_list(env._frozen_xy),
    }


def reset_record(env, observation, runtime, experiment, episode):
    snapshot = env.scene.snapshot()
    raw, normalized, first_action = policy_action(runtime, observation)
    record = {
        "experiment": experiment,
        "episode": episode,
        "reset_seed": RESET_SEED,
        **state_fields(env, snapshot),
        "raw_observation": as_list(raw),
        "normalized_observation": as_list(normalized),
        "first_deterministic_actor_action": as_list(first_action),
        "bootstrap_attempts_for_reset": int(
            env._bootstrap_last.get("attempts_for_reset", 0)),
        "reset_valid": True,
    }
    return record, tuple(float(v) for v in first_action)


def trace_record(env, info, action, control_step):
    snapshot = env.scene.snapshot()
    return {
        "control_step": control_step,
        **state_fields(env, snapshot),
        "actor_action": as_list(action),
        "workspace_violation": bool(info.get("workspace_violation", False)),
        "failure_reason": info.get("failure_reason") or "",
        "transport_target_reached": bool(info.get("transport_target_reached", False)),
        "transport_success": bool(info.get("transport_success", False)),
        "policy_switched_to_lower": bool(info.get("policy_switched_to_lower", False)),
        "detached": bool(info.get("detached", False)),
        "success": bool(info.get("success", False)),
    }


def run_from_reset(env, observation, first_action, transport, lower):
    done = False; step = 0; traces = []; info = {}
    while not done:
        if step == 0:
            action = first_action
        elif env.staged_phase == "TRANSPORT":
            _, _, tensor = policy_action(transport, observation)
            action = tuple(float(v) for v in tensor)
        elif env.staged_phase == "LOWER":
            _, _, tensor = policy_action(lower, observation)
            action = tuple(float(v) for v in tensor)
        else:
            action = (0.0, 0.0, 0.0, 0.0)
        observation, _, done, info = env.step(action)
        step += 1
        traces.append(trace_record(env, info, action, step))
        if step > env.max_steps + 1:
            raise RuntimeError("R9 audit exceeded Full Place horizon")
    final = {
        "steps": step,
        "transport_reached": bool(info.get("transport_target_reached")),
        "transport_success": bool(info.get("transport_success")),
        "switched_lower": bool(info.get("policy_switched_to_lower")),
        "detached": bool(info.get("detached")),
        "success": bool(info.get("success")),
        "failure_reason": info.get("failure_reason") or "SUCCESS",
        "workspace_violation": bool(info.get("workspace_violation")),
    }
    return final, traces


def run_experiment(name, fresh, transport, lower, count=10):
    reset_records = []; episodes = []; all_traces = []
    shared_env = None
    try:
        if not fresh:
            seed_everything(); shared_env = StagedFullPlaceEnvironment(); shared_env.initialize()
        for episode in range(count):
            seed_everything()
            env = StagedFullPlaceEnvironment() if fresh else shared_env
            if fresh:
                env.initialize()
            try:
                observation = env.reset()
                reset, first_action = reset_record(
                    env, observation, transport, name, episode)
                final, traces = run_from_reset(
                    env, observation, first_action, transport, lower)
                reset_records.append(reset)
                episodes.append({"experiment": name, "episode": episode, **final})
                all_traces.append(traces)
            finally:
                if fresh:
                    env.close()
    finally:
        if shared_env is not None:
            shared_env.close()
    directory = OUTPUT / name
    write_jsonl(directory / "reset_records.jsonl", reset_records)
    write_jsonl(directory / "episodes.jsonl", episodes)
    for index, traces in enumerate(all_traces):
        write_jsonl(directory / "traces" / ("episode_%02d.jsonl" % index), traces)
    return reset_records, episodes, all_traces


def flatten(value, prefix=""):
    rows = {}
    if isinstance(value, dict):
        for key in sorted(value):
            child = "%s.%s" % (prefix, key) if prefix else key
            rows.update(flatten(value[key], child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.update(flatten(item, "%s[%d]" % (prefix, index)))
    else:
        rows[prefix] = value
    return rows


IGNORED_COMPARISON_FIELDS = {"episode", "experiment"}


def compare_records(left, right, tolerance=FLOAT_TOLERANCE):
    a = flatten(left); b = flatten(right); differences = []
    for field in sorted(set(a) | set(b)):
        if field.split(".", 1)[0] in IGNORED_COMPARISON_FIELDS:
            continue
        av, bv = a.get(field), b.get(field)
        if isinstance(av, (int, float)) and not isinstance(av, bool) \
                and isinstance(bv, (int, float)) and not isinstance(bv, bool):
            delta = abs(float(av) - float(bv))
            if delta > tolerance:
                differences.append({"field": field, "left": av, "right": bv,
                                    "absolute_delta": delta})
        elif av != bv:
            differences.append({"field": field, "left": av, "right": bv,
                                "absolute_delta": None})
    return differences


def consistency(records):
    baseline = records[0]
    pairs = []
    for index, record in enumerate(records[1:], 1):
        differences = compare_records(baseline, record)
        pairs.append({"episode": index, "difference_count": len(differences),
                      "first_difference": differences[0] if differences else None,
                      "maximum_float_delta": max(
                          (d["absolute_delta"] for d in differences
                           if d["absolute_delta"] is not None), default=0.0)})
    return {"all_consistent": all(not row["difference_count"] for row in pairs),
            "compared_to_episode": 0, "pairs": pairs}


def cross_compare(a_resets, b_resets, a_traces, b_traces):
    episode_rows = []; first = None
    for episode in range(10):
        reset_diff = compare_records(a_resets[episode], b_resets[episode])
        row = {"episode": episode, "reset_difference_count": len(reset_diff),
               "reset_first_difference": reset_diff[0] if reset_diff else None}
        if reset_diff and first is None:
            first = {"episode": episode, "control_step": 0, **reset_diff[0]}
        trace_first = None
        for index, (left, right) in enumerate(zip(a_traces[episode], b_traces[episode]), 1):
            diff = compare_records(left, right)
            if diff:
                trace_first = {"control_step": index, **diff[0]}
                if first is None:
                    first = {"episode": episode, **trace_first}
                break
        if trace_first is None and len(a_traces[episode]) != len(b_traces[episode]):
            trace_first = {"control_step": min(len(a_traces[episode]),
                                               len(b_traces[episode])) + 1,
                           "field": "trace_length", "left": len(a_traces[episode]),
                           "right": len(b_traces[episode]), "absolute_delta": None}
            if first is None:
                first = {"episode": episode, **trace_first}
        row["trace_first_difference"] = trace_first
        episode_rows.append(row)
    return {"all_pairs_consistent": first is None,
            "first_divergence": first, "episodes": episode_rows}


def experiment_summary(records, episodes):
    outcomes = [row["failure_reason"] for row in episodes]
    return {
        "episodes": 10,
        "success_count": sum(row["success"] for row in episodes),
        "workspace_violation_count": sum(row["workspace_violation"] for row in episodes),
        "transport_reached_count": sum(row["transport_reached"] for row in episodes),
        "transport_success_count": sum(row["transport_success"] for row in episodes),
        "switched_lower_count": sum(row["switched_lower"] for row in episodes),
        "detached_count": sum(row["detached"] for row in episodes),
        "outcome_sequence": outcomes,
        "even_outcomes": outcomes[0::2], "odd_outcomes": outcomes[1::2],
        "reset_consistency": consistency(records),
    }


def isolation():
    catalog = task_catalog_document()
    experimental = [row["id"] for row in catalog["tasks"]
                    if row["id"] in ("s1_grasp", "s1_lift", "s1_place")]
    return {"production_task_count": len(catalog["tasks"]),
            "experimental_registered": experimental,
            "passed": len(catalog["tasks"]) == 7 and not experimental}


def run():
    os.environ.pop("DISPLAY", None)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    transport, lower = runtime_pair()
    write_json(OUTPUT / "config.json", {
        "reset_seed": RESET_SEED, "float_tolerance": FLOAT_TOLERANCE,
        "deterministic_policy": True, "training_transitions": 0,
        "transport_checkpoint": transport.metadata,
        "lower_checkpoint": lower.metadata,
        "r8_system_locked": True,
    })
    a_reset, a_episodes, a_traces = run_experiment(
        "same_env_10", False, transport, lower)
    b_reset, b_episodes, b_traces = run_experiment(
        "fresh_env_10", True, transport, lower)
    comparison = cross_compare(a_reset, b_reset, a_traces, b_traces)
    a_summary = experiment_summary(a_reset, a_episodes)
    b_summary = experiment_summary(b_reset, b_episodes)
    periodic = {
        "same_env_even_outcomes_equal": len(set(a_summary["even_outcomes"])) == 1,
        "same_env_odd_outcomes_equal": len(set(a_summary["odd_outcomes"])) == 1,
        "same_env_even_odd_differ": (
            a_summary["even_outcomes"] != a_summary["odd_outcomes"]),
        "fresh_env_all_outcomes_equal": len(set(b_summary["outcome_sequence"])) == 1,
    }
    # Leakage is established when a reused simulator's post-reset state is not
    # isolated while independently recreated simulators remain consistent.
    leakage_this_run = (not a_summary["reset_consistency"]["all_consistent"]
                        and b_summary["reset_consistency"]["all_consistent"])
    pre_fix_path = OUTPUT / "pre_fix_summary.json"
    pre_fix = (json.loads(pre_fix_path.read_text(encoding="utf-8"))
               if pre_fix_path.exists() else None)
    leakage_found = bool(leakage_this_run or (
        pre_fix and pre_fix.get("reset_leakage_found")))
    result = {
        "same_env_10": a_summary, "fresh_env_10": b_summary,
        "cross_experiment_comparison": comparison,
        "periodicity": periodic,
        "reset_leakage_found": leakage_found,
        "leakage_detected_in_current_run": leakage_this_run,
        "reset_isolation_clean": not leakage_this_run,
        "reset_fix": {
            "applied": pre_fix is not None,
            "description": (
                "one discarded full reset priming pass before the episode-visible reset"),
            "control_architecture_modified": False,
        },
        "pre_fix_summary_file": (
            "pre_fix_summary.json" if pre_fix is not None else None),
        "training_transitions": 0,
        "final_100_run": False,
        "isolation": isolation(),
    }
    result["p4_r9_reset_isolation_gate"] = (
        "PASS" if result["isolation"]["passed"]
        and result["reset_isolation_clean"] else "FAIL")
    write_json(OUTPUT / "comparison.json", comparison)
    write_json(OUTPUT / "summary.json", result)
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
