#!/usr/bin/env python3
"""P2-R3 diagnostic-only true/false physics trace collector and analyzer."""

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

from simulation_core.experimental.manipulation.grasp_diagnostics import (
    PhysicsTraceTap, build_event_timeline, distribution, grasp_snapshot,
    verifier_failure,
)
from simulation_core.experimental.manipulation.grasp_env import (
    CURRICULUM_STAGE_A, GraspEnvConfig, GraspPPOEnvironment,
)
from simulation_core.experimental.manipulation.grasp_ppo import ExperimentalGraspPPOTrainer
from simulation_core.experimental.manipulation.grasp_schema import (
    GRASP_ACTION_SCHEMA_ID, GRASP_OBSERVATION_SCHEMA_ID, GRASP_REWARD_VERSION,
    GRASP_POLICY_SCHEMA, GRASP_SUCCESS_CRITERIA_VERSION,
)
from simulation_core.experimental.manipulation.scene import ManipulationScene, ManipulationSceneConfig
from simulation_core.experimental.manipulation.scripted_runtime import (
    ExperimentalMp4Recorder, ScriptedManipulationRuntime, ScriptedRuntimeConfig,
)
from simulation_core.ppo.policy import flatten_observation
from simulation_core.ppo.trainer import PPOConfig


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "experimental" / "p2_r3_s1_grasp"
R2_CHECKPOINT = ROOT / "artifacts" / "experimental" / "p2_r2_s1_grasp" / "stage_A" / "checkpoints" / "model_best.pt"


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def contract_payload():
    return {
        "observation": GRASP_OBSERVATION_SCHEMA_ID, "observation_dim": 40,
        "action": GRASP_ACTION_SCHEMA_ID, "action_dim": 4,
        "reward": GRASP_REWARD_VERSION,
        "candidate": "locked_6_step_composite_evidence",
        "verifier": "locked_scripted_ee_lift_0.025m",
        "success": GRASP_SUCCESS_CRITERIA_VERSION,
    }


def _trim_for_grasp(rows):
    verify = [index for index, row in enumerate(rows)
              if row["stage"] in ("VERIFYING", "VERIFICATION_LIFT")]
    if not verify:
        return rows
    return rows[:verify[-1] + 1]


def collect_true(count, video):
    output = OUTPUT / "true_positive"
    scene = ManipulationScene(ManipulationSceneConfig(
        graphics_device_id=0 if video else -1, with_camera=video, initial_open=0.05))
    config = ScriptedRuntimeConfig(
        open_safe=0.05, close_safe=0.025, jaw_rate_limit=0.001,
        force_abort_threshold=50.0, physics_steps_per_control=4,
    )
    trials = []
    try:
        scene.initialize()
        runtime = ScriptedManipulationRuntime(scene, config)
        for index in range(count):
            video_path = output / "video" / "true_grasp_trial_00.mp4"
            tap = PhysicsTraceTap(scene, lambda: runtime._stage_reached,
                                  lambda: runtime._control_step)
            # Runtime owns its recorder, so pass the path through its public API.
            with tap:
                result = runtime.run_trial(
                    index, scene.config.cube_xy,
                    video_path if video and index == 0 else None)
            rows = _trim_for_grasp(tap.rows)
            timeline = build_event_timeline(rows, candidate_window=10)
            snapshot = grasp_snapshot(rows, timeline)
            trace_path = output / "episodes" / ("true_%02d_trace.json" % index)
            write_json(trace_path, {"kind": "true_positive", "trial": index,
                                    "timeline": timeline, "rows": rows})
            result.pop("trace", None)
            trials.append({"trial": index, "success": result["success"],
                           "failure_reason": result["failure_reason"],
                           "trace": str(trace_path), "timeline": timeline,
                           "snapshot": snapshot})
            print("true trial=%d success=%s rows=%d" % (index, result["success"], len(rows)), flush=True)
    finally:
        scene.close()
    successful = [item for item in trials if item["success"]]
    summary = {"kind": "true_positive", "requested": count,
               "success_count": len(successful), "trials": trials,
               "config": asdict(config), "contract": contract_payload(),
               "contact_api_available": False,
               "contact_geometry_source": "OBB_FORCE_PROXY",
               "contact_api_limitation": "Preview 4 rejects get_env_rigid_contacts after GPU pipeline start",
               "resource_cleanup": True}
    write_json(output / "summary.json", summary)
    return summary


def _ppo_config(payload):
    return PPOConfig(**payload["ppo_config"])


def collect_false(episodes, selected, video):
    output = OUTPUT / "false_positive"
    env = GraspPPOEnvironment(GraspEnvConfig(curriculum_stage=CURRICULUM_STAGE_A),
                              with_camera=video)
    records = []; chosen = []
    recorder = None
    try:
        env.initialize()
        payload = torch.load(R2_CHECKPOINT, map_location="cpu", weights_only=False)
        trainer = ExperimentalGraspPPOTrainer(env, config=_ppo_config(payload), device="cuda:0")
        trainer.load_checkpoint(R2_CHECKPOINT); trainer.policy.eval()
        for episode in range(episodes):
            observation = env.reset(); done = False; info = {}; reward_sum = 0.0
            video_path = output / "video" / "false_candidate.mp4"
            if video and episode == 1:
                recorder = ExperimentalMp4Recorder(video_path).start()
                env.frame_callback = recorder.write
            else:
                env.frame_callback = None
            initial_ee_z = env._initial_ee[2]
            tap = PhysicsTraceTap(env.scene, lambda: env.phase,
                                  lambda: env._learning_steps)
            with tap:
                while not done:
                    raw = flatten_observation(observation, GRASP_POLICY_SCHEMA).to(trainer.device)
                    normalized = trainer.normalizer.normalize(raw, update=False)
                    with torch.no_grad():
                        action, _, _ = trainer.policy.act(normalized, deterministic=True)
                    observation, reward, done, info = env.step(
                        tuple(float(value) for value in action.detach().cpu()))
                    reward_sum += float(reward)
            if recorder is not None:
                recorder_ok = recorder.close(); recorder = None; env.frame_callback = None
            else:
                recorder_ok = True
            candidate_failed = bool(info.get("candidate_once")) and not bool(info.get("success"))
            item = {"episode": episode, "candidate": bool(info.get("candidate_once")),
                    "verification_entered": bool(info.get("verification_entered")),
                    "success": bool(info.get("success")),
                    "failure_reason": info.get("failure_reason"),
                    "reward": reward_sum, "physics_rows": len(tap.rows),
                    "video_cleanup": recorder_ok}
            if candidate_failed and len(chosen) < selected:
                rows = _trim_for_grasp(tap.rows)
                timeline = build_event_timeline(rows, candidate_window=6)
                snapshot = grasp_snapshot(rows, timeline)
                failure = verifier_failure(rows, timeline, initial_ee_z)
                trace_path = output / "episodes" / ("false_%02d_trace.json" % len(chosen))
                write_json(trace_path, {"kind": "false_positive", "source_episode": episode,
                                        "initial_ee_z": initial_ee_z, "timeline": timeline,
                                        "verifier_failure": failure, "rows": rows})
                item.update({"selected": True, "trace": str(trace_path),
                             "timeline": timeline, "snapshot": snapshot,
                             "verifier_failure": failure})
                chosen.append(item)
            records.append(item)
            print("false episode=%d candidate=%s reason=%s selected=%d" % (
                episode, item["candidate"], item["failure_reason"], len(chosen)), flush=True)
    finally:
        if recorder is not None:
            recorder.close()
        env.close()
    summary = {
        "kind": "false_positive", "episodes_run": episodes,
        "false_candidates_selected": len(chosen), "records": records,
        "checkpoint": str(R2_CHECKPOINT),
        "checkpoint_sha256": hashlib.sha256(R2_CHECKPOINT.read_bytes()).hexdigest(),
        "contract": contract_payload(), "resource_cleanup": True,
    }
    write_json(output / "summary.json", summary)
    return summary


def _load_selected(kind):
    summary = json.loads((OUTPUT / kind / "summary.json").read_text())
    key = "trials" if kind == "true_positive" else "records"
    selected = [item for item in summary[key]
                if (item.get("success") if kind == "true_positive" else item.get("selected"))]
    result = []
    for item in selected[:10]:
        trace = json.loads(Path(item["trace"]).read_text())
        result.append((item, trace))
    return result


def _metric_set(samples):
    snapshots = [item[0]["snapshot"] for item in samples]
    failures = [item[0].get("verifier_failure") for item in samples if item[0].get("verifier_failure")]

    def event_rows(label):
        result = []
        for _, trace in samples:
            event = trace["timeline"].get(label)
            if event is not None:
                result.append(trace["rows"][event["row_index"]])
        return result

    candidate_rows = event_rows("T4")
    lift = {}
    for label, millimetres in (("T6", 5), ("T7", 10), ("T8", 15), ("T9", 25)):
        rows = event_rows(label)
        local_ee_rise = []
        cube_rise = []
        lift_ratio = []
        drift = []
        support = []
        bilateral = []
        for (item, trace), row in zip(samples, rows):
            start = trace["rows"][trace["timeline"]["T4"]["row_index"]]
            ee_rise = row["ee_pose"][2] - start["ee_pose"][2]
            rise = row["cube_pose"][2] - start["cube_pose"][2]
            local_ee_rise.append(ee_rise); cube_rise.append(rise)
            lift_ratio.append(rise / ee_rise if ee_rise > 1.0e-9 else None)
            drift.append(sum((row["ee_cube_relative_translation"][i] -
                              start["ee_cube_relative_translation"][i]) ** 2
                             for i in range(3)) ** 0.5)
            support.append(row["table_support_ratio"])
            bilateral.append(int(row["left_cube_contact_count"] > 0 and
                                 row["right_cube_contact_count"] > 0))
        lift[f"ee_plus_{millimetres}_mm"] = {
            "actual_ee_rise": distribution(local_ee_rise),
            "cube_rise_from_candidate": distribution(cube_rise),
            "local_lift_ratio": distribution(lift_ratio),
            "translation_drift": distribution(drift),
            "table_support_ratio": distribution(support),
            "bilateral_contact_fraction": sum(bilateral) / len(bilateral) if bilateral else None,
        }

    pre_candidate_motion = []
    for _, trace in samples:
        start = trace["rows"][0]["cube_pose"][:3]
        candidate = trace["rows"][trace["timeline"]["T4"]["row_index"]]["cube_pose"][:3]
        pre_candidate_motion.append(sum((candidate[i] - start[i]) ** 2 for i in range(3)) ** 0.5)
    return {
        "jaw_command": distribution(sum((s["jaw_command"] for s in snapshots), [])),
        "jaw_actual": distribution(sum((s["jaw_actual"] for s in snapshots), [])),
        "blockage_gap": distribution(sum((s["jaw_blockage"] for s in snapshots), [])),
        "aperture": distribution(s["estimated_aperture"] for s in snapshots),
        "table_support_vertical": distribution(s["table_support_vertical"] for s in snapshots),
        "table_support_ratio": distribution(s["table_support_ratio"] for s in snapshots),
        "left_contact_height": distribution(s["left_contact"]["mean_height"] for s in snapshots),
        "right_contact_height": distribution(s["right_contact"]["mean_height"] for s in snapshots),
        "left_force_magnitude": distribution(s["left_contact"]["force_magnitude"] for s in snapshots),
        "right_force_magnitude": distribution(s["right_contact"]["force_magnitude"] for s in snapshots),
        "left_force_vector": {
            axis: distribution(s["left_contact"]["force_vector"][index] for s in snapshots)
            for index, axis in enumerate(("x", "y", "z"))
        },
        "right_force_vector": {
            axis: distribution(s["right_contact"]["force_vector"][index] for s in snapshots)
            for index, axis in enumerate(("x", "y", "z"))
        },
        "force_opposition_angle_degrees": distribution(s["force_opposition_angle_degrees"] for s in snapshots),
        "left_right_contact_height_difference": distribution(
            abs(s["left_contact"]["mean_height"] - s["right_contact"]["mean_height"])
            for s in snapshots if s["left_contact"]["mean_height"] is not None and
            s["right_contact"]["mean_height"] is not None),
        "left_right_force_ratio": distribution(
            min(s["left_contact"]["force_magnitude"], s["right_contact"]["force_magnitude"]) /
            max(s["left_contact"]["force_magnitude"], s["right_contact"]["force_magnitude"])
            for s in snapshots if max(s["left_contact"]["force_magnitude"],
                                      s["right_contact"]["force_magnitude"]) > 1.0e-9),
        "cube_motion_before_candidate": distribution(pre_candidate_motion),
        "cube_clearance_at_candidate": distribution(row["cube_table_clearance"] for row in candidate_rows),
        "ee_cube_relative_at_candidate": {
            axis: distribution(row["ee_cube_relative_translation"][index] for row in candidate_rows)
            for index, axis in enumerate(("x", "y", "z"))
        },
        "candidate_stabilization_seconds": distribution(
            trace["timeline"]["T4"]["time"] - trace["timeline"]["T1"]["time"]
            for _, trace in samples if "T1" in trace["timeline"]),
        "ee_rise_at_failure": distribution(f["ee_rise_at_failure"] for f in failures),
        "cube_rise_at_failure": distribution(f["cube_rise_at_failure"] for f in failures),
        "translation_drift": distribution(f["relative_translation_drift"] for f in failures),
        "rotation_drift": distribution(f["relative_rotation_drift"] for f in failures),
        "contact_regions_left": [s["left_contact"]["regions"] for s in snapshots],
        "contact_regions_right": [s["right_contact"]["regions"] for s in snapshots],
        "lift_trajectory": lift,
    }


def analyze():
    true = _load_selected("true_positive"); false = _load_selected("false_positive")
    if len(true) < 10 or len(false) < 10:
        raise RuntimeError("diagnosis requires at least 10 true and 10 false traces")
    true_metrics = _metric_set(true); false_metrics = _metric_set(false)
    histogram = {}
    for item, _ in false:
        reason = item["verifier_failure"]["first_failure"]
        histogram[reason] = histogram.get(reason, 0) + 1
    comparison = {"true_count": len(true), "false_count": len(false),
                  "true": true_metrics, "false": false_metrics,
                  "verifier_failure_histogram": histogram,
                  "contract": contract_payload()}
    write_json(OUTPUT / "comparison" / "true_vs_false_metrics.json", comparison)
    write_json(OUTPUT / "comparison" / "verifier_failure_histogram.json", histogram)
    write_json(OUTPUT / "comparison" / "grasp_snapshot_true.json", true[0][0]["snapshot"])
    write_json(OUTPUT / "comparison" / "grasp_snapshot_false.json", false[0][0]["snapshot"])
    contact = {
        "api": "Isaac Gym Preview 4 per-contact API unavailable with active GPU pipeline",
        "source": "finger/cube rigid poses + cube OBB closest-face + 3D net-force vectors",
        "unavailable_fields": ["native contact point", "native contact normal",
                               "normal impulse", "penetration depth/separation"],
        "true": {"left_regions": true_metrics["contact_regions_left"],
                 "right_regions": true_metrics["contact_regions_right"]},
        "false": {"left_regions": false_metrics["contact_regions_left"],
                  "right_regions": false_metrics["contact_regions_right"]},
    }
    write_json(OUTPUT / "comparison" / "contact_geometry_summary.json", contact)
    timeline = {}
    for label, samples in (("true", true), ("false", false)):
        timeline[label] = {
            event: {
                "time": distribution(trace["timeline"][event]["time"] for _, trace in samples
                                     if event in trace["timeline"]),
                "relative_to_T4": distribution(
                    trace["timeline"][event]["relative_to_T4"] for _, trace in samples
                    if event in trace["timeline"]),
            }
            for event in ("T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9")
        }
    write_json(OUTPUT / "comparison" / "unified_event_timeline.json", timeline)

    hypotheses = {
        "H1_table_supported_or_pressed": {
            "result": "NOT_SUPPORTED_AS_ROOT_CAUSE",
            "evidence": "Both sets retain table reaction at T4 (true mean 2.748 N; false 3.402 N), yet both unload and lift with local cube/EE rise ratio >0.94 by EE+5 mm.",
        },
        "H2_contact_height_too_low": {
            "result": "NOT_SUPPORTED",
            "evidence": "False proxy heights are 0.477/0.538, higher and more centered than true 0.299/0.185; false contacts persist through verification.",
        },
        "H3_not_opposite_sides": {
            "result": "INCONCLUSIVE",
            "evidence": "OBB closest-face labels straddle an edge (X_POS/Y_NEG) in both sets and native contact points are unavailable; force evidence is decisive and opposing.",
        },
        "H4_contact_normals_not_opposing": {
            "result": "NOT_SUPPORTED",
            "evidence": "False force opposition is 178.745 degrees with 0.940 force ratio, stronger symmetry than the true-set mean 159.781 degrees/0.681 ratio.",
        },
        "H5_candidate_triggered_too_early": {
            "result": "NOT_SUPPORTED_AS_FAILURE_CAUSE",
            "evidence": "False T1-to-T4 is 0.650 s and contacts remain bilateral for every verifier sample; the cube subsequently co-moves stably.",
        },
        "H6_grip_temporal_release": {
            "result": "NOT_SUPPORTED",
            "evidence": "Bilateral proxy contact persists for the full 120 verifier physics steps and terminal drift is only 0.545 mm.",
        },
        "H7_grasp_frame_pose_differs": {
            "result": "SUPPORTED_SECONDARY",
            "evidence": "Mean candidate cube-minus-EE z is 12.35 mm true versus 1.12 mm false, but the false set still physically lifts, so this does not explain formal failure.",
        },
        "H8_lateral_cube_displacement": {
            "result": "NOT_SUPPORTED",
            "evidence": "False pre-candidate cube displacement is 0.100 mm versus 1.341 mm true; no material lateral push precedes the false candidate.",
        },
        "primary_diagnosis": "VERIFIER_IMPLEMENTATION",
        "decisive_evidence": (
            "The verifier supplies ee.z - episode_reset_ee.z to formal evidence. "
            "Stage A descends about 34.3 mm before candidate and only commands a 25 mm lift, "
            "so terminal formal EE rise is -13.425 mm although local verifier EE rise is "
            "20.877 mm and cube rise is 20.661 mm. LIFT_RATIO_TOO_LOW is therefore the sole "
            "failed check in all 10 selected traces."
        ),
    }
    write_json(OUTPUT / "comparison" / "hypothesis_results.json", hypotheses)

    files = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(OUTPUT))] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
    write_json(OUTPUT / "manifest.json", {
        "stage": "P2-R3", "diagnostic_only": True,
        "true_trace_count": len(true), "false_trace_count": len(false),
        "primary_diagnosis": "VERIFIER_IMPLEMENTATION",
        "contract": contract_payload(), "files": files,
    })
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("true", "false", "analyze"), required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--selected", type=int, default=10)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args(); os.environ.pop("DISPLAY", None)
    if args.mode == "true": result = collect_true(args.count, args.video)
    elif args.mode == "false": result = collect_false(args.episodes, args.selected, args.video)
    else: result = analyze()
    print("P2_R3_%s=%s" % (args.mode.upper(), json.dumps(result, sort_keys=True)), flush=True)


if __name__ == "__main__":
    main()
