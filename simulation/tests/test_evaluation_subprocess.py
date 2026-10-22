import json
import math
import tempfile
import unittest
from pathlib import Path

from simulation_core.evaluation_subprocess import (
    EVALUATION_EPISODES,
    EVALUATION_REPLAY_FPS,
    EVALUATION_VIDEO_PATH,
    EVALUATION_VIDEO_PARTIAL_SUFFIX,
    EvaluationInfrastructureError,
    ProductionEvaluationLauncher,
    evaluation_contract,
    validate_evaluation_outputs,
)
from simulation_core.evaluation import wilson_confidence_interval


MODEL_SHA = "a" * 64


def valid_summary(success_count=0, selected_seed=None, video=None):
    lower, upper = wilson_confidence_interval(success_count, 100)
    return {
        "episode_count": 100,
        "success_count": success_count,
        "success_rate": success_count / 100,
        "confidence_interval_95": {
            "method": "wilson", "lower": lower, "upper": upper,
        },
        "timeout_rate": 1.0 - success_count / 100,
        "failure_rate": 0.0,
        "mean_episode_length": 256.0,
        "mean_episode_return": 0.0,
        "mean_final_distance": 0.2,
        "mean_minimum_distance": 0.1,
        "mean_object_displacement": 0.0,
        "mean_action_norm": 0.0,
        "action_saturation_rate": 0.0,
        "checkpoint_sha256": MODEL_SHA,
        "selected_success_seed": selected_seed,
        "evaluation_video": video or {
            "status": "unavailable", "failure_reason": "replay_not_attempted",
        },
    }


def write_outputs(run_root, summary=None, successful_seeds=()):
    summary = summary or valid_summary(len(successful_seeds), min(successful_seeds, default=None))
    output = run_root / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
    with (output / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for seed in range(EVALUATION_EPISODES):
            success = seed in successful_seeds
            stream.write(json.dumps({
                "seed": seed, "success": success, "timeout": not success,
                "failure": False, "episode_return": 0.0,
                "episode_length": 256, "final_distance": 0.2,
                "minimum_distance": 0.1, "object_displacement": 0.0,
                "action_norm": 0.0, "action_saturation_rate": 0.0,
            }) + "\n")


class FakeProcess:
    def __init__(self, return_code=0):
        self.return_code = return_code
        self.terminated = False

    def wait(self, timeout=None):
        return self.return_code

    def poll(self):
        return self.return_code

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True


class EvaluationSubprocessTestCase(unittest.TestCase):
    def prepared_run(self, directory, successful_seeds=()):
        artifacts = Path(directory) / "artifacts"
        run_root = artifacts / "project" / "run"
        checkpoint = run_root / "checkpoints/model_latest.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        write_outputs(run_root, successful_seeds=successful_seeds)
        return artifacts, run_root

    def test_headless_suite_uses_one_fresh_process_and_no_video_for_zero_success(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts, run_root = self.prepared_run(directory)
            calls = []

            def popen(command, **kwargs):
                calls.append((command, kwargs))
                return FakeProcess()

            summary = ProductionEvaluationLauncher(
                artifacts, popen_factory=popen
            ).run("project", "run")

        self.assertEqual(0.0, summary["success_rate"])
        self.assertEqual("no_successful_seed", summary["evaluation_video"]["failure_reason"])
        self.assertEqual(1, len(calls))
        command, kwargs = calls[0]
        self.assertEqual(("--run-root", str(run_root), "--episodes", "100"), command[2:])
        self.assertNotIn("--replay-seed", command)
        self.assertFalse(kwargs["shell"])

    def test_minimum_success_seed_is_replayed_in_second_isolated_process(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts, run_root = self.prepared_run(directory, (7, 3, 20))
            calls = []

            def popen(command, **kwargs):
                calls.append(command)
                if "--replay-seed" in command:
                    video = run_root / EVALUATION_VIDEO_PATH
                    video.parent.mkdir(parents=True, exist_ok=True)
                    video.write_bytes(b"mp4")
                return FakeProcess()

            summary = ProductionEvaluationLauncher(
                artifacts, popen_factory=popen
            ).run("project", "run")

        self.assertEqual(2, len(calls))
        replay = calls[1]
        self.assertEqual("3", replay[replay.index("--replay-seed") + 1])
        self.assertEqual(str(run_root / EVALUATION_VIDEO_PATH), replay[replay.index("--video-output") + 1])
        self.assertEqual(str(EVALUATION_REPLAY_FPS), replay[replay.index("--video-fps") + 1])
        self.assertEqual("available", summary["evaluation_video"]["status"])
        self.assertEqual(MODEL_SHA, summary["evaluation_video"]["model_sha256"])

    def test_manipulation_uses_shared_entrypoint_with_explicit_task_id(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts, run_root = self.prepared_run(directory)
            calls = []
            launcher = ProductionEvaluationLauncher(
                artifacts, task_id="s1_lower_release",
                popen_factory=lambda command, **kwargs: (
                    calls.append(command) or FakeProcess()
                ),
            )
            launcher.run("project", "run")
        self.assertEqual(1, len(calls))
        self.assertIn("manipulation_diagnostic_evaluation.py", calls[0][1])
        self.assertEqual(
            "s1_lower_release", calls[0][calls[0].index("--task-id") + 1]
        )
        self.assertEqual(str(run_root), calls[0][calls[0].index("--run-root") + 1])

    def test_replay_sigsegv_only_marks_video_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts, run_root = self.prepared_run(directory, tuple(range(80)))
            calls = 0

            def popen(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    video = run_root / EVALUATION_VIDEO_PATH
                    video.parent.mkdir(parents=True, exist_ok=True)
                    video.write_bytes(b"stale")
                    video.with_name(
                        video.name + EVALUATION_VIDEO_PARTIAL_SUFFIX
                    ).write_bytes(b"partial")
                return FakeProcess(0 if calls == 1 else -11)

            summary = ProductionEvaluationLauncher(
                artifacts, popen_factory=popen
            ).run("project", "run")

        self.assertEqual("passed", evaluation_contract(summary)["status"])
        self.assertEqual("unavailable", summary["evaluation_video"]["status"])
        self.assertEqual("replay_subprocess_sigsegv", summary["evaluation_video"]["failure_reason"])
        self.assertFalse((run_root / EVALUATION_VIDEO_PATH).exists())
        self.assertFalse(
            (run_root / EVALUATION_VIDEO_PATH).with_name(
                (run_root / EVALUATION_VIDEO_PATH).name
                + EVALUATION_VIDEO_PARTIAL_SUFFIX
            ).exists()
        )

    def test_nonzero_headless_exit_is_infrastructure_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts, _ = self.prepared_run(directory)
            launcher = ProductionEvaluationLauncher(
                artifacts, popen_factory=lambda *args, **kwargs: FakeProcess(7)
            )
            with self.assertRaisesRegex(EvaluationInfrastructureError, "code 7"):
                launcher.run("project", "run")

    def test_missing_summary_is_infrastructure_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "run"
            (run_root / "evaluation").mkdir(parents=True)
            (run_root / "evaluation/episodes.jsonl").touch()
            with self.assertRaisesRegex(EvaluationInfrastructureError, "summary"):
                validate_evaluation_outputs(run_root)

    def test_zero_success_is_valid_policy_performance_without_video(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "run"
            write_outputs(run_root, valid_summary())
            summary = validate_evaluation_outputs(run_root)
        self.assertEqual(0.0, summary["success_rate"])
        self.assertEqual("failed", evaluation_contract(summary)["status"])

    def test_policy_gate_requires_rate_and_wilson_lower_bound(self):
        contract = evaluation_contract(valid_summary(80, 0))
        self.assertEqual("passed", contract["status"])
        self.assertEqual(0.2, contract["mean_final_distance"])
        self.assertEqual(0.1, contract["mean_minimum_distance"])
        self.assertEqual(0.0, contract["mean_object_displacement"])
        self.assertAlmostEqual(0.2, contract["timeout_rate"])
        self.assertEqual("failed", evaluation_contract(valid_summary(79, 0))["status"])

    def test_selection_must_be_minimum_successful_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "run"
            summary = valid_summary(2, 7)
            write_outputs(run_root, summary, successful_seeds=(3, 7))
            with self.assertRaisesRegex(EvaluationInfrastructureError, "minimum"):
                validate_evaluation_outputs(run_root)

    def test_inconsistent_counts_and_ci_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "run"
            summary = valid_summary(80, 0)
            summary["success_rate"] = 0.81
            write_outputs(run_root, summary, successful_seeds=tuple(range(80)))
            with self.assertRaisesRegex(EvaluationInfrastructureError, "inconsistent"):
                validate_evaluation_outputs(run_root)

    def test_non_finite_summary_is_infrastructure_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "run"
            summary = valid_summary()
            summary["mean_episode_return"] = math.nan
            write_outputs(run_root, summary)
            with self.assertRaisesRegex(EvaluationInfrastructureError, "non-finite"):
                validate_evaluation_outputs(run_root)


if __name__ == "__main__":
    unittest.main()
