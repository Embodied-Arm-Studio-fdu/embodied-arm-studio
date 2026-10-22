import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from simulation_core import (
    ArtifactFinalizer,
    ArtifactValidationError,
    TaskStatus,
    TrainingTaskConfig,
    TrainingTaskSpec,
)
from simulation_core.video_recorder import Mp4ProbeResult


def task_spec(run_id="run-artifacts", task_id="task-artifacts"):
    return TrainingTaskSpec(
        task_id=task_id,
        run_id=run_id,
        project_id="project-artifacts",
        config=TrainingTaskConfig(
            task_type="training",
            environment_type="s1_push",
            seed=23,
            rollout_steps=4,
            num_updates=2,
            checkpoint_every=1,
            episode_steps=8,
            physics_steps_per_action=1,
            batch_size=4,
        ),
        created_at="2026-08-22T00:00:00+00:00",
    )


def video_task_spec(run_id="run-video-artifacts", task_id="task-video-artifacts"):
    base = task_spec(run_id=run_id, task_id=task_id)
    return TrainingTaskSpec(
        task_id=base.task_id,
        run_id=base.run_id,
        project_id=base.project_id,
        config=TrainingTaskConfig(**dict(
            base.config.to_dict(),
            video_enabled=True,
            video_width=640,
            video_height=480,
            video_fps=15.0,
        )),
        created_at=base.created_at,
    )


def production_task_spec(run_id="run-production-artifacts"):
    base = task_spec(run_id=run_id)
    return TrainingTaskSpec(
        task_id=base.task_id, run_id=base.run_id, project_id=base.project_id,
        config=TrainingTaskConfig(**dict(
            base.config.to_dict(), run_mode="training", episode_steps=256,
            physics_steps_per_action=8,
        )), created_at=base.created_at,
    )


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ArtifactFinalizerTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.artifacts_root = Path(self.temporary_directory.name) / "artifacts"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_training_outputs(self, root, include_best=True):
        metrics = root / "metrics" / "training.jsonl"
        latest = root / "checkpoints" / "model_latest.pt"
        best = root / "checkpoints" / "model_best.pt"
        record = {
            "update_index": 2,
            "training_steps": 8,
            "average_reward": 0.75,
        }
        metrics.write_text(json.dumps(record) + "\n", encoding="utf-8")
        latest.write_bytes(b"latest-checkpoint")
        if include_best:
            best.write_bytes(b"best-checkpoint")
        return record

    def test_completed_snapshot_manifest_relative_metadata_and_logs(self):
        task = task_spec()
        finalizer = ArtifactFinalizer(self.artifacts_root)
        root = finalizer.prepare(task, child_pid=321)
        with finalizer.capture_process_output():
            print("important child output", flush=True)
            print("important child error", file=sys.stderr, flush=True)
        record = self.write_training_outputs(root)

        summary = finalizer.finalize(
            TaskStatus.COMPLETED,
            training_steps=8,
            update_steps=2,
            final_metrics_summary=record,
        )

        config_path = root / "config" / "training_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        manifest = json.loads(Path(summary.manifest_path).read_text(encoding="utf-8"))
        self.assertEqual(task.config.to_dict(), config)
        self.assertEqual(23, manifest["seed"])
        self.assertEqual("s1_push", manifest["environment_type"])
        self.assertEqual("completed", manifest["final_status"])
        self.assertEqual(record, manifest["final_metrics_summary"])
        self.assertIn("important child output", (root / "logs/stdout.log").read_text())
        self.assertIn("important child error", (root / "logs/stderr.log").read_text())
        self.assertTrue(summary.best_checkpoint_path.endswith("model_best.pt"))

        callback_metadata = {
            item["path"]: item for item in summary.artifacts
        }
        manifest_path = root / "manifest.json"
        self.assertNotIn(
            "manifest.json", {item["path"] for item in manifest["artifacts"]}
        )
        self.assertEqual(
            manifest_path.stat().st_size,
            callback_metadata["manifest.json"]["size_bytes"],
        )
        self.assertEqual(
            sha256(manifest_path), callback_metadata["manifest.json"]["sha256"]
        )

        for item in manifest["artifacts"]:
            self.assertFalse(Path(item["path"]).is_absolute())
            path = root / item["path"]
            self.assertEqual(path.stat().st_size, item["size_bytes"])
            self.assertEqual(sha256(path), item["sha256"])

        self.assertEqual(8, len(summary.artifacts))

    def test_video_false_completed_remains_eight_artifacts(self):
        finalizer = ArtifactFinalizer(self.artifacts_root)
        root = finalizer.prepare(task_spec(run_id="run-video-disabled"))
        record = self.write_training_outputs(root)
        summary = finalizer.finalize(
            TaskStatus.COMPLETED, final_metrics_summary=record
        )
        self.assertEqual(8, len(summary.artifacts))
        self.assertNotIn(
            "training_video", {item["type"] for item in summary.artifacts}
        )

    def test_production_registers_evaluation_artifacts_and_relative_paths(self):
        finalizer = ArtifactFinalizer(self.artifacts_root)
        root = finalizer.prepare(production_task_spec())
        record = self.write_training_outputs(root)
        for relative, content in (
            ("evaluation/summary.json", b"{}"),
            ("evaluation/episodes.jsonl", b"{}\n"),
            ("evaluation_video/successful_episode.mp4", b"mp4"),
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        evaluation = {
            "status": "failed", "deterministic": True,
            "policy_passed": False,
        }
        summary = finalizer.finalize(
            TaskStatus.COMPLETED, final_metrics_summary=record,
            evaluation=evaluation,
        )
        types = [item["type"] for item in summary.artifacts]
        self.assertEqual(len(types), len(set(types)))
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["artifacts"]) + 1, len(summary.artifacts))
        self.assertIn("training_episode_metrics", types)
        self.assertNotIn("training_metrics", types)
        self.assertEqual(
            {"evaluation_summary", "evaluation_episodes", "evaluation_video"},
            set(types).intersection({
                "evaluation_summary", "evaluation_episodes", "evaluation_video",
            }),
        )
        self.assertTrue(summary.metrics_path.endswith("metrics/training.jsonl"))
        self.assertTrue(all(not Path(item["path"]).is_absolute() for item in summary.artifacts))

    def test_smoke_run_registers_evaluation_when_evaluation_was_executed(self):
        finalizer = ArtifactFinalizer(self.artifacts_root)
        root = finalizer.prepare(task_spec(run_id="run-smoke-evaluation"))
        record = self.write_training_outputs(root)
        for relative, content in (
            ("evaluation/summary.json", b"{}"),
            ("evaluation/episodes.jsonl", b"{}\n"),
            ("evaluation_video/successful_episode.mp4", b"mp4"),
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        summary = finalizer.finalize(
            TaskStatus.COMPLETED,
            final_metrics_summary=record,
            evaluation={
                "status": "passed", "deterministic": True,
                "policy_passed": True,
            },
        )
        types = {item["type"] for item in summary.artifacts}
        self.assertTrue({
            "evaluation_summary", "evaluation_episodes", "evaluation_video",
        }.issubset(types))

    def test_video_failure_partial_is_not_registered(self):
        finalizer = ArtifactFinalizer(self.artifacts_root)
        root = finalizer.prepare(video_task_spec(run_id="run-video-partial"))
        record = self.write_training_outputs(root)
        partial = root / "video" / "overview.mp4.partial"
        partial.parent.mkdir()
        partial.write_bytes(b"partial")
        summary = finalizer.finalize(
            TaskStatus.COMPLETED, final_metrics_summary=record
        )
        self.assertEqual(8, len(summary.artifacts))
        self.assertTrue(partial.is_file())

    def test_video_success_is_ninth_callback_artifact_with_measured_manifest_fps(self):
        finalizer = ArtifactFinalizer(self.artifacts_root)
        root = finalizer.prepare(video_task_spec())
        record = self.write_training_outputs(root)
        video = root / "video" / "overview.mp4"
        video.parent.mkdir()
        video.write_bytes(b"valid-mp4")
        measured = Mp4ProbeResult("h264", 640, 480, 14.98, 3.25)
        with mock.patch.dict(
            os.environ,
            {"SIMULATION_FFMPEG_PATH": "/opt/private/bin/ffmpeg"},
            clear=False,
        ), mock.patch(
            "simulation_core.artifact_finalizer.probe_mp4", return_value=measured
        ) as probe:
            summary = finalizer.finalize(
                TaskStatus.COMPLETED, final_metrics_summary=record
            )

        self.assertEqual(9, len(summary.artifacts))
        videos = [item for item in summary.artifacts if item["type"] == "training_video"]
        self.assertEqual(1, len(videos))
        manifest = json.loads(Path(summary.manifest_path).read_text())
        manifest_videos = [
            item for item in manifest["artifacts"]
            if item["type"] == "training_video"
        ]
        self.assertEqual(1, len(manifest_videos))
        self.assertEqual(14.98, manifest_videos[0]["metadata"]["actual_fps"])
        self.assertEqual(3.25, manifest_videos[0]["metadata"]["duration_seconds"])
        self.assertEqual(len(manifest["artifacts"]) + 1, len(summary.artifacts))
        self.assertEqual(15.0, probe.call_args.args[4])

    def test_late_video_probe_failure_omits_but_does_not_delete_final(self):
        finalizer = ArtifactFinalizer(self.artifacts_root)
        root = finalizer.prepare(video_task_spec(run_id="run-video-late-failure"))
        record = self.write_training_outputs(root)
        video = root / "video" / "overview.mp4"
        video.parent.mkdir()
        video.write_bytes(b"retain-ready-file")
        with mock.patch.dict(
            os.environ,
            {"SIMULATION_FFMPEG_PATH": "/opt/private/bin/ffmpeg"},
            clear=False,
        ), mock.patch(
            "simulation_core.artifact_finalizer.probe_mp4",
            side_effect=RuntimeError("late failure"),
        ):
            summary = finalizer.finalize(
                TaskStatus.COMPLETED, final_metrics_summary=record
            )
        self.assertEqual(8, len(summary.artifacts))
        self.assertTrue(video.is_file())

    def test_completed_validation_rejects_missing_best_without_creating_it(self):
        finalizer = ArtifactFinalizer(self.artifacts_root)
        root = finalizer.prepare(task_spec())
        self.write_training_outputs(root, include_best=False)

        with self.assertRaisesRegex(ArtifactValidationError, "model_best.pt"):
            finalizer.finalize(TaskStatus.COMPLETED)

        self.assertFalse((root / "checkpoints/model_best.pt").exists())

    def test_manifest_digest_is_computed_from_complete_final_json(self):
        finalizer = ArtifactFinalizer(self.artifacts_root)
        root = finalizer.prepare(task_spec())
        record = self.write_training_outputs(root)
        from simulation_core import artifact_finalizer as module

        original = module._file_sha256
        observed_manifest = []

        def inspect_complete_file(path):
            if path.name == "manifest.json":
                observed_manifest.append(json.loads(path.read_text(encoding="utf-8")))
            return original(path)

        with mock.patch.object(module, "_file_sha256", side_effect=inspect_complete_file):
            summary = finalizer.finalize(
                TaskStatus.COMPLETED, final_metrics_summary=record
            )

        self.assertEqual(1, len(observed_manifest))
        self.assertEqual("completed", observed_manifest[0]["final_status"])
        manifest_item = next(
            item for item in summary.artifacts if item["type"] == "manifest"
        )
        self.assertEqual(sha256(root / "manifest.json"), manifest_item["sha256"])

    def test_registered_symlink_outside_run_is_never_read_or_sent(self):
        finalizer = ArtifactFinalizer(self.artifacts_root)
        root = finalizer.prepare(task_spec(run_id="run-symlink"))
        outside = Path(self.temporary_directory.name) / "outside-secret"
        outside.write_text("not-a-real-secret", encoding="utf-8")
        metrics = root / "metrics" / "training.jsonl"
        metrics.symlink_to(outside)

        with self.assertRaisesRegex(ArtifactValidationError, "regular file"):
            finalizer.finalize(TaskStatus.STOPPED)

    def test_stopped_manifest_accepts_partial_artifacts_without_fake_best(self):
        finalizer = ArtifactFinalizer(self.artifacts_root)
        root = finalizer.prepare(task_spec(run_id="run-stopped"))
        (root / "metrics/training.jsonl").write_text(
            json.dumps(
                {"update_index": 1, "training_steps": 4, "average_reward": 0.1}
            )
            + "\n",
            encoding="utf-8",
        )

        summary = finalizer.finalize(TaskStatus.STOPPED)
        manifest = json.loads(Path(summary.manifest_path).read_text())

        self.assertEqual("stopped", manifest["final_status"])
        self.assertEqual(4, manifest["training_steps"])
        self.assertTrue(summary.metrics_path)
        self.assertEqual("", summary.latest_checkpoint_path)
        self.assertEqual("", summary.best_checkpoint_path)
        self.assertFalse((root / "checkpoints/model_best.pt").exists())

    def test_failed_manifest_preserves_error_and_diagnostic_logs(self):
        finalizer = ArtifactFinalizer(self.artifacts_root)
        root = finalizer.prepare(task_spec(run_id="run-failed"))
        (root / "logs/stderr.log").write_text("traceback details\n")
        finalizer.log_event("error", error_type="RuntimeError")

        summary = finalizer.finalize(
            TaskStatus.FAILED,
            error={"type": "RuntimeError", "message": "training exploded"},
        )
        manifest = json.loads(Path(summary.manifest_path).read_text())

        self.assertEqual("failed", manifest["final_status"])
        self.assertEqual("training exploded", manifest["error"]["message"])
        self.assertIn("traceback details", Path(summary.log_paths["stderr"]).read_text())

    def test_different_run_ids_are_strictly_isolated(self):
        first = ArtifactFinalizer(self.artifacts_root)
        second = ArtifactFinalizer(self.artifacts_root)
        first_root = first.prepare(task_spec(run_id="run-one", task_id="task-one"))
        second_root = second.prepare(task_spec(run_id="run-two", task_id="task-two"))
        first.finalize(TaskStatus.STOPPED)
        second.finalize(TaskStatus.FAILED, error={"type": "FakeError"})

        self.assertNotEqual(first_root, second_root)
        self.assertTrue((first_root / "manifest.json").is_file())
        self.assertTrue((second_root / "manifest.json").is_file())
        self.assertFalse((first_root / "run-two").exists())


if __name__ == "__main__":
    unittest.main()
