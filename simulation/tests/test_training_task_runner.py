import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from simulation_core import (
    FakeBackendClient,
    FakeScheduler,
    InMemoryTaskRepository,
    SimulationService,
    TaskStatus,
    TrainingTaskConfig,
    ManipulationTrainingTaskConfig,
    TrainingTaskRunner,
    TrainingTaskSpec,
)
from simulation_core.ppo import TrainingCancelled
from simulation_core.domain import (
    TRAINING_EPISODE_STEPS,
    TRAINING_PHYSICS_STEPS_PER_ACTION,
)
from simulation_core.video_frame_pipeline import FramePipeline, PixelFormat
from simulation_core.training_task_runner import (
    P2_R8_GRASP_TRANSFER_CHECKPOINT,
    P2_R8_GRASP_TRANSFER_SHA256,
    P3_1_LIFT_TRANSFER_CHECKPOINT,
    P3_1_LIFT_TRANSFER_SHA256,
    P4_1_PLACE_TRANSFER_CHECKPOINT,
)


def training_config():
    return TrainingTaskConfig(
        task_type="training",
        environment_type="s1_push",
        seed=7,
        rollout_steps=4,
        num_updates=1,
        checkpoint_every=1,
        episode_steps=8,
        physics_steps_per_action=1,
        batch_size=4,
    )


def production_training_config():
    return TrainingTaskConfig(**dict(
        training_config().to_dict(),
        run_mode="training",
        episode_steps=TRAINING_EPISODE_STEPS,
        physics_steps_per_action=TRAINING_PHYSICS_STEPS_PER_ACTION,
    ))


class FakeEnvironment:
    def __init__(self, fail_initialize=False):
        self.fail_initialize = fail_initialize
        self.closed = False

    def initialize(self):
        if self.fail_initialize:
            raise RuntimeError("intentional initialization failure")

    def close(self):
        self.closed = True


class FakeTrainer:
    def __init__(self, artifacts_root, fail_training=False):
        self.artifacts_root = artifacts_root
        self.fail_training = fail_training
        self.training_steps = 0
        self.update_steps = 0

    def run_training(self, config, cancel_requested=None):
        if self.fail_training:
            raise RuntimeError("intentional training failure")
        self.training_steps = config.rollout_steps * config.num_updates
        self.update_steps = config.num_updates
        root = self.artifacts_root / config.project_id / config.run_id
        metrics = root / "metrics" / "training.jsonl"
        checkpoint = root / "checkpoints" / "model_latest.pt"
        metrics.parent.mkdir(parents=True)
        checkpoint.parent.mkdir(parents=True)
        record = {"training_steps": self.training_steps, "average_reward": 0.25}
        metrics.write_text(json.dumps(record) + "\n", encoding="utf-8")
        checkpoint.touch()
        return [record]


class FakeCancelledTrainer:
    def __init__(self, artifacts_root):
        self.artifacts_root = artifacts_root
        self.training_steps = 3
        self.update_steps = 0

    def run_training(self, config, cancel_requested=None):
        self.assert_cancel_requested = bool(cancel_requested and cancel_requested())
        root = self.artifacts_root / config.project_id / config.run_id
        checkpoint = root / "checkpoints" / "model_latest.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        raise TrainingCancelled()


class FakeRecorder:
    def __init__(self, fail_start=False):
        self.fail_start = fail_start
        self.started = False
        self.closed = False
        self.failure_reason = None
        self.offered = []

    def start(self):
        if self.fail_start:
            self.failure_reason = "recorder_open_failed"
            raise RuntimeError("sensitive argv must not escape")
        self.started = True

    def offer(self, data):
        self.offered.append(data)
        return True

    def close(self):
        self.closed = True
        return SimpleNamespace(failure_reason=self.failure_reason)

    def record_external_failure(self, reason):
        self.failure_reason = self.failure_reason or reason


class FailingRecorderCleanupLifecycle:
    def on_status(self, run_id, task_id, status):
        if status in (TaskStatus.COMPLETED, TaskStatus.STOPPED, TaskStatus.FAILED):
            raise RuntimeError("recorder cleanup failed")
        return ()


class RecordingEvaluationLauncher:
    def __init__(self, environment, fail=None):
        self.environment = environment
        self.fail = fail
        self.calls = []

    def run(self, project_id, run_id, cancel_requested=lambda: False):
        self.calls.append((project_id, run_id, self.environment.closed))
        if self.fail is not None:
            raise self.fail
        return {
            "episode_count": 100,
            "success_count": 100,
            "success_rate": 1.0,
            "confidence_interval_95": {
                "method": "wilson", "lower": 0.9630065017930143, "upper": 1.0,
            },
        }


class TrainingTaskRunnerTestCase(unittest.TestCase):
    def setUp(self):
        self.repository = InMemoryTaskRepository()
        self.backend = FakeBackendClient()
        self.service = SimulationService(
            self.repository,
            FakeScheduler(),
            self.backend,
            task_id_factory=lambda: "task-training",
        )

    def create_task(self):
        config = training_config()
        return self.service.create_task("run-training", "project-training", config.to_dict())

    def test_grasp_trainer_is_initialized_from_locked_p2_r8_checkpoint(self):
        calls = []

        class FakeGraspTrainer:
            def __init__(self, environment, **kwargs):
                self.environment = environment
                self.kwargs = kwargs

            def initialize_from_p2_r8(self, checkpoint, expected_sha256):
                calls.append((Path(checkpoint), expected_sha256))

        module = ModuleType(
            "simulation_core.experimental.manipulation.grasp_ppo"
        )
        module.ExperimentalGraspPPOTrainer = FakeGraspTrainer
        config = SimpleNamespace(
            environment_type="s1_grasp", rollout_steps=200,
            update_epochs=1, batch_size=16, device="cuda:0",
        )
        with patch.dict(sys.modules, {module.__name__: module}):
            trainer = TrainingTaskRunner._create_trainer(object(), config)

        self.assertIsInstance(trainer, FakeGraspTrainer)
        self.assertEqual([
            (P2_R8_GRASP_TRANSFER_CHECKPOINT, P2_R8_GRASP_TRANSFER_SHA256),
        ], calls)
        self.assertTrue(P2_R8_GRASP_TRANSFER_CHECKPOINT.is_file())

    def test_lower_release_trainer_is_initialized_from_locked_p4_1_checkpoint(self):
        calls = []

        class FakeLowerReleaseTrainer:
            def __init__(self, environment, **kwargs):
                self.environment = environment
                self.kwargs = kwargs

            def initialize_from_p4_1(self, checkpoint):
                calls.append(Path(checkpoint))

        module = ModuleType(
            "simulation_core.experimental.manipulation.lower_release_ppo"
        )
        module.ExperimentalLowerReleasePPOTrainer = FakeLowerReleaseTrainer
        config = SimpleNamespace(
            environment_type="s1_lower_release", rollout_steps=200,
            update_epochs=2, batch_size=32, device="cuda:0",
        )
        with patch.dict(sys.modules, {module.__name__: module}):
            trainer = TrainingTaskRunner._create_trainer(object(), config)

        self.assertIsInstance(trainer, FakeLowerReleaseTrainer)
        self.assertEqual([P4_1_PLACE_TRANSFER_CHECKPOINT], calls)
        self.assertTrue(P4_1_PLACE_TRANSFER_CHECKPOINT.is_file())

    def test_lift_trainer_is_initialized_from_locked_p3_1_checkpoint(self):
        calls = []

        class FakeLiftTrainer:
            def __init__(self, environment, **kwargs):
                self.environment = environment
                self.kwargs = kwargs

            def initialize_from_p3_1(self, checkpoint, expected_sha256):
                calls.append((Path(checkpoint), expected_sha256))

        module = ModuleType(
            "simulation_core.experimental.manipulation.lift_ppo"
        )
        module.ExperimentalLiftPPOTrainer = FakeLiftTrainer
        config = SimpleNamespace(
            environment_type="s1_lift", rollout_steps=200,
            update_epochs=1, batch_size=16, device="cuda:0",
        )
        with patch.dict(sys.modules, {module.__name__: module}):
            trainer = TrainingTaskRunner._create_trainer(object(), config)

        self.assertIsInstance(trainer, FakeLiftTrainer)
        self.assertEqual([
            (P3_1_LIFT_TRANSFER_CHECKPOINT, P3_1_LIFT_TRANSFER_SHA256),
        ], calls)
        self.assertTrue(P3_1_LIFT_TRANSFER_CHECKPOINT.is_file())

    def test_default_stream_lifecycle_uses_configured_lan_descriptor(self):
        spec = TrainingTaskSpec("task-video", "run-video", "project-video", training_config())
        environment = SimpleNamespace(
            backend=SimpleNamespace(
                frame_pipeline=FramePipeline("run-video", "task-video", PixelFormat.RGBA8)
            )
        )
        values = {
            "SIMULATION_STREAM_RTSP_BASE": "rtsp://127.0.0.1:18554",
            "SIMULATION_STREAM_PUBLIC_BASE_URL": "http://192.0.2.10:18889",
            "SIMULATION_FFMPEG_PATH": "/opt/ffmpeg/bin/ffmpeg",
        }
        with patch.dict(os.environ, values, clear=False):
            lifecycle = TrainingTaskRunner(Path("unused"))._create_stream_lifecycle(environment, spec)
        session = lifecycle.manager.create_session(spec.run_id, spec.task_id)
        self.assertTrue(session.rtsp_internal_url.startswith("rtsp://127.0.0.1:18554/"))
        self.assertTrue(session.whep_url.startswith("http://192.0.2.10:18889/"))
        self.assertNotIn("?", session.whep_url)

    def test_default_stream_lifecycle_limits_live_gop_to_half_second(self):
        spec = TrainingTaskSpec("task-video", "run-video", "project-video", training_config())
        environment = SimpleNamespace(
            backend=SimpleNamespace(
                frame_pipeline=FramePipeline("run-video", "task-video", PixelFormat.RGBA8)
            )
        )
        values = {
            "SIMULATION_STREAM_RTSP_BASE": "rtsp://127.0.0.1:18554",
            "SIMULATION_STREAM_PUBLIC_BASE_URL": "http://192.0.2.10:18889",
            "SIMULATION_FFMPEG_PATH": "/opt/ffmpeg/bin/ffmpeg",
        }
        worker = Mock()
        worker_type = Mock(return_value=worker)
        encoder_type = Mock(return_value=Mock())
        publisher = Mock()
        publisher_type = Mock(return_value=publisher)
        recorder = FakeRecorder()
        with patch.dict(os.environ, values, clear=False), patch(
            "simulation_core.video_encoder.EncoderWorker", worker_type
        ), patch(
            "simulation_core.video_encoder.FFmpegNvencEncoder", encoder_type
        ), patch(
            "simulation_core.media_streaming.MediaMtxPublisherSink", publisher_type
        ):
            lifecycle = TrainingTaskRunner(
                Path("unused"), recorder_factory=lambda config: recorder,
            )._create_stream_lifecycle(environment, spec)
            session = lifecycle.manager.create_session(
                spec.run_id, spec.task_id, fps=15
            )
            lifecycle.manager.start_session(session.stream_id)

        encoder_config = worker_type.call_args.args[2]
        self.assertEqual(7, encoder_config.gop_size)
        self.assertLessEqual(encoder_config.gop_size / encoder_config.fps, 0.5)
        publisher.open.assert_called_once_with()
        self.assertTrue(recorder.started)
        encoder_type.assert_called_once_with(Path(values["SIMULATION_FFMPEG_PATH"]))
        worker.start.assert_called_once_with()

    def test_recorder_open_failure_does_not_prevent_live_worker_start(self):
        config = TrainingTaskConfig(**dict(training_config().to_dict(), video_enabled=True))
        spec = TrainingTaskSpec("task-video", "run-video", "project-video", config)
        environment = SimpleNamespace(
            backend=SimpleNamespace(
                frame_pipeline=FramePipeline("run-video", "task-video", PixelFormat.RGBA8)
            )
        )
        values = {
            "SIMULATION_STREAM_RTSP_BASE": "rtsp://127.0.0.1:18554",
            "SIMULATION_STREAM_PUBLIC_BASE_URL": "http://192.0.2.10:18889",
            "SIMULATION_FFMPEG_PATH": "/opt/ffmpeg/bin/ffmpeg",
        }
        worker = Mock()
        publisher = Mock()
        recorder = FakeRecorder(fail_start=True)
        with patch.dict(os.environ, values, clear=False), patch(
            "simulation_core.video_encoder.EncoderWorker", return_value=worker,
        ), patch(
            "simulation_core.media_streaming.MediaMtxPublisherSink",
            return_value=publisher,
        ):
            lifecycle = TrainingTaskRunner(
                Path("unused"), recorder_factory=lambda recorder_config: recorder,
            )._create_stream_lifecycle(environment, spec)
            session = lifecycle.manager.create_session(spec.run_id, spec.task_id)
            lifecycle.manager.start_session(session.stream_id)

        publisher.open.assert_called_once_with()
        worker.start.assert_called_once_with()
        self.assertEqual("recorder_open_failed", recorder.failure_reason)

    def test_video_initialization_failure_does_not_fail_training(self):
        config = training_config().to_dict()
        config["video_enabled"] = True
        task = self.service.create_task("run-video-failure", "project-training", config)
        environment = FakeEnvironment()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            runner = TrainingTaskRunner(
                root,
                environment_factory=lambda config: environment,
                trainer_factory=lambda environment, config: FakeTrainer(root),
                stream_lifecycle_factory=lambda environment, spec: (_ for _ in ()).throw(
                    RuntimeError("video unavailable")
                ),
            )
            result = runner.run(TrainingTaskSpec.from_task(task))
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertTrue(environment.closed)

    def test_recorder_cleanup_failure_preserves_completed_result(self):
        config = TrainingTaskConfig(**dict(training_config().to_dict(), video_enabled=True))
        task = self.service.create_task(
            "run-video-completed-cleanup", "project-training", config.to_dict()
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            runner = TrainingTaskRunner(
                root,
                environment_factory=lambda current: FakeEnvironment(),
                trainer_factory=lambda environment, current: FakeTrainer(root),
                stream_lifecycle_factory=lambda environment, spec: FailingRecorderCleanupLifecycle(),
            )
            result = runner.run(TrainingTaskSpec.from_task(task))
        self.assertEqual(TaskStatus.COMPLETED, result.status)

    def test_recorder_cleanup_failure_preserves_stopped_result(self):
        config = TrainingTaskConfig(**dict(training_config().to_dict(), video_enabled=True))
        task = self.service.create_task(
            "run-video-stopped-cleanup", "project-training", config.to_dict()
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            runner = TrainingTaskRunner(
                root,
                environment_factory=lambda current: FakeEnvironment(),
                trainer_factory=lambda environment, current: FakeCancelledTrainer(root),
                cancel_requested=lambda: True,
                stream_lifecycle_factory=lambda environment, spec: FailingRecorderCleanupLifecycle(),
            )
            result = runner.run(TrainingTaskSpec.from_task(task))
        self.assertEqual(TaskStatus.STOPPED, result.status)

    def test_recorder_cleanup_failure_does_not_replace_training_failure(self):
        config = TrainingTaskConfig(**dict(training_config().to_dict(), video_enabled=True))
        task = self.service.create_task(
            "run-video-failed-cleanup", "project-training", config.to_dict()
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            runner = TrainingTaskRunner(
                root,
                environment_factory=lambda current: FakeEnvironment(),
                trainer_factory=lambda environment, current: FakeTrainer(
                    root, fail_training=True
                ),
                stream_lifecycle_factory=lambda environment, spec: FailingRecorderCleanupLifecycle(),
            )
            with self.assertRaisesRegex(RuntimeError, "intentional training failure"):
                runner.run(TrainingTaskSpec.from_task(task))

    def test_config_is_json_serializable_and_validated(self):
        config = training_config()
        self.assertEqual(config.to_dict(), json.loads(json.dumps(config.to_dict())))
        with self.assertRaisesRegex(ValueError, "rollout_steps"):
            TrainingTaskConfig(**dict(config.to_dict(), rollout_steps=0))

    def test_scene_geometry_round_trips_through_worker_config(self):
        config = TrainingTaskConfig(**dict(
            training_config().to_dict(),
            object_size=[0.10, 0.12, 0.08],
            object_start=[0.48, -0.08, 0.415],
            goal_position=[0.66, 0.14, 0.415],
        ))

        self.assertEqual(config.object_size, (0.10, 0.12, 0.08))
        self.assertEqual(config.object_start, (0.48, -0.08, 0.415))
        self.assertEqual(config.goal_position, (0.66, 0.14, 0.415))
        self.assertEqual(
            TrainingTaskConfig(**config.to_dict()).to_dict(), config.to_dict()
        )

    def test_smoke_and_training_modes_keep_rollout_and_episode_steps_distinct(self):
        smoke = training_config()
        self.assertEqual("smoke", smoke.run_mode)
        self.assertEqual(4, smoke.rollout_steps)
        self.assertEqual(8, smoke.episode_steps)
        training = TrainingTaskConfig(**dict(
            smoke.to_dict(),
            run_mode="training",
            episode_steps=TRAINING_EPISODE_STEPS,
            physics_steps_per_action=TRAINING_PHYSICS_STEPS_PER_ACTION,
        ))
        self.assertEqual(4, training.rollout_steps)
        self.assertEqual(256, training.episode_steps)
        self.assertEqual(8, training.physics_steps_per_action)
        with self.assertRaisesRegex(ValueError, "episode_steps=256"):
            TrainingTaskConfig(**dict(
                smoke.to_dict(), run_mode="training", episode_steps=64
            ))
        with self.assertRaisesRegex(ValueError, "physics_steps_per_action=8"):
            TrainingTaskConfig(**dict(
                smoke.to_dict(),
                run_mode="training",
                episode_steps=TRAINING_EPISODE_STEPS,
                physics_steps_per_action=2,
            ))

    def test_final_e2e_smoke_profile_remains_320_total_steps(self):
        smoke = TrainingTaskConfig(
            task_type="training",
            environment_type="s1_push",
            seed=66,
            rollout_steps=16,
            num_updates=20,
            checkpoint_every=1,
            episode_steps=64,
            physics_steps_per_action=2,
            update_epochs=1,
            batch_size=16,
            video_enabled=True,
            run_mode="smoke",
        )
        self.assertEqual(320, smoke.rollout_steps * smoke.num_updates)
        self.assertEqual(64, smoke.episode_steps)

    def test_success_transitions_and_result(self):
        task = self.create_task()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            environment = FakeEnvironment()
            progress = []
            runner = TrainingTaskRunner(
                root,
                environment_factory=lambda config: environment,
                trainer_factory=lambda environment, config: FakeTrainer(root),
                progress_callback=progress.append,
            )
            result = runner.run(TrainingTaskSpec.from_task(task))
        self.assertEqual(TaskStatus.COMPLETED, result.status)
        self.assertEqual(4, result.training_steps)
        self.assertEqual(1, result.update_steps)
        self.assertEqual(0.25, result.final_metrics_summary["average_reward"])
        self.assertTrue(environment.closed)
        self.assertEqual(
            [TaskStatus.RUNNING, TaskStatus.FINALIZING], progress
        )

    def test_smoke_mode_does_not_launch_final_evaluation(self):
        task = self.create_task()
        launcher = Mock(side_effect=AssertionError("evaluation must stay disabled"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            result = TrainingTaskRunner(
                root,
                environment_factory=lambda config: FakeEnvironment(),
                trainer_factory=lambda environment, config: FakeTrainer(root),
                evaluation_launcher=launcher,
            ).run(TrainingTaskSpec.from_task(task))
        self.assertEqual(TaskStatus.COMPLETED, result.status)
        launcher.run.assert_not_called()

    def test_manipulation_smoke_mode_launches_task_specific_evaluation(self):
        config = ManipulationTrainingTaskConfig(
            task_type="training", environment_type="s1_lower_release", seed=7,
            rollout_steps=4, num_updates=1, checkpoint_every=1,
            episode_steps=120, physics_steps_per_action=4,
            action_delta_m=0.003, video_enabled=False,
        )
        task = TrainingTaskSpec(
            "task-manipulation", "run-manipulation", "project-manipulation", config,
        )
        environment = FakeEnvironment()
        launcher = RecordingEvaluationLauncher(environment)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            result = TrainingTaskRunner(
                root,
                environment_factory=lambda current: environment,
                trainer_factory=lambda current, ignored: FakeTrainer(root),
                evaluation_launcher=launcher,
            ).run(task)
        self.assertEqual(TaskStatus.COMPLETED, result.status)
        self.assertEqual(
            [("project-manipulation", "run-manipulation", True)], launcher.calls
        )
        self.assertEqual("passed", result.evaluation["status"])

    def test_training_mode_launches_evaluation_after_environment_cleanup(self):
        config = production_training_config()
        task = self.service.create_task(
            "run-production", "project-training", config.to_dict()
        )
        environment = FakeEnvironment()
        launcher = RecordingEvaluationLauncher(environment)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            result = TrainingTaskRunner(
                root,
                environment_factory=lambda current: environment,
                trainer_factory=lambda current, ignored: FakeTrainer(root),
                evaluation_launcher=launcher,
            ).run(TrainingTaskSpec.from_task(task))
        self.assertEqual(TaskStatus.COMPLETED, result.status)
        self.assertEqual(
            [("project-training", "run-production", True)], launcher.calls
        )
        self.assertEqual("passed", result.evaluation["status"])
        self.assertTrue(result.evaluation["policy_passed"])

    def test_training_mode_evaluation_failure_preserves_completed_training(self):
        config = production_training_config()
        task = self.service.create_task(
            "run-eval-failure", "project-training", config.to_dict()
        )
        environment = FakeEnvironment()
        launcher = RecordingEvaluationLauncher(
            environment, fail=RuntimeError("evaluation process failed")
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            runner = TrainingTaskRunner(
                root,
                environment_factory=lambda current: environment,
                trainer_factory=lambda current, ignored: FakeTrainer(root),
                evaluation_launcher=launcher,
            )
            result = runner.run(TrainingTaskSpec.from_task(task))
        self.assertEqual(TaskStatus.COMPLETED, result.status)
        self.assertEqual("unavailable", result.evaluation["status"])
        self.assertEqual("RuntimeError", result.evaluation["failure_reason"])
        self.assertTrue(environment.closed)

    def test_stop_observed_at_training_boundary_skips_evaluation(self):
        config = production_training_config()
        task = self.service.create_task(
            "run-boundary-stop", "project-training", config.to_dict()
        )
        launcher = Mock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            result = TrainingTaskRunner(
                root,
                environment_factory=lambda current: FakeEnvironment(),
                trainer_factory=lambda current, ignored: FakeTrainer(root),
                cancel_requested=lambda: True,
                evaluation_launcher=launcher,
            ).run(TrainingTaskSpec.from_task(task))
        self.assertEqual(TaskStatus.STOPPED, result.status)
        launcher.run.assert_not_called()

    def test_video_false_has_zero_recorder_side_effects(self):
        task = self.create_task()
        recorder_factory = Mock(side_effect=AssertionError("recorder must stay disabled"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            runner = TrainingTaskRunner(
                root,
                environment_factory=lambda config: FakeEnvironment(),
                trainer_factory=lambda environment, config: FakeTrainer(root),
                recorder_factory=recorder_factory,
            )
            result = runner.run(TrainingTaskSpec.from_task(task))
            run_root = root / task.project_id / task.run_id
            self.assertFalse((run_root / "video").exists())
        self.assertEqual(TaskStatus.COMPLETED, result.status)
        recorder_factory.assert_not_called()

    def test_initializing_exception_transitions_to_failed_without_gpu(self):
        task = self.create_task()
        environment = FakeEnvironment(fail_initialize=True)
        runner = TrainingTaskRunner(
            Path("unused"),
            environment_factory=lambda config: environment,
        )
        with self.assertRaisesRegex(RuntimeError, "initialization failure"):
            runner.run(TrainingTaskSpec.from_task(task))
        self.assertTrue(environment.closed)

    def test_running_exception_transitions_to_failed_without_gpu(self):
        task = self.create_task()
        environment = FakeEnvironment()
        runner = TrainingTaskRunner(
            Path("unused"),
            environment_factory=lambda config: environment,
            trainer_factory=lambda environment, config: FakeTrainer(
                Path("unused"), fail_training=True
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "training failure"):
            runner.run(TrainingTaskSpec.from_task(task))
        self.assertTrue(environment.closed)

    def test_cooperative_cancel_returns_stopped_data_and_closes_environment(self):
        task = self.create_task()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            environment = FakeEnvironment()
            trainer = FakeCancelledTrainer(root)
            runner = TrainingTaskRunner(
                root,
                environment_factory=lambda config: environment,
                trainer_factory=lambda environment, config: trainer,
                cancel_requested=lambda: True,
            )

            result = runner.run(TrainingTaskSpec.from_task(task))

            self.assertEqual(TaskStatus.STOPPED, result.status)
            self.assertEqual(3, result.training_steps)
            self.assertEqual(0, result.update_steps)
            self.assertEqual("", result.metrics_path)
            self.assertTrue(Path(result.checkpoint_path).is_file())
            self.assertTrue(trainer.assert_cancel_requested)
            self.assertTrue(environment.closed)


if __name__ == "__main__":
    unittest.main()
