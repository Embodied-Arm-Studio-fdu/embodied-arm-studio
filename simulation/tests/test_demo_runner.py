import json
from pathlib import Path
import tempfile
import unittest

from simulation_core import (
    ArtifactFinalizer, DEMO_ARTIFACT_POLICY, DemoTaskConfig, DemoTaskRunner,
    ExecutionTaskSpec, TaskStatus,
)


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class FakeRuntime:
    def __init__(self, fail=False):
        self.fail = fail
        self.initialized = False
        self.closed = False
        self.steps = 0

    def initialize(self):
        self.initialized = True

    def step(self):
        self.steps += 1
        if self.fail:
            raise RuntimeError("fake demo failed")
        return {"position_error": 1.0 / self.steps}

    def close(self):
        self.closed = True


def spec(config):
    return ExecutionTaskSpec(
        "task-demo", "run-demo", "project-demo", config,
        config.environment_type, "2026-09-20T00:00:00+00:00",
    )


def config(**changes):
    values = {
        "task_type": "demo", "environment_type": "fake_demo",
        "duration_seconds": 0.3, "tick_interval_seconds": 0.1,
    }
    values.update(changes)
    return DemoTaskConfig(**values)


class DemoRunnerTests(unittest.TestCase):
    def test_completed_has_metrics_and_no_training_outputs(self):
        clock = Clock()
        runtime = FakeRuntime()
        statuses = []
        with tempfile.TemporaryDirectory() as directory:
            runner = DemoTaskRunner(
                Path(directory), lambda current: runtime,
                progress_callback=statuses.append,
                monotonic_clock=clock, sleeper=clock.sleep,
            )
            result = runner.run(spec(config()))
            records = [json.loads(line) for line in Path(result.metrics_path).read_text().splitlines()]
        self.assertEqual(TaskStatus.COMPLETED, result.status)
        self.assertEqual([TaskStatus.RUNNING, TaskStatus.FINALIZING], statuses)
        self.assertGreaterEqual(len(records), 3)
        self.assertEqual(0, result.training_steps)
        self.assertEqual("", result.checkpoint_path)
        self.assertTrue(runtime.initialized)
        self.assertTrue(runtime.closed)

    def test_stopped_checks_cancel_each_tick_and_closes(self):
        clock = Clock()
        runtime = FakeRuntime()
        calls = {"count": 0}

        def cancel():
            calls["count"] += 1
            return calls["count"] >= 2

        with tempfile.TemporaryDirectory() as directory:
            result = DemoTaskRunner(
                Path(directory), lambda current: runtime,
                cancel_requested=cancel,
                monotonic_clock=clock, sleeper=clock.sleep,
            ).run(spec(config(duration_seconds=5)))
        self.assertEqual(TaskStatus.STOPPED, result.status)
        self.assertEqual(1, runtime.steps)
        self.assertTrue(runtime.closed)

    def test_runtime_exception_escapes_for_worker_failed_transition(self):
        clock = Clock()
        runtime = FakeRuntime(fail=True)
        with tempfile.TemporaryDirectory() as directory:
            runner = DemoTaskRunner(
                Path(directory), lambda current: runtime,
                monotonic_clock=clock, sleeper=clock.sleep,
            )
            with self.assertRaisesRegex(RuntimeError, "fake demo failed"):
                runner.run(spec(config()))
        self.assertTrue(runtime.closed)

    def test_video_disabled_never_constructs_video_runtime(self):
        clock = Clock()
        runtime = FakeRuntime()
        with tempfile.TemporaryDirectory() as directory:
            runner = DemoTaskRunner(
                Path(directory), lambda current: runtime,
                stream_lifecycle_factory=lambda *args: self.fail("video started"),
                monotonic_clock=clock, sleeper=clock.sleep,
            )
            self.assertEqual(TaskStatus.COMPLETED, runner.run(spec(config())).status)

    def test_demo_artifact_policy_completes_without_checkpoint(self):
        clock = Clock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = spec(config())
            finalizer = ArtifactFinalizer(root, policy=DEMO_ARTIFACT_POLICY)
            run_root = finalizer.prepare(task)
            result = DemoTaskRunner(
                root, lambda current: FakeRuntime(),
                monotonic_clock=clock, sleeper=clock.sleep,
            ).run(task)
            summary = finalizer.finalize(
                result.status,
                final_metrics_summary=result.final_metrics_summary,
            )
            manifest = json.loads(Path(summary.manifest_path).read_text())
        self.assertEqual("completed", manifest["final_status"])
        self.assertEqual("", summary.latest_checkpoint_path)
        self.assertFalse((run_root / "checkpoints").exists())
        self.assertIn("metrics/demo.jsonl", [x["path"] for x in summary.artifacts])


if __name__ == "__main__":
    unittest.main()
