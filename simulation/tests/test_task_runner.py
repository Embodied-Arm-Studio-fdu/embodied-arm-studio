from pathlib import Path
import tempfile
import unittest

from simulation_core import (
    DEMO_ARTIFACT_POLICY, DemoTaskConfig, ExecutionTaskSpec, TaskDefinition,
    TaskExecutionResult, TaskRegistry, TaskRunner, TaskStatus,
    TrainingTaskConfig, TrainingTaskResult,
)


class FakeTrainingRunner:
    def __init__(self):
        self.training_steps = 4
        self.update_steps = 1
        self.received = None

    def run(self, task):
        self.received = task
        return TrainingTaskResult(
            task_id=task.task_id, run_id=task.run_id,
            status=TaskStatus.COMPLETED, training_steps=4, update_steps=1,
            metrics_path="metrics/training.jsonl", checkpoint_path="model.pt",
            final_metrics_summary={"training_steps": 4},
        )


class Runtime:
    def initialize(self): pass
    def step(self): return {"ok": 1}
    def close(self): pass


class TaskRunnerTests(unittest.TestCase):
    def test_push_uses_training_runner_without_demo_branch(self):
        config = TrainingTaskConfig(
            task_type="training", environment_type="s1_push", seed=7,
            rollout_steps=4, num_updates=1, checkpoint_every=1,
            episode_steps=8, physics_steps_per_action=1, batch_size=4,
        )
        fake = FakeTrainingRunner()
        with tempfile.TemporaryDirectory() as directory:
            runner = TaskRunner(
                Path(directory), training_runner_factory=lambda: fake,
                demo_runner_factory=lambda: self.fail("demo runner selected"),
            )
            result = runner.run(ExecutionTaskSpec(
                "t", "r", "p", config, "s1_push"
            ))
        self.assertEqual(TaskStatus.COMPLETED, result.status)
        self.assertEqual("s1_push", fake.received.config.environment_type)
        self.assertEqual(4, runner.training_steps)

    def test_fake_demo_uses_demo_runner_and_never_constructs_training(self):
        definition = TaskDefinition(
            id="fake_demo", display_name="Fake", task_type="demo", robot="s1",
            supports_training=False, supports_video=False, runner_kind="demo",
            config_parser=lambda values: DemoTaskConfig(**values),
            runtime_factory=lambda current: Runtime(),
            artifact_policy=DEMO_ARTIFACT_POLICY,
        )
        registry = TaskRegistry((definition,))

        class FakeDemoRunner:
            training_steps = 0
            update_steps = 0
            def run(self, task):
                return TaskExecutionResult(task.task_id, task.run_id, TaskStatus.COMPLETED)

        with tempfile.TemporaryDirectory() as directory:
            result = TaskRunner(
                Path(directory), registry=registry,
                training_runner_factory=lambda: self.fail("PPO path constructed"),
                demo_runner_factory=lambda: FakeDemoRunner(),
            ).run(ExecutionTaskSpec(
                "t", "r", "p",
                DemoTaskConfig("demo", "fake_demo", duration_seconds=1),
                "fake_demo",
            ))
        self.assertEqual(TaskStatus.COMPLETED, result.status)


if __name__ == "__main__":
    unittest.main()
