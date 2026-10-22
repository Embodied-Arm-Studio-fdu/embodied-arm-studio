import json
import tempfile
import time
import unittest
from pathlib import Path

from robotics.adapters.mock import MockSimulatorAdapter
from robotics.contracts import RunRequest, RunStatus as LegacyRunStatus
from robotics.runs import (
    ExecutionMode,
    InMemoryRunRepository,
    RunEventType,
    RunKind,
    RunService,
    RunStatus,
)


class MockCanonicalIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = InMemoryRunRepository()
        self.adapter = MockSimulatorAdapter(
            Path(self.temp_dir.name),
            tick_seconds=0,
            repository=self.repository,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def launch(self, kind="train", steps=8, adapter=None):
        adapter = adapter or self.adapter
        request = RunRequest(
            environment_id="mock-reach",
            strategy_id="joint-p-control",
            seed=13,
            max_steps=steps,
            config={"action_scale": 0.4},
        )
        launcher = adapter.launch_training if kind == "train" else adapter.launch_play
        return launcher(request)

    def wait_terminal(self, run_id, adapter=None):
        adapter = adapter or self.adapter
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            run = adapter.run_service.get_run(run_id)
            if run.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.STOPPED}:
                adapter.get_run(run_id)
                return run
            time.sleep(0.002)
        self.fail("canonical Mock run did not finish")

    def test_training_uses_canonical_mock_run_and_keeps_legacy_mode(self):
        record = self.launch("train")
        run = self.wait_terminal(record.run_id)
        self.assertEqual(run.execution_mode, ExecutionMode.MOCK)
        self.assertEqual(run.run_kind, RunKind.TRAIN)
        self.assertEqual(self.adapter.get_run(record.run_id).mode, "train")
        self.assertTrue(record.run_id.startswith("run-"))

    def test_run_service_can_be_injected(self):
        service = RunService(
            InMemoryRunRepository(),
            run_id_factory=lambda: "run-injected",
        )
        adapter = MockSimulatorAdapter(
            Path(self.temp_dir.name),
            tick_seconds=0,
            run_service=service,
        )
        record = self.launch(adapter=adapter)
        run = self.wait_terminal(record.run_id, adapter)
        self.assertEqual(record.run_id, "run-injected")
        self.assertEqual(run.run_kind, RunKind.TRAIN)
        self.assertIs(adapter.repository, service.repository)

    def test_play_is_operation_not_execution_mode(self):
        record = self.launch("play")
        run = self.wait_terminal(record.run_id)
        self.assertEqual(run.run_kind, RunKind.PLAY)
        self.assertEqual(run.execution_mode, ExecutionMode.MOCK)
        self.assertEqual(self.adapter.get_run(record.run_id).mode, "play")

    def test_complete_lifecycle_uses_canonical_state_events(self):
        record = self.launch()
        run = self.wait_terminal(record.run_id)
        self.assertEqual(run.status, RunStatus.COMPLETED)
        state_events = [
            event for event in self.repository.list_events(record.run_id)
            if event.type == RunEventType.STATE
        ]
        self.assertEqual(
            [event.payload["status"] for event in state_events],
            ["initializing", "running", "finalizing", "completed"],
        )
        legacy = self.adapter.get_run(record.run_id)
        self.assertEqual(legacy.status, LegacyRunStatus.COMPLETED)
        self.assertNotIn(legacy.status.value, {"initializing", "finalizing", "stopping"})

    def test_event_sequences_increase_and_legacy_events_convert(self):
        record = self.launch(steps=5)
        self.wait_terminal(record.run_id)
        canonical = self.repository.list_events(record.run_id)
        self.assertEqual([event.sequence for event in canonical], list(range(1, len(canonical) + 1)))
        legacy = list(self.adapter.stream_events(record.run_id))
        self.assertTrue(all(event.type in {"status", "metric", "error"} for event in legacy))
        metric = next(event for event in legacy if event.type == "metric")
        self.assertIn("mean_reward", metric.payload["metrics"])
        statuses = [event.payload["status"] for event in legacy if event.type == "status"]
        self.assertEqual(statuses, ["queued", "running", "running", "completed"])

    def test_metrics_and_run_files_keep_legacy_format(self):
        record = self.launch(steps=4)
        self.wait_terminal(record.run_id)
        run_dir = Path(self.temp_dir.name) / record.run_id
        metric_lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(metric_lines), 4)
        self.assertEqual(json.loads(metric_lines[-1])["type"], "metric")
        persisted = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["status"], "completed")
        self.assertEqual(persisted["mode"], "train")
        self.assertEqual(persisted["exit_code"], 0)

    def test_running_stop_reaches_stopped_without_terminal_regression(self):
        adapter = MockSimulatorAdapter(Path(self.temp_dir.name), tick_seconds=0.003)
        record = self.launch(steps=500, adapter=adapter)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if adapter.run_service.get_run(record.run_id).status == RunStatus.RUNNING:
                break
            time.sleep(0.001)
        else:
            self.fail("run never reached running")
        adapter.stop_run(record.run_id)
        run = self.wait_terminal(record.run_id, adapter)
        self.assertEqual(run.status, RunStatus.STOPPED)
        statuses = [
            event.payload["status"] for event in adapter.repository.list_events(record.run_id)
            if event.type == RunEventType.STATE
        ]
        self.assertEqual(statuses[-2:], ["stopping", "stopped"])
        time.sleep(0.01)
        self.assertEqual(adapter.run_service.get_run(record.run_id).status, RunStatus.STOPPED)
        self.assertEqual(adapter.stop_run(record.run_id).status, LegacyRunStatus.STOPPED)

    def test_persisted_queued_run_can_stop_without_local_execution_handle(self):
        run = self.adapter.run_service.create_run(
            user_id="user-1",
            project_id="project-1",
            execution_mode=ExecutionMode.MOCK,
            run_kind=RunKind.TRAIN,
            parameters={"seed": 13},
            project_config_snapshot={"config_version": 1},
            code="print('queued')\n",
        )

        stopped = self.adapter.stop_canonical_run(run.run_id)

        self.assertEqual(stopped.status, RunStatus.STOPPED)
        self.assertEqual(
            self.adapter.run_service.get_run(run.run_id).status,
            RunStatus.STOPPED,
        )
        events = self.repository.list_events(run.run_id)
        self.assertEqual(
            [event.payload["status"] for event in events],
            ["stopped"],
        )

    def test_list_get_final_metric_artifacts_and_run_isolation(self):
        first = self.launch(steps=3)
        second = self.launch("play", steps=6)
        self.wait_terminal(first.run_id)
        self.wait_terminal(second.run_id)
        listed_ids = {run.run_id for run in self.adapter.list_runs()}
        self.assertEqual(listed_ids, {first.run_id, second.run_id})
        self.assertEqual(self.adapter.get_run(first.run_id).run_id, first.run_id)
        first_metrics = [event for event in self.adapter.stream_events(first.run_id) if event.type == "metric"]
        second_metrics = [event for event in self.adapter.stream_events(second.run_id) if event.type == "metric"]
        self.assertEqual(len(first_metrics), 3)
        self.assertEqual(len(second_metrics), 6)
        self.assertIn("distance", first_metrics[-1].payload["metrics"])
        self.assertIn("metrics.jsonl", {item["name"] for item in self.adapter.list_artifacts(first.run_id)})
        first_event_ids = {event.event_id for event in self.repository.list_events(first.run_id)}
        second_event_ids = {event.event_id for event in self.repository.list_events(second.run_id)}
        self.assertFalse(first_event_ids & second_event_ids)


if __name__ == "__main__":
    unittest.main()
