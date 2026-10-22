import json
import tempfile
import time
import unittest
from pathlib import Path

from robotics.adapters.mock import MockSimulatorAdapter
from robotics.contracts import RunRequest, RunStatus


class MockSimulatorAdapterTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.adapter = MockSimulatorAdapter(Path(self.temp_dir.name), tick_seconds=0)

    def tearDown(self):
        self.temp_dir.cleanup()

    def wait_until_finished(self, run_id):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            record = self.adapter.get_run(run_id)
            if record.status not in (RunStatus.QUEUED, RunStatus.RUNNING):
                return record
            time.sleep(0.005)
        self.fail("mock run did not finish")

    def test_training_persists_metrics_and_export_manifest(self):
        record = self.adapter.launch_training(RunRequest(
            environment_id="mock-reach",
            strategy_id="policy-gradient-demo",
            seed=7,
            max_steps=12,
            config={"action_scale": 0.4},
        ))
        finished = self.wait_until_finished(record.run_id)

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        events = list(self.adapter.stream_events(record.run_id))
        metric_events = [event for event in events if event.type == "metric"]
        self.assertEqual(len(metric_events), 12)
        self.assertLess(metric_events[-1].payload["metrics"]["distance"], metric_events[0].payload["metrics"]["distance"])

        manifest = self.adapter.export_policy(record.run_id, "manifest")
        self.assertFalse(manifest["real_robot_control_enabled"])
        self.assertEqual(len(manifest["checksum_sha256"]), 64)
        persisted = json.loads((Path(self.temp_dir.name) / record.run_id / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["status"], "completed")

    def test_validation_rejects_unknown_and_out_of_range_parameters(self):
        errors = self.adapter.validate_config("mock-reach", {"action_scale": 4, "unsafe_shell": True})
        self.assertEqual(len(errors), 2)

    def test_pick_place_environment_emits_structured_evidence(self):
        environment = next(
            item for item in self.adapter.list_environments()
            if item.id == "mock-pick-place"
        )
        self.assertEqual(environment.revision, 1)
        self.assertEqual(environment.capability, "mock")
        self.assertIn("reward_terms", environment.evidence_channels)

        record = self.adapter.launch_training(RunRequest(
            environment_id="mock-pick-place",
            strategy_id="joint-p-control",
            seed=11,
            max_steps=8,
        ))
        self.wait_until_finished(record.run_id)
        metric_events = [
            event for event in self.adapter.stream_events(record.run_id)
            if event.type == "metric"
        ]
        final_payload = metric_events[-1].payload
        self.assertEqual(set(final_payload["reward_terms"]), {"reach", "grasp", "place", "action_penalty"})
        self.assertIn(final_payload["termination"]["reason"], {"running", "success", "time_limit"})
        self.assertEqual(len(final_payload["trajectory"]["end_effector_xyz"]), 3)
        self.assertEqual(final_payload["collision"]["model"], "geometric-approximation")

    def test_stop_transitions_running_job(self):
        adapter = MockSimulatorAdapter(Path(self.temp_dir.name), tick_seconds=0.01)
        record = adapter.launch_training(RunRequest(
            environment_id="mock-reach",
            strategy_id="joint-p-control",
            max_steps=500,
        ))
        adapter.stop_run(record.run_id)
        finished = self.wait_with_adapter(adapter, record.run_id)
        self.assertEqual(finished.status, RunStatus.STOPPED)

    def wait_with_adapter(self, adapter, run_id):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            record = adapter.get_run(run_id)
            if record.status not in (RunStatus.QUEUED, RunStatus.RUNNING):
                return record
            time.sleep(0.005)
        self.fail("mock run did not stop")


if __name__ == "__main__":
    unittest.main()