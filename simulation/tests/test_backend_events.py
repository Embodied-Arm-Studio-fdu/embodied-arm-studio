import json
import unittest

from simulation_core import (
    BackendMessageFactory,
    EventType,
    RecordingBackendClient,
    RunSequenceGenerator,
    SimulationEvent,
    TerminalCallback,
    TerminalStatus,
)


NOW = "2026-08-23T12:00:00+00:00"
LATER = "2026-08-23T12:01:00Z"


def callback(status, **overrides):
    values = dict(
        event_id="evt_terminal", sequence=1, run_id="run-a", task_id="task-a",
        status=status, started_at=NOW, ended_at=LATER, training_steps=8000,
        update_steps=8, stop_reason=None, metrics={"average_reward": 0.82},
        model={"type": "checkpoint_best", "path": "checkpoints/model_best.pt"},
        artifacts=[{"type": "manifest", "path": "manifest.json"}], error=None,
    )
    values.update(overrides)
    return TerminalCallback(**values)


class SimulationEventTests(unittest.TestCase):
    def make_event(self, event_type, payload):
        return SimulationEvent(
            event_id="evt_1", sequence=1, type=event_type, run_id="run-a",
            task_id="task-a", occurred_at=NOW, payload=payload,
        )

    def test_event_is_json_serializable(self):
        event = self.make_event(EventType.STATE, {
            "previous_status": "initializing", "status": "running"
        })
        self.assertEqual("state", json.loads(json.dumps(event.to_dict()))["type"])

    def test_all_four_payloads_validate(self):
        cases = (
            ("state", {"previous_status": "queued", "status": "initializing"}),
            ("metric", {"step": 8000, "metrics": {
                "update_index": 20, "average_reward": 0.82, "policy_loss": 0.12,
                "value_loss": 0.08, "entropy": 7.4, "approx_kl": 0.01,
                "clip_fraction": 0.2, "explained_variance": 0.7,
            }}),
            ("error", {"code": "TRAINING_FAILED", "message": "failed", "retryable": False}),
            ("heartbeat", {"status": "running", "training_steps": 8000}),
        )
        for event_type, payload in cases:
            with self.subTest(event_type=event_type):
                self.make_event(event_type, payload)

    def test_invalid_payloads_and_runtime_objects_are_rejected(self):
        with self.assertRaises(ValueError):
            self.make_event("metric", {"step": 1, "metrics": {"invented": 1.0}})
        with self.assertRaises(ValueError):
            self.make_event("heartbeat", {"status": "running", "training_steps": object()})
        with self.assertRaises(ValueError):
            self.make_event("unknown", {})

    def test_live_update_index_profile_ranges_and_strict_validation(self):
        for update_index in (1, 2, *range(3, 21)):
            with self.subTest(update_index=update_index):
                event = self.make_event("metric", {
                    "step": update_index * 16,
                    "metrics": {
                        "update_index": update_index,
                        "average_reward": 0.1,
                    },
                })
                self.assertEqual(
                    update_index, event.payload["metrics"]["update_index"]
                )
                self.assertNotIn("update_index", event.payload)

        for invalid in (0, -1, 1.0, "1", True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "update_index"):
                    self.make_event("metric", {
                        "step": 16,
                        "metrics": {
                            "update_index": invalid,
                            "average_reward": 0.1,
                        },
                    })

        with self.assertRaisesRegex(ValueError, "update_index"):
            self.make_event(
                "metric", {"step": 16, "metrics": {"average_reward": 0.1}}
            )


class TerminalCallbackTests(unittest.TestCase):
    def test_three_terminal_statuses_validate_and_serialize(self):
        completed = callback("completed")
        failed = callback("failed", model=None, error={
            "code": "PPO_ERROR", "message": "training failed", "retryable": False
        })
        stopped = callback("stopped", stop_reason="requested_by_user")
        for item in (completed, failed, stopped):
            json.dumps(item.to_dict())

    def test_completed_constraints(self):
        with self.assertRaises(ValueError):
            callback("completed", error={"code": "X", "message": "x", "retryable": False})
        with self.assertRaises(ValueError):
            callback("completed", model=None)
        with self.assertRaises(ValueError):
            callback("completed", artifacts=[])

    def test_completed_demo_has_no_fabricated_model(self):
        result = callback(
            "completed", model=None, training_steps=0, update_steps=0,
            metrics={"tracking_error_m": 0.01},
        )
        self.assertIsNone(result.model)

    def test_failed_requires_structured_error(self):
        with self.assertRaises(ValueError):
            callback("failed", error=None)
        with self.assertRaises(ValueError):
            callback("failed", error={"code": "X", "message": "x"})

    def test_stopped_requires_reason_and_may_keep_checkpoint(self):
        with self.assertRaises(ValueError):
            callback("stopped", stop_reason=None)
        result = callback("stopped", stop_reason="requested_by_user")
        self.assertEqual("checkpoint_best", result.model["type"])

    def test_host_absolute_artifact_paths_are_rejected(self):
        with self.assertRaises(ValueError):
            callback("completed", model={
                "type": "checkpoint_best", "path": "/tmp/model_best.pt"
            })
        with self.assertRaises(ValueError):
            callback("completed", artifacts=[{
                "type": "manifest", "path": "C:\\runs\\manifest.json"
            }])
        with self.assertRaises(ValueError):
            callback("completed", model={
                "type": "checkpoint_best",
                "uri": "artifact://run-a/checkpoints/model_best.pt",
            })
        with self.assertRaises(ValueError):
            callback("completed", artifacts=[{
                "type": "manifest", "path": "../manifest.json",
            }])

    def test_evaluation_contract_and_unique_artifact_types(self):
        evaluation = {
            "status": "passed", "deterministic": True,
            "policy_passed": True, "episode_count": 100,
            "success_count": 80, "success_rate": 0.8,
            "wilson_95_lower_bound": 0.7111708344068411,
            "wilson_95_upper_bound": 0.8666330666689676,
            "mean_episode_return": 1.25, "mean_episode_length": 240.0,
            "mean_final_distance": 0.2, "mean_minimum_distance": 0.1,
            "mean_object_displacement": 0.15,
            "timeout_rate": 0.2, "failure_rate": 0.0,
        }
        self.assertEqual("passed", callback("completed", evaluation=evaluation).evaluation["status"])
        with self.assertRaisesRegex(ValueError, "each type"):
            callback("completed", artifacts=[
                {"type": "manifest", "path": "manifest.json"},
                {"type": "manifest", "path": "manifest-copy.json"},
            ])


class SequenceAndBackendTests(unittest.TestCase):
    def test_same_run_strictly_increases_and_runs_are_independent(self):
        sequences = RunSequenceGenerator()
        self.assertEqual([1, 2, 3], [sequences.next("a") for _ in range(3)])
        self.assertEqual(1, sequences.next("b"))
        self.assertEqual(4, sequences.next("a"))

    def test_factory_ids_are_unique_and_callback_shares_sequence_space(self):
        factory = BackendMessageFactory()
        first = factory.event("state", "run-a", "task-a", {
            "previous_status": "queued", "status": "initializing"
        }, occurred_at=NOW)
        second = factory.event("heartbeat", "run-a", "task-a", {
            "status": "running", "training_steps": 0
        }, occurred_at=NOW)
        terminal = factory.terminal_callback(
            "run-a", "task-a", status=TerminalStatus.STOPPED,
            started_at=NOW, ended_at=LATER, training_steps=0, update_steps=0,
            stop_reason="requested_by_user", metrics={}, model=None,
            artifacts=[], error=None,
        )
        self.assertEqual([1, 2, 3], [first.sequence, second.sequence, terminal.sequence])
        self.assertEqual(3, len({first.event_id, second.event_id, terminal.event_id}))
        self.assertIs(first, first)  # Re-publishing the same logical object preserves its ID.

    def test_recording_backend_records_and_filters(self):
        backend = RecordingBackendClient()
        factory = BackendMessageFactory()
        event = factory.event("heartbeat", "run-a", "task-a", {
            "status": "running", "training_steps": 1
        }, occurred_at=NOW)
        terminal = factory.terminal_callback(
            "run-b", "task-b", status="failed", started_at=NOW, ended_at=LATER,
            training_steps=1, update_steps=0, stop_reason=None, metrics={}, model=None,
            artifacts=[], error={"code": "X", "message": "x", "retryable": False},
        )
        backend.publish_event(event)
        backend.publish_terminal_callback(terminal)
        self.assertEqual([event], backend.events_for_run("run-a"))
        self.assertEqual([], backend.events_for_run("run-b"))
        self.assertEqual([terminal], backend.terminal_callbacks_for_run("run-b"))


if __name__ == "__main__":
    unittest.main()
