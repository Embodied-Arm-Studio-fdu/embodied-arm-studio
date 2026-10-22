import unittest
from datetime import datetime, timedelta, timezone

from robotics.runs import (
    ExecutionMode,
    Run,
    RunEvent,
    RunEventType,
    RunKind,
    RunStatus,
)


UTC_TIME = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)


def make_run(run_id="run-1", execution_mode=ExecutionMode.MOCK, **changes):
    values = {
        "run_id": run_id,
        "user_id": "user-1",
        "project_id": "project-1",
        "status": RunStatus.QUEUED,
        "execution_mode": execution_mode,
        "code_sha256": "a" * 64,
        "created_at": UTC_TIME,
        "updated_at": UTC_TIME,
    }
    values.update(changes)
    return Run(**values)


def make_event(**changes):
    values = {
        "event_id": "event-1",
        "sequence": 1,
        "type": RunEventType.STATE,
        "run_id": "run-1",
        "occurred_at": UTC_TIME,
        "payload": {"status": "queued"},
    }
    values.update(changes)
    return RunEvent(**values)


class RunModelsTest(unittest.TestCase):
    def test_create_valid_mock_run(self):
        run = make_run(parameters={"seed": 7})
        self.assertEqual(run.execution_mode, ExecutionMode.MOCK)
        self.assertIsNone(run.simulation_task_id)
        self.assertEqual(run.parameters, {"seed": 7})

    def test_create_valid_simulation_run_without_task_id(self):
        run = make_run(
            execution_mode=ExecutionMode.SIMULATION, class_id="class-1"
        )
        self.assertEqual(run.execution_mode, ExecutionMode.SIMULATION)
        self.assertEqual(run.class_id, "class-1")
        self.assertIsNone(run.simulation_task_id)

    def test_historical_run_may_have_null_class_id(self):
        self.assertIsNone(make_run().class_id)

    def test_progress_out_of_range_fails(self):
        for progress in (-0.01, 1.01, True):
            with self.subTest(progress=progress):
                with self.assertRaises(ValueError):
                    make_run(progress=progress)

    def test_event_sequence_must_be_positive(self):
        for sequence in (0, -1, True):
            with self.subTest(sequence=sequence):
                with self.assertRaises(ValueError):
                    make_event(sequence=sequence)

    def test_datetimes_must_be_timezone_aware_utc(self):
        invalid_times = (
            datetime(2026, 8, 23, 12, 0),
            datetime(2026, 8, 23, 12, 0, tzinfo=timezone(timedelta(hours=8))),
        )
        for timestamp in invalid_times:
            with self.subTest(timestamp=timestamp):
                with self.assertRaises(ValueError):
                    make_run(created_at=timestamp)
                with self.assertRaises(ValueError):
                    make_event(occurred_at=timestamp)

    def test_mutable_defaults_are_not_shared(self):
        first = make_run("run-1")
        second = make_run("run-2")
        first.parameters["seed"] = 9
        first.project_config_snapshot["scene"] = "push"
        first.metrics_summary["reward"] = 1
        self.assertEqual(second.parameters, {})
        self.assertEqual(second.project_config_snapshot, {})
        self.assertEqual(second.metrics_summary, {})

        first_event = make_event(payload={})
        second_event = make_event(event_id="event-2", payload={})
        first_event.payload["step"] = 1
        self.assertEqual(second_event.payload, {})

    def test_unknown_execution_mode_fails(self):
        with self.assertRaises(ValueError):
            make_run(execution_mode="real_robot")

    def test_run_kind_is_separate_from_execution_mode(self):
        run = make_run(run_kind=RunKind.PLAY)
        self.assertEqual(run.run_kind, RunKind.PLAY)
        self.assertEqual(run.execution_mode, ExecutionMode.MOCK)

    def test_unknown_run_kind_fails(self):
        with self.assertRaisesRegex(ValueError, "Run kind"):
            make_run(run_kind="evaluate")

    def test_unsupported_schema_version_fails(self):
        with self.assertRaises(ValueError):
            make_event(schema_version="2.0")


if __name__ == "__main__":
    unittest.main()
