from copy import deepcopy
import unittest
from datetime import datetime, timezone

from robotics.contracts import RunRequest, RunStatus as LegacyRunStatus
from robotics.runs import (
    ExecutionMode,
    LegacyCompatibilityError,
    Run,
    RunError,
    RunEvent,
    RunEventType,
    RunKind,
    RunStatus,
    legacy_request_to_create_command,
    to_legacy_run_event,
    to_legacy_run_record,
    to_legacy_status,
)


UTC_TIME = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)


def make_run(execution_mode=ExecutionMode.MOCK, **changes):
    values = {
        "run_id": "run-compat-1",
        "user_id": "user-1",
        "project_id": "project-1",
        "status": RunStatus.RUNNING,
        "execution_mode": execution_mode,
        "code_sha256": "a" * 64,
        "parameters": {"seed": 17, "max_steps": 240, "config": {"lr": 0.1}},
        "project_config_snapshot": {
            "environment_id": "mock-reach",
            "strategy_id": "joint-p-control",
            "config": {"lr": 0.1},
        },
        "created_at": UTC_TIME,
        "updated_at": UTC_TIME,
    }
    values.update(changes)
    return Run(**values)


def make_event(event_type, payload):
    return RunEvent(
        event_id="event-compat-1",
        sequence=9,
        type=event_type,
        run_id="run-compat-1",
        task_id=None,
        occurred_at=UTC_TIME,
        payload=payload,
    )


class RunCompatibilityTest(unittest.TestCase):
    def test_all_canonical_statuses_project_to_legacy_statuses(self):
        expected = {
            RunStatus.QUEUED: LegacyRunStatus.QUEUED,
            RunStatus.INITIALIZING: LegacyRunStatus.QUEUED,
            RunStatus.RUNNING: LegacyRunStatus.RUNNING,
            RunStatus.FINALIZING: LegacyRunStatus.RUNNING,
            RunStatus.STOPPING: LegacyRunStatus.RUNNING,
            RunStatus.COMPLETED: LegacyRunStatus.COMPLETED,
            RunStatus.FAILED: LegacyRunStatus.FAILED,
            RunStatus.STOPPED: LegacyRunStatus.STOPPED,
        }
        for canonical, legacy in expected.items():
            with self.subTest(canonical=canonical):
                self.assertEqual(to_legacy_status(canonical), legacy)

    def test_unknown_status_fails_explicitly(self):
        with self.assertRaisesRegex(LegacyCompatibilityError, "unknown"):
            to_legacy_status("mystery")

    def test_legacy_request_becomes_create_command(self):
        request = RunRequest(
            environment_id="mock-reach",
            strategy_id="joint-p-control",
            seed=7,
            max_steps=80,
            config={"nested": {"scale": 0.4}},
        )
        original = deepcopy(request)
        command = legacy_request_to_create_command(
            request,
            user_id="caller-user",
            project_id="caller-project",
            execution_mode=ExecutionMode.MOCK,
            code="policy = 'demo'\n",
        )
        self.assertEqual(command["user_id"], "caller-user")
        self.assertEqual(command["project_id"], "caller-project")
        self.assertEqual(command["execution_mode"], ExecutionMode.MOCK)
        self.assertEqual(command["run_kind"], RunKind.TRAIN)
        self.assertEqual(
            command["parameters"],
            {"seed": 7, "max_steps": 80, "config": {"nested": {"scale": 0.4}}},
        )
        self.assertEqual(
            command["project_config_snapshot"],
            {
                "environment_id": "mock-reach",
                "strategy_id": "joint-p-control",
                "config": {"nested": {"scale": 0.4}},
            },
        )
        self.assertEqual(command["code"], "policy = 'demo'\n")
        self.assertEqual(request, original)

    def test_request_config_is_deep_copied(self):
        request = RunRequest("mock-reach", "joint-p-control", config={"x": {"y": 1}})
        command = legacy_request_to_create_command(
            request,
            user_id="user",
            project_id="project",
            execution_mode=ExecutionMode.MOCK,
        )
        request.config["x"]["y"] = 2
        command["parameters"]["config"]["x"]["y"] = 3
        self.assertEqual(command["project_config_snapshot"]["config"]["x"]["y"], 1)

    def test_request_adapter_does_not_validate_training_ranges(self):
        request = RunRequest(
            "mock-reach", "joint-p-control", seed=-1, max_steps=0
        )
        command = legacy_request_to_create_command(
            request,
            user_id="user",
            project_id="project",
            execution_mode=ExecutionMode.MOCK,
        )
        self.assertEqual(command["parameters"]["seed"], -1)
        self.assertEqual(command["parameters"]["max_steps"], 0)

    def test_caller_must_supply_identity(self):
        request = RunRequest("mock-reach", "joint-p-control")
        for user_id, project_id in (("", "project"), ("user", "")):
            with self.subTest(user_id=user_id, project_id=project_id):
                with self.assertRaises(LegacyCompatibilityError):
                    legacy_request_to_create_command(
                        request,
                        user_id=user_id,
                        project_id=project_id,
                        execution_mode=ExecutionMode.MOCK,
                    )

    def test_canonical_mock_run_becomes_legacy_record(self):
        run = make_run(
            started_at=UTC_TIME,
            ended_at=UTC_TIME,
            error=RunError("DEMO", "legacy message", False),
        )
        legacy = to_legacy_run_record(run)
        self.assertEqual(legacy.run_id, run.run_id)
        self.assertEqual(legacy.mode, "train")
        self.assertEqual(legacy.environment_id, "mock-reach")
        self.assertEqual(legacy.strategy_id, "joint-p-control")
        self.assertEqual(legacy.seed, 17)
        self.assertEqual(legacy.max_steps, 240)
        self.assertEqual(legacy.status, LegacyRunStatus.RUNNING)
        self.assertEqual(legacy.started_at, UTC_TIME.isoformat())
        self.assertEqual(legacy.ended_at, UTC_TIME.isoformat())
        self.assertEqual(legacy.error, "legacy message")

    def test_execution_mode_does_not_replace_legacy_run_kind(self):
        legacy = to_legacy_run_record(
            make_run(ExecutionMode.SIMULATION, run_kind=RunKind.PLAY)
        )
        self.assertEqual(legacy.mode, "play")
        self.assertEqual(
            make_run(ExecutionMode.SIMULATION).execution_mode,
            ExecutionMode.SIMULATION,
        )

    def test_missing_frozen_run_field_fails(self):
        run = make_run(
            project_config_snapshot={
                "strategy_id": "joint-p-control",
                "config": {},
            }
        )
        with self.assertRaisesRegex(LegacyCompatibilityError, "environment_id"):
            to_legacy_run_record(run)

    def test_state_event_becomes_status_with_projected_payload(self):
        event = make_event(
            RunEventType.STATE,
            {"previous_status": "initializing", "status": "finalizing", "detail": "ok"},
        )
        legacy = to_legacy_run_event(event)
        self.assertEqual(legacy.type, "status")
        self.assertEqual(legacy.payload["previous_status"], "queued")
        self.assertEqual(legacy.payload["status"], "running")
        self.assertEqual(legacy.payload["detail"], "ok")
        self.assertEqual(legacy.timestamp, UTC_TIME.isoformat())
        self.assertEqual(legacy.version, 1)
        self.assertEqual(legacy.event_id, event.event_id)

    def test_metric_event_preserves_payload(self):
        payload = {"step": 8, "metrics": {"average_reward": 0.9}}
        legacy = to_legacy_run_event(make_event(RunEventType.METRIC, payload))
        self.assertEqual(legacy.type, "metric")
        self.assertEqual(legacy.payload, payload)

    def test_error_event_preserves_payload(self):
        payload = {"code": "FAILED", "message": "worker failed", "retryable": False}
        legacy = to_legacy_run_event(make_event(RunEventType.ERROR, payload))
        self.assertEqual(legacy.type, "error")
        self.assertEqual(legacy.payload, payload)

    def test_heartbeat_is_explicitly_skipped(self):
        event = make_event(
            RunEventType.HEARTBEAT,
            {"status": "running", "training_steps": 12},
        )
        self.assertIsNone(to_legacy_run_event(event))

    def test_event_payload_is_not_modified(self):
        event = make_event(
            RunEventType.STATE,
            {"previous_status": "initializing", "status": "stopping", "nested": {"x": 1}},
        )
        original = deepcopy(event.payload)
        legacy = to_legacy_run_event(event)
        legacy.payload["nested"]["x"] = 2
        self.assertEqual(event.payload, original)

    def test_state_event_missing_status_fails(self):
        event = make_event(RunEventType.STATE, {"previous_status": "queued"})
        with self.assertRaisesRegex(LegacyCompatibilityError, "status"):
            to_legacy_run_event(event)


if __name__ == "__main__":
    unittest.main()
