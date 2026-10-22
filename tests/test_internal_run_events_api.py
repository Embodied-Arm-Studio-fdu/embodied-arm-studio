import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from sqlalchemy import delete, func, select

from apps.api.database import create_postgres_engine, create_session_factory
from apps.api.main import create_app
from tests.auth import StaticCurrentUserProvider
from robotics.runs import (
    ApplyEventResult,
    ExecutionMode,
    InMemoryRunRepository,
    RunEvent,
    RunEventType,
    RunKind,
    RunStatus,
)
from robotics.runs.db_models import RunEventRow, RunRow
from tests.http_client import ASGITestClient
from tests.postgres_users import delete_test_user, seed_test_user
from tests.postgres_safety import safe_test_database_url


TEST_SERVICE_TOKEN = "phase-5-2-explicit-test-token"
OCCURRED_AT = "2026-08-27T10:00:00Z"
SAFE_POSTGRES_URL = safe_test_database_url()


class InternalRunEventsApiTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.app = create_app(
            root / "runs",
            tick_seconds=0,
            project_root=root / "projects",
            run_repository=InMemoryRunRepository(),
            internal_run_event_service_token=TEST_SERVICE_TOKEN,
            current_user_provider=StaticCurrentUserProvider("internal-api-test"),
        )
        self.client = ASGITestClient(self.app)
        self.service = self.app.state.run_service
        self.run = self.create_run()

    def tearDown(self):
        self.client.close()
        self.temp_dir.cleanup()

    def create_run(
        self,
        mode=ExecutionMode.SIMULATION,
        bind_task=True,
        task_id="task-1",
    ):
        run = self.service.create_run(
            user_id="internal-api-test",
            project_id="project-1",
            execution_mode=mode,
            run_kind=RunKind.TRAIN,
            parameters={"max_steps": 100},
            project_config_snapshot={
                "environment_id": "arm-push",
                "expected_training_steps": 100,
            },
            code="",
        )
        if bind_task and mode == ExecutionMode.SIMULATION:
            self.service.bind_simulation_task(run.run_id, task_id)
        return self.service.get_run(run.run_id)

    def event(self, event_type, sequence, payload, **changes):
        value = {
            "schema_version": "1.0",
            "event_id": "evt-%s" % sequence,
            "sequence": sequence,
            "type": event_type,
            "run_id": self.run.run_id,
            "task_id": "task-1",
            "occurred_at": OCCURRED_AT,
            "payload": payload,
        }
        value.update(changes)
        return value

    def post(self, payload, path_run_id=None, token=TEST_SERVICE_TOKEN):
        headers = {}
        if token is not None:
            headers["Authorization"] = "Bearer %s" % token
        return self.client.post(
            "/api/internal/v1/runs/%s/events"
            % (path_run_id or payload["run_id"]),
            json=payload,
            headers=headers,
        )

    def assert_accepted(self, response, *, duplicate=False, late=False):
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(
            response.json(),
            {"accepted": True, "duplicate": duplicate, "late": late},
        )

    def test_four_legal_event_types_and_canonical_aggregation(self):
        events = [
            self.event(
                "state",
                1,
                {"previous_status": "queued", "status": "initializing"},
            ),
            self.event(
                "state",
                2,
                {"previous_status": "initializing", "status": "running"},
            ),
            self.event(
                "heartbeat",
                3,
                {"status": "running", "training_steps": 20},
            ),
            self.event(
                "metric",
                4,
                {"step": 40, "metrics": {
                    "update_index": 2,
                    "average_reward": 0.82,
                    "custom": 7,
                }},
            ),
            self.event(
                "error",
                5,
                {"code": "WORKER_WARNING", "message": "bounded summary", "retryable": False},
            ),
        ]
        for event in events:
            self.assert_accepted(self.post(event))

        history = self.client.get(
            "/api/v1/runs/%s/events" % self.run.run_id
        )
        self.assertEqual(history.status_code, 200)
        self.assertEqual(
            [item["type"] for item in history.json()],
            ["state", "state", "heartbeat", "metric", "error"],
        )
        self.assertEqual(history.json()[3]["occurred_at"], OCCURRED_AT)

        detail = self.client.get("/api/v1/runs/%s" % self.run.run_id).json()
        self.assertEqual(detail["status"], "running")
        self.assertEqual(detail["last_sequence"], 5)
        self.assertEqual(detail["progress"], 0.4)
        self.assertEqual(detail["training_steps"], 40)
        self.assertEqual(
            detail["metrics_summary"],
            {
                "update_index": 2,
                "average_reward": 0.82,
                "custom": 7,
                "step": 40,
            },
        )
        self.assertEqual(
            history.json()[3]["payload"]["metrics"]["update_index"], 2
        )
        self.assertEqual(detail["error"]["code"], "WORKER_WARNING")

    def test_path_and_body_run_id_mismatch_is_400(self):
        response = self.post(
            self.event("heartbeat", 1, {"status": "running", "training_steps": 0}),
            path_run_id="other-run",
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["code"], "run_id_mismatch")

    def test_unknown_schema_type_and_extra_fields_are_rejected(self):
        base = self.event(
            "heartbeat", 1, {"status": "running", "training_steps": 0}
        )
        cases = []
        unknown_schema = deepcopy(base)
        unknown_schema["schema_version"] = "2.0"
        cases.append(unknown_schema)
        unknown_type = deepcopy(base)
        unknown_type["type"] = "artifact"
        cases.append(unknown_type)
        extra_top = deepcopy(base)
        extra_top["runtime"] = {"device": "cuda"}
        cases.append(extra_top)
        extra_payload = deepcopy(base)
        extra_payload["payload"]["tensor"] = [1, 2]
        cases.append(extra_payload)
        for payload in cases:
            with self.subTest(payload=payload):
                self.assertEqual(self.post(payload).status_code, 422)

    def test_non_utc_time_and_invalid_payloads_are_rejected(self):
        cases = [
            self.event(
                "heartbeat",
                1,
                {"status": "running", "training_steps": 0},
                occurred_at="2026-08-27T12:00:00+02:00",
            ),
            self.event("metric", 1, {"step": -1, "metrics": {"reward": 1}}),
            self.event("metric", 1, {"step": 1, "metrics": {"reward": True}}),
            self.event(
                "heartbeat", 1, {"status": "completed", "training_steps": 0}
            ),
            self.event(
                "error",
                1,
                {"code": "FAILED", "message": "token=not-redacted", "retryable": False},
            ),
            self.event(
                "state",
                1,
                {"previous_status": "running", "status": "completed"},
            ),
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                self.assertEqual(self.post(payload).status_code, 422)

    def test_update_index_is_optional_but_strict_when_present(self):
        legacy = self.event(
            "metric", 1, {"step": 16, "metrics": {"average_reward": 0.1}}
        )
        self.assert_accepted(self.post(legacy))
        for sequence, invalid in enumerate((0, -1, 1.0, "1", True), start=2):
            with self.subTest(invalid=invalid):
                candidate = self.event(
                    "metric",
                    sequence,
                    {
                        "step": 16,
                        "metrics": {"update_index": invalid},
                    },
                )
                self.assertEqual(self.post(candidate).status_code, 422)

    def test_unknown_mock_unbound_and_mismatched_runs_are_distinguished(self):
        heartbeat = self.event(
            "heartbeat", 1, {"status": "running", "training_steps": 0}
        )
        unknown = deepcopy(heartbeat)
        unknown["run_id"] = "missing-run"
        self.assertEqual(self.post(unknown).status_code, 404)

        mock_run = self.create_run(mode=ExecutionMode.MOCK, bind_task=False)
        mock_event = deepcopy(heartbeat)
        mock_event["run_id"] = mock_run.run_id
        self.assertEqual(self.post(mock_event).status_code, 409)

        unbound = self.create_run(bind_task=False)
        unbound_event = deepcopy(heartbeat)
        unbound_event["run_id"] = unbound.run_id
        self.assertEqual(self.post(unbound_event).status_code, 409)

        mismatch = deepcopy(heartbeat)
        mismatch["task_id"] = "task-other"
        self.assertEqual(self.post(mismatch).status_code, 409)

    def test_identical_event_replay_succeeds_without_side_effects(self):
        event = self.event(
            "metric", 1, {"step": 50, "metrics": {"average_reward": 1.0}}
        )
        self.assert_accepted(self.post(event))
        before = self.service.get_run(self.run.run_id)
        self.assert_accepted(self.post(event), duplicate=True)
        self.assertEqual(self.service.get_run(self.run.run_id), before)
        self.assertEqual(len(self.service.list_events(self.run.run_id)), 1)

    def test_duplicate_repairs_snapshot_after_prior_append_only(self):
        self.service.transition_status(self.run.run_id, RunStatus.INITIALIZING)
        self.service.transition_status(self.run.run_id, RunStatus.RUNNING)
        event = RunEvent(
            schema_version="1.0",
            event_id="append-only-event",
            sequence=4,
            type=RunEventType.METRIC,
            run_id=self.run.run_id,
            task_id="task-1",
            occurred_at=self.run.created_at,
            payload={"step": 30, "metrics": {"average_reward": 0.3}},
        )
        self.app.state.repository.append_event(event)
        self.assertEqual(
            self.service.apply_simulation_event(event),
            ApplyEventResult.DUPLICATE,
        )
        repaired = self.service.get_run(self.run.run_id)
        self.assertEqual(repaired.last_sequence, 4)
        self.assertEqual(repaired.progress, 0.3)
        self.assertEqual(repaired.metrics_summary["average_reward"], 0.3)

    def test_event_id_and_sequence_content_conflicts_are_409(self):
        first = self.event(
            "metric", 1, {"step": 1, "metrics": {"average_reward": 0.1}}
        )
        self.assert_accepted(self.post(first))

        event_id_conflict = self.event(
            "metric",
            2,
            {"step": 2, "metrics": {"average_reward": 0.2}},
            event_id=first["event_id"],
        )
        response = self.post(event_id_conflict)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["code"], "event_id_conflict")

        sequence_conflict = self.event(
            "metric",
            1,
            {"step": 9, "metrics": {"average_reward": 0.9}},
            event_id="evt-other",
        )
        response = self.post(sequence_conflict)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["code"], "sequence_conflict")

    def test_current_state_conflict_is_not_persisted(self):
        event = self.event(
            "state",
            1,
            {"previous_status": "initializing", "status": "running"},
        )
        response = self.post(event)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.json()["detail"]["code"], "event_state_conflict"
        )
        self.assertEqual(self.service.list_events(self.run.run_id), [])

    def test_sequence_gap_and_late_event_preserve_newer_snapshot(self):
        newer = self.event(
            "metric", 5, {"step": 80, "metrics": {"average_reward": 0.8}}
        )
        self.assert_accepted(self.post(newer))
        snapshot = self.service.get_run(self.run.run_id)
        older = self.event(
            "metric", 2, {"step": 10, "metrics": {"average_reward": 0.1}}
        )
        self.assert_accepted(self.post(older), late=True)
        self.assertEqual(self.service.get_run(self.run.run_id), snapshot)
        self.assertEqual(
            [event.sequence for event in self.service.list_events(self.run.run_id)],
            [2, 5],
        )

    def test_terminal_run_cannot_be_revived_by_higher_sequence_event(self):
        self.service.transition_status(self.run.run_id, RunStatus.INITIALIZING)
        self.service.transition_status(self.run.run_id, RunStatus.RUNNING)
        self.service.transition_status(self.run.run_id, RunStatus.FINALIZING)
        terminal = self.service.transition_status(self.run.run_id, RunStatus.COMPLETED)
        event = self.event(
            "state",
            8,
            {"previous_status": "queued", "status": "initializing"},
        )
        self.assert_accepted(self.post(event), late=True)
        after = self.service.get_run(self.run.run_id)
        self.assertEqual(after, terminal)
        self.assertEqual(after.status, RunStatus.COMPLETED)
        self.assertEqual(after.progress, 1.0)

    def test_configured_token_accepts_correct_and_rejects_missing_or_wrong(self):
        event = self.event(
            "heartbeat", 1, {"status": "running", "training_steps": 0}
        )
        missing = self.post(event, token=None)
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(missing.headers["www-authenticate"], "Bearer")
        self.assertEqual(self.post(event, token="wrong-test-token").status_code, 403)
        self.assert_accepted(self.post(event))


@unittest.skipUnless(
    SAFE_POSTGRES_URL is not None,
    "PostgreSQL test configuration is unavailable",
)
class InternalRunEventsPostgreSQLTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = create_postgres_engine(SAFE_POSTGRES_URL)
        cls.factory = create_session_factory(cls.engine)

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.user_id = "p52-%s" % uuid4().hex
        seed_test_user(self.factory, self.user_id)
        root = Path(self.temp_dir.name)
        self.app = create_app(
            root / "runs",
            tick_seconds=0,
            project_root=root / "projects",
            run_repository_mode="postgresql",
            database_url=SAFE_POSTGRES_URL,
            internal_run_event_service_token=TEST_SERVICE_TOKEN,
            current_user_provider=StaticCurrentUserProvider(self.user_id),
        )
        self.client = ASGITestClient(self.app)
        service = self.app.state.run_service
        run = service.create_run(
            user_id=self.user_id,
            project_id="project-1",
            execution_mode=ExecutionMode.SIMULATION,
            run_kind=RunKind.TRAIN,
            parameters={"max_steps": 100},
            project_config_snapshot={"expected_training_steps": 100},
            code="",
        )
        service.bind_simulation_task(run.run_id, "postgres-task-1")
        service.transition_status(run.run_id, RunStatus.INITIALIZING)
        service.transition_status(run.run_id, RunStatus.RUNNING)
        self.run_id = run.run_id

    def tearDown(self):
        self.client.close()
        with self.factory() as session:
            run_ids = select(RunRow.run_id).where(RunRow.user_id == self.user_id)
            session.execute(
                delete(RunEventRow).where(RunEventRow.run_id.in_(run_ids))
            )
            session.execute(delete(RunRow).where(RunRow.user_id == self.user_id))
            session.commit()
            residual = session.scalar(
                select(func.count()).select_from(RunRow).where(
                    RunRow.user_id == self.user_id
                )
            )
        self.assertEqual(residual, 0)
        delete_test_user(self.factory, self.user_id)
        self.temp_dir.cleanup()

    def test_event_round_trip_duplicate_and_aggregate(self):
        event = {
            "schema_version": "1.0",
            "event_id": "pg-event-%s" % uuid4().hex,
            "sequence": 3,
            "type": "metric",
            "run_id": self.run_id,
            "task_id": "postgres-task-1",
            "occurred_at": OCCURRED_AT,
            "payload": {
                "step": 25,
                "metrics": {"average_reward": 0.5},
            },
        }
        headers = {"Authorization": "Bearer %s" % TEST_SERVICE_TOKEN}
        url = "/api/internal/v1/runs/%s/events" % self.run_id
        first = self.client.post(url, json=event, headers=headers)
        replay = self.client.post(url, json=event, headers=headers)
        self.assertEqual(first.status_code, 202, first.text)
        self.assertEqual(replay.status_code, 202, replay.text)
        self.assertFalse(first.json()["duplicate"])
        self.assertTrue(replay.json()["duplicate"])
        history = self.client.get("/api/v1/runs/%s/events" % self.run_id)
        detail = self.client.get("/api/v1/runs/%s" % self.run_id)
        self.assertEqual(len(history.json()), 1)
        self.assertEqual(detail.json()["last_sequence"], 3)
        self.assertEqual(detail.json()["progress"], 0.25)


if __name__ == "__main__":
    unittest.main()
