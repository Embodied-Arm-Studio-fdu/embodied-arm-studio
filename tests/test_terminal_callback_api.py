import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from alembic import command as alembic_command
from alembic.config import Config
from sqlalchemy import delete, func, inspect, select

from apps.api.database import create_postgres_engine, create_session_factory
from apps.api.main import create_app
from apps.api.schemas.internal_runs import TerminalCallbackRequest
from tests.auth import StaticCurrentUserProvider
from robotics.runs import ExecutionMode, RunEvent, RunEventType, RunKind, RunStatus
from robotics.runs.db_models import (
    RunArtifactRow,
    RunEventRow,
    RunIdempotencyKeyRow,
    RunRow,
)
from robotics.runs.repository import RepositoryIntegrityError
from tests.http_client import ASGITestClient
from tests.postgres_users import delete_test_user, seed_test_user
from tests.postgres_safety import safe_test_database_url


ROOT = Path(__file__).resolve().parents[1]
TEST_SERVICE_TOKEN = "phase-5-3-explicit-test-token"
STARTED_AT = "2026-08-27T10:00:00Z"
ENDED_AT = "2026-08-27T10:30:00Z"
SHA256 = "a" * 64


SAFE_POSTGRES_URL = safe_test_database_url()


class TerminalCallbackVideoSchemaTest(unittest.TestCase):
    def test_nine_artifact_video_callback_is_accepted_without_extra_metadata(self):
        artifact_types = [
            "checkpoint_best", "checkpoint_latest", "training_config",
            "stderr_log", "stdout_log", "task_log", "training_metrics",
            "training_video", "manifest",
        ]
        artifacts = [
            {
                "type": artifact_type,
                "path": "video/overview.mp4" if artifact_type == "training_video"
                else "files/%s-%d.bin" % (artifact_type, index),
                "size_bytes": index + 1,
                "sha256": SHA256,
            }
            for index, artifact_type in enumerate(artifact_types)
        ]
        payload = {
            "schema_version": "1.0",
            "event_id": "callback-video-schema",
            "sequence": 7,
            "run_id": "run-video-schema",
            "task_id": "task-video-schema",
            "status": "completed",
            "started_at": STARTED_AT,
            "ended_at": ENDED_AT,
            "training_steps": 320,
            "update_steps": 20,
            "stop_reason": None,
            "metrics": {"average_reward": 0.5},
            "model": artifacts[0],
            "artifacts": artifacts,
            "error": None,
        }
        parsed = TerminalCallbackRequest.model_validate(payload)
        self.assertEqual(9, len(parsed.artifacts))
        self.assertEqual(
            1,
            sum(item.artifact_type == "training_video" for item in parsed.artifacts),
        )

    def test_production_evaluation_callback_is_strict_and_backward_compatible(self):
        base = {
            "schema_version": "1.0", "event_id": "callback-eval",
            "sequence": 8, "run_id": "run-eval", "task_id": "task-eval",
            "status": "completed", "started_at": STARTED_AT,
            "ended_at": ENDED_AT, "training_steps": 20000,
            "update_steps": 100, "stop_reason": None, "metrics": {},
            "model": {"type": "checkpoint_best", "path": "checkpoints/model_best.pt"},
            "artifacts": [{"type": "manifest", "path": "manifest.json"}],
            "error": None,
        }
        self.assertIsNone(TerminalCallbackRequest.model_validate(base).evaluation)
        base["evaluation"] = {
            "status": "passed", "deterministic": True,
            "policy_passed": True, "episode_count": 100,
            "success_count": 80, "success_rate": 0.8,
            "wilson_95_lower_bound": 0.7111708344068411,
            "wilson_95_upper_bound": 0.8666330666689676,
            "mean_episode_length": 240.0, "mean_episode_return": 1.25,
            "mean_final_distance": 0.2, "mean_minimum_distance": 0.1,
            "mean_object_displacement": 0.15,
            "timeout_rate": 0.2, "failure_rate": 0.0,
        }
        self.assertTrue(
            TerminalCallbackRequest.model_validate(base).evaluation.policy_passed
        )
        invalid = dict(base)
        invalid["evaluation"] = dict(base["evaluation"], policy_passed=False)
        with self.assertRaises(ValueError):
            TerminalCallbackRequest.model_validate(invalid)
        unavailable = dict(base)
        unavailable["evaluation"] = {
            "status": "unavailable", "deterministic": True,
            "policy_passed": False, "failure_reason": "evaluation_failed",
            "mean_final_distance": 0.0,
        }
        with self.assertRaises(ValueError):
            TerminalCallbackRequest.model_validate(unavailable)

    def test_absolute_parent_and_duplicate_type_artifacts_are_rejected(self):
        artifact = {"type": "manifest", "path": "manifest.json"}
        payload = {
            "schema_version": "1.0", "event_id": "callback-path",
            "sequence": 9, "run_id": "run-path", "task_id": "task-path",
            "status": "completed", "started_at": STARTED_AT,
            "ended_at": ENDED_AT, "training_steps": 1, "update_steps": 1,
            "stop_reason": None, "metrics": {},
            "model": {"type": "checkpoint_best", "path": "checkpoints/model_best.pt"},
            "artifacts": [artifact], "error": None,
        }
        for bad_path in ("/tmp/manifest.json", "../manifest.json"):
            with self.subTest(path=bad_path):
                bad = dict(payload, artifacts=[dict(artifact, path=bad_path)])
                with self.assertRaises(ValueError):
                    TerminalCallbackRequest.model_validate(bad)
        duplicate = dict(payload, artifacts=[artifact, dict(artifact, path="copy.json")])
        with self.assertRaises(ValueError):
            TerminalCallbackRequest.model_validate(duplicate)


@unittest.skipUnless(
    SAFE_POSTGRES_URL is not None,
    "PostgreSQL test configuration is unavailable",
)
class TerminalCallbackMigrationTest(unittest.TestCase):
    def test_upgrade_downgrade_upgrade_round_trip(self):
        engine = create_postgres_engine(SAFE_POSTGRES_URL)
        config = Config(str(ROOT / "alembic.ini"))
        environment = {
            "DATABASE_URL": SAFE_POSTGRES_URL.render_as_string(False)
        }
        database_has_run_facts = False
        try:
            with engine.connect() as connection:
                counts = []
                for table in ("runs", "run_events", "run_idempotency_keys"):
                    counts.append(
                        connection.scalar(select(func.count()).select_from(
                            {"runs": RunRow, "run_events": RunEventRow,
                             "run_idempotency_keys": RunIdempotencyKeyRow}[table]
                        ))
                    )
                database_has_run_facts = any(counts)
            with patch.dict(os.environ, environment, clear=False):
                alembic_command.upgrade(config, "head")
                # Downgrading to 0001 drops later tables and their rows.  Keep
                # this legacy round trip only for an empty test database; a
                # populated shared test target is validated non-destructively.
                if not database_has_run_facts:
                    alembic_command.downgrade(config, "20260824_0001")
            if not database_has_run_facts:
                old_inspector = inspect(engine)
                self.assertNotIn("run_artifacts", old_inspector.get_table_names())
                self.assertNotIn(
                    "training_steps",
                    {column["name"] for column in old_inspector.get_columns("runs")},
                )
        finally:
            with patch.dict(os.environ, environment, clear=False):
                alembic_command.upgrade(config, "head")
            engine.dispose()

        verify_engine = create_postgres_engine(SAFE_POSTGRES_URL)
        try:
            current = inspect(verify_engine)
            self.assertIn("run_artifacts", current.get_table_names())
            self.assertTrue(
                {"training_steps", "update_steps", "stop_reason"}.issubset(
                    {column["name"] for column in current.get_columns("runs")}
                )
            )
            self.assertIn(
                {"run_id", "path"},
                [
                    set(item["column_names"])
                    for item in current.get_unique_constraints("run_artifacts")
                ],
            )
        finally:
            verify_engine.dispose()


@unittest.skipUnless(
    SAFE_POSTGRES_URL is not None,
    "PostgreSQL test configuration is unavailable",
)
class TerminalCallbackPostgreSQLTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cleanup_engine = create_postgres_engine(SAFE_POSTGRES_URL)
        cls.factory = create_session_factory(cls.cleanup_engine)

    @classmethod
    def tearDownClass(cls):
        cls.cleanup_engine.dispose()

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.user_id = "p53-%s" % uuid4().hex
        seed_test_user(self.factory, self.user_id)
        self.apps = []
        self.clients = []
        self.app, self.client = self.make_app("primary")
        self.run = self.create_run()

    def tearDown(self):
        for client in reversed(self.clients):
            client.close()
        with self.factory() as session:
            run_ids = select(RunRow.run_id).where(RunRow.user_id == self.user_id)
            session.execute(
                delete(RunArtifactRow).where(RunArtifactRow.run_id.in_(run_ids))
            )
            session.execute(
                delete(RunIdempotencyKeyRow).where(
                    RunIdempotencyKeyRow.run_id.in_(run_ids)
                )
            )
            session.execute(
                delete(RunEventRow).where(RunEventRow.run_id.in_(run_ids))
            )
            session.execute(delete(RunRow).where(RunRow.user_id == self.user_id))
            session.commit()
            self.assertEqual(
                session.scalar(
                    select(func.count()).select_from(RunRow).where(
                        RunRow.user_id == self.user_id
                    )
                ),
                0,
            )
        self.temp_dir.cleanup()
        delete_test_user(self.factory, self.user_id)

    def make_app(self, name):
        root = Path(self.temp_dir.name)
        app = create_app(
            root / (name + "-runs"),
            tick_seconds=0,
            project_root=root / (name + "-projects"),
            run_repository_mode="postgresql",
            database_url=SAFE_POSTGRES_URL,
            current_user_provider=StaticCurrentUserProvider(self.user_id),
            internal_run_event_service_token=TEST_SERVICE_TOKEN,
        )
        client = ASGITestClient(app)
        self.apps.append(app)
        self.clients.append(client)
        return app, client

    def create_run(
        self,
        mode=ExecutionMode.SIMULATION,
        bind=True,
        task_id="task-1",
    ):
        service = self.app.state.run_service
        run = service.create_run(
            user_id=self.user_id,
            project_id="project-1",
            execution_mode=mode,
            run_kind=RunKind.TRAIN,
            parameters={"max_steps": 100},
            project_config_snapshot={"expected_training_steps": 100},
            code="",
        )
        if bind and mode == ExecutionMode.SIMULATION:
            service.bind_simulation_task(run.run_id, task_id)
        return service.get_run(run.run_id)

    def payload(self, run=None, **changes):
        run = run or self.run
        value = {
            "schema_version": "1.0",
            "event_id": "callback-%s" % uuid4().hex,
            "sequence": 7,
            "run_id": run.run_id,
            "task_id": "task-1",
            "status": "completed",
            "started_at": STARTED_AT,
            "ended_at": ENDED_AT,
            "training_steps": 80,
            "update_steps": 20,
            "stop_reason": None,
            "metrics": {"average_reward": 0.82},
            "model": {
                "type": "checkpoint_best",
                "path": "checkpoints/model_best.pt",
                "size_bytes": 42,
                "sha256": SHA256,
            },
            "artifacts": [
                {
                    "type": "checkpoint_best",
                    "path": "checkpoints/model_best.pt",
                    "sha256": SHA256,
                },
                {"type": "manifest", "path": "manifest.json", "size_bytes": 12},
            ],
            "error": None,
        }
        value.update(changes)
        return value

    def failed_payload(self, run=None, **changes):
        value = self.payload(
            run,
            status="failed",
            model=None,
            artifacts=[],
            metrics={},
            error={
                "code": "TRAINING_FAILED",
                "message": "training process exited",
                "retryable": False,
            },
        )
        value.update(changes)
        return value

    def stopped_payload(self, run=None, **changes):
        value = self.payload(
            run,
            status="stopped",
            model=None,
            artifacts=[],
            stop_reason="requested_by_user",
            metrics={},
        )
        value.update(changes)
        return value

    def post(self, payload, client=None, path_run_id=None, token=TEST_SERVICE_TOKEN):
        headers = {}
        if token is not None:
            headers["Authorization"] = "Bearer %s" % token
        return (client or self.client).post(
            "/api/internal/v1/runs/%s/callback"
            % (path_run_id or payload["run_id"]),
            json=payload,
            headers=headers,
        )

    def assert_success(self, response, duplicate=False):
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(
            response.json(), {"accepted": True, "duplicate": duplicate}
        )

    def test_completed_converges_from_queued_and_merges_primary_model(self):
        callback = self.payload(sequence=11)
        self.assert_success(self.post(callback))
        saved = self.app.state.run_service.get_run(self.run.run_id)
        self.assertEqual(saved.status, RunStatus.COMPLETED)
        self.assertEqual(saved.progress, 1.0)
        self.assertEqual(saved.last_sequence, 11)
        self.assertEqual(saved.training_steps, 80)
        self.assertEqual(saved.update_steps, 20)
        self.assertEqual(saved.metrics_summary["average_reward"], 0.82)
        self.assertEqual(saved.metrics_summary["training_steps"], 80)
        self.assertEqual(saved.started_at.isoformat(), "2026-08-27T10:00:00+00:00")
        self.assertEqual(saved.ended_at.isoformat(), "2026-08-27T10:30:00+00:00")

        events = self.client.get(
            "/api/v1/runs/%s/events" % self.run.run_id
        ).json()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "state")
        self.assertEqual(events[0]["payload"]["status"], "completed")
        detail = self.client.get(
            "/api/v1/runs/%s" % self.run.run_id
        ).json()
        self.assertEqual(detail["training_steps"], 80)
        self.assertEqual(detail["update_steps"], 20)
        self.assertIsNone(detail["stop_reason"])
        artifacts = self.client.get(
            "/api/v1/runs/%s/artifacts" % self.run.run_id
        ).json()
        self.assertEqual(len(artifacts), 2)
        model = next(item for item in artifacts if item["is_primary_model"])
        self.assertEqual(model["path"], "checkpoints/model_best.pt")
        self.assertEqual(model["size_bytes"], 42)

    def test_completed_callback_registers_exactly_one_video_as_ninth_artifact(self):
        artifact_types = [
            "checkpoint_best", "checkpoint_latest", "training_config",
            "stderr_log", "stdout_log", "task_log", "training_metrics",
            "training_video", "manifest",
        ]
        artifacts = [
            {
                "type": artifact_type,
                "path": (
                    "video/overview.mp4"
                    if artifact_type == "training_video"
                    else "files/%s-%d.bin" % (artifact_type, index)
                ),
                "size_bytes": index + 1,
                "sha256": SHA256,
            }
            for index, artifact_type in enumerate(artifact_types)
        ]
        callback = self.payload(
            artifacts=artifacts,
            model=artifacts[0],
        )
        self.assert_success(self.post(callback))
        registered = self.client.get(
            "/api/v1/runs/%s/artifacts" % self.run.run_id
        ).json()
        self.assertEqual(9, len(registered))
        self.assertEqual(
            1,
            sum(item["artifact_type"] == "training_video" for item in registered),
        )

    def test_evaluation_is_persisted_inside_metrics_summary(self):
        evaluation = {
            "status": "passed", "deterministic": True,
            "policy_passed": True, "episode_count": 100,
            "success_count": 80, "success_rate": 0.8,
            "wilson_95_lower_bound": 0.7111708344068411,
            "wilson_95_upper_bound": 0.8666330666689676,
        }
        self.assert_success(self.post(self.payload(evaluation=evaluation)))
        saved = self.app.state.run_service.get_run(self.run.run_id)
        self.assertEqual(evaluation, saved.metrics_summary["evaluation"])

    def test_failed_and_stopped_terminal_facts(self):
        failed_run = self.create_run(task_id="task-failed")
        failed = self.failed_payload(
            failed_run, task_id="task-failed", sequence=3
        )
        self.assert_success(self.post(failed))
        failed_saved = self.app.state.run_service.get_run(failed_run.run_id)
        self.assertEqual(failed_saved.status, RunStatus.FAILED)
        self.assertEqual(failed_saved.error.code, "TRAINING_FAILED")
        self.assertEqual(failed_saved.progress, 0.8)

        stopped_run = self.create_run(task_id="task-stopped")
        stopped = self.stopped_payload(
            stopped_run, task_id="task-stopped", sequence=9
        )
        self.assert_success(self.post(stopped))
        stopped_saved = self.app.state.run_service.get_run(stopped_run.run_id)
        self.assertEqual(stopped_saved.status, RunStatus.STOPPED)
        self.assertEqual(stopped_saved.stop_reason, "requested_by_user")
        self.assertEqual(stopped_saved.progress, 0.8)

    def test_failed_progress_caps_and_training_steps_do_not_regress(self):
        service = self.app.state.run_service
        service.transition_status(self.run.run_id, RunStatus.INITIALIZING)
        service.transition_status(self.run.run_id, RunStatus.RUNNING)
        service.apply_simulation_event(
            RunEvent(
                event_id="live-before-callback",
                sequence=3,
                type=RunEventType.METRIC,
                run_id=self.run.run_id,
                task_id="task-1",
                occurred_at=self.run.created_at,
                payload={
                    "step": 120,
                    "metrics": {
                        "update_index": 3,
                        "average_reward": 0.5,
                    },
                },
            )
        )
        callback = self.failed_payload(
            sequence=7,
            training_steps=80,
            update_steps=2,
            metrics={"update_index": 2, "average_reward": 0.4},
        )
        self.assert_success(self.post(callback))
        saved = service.get_run(self.run.run_id)
        self.assertEqual(saved.training_steps, 120)
        self.assertEqual(saved.progress, 0.99)
        self.assertEqual(saved.update_steps, 2)
        self.assertEqual(saved.metrics_summary["training_steps"], 120)
        self.assertEqual(saved.metrics_summary["update_steps"], 2)
        self.assertEqual(saved.metrics_summary["update_index"], 2)

    def test_zero_update_terminal_callback_may_omit_update_index(self):
        callback = self.stopped_payload(
            training_steps=10, update_steps=0, metrics={}
        )
        self.assert_success(self.post(callback))
        saved = self.app.state.run_service.get_run(self.run.run_id)
        self.assertEqual(saved.progress, 0.1)
        self.assertNotIn("update_index", saved.metrics_summary)

    def test_terminal_conditional_requirements(self):
        cases = [
            self.payload(model=None),
            self.payload(artifacts=[]),
            self.failed_payload(error=None),
            self.stopped_payload(stop_reason=None),
        ]
        for callback in cases:
            with self.subTest(callback=callback):
                self.assertEqual(self.post(callback).status_code, 422)

    def test_time_count_sha_and_path_validation(self):
        cases = [
            self.payload(started_at="2026-08-27T12:00:00+02:00"),
            self.payload(ended_at="2026-08-27T09:00:00Z"),
            self.payload(training_steps=-1),
            self.payload(update_steps=-1),
            self.payload(metrics={"update_index": 0}),
            self.payload(metrics={"update_index": -1}),
            self.payload(metrics={"update_index": 1.0}),
            self.payload(metrics={"update_index": "1"}),
            self.payload(metrics={"update_index": True}),
            self.payload(model={"type": "best", "path": "model.pt", "size_bytes": -1}),
            self.payload(model={"type": "best", "path": "model.pt", "sha256": "xyz"}),
            self.payload(model={"type": "best", "path": "/home/user/model.pt"}),
            self.payload(model={"type": "best", "path": "../model.pt"}),
            self.payload(model={"type": "best", "path": "checkpoints\\model.pt"}),
            self.failed_payload(
                error={"code": "FAILED", "message": "token=secret", "retryable": False}
            ),
        ]
        for callback in cases:
            with self.subTest(callback=callback):
                self.assertEqual(self.post(callback).status_code, 422)

    def test_run_and_task_conflicts(self):
        unknown = self.payload()
        unknown["run_id"] = "missing-run"
        self.assertEqual(self.post(unknown).status_code, 404)

        mock_run = self.create_run(mode=ExecutionMode.MOCK, bind=False)
        self.assertEqual(self.post(self.payload(mock_run)).status_code, 409)
        unbound = self.create_run(bind=False)
        self.assertEqual(self.post(self.payload(unbound)).status_code, 409)
        mismatch = self.payload(task_id="task-other")
        self.assertEqual(self.post(mismatch).status_code, 409)

    def test_exact_replay_is_idempotent_and_preserves_facts(self):
        callback = self.payload()
        self.assert_success(self.post(callback))
        before = self.app.state.run_service.get_run(self.run.run_id)
        artifacts_before = self.client.get(
            "/api/v1/runs/%s/artifacts" % self.run.run_id
        ).json()
        self.assert_success(self.post(callback), duplicate=True)
        self.assertEqual(self.app.state.run_service.get_run(self.run.run_id), before)
        self.assertEqual(
            self.client.get(
                "/api/v1/runs/%s/artifacts" % self.run.run_id
            ).json(),
            artifacts_before,
        )

    def test_event_id_sequence_and_terminal_conflicts(self):
        callback = self.payload()
        self.assert_success(self.post(callback))
        changed = deepcopy(callback)
        changed["training_steps"] = 81
        self.assertEqual(self.post(changed).status_code, 409)

        terminal_conflict = self.failed_payload(
            event_id="different-terminal-event", sequence=8
        )
        self.assertEqual(self.post(terminal_conflict).status_code, 409)

        occupied_run = self.create_run(task_id="task-occupied")
        self.app.state.run_service.apply_simulation_event(
            RunEvent(
                event_id="occupied-event",
                sequence=5,
                type=RunEventType.HEARTBEAT,
                run_id=occupied_run.run_id,
                task_id="task-occupied",
                occurred_at=occupied_run.created_at,
                payload={"status": "running", "training_steps": 0},
            )
        )
        occupied = self.payload(
            occupied_run, task_id="task-occupied", sequence=5
        )
        self.assertEqual(self.post(occupied).status_code, 409)

    def test_transaction_failure_rolls_back_every_fact_and_retry_succeeds(self):
        callback = self.payload()

        def fail_before_commit(_session):
            raise RepositoryIntegrityError("forced_callback_failure")

        callback_service = self.app.state.terminal_callback_service
        callback_service._before_commit = fail_before_commit
        failed = self.post(callback)
        callback_service._before_commit = None
        self.assertEqual(failed.status_code, 503)
        saved = self.app.state.run_service.get_run(self.run.run_id)
        self.assertEqual(saved.status, RunStatus.QUEUED)
        self.assertEqual(saved.last_sequence, 0)
        self.assertEqual(
            self.client.get(
                "/api/v1/runs/%s/events" % self.run.run_id
            ).json(),
            [],
        )
        self.assertEqual(
            self.client.get(
                "/api/v1/runs/%s/artifacts" % self.run.run_id
            ).json(),
            [],
        )
        self.assert_success(self.post(callback))

    def test_terminal_run_rejects_late_event_snapshot_regression(self):
        callback = self.payload(sequence=10)
        self.assert_success(self.post(callback))
        before = self.app.state.run_service.get_run(self.run.run_id)
        late = {
            "schema_version": "1.0",
            "event_id": "late-heartbeat",
            "sequence": 12,
            "type": "heartbeat",
            "run_id": self.run.run_id,
            "task_id": "task-1",
            "occurred_at": ENDED_AT,
            "payload": {"status": "running", "training_steps": 10},
        }
        response = self.client.post(
            "/api/internal/v1/runs/%s/events" % self.run.run_id,
            json=late,
            headers={"Authorization": "Bearer %s" % TEST_SERVICE_TOKEN},
        )
        self.assertEqual(response.status_code, 202)
        self.assertTrue(response.json()["late"])
        self.assertEqual(
            self.app.state.run_service.get_run(self.run.run_id), before
        )

    def test_app_rebuild_recovers_run_event_and_artifacts(self):
        callback = self.payload()
        self.assert_success(self.post(callback))
        expected_events = self.client.get(
            "/api/v1/runs/%s/events" % self.run.run_id
        ).json()
        expected_artifacts = self.client.get(
            "/api/v1/runs/%s/artifacts" % self.run.run_id
        ).json()
        second_app, second = self.make_app("rebuilt")
        self.assertIsNot(
            second_app.state.terminal_callback_service,
            self.app.state.terminal_callback_service,
        )
        self.assertEqual(
            second.get("/api/v1/runs/%s" % self.run.run_id).json()["status"],
            "completed",
        )
        self.assertEqual(
            second.get("/api/v1/runs/%s/events" % self.run.run_id).json(),
            expected_events,
        )
        self.assertEqual(
            second.get("/api/v1/runs/%s/artifacts" % self.run.run_id).json(),
            expected_artifacts,
        )

    def test_authentication_and_path_body_match(self):
        callback = self.payload()
        self.assertEqual(self.post(callback, token=None).status_code, 401)
        self.assertEqual(self.post(callback, token="wrong").status_code, 403)
        self.assertEqual(
            self.post(callback, path_run_id="other-run").status_code, 400
        )
        self.assert_success(self.post(callback))


if __name__ == "__main__":
    unittest.main()
