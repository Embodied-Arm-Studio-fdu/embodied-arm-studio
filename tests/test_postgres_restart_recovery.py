import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from sqlalchemy import delete, func, select

from apps.api.database import create_postgres_engine, create_session_factory
from apps.api.main import create_app
from tests.auth import StaticCurrentUserProvider
from robotics.projects.models import (
    TrainingProfile,
    evaluation_rubric,
    parameter_schema,
    starter_code_digest,
)
from robotics.runs import (
    PostgreSQLIdempotentRunCreationService,
    PostgreSQLRunRepository,
    RunEvent,
    RunEventType,
)
from robotics.runs.db_models import (
    CourseRow,
    OrganizationMembershipRow,
    OrganizationRow,
    RunEventRow,
    RunIdempotencyKeyRow,
    RunRow,
    TeachingClassRow,
    TeachingProjectRow,
    UserRow,
)
from robotics.simulation import SimulationTaskResponse
from tests.http_client import ASGITestClient
from tests.postgres_safety import safe_test_database_url


SAFE_TEST_DATABASE_URL = safe_test_database_url()


class RestartSimulationClient:
    def create_task(self, run_id, project_id, config):
        now = datetime.now(timezone.utc)
        return SimulationTaskResponse(
            task_id="task-" + run_id,
            run_id=run_id,
            project_id=project_id,
            status="queued",
            created_at=now,
            updated_at=now,
            config=config,
        )

    def close(self):
        pass

    def stop_task(self, task_id):
        raise AssertionError("restart creation test never stops a task")


@unittest.skipUnless(
    SAFE_TEST_DATABASE_URL is not None,
    "PostgreSQL test configuration is unavailable",
)
class PostgreSQLRestartRecoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cleanup_engine = create_postgres_engine(SAFE_TEST_DATABASE_URL)
        cls.cleanup_factory = create_session_factory(cls.cleanup_engine)

    @classmethod
    def tearDownClass(cls):
        cls.cleanup_engine.dispose()

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.prefix = "p742-restart-%s-" % uuid4().hex
        self.user_id = self.prefix + "user"
        self.organization_id = self.prefix + "organization"
        self.course_id = self.prefix + "course"
        self.class_id = self.prefix + "class"
        self.project_id = self.prefix + "project"
        now = datetime.now(timezone.utc)
        with self.cleanup_factory() as session:
            session.add(UserRow(
                user_id=self.user_id, username=self.user_id,
                display_name="Restart Instructor", password_hash="test-only",
                enabled=True, created_at=now, updated_at=now,
            ))
            session.flush()
            session.add(OrganizationRow(
                organization_id=self.organization_id, slug=self.organization_id,
                name="Restart Test", enabled=True,
                created_at=now, updated_at=now,
            ))
            session.flush()
            session.add(OrganizationMembershipRow(
                membership_id=self.prefix + "membership",
                organization_id=self.organization_id, user_id=self.user_id,
                role="instructor", enabled=True,
                created_at=now, updated_at=now,
            ))
            session.add(CourseRow(
                course_id=self.course_id, organization_id=self.organization_id,
                code=self.prefix + "code", title="Restart Course",
                description="", owner_instructor_user_id=self.user_id,
                enabled=True, created_at=now, updated_at=now,
            ))
            session.flush()
            session.add(TeachingClassRow(
                class_id=self.class_id, course_id=self.course_id, code="A",
                name="A", term=self.prefix,
                instructor_user_id=self.user_id, enabled=True,
                created_at=now, updated_at=now,
            ))
            session.add(TeachingProjectRow(
                project_id=self.project_id, course_id=self.course_id,
                title="Restart Project", description="", instructions="",
                starter_code="", starter_code_sha256=starter_code_digest(""),
                training_profile="short", video_enabled=False,
                parameter_schema=parameter_schema(),
                evaluation_rubric=evaluation_rubric(TrainingProfile.SHORT),
                config_version=1, enabled=True,
                created_by_user_id=self.user_id,
                created_at=now, updated_at=now,
            ))
            session.commit()

    def tearDown(self):
        with self.cleanup_factory() as session:
            run_ids = select(RunRow.run_id).where(
                RunRow.user_id == self.user_id
            )
            session.execute(
                delete(RunIdempotencyKeyRow).where(
                    RunIdempotencyKeyRow.run_id.in_(run_ids)
                )
            )
            session.execute(
                delete(RunEventRow).where(RunEventRow.run_id.in_(run_ids))
            )
            session.execute(
                delete(RunRow).where(RunRow.user_id == self.user_id)
            )
            session.execute(delete(TeachingProjectRow).where(
                TeachingProjectRow.project_id == self.project_id
            ))
            session.execute(delete(TeachingClassRow).where(
                TeachingClassRow.class_id == self.class_id
            ))
            session.execute(delete(CourseRow).where(
                CourseRow.course_id == self.course_id
            ))
            session.execute(delete(OrganizationMembershipRow).where(
                OrganizationMembershipRow.organization_id == self.organization_id
            ))
            session.execute(delete(OrganizationRow).where(
                OrganizationRow.organization_id == self.organization_id
            ))
            session.execute(delete(UserRow).where(UserRow.user_id == self.user_id))
            session.commit()
        self.temp_dir.cleanup()

    def _app(self, name):
        root = Path(self.temp_dir.name)
        return create_app(
            root / (name + "-runs"),
            tick_seconds=0,
            project_root=root / (name + "-projects"),
            run_repository_mode="postgresql",
            database_url=SAFE_TEST_DATABASE_URL,
            current_user_provider=StaticCurrentUserProvider(self.user_id),
            simulation_client=RestartSimulationClient(),
        )

    def _payload(self, seed=44):
        return {
            "project_id": self.project_id,
            "class_id": self.class_id,
            "project_config_version": 1,
            "execution_mode": "simulation",
            "run_kind": "train",
            "parameters": {"seed": seed},
            "code": "print('persistence recovery')\n",
        }

    def _counts(self):
        with self.cleanup_factory() as session:
            run_ids = select(RunRow.run_id).where(
                RunRow.user_id == self.user_id
            )
            return (
                session.scalar(
                    select(func.count())
                    .select_from(RunRow)
                    .where(RunRow.user_id == self.user_id)
                ),
                session.scalar(
                    select(func.count())
                    .select_from(RunEventRow)
                    .where(RunEventRow.run_id.in_(run_ids))
                ),
                session.scalar(
                    select(func.count())
                    .select_from(RunIdempotencyKeyRow)
                    .where(RunIdempotencyKeyRow.user_id == self.user_id)
                ),
            )

    def test_run_events_and_idempotency_survive_independent_app_rebuild(self):
        first_app = self._app("app-a")
        with patch.object(
            first_app.state.engine,
            "dispose",
            wraps=first_app.state.engine.dispose,
        ) as dispose:
            with ASGITestClient(first_app) as first_client:
                created = first_client.post(
                    "/api/v1/runs",
                    json=self._payload(),
                    headers={"Idempotency-Key": "restart-key"},
                )
                self.assertEqual(created.status_code, 202)
                self.assertEqual(
                    created.headers["Idempotency-Replayed"], "false"
                )
                run_id = created.json()["run_id"]
                task_id = created.json()["simulation_task_id"]
                now = datetime.now(timezone.utc)
                first_app.state.run_service.apply_event(
                    RunEvent(
                        event_id=self.prefix + "state-1",
                        sequence=1,
                        type=RunEventType.STATE,
                        run_id=run_id,
                        task_id=task_id,
                        occurred_at=now,
                        payload={
                            "previous_status": "queued",
                            "status": "initializing",
                        },
                    )
                )
                first_app.state.run_service.apply_event(
                    RunEvent(
                        event_id=self.prefix + "metric-2",
                        sequence=2,
                        type=RunEventType.METRIC,
                        run_id=run_id,
                        task_id=task_id,
                        occurred_at=now,
                        payload={
                            "step": 30,
                            "metrics": {"average_reward": 0.25},
                            "progress": 0.25,
                        },
                    )
                )
                expected_run = first_client.get(
                    "/api/v1/runs/%s" % run_id
                ).json()
                expected_events = first_client.get(
                    "/api/v1/runs/%s/events" % run_id
                ).json()
            dispose.assert_called_once_with()

        second_app = self._app("app-b")
        self.assertIsNot(first_app, second_app)
        for name in (
            "engine",
            "session_factory",
            "repository",
            "run_service",
            "run_creation_service",
            "adapter",
        ):
            self.assertIsNot(
                getattr(first_app.state, name), getattr(second_app.state, name)
            )
        self.assertIsInstance(
            second_app.state.repository, PostgreSQLRunRepository
        )
        self.assertIsInstance(
            second_app.state.run_creation_service,
            PostgreSQLIdempotentRunCreationService,
        )
        self.assertIs(
            second_app.state.repository,
            second_app.state.run_service.repository,
        )
        self.assertIs(
            second_app.state.run_service,
            second_app.state.adapter.run_service,
        )

        with ASGITestClient(second_app) as second_client:
            detail = second_client.get("/api/v1/runs/%s" % run_id)
            events = second_client.get(
                "/api/v1/runs/%s/events" % run_id
            )
            catchup = second_client.get(
                "/api/v1/runs/%s/events?after_sequence=1" % run_id
            )
            listed = second_client.get("/api/v1/runs")
            replay = second_client.post(
                "/api/v1/runs",
                json=self._payload(),
                headers={"Idempotency-Key": "restart-key"},
            )
            conflict = second_client.post(
                "/api/v1/runs",
                json=self._payload(seed=45),
                headers={"Idempotency-Key": "restart-key"},
            )
            legacy_detail = second_client.get("/api/runs/%s" % run_id)
            legacy_list = second_client.get("/api/runs")

            self.assertEqual(detail.json(), expected_run)
            self.assertEqual(detail.json()["status"], "initializing")
            self.assertEqual(detail.json()["last_sequence"], 2)
            self.assertEqual(detail.json()["simulation_task_id"], task_id)
            self.assertEqual(events.json(), expected_events)
            self.assertEqual(catchup.json(), expected_events[1:])
            self.assertIn(run_id, [run["run_id"] for run in listed.json()])
            self.assertEqual(replay.status_code, 202)
            self.assertEqual(replay.json()["run_id"], run_id)
            self.assertEqual(replay.headers["Idempotency-Replayed"], "true")
            self.assertEqual(conflict.status_code, 409)
            self.assertEqual(legacy_detail.status_code, 200)
            self.assertEqual(legacy_detail.json()["run_id"], run_id)
            self.assertEqual(legacy_detail.json()["status"], "queued")
            self.assertIn(
                run_id, [run["run_id"] for run in legacy_list.json()]
            )
            self.assertEqual(self._counts(), (1, 2, 1))


if __name__ == "__main__":
    unittest.main()
