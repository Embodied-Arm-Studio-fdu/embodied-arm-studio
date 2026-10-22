import asyncio
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import delete

from apps.api.database import (
    DatabaseConfigurationError,
    DatabaseReadinessError,
    create_postgres_engine,
    create_session_factory,
)
from apps.api.dependencies.runs import RunRepositoryConfigurationError
from apps.api.main import create_app
from tests.auth import StaticCurrentUserProvider
from robotics.runs import (
    InMemoryRunRepository,
    PostgreSQLIdempotentRunCreationService,
    PostgreSQLRunRepository,
)
from robotics.runs.db_models import RunEventRow, RunIdempotencyKeyRow, RunRow
from tests.http_client import ASGITestClient
from tests.postgres_users import delete_test_user, seed_test_user
from tests.postgres_safety import safe_test_database_url


SAFE_TEST_DATABASE_URL = safe_test_database_url()


@unittest.skipUnless(
    SAFE_TEST_DATABASE_URL is not None,
    "TEST_DATABASE_URL is unset; PostgreSQL app wiring tests were not run",
)
class PostgreSQLAppWiringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cleanup_engine = create_postgres_engine(SAFE_TEST_DATABASE_URL)
        cls.cleanup_factory = create_session_factory(cls.cleanup_engine)
        cls.created_mock_user = seed_test_user(
            cls.cleanup_factory, "mock-adapter"
        )

    @classmethod
    def tearDownClass(cls):
        if cls.created_mock_user:
            delete_test_user(cls.cleanup_factory, "mock-adapter")
        cls.cleanup_engine.dispose()

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.created_run_ids = []

    def tearDown(self):
        if self.created_run_ids:
            with self.cleanup_factory() as session:
                session.execute(
                    delete(RunIdempotencyKeyRow).where(
                        RunIdempotencyKeyRow.run_id.in_(self.created_run_ids)
                    )
                )
                session.execute(
                    delete(RunEventRow).where(
                        RunEventRow.run_id.in_(self.created_run_ids)
                    )
                )
                session.execute(
                    delete(RunRow).where(
                        RunRow.run_id.in_(self.created_run_ids)
                    )
                )
                session.commit()
        self.temp_dir.cleanup()

    def _postgres_app(self):
        return create_app(
            self.root / "runs",
            tick_seconds=0,
            project_root=self.root / "projects",
            run_repository_mode="postgresql",
            database_url=SAFE_TEST_DATABASE_URL,
            current_user_provider=StaticCurrentUserProvider("mock-adapter"),
        )

    def _launch(self, client):
        response = client.post(
            "/api/runs/train",
            json={
                "environment_id": "mock-reach",
                "strategy_id": "joint-p-control",
                "seed": 41,
                "max_steps": 6,
                "config": {"action_scale": 0.4},
            },
        )
        self.assertEqual(response.status_code, 202)
        run_id = response.json()["run_id"]
        self.created_run_ids.append(run_id)
        return run_id

    def _wait_terminal(self, client, run_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            response = client.get("/api/v1/runs/%s" % run_id)
            self.assertEqual(response.status_code, 200)
            if response.json()["status"] in {"completed", "failed", "stopped"}:
                return response.json()
            time.sleep(0.005)
        self.fail("PostgreSQL-backed Mock Run did not finish")

    def test_mock_run_and_events_survive_app_recreation_and_engine_disposes(self):
        first_app = self._postgres_app()
        self.assertIsInstance(
            first_app.state.repository, PostgreSQLRunRepository
        )
        self.assertIs(
            first_app.state.repository,
            first_app.state.run_service.repository,
        )
        self.assertIs(
            first_app.state.run_service,
            first_app.state.adapter.run_service,
        )
        self.assertIsInstance(
            first_app.state.run_creation_service,
            PostgreSQLIdempotentRunCreationService,
        )
        train_route = next(
            route
            for route in first_app.routes
            if getattr(route, "path", None) == "/api/runs/train"
        )
        closure_values = {
            cell.cell_contents for cell in train_route.endpoint.__closure__ or ()
        }
        self.assertIn(first_app.state.adapter, closure_values)

        with patch.object(
            first_app.state.engine,
            "dispose",
            wraps=first_app.state.engine.dispose,
        ) as dispose:
            with ASGITestClient(first_app) as first_client:
                run_id = self._launch(first_client)
                terminal = self._wait_terminal(first_client, run_id)
                self.assertEqual(terminal["status"], "completed")
                self.assertEqual(terminal["execution_mode"], "mock")
                events = first_client.get(
                    "/api/v1/runs/%s/events" % run_id
                ).json()
                self.assertGreater(len(events), 6)
                self.assertTrue(any(event["type"] == "metric" for event in events))
                self.assertTrue((self.root / "runs" / run_id / "run.json").is_file())
                self.assertTrue(
                    (self.root / "runs" / run_id / "metrics.jsonl").is_file()
                )
            dispose.assert_called_once_with()

        second_app = self._postgres_app()
        self.assertIsNot(second_app.state.engine, first_app.state.engine)
        self.assertIsNot(second_app.state.repository, first_app.state.repository)
        self.assertIs(
            second_app.state.repository,
            second_app.state.run_service.repository,
        )
        self.assertIs(
            second_app.state.run_service,
            second_app.state.adapter.run_service,
        )
        self.assertIsInstance(
            second_app.state.run_creation_service,
            PostgreSQLIdempotentRunCreationService,
        )
        self.assertIsNot(
            second_app.state.run_creation_service,
            first_app.state.run_creation_service,
        )
        with ASGITestClient(second_app) as second_client:
            listed = second_client.get("/api/v1/runs").json()
            self.assertIn(run_id, [run["run_id"] for run in listed])
            detail = second_client.get("/api/v1/runs/%s" % run_id)
            self.assertEqual(detail.status_code, 200)
            self.assertEqual(detail.json(), terminal)
            recovered_events = second_client.get(
                "/api/v1/runs/%s/events?after_sequence=2" % run_id
            ).json()
            self.assertEqual(recovered_events, events[2:])
            legacy = second_client.get("/api/runs/%s" % run_id)
            self.assertEqual(legacy.status_code, 200)
            self.assertEqual(legacy.json()["status"], "completed")

    def test_postgresql_mode_requires_database_url(self):
        with self.assertRaisesRegex(
            DatabaseConfigurationError, "DATABASE_URL is required"
        ):
            create_app(
                self.root / "missing-url-runs",
                project_root=self.root / "missing-url-projects",
                run_repository_mode="postgresql",
                database_environment={},
            )

    def test_default_mode_is_postgresql(self):
        app = create_app(
            self.root / "default-runs",
            project_root=self.root / "default-projects",
            database_environment={
                "DATABASE_URL": SAFE_TEST_DATABASE_URL.render_as_string(False)
            },
            current_user_provider=StaticCurrentUserProvider("mock-adapter"),
        )
        try:
            self.assertEqual(app.state.run_repository_mode, "postgresql")
            self.assertIsInstance(
                app.state.repository, PostgreSQLRunRepository
            )
        finally:
            app.state.engine.dispose()

    def test_readiness_failure_is_clear_and_disposes_engine(self):
        app = self._postgres_app()
        with patch(
            "apps.api.main.check_postgres_readiness",
            side_effect=DatabaseReadinessError("readiness failed"),
        ), patch.object(
            app.state.engine,
            "dispose",
            wraps=app.state.engine.dispose,
        ) as dispose:
            with self.assertRaisesRegex(
                DatabaseReadinessError, "readiness failed"
            ):
                asyncio.run(app.router.startup())
            dispose.assert_called_once_with()

    def test_explicit_memory_modes_never_build_a_postgres_engine(self):
        injected = InMemoryRunRepository()
        with patch(
            "apps.api.dependencies.runs.create_postgres_engine"
        ) as create_engine:
            injected_app = create_app(
                self.root / "injected-runs",
                tick_seconds=0,
                project_root=self.root / "injected-projects",
                run_repository=injected,
                current_user_provider=StaticCurrentUserProvider("mock-adapter"),
            )
            configured_app = create_app(
                self.root / "configured-runs",
                tick_seconds=0,
                project_root=self.root / "configured-projects",
                run_repository_mode="memory",
                current_user_provider=StaticCurrentUserProvider("mock-adapter"),
            )
            create_engine.assert_not_called()

        self.assertIs(injected_app.state.repository, injected)
        self.assertIsNone(injected_app.state.engine)
        self.assertIsInstance(
            configured_app.state.repository, InMemoryRunRepository
        )
        self.assertIsNot(
            configured_app.state.repository, injected_app.state.repository
        )
        self.assertIsNone(injected_app.state.run_creation_service)
        self.assertIsNone(configured_app.state.run_creation_service)
        with ASGITestClient(configured_app) as client:
            run_id = self._launch(client)
            self._wait_terminal(client, run_id)
        self.created_run_ids.remove(run_id)

    def test_unknown_mode_and_sqlite_are_rejected(self):
        for mode in ("sqlite", "automatic", ""):
            with self.subTest(mode=mode):
                with self.assertRaises(RunRepositoryConfigurationError):
                    create_app(
                        self.root / ("mode-" + (mode or "empty")),
                        project_root=self.root / "mode-projects",
                        run_repository_mode=mode,
                    )
        with self.assertRaisesRegex(
            DatabaseConfigurationError, "must use PostgreSQL"
        ):
            create_app(
                self.root / "sqlite-url-runs",
                project_root=self.root / "sqlite-url-projects",
                run_repository_mode="postgresql",
                database_url="sqlite:///forbidden.db",
            )


if __name__ == "__main__":
    unittest.main()
