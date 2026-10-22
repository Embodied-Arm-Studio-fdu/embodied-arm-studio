"""Run deletion: authenticated ownership, terminal gating and atomic cleanup."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from apps.api.main import create_app
from robotics.runs import RunStatus
from robotics.runs.db_models import RunRow, RunEventRow, RunArtifactRow, EvaluationReportRow, RunIdempotencyKeyRow
from tests.auth import StaticCurrentUserProvider
from tests.http_client import ASGITestClient
from tests.test_run_repository import make_run, make_event
from tests import test_v1_run_comparison as comparison_fixtures
from sqlalchemy import select, func


class RunDeletionMemoryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.app = create_app(Path(self.temp.name)/"runs", project_root=Path(self.temp.name)/"projects", run_repository_mode="memory", current_user_provider=StaticCurrentUserProvider("user-1"))
        self.client = ASGITestClient(self.app)
        self.addCleanup(self.client.close)
        self.repository = self.app.state.run_service.repository
        self.repository.create_run(replace(make_run("done"), status=RunStatus.COMPLETED))
        self.repository.create_run(make_run("active"))
        self.repository.create_run(replace(make_run("other"), status=RunStatus.COMPLETED, user_id="user-2"))
        self.repository.append_event(make_event("done"))

    def delete(self, ids, **kwargs):
        return self.client.post("/api/v1/runs/delete", json={"run_ids": ids}, **kwargs)

    def test_delete_removes_record_and_events(self):
        self.assertEqual(self.delete(["done"]).json(), {"deleted_count": 1, "deleted_run_ids": ["done"]})
        self.assertEqual(self.client.get("/api/v1/runs/done").status_code, 404)
        self.assertNotIn("done", [r["run_id"] for r in self.client.get("/api/v1/runs").json()])
        self.assertEqual(self.delete(["done"]).status_code, 404)
        self.repository.create_run(replace(make_run("done"), status=RunStatus.COMPLETED))
        self.assertEqual(self.repository.list_events("done"), [])
        self.repository.append_event(make_event("done"))  # Event ID was removed too.

    def test_mixed_active_batch_is_atomic(self):
        self.assertEqual(self.delete(["done", "active"]).status_code, 409)
        self.assertEqual(self.repository.get_run("done").status, RunStatus.COMPLETED)

    def test_unknown_or_other_owner_batch_is_atomic(self):
        for invalid in ("missing", "other"):
            self.assertEqual(self.delete(["done", invalid]).status_code, 404)
            self.assertEqual(self.repository.get_run("done").status, RunStatus.COMPLETED)

    def test_auth_and_batch_validation(self):
        self.assertEqual(self.delete(["done"], headers={"Authorization": ""}).status_code, 401)
        for ids in ([], ["done", "done"], [" "], ["x%d" % i for i in range(11)]):
            self.assertEqual(self.delete(ids).status_code, 422)


@unittest.skipUnless(comparison_fixtures.SAFE_POSTGRES_URL is not None, "Isolated PostgreSQL test database is unavailable")
class RunDeletionPostgreSQLTest(unittest.TestCase):
    # Reuse the canonical comparison fixtures, including events/artifacts/evaluations.
    setUp = comparison_fixtures.V1RunComparisonExportTest.setUp
    tearDown = comparison_fixtures.V1RunComparisonExportTest.tearDown
    _app = comparison_fixtures.V1RunComparisonExportTest._app
    _run_row = comparison_fixtures.V1RunComparisonExportTest._run_row
    _evaluation = comparison_fixtures.V1RunComparisonExportTest._evaluation
    _insert_facts = comparison_fixtures.V1RunComparisonExportTest._insert_facts
    _cleanup = comparison_fixtures.V1RunComparisonExportTest._cleanup

    def delete(self, ids):
        return self.client.post("/api/v1/runs/delete", json={"run_ids": ids})

    def test_transaction_cleans_all_indexes_and_survives_restart(self):
        with self.session_factory() as session:
            session.add(RunIdempotencyKeyRow(user_id=self.user_id, idempotency_key=self.prefix, request_hash="f"*64, run_id=self.run_a, created_at=self.now))
            session.commit()
        self.assertEqual(self.delete([self.run_a, self.run_b]).json()["deleted_count"], 2)
        with self.session_factory() as session:
            for model in (RunRow, RunEventRow, RunArtifactRow, EvaluationReportRow, RunIdempotencyKeyRow):
                count = session.scalar(select(func.count()).select_from(model).where(model.run_id.in_([self.run_a, self.run_b])))
                self.assertEqual(count, 0, model.__tablename__)
            self.assertIsNotNone(session.get(RunRow, self.other_run))
        with ASGITestClient(self._app("restarted")) as client:
            self.assertEqual(client.get("/api/v1/runs/"+self.run_a).status_code, 404)

    def test_other_owner_and_active_batches_leave_all_records(self):
        self.assertEqual(self.delete([self.run_a, self.other_run]).status_code, 404)
        with self.session_factory() as session:
            session.get(RunRow, self.run_b).status = "running"
            session.commit()
        self.assertEqual(self.delete([self.run_a, self.run_b]).status_code, 409)
        with self.session_factory() as session:
            self.assertIsNotNone(session.get(RunRow, self.run_a))
            self.assertIsNotNone(session.scalar(select(RunArtifactRow).where(RunArtifactRow.run_id == self.run_a).limit(1)))


if __name__ == "__main__":
    unittest.main()
