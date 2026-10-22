import hashlib
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import delete

from apps.api.database import create_postgres_engine, create_session_factory
from apps.api.main import create_app
from tests.auth import StaticCurrentUserProvider
from robotics.runs.db_models import (
    EvaluationReportRow,
    RunArtifactRow,
    RunEventRow,
    RunIdempotencyKeyRow,
    RunRow,
)
from tests.http_client import ASGITestClient
from tests.postgres_users import delete_test_user, seed_test_user
from tests.test_terminal_callback_api import SAFE_POSTGRES_URL


@unittest.skipUnless(
    SAFE_POSTGRES_URL is not None,
    "PostgreSQL test configuration is unavailable",
)
class V1RunComparisonExportTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.prefix = "p66-%s" % uuid4().hex
        self.user_id = self.prefix + "-user"
        self.other_user_id = self.prefix + "-other-user"
        self.engine = create_postgres_engine(SAFE_POSTGRES_URL)
        self.session_factory = create_session_factory(self.engine)
        seed_test_user(self.session_factory, self.user_id)
        seed_test_user(self.session_factory, self.other_user_id)
        self.run_a = self.prefix + "-run-a"
        self.run_b = self.prefix + "-run-b"
        self.run_c = self.prefix + "-run-c"
        self.run_missing_evaluation = self.prefix + "-run-no-eval"
        self.other_run = self.prefix + "-run-other"
        self.now = datetime(2026, 8, 28, 12, tzinfo=timezone.utc)
        self._insert_facts()
        self.app = self._app("app-a")
        self.client = ASGITestClient(self.app)

    def tearDown(self):
        self.client.close()
        self._cleanup()
        delete_test_user(self.session_factory, self.user_id)
        delete_test_user(self.session_factory, self.other_user_id)
        self.engine.dispose()
        self.temp_dir.cleanup()

    def _app(self, name):
        root = Path(self.temp_dir.name) / name
        return create_app(
            root / "runs",
            tick_seconds=0,
            project_root=root / "projects",
            run_repository_mode="postgresql",
            database_url=SAFE_POSTGRES_URL,
            current_user_provider=StaticCurrentUserProvider(self.user_id),
            simulation_environment={},
        )

    def _run_row(
        self,
        run_id,
        *,
        user_id=None,
        project_id="project-1",
        status="completed",
        parameters=None,
        config=None,
        metrics=None,
        error=None,
        training_steps=100,
        update_steps=25,
        stop_reason=None,
        last_sequence=0,
    ):
        return RunRow(
            run_id=run_id,
            user_id=user_id or self.user_id,
            project_id=project_id,
            status=status,
            execution_mode="simulation",
            run_kind="train",
            parameters=parameters or {},
            project_config_snapshot=config or {},
            code_sha256=(run_id[-1] if run_id[-1] in "abcdef" else "a") * 64,
            simulation_task_id=self.prefix + "-task-" + run_id[-1],
            progress=1.0 if status == "completed" else 0.6,
            metrics_summary=metrics or {},
            error=error,
            last_sequence=last_sequence,
            training_steps=training_steps,
            update_steps=update_steps,
            stop_reason=stop_reason,
            created_at=self.now,
            started_at=self.now,
            ended_at=self.now,
            updated_at=self.now,
        )

    def _evaluation(
        self,
        evaluation_id,
        run_id,
        *,
        status="assessed",
        score=90.0,
        grade="pass",
        rubric_version="rubric-1",
    ):
        return EvaluationReportRow(
            evaluation_id=evaluation_id,
            run_id=run_id,
            requested_by_user_id=self.user_id,
            status=status,
            score=score,
            grade=grade,
            rubric_version=rubric_version,
            rubric_snapshot={
                "version": rubric_version,
                "criteria": [],
                "service_token": "do-not-export-token",
            },
            evidence={
                "run": {"run_id": run_id},
                "host_path": "/private/evidence.json",
            },
            evidence_sha256=("b" if run_id == self.run_b else "c") * 64,
            summary={"criterion_count": 1},
            limitations=(
                ["Frozen evidence was insufficient."]
                if status == "insufficient_evidence" else []
            ),
            evaluator_version="canonical-rubric-v1",
            created_at=self.now,
        )

    def _insert_facts(self):
        with self.session_factory() as session:
            session.add_all(
                [
                    self._run_row(
                        self.run_a,
                        parameters={"learning_rate": 0.001, "seed": 7},
                        config={"config_version": "config-1", "task": "push"},
                        metrics={
                            "reward": 0.8,
                            "success_rate": 0.9,
                            "update_index": 3,
                        },
                        last_sequence=3,
                    ),
                    self._run_row(
                        self.run_b,
                        status="failed",
                        parameters={"learning_rate": 0.002, "seed": 7},
                        config={
                            "config_version": "config-2",
                            "database_url": "postgresql://secret@db/prod",
                        },
                        metrics={"reward": 0.4},
                        error={
                            "code": "TRAINING_FAILED",
                            "message": "token=private /private/internal.log",
                            "retryable": False,
                        },
                        training_steps=80,
                        update_steps=20,
                    ),
                    self._run_row(
                        self.run_c,
                        project_id="project-2",
                        status="stopped",
                        config={"config_version": "config-1"},
                        training_steps=10,
                        update_steps=2,
                        stop_reason="requested_by_user",
                    ),
                    self._run_row(self.run_missing_evaluation),
                    self._run_row(
                        self.other_run,
                        user_id=self.other_user_id,
                    ),
                ]
            )
            session.flush()
            session.add_all(
                [
                    RunEventRow(
                        event_id=self.prefix + "-event-1",
                        run_id=self.run_a,
                        sequence=1,
                        type="state",
                        task_id=self.prefix + "-task-a",
                        occurred_at=self.now,
                        schema_version="1.0",
                        payload={"status": "running", "service_token": "secret"},
                    ),
                    RunEventRow(
                        event_id=self.prefix + "-event-3",
                        run_id=self.run_a,
                        sequence=3,
                        type="metric",
                        task_id=self.prefix + "-task-a",
                        occurred_at=self.now,
                        schema_version="1.0",
                        payload={
                            "step": 100,
                            "metrics": {"reward": 0.8, "update_index": 3},
                        },
                    ),
                    RunArtifactRow(
                        artifact_id=self.prefix + "-artifact-manifest",
                        run_id=self.run_a,
                        artifact_type="manifest",
                        path="manifest.json",
                        size_bytes=42,
                        sha256="d" * 64,
                        is_primary_model=False,
                        created_at=self.now,
                    ),
                    RunArtifactRow(
                        artifact_id=self.prefix + "-artifact-model",
                        run_id=self.run_a,
                        artifact_type="checkpoint_best",
                        path="checkpoints/model.pt",
                        size_bytes=None,
                        sha256=None,
                        is_primary_model=True,
                        created_at=self.now,
                    ),
                    RunArtifactRow(
                        artifact_id=self.prefix + "-artifact-log",
                        run_id=self.run_b,
                        artifact_type="log",
                        path="logs/task.log",
                        size_bytes=12,
                        sha256="e" * 64,
                        is_primary_model=False,
                        created_at=self.now,
                    ),
                    self._evaluation(
                        self.prefix + "-evaluation-a", self.run_a
                    ),
                    self._evaluation(
                        self.prefix + "-evaluation-b",
                        self.run_b,
                        status="insufficient_evidence",
                        score=None,
                        grade="insufficient_evidence",
                    ),
                    self._evaluation(
                        self.prefix + "-evaluation-c",
                        self.run_c,
                        rubric_version="rubric-2",
                    ),
                ]
            )
            session.commit()

    def _cleanup(self):
        run_ids = [
            self.run_a,
            self.run_b,
            self.run_c,
            self.run_missing_evaluation,
            self.other_run,
        ]
        with self.session_factory() as session:
            session.execute(
                delete(EvaluationReportRow).where(
                    EvaluationReportRow.run_id.in_(run_ids)
                )
            )
            session.execute(
                delete(RunArtifactRow).where(RunArtifactRow.run_id.in_(run_ids))
            )
            session.execute(
                delete(RunEventRow).where(RunEventRow.run_id.in_(run_ids))
            )
            session.execute(
                delete(RunIdempotencyKeyRow).where(
                    RunIdempotencyKeyRow.run_id.in_(run_ids)
                )
            )
            session.execute(delete(RunRow).where(RunRow.run_id.in_(run_ids)))
            session.commit()

    def compare(self, *run_ids):
        return self.client.post(
            "/api/v1/runs/compare", json={"run_ids": list(run_ids)}
        )

    def export(self, *run_ids, format="json"):
        return self.client.post(
            "/api/v1/runs/export",
            json={"run_ids": list(run_ids), "format": format},
        )

    def test_compare_preserves_order_and_reports_parameter_metric_step_differences(self):
        response = self.compare(self.run_b, self.run_a)
        self.assertEqual(200, response.status_code, response.text)
        body = response.json()
        self.assertEqual([self.run_b, self.run_a], body["run_ids"])
        self.assertEqual([self.run_b, self.run_a], [item["run_id"] for item in body["runs"]])
        self.assertEqual(
            [0.002, 0.001],
            body["differences"]["parameters"]["learning_rate"]["values"],
        )
        self.assertEqual(
            [None, 0.9],
            body["differences"]["metrics"]["success_rate"]["values"],
        )
        self.assertEqual(
            [None, 3],
            body["differences"]["metrics"]["update_index"]["values"],
        )
        self.assertTrue(body["differences"]["training_steps"]["different"])
        self.assertTrue(body["differences"]["config_versions"]["different"])
        self.assertNotIn("token=private", response.text)

    def test_request_limits_duplicates_unknown_and_cross_user_are_enforced(self):
        for run_ids in ([self.run_a], [self.run_a] * 11, [self.run_a, self.run_a]):
            with self.subTest(run_ids=len(run_ids)):
                response = self.client.post(
                    "/api/v1/runs/compare", json={"run_ids": run_ids}
                )
                self.assertEqual(422, response.status_code)
        accepted_maximum = self.client.post(
            "/api/v1/runs/compare",
            json={"run_ids": ["missing-%d" % index for index in range(10)]},
        )
        self.assertEqual(404, accepted_maximum.status_code)
        self.assertEqual(404, self.compare(self.run_a, "missing-run").status_code)
        self.assertEqual(404, self.compare(self.run_a, self.other_run).status_code)

    def test_same_and_cross_project_comparison_rules_are_explicit(self):
        same = self.compare(self.run_a, self.run_b).json()
        cross = self.compare(self.run_a, self.run_c).json()
        self.assertFalse(same["cross_project"])
        self.assertTrue(cross["cross_project"])
        self.assertIn("Cross-project comparison", " ".join(cross["limitations"]))

    def test_terminal_states_and_missing_metrics_are_neutral(self):
        body = self.compare(self.run_a, self.run_b, self.run_c).json()
        self.assertEqual(
            ["completed", "failed", "stopped"],
            body["differences"]["statuses"]["values"],
        )
        self.assertIsNone(
            body["differences"]["metrics"]["success_rate"]["values"][2]
        )
        limitations = " ".join(body["limitations"])
        self.assertIn("not quality rankings", limitations)
        self.assertIn("never replaced with zero", limitations)

    def test_evaluation_rubric_compatibility_missing_evidence_and_artifacts(self):
        same = self.compare(self.run_a, self.run_b).json()
        different = self.compare(self.run_a, self.run_c).json()
        missing = self.compare(self.run_a, self.run_missing_evaluation).json()
        self.assertTrue(
            same["differences"]["evaluations"]["rubric_versions_comparable"]
        )
        self.assertFalse(
            different["differences"]["evaluations"]["rubric_versions_comparable"]
        )
        self.assertIsNone(missing["runs"][1]["evaluation"])
        self.assertEqual(
            "insufficient_evidence", same["runs"][1]["evaluation"]["status"]
        )
        artifacts = same["runs"][0]["artifacts"]
        self.assertEqual(2, artifacts["total_count"])
        self.assertEqual(1, artifacts["missing_sha256_count"])

    def test_json_export_contains_canonical_facts_and_preserves_event_gap(self):
        response = self.export(self.run_a)
        self.assertEqual(200, response.status_code, response.text)
        body = response.json()
        self.assertEqual("1.0", body["schema_version"])
        item = body["runs"][0]
        self.assertEqual(self.run_a, item["run"]["run_id"])
        self.assertEqual([1, 3], [event["sequence"] for event in item["events"]])
        self.assertEqual(3, item["run"]["metrics_summary"]["update_index"])
        self.assertEqual(
            3, item["events"][1]["payload"]["metrics"]["update_index"]
        )
        self.assertEqual(2, len(item["artifacts"]))
        self.assertEqual(1, len(item["evaluations"]))
        self.assertIn("download_path", item["artifacts"][0])
        self.assertIn("logical_path", item["artifacts"][0])
        self.assertFalse(item["artifacts"][0]["logical_path"].startswith("/"))

    def test_export_redacts_secrets_database_urls_absolute_paths_and_error_message(self):
        response = self.export(self.run_a, self.run_b)
        text = response.text
        self.assertNotIn("do-not-export-token", text)
        self.assertNotIn("postgresql://", text)
        self.assertNotIn("/home/user/", text)
        self.assertNotIn("token=private", text)
        failed = response.json()["runs"][1]["run"]
        self.assertEqual("Run reported an error", failed["error"]["message"])
        self.assertEqual(
            "[redacted]", failed["project_config_snapshot"]["database_url"]
        )

    def test_export_headers_etag_schema_and_format_are_stable(self):
        first = self.export(self.run_a, self.run_b)
        second = self.export(self.run_a, self.run_b)
        self.assertEqual(first.content, second.content)
        self.assertEqual(first.headers["etag"], second.headers["etag"])
        self.assertEqual(
            '"%s"' % hashlib.sha256(first.content).hexdigest(),
            first.headers["etag"],
        )
        self.assertEqual(
            'attachment; filename="runs-export.json"',
            first.headers["content-disposition"],
        )
        self.assertEqual("nosniff", first.headers["x-content-type-options"])
        self.assertEqual("application/json", first.headers["content-type"])
        self.assertEqual(422, self.export(self.run_a, format="pdf").status_code)
        self.assertEqual(
            422,
            self.client.post(
                "/api/v1/runs/export",
                json={"run_ids": ["missing-%d" % index for index in range(11)]},
            ).status_code,
        )
        self.assertEqual(
            422,
            self.client.post(
                "/api/v1/runs/export", json={"run_ids": [self.run_a] * 2}
            ).status_code,
        )

    def test_app_rebuild_returns_identical_comparison_and_export(self):
        comparison = self.compare(self.run_a, self.run_b).content
        exported = self.export(self.run_a, self.run_b).content
        rebuilt = self._app("app-b")
        with ASGITestClient(rebuilt) as client:
            comparison_after = client.post(
                "/api/v1/runs/compare",
                json={"run_ids": [self.run_a, self.run_b]},
            ).content
            exported_after = client.post(
                "/api/v1/runs/export",
                json={"run_ids": [self.run_a, self.run_b], "format": "json"},
            ).content
        self.assertEqual(comparison, comparison_after)
        self.assertEqual(exported, exported_after)

    def test_memory_app_does_not_publish_memory_as_canonical_source(self):
        root = Path(self.temp_dir.name) / "memory"
        app = create_app(
            root / "runs",
            tick_seconds=0,
            project_root=root / "projects",
            run_repository_mode="memory",
            current_user_provider=StaticCurrentUserProvider(self.user_id),
            simulation_environment={},
        )
        with ASGITestClient(app) as client:
            self.assertEqual(
                503,
                client.post(
                    "/api/v1/runs/compare",
                    json={"run_ids": [self.run_a, self.run_b]},
                ).status_code,
            )
            self.assertEqual(
                503,
                client.post(
                    "/api/v1/runs/export",
                    json={"run_ids": [self.run_a]},
                ).status_code,
            )


if __name__ == "__main__":
    unittest.main()
