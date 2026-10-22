import os
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from alembic import command as alembic_command
from alembic.config import Config
from sqlalchemy import delete, func, inspect, select

from apps.api.database import create_postgres_engine, create_session_factory
from apps.api.main import create_app
from tests.auth import StaticCurrentUserProvider
from tests.postgres_users import delete_test_user, seed_test_user
from robotics.evaluations import (
    EVALUATOR_VERSION,
    EvaluationNotAllowed,
    EvaluationReportNotFound,
    EvaluationService,
    EvaluationStatus,
    canonical_evidence_json,
    evidence_sha256,
)
from robotics.runs import (
    ExecutionMode,
    InMemoryRunRepository,
    RunArtifact,
    RunError,
    RunKind,
    RunService,
    RunStatus,
)
from robotics.runs.db_models import (
    EvaluationReportRow,
    RunArtifactRow,
    RunEventRow,
    RunIdempotencyKeyRow,
    RunRow,
)
from tests.http_client import ASGITestClient
from tests.test_terminal_callback_api import SAFE_POSTGRES_URL


ROOT = Path(__file__).resolve().parents[1]


def rubric(version="rubric-1", operator="gte"):
    return {
        "version": version,
        "criteria": [
            {
                "criterion_id": "success-rate",
                "evidence_path": "run.metrics_summary.success_rate",
                "operator": operator,
                "target": 0.8,
                "weight": 70,
            },
            {
                "criterion_id": "training-budget",
                "evidence_path": "run.training_steps",
                "operator": "lte",
                "target": 100,
                "weight": 30,
            },
        ],
        "grade_bands": [
            {"minimum_score": 90, "grade": "excellent"},
            {"minimum_score": 60, "grade": "pass"},
            {"minimum_score": 0, "grade": "needs_review"},
        ],
    }


class MemoryEvaluationRepository:
    def __init__(self):
        self.reports = {}
        self.by_fingerprint = {}

    def create_or_get(self, report):
        key = (
            report.run_id,
            report.rubric_version,
            report.evaluator_version,
            report.evidence_sha256,
        )
        existing_id = self.by_fingerprint.get(key)
        if existing_id is not None:
            return self.reports[existing_id], True
        self.reports[report.evaluation_id] = report
        self.by_fingerprint[key] = report.evaluation_id
        return report, False

    def get(self, evaluation_id):
        try:
            return self.reports[evaluation_id]
        except KeyError as exc:
            raise EvaluationReportNotFound() from exc

    def list_for_run(self, run_id):
        return sorted(
            [report for report in self.reports.values() if report.run_id == run_id],
            key=lambda report: (report.created_at, report.evaluation_id),
            reverse=True,
        )


class MemoryArtifactMetadataStore:
    def __init__(self):
        self.items = []

    def list_artifacts(self, run_id):
        return [item for item in self.items if item.run_id == run_id]


class PersistentEvaluationServiceAndApiTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.run_repository = InMemoryRunRepository()
        self.run_service = RunService(self.run_repository)
        self.artifacts = MemoryArtifactMetadataStore()
        self.evaluations = MemoryEvaluationRepository()
        self.next_id = 0

        def id_factory():
            self.next_id += 1
            return "evaluation-%d" % self.next_id

        self.service = EvaluationService(
            self.run_service,
            self.artifacts,
            self.evaluations,
            clock=lambda: datetime(2026, 8, 27, 12, tzinfo=timezone.utc),
            evaluation_id_factory=id_factory,
        )
        root = Path(self.temp_dir.name)
        self.app = create_app(
            root / "runs",
            tick_seconds=0,
            project_root=root / "projects",
            run_repository=self.run_repository,
            current_user_provider=StaticCurrentUserProvider("trusted-user"),
            canonical_evaluation_service=self.service,
        )
        self.client = ASGITestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.temp_dir.cleanup()

    def make_run(
        self,
        *,
        status=RunStatus.COMPLETED,
        user_id="trusted-user",
        frozen_rubric=None,
        metrics=None,
    ):
        snapshot = {}
        if frozen_rubric is not None:
            snapshot["evaluation_rubric"] = deepcopy(frozen_rubric)
        run = self.run_service.create_run(
            user_id=user_id,
            project_id="project-1",
            execution_mode=ExecutionMode.SIMULATION,
            run_kind=RunKind.TRAIN,
            parameters={"seed": 7, "max_steps": 100},
            project_config_snapshot=snapshot,
            code="policy-code",
        )
        if status != RunStatus.QUEUED:
            run = self.run_service.transition_status(
                run.run_id, RunStatus.INITIALIZING
            )
        if status not in {RunStatus.QUEUED, RunStatus.INITIALIZING}:
            run = self.run_service.transition_status(run.run_id, RunStatus.RUNNING)
        if status == RunStatus.COMPLETED:
            run = self.run_service.transition_status(
                run.run_id, RunStatus.FINALIZING
            )
            run = self.run_service.transition_status(
                run.run_id, RunStatus.COMPLETED
            )
        elif status == RunStatus.FAILED:
            run = self.run_service.transition_status(run.run_id, RunStatus.FAILED)
        elif status == RunStatus.STOPPED:
            run = self.run_service.transition_status(
                run.run_id, RunStatus.STOPPING
            )
            run = self.run_service.transition_status(run.run_id, RunStatus.STOPPED)
        elif status == RunStatus.FINALIZING:
            run = self.run_service.transition_status(
                run.run_id, RunStatus.FINALIZING
            )
        elif status == RunStatus.STOPPING:
            run = self.run_service.transition_status(
                run.run_id, RunStatus.STOPPING
            )
        updated = replace(
            run,
            metrics_summary=deepcopy(metrics or {}),
            training_steps=80,
            update_steps=20,
            error=(
                RunError("TRAINING_FAILED", "training failed", False)
                if status == RunStatus.FAILED
                else None
            ),
            stop_reason=(
                "requested_by_user" if status == RunStatus.STOPPED else None
            ),
        )
        return self.run_repository.save_run(updated)

    def test_commentary_uses_authorized_report_without_workspace(self):
        run = self.make_run(frozen_rubric=rubric(), metrics={"success_rate": 0.5})
        report = self.service.create(run.run_id, "trusted-user").report
        response = self.client.post(f"/api/v1/evaluations/{report.evaluation_id}/commentary", json={"provider": "mock", "focus": "failure_analysis", "assistance_level": 5})
        self.assertEqual(response.status_code, 200)
        self.assertIn(run.run_id, response.json()["content"])
        self.assertIn("canonical_evaluation_read", response.json()["tools_used"])
        self.assertEqual(self.service.get(report.evaluation_id, "trusted-user"), report)
        denied = self.client.post(f"/api/v1/evaluations/{report.evaluation_id}/commentary", json={"provider": "mock", "assistance_level": 4})
        self.assertEqual(denied.status_code, 422)

    def test_commentary_rejects_other_users_report(self):
        run = self.make_run(user_id="other-user", frozen_rubric=rubric())
        report = self.service.create(run.run_id, "other-user").report
        response = self.client.post(f"/api/v1/evaluations/{report.evaluation_id}/commentary", json={"provider": "mock", "assistance_level": 5})
        self.assertEqual(response.status_code, 404)

    def test_deepseek_commentary_receives_server_evidence(self):
        run = self.make_run(frozen_rubric=rubric())
        report = self.service.create(run.run_id, "trusted-user").report
        self.app.state.agent_provider_configs["deepseek"] = {"api_key": "test-only-not-a-real-key", "model": "deepseek-chat", "base_url": "https://api.deepseek.com"}
        with patch("apps.api.routers.evaluations.PROVIDERS") as providers:
            providers["deepseek"].complete.return_value = "基于真实证据的点评"
            response = self.client.post(f"/api/v1/evaluations/{report.evaluation_id}/commentary", json={"provider": "deepseek", "assistance_level": 5, "focus": "failure_analysis"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["content"], "基于真实证据的点评")
            messages = providers["deepseek"].complete.call_args[0][0]
            self.assertIn(report.evaluation_id, messages[1]["content"])
            self.assertNotIn("test-only-not-a-real-key", messages[1]["content"])

    def add_artifact(self, run, artifact_id="artifact-1"):
        self.artifacts.items.append(
            RunArtifact(
                artifact_id=artifact_id,
                run_id=run.run_id,
                artifact_type="manifest",
                path="manifest.json",
                size_bytes=42,
                sha256="a" * 64,
                is_primary_model=False,
                created_at=datetime(2026, 8, 27, 11, tzinfo=timezone.utc),
            )
        )

    def post(self, run_id, **body):
        payload = {"run_id": run_id}
        payload.update(body)
        return self.client.post("/api/v1/evaluations", json=payload)

    def test_completed_run_is_assessed_from_frozen_rubric(self):
        run = self.make_run(
            frozen_rubric=rubric(), metrics={"success_rate": 0.9}
        )
        self.add_artifact(run)
        response = self.post(run.run_id)
        self.assertEqual(response.status_code, 201, response.text)
        report = response.json()
        self.assertEqual(report["status"], "assessed")
        self.assertEqual(report["score"], 100)
        self.assertEqual(report["grade"], "excellent")
        self.assertEqual(report["rubric_version"], "rubric-1")
        self.assertEqual(report["evaluator_version"], EVALUATOR_VERSION)
        self.assertEqual(report["created_at"], "2026-08-27T12:00:00Z")

    def test_evidence_snapshots_metrics_run_facts_and_artifact_metadata(self):
        run = self.make_run(
            frozen_rubric=rubric(),
            metrics={"success_rate": 0.9, "update_index": 5},
        )
        self.add_artifact(run)
        report = self.service.create(run.run_id, "trusted-user").report
        evidence = report.evidence
        self.assertEqual(evidence["run"]["parameters"]["seed"], 7)
        self.assertEqual(evidence["run"]["seed"], 7)
        self.assertEqual(evidence["run"]["training_steps"], 80)
        self.assertEqual(evidence["run"]["update_steps"], 20)
        self.assertEqual(evidence["run"]["metrics_summary"]["success_rate"], 0.9)
        self.assertEqual(evidence["run"]["metrics_summary"]["update_index"], 5)
        self.assertEqual(evidence["artifacts"][0]["path"], "manifest.json")
        self.assertNotIn("created_at", evidence["artifacts"][0])
        self.assertNotIn(str(Path(self.temp_dir.name)), canonical_evidence_json(evidence))

    def test_missing_rubric_and_missing_artifact_are_explicitly_insufficient(self):
        run = self.make_run(metrics={"success_rate": 0.9})
        report = self.service.create(run.run_id, "trusted-user").report
        self.assertEqual(report.status, EvaluationStatus.INSUFFICIENT_EVIDENCE)
        self.assertIsNone(report.score)
        self.assertEqual(report.grade, "insufficient_evidence")
        self.assertIn("Frozen evaluation_rubric is missing.", report.limitations)
        self.assertIn(
            "No registered Artifact metadata is available.", report.limitations
        )

    def test_failed_and_stopped_are_not_automatically_scored_zero(self):
        for status in (RunStatus.FAILED, RunStatus.STOPPED):
            with self.subTest(status=status):
                run = self.make_run(status=status)
                report = self.service.create(run.run_id, "trusted-user").report
                self.assertEqual(
                    report.status, EvaluationStatus.INSUFFICIENT_EVIDENCE
                )
                self.assertIsNone(report.score)
                if status == RunStatus.FAILED:
                    self.assertEqual(
                        report.evidence["run"]["error"]["code"],
                        "TRAINING_FAILED",
                    )
                else:
                    self.assertEqual(
                        report.evidence["run"]["stop_reason"],
                        "requested_by_user",
                    )

    def test_all_nonterminal_states_are_rejected(self):
        for status in (
            RunStatus.QUEUED,
            RunStatus.INITIALIZING,
            RunStatus.RUNNING,
            RunStatus.FINALIZING,
            RunStatus.STOPPING,
        ):
            with self.subTest(status=status):
                run = self.make_run(status=status, frozen_rubric=rubric())
                with self.assertRaises(EvaluationNotAllowed):
                    self.service.create(run.run_id, "trusted-user")
                self.assertEqual(self.post(run.run_id).status_code, 409)

    def test_unknown_operator_is_not_guessed(self):
        run = self.make_run(
            frozen_rubric=rubric(operator="approximately"),
            metrics={"success_rate": 0.9},
        )
        report = self.service.create(run.run_id, "trusted-user").report
        self.assertEqual(report.status, EvaluationStatus.INSUFFICIENT_EVIDENCE)
        self.assertIsNone(report.score)
        self.assertIn(
            "Unknown rubric operator: approximately", report.limitations
        )

    def test_missing_or_incompatible_evidence_is_not_scored(self):
        for metrics in ({}, {"success_rate": "high"}):
            with self.subTest(metrics=metrics):
                run = self.make_run(frozen_rubric=rubric(), metrics=metrics)
                report = self.service.create(run.run_id, "trusted-user").report
                self.assertEqual(
                    report.status, EvaluationStatus.INSUFFICIENT_EVIDENCE
                )
                self.assertIsNone(report.score)

    def test_identical_evidence_replays_same_immutable_report(self):
        run = self.make_run(
            frozen_rubric=rubric(), metrics={"success_rate": 0.9}
        )
        first = self.post(run.run_id)
        second = self.post(run.run_id)
        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json(), second.json())
        self.assertEqual(second.headers["idempotency-replayed"], "true")
        self.assertEqual(len(self.evaluations.reports), 1)

    def test_list_requires_run_id_and_empty_history_does_not_evaluate(self):
        run = self.make_run(frozen_rubric=rubric())
        self.assertEqual(self.client.get("/api/v1/evaluations").status_code, 422)
        response = self.client.get(
            "/api/v1/evaluations", params={"run_id": run.run_id}
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), [])
        self.assertEqual(self.evaluations.reports, {})

    def test_list_returns_immutable_history_newest_first(self):
        run = self.make_run(
            frozen_rubric=rubric("rubric-1"), metrics={"success_rate": 0.9}
        )
        first = self.service.create(run.run_id, "trusted-user").report
        changed_snapshot = deepcopy(run.project_config_snapshot)
        changed_snapshot["evaluation_rubric"] = rubric("rubric-2")
        self.run_repository.save_run(
            replace(run, project_config_snapshot=changed_snapshot)
        )
        second = self.service.create(run.run_id, "trusted-user").report
        before = deepcopy(first.evidence)
        response = self.client.get(
            "/api/v1/evaluations", params={"run_id": run.run_id}
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            [item["evaluation_id"] for item in response.json()],
            sorted([first.evaluation_id, second.evaluation_id], reverse=True),
        )
        self.assertEqual(self.evaluations.get(first.evaluation_id).evidence, before)
        self.assertEqual(len(self.evaluations.reports), 2)
        self.assertNotIn("success_rate", response.json()[0]["summary"])

    def test_list_unknown_and_cross_user_runs_are_hidden(self):
        self.assertEqual(
            self.client.get(
                "/api/v1/evaluations", params={"run_id": "missing-run"}
            ).status_code,
            404,
        )
        other = self.make_run(user_id="other-user", frozen_rubric=rubric())
        self.assertEqual(
            self.client.get(
                "/api/v1/evaluations", params={"run_id": other.run_id}
            ).status_code,
            404,
        )

    def test_rubric_or_evidence_change_creates_new_immutable_report(self):
        run = self.make_run(
            frozen_rubric=rubric("rubric-1"),
            metrics={"success_rate": 0.9},
        )
        first = self.service.create(run.run_id, "trusted-user").report
        changed_snapshot = deepcopy(run.project_config_snapshot)
        changed_snapshot["evaluation_rubric"] = rubric("rubric-2")
        self.run_repository.save_run(
            replace(run, project_config_snapshot=changed_snapshot)
        )
        second = self.service.create(run.run_id, "trusted-user").report
        self.assertNotEqual(first.evaluation_id, second.evaluation_id)
        self.assertNotEqual(first.evidence_sha256, second.evidence_sha256)
        self.assertEqual(
            self.evaluations.get(first.evaluation_id).rubric_version,
            "rubric-1",
        )
        current = self.run_repository.get_run(run.run_id)
        self.run_repository.save_run(
            replace(current, metrics_summary={"success_rate": 0.1})
        )
        third = self.service.create(run.run_id, "trusted-user").report
        self.assertNotEqual(second.evaluation_id, third.evaluation_id)
        self.assertNotEqual(second.evidence_sha256, third.evidence_sha256)
        self.assertEqual(
            self.evaluations.get(second.evaluation_id).score, second.score
        )

    def test_evidence_sha_is_stable_canonical_json(self):
        left = {"b": [2, 1], "a": {"x": 1}}
        right = {"a": {"x": 1}, "b": [2, 1]}
        self.assertEqual(canonical_evidence_json(left), canonical_evidence_json(right))
        self.assertEqual(evidence_sha256(left), evidence_sha256(right))
        self.assertEqual(len(evidence_sha256(left)), 64)

    def test_report_rejects_out_of_range_score_and_naive_time(self):
        run = self.make_run(
            frozen_rubric=rubric(), metrics={"success_rate": 0.9}
        )
        report = self.service.create(run.run_id, "trusted-user").report
        with self.assertRaisesRegex(ValueError, "between 0 and 100"):
            replace(report, score=101)
        with self.assertRaisesRegex(ValueError, "timezone-aware UTC"):
            replace(report, created_at=datetime(2026, 8, 27, 12))

    def test_unknown_run_unauthorized_and_cross_user_get_are_hidden(self):
        self.assertEqual(self.post("missing-run").status_code, 404)
        other = self.make_run(user_id="other-user", frozen_rubric=rubric())
        self.assertEqual(self.post(other.run_id).status_code, 404)
        owned = self.make_run(
            frozen_rubric=rubric(), metrics={"success_rate": 0.9}
        )
        report = self.service.create(owned.run_id, "trusted-user").report
        root = Path(self.temp_dir.name)
        other_app = create_app(
            root / "other-runs",
            tick_seconds=0,
            project_root=root / "other-projects",
            run_repository=self.run_repository,
            current_user_provider=StaticCurrentUserProvider("other-user"),
            canonical_evaluation_service=self.service,
        )
        with ASGITestClient(other_app) as other_client:
            response = other_client.get(
                "/api/v1/evaluations/%s" % report.evaluation_id
            )
            self.assertEqual(response.status_code, 404)
        self.assertEqual(
            self.client.get("/api/v1/evaluations/missing").status_code, 404
        )

    def test_client_cannot_submit_score_or_overwrite_report(self):
        run = self.make_run(
            frozen_rubric=rubric(), metrics={"success_rate": 0.9}
        )
        response = self.post(run.run_id, score=0, grade="failed")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.evaluations.reports, {})


@unittest.skipUnless(
    SAFE_POSTGRES_URL is not None,
    "PostgreSQL test configuration is unavailable",
)
class PersistentEvaluationMigrationTest(unittest.TestCase):
    def test_upgrade_downgrade_upgrade_only_owns_evaluation_table(self):
        engine = create_postgres_engine(SAFE_POSTGRES_URL)
        config = Config(str(ROOT / "alembic.ini"))
        environment = {
            "DATABASE_URL": SAFE_POSTGRES_URL.render_as_string(False)
        }
        try:
            with patch.dict(os.environ, environment, clear=False):
                alembic_command.upgrade(config, "head")
            self.assertIn("evaluation_reports", inspect(engine).get_table_names())
            with engine.connect() as connection:
                existing_reports = connection.scalar(
                    select(func.count()).select_from(EvaluationReportRow)
                )
            if existing_reports:
                self.skipTest(
                    "shared test evaluation_reports is nonempty; refusing destructive downgrade"
                )
            with patch.dict(os.environ, environment, clear=False):
                alembic_command.downgrade(config, "20260827_0002")
            tables = inspect(engine).get_table_names()
            self.assertNotIn("evaluation_reports", tables)
            self.assertIn("run_artifacts", tables)
        finally:
            with patch.dict(os.environ, environment, clear=False):
                alembic_command.upgrade(config, "head")
            engine.dispose()


@unittest.skipUnless(
    SAFE_POSTGRES_URL is not None,
    "PostgreSQL test configuration is unavailable",
)
class PersistentEvaluationPostgreSQLTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = create_postgres_engine(SAFE_POSTGRES_URL)
        cls.factory = create_session_factory(cls.engine)

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def test_app_rebuild_reads_persisted_evaluation(self):
        temp_dir = tempfile.TemporaryDirectory()
        root = Path(temp_dir.name)
        user_id = "p64-%s" % uuid4().hex
        seed_test_user(self.factory, user_id)
        clients = []
        run_id = None
        try:
            first = create_app(
                root / "first-runs",
                tick_seconds=0,
                project_root=root / "first-projects",
                run_repository_mode="postgresql",
                database_url=SAFE_POSTGRES_URL,
                current_user_provider=StaticCurrentUserProvider(user_id),
            )
            first_client = ASGITestClient(first)
            clients.append(first_client)
            run = first.state.run_service.create_run(
                user_id=user_id,
                project_id="project-1",
                execution_mode=ExecutionMode.SIMULATION,
                run_kind=RunKind.TRAIN,
                parameters={"seed": 11},
                project_config_snapshot={"evaluation_rubric": rubric()},
                code="policy",
            )
            run_id = run.run_id
            for status in (
                RunStatus.INITIALIZING,
                RunStatus.RUNNING,
                RunStatus.FINALIZING,
                RunStatus.COMPLETED,
            ):
                run = first.state.run_service.transition_status(run.run_id, status)
            run = first.state.repository.save_run(
                replace(
                    run,
                    metrics_summary={"success_rate": 0.9},
                    training_steps=80,
                    update_steps=20,
                )
            )
            with self.factory() as session:
                session.add(
                    RunArtifactRow(
                        artifact_id="artifact-%s" % uuid4().hex,
                        run_id=run.run_id,
                        artifact_type="manifest",
                        path="manifest.json",
                        size_bytes=42,
                        sha256="a" * 64,
                        is_primary_model=False,
                        created_at=datetime.now(timezone.utc),
                    )
                )
                session.commit()
            created = first_client.post(
                "/api/v1/evaluations", json={"run_id": run.run_id}
            )
            self.assertEqual(created.status_code, 201, created.text)
            evaluation_id = created.json()["evaluation_id"]
            replay = first_client.post(
                "/api/v1/evaluations", json={"run_id": run.run_id}
            )
            self.assertEqual(replay.status_code, 200, replay.text)
            self.assertEqual(replay.json(), created.json())
            self.assertEqual(replay.headers["idempotency-replayed"], "true")
            self.assertEqual(
                created.json()["evidence"]["artifacts"][0]["path"],
                "manifest.json",
            )

            second = create_app(
                root / "second-runs",
                tick_seconds=0,
                project_root=root / "second-projects",
                run_repository_mode="postgresql",
                database_url=SAFE_POSTGRES_URL,
                current_user_provider=StaticCurrentUserProvider(user_id),
            )
            second_client = ASGITestClient(second)
            clients.append(second_client)
            recovered = second_client.get(
                "/api/v1/evaluations/%s" % evaluation_id
            )
            self.assertEqual(recovered.status_code, 200, recovered.text)
            self.assertEqual(recovered.json(), created.json())
        finally:
            for client in reversed(clients):
                client.close()
            if run_id is not None:
                with self.factory() as session:
                    session.execute(
                        delete(EvaluationReportRow).where(
                            EvaluationReportRow.run_id == run_id
                        )
                    )
                    session.execute(
                        delete(RunArtifactRow).where(
                            RunArtifactRow.run_id == run_id
                        )
                    )
                    session.execute(
                        delete(RunIdempotencyKeyRow).where(
                            RunIdempotencyKeyRow.run_id == run_id
                        )
                    )
                    session.execute(
                        delete(RunEventRow).where(RunEventRow.run_id == run_id)
                    )
                    session.execute(delete(RunRow).where(RunRow.run_id == run_id))
                    session.commit()
            delete_test_user(self.factory, user_id)
            temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
