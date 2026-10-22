import io
import logging
import os
import shutil
import subprocess
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import yaml
from alembic import command
from alembic.config import Config
from sqlalchemy import CheckConstraint, DateTime, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB

from apps.api.database import (
    DatabaseConfigurationError,
    create_postgres_engine,
    resolve_database_url,
    session_scope,
)
from robotics.runs.db_models import Base


ROOT = Path(__file__).resolve().parents[1]


class PostgreSQLFoundationTest(unittest.TestCase):
    def test_metadata_contains_only_the_twelve_canonical_tables(self):
        self.assertEqual(
            set(Base.metadata.tables),
            {
                "runs",
                "run_events",
                "run_idempotency_keys",
                "run_artifacts",
                "evaluation_reports",
                "users",
                "organizations",
                "organization_memberships",
                "courses",
                "classes",
                "enrollments",
                "teaching_projects",
            },
        )

    def test_primary_keys_unique_constraints_and_indexes_exist(self):
        runs = Base.metadata.tables["runs"]
        events = Base.metadata.tables["run_events"]
        keys = Base.metadata.tables["run_idempotency_keys"]
        artifacts = Base.metadata.tables["run_artifacts"]
        evaluations = Base.metadata.tables["evaluation_reports"]
        users = Base.metadata.tables["users"]
        organizations = Base.metadata.tables["organizations"]
        memberships = Base.metadata.tables["organization_memberships"]
        courses = Base.metadata.tables["courses"]
        classes = Base.metadata.tables["classes"]
        enrollments = Base.metadata.tables["enrollments"]
        projects = Base.metadata.tables["teaching_projects"]
        self.assertEqual([column.name for column in runs.primary_key.columns], ["run_id"])
        self.assertEqual(
            [column.name for column in events.primary_key.columns], ["event_id"]
        )
        self.assertEqual([column.name for column in keys.primary_key.columns], ["id"])
        self.assertIn(
            ("simulation_task_id",), self._unique_column_sets(runs)
        )
        self.assertIn(("run_id", "sequence"), self._unique_column_sets(events))
        self.assertIn(
            ("user_id", "idempotency_key"), self._unique_column_sets(keys)
        )
        self.assertIn(("run_id", "path"), self._unique_column_sets(artifacts))
        self.assertEqual(
            [column.name for column in evaluations.primary_key.columns],
            ["evaluation_id"],
        )
        self.assertIn(
            (
                "run_id",
                "rubric_version",
                "evaluator_version",
                "evidence_sha256",
            ),
            self._unique_column_sets(evaluations),
        )
        self.assertEqual(
            [column.name for column in users.primary_key.columns], ["user_id"]
        )
        self.assertIn(("username",), self._unique_column_sets(users))
        self.assertEqual(
            [column.name for column in organizations.primary_key.columns],
            ["organization_id"],
        )
        self.assertIn(("slug",), self._unique_column_sets(organizations))
        self.assertEqual(
            [column.name for column in memberships.primary_key.columns],
            ["membership_id"],
        )
        self.assertIn(
            ("organization_id", "user_id"),
            self._unique_column_sets(memberships),
        )
        self.assertEqual(
            [column.name for column in courses.primary_key.columns],
            ["course_id"],
        )
        self.assertIn(
            ("organization_id", "code"), self._unique_column_sets(courses)
        )
        self.assertEqual(
            [column.name for column in classes.primary_key.columns],
            ["class_id"],
        )
        self.assertIn(
            ("course_id", "term", "code"),
            self._unique_column_sets(classes),
        )
        self.assertEqual(
            [column.name for column in enrollments.primary_key.columns],
            ["enrollment_id"],
        )
        self.assertIn(
            ("class_id", "user_id"),
            self._unique_column_sets(enrollments),
        )
        self.assertEqual(
            [column.name for column in projects.primary_key.columns],
            ["project_id"],
        )
        self.assertIn(
            ("course_id", "title"), self._unique_column_sets(projects)
        )
        role_checks = {
            str(item.sqltext)
            for item in memberships.constraints
            if isinstance(item, CheckConstraint)
        }
        self.assertEqual(
            role_checks,
            {"role IN ('student', 'instructor', 'administrator')"},
        )
        project_checks = {
            str(item.sqltext)
            for item in projects.constraints
            if isinstance(item, CheckConstraint)
        }
        self.assertIn(
            "training_profile IN ('short', 'standard')", project_checks
        )
        self.assertIn("config_version >= 1", project_checks)
        self.assertIn("ix_runs_created_at", {index.name for index in runs.indexes})
        self.assertIn("ix_runs_class_id", {index.name for index in runs.indexes})
        self.assertIn("ix_run_events_run_id", {index.name for index in events.indexes})
        self.assertIn(
            "ix_run_artifacts_run_id", {index.name for index in artifacts.indexes}
        )
        self.assertIn(
            "ix_evaluation_reports_run_id",
            {index.name for index in evaluations.indexes},
        )

    def test_foreign_keys_target_authoritative_parent_tables(self):
        runs = Base.metadata.tables["runs"]
        events = Base.metadata.tables["run_events"]
        keys = Base.metadata.tables["run_idempotency_keys"]
        artifacts = Base.metadata.tables["run_artifacts"]
        evaluations = Base.metadata.tables["evaluation_reports"]
        self.assertEqual(
            {foreign_key.target_fullname for foreign_key in runs.foreign_keys},
            {"classes.class_id"},
        )
        self.assertEqual(
            {foreign_key.target_fullname for foreign_key in events.foreign_keys},
            {"runs.run_id"},
        )
        memberships = Base.metadata.tables["organization_memberships"]
        self.assertEqual(
            {
                foreign_key.target_fullname
                for foreign_key in memberships.foreign_keys
            },
            {"organizations.organization_id", "users.user_id"},
        )
        self.assertEqual(
            {foreign_key.target_fullname for foreign_key in keys.foreign_keys},
            {"runs.run_id"},
        )
        self.assertEqual(
            {foreign_key.target_fullname for foreign_key in artifacts.foreign_keys},
            {"runs.run_id"},
        )
        self.assertEqual(
            {
                foreign_key.target_fullname
                for foreign_key in evaluations.foreign_keys
            },
            {"runs.run_id"},
        )
        courses = Base.metadata.tables["courses"]
        classes = Base.metadata.tables["classes"]
        enrollments = Base.metadata.tables["enrollments"]
        projects = Base.metadata.tables["teaching_projects"]
        self.assertEqual(
            {foreign_key.target_fullname for foreign_key in courses.foreign_keys},
            {"organizations.organization_id", "users.user_id"},
        )
        self.assertEqual(
            {foreign_key.target_fullname for foreign_key in classes.foreign_keys},
            {"courses.course_id", "users.user_id"},
        )
        self.assertEqual(
            {
                foreign_key.target_fullname
                for foreign_key in enrollments.foreign_keys
            },
            {"classes.class_id", "users.user_id"},
        )
        self.assertEqual(
            {
                foreign_key.target_fullname
                for foreign_key in projects.foreign_keys
            },
            {"courses.course_id", "users.user_id"},
        )

    def test_structured_fields_use_postgresql_jsonb(self):
        runs = Base.metadata.tables["runs"]
        events = Base.metadata.tables["run_events"]
        for name in (
            "parameters",
            "project_config_snapshot",
            "metrics_summary",
            "error",
        ):
            self.assertIsInstance(runs.c[name].type, JSONB)
        self.assertIsInstance(events.c.payload.type, JSONB)
        evaluations = Base.metadata.tables["evaluation_reports"]
        for name in (
            "rubric_snapshot",
            "evidence",
            "summary",
            "limitations",
        ):
            self.assertIsInstance(evaluations.c[name].type, JSONB)
        projects = Base.metadata.tables["teaching_projects"]
        self.assertIsInstance(projects.c.parameter_schema.type, JSONB)
        self.assertIsInstance(projects.c.evaluation_rubric.type, JSONB)

    def test_all_time_columns_are_timezone_aware(self):
        expected = {
            "runs": {"created_at", "started_at", "ended_at", "updated_at"},
            "run_events": {"occurred_at"},
            "run_idempotency_keys": {"created_at"},
            "run_artifacts": {"created_at"},
            "evaluation_reports": {"created_at"},
            "users": {"created_at", "updated_at"},
            "organizations": {"created_at", "updated_at"},
            "organization_memberships": {"created_at", "updated_at"},
            "courses": {"created_at", "updated_at"},
            "classes": {"created_at", "updated_at"},
            "enrollments": {"created_at", "updated_at"},
            "teaching_projects": {"created_at", "updated_at"},
        }
        for table_name, column_names in expected.items():
            table = Base.metadata.tables[table_name]
            for column_name in column_names:
                self.assertIsInstance(table.c[column_name].type, DateTime)
                self.assertTrue(table.c[column_name].type.timezone)

    def test_large_artifact_content_is_not_in_core_tables(self):
        forbidden = {"checkpoint", "model", "log", "artifact", "file_content"}
        persisted = {
            column.name
            for table in Base.metadata.tables.values()
            for column in table.columns
        }
        self.assertTrue(forbidden.isdisjoint(persisted))

    def test_alembic_configuration_contains_no_database_url_or_password(self):
        contents = (ROOT / "alembic.ini").read_text(encoding="utf-8")
        self.assertNotIn("sqlalchemy.url", contents)
        self.assertNotIn("postgresql://", contents)
        self.assertNotIn("postgresql+psycopg://", contents)

    def test_offline_upgrade_sql_is_generated_without_secret(self):
        output = self._offline_sql("upgrade")
        lowered = output.lower()
        self.assertIn("create table runs", lowered)
        self.assertIn("create table run_events", lowered)
        self.assertIn("create table run_idempotency_keys", lowered)
        self.assertIn("create table run_artifacts", lowered)
        self.assertIn("create table evaluation_reports", lowered)
        self.assertIn("create table users", lowered)
        self.assertIn("create table organizations", lowered)
        self.assertIn("create table organization_memberships", lowered)
        self.assertIn("create table courses", lowered)
        self.assertIn("create table classes", lowered)
        self.assertIn("create table enrollments", lowered)
        self.assertIn("create table teaching_projects", lowered)
        self.assertIn("jsonb", lowered)
        self.assertNotIn("phase32-secret", output)

    def test_offline_downgrade_sql_is_generated_without_secret(self):
        output = self._offline_sql("downgrade")
        lowered = output.lower()
        self.assertIn("drop table run_idempotency_keys", lowered)
        self.assertIn("drop table run_artifacts", lowered)
        self.assertIn("drop table evaluation_reports", lowered)
        self.assertIn("drop table users", lowered)
        self.assertIn("drop table organizations", lowered)
        self.assertIn("drop table organization_memberships", lowered)
        self.assertIn("drop table courses", lowered)
        self.assertIn("drop table classes", lowered)
        self.assertIn("drop table enrollments", lowered)
        self.assertIn("drop table teaching_projects", lowered)
        self.assertIn("drop table run_events", lowered)
        self.assertIn("drop table runs", lowered)
        self.assertNotIn("phase32-secret", output)

    def test_database_url_missing_has_clear_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(DatabaseConfigurationError, "DATABASE_URL"):
                resolve_database_url()

    def test_non_postgresql_url_is_rejected_without_fallback(self):
        with self.assertRaisesRegex(DatabaseConfigurationError, "PostgreSQL"):
            resolve_database_url("sqlite:///temporary.db")

    def test_test_database_url_can_be_injected_without_logging_password(self):
        password = "test-password-not-for-logs"
        url = "postgresql+psycopg://tester:%s@db/schema_test" % password
        resolved = resolve_database_url(url)
        engine = create_postgres_engine(url)
        try:
            self.assertEqual(resolved.database, "schema_test")
            self.assertNotIn(password, resolved.render_as_string(hide_password=True))
            self.assertNotIn(password, repr(engine.url))
            self.assertTrue(engine.hide_parameters)
        finally:
            engine.dispose()

    def test_session_scope_commits_success_and_rolls_back_failure(self):
        successful = _FakeSession()
        with session_scope(lambda: successful):
            pass
        self.assertEqual(successful.calls, ["commit", "close"])

        failed = _FakeSession()
        with self.assertRaisesRegex(RuntimeError, "transaction failed"):
            with session_scope(lambda: failed):
                raise RuntimeError("transaction failed")
        self.assertEqual(failed.calls, ["rollback", "close"])

    def test_compose_yaml_defines_postgres_health_and_named_volume(self):
        compose_path = ROOT / "docker-compose.yml"
        compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
        postgres = compose["services"]["postgres"]
        api = compose["services"]["api"]
        self.assertEqual(postgres["image"], "postgres:16.15-bookworm")
        self.assertIn("healthcheck", postgres)
        self.assertEqual(postgres["restart"], "unless-stopped")
        self.assertIn("postgres_data", compose["volumes"])
        self.assertIn("studio_data", compose["volumes"])
        self.assertIn("postgres:5432", api["environment"]["DATABASE_URL"])
        self.assertEqual(
            {
                "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL",
                "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL",
                "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
            },
            {
                name
                for name in api["environment"]
                if name.startswith(("OPENAI_", "DEEPSEEK_", "ANTHROPIC_"))
            },
        )
        self.assertEqual(api["restart"], "unless-stopped")
        self.assertIn("build", api)
        self.assertIn("healthcheck", api)
        self.assertEqual(
            api["depends_on"]["postgres"]["condition"], "service_healthy"
        )
        self.assertEqual(
            api["depends_on"]["migrate"]["condition"],
            "service_completed_successfully",
        )
        self.assertEqual(compose["services"]["web"]["restart"], "unless-stopped")

    def test_docker_compose_config_when_command_is_available(self):
        if not self._compose_command_available():
            self.assertIsNotNone(
                yaml.safe_load((ROOT / "docker-compose.yml").read_text("utf-8"))
            )
            return
        completed = subprocess.run(
            ["docker", "compose", "config", "--quiet"],
            cwd=str(ROOT),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
            universal_newlines=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    @staticmethod
    def _unique_column_sets(table):
        return {
            tuple(column.name for column in constraint.columns)
            for constraint in table.constraints
            if isinstance(constraint, UniqueConstraint)
        }

    @staticmethod
    def _compose_command_available():
        if shutil.which("docker") is None:
            return False
        completed = subprocess.run(
            ["docker", "compose", "version"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        return completed.returncode == 0

    @staticmethod
    def _offline_sql(direction):
        config = Config(str(ROOT / "alembic.ini"))
        output = io.StringIO()
        environment = {
            "DATABASE_URL": (
                "postgresql+psycopg://schema_user:phase32-secret@db/schema_test"
            )
        }
        with patch.dict(os.environ, environment, clear=False):
            with redirect_stdout(output), redirect_stderr(output):
                if direction == "upgrade":
                    command.upgrade(config, "head", sql=True)
                else:
                    command.downgrade(config, "20260831_0007:base", sql=True)
        return output.getvalue()


class _FakeSession:
    def __init__(self):
        self.calls = []

    def commit(self):
        self.calls.append("commit")

    def rollback(self):
        self.calls.append("rollback")

    def close(self):
        self.calls.append("close")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
