import argparse
from contextlib import contextmanager
from io import StringIO
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from apps.api.main import create_app
from scripts import check_midterm_runtime as check
from scripts import configure_midterm_runtime as configure
from scripts.bootstrap_midterm_demo import demo_names
from tests.http_client import ASGITestClient


class FakeResult:
    def __init__(self, rows=None, scalar=None):
        self.rows = rows or []
        self.scalar = scalar

    def mappings(self):
        return self

    def all(self):
        return self.rows

    def scalar_one(self):
        return self.scalar


class DemoConnection:
    def __init__(self, namespace="midterm-demo", complete=True, project_overrides=None):
        names = demo_names(namespace)
        self.names = names
        self.complete = complete
        self.project_overrides = project_overrides or {}
        self.users = {
            names.administrator_username: {"user_id": "user-admin", "username": names.administrator_username, "display_name": "Midterm Demo Administrator [%s]" % namespace, "enabled": True},
            names.instructor_username: {"user_id": "user-instructor", "username": names.instructor_username, "display_name": "Midterm Demo Instructor [%s]" % namespace, "enabled": True},
            names.student_username: {"user_id": "user-student", "username": names.student_username, "display_name": "Midterm Demo Student [%s]" % namespace, "enabled": True},
        }

    def execute(self, statement, parameters=None):
        sql = str(statement)
        if "FROM users WHERE username IN" in sql:
            rows = list(self.users.values())
            if not self.complete:
                rows = rows[:-1]
            return FakeResult(rows)
        if "FROM organizations WHERE" in sql:
            return FakeResult([{"organization_id": "org-1", "slug": self.names.organization_slug, "name": self.names.organization_name, "enabled": True}])
        if "FROM organization_memberships" in sql:
            return FakeResult([
                {"username": self.names.administrator_username, "role": "administrator", "enabled": True},
                {"username": self.names.instructor_username, "role": "instructor", "enabled": True},
                {"username": self.names.student_username, "role": "student", "enabled": True},
            ])
        if "FROM courses" in sql:
            return FakeResult([{"course_id": "course-1", "code": self.names.course_code, "title": self.names.course_title, "description": "S1 Push midterm demonstration course.", "owner_instructor_user_id": "user-instructor", "enabled": True}])
        if "FROM classes" in sql:
            return FakeResult([{"class_id": "class-1", "code": self.names.class_code, "name": self.names.class_name, "term": self.names.class_term, "instructor_user_id": "user-instructor", "enabled": True}])
        if "FROM enrollments" in sql:
            return FakeResult([{"enrollment_id": "enrollment-1", "enabled": True}])
        if "FROM teaching_projects" in sql:
            empty_sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            rows = [
                {"project_id": "project-short", "course_id": "course-1", "title": self.names.short_project_title, "description": "Canonical S1 Push short training profile.", "instructions": "Set only seed and inspect real training evidence.", "starter_code": "", "starter_code_sha256": empty_sha256, "training_profile": "short", "video_enabled": False, "parameter_schema": check.parameter_schema(), "evaluation_rubric": check.evaluation_rubric(check.TrainingProfile.SHORT), "config_version": 1, "enabled": True, "created_by_user_id": "user-admin"},
                {"project_id": "project-standard", "course_id": "course-1", "title": self.names.standard_project_title, "description": "Canonical S1 Push standard training profile.", "instructions": "Set only seed and inspect real training evidence.", "starter_code": "", "starter_code_sha256": empty_sha256, "training_profile": "standard", "video_enabled": False, "parameter_schema": check.parameter_schema(), "evaluation_rubric": check.evaluation_rubric(check.TrainingProfile.STANDARD, 3), "config_version": 3, "enabled": True, "created_by_user_id": "user-admin"},
            ]
            for row in rows:
                row.update(self.project_overrides)
            return FakeResult(rows)
        if "FROM runs r LEFT JOIN users" in sql:
            return FakeResult([])
        if "SELECT 1" in sql:
            return FakeResult(scalar=1)
        if "alembic_version" in sql:
            return FakeResult(scalar=check.EXPECTED_REVISION)
        raise AssertionError("unexpected SQL: %s" % sql)


class FakeEngine:
    def __init__(self, connection):
        self.connection = connection
        self.disposed = False

    @contextmanager
    def connect(self):
        yield self.connection

    def dispose(self):
        self.disposed = True


class ConfigureRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def args(self, *extra):
        return configure._parser().parse_args([
            "--config", str(self.root / "runtime.env"),
            "--database-name", "demo_db", "--database-user", "demo_user",
            "--project-root", str(self.root),
            "--artifact-root", str(self.root / "var" / "artifacts"),
            *extra,
        ])

    @staticmethod
    def defaults(prompt):
        return ""

    def test_default_is_preview_and_does_not_read_password_or_write(self):
        output = StringIO()
        code = configure.run(
            self.args(), output=output, error=StringIO(),
            password_reader=lambda prompt: self.fail("password reader called"),
        )
        self.assertEqual(code, 0)
        self.assertFalse((self.root / "runtime.env").exists())
        self.assertIn("zero writes", output.getvalue())

    def test_apply_writes_0600_and_never_prints_secrets(self):
        password = "p@ss:/?#[]!' special"
        secrets = ["jwt-secret-" + "a" * 48, "adapter-secret-" + "b" * 48, "callback-secret-" + "c" * 48]
        reads = iter([password, password])
        output, error = StringIO(), StringIO()
        with patch.object(configure.secrets, "token_urlsafe", side_effect=secrets):
            code = configure.run(
                self.args("--apply"), output=output, error=error,
                input_reader=self.defaults,
                password_reader=lambda prompt: next(reads),
            )
        path = self.root / "runtime.env"
        self.assertEqual(code, 0)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        rendered = output.getvalue() + error.getvalue()
        self.assertNotIn(password, rendered)
        for secret in secrets:
            self.assertNotIn(secret, rendered)
        values = check.parse_env_file(path)
        database_url = check.resolve_database_url(values["DATABASE_URL"])
        self.assertEqual(database_url.password, password)
        self.assertEqual(values["VITE_API_PROXY_TARGET"], values["BACKEND_ORIGIN"])
        self.assertEqual(values["SIMULATION_BACKEND_SERVICE_TOKEN"], values["INTERNAL_RUN_EVENT_SERVICE_TOKEN"])
        self.assertEqual(values["ARTIFACT_STORAGE_ROOT"], values["SIMULATION_ARTIFACTS_ROOT"])
        self.assertEqual(values["MIDTERM_PROJECT_ROOT"], str(self.root))

    def test_existing_file_refuses_without_replace_before_password(self):
        target = self.root / "runtime.env"
        target.write_text("existing", encoding="utf-8")
        called = []
        code = configure.run(
            self.args("--apply"), output=StringIO(), error=StringIO(),
            password_reader=lambda prompt: called.append(prompt),
        )
        self.assertEqual(code, 2)
        self.assertEqual(called, [])
        self.assertEqual(target.read_text(encoding="utf-8"), "existing")

    def test_apply_interactively_collects_all_non_secret_runtime_values(self):
        answers = iter([
            "db.internal", "5544", "demo_db", "demo_user",
            "http://127.0.0.1:8000", "http://127.0.0.1:8090",
            "redis://127.0.0.1:6379/7", "demo-isolated", "demo-gpu",
            str(self.root / "var" / "interactive-artifacts"),
        ])
        passwords = iter(["db-password", "db-password"])
        code = configure.run(
            configure._parser().parse_args([
                "--apply", "--config", str(self.root / "interactive.env"),
                "--project-root", str(self.root),
            ]),
            output=StringIO(), error=StringIO(),
            input_reader=lambda prompt: next(answers),
            password_reader=lambda prompt: next(passwords),
        )
        self.assertEqual(code, 0)
        values = check.parse_env_file(self.root / "interactive.env")
        database_url = check.resolve_database_url(values["DATABASE_URL"])
        self.assertEqual((database_url.host, database_url.port), ("db.internal", 5544))
        self.assertEqual(values["SIMULATION_REDIS_NAMESPACE"], "demo-isolated")
        self.assertEqual(values["SIMULATION_RQ_QUEUE"], "demo-gpu")

    def test_replace_requires_exact_interactive_confirmation(self):
        target = self.root / "runtime.env"
        target.write_text("existing", encoding="utf-8")
        code = configure.run(
            self.args("--apply", "--replace"), output=StringIO(), error=StringIO(),
            input_reader=lambda prompt: "no",
            password_reader=lambda prompt: self.fail("password reader called"),
        )
        self.assertEqual(code, 2)
        self.assertEqual(target.read_text(encoding="utf-8"), "existing")

    def test_template_and_gitignore_contain_no_usable_secret(self):
        template = (configure.PROJECT_ROOT / "config" / "midterm-runtime.env.example").read_text(encoding="utf-8")
        self.assertIn("<GENERATED_SECRET>", template)
        self.assertNotIn("password_hash", template)
        self.assertNotRegex(template, r"(?m)^(USER_JWT_SECRET|SIMULATION_SERVICE_TOKEN)='[A-Za-z0-9_-]{32,}'$")
        ignored = (configure.PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("midterm-runtime.env", ignored)
        self.assertNotEqual(configure.DEFAULT_CONFIG.parent, configure.PROJECT_ROOT)

    def video_args(self, *extra):
        for name in ("mediamtx.yml", "mediamtx", "ffmpeg", "ffprobe"):
            (self.root / name).touch()
        return configure._parser().parse_args([
            "--video-only", "--config", str(self.root / "runtime.env"),
            "--stream-public-base", "http://192.0.2.10:18889",
            "--mediamtx-ice-additional-host", "192.0.2.10",
            "--mediamtx-allowed-origin", "http://192.0.2.10:5173",
            "--mediamtx-config-path", str(self.root / "mediamtx.yml"),
            "--mediamtx-binary-path", str(self.root / "mediamtx"),
            "--simulation-ffmpeg-path", str(self.root / "ffmpeg"),
            *extra,
        ])

    def test_video_only_preview_is_zero_write_and_apply_preserves_secrets(self):
        target = self.root / "runtime.env"
        original = "USER_JWT_SECRET='never-display-or-change'\nSIMULATION_REDIS_NAMESPACE='midterm-demo'\n"
        target.write_text(original, encoding="utf-8")
        target.chmod(0o600)
        preview = StringIO()
        self.assertEqual(configure.run(self.video_args(), output=preview, error=StringIO()), 0)
        self.assertEqual(target.read_text(encoding="utf-8"), original)
        self.assertNotIn("never-display-or-change", preview.getvalue())
        output = StringIO()
        self.assertEqual(configure.run(self.video_args("--apply"), output=output, error=StringIO()), 0)
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
        rendered = target.read_text(encoding="utf-8")
        self.assertTrue(rendered.startswith(original.rstrip("\n")))
        self.assertIn("USER_JWT_SECRET='never-display-or-change'", rendered)
        self.assertNotIn("never-display-or-change", output.getvalue())

    def test_video_only_conflict_refuses_without_overwrite(self):
        target = self.root / "runtime.env"
        original = "SIMULATION_STREAM_PUBLIC_BASE_URL='http://10.0.0.1:18889'\n"
        target.write_text(original, encoding="utf-8")
        target.chmod(0o600)
        error = StringIO()
        self.assertEqual(configure.run(self.video_args("--apply"), output=StringIO(), error=error), 2)
        self.assertEqual(target.read_text(encoding="utf-8"), original)
        self.assertIn("conflict", error.getvalue())


class RuntimePreflightTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def add_process(self, proc_root, pid, state, executable, comm="mediamtx"):
        process = proc_root / str(pid)
        process.mkdir(parents=True)
        (process / "stat").write_text(
            "%d (%s) %s 1 1 1\n" % (pid, comm, state),
            encoding="ascii",
        )
        (process / "comm").write_text(comm + "\n", encoding="ascii")
        (process / "exe").symlink_to(executable)

    def test_database_audit_source_only_overrides_database_url_and_redacts_secret(self):
        database_config = self.root / "api.env"
        secret_url = "postgresql+psycopg://api-user:never-print@postgres:5432/api-db"
        database_config.write_text(
            "DATABASE_URL='%s'\nSIMULATION_REDIS_URL='redis://wrong:6379/0'\n"
            % secret_url,
            encoding="utf-8",
        )
        database_config.chmod(0o600)
        runtime_values = {
            "DATABASE_URL": "postgresql+psycopg://runtime:secret@127.0.0.1:5433/test",
            "SIMULATION_REDIS_URL": "redis://127.0.0.1:6379/0",
            "SIMULATION_SERVICE_TOKEN": "runtime-token",
        }
        output = StringIO()
        reporter = check.Reporter(output)

        resolved = check.database_values_for_audit(
            runtime_values, database_config, "172.18.0.2", reporter
        )

        self.assertIsNotNone(resolved)
        url, _ = check.redact_database_url(resolved["DATABASE_URL"])
        self.assertEqual(url.host, "172.18.0.2")
        self.assertEqual(url.database, "api-db")
        self.assertEqual(url.username, "api-user")
        self.assertEqual(
            resolved["SIMULATION_REDIS_URL"], runtime_values["SIMULATION_REDIS_URL"]
        )
        self.assertEqual(
            resolved["SIMULATION_SERVICE_TOKEN"], runtime_values["SIMULATION_SERVICE_TOKEN"]
        )
        self.assertNotIn("never-print", output.getvalue())
        self.assertNotIn(secret_url, output.getvalue())

    def test_database_audit_source_rejects_invalid_host_override(self):
        database_config = self.root / "api.env"
        database_config.write_text(
            "DATABASE_URL='postgresql+psycopg://api:secret@postgres:5432/api-db'\n",
            encoding="utf-8",
        )
        database_config.chmod(0o600)
        reporter = check.Reporter(StringIO())
        self.assertIsNone(
            check.database_values_for_audit(
                {}, database_config, "postgres;unsafe", reporter
            )
        )
        self.assertEqual(reporter.failures, 1)

    def test_mediamtx_process_count_zombie_only_is_zero(self):
        proc_root = self.root / "proc"
        binary = self.root / "mediamtx"
        binary.touch()
        self.add_process(proc_root, 101, "Z", binary)
        self.assertEqual(
            check._count_matching_processes(binary, proc_root=proc_root), 0
        )

    def test_mediamtx_process_count_zombie_and_live_is_one(self):
        proc_root = self.root / "proc"
        binary = self.root / "mediamtx"
        binary.touch()
        self.add_process(proc_root, 101, "Z", binary)
        self.add_process(proc_root, 102, "S", binary)
        self.assertEqual(
            check._count_matching_processes(binary, proc_root=proc_root), 1
        )

    def test_mediamtx_process_count_ignores_dead_and_similar_names(self):
        proc_root = self.root / "proc"
        binary = self.root / "mediamtx"
        other = self.root / "mediamtx-helper"
        binary.touch()
        other.touch()
        self.add_process(proc_root, 101, "X", binary)
        self.add_process(proc_root, 102, "S", other, comm="mediamtx-helper")
        self.assertEqual(
            check._count_matching_processes(binary, proc_root=proc_root), 0
        )

    def test_demo_namespace_complete_and_missing(self):
        complete_report = check.Reporter(StringIO())
        self.assertTrue(check.check_demo_namespace(DemoConnection(), "midterm-demo", complete_report))
        self.assertEqual(complete_report.failures, 0)
        missing_report = check.Reporter(StringIO())
        self.assertFalse(check.check_demo_namespace(DemoConnection(complete=False), "midterm-demo", missing_report))
        self.assertGreater(missing_report.failures, 0)

    def test_demo_namespace_rejects_tampered_project_facts(self):
        tampered_values = (
            {"created_by_user_id": "user-instructor"},
            {"training_profile": "SHORT"},
            {"video_enabled": 0},
            {"config_version": True},
            {"parameter_schema": {"type": "object"}},
            {"evaluation_rubric": {"version": "tampered"}},
        )
        for overrides in tampered_values:
            with self.subTest(overrides=overrides):
                reporter = check.Reporter(StringIO())
                self.assertFalse(
                    check.check_demo_namespace(
                        DemoConnection(project_overrides=overrides),
                        "midterm-demo",
                        reporter,
                    )
                )
                self.assertGreater(reporter.failures, 0)

    def test_database_accepts_head_and_rejects_non_head_or_bad_database(self):
        engine = FakeEngine(DemoConnection())
        report = check.Reporter(StringIO())
        with patch.object(check, "create_engine", return_value=engine), patch.object(check, "check_database_schema", return_value=True):
            returned = check.run_database_checks({"DATABASE_URL": "postgresql+psycopg://user:pass@127.0.0.1:5433/demo"}, "midterm-demo", report)
        self.assertIs(returned, engine)
        self.assertEqual(report.failures, 0)

        wrong_connection = DemoConnection()
        original_execute = wrong_connection.execute
        wrong_connection.execute = lambda statement, parameters=None: FakeResult(scalar="old") if "alembic_version" in str(statement) else original_execute(statement, parameters)
        wrong_report = check.Reporter(StringIO())
        with patch.object(check, "create_engine", return_value=FakeEngine(wrong_connection)), patch.object(check, "check_database_schema", return_value=True):
            check.run_database_checks({"DATABASE_URL": "postgresql+psycopg://user:pass@127.0.0.1:5433/demo"}, "midterm-demo", wrong_report)
        self.assertGreater(wrong_report.failures, 0)

        bad_report = check.Reporter(StringIO())
        self.assertIsNone(check.run_database_checks({"DATABASE_URL": "sqlite:///bad"}, "midterm-demo", bad_report))
        self.assertGreater(bad_report.failures, 0)

    def test_recovery_gate_lists_only_public_fields_and_fails(self):
        class RecoveryConnection:
            def execute(self, statement, parameters=None):
                return FakeResult([{
                    "run_id": "run-queued", "status": "queued", "created_at": "2026-09-06T00:00:00Z",
                    "username": "midterm-demo-student", "user_id": "user-student",
                    "project_id": "project-short", "class_id": "class-1",
                }])
        output = StringIO()
        reporter = check.Reporter(output)
        rows = check.list_recovery_targets(RecoveryConnection(), reporter)
        self.assertEqual(len(rows), 1)
        self.assertGreater(reporter.failures, 0)
        rendered = output.getvalue()
        self.assertIn("run-queued", rendered)
        self.assertIn("queued/unbound count", rendered)
        self.assertIn("stopping/bound count", rendered)
        self.assertNotIn("parameters", rendered)
        self.assertNotIn("snapshot", rendered)

    def test_token_mapping_only_reports_match_or_mismatch(self):
        secret = "never-print-this-" + "x" * 40
        output = StringIO()
        reporter = check.Reporter(output)
        self.assertTrue(check.check_token_mapping({
            "SIMULATION_SERVICE_TOKEN": secret,
            "INTERNAL_RUN_EVENT_SERVICE_TOKEN": secret,
            "SIMULATION_BACKEND_SERVICE_TOKEN": secret,
        }, reporter))
        self.assertNotIn(secret, output.getvalue())
        self.assertIn("MATCH", output.getvalue())
        mismatch = check.Reporter(StringIO())
        self.assertFalse(check.check_token_mapping({
            "SIMULATION_SERVICE_TOKEN": secret,
            "INTERNAL_RUN_EVENT_SERVICE_TOKEN": secret,
            "SIMULATION_BACKEND_SERVICE_TOKEN": "different-" + "y" * 40,
        }, mismatch))

    def test_origin_mapping_requires_frontend_and_simulation_callbacks_to_8000(self):
        values = {
            "BACKEND_ORIGIN": "http://127.0.0.1:8000",
            "VITE_API_PROXY_TARGET": "http://127.0.0.1:8000",
            "SIMULATION_BACKEND_BASE_URL": "http://127.0.0.1:8000",
            "SIMULATION_BASE_URL": "http://127.0.0.1:8090",
        }
        reporter = check.Reporter(StringIO())
        self.assertTrue(check.check_origin_mapping(values, reporter))
        values["SIMULATION_BACKEND_BASE_URL"] = "http://127.0.0.1:8002"
        mismatch = check.Reporter(StringIO())
        self.assertFalse(check.check_origin_mapping(values, mismatch))

    def test_artifact_root_must_match_and_stay_in_boundary(self):
        allowed = self.root / "var"
        target = allowed / "artifacts"
        reporter = check.Reporter(StringIO())
        self.assertTrue(check.check_artifacts({
            "ARTIFACT_STORAGE_ROOT": str(target),
            "SIMULATION_ARTIFACTS_ROOT": str(target),
        }, reporter, allowed_parent=allowed))
        self.assertTrue(target.is_dir())
        outside = check.Reporter(StringIO())
        self.assertFalse(check.check_artifacts({
            "ARTIFACT_STORAGE_ROOT": str(self.root / "outside"),
            "SIMULATION_ARTIFACTS_ROOT": str(self.root / "outside"),
        }, outside, allowed_parent=allowed))

    def test_artifact_root_uses_explicit_canonical_project_root(self):
        target = self.root / "var" / "artifacts"
        reporter = check.Reporter(StringIO())
        self.assertTrue(check.check_artifacts({
            "MIDTERM_PROJECT_ROOT": str(self.root),
            "ARTIFACT_STORAGE_ROOT": str(target),
            "SIMULATION_ARTIFACTS_ROOT": str(target),
        }, reporter))
        missing = check.Reporter(StringIO())
        self.assertFalse(check.check_artifacts({
            "ARTIFACT_STORAGE_ROOT": str(target),
            "SIMULATION_ARTIFACTS_ROOT": str(target),
        }, missing))

    def test_redis_check_is_read_only_and_requires_one_worker_empty_queue(self):
        commands = []
        class Probe:
            def __init__(self, url):
                pass
            def command(self, *parts):
                commands.append(parts)
                if parts[0] == "PING": return "PONG"
                if parts[0] == "SMEMBERS":
                    return (
                        [b"rq:worker:one"]
                        if parts[1].startswith("rq:workers:") else []
                    )
                if parts[0] == "HGET": return b"midterm-demo-gpu"
                return 0
        reporter = check.Reporter(StringIO())
        with patch.object(check, "RedisProbe", Probe):
            self.assertTrue(check.check_redis({
                "SIMULATION_REDIS_URL": "redis://127.0.0.1:6379/0",
                "SIMULATION_REDIS_NAMESPACE": "midterm-demo",
                "SIMULATION_RQ_QUEUE": "midterm-demo-gpu",
            }, reporter, before_run=True))
        self.assertEqual(reporter.failures, 0)
        self.assertTrue({item[0] for item in commands}.issubset({"PING", "SMEMBERS", "HGET", "LLEN", "ZCARD", "GET"}))
        self.assertIn(("SMEMBERS", "rq:workers:midterm-demo-gpu"), commands)

    def test_before_run_blocks_undelivered_callback_outbox(self):
        class Probe:
            def __init__(self, url):
                pass
            def command(self, *parts):
                if parts[0] == "PING": return "PONG"
                if parts[0] == "SMEMBERS":
                    if parts[1].startswith("rq:workers:"):
                        return [b"rq:worker:one"]
                    return [b"evt-pending"]
                if parts[0] == "HGET": return b"midterm-demo-gpu"
                if parts[0] == "GET":
                    return json.dumps({"delivery_status": "pending"}).encode()
                return 0
        output = StringIO()
        reporter = check.Reporter(output)
        with patch.object(check, "RedisProbe", Probe):
            self.assertFalse(check.check_redis({
                "SIMULATION_REDIS_URL": "redis://127.0.0.1:6379/0",
                "SIMULATION_REDIS_NAMESPACE": "midterm-demo",
                "SIMULATION_RQ_QUEUE": "midterm-demo-gpu",
            }, reporter, before_run=True))
        self.assertIn("undelivered=1", output.getvalue())

    def test_before_run_checks_canonical_ports_adapter_and_no_side_effect(self):
        output = StringIO()
        reporter = check.Reporter(output)
        def get_json(origin, path, token=None):
            if path == "/api/ready":
                return 200, {"database_ready": True, "simulation_configured": True, "artifact_download_configured": True}
            if path == "/openapi.json":
                return 200, {"paths": {"/api/v1/runs": {}}}
            return 404, {"code": "task_not_found"}
        with patch.object(check, "_get_json", side_effect=get_json), patch.object(check, "_reachable", return_value=True):
            check.check_before_run({
                "BACKEND_ORIGIN": "http://127.0.0.1:8000",
                "FRONTEND_ORIGIN": "http://127.0.0.1:5173",
                "SIMULATION_BASE_URL": "http://127.0.0.1:8090",
                "SIMULATION_SERVICE_TOKEN": "z" * 48,
            }, reporter)
        self.assertEqual(reporter.failures, 0)
        self.assertIn("no task created", output.getvalue())

    def test_before_run_rejects_noncanonical_ports(self):
        reporter = check.Reporter(StringIO())
        with patch.object(check, "_get_json", return_value=(404, {"code": "task_not_found"})), patch.object(check, "_reachable", return_value=True):
            check.check_before_run({
                "BACKEND_ORIGIN": "http://127.0.0.1:8001",
                "FRONTEND_ORIGIN": "http://127.0.0.1:5174",
                "SIMULATION_BASE_URL": "http://127.0.0.1:8091",
                "SIMULATION_SERVICE_TOKEN": "z" * 48,
            }, reporter)
        self.assertGreaterEqual(reporter.failures, 3)

    def test_before_video_pass_and_loopback_public_failure_are_redacted(self):
        for name in ("mediamtx.yml", "mediamtx", "ffmpeg", "ffprobe"):
            (self.root / name).touch()
        for name in ("mediamtx", "ffmpeg", "ffprobe"):
            (self.root / name).chmod(0o755)
        values = {
            "SIMULATION_STREAM_RTSP_BASE": "rtsp://127.0.0.1:18554",
            "SIMULATION_STREAM_PUBLIC_BASE_URL": "http://192.0.2.10:18889",
            "SIMULATION_FFMPEG_PATH": str(self.root / "ffmpeg"),
            "MEDIAMTX_CONTROL_URL": "http://127.0.0.1:19997",
            "MEDIAMTX_ICE_UDP_PORT": "18189",
            "MEDIAMTX_ICE_ADDITIONAL_HOST": "192.0.2.10",
            "MEDIAMTX_WEBRTC_ALLOWED_ORIGIN": "http://192.0.2.10:5173",
            "MEDIAMTX_CONFIG_PATH": str(self.root / "mediamtx.yml"),
            "MEDIAMTX_BINARY_PATH": str(self.root / "mediamtx"),
            "USER_JWT_SECRET": "must-not-appear",
        }
        actual = {
            "rtsp": True, "rtspAddress": "127.0.0.1:18554", "rtspTransports": ["tcp"],
            "webrtc": True, "webrtcAddress": ":18889",
            "webrtcLocalUDPAddress": ":18189", "webrtcIPsFromInterfaces": False,
            "webrtcAdditionalHosts": ["192.0.2.10"],
            "webrtcAllowOrigins": ["http://192.0.2.10:5173"],
            "rtmp": False, "hls": False, "srt": False, "moq": False,
            "metrics": False, "pprof": False, "playback": False,
        }
        output = StringIO(); reporter = check.Reporter(output)
        with patch.object(check, "_get_json", return_value=(200, actual)), \
             patch.object(check, "_tcp_listening", return_value=True), \
             patch.object(check, "_udp_listening", return_value=True), \
             patch.object(check, "_count_matching_processes", return_value=1):
            check.check_before_video(values, reporter)
        self.assertEqual(reporter.failures, 0)
        self.assertNotIn("must-not-appear", output.getvalue())

        (self.root / "ffprobe").unlink()
        missing_probe = check.Reporter(StringIO())
        with patch.object(check, "_get_json", return_value=(200, actual)), \
             patch.object(check, "_tcp_listening", return_value=True), \
             patch.object(check, "_udp_listening", return_value=True), \
             patch.object(check, "_count_matching_processes", return_value=1):
            check.check_before_video(values, missing_probe)
        self.assertEqual(missing_probe.failures, 1)
        (self.root / "ffprobe").touch()
        non_executable_probe = check.Reporter(StringIO())
        with patch.object(check, "_get_json", return_value=(200, actual)), \
             patch.object(check, "_tcp_listening", return_value=True), \
             patch.object(check, "_udp_listening", return_value=True), \
             patch.object(check, "_count_matching_processes", return_value=1):
            check.check_before_video(values, non_executable_probe)
        self.assertEqual(non_executable_probe.failures, 1)
        (self.root / "ffprobe").chmod(0o755)

        two_live_output = StringIO()
        two_live = check.Reporter(two_live_output)
        with patch.object(check, "_get_json", return_value=(200, actual)), \
             patch.object(check, "_tcp_listening", return_value=True), \
             patch.object(check, "_udp_listening", return_value=True), \
             patch.object(check, "_count_matching_processes", return_value=2):
            check.check_before_video(values, two_live)
        self.assertEqual(two_live.failures, 1)
        self.assertIn("expected 1, found 2", two_live_output.getvalue())

        values["SIMULATION_STREAM_PUBLIC_BASE_URL"] = "http://127.0.0.1:18889"
        failed = check.Reporter(StringIO())
        with patch.object(check, "_get_json", return_value=(200, actual)), \
             patch.object(check, "_tcp_listening", return_value=True), \
             patch.object(check, "_udp_listening", return_value=True), \
             patch.object(check, "_count_matching_processes", return_value=1):
            check.check_before_video(values, failed)
        self.assertGreater(failed.failures, 0)

    def test_before_video_stage_skips_database_and_is_read_only(self):
        args = check._parser().parse_args([
            "--stage", "before-video", "--namespace", "midterm-demo",
            "--config", str(self.root / "runtime.env"),
        ])
        values = {"SIMULATION_REDIS_NAMESPACE": "midterm-demo"}
        with patch.object(check, "check_config", return_value=values), \
             patch.object(check, "check_redis", return_value=True) as redis_check, \
             patch.object(check, "check_before_video") as video_check, \
             patch.object(check, "run_database_checks", side_effect=AssertionError("database touched")):
            self.assertEqual(check.run(args, output=StringIO()), 0)
        redis_check.assert_called_once_with(values, unittest.mock.ANY, before_run=True)
        video_check.assert_called_once()

    def test_midterm_mediamtx_yaml_matches_v1193_surface(self):
        import yaml
        path = configure.PROJECT_ROOT / "simulation" / "config" / "mediamtx_midterm.yml"
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(values["rtspAddress"], "127.0.0.1:18554")
        self.assertEqual(values["webrtcAddress"], ":18889")
        self.assertEqual(values["webrtcLocalUDPAddress"], ":18189")
        self.assertEqual(values["apiAddress"], "127.0.0.1:19997")
        self.assertEqual(values["webrtcAdditionalHosts"], ["192.0.2.10"])
        self.assertEqual(values["webrtcAllowOrigins"], ["http://192.0.2.10:5173"])


class ReadinessEndpointTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def app(self):
        return create_app(
            storage_root=self.root / "runs",
            project_root=self.root / "projects",
            run_repository_mode="memory",
        )

    def test_readiness_503_is_redacted_and_health_remains_compatible(self):
        with ASGITestClient(self.app()) as client:
            response = client.get("/api/ready")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json(), {
                "status": "not_ready", "database_ready": False,
                "simulation_configured": False, "artifact_download_configured": False,
            })
            self.assertEqual(client.get("/api/health").json()["adapter"], "mock")
            rendered = response.text.lower()
            for forbidden in ("postgresql://", "token", "secret", "/home/"):
                self.assertNotIn(forbidden, rendered)

    def test_readiness_200_reports_only_capability_booleans(self):
        app = self.app()
        app.state.engine = object()
        app.state.simulation_client = object()
        app.state.artifact_download_service = object()
        with patch("apps.api.main.check_postgres_readiness", return_value=None):
            with ASGITestClient(app) as client:
                response = client.get("/api/ready")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "status": "ready", "database_ready": True,
            "simulation_configured": True, "artifact_download_configured": True,
        })

    def test_readiness_returns_redacted_503_when_database_probe_raises(self):
        app = self.app()
        app.state.engine = object()
        with patch("apps.api.main.check_postgres_readiness", side_effect=RuntimeError("postgresql://user:secret@host/db")):
            with ASGITestClient(app) as client:
                response = client.get("/api/ready")
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("secret", response.text.lower())
        self.assertNotIn("postgresql://", response.text.lower())


if __name__ == "__main__":
    unittest.main()
