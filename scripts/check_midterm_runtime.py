#!/usr/bin/env python3
"""Read-only staged preflight for the canonical midterm runtime."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shlex
import socket
import ssl
import stat
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, TextIO, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL

from apps.api.database import resolve_database_url
from robotics.projects import (
    ALGORITHM,
    ENVIRONMENT_TYPE,
    TASK_TYPE,
    TrainingProfile,
    current_config_version,
    evaluation_rubric,
    parameter_schema,
    profile_config,
    starter_code_digest,
)
from scripts.bootstrap_midterm_demo import demo_names


EXPECTED_REVISION = "20260903_0008"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("/home/user/.config/embodied-arm-studio/midterm-runtime.env")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
SAFE_DATABASE_HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,252}$")
REQUIRED_COLUMNS = {
    "alembic_version": {"version_num"},
    "users": {"user_id", "username", "enabled"},
    "organizations": {"organization_id", "slug", "enabled"},
    "organization_memberships": {"organization_id", "user_id", "role", "enabled"},
    "courses": {"course_id", "organization_id", "code", "owner_instructor_user_id", "enabled"},
    "classes": {"class_id", "course_id", "instructor_user_id", "code", "term", "enabled"},
    "enrollments": {"enrollment_id", "class_id", "user_id", "enabled"},
    "teaching_projects": {
        "project_id", "course_id", "title", "description", "instructions",
        "starter_code", "starter_code_sha256", "training_profile",
        "video_enabled", "parameter_schema", "evaluation_rubric",
        "config_version", "enabled", "created_by_user_id",
    },
    "runs": {
        "run_id", "user_id", "project_id", "class_id", "status",
        "execution_mode", "simulation_task_id", "created_at",
    },
    "run_events": {"event_id", "run_id", "sequence", "payload"},
    "run_artifacts": {"artifact_id", "run_id", "path", "artifact_type"},
    "evaluation_reports": {"evaluation_id", "run_id", "created_at"},
}


@dataclass
class Reporter:
    output: TextIO = sys.stdout
    failures: int = 0

    def ok(self, label: str, detail: str = "PASS") -> None:
        print("PASS %-32s %s" % (label, detail), file=self.output)

    def fail(self, label: str, detail: str) -> None:
        self.failures += 1
        print("FAIL %-32s %s" % (label, detail), file=self.output)

    def info(self, label: str, detail: str) -> None:
        print("INFO %-32s %s" % (label, detail), file=self.output)


def parse_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError("invalid config line %d" % line_number)
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Z_a-z][A-Z_a-z0-9]*", key):
            raise ValueError("invalid config key on line %d" % line_number)
        try:
            tokens = shlex.split(raw_value, posix=True)
        except ValueError as exc:
            raise ValueError("invalid quoted value on line %d" % line_number) from exc
        if len(tokens) != 1:
            raise ValueError("config values must be one quoted scalar on line %d" % line_number)
        values[key] = tokens[0]
    return values


def redact_database_url(raw_url: str) -> Tuple[URL, str]:
    url = resolve_database_url(raw_url)
    return url, "%s://%s:%s/%s" % (
        url.drivername,
        url.host or "<host>",
        url.port or "<default>",
        url.database or "<database>",
    )


def check_config(path: Path, reporter: Reporter) -> Optional[Dict[str, str]]:
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode == 0o600:
            reporter.ok("config permission", "0600")
        else:
            reporter.fail("config permission", "must be 0600")
        values = parse_env_file(path)
        reporter.ok("config syntax")
        return values
    except (OSError, ValueError) as exc:
        reporter.fail("config file", str(exc))
        return None


def database_values_for_audit(
    runtime_values: Mapping[str, str],
    database_config: Optional[Path],
    database_host_override: Optional[str],
    reporter: Reporter,
) -> Optional[Dict[str, str]]:
    """Use a separately sourced Backend DATABASE_URL without replacing runtime gates."""

    resolved = dict(runtime_values)
    if database_config is None:
        return resolved
    source_path = database_config.expanduser()
    source_values = check_config(source_path, reporter)
    if source_values is None:
        return None
    resolved["DATABASE_URL"] = source_values.get("DATABASE_URL", "")
    if database_host_override:
        if not SAFE_DATABASE_HOST.fullmatch(database_host_override):
            reporter.fail("database host override", "invalid host")
            return None
        url = resolve_database_url(resolved["DATABASE_URL"])
        resolved["DATABASE_URL"] = url.set(
            host=database_host_override
        ).render_as_string(hide_password=False)
    reporter.info("database config source", str(source_path))
    return resolved


def check_database_schema(engine: Any, reporter: Reporter) -> bool:
    passed = True
    try:
        inspector = inspect(engine)
        for table_name, required in REQUIRED_COLUMNS.items():
            if not inspector.has_table(table_name):
                reporter.fail("schema " + table_name, "table missing")
                passed = False
                continue
            actual = {column["name"] for column in inspector.get_columns(table_name)}
            missing = sorted(required - actual)
            if missing:
                reporter.fail("schema " + table_name, "missing columns: " + ",".join(missing))
                passed = False
        if passed:
            reporter.ok("required schema")
    except Exception:
        reporter.fail("required schema", "inspection failed")
        passed = False
    return passed


def _rows(connection: Any, statement: str, parameters: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [dict(row) for row in connection.execute(text(statement), parameters).mappings().all()]


def _project_facts_match(
    row: Optional[Mapping[str, Any]],
    *,
    course_id: str,
    title: str,
    profile: TrainingProfile,
    video_enabled: bool,
    expected_training_steps: int,
    config_version: int,
    created_by_user_id: str,
    environment_type: str = "s1_push",
) -> bool:
    """Compare stored and canonically derived public Project facts strictly."""

    if row is None:
        return False
    expected_schema = parameter_schema()
    expected_schema["x-environment-type"] = environment_type
    legacy_schema = dict(expected_schema)
    legacy_schema.pop("x-environment-type", None)
    accepted_schemas = (
        (expected_schema, legacy_schema)
        if environment_type == "s1_push" and profile is TrainingProfile.SHORT
        else (expected_schema,)
    )
    expected_rubric = evaluation_rubric(profile, config_version)
    expected_rubric["version"] = "%s-training-evidence-v1" % environment_type.replace("_", "-")
    return (
        row["course_id"] == course_id
        and row["title"] == title
        and type(row["training_profile"]) is str
        and row["training_profile"] == profile.value
        and row["description"] == "Canonical %s %s training profile." % (
            "S1 Reach" if environment_type == "s1_reach" else "S1 Push",
            profile.value,
        )
        and row["instructions"] == "Set only seed and inspect real training evidence."
        and row["starter_code"] == ""
        and row["starter_code_sha256"] == starter_code_digest("")
        and type(row["video_enabled"]) is bool
        and row["video_enabled"] is video_enabled
        and type(row["config_version"]) is int
        and row["config_version"] == config_version
        and isinstance(row["parameter_schema"], dict)
        and row["parameter_schema"] in accepted_schemas
        and isinstance(row["evaluation_rubric"], dict)
        and row["evaluation_rubric"] == expected_rubric
        and type(row["enabled"]) is bool
        and row["enabled"] is True
        and row["created_by_user_id"] == created_by_user_id
        and TASK_TYPE == "training"
        and environment_type in {
            ENVIRONMENT_TYPE, "s1_reach", "s1_grasp", "s1_lift",
            "s1_lower_release",
        }
        and ALGORITHM == "PPO"
        and type(expected_training_steps) is int
        and profile_config(profile, config_version)["expected_training_steps"]
        == expected_training_steps
    )


def check_demo_namespace(connection: Any, namespace: str, reporter: Reporter) -> bool:
    names = demo_names(namespace)
    reach_project_title = names.reach_project_title
    good = True
    usernames = [
        names.administrator_username,
        names.instructor_username,
        names.student_username,
    ]
    expected_display_names = {
        names.administrator_username: "Midterm Demo Administrator [%s]" % namespace,
        names.instructor_username: "Midterm Demo Instructor [%s]" % namespace,
        names.student_username: "Midterm Demo Student [%s]" % namespace,
    }
    users = _rows(
        connection,
        "SELECT user_id, username, display_name, enabled FROM users "
        "WHERE username IN (:administrator, :instructor, :student)",
        {
            "administrator": usernames[0],
            "instructor": usernames[1],
            "student": usernames[2],
        },
    )
    by_username = {row["username"]: row for row in users}
    for username in usernames:
        row = by_username.get(username)
        if (
            row is None or not row["enabled"]
            or row["display_name"] != expected_display_names[username]
        ):
            reporter.fail("demo user", "%s missing-or-inconsistent" % username)
            good = False
        else:
            reporter.ok("demo user", "%s enabled" % username)

    organizations = _rows(
        connection,
        "SELECT organization_id, slug, name, enabled FROM organizations WHERE slug=:slug",
        {"slug": names.organization_slug},
    )
    organization = organizations[0] if len(organizations) == 1 else None
    if organization is None or organization["name"] != names.organization_name or not organization["enabled"]:
        reporter.fail("demo organization", "%s missing-or-inconsistent" % names.organization_slug)
        return False
    reporter.ok("demo organization", "%s enabled" % names.organization_slug)

    expected_roles = {
        names.administrator_username: "administrator",
        names.instructor_username: "instructor",
        names.student_username: "student",
    }
    memberships = _rows(
        connection,
        "SELECT u.username, m.role, m.enabled FROM organization_memberships m "
        "JOIN users u ON u.user_id=m.user_id WHERE m.organization_id=:organization_id",
        {"organization_id": organization["organization_id"]},
    )
    role_rows = {row["username"]: row for row in memberships if row["username"] in expected_roles}
    for username, expected_role in expected_roles.items():
        row = role_rows.get(username)
        if row is None or row["role"] != expected_role or not row["enabled"]:
            reporter.fail("demo membership", "%s/%s missing-or-disabled" % (username, expected_role))
            good = False
        else:
            reporter.ok("demo membership", "%s/%s enabled" % (username, expected_role))

    administrator = by_username.get(names.administrator_username)
    instructor = by_username.get(names.instructor_username)
    student = by_username.get(names.student_username)
    courses = _rows(
        connection,
        "SELECT course_id, code, title, description, owner_instructor_user_id, enabled FROM courses "
        "WHERE organization_id=:organization_id AND code=:code",
        {"organization_id": organization["organization_id"], "code": names.course_code},
    )
    course = courses[0] if len(courses) == 1 else None
    if (
        course is None or instructor is None or course["title"] != names.course_title
        or course["description"] != "S1 Push midterm demonstration course."
        or course["owner_instructor_user_id"] != instructor["user_id"] or not course["enabled"]
    ):
        reporter.fail("demo course", "%s missing-or-inconsistent" % names.course_code)
        return False
    reporter.ok("demo course", "%s enabled" % names.course_code)

    classes = _rows(
        connection,
        "SELECT class_id, code, name, term, instructor_user_id, enabled FROM classes "
        "WHERE course_id=:course_id AND code=:code AND term=:term",
        {"course_id": course["course_id"], "code": names.class_code, "term": names.class_term},
    )
    teaching_class = classes[0] if len(classes) == 1 else None
    if (
        teaching_class is None or instructor is None or teaching_class["name"] != names.class_name
        or teaching_class["instructor_user_id"] != instructor["user_id"] or not teaching_class["enabled"]
    ):
        reporter.fail("demo class", "%s missing-or-inconsistent" % names.class_code)
        return False
    reporter.ok("demo class", "%s enabled" % names.class_code)

    enrollments = _rows(
        connection,
        "SELECT enrollment_id, enabled FROM enrollments WHERE class_id=:class_id AND user_id=:user_id",
        {"class_id": teaching_class["class_id"], "user_id": student["user_id"] if student else ""},
    )
    if len(enrollments) != 1 or not enrollments[0]["enabled"]:
        reporter.fail("demo enrollment", "Student enrollment missing-or-disabled")
        good = False
    else:
        reporter.ok("demo enrollment", "%s enabled" % enrollments[0]["enrollment_id"])

    projects = _rows(
        connection,
        "SELECT project_id, course_id, title, description, instructions, starter_code, "
        "starter_code_sha256, training_profile, video_enabled, parameter_schema, "
        "evaluation_rubric, config_version, enabled, created_by_user_id "
        "FROM teaching_projects WHERE course_id=:course_id AND title IN "
        "(:short_title, :standard_title, :reach_title, :grasp_title, :lift_title, :lower_title)",
        {
            "course_id": course["course_id"],
            "short_title": names.short_project_title,
            "standard_title": names.standard_project_title,
            "reach_title": reach_project_title,
            "grasp_title": names.grasp_project_title,
            "lift_title": names.lift_project_title,
            "lower_title": names.lower_release_project_title,
        },
    )
    by_title = {row["title"]: row for row in projects}
    expected_projects = (
        (
            names.short_project_title, TrainingProfile.SHORT, False, 32,
            current_config_version(TrainingProfile.SHORT), "s1_push",
        ),
        (
            names.standard_project_title, TrainingProfile.STANDARD, True, 20000,
            4, "s1_push",
        ),
        (
            reach_project_title, TrainingProfile.STANDARD, True, 20000,
            4, "s1_reach",
        ),
        (
            names.grasp_project_title, TrainingProfile.STANDARD, True, 20000,
            4, "s1_grasp",
        ),
        (
            names.lift_project_title, TrainingProfile.STANDARD, True, 20000,
            4, "s1_lift",
        ),
        (
            names.lower_release_project_title, TrainingProfile.STANDARD, True, 20000,
            4, "s1_lower_release",
        ),
    )
    for title, profile, video, expected_steps, config_version, environment_type in expected_projects:
        row = by_title.get(title)
        valid = (
            administrator is not None
            and _project_facts_match(
                row,
                course_id=course["course_id"],
                title=title,
                profile=profile,
                video_enabled=video,
                expected_training_steps=expected_steps,
                config_version=config_version,
                created_by_user_id=administrator["user_id"],
                environment_type=environment_type,
            )
        )
        if not valid:
            reporter.fail("demo project", "%s missing-or-inconsistent" % title)
            good = False
        else:
            reporter.ok(
                "demo project",
                "%s profile=%s steps=%d video=%s version=%d enabled"
                % (
                    row["project_id"], profile.value, expected_steps,
                    str(video).lower(), config_version,
                ),
            )
    return good


def list_recovery_targets(connection: Any, reporter: Reporter) -> List[Dict[str, Any]]:
    rows = _rows(
        connection,
        "SELECT r.run_id, r.status, r.created_at, u.username, r.user_id, r.project_id, r.class_id "
        "FROM runs r LEFT JOIN users u ON u.user_id=r.user_id "
        "WHERE r.execution_mode='simulation' AND "
        "((r.status='queued' AND r.simulation_task_id IS NULL) OR "
        "(r.status='stopping' AND r.simulation_task_id IS NOT NULL)) "
        "ORDER BY r.created_at ASC, r.run_id ASC",
        {},
    )
    queued_count = sum(row["status"] == "queued" for row in rows)
    stopping_count = sum(row["status"] == "stopping" for row in rows)
    reporter.info("queued/unbound count", str(queued_count))
    reporter.info("stopping/bound count", str(stopping_count))
    for row in rows:
        owner = row["username"] or row["user_id"]
        reporter.info(
            "recovery target",
            "run_id=%s status=%s created_at=%s owner=%s project_id=%s class_id=%s"
            % (
                row["run_id"], row["status"], row["created_at"], owner,
                row["project_id"], row["class_id"] or "<none>",
            ),
        )
    if rows:
        reporter.fail("startup recovery gate", "operator decision required before Backend restart")
    else:
        reporter.ok("startup recovery gate", "no compensation targets")
    return rows


def run_database_checks(values: Mapping[str, str], namespace: str, reporter: Reporter) -> Optional[Any]:
    raw_url = values.get("DATABASE_URL", "")
    engine = None
    try:
        url, redacted = redact_database_url(raw_url)
        reporter.ok("DATABASE_URL", redacted)
        engine = create_engine(url, future=True, hide_parameters=True, pool_pre_ping=True)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            reporter.ok("PostgreSQL connection")
            revision = connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
            if revision == EXPECTED_REVISION:
                reporter.ok("Alembic current=head", EXPECTED_REVISION)
            else:
                reporter.fail("Alembic current=head", "expected %s" % EXPECTED_REVISION)
            check_database_schema(engine, reporter)
            check_demo_namespace(connection, namespace, reporter)
        return engine
    except Exception:
        if engine is not None:
            engine.dispose()
        reporter.fail("PostgreSQL preflight", "connection, migration, or read-only audit failed")
        return None


def _minimum_secret(values: Mapping[str, str], key: str, reporter: Reporter) -> bool:
    valid = len(values.get(key, "").encode("utf-8")) >= 32
    (reporter.ok if valid else reporter.fail)(key, "present" if valid else "missing-or-too-short")
    return valid


def check_token_mapping(values: Mapping[str, str], reporter: Reporter) -> bool:
    adapter = values.get("SIMULATION_SERVICE_TOKEN", "")
    callback = values.get("INTERNAL_RUN_EVENT_SERVICE_TOKEN", "")
    backend_callback = values.get("SIMULATION_BACKEND_SERVICE_TOKEN", "")
    adapter_match = len(adapter.encode("utf-8")) >= 32
    callback_match = len(callback.encode("utf-8")) >= 32 and callback == backend_callback
    (reporter.ok if adapter_match else reporter.fail)("Backend->Simulation token", "MATCH" if adapter_match else "MISMATCH")
    (reporter.ok if callback_match else reporter.fail)("Simulation->Backend token", "MATCH" if callback_match else "MISMATCH")
    return adapter_match and callback_match


def check_origin_mapping(values: Mapping[str, str], reporter: Reporter) -> bool:
    backend = values.get("BACKEND_ORIGIN", "")
    simulation = values.get("SIMULATION_BASE_URL", "")
    frontend_backend = values.get("VITE_API_PROXY_TARGET", "")
    simulation_backend = values.get("SIMULATION_BACKEND_BASE_URL", "")
    backend_valid = _is_origin_on_port(backend, 8000)
    simulation_valid = _is_origin_on_port(simulation, 8090)
    backend_match = backend_valid and frontend_backend == backend and simulation_backend == backend
    (reporter.ok if backend_match else reporter.fail)(
        "Backend origin mapping", "MATCH" if backend_match else "MISMATCH"
    )
    (reporter.ok if simulation_valid else reporter.fail)(
        "Simulation origin mapping", "MATCH" if simulation_valid else "MISMATCH"
    )
    return backend_match and simulation_valid


def check_artifacts(
    values: Mapping[str, str],
    reporter: Reporter,
    *,
    allowed_parent: Optional[Path] = None,
) -> bool:
    backend_raw = values.get("ARTIFACT_STORAGE_ROOT", "")
    simulation_raw = values.get("SIMULATION_ARTIFACTS_ROOT", "")
    try:
        backend = Path(backend_raw).expanduser().resolve()
        simulation = Path(simulation_raw).expanduser().resolve()
        if allowed_parent is None:
            configured_root = values.get("MIDTERM_PROJECT_ROOT", "")
            if not configured_root:
                raise ValueError("MIDTERM_PROJECT_ROOT is required")
            project_root = Path(configured_root).expanduser().resolve()
            if not project_root.is_absolute() or not project_root.is_dir():
                raise ValueError("MIDTERM_PROJECT_ROOT must be an existing absolute directory")
            allowed = (project_root / "var").resolve()
        else:
            allowed = allowed_parent.resolve()
        if not backend_raw or not simulation_raw or backend != simulation:
            raise ValueError("Backend and Simulation Artifact roots must MATCH")
        if backend == allowed or allowed not in backend.parents:
            raise ValueError("Artifact root must be a child of the approved var directory")
        backend.mkdir(mode=0o750, parents=True, exist_ok=True)
        if not backend.is_dir() or not os.access(backend, os.R_OK | os.W_OK | os.X_OK):
            raise ValueError("Artifact root needs same-user read/write/traverse permission")
        reporter.ok("Artifact root mapping", "MATCH")
        reporter.ok("Simulation Artifact access", "same-user writable/traversable")
        reporter.ok("Backend Artifact access", "same-user readable/traversable")
        return True
    except (OSError, ValueError) as exc:
        reporter.fail("Artifact root mapping", str(exc))
        return False


class RedisProbe:
    """Tiny read-only RESP client; it never issues a mutating Redis command."""

    def __init__(self, raw_url: str, timeout: float = 3.0) -> None:
        parsed = urlsplit(raw_url)
        if parsed.scheme not in {"redis", "rediss"} or not parsed.hostname:
            raise ValueError("Redis URL must use redis or rediss")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Redis URL credentials/query/fragment are not supported")
        database = parsed.path.lstrip("/")
        if not database.isdigit() or "/" in database:
            raise ValueError("Redis URL must select one numeric database")
        self.host = parsed.hostname
        self.port = parsed.port or 6379
        self.database = int(database)
        self.secure = parsed.scheme == "rediss"
        self.timeout = timeout

    def command(self, *parts: str) -> Any:
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        if self.secure:
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=self.host)
        try:
            stream = sock.makefile("rwb")
            if self.database:
                self._send(stream, "SELECT", str(self.database))
                self._read(stream)
            self._send(stream, *parts)
            return self._read(stream)
        finally:
            sock.close()

    @staticmethod
    def _send(stream: Any, *parts: str) -> None:
        payload = ["*%d\r\n" % len(parts)]
        for part in parts:
            encoded = part.encode("utf-8")
            payload.append("$%d\r\n" % len(encoded))
            payload.append(encoded)
            payload.append(b"\r\n")
        for item in payload:
            stream.write(item.encode("ascii") if isinstance(item, str) else item)
        stream.flush()

    @classmethod
    def _read(cls, stream: Any) -> Any:
        prefix = stream.read(1)
        line = stream.readline().rstrip(b"\r\n")
        if prefix == b"+":
            return line.decode("utf-8", "replace")
        if prefix == b"-":
            raise RuntimeError("Redis returned an error")
        if prefix == b":":
            return int(line)
        if prefix == b"$":
            length = int(line)
            if length < 0:
                return None
            value = stream.read(length)
            stream.read(2)
            return value
        if prefix == b"*":
            return [cls._read(stream) for _ in range(int(line))]
        raise RuntimeError("invalid Redis response")


def check_redis(values: Mapping[str, str], reporter: Reporter, *, before_run: bool = False) -> bool:
    namespace = values.get("SIMULATION_REDIS_NAMESPACE", "")
    queue = values.get("SIMULATION_RQ_QUEUE", "")
    if not SAFE_NAME.fullmatch(namespace) or not SAFE_NAME.fullmatch(queue):
        reporter.fail("Redis namespace/queue", "invalid safe name")
        return False
    if namespace == "simulation" or queue == "simulation-gpu":
        reporter.fail("Redis namespace/queue", "must not use shared defaults")
        return False
    try:
        probe = RedisProbe(values.get("SIMULATION_REDIS_URL", ""))
        if probe.command("PING") != "PONG":
            raise RuntimeError("PING failed")
        reporter.ok("Redis read-only PING", "namespace=%s queue=%s" % (namespace, queue))
        if not before_run:
            return True
        workers = probe.command("SMEMBERS", "rq:workers:%s" % queue) or []
        selected_workers = 0
        for worker_key in workers:
            key = worker_key.decode("utf-8", "replace") if isinstance(worker_key, bytes) else str(worker_key)
            queues = probe.command("HGET", key, "queues")
            if queues is not None:
                queue_text = queues.decode("utf-8", "replace") if isinstance(queues, bytes) else str(queues)
                if queue in queue_text.split(","):
                    selected_workers += 1
        if selected_workers == 1:
            reporter.ok("selected RQ workers", "1")
        else:
            reporter.fail("selected RQ workers", "expected exactly 1, found %d" % selected_workers)
        queued = int(probe.command("LLEN", "rq:queue:%s" % queue) or 0)
        active = 0
        for prefix in ("rq:wip:", "rq:deferred:", "rq:scheduled:"):
            try:
                active += int(probe.command("ZCARD", prefix + queue) or 0)
            except RuntimeError:
                pass
        if queued + active == 0:
            reporter.ok("selected RQ queue", "no queued/active old jobs")
        else:
            reporter.fail("selected RQ queue", "unexpected jobs=%d" % (queued + active))
        callback_ids = probe.command(
            "SMEMBERS", "%s:callback_outbox:index" % namespace
        ) or []
        undelivered = 0
        invalid = 0
        for event_id in callback_ids:
            event_text = (
                event_id.decode("utf-8", "replace")
                if isinstance(event_id, bytes) else str(event_id)
            )
            encoded = probe.command(
                "GET", "%s:callback_outbox:event:%s" % (namespace, event_text)
            )
            try:
                if isinstance(encoded, bytes):
                    encoded = encoded.decode("utf-8")
                record = json.loads(encoded)
                if record.get("delivery_status") != "delivered":
                    undelivered += 1
            except (TypeError, ValueError, UnicodeError):
                invalid += 1
        if undelivered == 0 and invalid == 0:
            reporter.ok("callback delivery gate", "no undelivered callbacks")
        else:
            reporter.fail(
                "callback delivery gate",
                "undelivered=%d invalid=%d; restore Backend and drain outbox"
                % (undelivered, invalid),
            )
        return (
            selected_workers == 1
            and queued + active == 0
            and undelivered == 0
            and invalid == 0
        )
    except Exception:
        reporter.fail("Redis read-only check", "unreachable or incompatible")
        return False


def _get_json(origin: str, path: str, *, token: Optional[str] = None) -> Tuple[int, Any]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = Request(origin.rstrip("/") + path, headers=headers, method="GET")
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(request, timeout=4) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except Exception:
            payload = None
        return exc.code, payload


def _reachable(origin: str) -> bool:
    try:
        request = Request(origin, method="GET")
        with build_opener(ProxyHandler({})).open(request, timeout=3) as response:
            return 200 <= response.status < 500
    except HTTPError as exc:
        return exc.code < 500
    except (OSError, URLError, ValueError):
        return False


def _is_origin_on_port(value: str, port: int) -> bool:
    try:
        parsed = urlsplit(value)
        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and parsed.port == port
            and parsed.username is None
            and parsed.password is None
            and parsed.path in {"", "/"}
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        return False


def check_before_run(values: Mapping[str, str], reporter: Reporter) -> None:
    backend = values.get("BACKEND_ORIGIN", "")
    frontend = values.get("FRONTEND_ORIGIN", "")
    simulation = values.get("SIMULATION_BASE_URL", "")
    try:
        if not _is_origin_on_port(backend, 8000):
            raise ValueError
        ready_status, ready = _get_json(backend, "/api/ready")
        openapi_status, openapi = _get_json(backend, "/openapi.json")
        canonical = (
            ready_status == 200 and isinstance(ready, dict) and ready.get("database_ready") is True
            and ready.get("simulation_configured") is True
            and ready.get("artifact_download_configured") is True
            and openapi_status == 200 and isinstance(openapi, dict)
            and "/api/v1/runs" in openapi.get("paths", {})
        )
        (reporter.ok if canonical else reporter.fail)(
            "Backend canonical API", "8000 ready" if canonical else "8000 readiness/OpenAPI failed"
        )
    except Exception:
        reporter.fail("Backend canonical API", "8000 unreachable")

    frontend_proxy_ready = False
    try:
        proxy_status, proxy_payload = _get_json(frontend, "/api/ready")
        frontend_proxy_ready = (
            proxy_status == 200 and isinstance(proxy_payload, dict)
            and proxy_payload.get("database_ready") is True
        )
    except Exception:
        frontend_proxy_ready = False
    if (
        frontend == "http://127.0.0.1:5173"
        and _reachable(frontend)
        and frontend_proxy_ready
    ):
        reporter.ok("Frontend formal port", "5173 reachable; /api proxy reaches Backend")
    else:
        reporter.fail(
            "Frontend formal port",
            "5173 and its /api proxy to Backend 8000 must be reachable",
        )
    reporter.info("Frontend 5174", "not a formal runtime port")

    try:
        if not _is_origin_on_port(simulation, 8090):
            raise ValueError
        probe_id = "midterm-preflight-%s" % uuid4().hex
        status, payload = _get_json(
            simulation,
            "/simulation/tasks/%s" % probe_id,
            token=values.get("SIMULATION_SERVICE_TOKEN", ""),
        )
        valid = status == 404 and isinstance(payload, dict) and payload.get("code") == "task_not_found"
        (reporter.ok if valid else reporter.fail)(
            "Simulation protected GET", "token accepted; no task created" if valid else "8090/token check failed"
        )
    except Exception:
        reporter.fail("Simulation protected GET", "8090 unreachable")
    reporter.info(
        "Project video contract",
        "Short=false; Standard/Reach=true (validated by database gate)",
    )


def _media_origin(value: str, scheme: str, port: int, *, loopback: bool) -> Optional[str]:
    try:
        parsed = urlsplit(value)
        is_loopback = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        if not (
            parsed.scheme == scheme and parsed.hostname and parsed.port == port
            and parsed.username is None and parsed.password is None
            and parsed.path in {"", "/"} and not parsed.query and not parsed.fragment
            and is_loopback == loopback
        ):
            return None
        return parsed.hostname
    except ValueError:
        return None


def _tcp_listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def _udp_listening(port: int) -> bool:
    expected = "%04X" % port
    for table in (Path("/proc/net/udp"), Path("/proc/net/udp6")):
        try:
            lines = table.read_text(encoding="ascii").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            columns = line.split()
            if len(columns) <= 3:
                continue
            address, actual_port = columns[1].rsplit(":", 1)
            if actual_port.upper() == expected and set(address) <= {"0"}:
                return True
    return False


def _count_matching_processes(
    executable: Path,
    *,
    proc_root: Path = Path("/proc"),
) -> int:
    expected = executable.resolve()
    count = 0
    for item in proc_root.iterdir():
        if not item.name.isdigit():
            continue
        try:
            stat_line = (item / "stat").read_text(encoding="ascii")
            closing_paren = stat_line.rfind(")")
            if closing_paren < 0:
                continue
            fields = stat_line[closing_paren + 1:].split()
            if not fields or fields[0].upper() in {"Z", "X"}:
                continue
            executable_matches = (item / "exe").resolve() == expected
            command_matches = (item / "comm").read_text(encoding="ascii").strip() == "mediamtx"
            if executable_matches or command_matches:
                count += 1
        except (OSError, RuntimeError):
            continue
    return count


def check_before_video(values: Mapping[str, str], reporter: Reporter) -> None:
    rtsp_base = values.get("SIMULATION_STREAM_RTSP_BASE", "")
    public_base = values.get("SIMULATION_STREAM_PUBLIC_BASE_URL", "")
    control = values.get("MEDIAMTX_CONTROL_URL", "")
    ice_host = values.get("MEDIAMTX_ICE_ADDITIONAL_HOST", "")
    allowed_origin = values.get("MEDIAMTX_WEBRTC_ALLOWED_ORIGIN", "")
    rtsp_host = _media_origin(rtsp_base, "rtsp", 18554, loopback=True)
    public_host = _media_origin(public_base, "http", 18889, loopback=False)
    control_host = _media_origin(control, "http", 19997, loopback=True)
    if rtsp_host:
        reporter.ok("internal RTSP base", "loopback:18554 MATCH")
    else:
        reporter.fail("internal RTSP base", "MISMATCH")
    if public_host and public_host == ice_host:
        reporter.ok("public WHEP base", "host=%s port=18889 MATCH" % public_host)
    else:
        reporter.fail("public WHEP base", "host/ICE additional host MISMATCH")
    if control_host:
        reporter.ok("MediaMTX control URL", "loopback:19997 MATCH")
    else:
        reporter.fail("MediaMTX control URL", "MISMATCH")
    try:
        ice_port = int(values.get("MEDIAMTX_ICE_UDP_PORT", ""))
    except ValueError:
        ice_port = 0
    config_path = Path(values.get("MEDIAMTX_CONFIG_PATH", "")).expanduser()
    binary_path = Path(values.get("MEDIAMTX_BINARY_PATH", "")).expanduser()
    ffmpeg_path = Path(values.get("SIMULATION_FFMPEG_PATH", "")).expanduser()
    ffprobe_path = ffmpeg_path.with_name("ffprobe")
    for label, path, executable in (
        ("MediaMTX config file", config_path, False),
        ("MediaMTX executable", binary_path, True),
        ("FFmpeg executable", ffmpeg_path, True),
        ("ffprobe executable", ffprobe_path, True),
    ):
        present = path.is_file()
        usable = present and (not executable or os.access(str(path), os.X_OK))
        (reporter.ok if usable else reporter.fail)(
            label,
            "present" if usable else ("not executable" if present else "missing"),
        )

    actual = None
    try:
        status, payload = _get_json(control, "/v3/config/global/get")
        if status == 200 and isinstance(payload, dict):
            actual = payload
            reporter.ok("MediaMTX control API", "ready")
        else:
            reporter.fail("MediaMTX control API", "not ready")
    except Exception:
        reporter.fail("MediaMTX control API", "unreachable")
    if actual is not None:
        checks = (
            ("MediaMTX RTSP config", actual.get("rtsp") is True and actual.get("rtspAddress") == "127.0.0.1:18554" and actual.get("rtspTransports") == ["tcp"]),
            ("MediaMTX WHEP config", actual.get("webrtc") is True and actual.get("webrtcAddress") == ":18889"),
            ("MediaMTX ICE config", actual.get("webrtcLocalUDPAddress") == ":18189" and actual.get("webrtcIPsFromInterfaces") is False and actual.get("webrtcAdditionalHosts") == [ice_host]),
            (
                "MediaMTX CORS config",
                set(actual.get("webrtcAllowOrigins") or [])
                == {
                    allowed_origin,
                    "http://192.0.2.10:4173",
                    "http://127.0.0.1:4173",
                    "http://localhost:4173",
                },
            ),
            ("MediaMTX service scope", all(actual.get(name) is False for name in ("rtmp", "hls", "srt", "moq", "metrics", "pprof", "playback"))),
        )
        for label, matched in checks:
            (reporter.ok if matched else reporter.fail)(label, "MATCH" if matched else "MISMATCH")

    for label, host, port in (
        ("RTSP TCP listener", rtsp_host or "127.0.0.1", 18554),
        ("WHEP TCP listener", public_host or "127.0.0.1", 18889),
    ):
        matched = _tcp_listening(host, port)
        (reporter.ok if matched else reporter.fail)(label, "LISTEN" if matched else "NOT LISTENING")
    udp_ok = ice_port == 18189 and _udp_listening(ice_port)
    (reporter.ok if udp_ok else reporter.fail)("ICE UDP listener", "18189 LISTEN" if udp_ok else "MISMATCH")
    process_count = _count_matching_processes(binary_path) if binary_path.is_file() else 0
    (reporter.ok if process_count == 1 else reporter.fail)(
        "MediaMTX process count", "1" if process_count == 1 else "expected 1, found %d" % process_count
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("database", "before-backend", "before-run", "before-video"), required=True)
    parser.add_argument("--namespace", default="midterm-demo")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--database-config", type=Path)
    parser.add_argument("--database-host-override")
    return parser


def run(
    args: argparse.Namespace,
    *,
    output: TextIO = sys.stdout,
    allowed_artifact_parent: Optional[Path] = None,
) -> int:
    reporter = Reporter(output)
    values = check_config(args.config.expanduser(), reporter)
    if values is None:
        return 1
    try:
        demo_names(args.namespace)
    except ValueError as exc:
        reporter.fail("demo namespace", str(exc))
        return 1
    if args.stage == "before-video":
        database_values = database_values_for_audit(
            values, args.database_config, args.database_host_override, reporter
        )
        engine = (
            run_database_checks(database_values, args.namespace, reporter)
            if database_values is not None else None
        )
        configured_namespace = values.get("SIMULATION_REDIS_NAMESPACE", "")
        (reporter.ok if configured_namespace == args.namespace else reporter.fail)(
            "namespace mapping", "MATCH" if configured_namespace == args.namespace else "MISMATCH"
        )
        check_redis(values, reporter, before_run=True)
        check_before_video(values, reporter)
        if engine is not None:
            engine.dispose()
        if reporter.failures:
            reporter.info(
                "remediation",
                "fix the named media/listener/single-process/Worker gate and rerun; do not create a Run",
            )
        reporter.info("preflight result", "PASS" if reporter.failures == 0 else "FAIL")
        return 0 if reporter.failures == 0 else 1
    database_values = database_values_for_audit(
        values,
        args.database_config,
        args.database_host_override,
        reporter,
    )
    if database_values is None:
        return 1
    engine = run_database_checks(database_values, args.namespace, reporter)
    if args.stage in {"before-backend", "before-run"}:
        configured_namespace = values.get("SIMULATION_REDIS_NAMESPACE", "")
        if configured_namespace == args.namespace:
            reporter.ok("namespace mapping", "MATCH")
        else:
            reporter.fail("namespace mapping", "MISMATCH")
        _minimum_secret(values, "USER_JWT_SECRET", reporter)
        _minimum_secret(values, "SIMULATION_SERVICE_TOKEN", reporter)
        _minimum_secret(values, "INTERNAL_RUN_EVENT_SERVICE_TOKEN", reporter)
        check_token_mapping(values, reporter)
        check_origin_mapping(values, reporter)
        check_redis(values, reporter, before_run=args.stage == "before-run")
        check_artifacts(values, reporter, allowed_parent=allowed_artifact_parent)
        if engine is not None:
            try:
                with engine.connect() as connection:
                    list_recovery_targets(connection, reporter)
            except Exception:
                reporter.fail("startup recovery gate", "read-only query failed")
    if args.stage == "before-run":
        check_before_run(values, reporter)
    if engine is not None:
        engine.dispose()
    if reporter.failures:
        if args.stage == "database":
            reporter.info(
                "remediation",
                "verify the private config, PostgreSQL reachability, and reviewed Alembic head; then rerun",
            )
        elif args.stage == "before-backend":
            reporter.info(
                "remediation",
                "fix the named config/service gate or decide each listed Run; never clear PostgreSQL or Redis",
            )
        else:
            reporter.info(
                "remediation",
                "start/fix the named 8000, 5173, 8090, or single-Worker gate and rerun; do not create a Run",
            )
    reporter.info("preflight result", "PASS" if reporter.failures == 0 else "FAIL")
    return 0 if reporter.failures == 0 else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
