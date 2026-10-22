import os
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from alembic import command as alembic_command
from alembic.config import Config
from sqlalchemy import delete, func, select
from sqlalchemy.engine import URL

from apps.api.database import create_postgres_engine, create_session_factory
from apps.api.main import create_app
from tests.auth import StaticCurrentUserProvider
from robotics.runs import (
    IdempotencyConflict,
    IdempotentRunCreateCommand,
    IdempotentRunCreateResult,
    InMemoryRunRepository,
    PostgreSQLIdempotentRunCreationService,
    RunService,
    fingerprint_run_create,
)
from robotics.projects.models import (
    TrainingProfile,
    evaluation_rubric,
    parameter_schema,
    starter_code_digest,
    success_condition,
    worker_config,
)
from robotics.runs.db_models import (
    CourseRow,
    EnrollmentRow,
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


ROOT = Path(__file__).resolve().parents[1]
TEST_ENV_PATH = Path(
    "/tmp/embodied-arm-studio-postgres-test.env"
)


class MemoryIdempotentRunCreationService:
    """Explicit test-only implementation; production never selects this."""

    def __init__(self, repository):
        self._service = RunService(repository)
        self._records = {}
        self._lock = threading.RLock()

    def create_run(self, command):
        if not isinstance(command, IdempotentRunCreateCommand):
            raise TypeError("command must be an IdempotentRunCreateCommand")
        request_hash = fingerprint_run_create(command)
        scope = (command.user_id, command.idempotency_key)
        with self._lock:
            existing = self._records.get(scope)
            if existing is not None:
                existing_hash, run_id = existing
                if existing_hash != request_hash:
                    raise IdempotencyConflict(*scope)
                return IdempotentRunCreateResult(
                    run=self._service.get_run(run_id), replayed=True
                )
            seed = command.parameters["seed"]
            config = worker_config(TrainingProfile.SHORT, False, seed)
            run = self._service.create_run(
                user_id=command.user_id,
                project_id=command.project_id,
                class_id=command.class_id,
                execution_mode=command.execution_mode,
                run_kind=command.run_kind,
                parameters={"seed": seed},
                project_config_snapshot={
                    "schema_version": "1.0",
                    "project_id": command.project_id,
                    "course_id": "course-1",
                    "class_id": command.class_id,
                    "title": "Project",
                    "description": "",
                    "instructions": "",
                    "task_type": "training",
                    "environment_type": "s1_push",
                    "algorithm": "PPO",
                    "success_condition": success_condition(),
                    "training_profile": "short",
                    "expected_training_steps": 32,
                    "video_enabled": False,
                    "parameter_schema": parameter_schema(),
                    "resolved_parameters": {"seed": seed},
                    "evaluation_rubric": evaluation_rubric(TrainingProfile.SHORT),
                    "config_version": command.project_config_version,
                    "enabled": True,
                    "project_enabled_at_creation": True,
                    "starter_code_sha256": "0" * 64,
                    "training_config": config,
                },
                code=command.code,
            )
            self._records[scope] = (request_hash, run.run_id)
            return IdempotentRunCreateResult(run=run, replayed=False)


class FakeSimulationClient:
    def __init__(self):
        self.calls = []
        self.closed = False

    def create_task(self, run_id, project_id, config):
        self.calls.append((run_id, project_id, config))
        now = datetime(2026, 8, 27, 10, 0, tzinfo=timezone.utc)
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
        self.closed = True

    def stop_task(self, task_id):
        raise AssertionError("Run creation tests must not stop simulation tasks")


class V1RunCreateApiMemoryTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = InMemoryRunRepository()
        self.creation_service = MemoryIdempotentRunCreationService(
            self.repository
        )
        self.app = self.make_app("trusted-user-1", "primary")
        self.client = ASGITestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.temp_dir.cleanup()

    def make_app(self, user_id, suffix):
        root = Path(self.temp_dir.name)
        simulation_client = FakeSimulationClient()
        return create_app(
            root / (suffix + "-runs"),
            tick_seconds=0,
            project_root=root / (suffix + "-projects"),
            run_repository=self.repository,
            run_creation_service=self.creation_service,
            current_user_provider=StaticCurrentUserProvider(user_id),
            simulation_client=simulation_client,
        )

    @staticmethod
    def payload(**changes):
        values = {
            "project_id": "project-1",
            "class_id": "class-1",
            "project_config_version": 3,
            "execution_mode": "simulation",
            "run_kind": "train",
            "parameters": {"seed": 17},
            "code": "print('queued only')\n",
        }
        values.update(changes)
        return values

    def create(self, key="create-1", payload=None, **kwargs):
        headers = dict(kwargs.pop("headers", {}))
        headers["Idempotency-Key"] = key
        return self.client.post(
            "/api/v1/runs",
            json=payload or self.payload(),
            headers=headers,
            **kwargs
        )

    def test_first_create_returns_queued_bound_canonical_run_without_events(self):
        response = self.create(headers={"X-User-Id": "forged-user"})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.headers["Idempotency-Replayed"], "false")
        body = response.json()
        self.assertEqual(body["user_id"], "trusted-user-1")
        self.assertEqual(body["project_id"], "project-1")
        self.assertEqual(body["status"], "queued")
        self.assertEqual(body["execution_mode"], "simulation")
        self.assertEqual(body["run_kind"], "train")
        self.assertEqual(body["class_id"], "class-1")
        self.assertEqual(body["parameters"], {"seed": 17})
        self.assertEqual(body["project_config_snapshot"]["config_version"], 3)
        self.assertEqual(body["simulation_task_id"], "task-" + body["run_id"])
        self.assertEqual(body["progress"], 0.0)
        self.assertEqual(body["last_sequence"], 0)
        self.assertEqual(
            self.client.get(
                "/api/v1/runs/%s/events" % body["run_id"]
            ).json(),
            [],
        )

    def test_key_validation_and_payload_validation_create_nothing(self):
        missing = self.client.post("/api/v1/runs", json=self.payload())
        blank = self.create(key="   ")
        overlong = self.create(key="x" * 256)
        invalid = self.create(key="invalid", payload=self.payload(project_id=""))
        forged_body = self.create(
            key="forged", payload=dict(self.payload(), user_id="forged")
        )
        snapshot = self.create(
            key="snapshot",
            payload=dict(self.payload(), project_config_snapshot={}),
        )
        bool_seed = self.create(
            key="bool-seed",
            payload=self.payload(parameters={"seed": True}),
        )
        worker_override = self.create(
            key="worker-override",
            payload=self.payload(parameters={"seed": 1, "batch_size": 1}),
        )
        self.assertEqual(missing.status_code, 422)
        self.assertEqual(blank.status_code, 422)
        self.assertEqual(overlong.status_code, 422)
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(forged_body.status_code, 422)
        self.assertEqual(snapshot.status_code, 422)
        self.assertEqual(bool_seed.status_code, 422)
        self.assertEqual(worker_override.status_code, 422)
        self.assertEqual(self.repository.list_runs(), [])

    def test_seed_boundaries_and_every_forbidden_formal_field(self):
        minimum = self.create(
            key="seed-min", payload=self.payload(parameters={"seed": 0})
        )
        maximum = self.create(
            key="seed-max",
            payload=self.payload(parameters={"seed": 2147483647}),
        )
        self.assertEqual((minimum.status_code, maximum.status_code), (202, 202))
        invalid_seeds = (-1, 2147483648, True, 1.5, "7")
        for index, seed in enumerate(invalid_seeds):
            with self.subTest(seed=seed):
                response = self.create(
                    key="bad-seed-%d" % index,
                    payload=self.payload(parameters={"seed": seed}),
                )
                self.assertEqual(response.status_code, 422)
        forbidden_top_level = (
            "project_config_snapshot", "max_steps", "action_scale",
            "environment_id", "strategy_id", "num_envs", "training_config",
            "rubric", "task_type", "environment_type", "algorithm",
        )
        for index, name in enumerate(forbidden_top_level):
            with self.subTest(field=name):
                response = self.create(
                    key="forbidden-%d" % index,
                    payload=dict(self.payload(), **{name: {}}),
                )
                self.assertEqual(response.status_code, 422)
        self.assertEqual(len(self.repository.list_runs()), 2)

    def test_replay_conflict_list_and_detail(self):
        first = self.create(key=" stable-key ")
        replay = self.create(key="stable-key")
        conflict = self.create(
            key="stable-key",
            payload=self.payload(parameters={"seed": 99}),
        )
        self.assertEqual(first.status_code, 202)
        self.assertEqual(replay.status_code, 202)
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(first.json()["run_id"], replay.json()["run_id"])
        self.assertEqual(replay.headers["Idempotency-Replayed"], "true")
        self.assertEqual(len(self.repository.list_runs()), 1)
        run_id = first.json()["run_id"]
        self.assertEqual(
            self.client.get("/api/v1/runs/%s" % run_id).json(), first.json()
        )
        self.assertEqual(
            [run["run_id"] for run in self.client.get("/api/v1/runs").json()],
            [run_id],
        )

    def test_different_trusted_users_can_reuse_key(self):
        second_app = self.make_app("trusted-user-2", "second")
        with ASGITestClient(second_app) as second:
            first = self.create(key="shared-key")
            other = second.post(
                "/api/v1/runs",
                json=self.payload(),
                headers={"Idempotency-Key": "shared-key"},
            )
        self.assertEqual(first.status_code, 202)
        self.assertEqual(other.status_code, 202)
        self.assertNotEqual(first.json()["run_id"], other.json()["run_id"])
        self.assertEqual(other.json()["user_id"], "trusted-user-2")
        self.assertEqual(len(self.repository.list_runs()), 2)


def _read_test_environment():
    values = {}
    if TEST_ENV_PATH.is_file():
        for raw_line in TEST_ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            values[name.strip()] = value.strip().strip("\"'")
    return values


def _safe_test_database_url():
    raw_url = os.environ.get("TEST_DATABASE_URL")
    if raw_url:
        from apps.api.database import resolve_database_url

        resolved = resolve_database_url(raw_url)
    else:
        values = _read_test_environment()
        database = values.get("POSTGRES_DB")
        username = values.get("POSTGRES_USER")
        password = values.get("POSTGRES_PASSWORD")
        if not all((database, username, password)):
            return None
        resolved = URL.create(
            "postgresql+psycopg",
            username=username,
            password=password,
            host=values.get("POSTGRES_HOST", "127.0.0.1"),
            port=int(values.get("POSTGRES_PORT", "5433")),
            database=database,
        )
    if resolved.database is None or "test" not in resolved.database.lower():
        raise RuntimeError("PostgreSQL API tests require a database name containing test")
    return resolved


SAFE_TEST_DATABASE_URL = _safe_test_database_url()


@unittest.skipUnless(
    SAFE_TEST_DATABASE_URL is not None,
    "PostgreSQL test configuration is unavailable",
)
class V1RunCreateApiPostgreSQLTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cleanup_engine = create_postgres_engine(SAFE_TEST_DATABASE_URL)
        cls.factory = create_session_factory(cls.cleanup_engine)
        config = Config(str(ROOT / "alembic.ini"))
        with patch.dict(
            os.environ,
            {"DATABASE_URL": SAFE_TEST_DATABASE_URL.render_as_string(False)},
            clear=False,
        ):
            alembic_command.upgrade(config, "head")

    @classmethod
    def tearDownClass(cls):
        cls.cleanup_engine.dispose()

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.prefix = "p742-legacy-api-%s-" % uuid4().hex
        self.clients = []
        self.user_1 = self.prefix + "user-1"
        self.user_2 = self.prefix + "user-2"
        self.organization_id = self.prefix + "org"
        self.course_id = self.prefix + "course"
        self.class_id = self.prefix + "class"
        self.project_id = self.prefix + "project"
        now = datetime.now(timezone.utc)
        with self.factory() as session:
            session.add_all([
                UserRow(
                    user_id=user_id, username=user_id, display_name=user_id,
                    password_hash="test-only", enabled=True,
                    created_at=now, updated_at=now,
                )
                for user_id in (self.user_1, self.user_2)
            ])
            session.flush()
            session.add(OrganizationRow(
                organization_id=self.organization_id, slug=self.organization_id,
                name="Test", enabled=True, created_at=now, updated_at=now,
            ))
            session.flush()
            session.add_all([
                OrganizationMembershipRow(
                    membership_id=user_id + "-membership",
                    organization_id=self.organization_id, user_id=user_id,
                    role="student", enabled=True, created_at=now, updated_at=now,
                )
                for user_id in (self.user_1, self.user_2)
            ])
            session.add(CourseRow(
                course_id=self.course_id, organization_id=self.organization_id,
                code=self.prefix + "code", title="Course", description="",
                owner_instructor_user_id=self.user_1, enabled=True,
                created_at=now, updated_at=now,
            ))
            session.flush()
            session.add(TeachingClassRow(
                class_id=self.class_id, course_id=self.course_id, code="A",
                name="A", term=self.prefix, instructor_user_id=self.user_1,
                enabled=True, created_at=now, updated_at=now,
            ))
            session.flush()
            session.add_all([
                EnrollmentRow(
                    enrollment_id=user_id + "-enrollment", class_id=self.class_id,
                    user_id=user_id, enabled=True, created_at=now, updated_at=now,
                )
                for user_id in (self.user_1, self.user_2)
            ])
            session.add(TeachingProjectRow(
                project_id=self.project_id, course_id=self.course_id,
                title=self.prefix + "project", description="", instructions="",
                starter_code="", starter_code_sha256=starter_code_digest(""),
                training_profile="short", video_enabled=False,
                parameter_schema=parameter_schema(),
                evaluation_rubric=evaluation_rubric(TrainingProfile.SHORT),
                config_version=1, enabled=True, created_by_user_id=self.user_1,
                created_at=now, updated_at=now,
            ))
            session.commit()

    def tearDown(self):
        for client in reversed(self.clients):
            client.close()
        with self.factory() as session:
            run_ids = select(RunRow.run_id).where(
                RunRow.user_id.like(self.prefix + "%")
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
                delete(RunRow).where(RunRow.user_id.like(self.prefix + "%"))
            )
            session.execute(delete(TeachingProjectRow).where(TeachingProjectRow.project_id == self.project_id))
            session.execute(delete(EnrollmentRow).where(EnrollmentRow.class_id == self.class_id))
            session.execute(delete(TeachingClassRow).where(TeachingClassRow.class_id == self.class_id))
            session.execute(delete(CourseRow).where(CourseRow.course_id == self.course_id))
            session.execute(delete(OrganizationMembershipRow).where(OrganizationMembershipRow.organization_id == self.organization_id))
            session.execute(delete(OrganizationRow).where(OrganizationRow.organization_id == self.organization_id))
            session.execute(delete(UserRow).where(UserRow.user_id.like(self.prefix + "%")))
            session.commit()
        self.temp_dir.cleanup()

    def app_client(self, user_id, suffix, simulation_client=None):
        root = Path(self.temp_dir.name)
        if simulation_client is None:
            simulation_client = FakeSimulationClient()
        app = create_app(
            root / (suffix + "-runs"),
            tick_seconds=0,
            project_root=root / (suffix + "-projects"),
            run_repository_mode="postgresql",
            database_url=SAFE_TEST_DATABASE_URL,
            current_user_provider=StaticCurrentUserProvider(user_id),
            simulation_client=simulation_client,
        )
        client = ASGITestClient(app)
        self.clients.append(client)
        return app, client

    def payload(self, **changes):
        values = {
            "project_id": self.project_id,
            "class_id": self.class_id,
            "project_config_version": 1,
            "execution_mode": "simulation",
            "run_kind": "train",
            "parameters": {"seed": 31},
            "code": "print('postgres queued')\n",
        }
        values.update(changes)
        return values

    def counts(self):
        with self.factory() as session:
            runs = session.scalar(
                select(func.count())
                .select_from(RunRow)
                .where(RunRow.user_id.like(self.prefix + "%"))
            )
            keys = session.scalar(
                select(func.count())
                .select_from(RunIdempotencyKeyRow)
                .where(RunIdempotencyKeyRow.user_id.like(self.prefix + "%"))
            )
            events = session.scalar(
                select(func.count())
                .select_from(RunEventRow)
                .join(RunRow)
                .where(RunRow.user_id.like(self.prefix + "%"))
            )
            return runs, keys, events

    def test_real_service_replay_app_rebuild_and_no_duplicates(self):
        user_id = self.user_1
        first_app, first_client = self.app_client(user_id, "first")
        self.assertIsInstance(
            first_app.state.run_creation_service,
            PostgreSQLIdempotentRunCreationService,
        )
        first = first_client.post(
            "/api/v1/runs",
            json=self.payload(),
            headers={"Idempotency-Key": "rebuild-key"},
        )
        second_app, second_client = self.app_client(user_id, "second")
        replay = second_client.post(
            "/api/v1/runs",
            json=self.payload(),
            headers={"Idempotency-Key": "rebuild-key"},
        )
        self.assertIsNot(
            first_app.state.run_creation_service,
            second_app.state.run_creation_service,
        )
        self.assertEqual(first.status_code, 202)
        self.assertEqual(replay.status_code, 202)
        self.assertEqual(first.json()["status"], "queued")
        self.assertEqual(first.json()["run_id"], replay.json()["run_id"])
        self.assertEqual(first.headers["Idempotency-Replayed"], "false")
        self.assertEqual(replay.headers["Idempotency-Replayed"], "true")
        self.assertEqual(self.counts(), (1, 1, 0))
        run_id = first.json()["run_id"]
        self.assertEqual(
            second_client.get("/api/v1/runs/%s" % run_id).json(),
            replay.json(),
        )
        self.assertIn(
            run_id,
            [run["run_id"] for run in second_client.get("/api/v1/runs").json()],
        )

    def test_conflict_different_user_and_invalid_payload(self):
        _, first_client = self.app_client(self.user_1, "first")
        _, second_client = self.app_client(self.user_2, "second")
        headers = {"Idempotency-Key": "shared-key"}
        first = first_client.post(
            "/api/v1/runs", json=self.payload(), headers=headers
        )
        conflict = first_client.post(
            "/api/v1/runs",
            json=self.payload(parameters={"seed": 32}),
            headers=headers,
        )
        other_user = second_client.post(
            "/api/v1/runs", json=self.payload(), headers=headers
        )
        invalid = first_client.post(
            "/api/v1/runs",
            json=self.payload(project_id=""),
            headers={"Idempotency-Key": "invalid"},
        )
        self.assertEqual(first.status_code, 202)
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(other_user.status_code, 202)
        self.assertEqual(invalid.status_code, 422)
        self.assertNotEqual(first.json()["run_id"], other_user.json()["run_id"])
        self.assertEqual(self.counts(), (2, 2, 0))

    def test_simulation_dispatch_observes_committed_run_and_idempotency_record(self):
        owner = self

        class CommitCheckingClient(FakeSimulationClient):
            def create_task(self, run_id, project_id, config):
                with owner.factory() as session:
                    row = session.get(RunRow, run_id)
                    key_count = session.scalar(
                        select(func.count())
                        .select_from(RunIdempotencyKeyRow)
                        .where(RunIdempotencyKeyRow.run_id == run_id)
                    )
                    owner.assertIsNotNone(row)
                    owner.assertIsNone(row.simulation_task_id)
                    owner.assertEqual(key_count, 1)
                return super().create_task(run_id, project_id, config)

        simulation = CommitCheckingClient()
        _, client = self.app_client(
            self.user_1, "simulation", simulation
        )
        response = client.post(
            "/api/v1/runs",
            json=self.payload(parameters={"seed": 7}),
            headers={"Idempotency-Key": "simulation-commit-order"},
        )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(
            response.json()["simulation_task_id"],
            "task-" + response.json()["run_id"],
        )
        self.assertEqual(len(simulation.calls), 1)
        self.assertEqual(self.counts(), (1, 1, 0))

    def test_scene_ref_freezes_document_and_rejects_stale_revision(self):
        _, client = self.app_client(self.user_1, "scene-binding")
        scene = client.get("/api/scenes/scene_mock_reach_lab")
        self.assertEqual(scene.status_code, 200)
        scene_document = scene.json()
        scene_ref = {
            "scene_id": scene_document["scene_id"],
            "revision": scene_document["revision"],
        }

        created = client.post(
            "/api/v1/runs",
            json=self.payload(scene_ref=scene_ref),
            headers={"Idempotency-Key": "scene-binding"},
        )
        self.assertEqual(created.status_code, 202)
        snapshot_scene = created.json()["project_config_snapshot"]["scene"]
        self.assertEqual(snapshot_scene["scene_ref"], scene_ref)
        self.assertEqual(snapshot_scene["document"], scene_document)

        stale = client.post(
            "/api/v1/runs",
            json=self.payload(
                scene_ref={
                    "scene_id": scene_ref["scene_id"],
                    "revision": scene_ref["revision"] + 1,
                }
            ),
            headers={"Idempotency-Key": "scene-binding-stale"},
        )
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(
            stale.json()["detail"]["code"], "scene_revision_conflict"
        )


if __name__ == "__main__":
    unittest.main()
