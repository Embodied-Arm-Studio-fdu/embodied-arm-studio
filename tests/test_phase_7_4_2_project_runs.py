import hashlib
import asyncio
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from alembic import command as alembic_command
from alembic.config import Config
from sqlalchemy import delete, func, inspect, select, text, update

from apps.api.database import create_postgres_engine, create_session_factory
from apps.api.main import create_app
from robotics.projects.models import (
    TrainingProfile,
    evaluation_rubric,
    parameter_schema,
    starter_code_digest,
    worker_config,
)
from robotics.runs.db_models import (
    CourseRow,
    EnrollmentRow,
    EvaluationReportRow,
    OrganizationMembershipRow,
    OrganizationRow,
    RunArtifactRow,
    RunEventRow,
    RunIdempotencyKeyRow,
    RunRow,
    TeachingClassRow,
    TeachingProjectRow,
    UserRow,
)
from robotics.runs import (
    ExecutionMode,
    IdempotentRunCreateCommand,
    PostgreSQLProjectRunCreationService,
    RunKind,
    RunStatus,
)
from robotics.simulation import (
    SimulationTaskResponse,
    SimulationTimeoutError,
    InputEventAck,
    StreamDescriptor,
    StreamSessionStatus,
)
from tests.auth import StaticCurrentUserProvider
from tests.http_client import ASGITestClient
from tests.test_v1_run_websocket import ASGIWebSocketDriver
from tests.test_projects import SAFE_POSTGRES_URL


ROOT = Path(__file__).resolve().parents[1]


class RecordingSimulationClient:
    def __init__(self, factory=None):
        self.calls = []
        self.error = None
        self.closed = False
        self._factory = factory
        self._lock = threading.Lock()

    def create_task(self, run_id, project_id, config):
        if self._factory is not None:
            with self._factory() as session:
                run = session.get(RunRow, run_id)
                keys = session.scalar(
                    select(func.count())
                    .select_from(RunIdempotencyKeyRow)
                    .where(RunIdempotencyKeyRow.run_id == run_id)
                )
                if run is None or keys != 1 or run.simulation_task_id is not None:
                    raise AssertionError("Simulation dispatch preceded database commit")
        with self._lock:
            self.calls.append((run_id, project_id, deepcopy(config)))
        if self.error is not None:
            raise self.error
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

    def get_task_catalog(self):
        from robotics.simulation.contracts import SimulationTaskCatalog
        from simulation_core.task_registry import PRODUCTION_TASK_REGISTRY
        return SimulationTaskCatalog(tasks=[definition.catalog_entry()
                                            for definition in PRODUCTION_TASK_REGISTRY.definitions()])

    def close(self):
        self.closed = True

    def stop_task(self, task_id):
        raise AssertionError("Phase 7.4.2 creation tests never stop tasks")

    def create_stream_session(self, task_id, camera_id="overview"):
        run_id = task_id[len("task-"):] if task_id.startswith("task-") else task_id
        return StreamDescriptor(
            stream_id="p743-stream",
            run_id=run_id,
            task_id=task_id,
            camera_id=camera_id,
            status=StreamSessionStatus.READY,
            webrtc_url="https://media.example/p743-stream",
            whep_url="https://media.example/p743-stream/whep",
            width=640,
            height=480,
            fps=15,
        )

    def send_input_event(self, event):
        return InputEventAck(True, event.event_id, False)


@unittest.skipUnless(
    SAFE_POSTGRES_URL is not None,
    "PostgreSQL test configuration is unavailable",
)
class Phase742MigrationTest(unittest.TestCase):
    def test_upgrade_downgrade_0007_upgrade_preserves_historical_null_class(self):
        config = Config(str(ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(ROOT / "migrations"))
        environment = {
            "DATABASE_URL": SAFE_POSTGRES_URL.render_as_string(False)
        }
        run_id = "p742-migration-" + uuid4().hex
        engine = create_postgres_engine(SAFE_POSTGRES_URL)
        try:
            with patch.dict(os.environ, environment, clear=False):
                alembic_command.upgrade(config, "head")
                alembic_command.downgrade(config, "20260831_0007")
            self.assertNotIn(
                "class_id",
                {item["name"] for item in inspect(engine).get_columns("runs")},
            )
            now = datetime.now(timezone.utc)
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "INSERT INTO runs "
                        "(run_id,user_id,project_id,status,execution_mode,run_kind,"
                        "parameters,project_config_snapshot,code_sha256,"
                        "simulation_task_id,progress,metrics_summary,error,"
                        "last_sequence,training_steps,update_steps,stop_reason,"
                        "created_at,started_at,ended_at,updated_at) VALUES "
                        "(:run_id,:user_id,:project_id,'queued','mock','train',"
                        "CAST('{}' AS jsonb),CAST('{}' AS jsonb),:digest,NULL,0,"
                        "CAST('{}' AS jsonb),NULL,0,0,0,NULL,:now,NULL,NULL,:now)"
                    ),
                    {
                        "run_id": run_id,
                        "user_id": run_id + "-user",
                        "project_id": run_id + "-project",
                        "digest": "a" * 64,
                        "now": now,
                    },
                )
            with patch.dict(os.environ, environment, clear=False):
                alembic_command.upgrade(config, "head")
            inspector = inspect(engine)
            self.assertIn(
                "class_id",
                {item["name"] for item in inspector.get_columns("runs")},
            )
            self.assertIn(
                "ix_runs_class_id",
                {item["name"] for item in inspector.get_indexes("runs")},
            )
            self.assertEqual(
                {item["referred_table"] for item in inspector.get_foreign_keys("runs")},
                {"classes"},
            )
            with engine.connect() as connection:
                self.assertIsNone(
                    connection.execute(
                        text("SELECT class_id FROM runs WHERE run_id=:run_id"),
                        {"run_id": run_id},
                    ).scalar_one()
                )
        finally:
            with engine.begin() as connection:
                connection.execute(
                    text("DELETE FROM runs WHERE run_id=:run_id"),
                    {"run_id": run_id},
                )
            engine.dispose()


@unittest.skipUnless(
    SAFE_POSTGRES_URL is not None,
    "PostgreSQL test configuration is unavailable",
)
class Phase742ProjectRunPostgreSQLTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = Config(str(ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(ROOT / "migrations"))
        with patch.dict(
            os.environ,
            {"DATABASE_URL": SAFE_POSTGRES_URL.render_as_string(False)},
            clear=False,
        ):
            alembic_command.upgrade(config, "head")
        cls.engine = create_postgres_engine(SAFE_POSTGRES_URL)
        cls.factory = create_session_factory(cls.engine)

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.prefix = "p742-" + uuid4().hex
        self.now = datetime.now(timezone.utc)
        self.ids = {
            name: self.prefix + "-" + name
            for name in (
                "admin", "owner", "assigned", "unassigned", "student", "student-2", "unenrolled",
                "other-admin", "org", "other-org", "course", "course-2",
                "other-course", "class", "class-sibling", "class-2", "other-class",
                "project", "standard-project",
            )
        }
        self._seed()
        self.clients = []

    def tearDown(self):
        for client in reversed(self.clients):
            client.close()
        with self.factory() as session:
            run_ids = select(RunRow.run_id).where(
                RunRow.project_id.like(self.prefix + "%")
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
                delete(EvaluationReportRow).where(
                    EvaluationReportRow.run_id.in_(run_ids)
                )
            )
            session.execute(
                delete(RunArtifactRow).where(
                    RunArtifactRow.run_id.in_(run_ids)
                )
            )
            session.execute(
                delete(RunRow).where(RunRow.project_id.like(self.prefix + "%"))
            )
            session.execute(
                delete(TeachingProjectRow).where(
                    TeachingProjectRow.project_id.like(self.prefix + "%")
                )
            )
            session.execute(
                delete(EnrollmentRow).where(
                    EnrollmentRow.class_id.like(self.prefix + "%")
                )
            )
            session.execute(
                delete(TeachingClassRow).where(
                    TeachingClassRow.class_id.like(self.prefix + "%")
                )
            )
            session.execute(
                delete(CourseRow).where(CourseRow.course_id.like(self.prefix + "%"))
            )
            session.execute(
                delete(OrganizationMembershipRow).where(
                    OrganizationMembershipRow.organization_id.like(self.prefix + "%")
                )
            )
            session.execute(
                delete(OrganizationRow).where(
                    OrganizationRow.organization_id.like(self.prefix + "%")
                )
            )
            session.execute(
                delete(UserRow).where(UserRow.user_id.like(self.prefix + "%"))
            )
            session.commit()
        self.temp.cleanup()

    def _seed(self):
        with self.factory() as session:
            for name in (
                "admin", "owner", "assigned", "unassigned", "student", "student-2", "unenrolled", "other-admin"
            ):
                session.add(
                    UserRow(
                        user_id=self.ids[name],
                        username=self.ids[name],
                        display_name=name,
                        password_hash="test-only",
                        enabled=True,
                        created_at=self.now,
                        updated_at=self.now,
                    )
                )
            session.flush()
            session.add_all(
                [
                    OrganizationRow(
                        organization_id=self.ids["org"], slug=self.ids["org"],
                        name="Phase 742", enabled=True,
                        created_at=self.now, updated_at=self.now,
                    ),
                    OrganizationRow(
                        organization_id=self.ids["other-org"], slug=self.ids["other-org"],
                        name="Other", enabled=True,
                        created_at=self.now, updated_at=self.now,
                    ),
                ]
            )
            session.flush()
            roles = {
                "admin": "administrator", "owner": "instructor",
                "assigned": "instructor", "unassigned": "instructor",
                "student": "student",
                "student-2": "student",
                "unenrolled": "student",
            }
            for name, role in roles.items():
                session.add(
                    OrganizationMembershipRow(
                        membership_id=self.ids[name] + "-membership",
                        organization_id=self.ids["org"], user_id=self.ids[name],
                        role=role, enabled=True,
                        created_at=self.now, updated_at=self.now,
                    )
                )
            session.add(
                OrganizationMembershipRow(
                    membership_id=self.ids["other-admin"] + "-membership",
                    organization_id=self.ids["other-org"],
                    user_id=self.ids["other-admin"], role="administrator",
                    enabled=True, created_at=self.now, updated_at=self.now,
                )
            )
            session.flush()
            session.add_all(
                [
                    self._course("course", "C1", "owner"),
                    self._course("course-2", "C2", "owner"),
                    self._course("other-course", "OC", "other-admin", "other-org"),
                ]
            )
            session.flush()
            session.add_all(
                [
                    self._class("class", "course", "assigned"),
                    self._class("class-sibling", "course", "assigned"),
                    self._class("class-2", "course-2", "assigned"),
                    self._class("other-class", "other-course", "other-admin"),
                ]
            )
            session.flush()
            session.add(
                EnrollmentRow(
                    enrollment_id=self.ids["student"] + "-enrollment",
                    class_id=self.ids["class"], user_id=self.ids["student"],
                    enabled=True, created_at=self.now, updated_at=self.now,
                )
            )
            session.add(
                EnrollmentRow(
                    enrollment_id=self.ids["student-2"] + "-enrollment",
                    class_id=self.ids["class"], user_id=self.ids["student-2"],
                    enabled=True, created_at=self.now, updated_at=self.now,
                )
            )
            session.add_all(
                [
                    self._project("project", "short", True),
                    self._project("standard-project", "standard", False),
                ]
            )
            session.commit()

    def _course(self, name, code, owner, organization="org"):
        return CourseRow(
            course_id=self.ids[name], organization_id=self.ids[organization],
            code=self.prefix + code, title=name, description="",
            owner_instructor_user_id=self.ids[owner], enabled=True,
            created_at=self.now, updated_at=self.now,
        )

    def _class(self, name, course, instructor):
        return TeachingClassRow(
            class_id=self.ids[name], course_id=self.ids[course], code=name,
            name=name, term=self.prefix, instructor_user_id=self.ids[instructor],
            enabled=True, created_at=self.now, updated_at=self.now,
        )

    def _project(self, name, profile, video):
        selected = TrainingProfile(profile)
        return TeachingProjectRow(
            project_id=self.ids[name], course_id=self.ids["course"],
            title=name, description="Description", instructions="Instructions",
            starter_code="# never executed", starter_code_sha256=starter_code_digest("# never executed"),
            training_profile=profile, video_enabled=video,
            parameter_schema=parameter_schema(), evaluation_rubric=evaluation_rubric(selected),
            config_version=1, enabled=True, created_by_user_id=self.ids["owner"],
            created_at=self.now, updated_at=self.now,
        )

    def payload(self, project="project", class_name="class", version=1, seed=7):
        return {
            "project_id": self.ids[project],
            "class_id": self.ids[class_name],
            "project_config_version": version,
            "execution_mode": "simulation",
            "run_kind": "train",
            "parameters": {"seed": seed},
            "code": "student code is hashed only",
        }

    def app_client(self, actor, simulation, **app_options):
        root = Path(self.temp.name) / (actor + uuid4().hex)
        app = create_app(
            root / "runs", project_root=root / "projects",
            run_repository_mode="postgresql", database_url=SAFE_POSTGRES_URL,
            current_user_provider=StaticCurrentUserProvider(self.ids[actor]),
            simulation_client=simulation,
            **app_options,
        )
        client = ASGITestClient(app)
        self.clients.append(client)
        return app, client

    def post(self, actor, key, simulation, payload=None):
        _, client = self.app_client(actor, simulation)
        return client.post(
            "/api/v1/runs", json=payload or self.payload(),
            headers={"Idempotency-Key": key},
        )

    def test_scene_revision_is_frozen_and_invalid_geometry_is_rejected_before_dispatch(self):
        from robotics.scenes import default_s1_scene
        simulation = RecordingSimulationClient(self.factory)
        app, client = self.app_client("student", simulation)
        document = default_s1_scene()
        document["scene_id"] = "scene-" + self.prefix
        document["objects"][1]["position"] = [0.48, 0.05, -0.12]
        document["objects"][1]["scale"] = [0.06, 0.10, 0.08]
        document["objects"][2]["position"] = [0.64, 0.01, 0.15]
        saved = client.post("/api/scenes", json={"document": document}).json()
        payload = self.payload()
        payload["environment_type"] = "s1_push"
        payload["scene_ref"] = {"scene_id": saved["scene_id"], "revision": saved["revision"]}
        response = client.post("/api/v1/runs", json=payload, headers={"Idempotency-Key": self.prefix + "-scene"})
        self.assertEqual(response.status_code, 202, response.text)
        record = response.json()
        self.assertEqual(record["project_config_snapshot"]["scene"]["document"], saved)
        self.assertEqual(simulation.calls[-1][2]["object_size"], [0.06, 0.08, 0.10])
        self.assertEqual(simulation.calls[-1][2]["object_start"], [0.48, -0.12, 0.425])
        self.assertEqual(simulation.calls[-1][2]["goal_position"], [0.64, 0.15, 0.425])
        # Even an external overwrite of the saved Scene cannot change the Run's snapshot.
        document["objects"][2]["position"][0] = 0.66
        updated = client.post("/api/scenes", json={"document": document}).json()
        frozen = client.get("/api/v1/runs/" + record["run_id"]).json()
        self.assertEqual(frozen["project_config_snapshot"]["scene"]["document"], saved)
        stale = client.post("/api/v1/runs", json=payload, headers={"Idempotency-Key": self.prefix + "-stale"})
        self.assertEqual(stale.status_code, 409, stale.text)
        document["objects"][1]["position"][0] = 0.19
        invalid = client.post("/api/scenes", json={"document": document}).json()
        payload["scene_ref"]["revision"] = invalid["revision"]
        rejected = client.post("/api/v1/runs", json=payload, headers={"Idempotency-Key": self.prefix + "-invalid"})
        self.assertEqual(rejected.status_code, 422, rejected.text)
        self.assertEqual(len(simulation.calls), 1)

    def test_creation_matrix_snapshot_exact_configs_and_commit_order(self):
        simulation = RecordingSimulationClient(self.factory)
        for actor in ("student", "assigned", "owner", "admin"):
            response = self.post(actor, "matrix-" + actor, simulation)
            self.assertEqual(response.status_code, 202, response.text)
            body = response.json()
            self.assertEqual(body["class_id"], self.ids["class"])
            self.assertEqual(body["parameters"], {"seed": 7})
            self.assertEqual(
                body["code_sha256"],
                hashlib.sha256(b"student code is hashed only").hexdigest(),
            )
            snapshot = body["project_config_snapshot"]
            required = {
                "schema_version", "project_id", "course_id", "class_id", "title",
                "description", "instructions", "task_type", "environment_type",
                "algorithm", "success_condition", "training_profile",
                "expected_training_steps", "video_enabled", "parameter_schema",
                "resolved_parameters", "evaluation_rubric", "config_version",
                "project_enabled_at_creation", "starter_code_sha256", "training_config",
            }
            self.assertTrue(required.issubset(snapshot))
            self.assertEqual(snapshot["resolved_parameters"], {"seed": 7})
            self.assertEqual(
                snapshot["training_config"],
                worker_config(TrainingProfile.SHORT, True, 7),
            )
            self.assertEqual(len(snapshot["training_config"]), 16)
        standard = self.post(
            "student", "standard", simulation,
            self.payload(project="standard-project", seed=2147483647),
        )
        self.assertEqual(standard.status_code, 202, standard.text)
        frozen = standard.json()["project_config_snapshot"]
        self.assertEqual(frozen["expected_training_steps"], 320)
        self.assertEqual(
            frozen["training_config"],
            worker_config(TrainingProfile.STANDARD, False, 2147483647),
        )

    def test_evaluation_history_list_role_scope_sort_restart_and_immutability(self):
        simulation = RecordingSimulationClient(self.factory)
        app, student_client = self.app_client("student", simulation)
        created = student_client.post(
            "/api/v1/runs",
            json=self.payload(),
            headers={"Idempotency-Key": "evaluation-history"},
        )
        self.assertEqual(created.status_code, 202, created.text)
        run_id = created.json()["run_id"]
        run = app.state.run_service.get_run(run_id)
        for status in (
            RunStatus.INITIALIZING,
            RunStatus.RUNNING,
            RunStatus.FINALIZING,
            RunStatus.COMPLETED,
        ):
            run = app.state.run_service.transition_status(run_id, status)
        first = student_client.post(
            "/api/v1/evaluations", json={"run_id": run_id}
        )
        self.assertEqual(first.status_code, 201, first.text)
        first_body = deepcopy(first.json())
        with self.factory() as session:
            session.add(
                RunArtifactRow(
                    artifact_id=self.prefix + "-history-artifact",
                    run_id=run_id,
                    artifact_type="manifest",
                    path="manifest.json",
                    size_bytes=2,
                    sha256="a" * 64,
                    is_primary_model=False,
                    created_at=datetime.now(timezone.utc),
                )
            )
            session.commit()
        second = student_client.post(
            "/api/v1/evaluations", json={"run_id": run_id}
        )
        self.assertEqual(second.status_code, 201, second.text)

        with self.factory() as session:
            before_count = session.scalar(
                select(func.count())
                .select_from(EvaluationReportRow)
                .where(EvaluationReportRow.run_id == run_id)
            )
        for actor in ("student", "assigned", "owner", "admin"):
            with self.subTest(actor=actor):
                _, client = self.app_client(actor, simulation)
                response = client.get(
                    "/api/v1/evaluations", params={"run_id": run_id}
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(len(response.json()), 2)
                self.assertEqual(
                    response.json()[0]["evaluation_id"],
                    second.json()["evaluation_id"],
                )
                self.assertEqual(response.json()[1], first_body)
                self.assertNotIn("success_rate", str(response.json()))
        for actor in ("student-2", "unassigned", "other-admin"):
            with self.subTest(hidden_actor=actor):
                _, client = self.app_client(actor, simulation)
                self.assertEqual(
                    client.get(
                        "/api/v1/evaluations", params={"run_id": run_id}
                    ).status_code,
                    404,
                )
        with self.factory() as session:
            session.execute(
                update(TeachingProjectRow)
                .where(TeachingProjectRow.project_id == self.ids["project"])
                .values(enabled=False)
            )
            session.commit()
        try:
            _, disabled_client = self.app_client("student", simulation)
            self.assertEqual(
                disabled_client.get(
                    "/api/v1/evaluations", params={"run_id": run_id}
                ).status_code,
                404,
            )
        finally:
            with self.factory() as session:
                session.execute(
                    update(TeachingProjectRow)
                    .where(TeachingProjectRow.project_id == self.ids["project"])
                    .values(enabled=True)
                )
                session.commit()
        _, rebuilt_client = self.app_client("student", simulation)
        rebuilt = rebuilt_client.get(
            "/api/v1/evaluations", params={"run_id": run_id}
        )
        self.assertEqual(rebuilt.status_code, 200, rebuilt.text)
        self.assertEqual(len(rebuilt.json()), 2)
        with self.factory() as session:
            after_count = session.scalar(
                select(func.count())
                .select_from(EvaluationReportRow)
                .where(EvaluationReportRow.run_id == run_id)
            )
            persisted_first = session.get(
                EvaluationReportRow, first_body["evaluation_id"]
            )
            self.assertEqual(persisted_first.evidence, first_body["evidence"])
        self.assertEqual(after_count, before_count)

    def test_visibility_permission_version_and_disabled_rules(self):
        simulation = RecordingSimulationClient()
        self.assertEqual(
            self.post("unenrolled", "no-enrollment", simulation).status_code, 403
        )
        self.assertEqual(
            self.post("other-admin", "cross-org", simulation).status_code, 404
        )
        self.assertEqual(
            self.post("unassigned", "unassigned", simulation).status_code, 403
        )
        self.assertEqual(
            self.post(
                "student", "unknown-project", simulation,
                dict(self.payload(), project_id=self.prefix + "-missing-project"),
            ).status_code,
            404,
        )
        self.assertEqual(
            self.post(
                "student", "unknown-class", simulation,
                dict(self.payload(), class_id=self.prefix + "-missing-class"),
            ).status_code,
            404,
        )
        self.assertEqual(
            self.post(
                "student", "mismatch", simulation,
                self.payload(class_name="class-2"),
            ).status_code,
            404,
        )
        stale = self.post(
            "student", "stale", simulation, self.payload(version=99)
        )
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(stale.json()["detail"]["code"], "project_version_conflict")
        disabled_resources = (
            (UserRow, UserRow.user_id == self.ids["student"]),
            (
                OrganizationMembershipRow,
                OrganizationMembershipRow.membership_id
                == self.ids["student"] + "-membership",
            ),
            (OrganizationRow, OrganizationRow.organization_id == self.ids["org"]),
            (CourseRow, CourseRow.course_id == self.ids["course"]),
            (TeachingClassRow, TeachingClassRow.class_id == self.ids["class"]),
            (TeachingProjectRow, TeachingProjectRow.project_id == self.ids["project"]),
        )
        for index, (model, condition) in enumerate(disabled_resources):
            with self.subTest(model=model.__name__):
                with self.factory() as session:
                    session.execute(update(model).where(condition).values(enabled=False))
                    session.commit()
                try:
                    self.assertEqual(
                        self.post(
                            "student", "disabled-%d" % index, simulation
                        ).status_code,
                        404,
                    )
                finally:
                    with self.factory() as session:
                        session.execute(update(model).where(condition).values(enabled=True))
                        session.commit()
        self.assertEqual(simulation.calls, [])

    def test_project_lock_prevents_snapshot_patch_race(self):
        simulation = RecordingSimulationClient()
        app, _ = self.app_client("student", simulation)
        service = app.state.run_creation_service
        self.assertIsInstance(service, PostgreSQLProjectRunCreationService)
        snapshot_entered = threading.Event()
        release_snapshot = threading.Event()
        patch_started = threading.Event()
        original_snapshot = service._snapshot
        command = IdempotentRunCreateCommand(
            user_id=self.ids["student"],
            idempotency_key="lock-race",
            project_id=self.ids["project"],
            class_id=self.ids["class"],
            project_config_version=1,
            execution_mode=ExecutionMode.SIMULATION,
            run_kind=RunKind.TRAIN,
            parameters={"seed": 7},
            code="student code is hashed only",
        )

        def blocking_snapshot(*args):
            snapshot_entered.set()
            if not release_snapshot.wait(timeout=5):
                raise AssertionError("snapshot lock test timed out")
            return original_snapshot(*args)

        def patch_project():
            with self.factory() as session:
                patch_started.set()
                session.execute(
                    update(TeachingProjectRow)
                    .where(TeachingProjectRow.project_id == self.ids["project"])
                    .values(config_version=2, training_profile="standard")
                )
                session.commit()

        with patch.object(service, "_snapshot", side_effect=blocking_snapshot):
            with ThreadPoolExecutor(max_workers=2) as executor:
                creation = executor.submit(
                    service.create_run, command
                )
                self.assertTrue(snapshot_entered.wait(timeout=5))
                mutation = executor.submit(patch_project)
                self.assertTrue(patch_started.wait(timeout=5))
                try:
                    with self.assertRaises(FutureTimeoutError):
                        mutation.result(timeout=0.1)
                finally:
                    release_snapshot.set()
                result = creation.result(timeout=5)
                mutation.result(timeout=5)
        self.assertEqual(
            result.run.project_config_snapshot["config_version"], 1
        )
        self.assertEqual(
            result.run.project_config_snapshot["training_profile"],
            "short",
        )

    def test_first_creation_rechecks_idempotency_after_project_lock(self):
        simulation = RecordingSimulationClient()
        app, client = self.app_client("student", simulation)
        service = app.state.run_creation_service
        with patch.object(
            service, "_find_key", wraps=service._find_key
        ) as find_key:
            response = client.post(
                "/api/v1/runs",
                json=self.payload(),
                headers={"Idempotency-Key": "double-check"},
            )
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(find_key.call_count, 2)

    def test_replay_project_upgrade_recovery_and_revocation(self):
        simulation = RecordingSimulationClient()
        first = self.post("student", "bound", simulation)
        self.assertEqual(first.status_code, 202, first.text)
        frozen = deepcopy(first.json()["project_config_snapshot"])
        with self.factory() as session:
            session.execute(
                update(TeachingProjectRow)
                .where(TeachingProjectRow.project_id == self.ids["project"])
                .values(
                    config_version=2,
                    training_profile="standard",
                    evaluation_rubric=evaluation_rubric(
                        TrainingProfile.STANDARD, 2
                    ),
                    title="Updated",
                )
            )
            session.commit()
        replay = self.post("student", "bound", simulation)
        self.assertEqual(replay.status_code, 202, replay.text)
        self.assertEqual(replay.headers["Idempotency-Replayed"], "true")
        self.assertEqual(replay.json()["project_config_snapshot"], frozen)
        self.assertEqual(len(simulation.calls), 1)

        failed_simulation = RecordingSimulationClient()
        failed_simulation.error = SimulationTimeoutError("private upstream")
        queued = self.post(
            "student", "queued", failed_simulation,
            self.payload(version=2),
        )
        self.assertEqual(queued.status_code, 503)
        old_config = deepcopy(failed_simulation.calls[-1][2])
        with self.factory() as session:
            session.execute(
                update(TeachingProjectRow)
                .where(TeachingProjectRow.project_id == self.ids["project"])
                .values(config_version=3, video_enabled=False)
            )
            session.commit()
        failed_simulation.error = None
        recovered = self.post(
            "student", "queued", failed_simulation,
            self.payload(version=2),
        )
        self.assertEqual(recovered.status_code, 202, recovered.text)
        self.assertEqual(failed_simulation.calls[-1][2], old_config)

        denied_simulation = RecordingSimulationClient()
        denied_simulation.error = SimulationTimeoutError("temporary")
        denied = self.post(
            "student", "revoked", denied_simulation,
            self.payload(version=3),
        )
        self.assertEqual(denied.status_code, 503)
        call_count = len(denied_simulation.calls)
        with self.factory() as session:
            session.execute(
                update(EnrollmentRow)
                .where(EnrollmentRow.user_id == self.ids["student"])
                .values(enabled=False)
            )
            session.commit()
        denied_simulation.error = None
        replay_denied = self.post(
            "student", "revoked", denied_simulation,
            self.payload(version=3),
        )
        self.assertEqual(replay_denied.status_code, 403)
        self.assertEqual(len(denied_simulation.calls), call_count)

    def test_tampered_snapshot_identity_fails_before_external_call(self):
        simulation = RecordingSimulationClient()
        simulation.error = SimulationTimeoutError("temporary")
        failed = self.post("student", "tampered-identity", simulation)
        self.assertEqual(failed.status_code, 503)
        call_count = len(simulation.calls)
        run_id = simulation.calls[-1][0]
        with self.factory() as session:
            row = session.get(RunRow, run_id)
            snapshot = deepcopy(row.project_config_snapshot)
            snapshot["course_id"] = self.ids["course-2"]
            row.project_config_snapshot = snapshot
            session.commit()
        simulation.error = None
        replay = self.post("student", "tampered-identity", simulation)
        self.assertEqual(replay.status_code, 502)
        self.assertEqual(len(simulation.calls), call_count)

    def test_concurrent_identical_request_creates_one_run_and_key(self):
        simulation = RecordingSimulationClient()
        barrier = threading.Barrier(2)

        def submit(index):
            _, client = self.app_client("student", simulation)
            barrier.wait(timeout=5)
            return client.post(
                "/api/v1/runs", json=self.payload(),
                headers={"Idempotency-Key": "concurrent"},
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(executor.map(submit, range(2)))
        self.assertEqual([item.status_code for item in responses], [202, 202])
        self.assertEqual(responses[0].json()["run_id"], responses[1].json()["run_id"])
        with self.factory() as session:
            self.assertEqual(
                session.scalar(
                    select(func.count()).select_from(RunRow).where(
                        RunRow.project_id == self.ids["project"]
                    )
                ),
                1,
            )
            self.assertEqual(
                session.scalar(
                    select(func.count()).select_from(RunIdempotencyKeyRow).where(
                        RunIdempotencyKeyRow.user_id == self.ids["student"]
                    )
                ),
                1,
            )

    def test_phase743_class_scoped_visibility_control_and_filters(self):
        simulation = RecordingSimulationClient()
        first = self.post("student", "p743-student", simulation)
        second = self.post("student-2", "p743-student-2", simulation)
        standard = self.post(
            "student", "p743-standard", simulation,
            self.payload(project="standard-project"),
        )
        sibling = self.post(
            "owner", "p743-sibling", simulation,
            self.payload(class_name="class-sibling"),
        )
        for response in (first, second, standard, sibling):
            self.assertEqual(response.status_code, 202, response.text)
        first_id = first.json()["run_id"]
        second_id = second.json()["run_id"]
        run_ids = [first_id, second_id, standard.json()["run_id"]]
        with self.factory() as session:
            session.execute(
                update(RunRow)
                .where(RunRow.run_id.in_(run_ids))
                .values(created_at=self.now)
            )
            session.commit()

        _, student = self.app_client("student", simulation)
        own = student.get("/api/v1/classes/%s/runs" % self.ids["class"])
        self.assertEqual(own.status_code, 200, own.text)
        self.assertEqual(
            {item["run_id"] for item in own.json()},
            {first_id, standard.json()["run_id"]},
        )
        self.assertEqual(
            [item["run_id"] for item in own.json()],
            sorted([first_id, standard.json()["run_id"]]),
        )
        default_list = student.get("/api/v1/runs")
        self.assertEqual(default_list.status_code, 200, default_list.text)
        self.assertEqual(
            {item["run_id"] for item in default_list.json()},
            {first_id, standard.json()["run_id"]},
        )
        self.assertEqual(
            student.get("/api/v1/runs/%s" % second_id).status_code, 404
        )
        self.assertEqual(
            student.post(
                "/api/v1/runs/compare", json={"run_ids": [first_id, second_id]}
            ).status_code,
            404,
        )
        self.assertEqual(
            student.post(
                "/api/v1/runs/export", json={"run_ids": [first_id, second_id]}
            ).status_code,
            404,
        )
        filtered = student.get(
            "/api/v1/classes/%s/runs?project_id=%s&status=queued"
            % (self.ids["class"], self.ids["project"])
        )
        self.assertEqual([item["run_id"] for item in filtered.json()], [first_id])

        for actor in ("assigned", "owner", "admin"):
            with self.subTest(actor=actor):
                _, client = self.app_client(actor, simulation)
                visible = client.get(
                    "/api/v1/classes/%s/runs" % self.ids["class"]
                )
                self.assertEqual(visible.status_code, 200, visible.text)
                self.assertEqual(len(visible.json()), 3)
                self.assertEqual(
                    [item["run_id"] for item in visible.json()], sorted(run_ids)
                )
                self.assertEqual(
                    client.get("/api/v1/runs/%s/events" % first_id).status_code,
                    200,
                )
                self.assertEqual(
                    client.get("/api/v1/runs/%s/artifacts" % first_id).status_code,
                    200,
                )
                self.assertEqual(
                    client.post(
                        "/api/v1/runs/compare",
                        json={"run_ids": [first_id, second_id]},
                    ).status_code,
                    200,
                )
                self.assertEqual(
                    client.post(
                        "/api/v1/runs/export",
                        json={"run_ids": [first_id, second_id]},
                    ).status_code,
                    200,
                )
                self.assertEqual(
                    client.post("/api/v1/runs/%s/stop" % first_id).status_code,
                    403,
                )
                sibling_runs = client.get(
                    "/api/v1/classes/%s/runs" % self.ids["class-sibling"]
                )
                self.assertEqual(sibling_runs.status_code, 200, sibling_runs.text)
                self.assertEqual(
                    [item["run_id"] for item in sibling_runs.json()],
                    [sibling.json()["run_id"]],
                )

        self.assertEqual(
            student.get(
                "/api/v1/classes/%s/runs" % self.ids["class-sibling"]
            ).status_code,
            404,
        )

        with self.factory() as session:
            session.execute(
                update(RunRow)
                .where(RunRow.run_id == first_id)
                .values(status="running")
            )
            session.commit()
        instructor_app, instructor = self.app_client(
            "assigned", simulation, websocket_auth_timeout_seconds=0.01
        )
        student_app, _ = self.app_client(
            "student", simulation, websocket_auth_timeout_seconds=0.01
        )
        self.assertEqual(
            instructor.get("/api/v1/runs/%s/stream" % first_id).status_code, 200
        )

        async def websocket_checks():
            events = ASGIWebSocketDriver(
                instructor_app, "/api/v1/runs/%s/events" % first_id
            )
            self.assertEqual((await events.connect())["type"], "websocket.accept")
            await events.disconnect()
            denied_input = ASGIWebSocketDriver(
                instructor_app, "/api/v1/runs/%s/input" % first_id
            )
            self.assertEqual((await denied_input.connect())["code"], 4403)
            owner_input = ASGIWebSocketDriver(
                student_app, "/api/v1/runs/%s/input" % first_id
            )
            self.assertEqual(
                (await owner_input.connect())["type"], "websocket.accept"
            )
            await owner_input.send_json(
                {
                    "event_id": "p743-input",
                    "run_id": first_id,
                    "task_id": "task-" + first_id,
                    "type": "KEYBOARD",
                    "occurred_at": "2026-09-03T00:00:00+00:00",
                    "key": "W",
                    "action": "pressed",
                }
            )
            self.assertEqual((await owner_input.receive_json())["type"], "ACK")
            await owner_input.disconnect()
            unauthenticated = ASGIWebSocketDriver(
                instructor_app,
                "/api/v1/runs/%s/events" % first_id,
                headers=[],
            )
            self.assertEqual(
                (await unauthenticated.connect())["type"], "websocket.accept"
            )
            self.assertEqual((await unauthenticated.receive())["code"], 4401)

        asyncio.run(websocket_checks())

        with self.factory() as session:
            session.execute(
                update(RunRow)
                .where(RunRow.run_id == first_id)
                .values(status="completed", ended_at=self.now)
            )
            session.commit()
        evaluation = instructor.post(
            "/api/v1/evaluations", json={"run_id": first_id}
        )
        self.assertEqual(evaluation.status_code, 201, evaluation.text)
        self.assertEqual(evaluation.json()["requested_by_user_id"], self.ids["assigned"])
        _, other_student = self.app_client("student-2", simulation)
        self.assertEqual(
            other_student.get(
                "/api/v1/evaluations/%s" % evaluation.json()["evaluation_id"]
            ).status_code,
            404,
        )

        for actor in ("unassigned", "other-admin"):
            _, client = self.app_client(actor, simulation)
            self.assertEqual(
                client.get("/api/v1/classes/%s/runs" % self.ids["class"]).status_code,
                404,
            )
            self.assertEqual(
                client.get("/api/v1/runs/%s" % first_id).status_code, 404
            )

    def test_phase743_revocation_and_legacy_owner_only(self):
        simulation = RecordingSimulationClient()
        created = self.post("student", "p743-revoke", simulation)
        self.assertEqual(created.status_code, 202, created.text)
        run_id = created.json()["run_id"]
        _, student = self.app_client("student", simulation)
        checks = (
            (UserRow, UserRow.user_id == self.ids["student"]),
            (OrganizationRow, OrganizationRow.organization_id == self.ids["org"]),
            (
                OrganizationMembershipRow,
                OrganizationMembershipRow.user_id == self.ids["student"],
            ),
            (CourseRow, CourseRow.course_id == self.ids["course"]),
            (TeachingClassRow, TeachingClassRow.class_id == self.ids["class"]),
            (
                EnrollmentRow,
                EnrollmentRow.user_id == self.ids["student"],
            ),
            (TeachingProjectRow, TeachingProjectRow.project_id == self.ids["project"]),
        )
        for model, condition in checks:
            with self.subTest(model=model.__name__):
                with self.factory() as session:
                    session.execute(update(model).where(condition).values(enabled=False))
                    session.commit()
                try:
                    self.assertEqual(
                        student.get("/api/v1/runs/%s" % run_id).status_code, 404
                    )
                finally:
                    with self.factory() as session:
                        session.execute(update(model).where(condition).values(enabled=True))
                        session.commit()

        legacy = self.post("student", "p743-legacy-source", simulation)
        legacy_id = legacy.json()["run_id"]
        with self.factory() as session:
            row = session.get(RunRow, legacy_id)
            row.class_id = None
            row.project_id = self.prefix + "-arbitrary-legacy-project"
            row.project_config_snapshot = {}
            session.commit()
        self.assertEqual(
            student.get("/api/v1/runs/%s" % legacy_id).status_code, 200
        )
        for actor in ("assigned", "owner", "admin"):
            _, client = self.app_client(actor, simulation)
            self.assertEqual(
                client.get("/api/v1/runs/%s" % legacy_id).status_code, 404
            )


if __name__ == "__main__":
    unittest.main()
