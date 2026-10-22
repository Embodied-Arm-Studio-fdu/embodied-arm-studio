import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from alembic import command as alembic_command
from alembic.config import Config
from sqlalchemy import delete, func, inspect, select, update
from sqlalchemy.exc import IntegrityError

from apps.api.database import create_postgres_engine, create_session_factory
from apps.api.main import create_app
from robotics.auth import AuthenticationService, PasswordHasher, PostgreSQLUserRepository
from robotics.courses import CourseService, PostgreSQLCourseRepository
from robotics.organizations import (
    OrganizationRole,
    OrganizationService,
    PostgreSQLOrganizationRepository,
)
from robotics.projects import (
    ALGORITHM,
    ENVIRONMENT_TYPE,
    TASK_TYPE,
    PostgreSQLTeachingProjectRepository,
    TeachingProject,
    TeachingProjectConflict,
    TeachingProjectNotFound,
    TeachingProjectPermissionDenied,
    TeachingProjectService,
    TeachingProjectVersionConflict,
    TrainingProfile,
    current_config_version,
    evaluation_rubric,
    parameter_schema,
    profile_config,
    starter_code_digest,
    success_condition,
    worker_config,
)
from robotics.runs import InMemoryRunRepository
from robotics.runs.db_models import (
    CourseRow,
    EnrollmentRow,
    OrganizationMembershipRow,
    OrganizationRow,
    TeachingClassRow,
    TeachingProjectRow,
    UserRow,
)
from tests.auth import StaticCurrentUserProvider
from tests.http_client import ASGITestClient
from tests.postgres_safety import safe_test_database_url


ROOT = Path(__file__).resolve().parents[1]
SAFE_POSTGRES_URL = safe_test_database_url()


class TeachingProjectContractTest(unittest.TestCase):
    def test_frozen_s1_contract_profiles_schema_rubric_and_worker_keys(self):
        self.assertEqual(TASK_TYPE, "training")
        self.assertEqual(ENVIRONMENT_TYPE, "s1_push")
        self.assertEqual(ALGORITHM, "PPO")
        self.assertEqual(success_condition(), {
            "metric": "object_to_goal_distance", "operator": "lte",
            "value": 0.05, "unit": "m",
        })
        schema = parameter_schema()
        self.assertEqual(schema["required"], ["seed"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["seed"]["minimum"], 0)
        self.assertEqual(schema["properties"]["seed"]["maximum"], 2147483647)
        self.assertEqual(profile_config(TrainingProfile.SHORT), {
            "rollout_steps": 16, "num_updates": 2,
            "episode_steps": 32, "expected_training_steps": 32,
        })
        self.assertEqual(profile_config(TrainingProfile.STANDARD), {
            "rollout_steps": 16, "num_updates": 20,
            "episode_steps": 64, "expected_training_steps": 320,
        })
        self.assertEqual(current_config_version(TrainingProfile.SHORT), 1)
        self.assertEqual(current_config_version(TrainingProfile.STANDARD), 4)
        self.assertEqual(profile_config(TrainingProfile.STANDARD, 2), {
            "rollout_steps": 200, "num_updates": 100,
            "episode_steps": 256, "expected_training_steps": 20000,
        })
        self.assertEqual(
            profile_config(TrainingProfile.STANDARD, 3),
            profile_config(TrainingProfile.STANDARD, 2),
        )
        self.assertEqual(
            profile_config(TrainingProfile.STANDARD, 4),
            profile_config(TrainingProfile.STANDARD, 2),
        )
        short = worker_config(TrainingProfile.SHORT, True, 42)
        standard = worker_config(TrainingProfile.STANDARD, False, 2147483647)
        self.assertEqual(len(short), 16)
        self.assertEqual(short["camera_id"], "overview")
        self.assertEqual((short["video_width"], short["video_height"], short["video_fps"]), (640, 480, 15.0))
        self.assertEqual(standard["num_updates"], 20)
        production = worker_config(
            TrainingProfile.STANDARD, True, 42, config_version=2
        )
        self.assertEqual(len(production), 17)
        self.assertEqual(production["run_mode"], "training")
        self.assertEqual(production["rollout_steps"], 200)
        self.assertEqual(production["num_updates"], 100)
        self.assertEqual(production["episode_steps"], 256)
        self.assertEqual(production["physics_steps_per_action"], 8)
        self.assertEqual(production["checkpoint_every"], 10)
        self.assertEqual(production["update_epochs"], 1)
        self.assertEqual(production["batch_size"], 16)
        self.assertEqual(evaluation_rubric(TrainingProfile.SHORT)["criteria"][1]["target"], 32)
        self.assertEqual(evaluation_rubric(TrainingProfile.STANDARD)["criteria"][2]["target"], 20)
        for invalid in (-1, 2147483648, True):
            with self.assertRaises(ValueError):
                worker_config(TrainingProfile.SHORT, False, invalid)

    def test_starter_digest_size_deep_copy_and_utc_validation(self):
        self.assertEqual(
            starter_code_digest(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        )
        with self.assertRaisesRegex(ValueError, "512 KiB"):
            starter_code_digest("测" * 180000)

        schema = parameter_schema()
        rubric = evaluation_rubric(TrainingProfile.SHORT)
        now = datetime.now(timezone.utc)
        project = TeachingProject(
            "p", "c", "Title", "", "", "", starter_code_digest(""),
            TrainingProfile.SHORT, False, schema, rubric, 1, True, "u", now, now,
        )
        schema["required"].append("unsafe")
        rubric["criteria"][0]["target"] = "failed"
        self.assertEqual(project.parameter_schema["required"], ["seed"])
        self.assertEqual(project.evaluation_rubric["criteria"][0]["target"], "completed")
        naive = datetime.now()
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            TeachingProject(
                "p", "c", "Title", "", "", "", starter_code_digest(""),
                TrainingProfile.SHORT, False, parameter_schema(),
                evaluation_rubric(TrainingProfile.SHORT), 1, True, "u", naive, naive,
            )
        non_utc = datetime.now(timezone(timedelta(hours=8)))
        with self.assertRaisesRegex(ValueError, "must use UTC"):
            TeachingProject(
                "p", "c", "Title", "", "", "", starter_code_digest(""),
                TrainingProfile.SHORT, False, parameter_schema(),
                evaluation_rubric(TrainingProfile.SHORT), 1, True, "u", non_utc, non_utc,
            )

    def test_new_standard_project_uses_current_v4_contract(self):
        class CaptureRepository:
            project = None

            def create_project(self, project, _actor_user_id):
                self.project = project
                return project

        repository = CaptureRepository()
        created = TeachingProjectService(repository).create_project(
            "course",
            "owner",
            title="Production",
            description="",
            instructions="",
            starter_code="",
            training_profile=TrainingProfile.STANDARD,
            video_enabled=True,
            enabled=True,
        )
        self.assertIs(created, repository.project)
        self.assertEqual(created.config_version, 4)
        self.assertEqual(created.expected_training_steps, 20000)
        self.assertEqual(
            created.evaluation_rubric,
            evaluation_rubric(TrainingProfile.STANDARD, 4),
        )


@unittest.skipUnless(
    SAFE_POSTGRES_URL is not None,
    "PostgreSQL test configuration is unavailable",
)
class TeachingProjectPostgreSQLTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Config(str(ROOT / "alembic.ini"))
        cls.config.set_main_option("script_location", str(ROOT / "migrations"))
        cls.environment = {"DATABASE_URL": SAFE_POSTGRES_URL.render_as_string(False)}
        with patch.dict(os.environ, cls.environment, clear=False):
            alembic_command.upgrade(cls.config, "head")
        cls.engine = create_postgres_engine(SAFE_POSTGRES_URL)
        cls.factory = create_session_factory(cls.engine)

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def setUp(self):
        self.prefix = "p741-%s" % uuid4().hex
        self.users = PostgreSQLUserRepository(self.factory)
        self.auth = AuthenticationService(self.users, PasswordHasher())
        self.organizations = OrganizationService(
            PostgreSQLOrganizationRepository(self.factory)
        )
        self.courses = CourseService(PostgreSQLCourseRepository(self.factory))
        self.repository = PostgreSQLTeachingProjectRepository(self.factory)
        self.projects = TeachingProjectService(self.repository)
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        with self.factory() as session:
            organization_ids = session.execute(
                select(OrganizationRow.organization_id).where(
                    OrganizationRow.slug.like(self.prefix + "%")
                )
            ).scalars().all()
            course_ids = session.execute(
                select(CourseRow.course_id).where(
                    CourseRow.organization_id.in_(organization_ids)
                )
            ).scalars().all() if organization_ids else []
            class_ids = session.execute(
                select(TeachingClassRow.class_id).where(
                    TeachingClassRow.course_id.in_(course_ids)
                )
            ).scalars().all() if course_ids else []
            if course_ids:
                session.execute(delete(TeachingProjectRow).where(TeachingProjectRow.course_id.in_(course_ids)))
            if class_ids:
                session.execute(delete(EnrollmentRow).where(EnrollmentRow.class_id.in_(class_ids)))
                session.execute(delete(TeachingClassRow).where(TeachingClassRow.class_id.in_(class_ids)))
            if course_ids:
                session.execute(delete(CourseRow).where(CourseRow.course_id.in_(course_ids)))
            if organization_ids:
                session.execute(delete(OrganizationMembershipRow).where(OrganizationMembershipRow.organization_id.in_(organization_ids)))
                session.execute(delete(OrganizationRow).where(OrganizationRow.organization_id.in_(organization_ids)))
            session.execute(delete(UserRow).where(UserRow.username.like(self.prefix + "%")))
            session.commit()
        self.temp.cleanup()

    def create_user(self, suffix):
        return self.auth.create_user(
            self.prefix + suffix, "Phase 7.4.1 " + suffix, "safe-test-password"
        )

    def create_context(self, suffix=""):
        admin = self.create_user(suffix + "-admin")
        owner = self.create_user(suffix + "-owner")
        assigned = self.create_user(suffix + "-assigned")
        student = self.create_user(suffix + "-student")
        outsider = self.create_user(suffix + "-outsider")
        organization, _ = self.organizations.create_organization(
            self.prefix + suffix + "-org", "Project Test Organization", admin.user_id
        )
        self.organizations.add_membership(organization.organization_id, owner.user_id, OrganizationRole.INSTRUCTOR)
        self.organizations.add_membership(organization.organization_id, assigned.user_id, OrganizationRole.INSTRUCTOR)
        self.organizations.add_membership(organization.organization_id, student.user_id, OrganizationRole.STUDENT)
        course = self.courses.create_course(
            organization.organization_id, self.prefix + suffix + "-COURSE",
            "Robotics", "", owner.user_id, owner.user_id,
        )
        teaching_class = self.courses.create_class(
            course.course_id, "A", "Class A", "2026", assigned.user_id, owner.user_id
        )
        return organization, course, teaching_class, admin, owner, assigned, student, outsider

    def create_project(self, course, actor, title="S1 Push", **changes):
        values = {
            "title": title,
            "description": "Push with PPO",
            "instructions": "Observe real metrics.",
            "starter_code": "# stored only\n",
            "training_profile": TrainingProfile.SHORT,
            "video_enabled": False,
            "enabled": True,
        }
        values.update(changes)
        return self.projects.create_project(course.course_id, actor.user_id, **values)

    def app_for(self, user_id):
        root = Path(self.temp.name) / user_id
        return create_app(
            root / "runs",
            project_root=root / "workspaces",
            run_repository=InMemoryRunRepository(),
            current_user_provider=StaticCurrentUserProvider(user_id),
            organization_service=self.organizations,
            course_service=self.courses,
            teaching_project_service=self.projects,
        )

    def test_01_migration_round_trip_constraints_indexes_utc_and_recovery(self):
        with self.factory() as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(TeachingProjectRow)), 0)
        with patch.dict(os.environ, self.environment, clear=False):
            alembic_command.downgrade(self.config, "20260831_0006")
        self.assertNotIn("teaching_projects", inspect(self.engine).get_table_names())
        with patch.dict(os.environ, self.environment, clear=False):
            alembic_command.upgrade(self.config, "head")
        inspector = inspect(self.engine)
        self.assertIn("teaching_projects", inspector.get_table_names())
        self.assertIn(
            {"course_id", "title"},
            [set(item["column_names"]) for item in inspector.get_unique_constraints("teaching_projects")],
        )
        self.assertEqual(
            {"courses", "users"},
            {item["referred_table"] for item in inspector.get_foreign_keys("teaching_projects")},
        )
        self.assertGreaterEqual(len(inspector.get_indexes("teaching_projects")), 3)
        checks = {item["name"] for item in inspector.get_check_constraints("teaching_projects")}
        self.assertTrue({
            "ck_teaching_projects_training_profile",
            "ck_teaching_projects_config_version_positive",
            "ck_teaching_projects_starter_code_size",
        }.issubset(checks))
        _, course, _, _, owner, _, _, _ = self.create_context("-migration")
        created = self.create_project(course, owner).project
        self.assertIsNotNone(created.created_at.utcoffset())
        rebuilt = TeachingProjectService(PostgreSQLTeachingProjectRepository(self.factory))
        self.assertEqual(
            rebuilt.get_project(course.course_id, created.project_id, owner.user_id).project,
            created,
        )

    def test_create_update_version_replay_disable_hash_and_json_isolation(self):
        _, course, _, _, owner, _, student, _ = self.create_context("-version")
        created = self.create_project(course, owner)
        self.assertEqual(created.project.config_version, 1)
        self.assertEqual(created.project.starter_code_sha256, starter_code_digest("# stored only\n"))
        created.project.parameter_schema["required"].append("unsafe")
        fresh = self.projects.get_project(course.course_id, created.project.project_id, owner.user_id)
        self.assertEqual(fresh.project.parameter_schema["required"], ["seed"])
        changed = self.projects.update_project(
            course.course_id, fresh.project.project_id, owner.user_id, 1,
            {"training_profile": "standard", "starter_code": "updated"},
        )
        self.assertEqual(changed.project.config_version, 2)
        self.assertEqual(changed.project.expected_training_steps, 20000)
        self.assertEqual(
            changed.project.evaluation_rubric["criteria"][1]["target"], 20000
        )
        replay = self.projects.update_project(
            course.course_id, fresh.project.project_id, owner.user_id, 1,
            {"training_profile": "standard", "starter_code": "updated"},
        )
        self.assertEqual(replay.project.config_version, 2)
        self.assertEqual(replay.project.updated_at, changed.project.updated_at)
        same = self.projects.update_project(
            course.course_id, fresh.project.project_id, owner.user_id, 2,
            {"training_profile": "standard"},
        )
        self.assertEqual(same.project.updated_at, changed.project.updated_at)
        with self.assertRaises(TeachingProjectVersionConflict):
            self.projects.update_project(
                course.course_id, fresh.project.project_id, owner.user_id, 1,
                {"title": "Different"},
            )
        disabled = self.projects.update_project(
            course.course_id, fresh.project.project_id, owner.user_id, 2,
            {"enabled": False},
        )
        self.assertEqual(disabled.project.config_version, 3)
        self.assertFalse(disabled.permissions.can_use)
        with self.assertRaises(TeachingProjectNotFound):
            self.projects.get_project(course.course_id, fresh.project.project_id, student.user_id)

    def test_unique_and_database_check_constraints(self):
        organization, course, _, admin, owner, _, _, _ = self.create_context("-constraints")
        created = self.create_project(course, owner)
        with self.assertRaises(TeachingProjectConflict):
            self.create_project(course, owner)
        other = self.courses.create_course(
            organization.organization_id, self.prefix + "-OTHER", "Other", "",
            owner.user_id, admin.user_id,
        )
        reused = self.create_project(other, admin)
        self.assertEqual(reused.project.title, created.project.title)
        with self.factory() as session:
            with self.assertRaises(IntegrityError):
                session.execute(
                    update(TeachingProjectRow)
                    .where(TeachingProjectRow.project_id == created.project.project_id)
                    .values(training_profile="unsafe")
                )
            session.rollback()
            with self.assertRaises(IntegrityError):
                session.execute(
                    update(TeachingProjectRow)
                    .where(TeachingProjectRow.project_id == created.project.project_id)
                    .values(config_version=0)
                )
            session.rollback()

    def test_rbac_and_disabled_resource_matrix(self):
        organization, course, teaching_class, admin, owner, assigned, student, outsider = self.create_context("-rbac")
        created = self.create_project(course, owner)
        project_id = created.project.project_id
        with self.assertRaises(TeachingProjectNotFound):
            self.projects.list_projects(course.course_id, student.user_id)
        self.courses.put_enrollment(teaching_class.class_id, student.user_id, True, assigned.user_id)
        student_view = self.projects.get_project(course.course_id, project_id, student.user_id)
        self.assertEqual(student_view.permissions, type(student_view.permissions)(True, True, False))
        assigned_view = self.projects.get_project(course.course_id, project_id, assigned.user_id)
        self.assertFalse(assigned_view.permissions.can_modify)
        with self.assertRaises(TeachingProjectPermissionDenied):
            self.projects.update_project(course.course_id, project_id, assigned.user_id, 1, {"title": "Denied"})
        self.assertTrue(self.projects.get_project(course.course_id, project_id, owner.user_id).permissions.can_modify)
        self.assertTrue(self.projects.get_project(course.course_id, project_id, admin.user_id).permissions.can_modify)
        with self.assertRaises(TeachingProjectNotFound):
            self.projects.get_project(course.course_id, project_id, outsider.user_id)
        self.courses.put_enrollment(teaching_class.class_id, student.user_id, False, assigned.user_id)
        with self.assertRaises(TeachingProjectNotFound):
            self.projects.get_project(course.course_id, project_id, student.user_id)
        self.courses.put_enrollment(teaching_class.class_id, student.user_id, True, assigned.user_id)
        self.courses.update_class(teaching_class.class_id, owner.user_id, {"enabled": False})
        with self.assertRaises(TeachingProjectNotFound):
            self.projects.get_project(course.course_id, project_id, assigned.user_id)
        with self.assertRaises(TeachingProjectNotFound):
            self.projects.get_project(course.course_id, project_id, student.user_id)
        self.assertTrue(self.projects.get_project(course.course_id, project_id, owner.user_id).permissions.can_modify)
        self.courses.update_class(teaching_class.class_id, owner.user_id, {"enabled": True})
        self.organizations.set_membership(organization.organization_id, student.user_id, OrganizationRole.STUDENT, False)
        with self.assertRaises(TeachingProjectNotFound):
            self.projects.get_project(course.course_id, project_id, student.user_id)
        self.organizations.set_membership(organization.organization_id, student.user_id, OrganizationRole.STUDENT, True)
        with self.factory() as session:
            session.execute(update(UserRow).where(UserRow.user_id == student.user_id).values(enabled=False))
            session.commit()
        with self.assertRaises(TeachingProjectNotFound):
            self.projects.get_project(course.course_id, project_id, student.user_id)
        with self.factory() as session:
            session.execute(update(UserRow).where(UserRow.user_id == student.user_id).values(enabled=True))
            session.execute(update(CourseRow).where(CourseRow.course_id == course.course_id).values(enabled=False))
            session.commit()
        with self.assertRaises(TeachingProjectNotFound):
            self.projects.get_project(course.course_id, project_id, admin.user_id)
        with self.factory() as session:
            session.execute(update(CourseRow).where(CourseRow.course_id == course.course_id).values(enabled=True))
            session.execute(update(OrganizationRow).where(OrganizationRow.organization_id == organization.organization_id).values(enabled=False))
            session.commit()
        with self.assertRaises(TeachingProjectNotFound):
            self.projects.get_project(course.course_id, project_id, owner.user_id)

    def test_api_contract_injection_rejection_version_error_and_workspace_compatibility(self):
        _, course, teaching_class, admin, owner, assigned, student, _ = self.create_context("-api")
        self.courses.put_enrollment(teaching_class.class_id, student.user_id, True, assigned.user_id)
        payload = {
            "title": "API S1 Push", "description": "", "instructions": "",
            "starter_code": "", "training_profile": "short",
            "video_enabled": True, "enabled": True,
        }
        path = "/api/v1/courses/%s/projects" % course.course_id
        with ASGITestClient(self.app_for(owner.user_id)) as client:
            created = client.post(path, json=payload)
            self.assertEqual(created.status_code, 201, created.text)
            body = created.json()
            self.assertEqual((body["task_type"], body["environment_type"], body["algorithm"]), ("training", "s1_push", "PPO"))
            self.assertEqual(body["expected_training_steps"], 32)
            self.assertEqual(body["permissions"], {"can_view": True, "can_use": True, "can_modify": True})
            project_id = body["project_id"]
            for forbidden in (
                {"task_type": "training"}, {"rollout_steps": 16},
                {"parameter_schema": parameter_schema()},
                {"evaluation_rubric": evaluation_rubric(TrainingProfile.SHORT)},
                {"num_envs": 1},
            ):
                invalid = dict(payload)
                invalid.update(forbidden)
                self.assertEqual(client.post(path, json=invalid).status_code, 422)
            conflict = client.request(
                "PATCH", path + "/" + project_id,
                json={"config_version": 999, "title": "Conflict"},
            )
            self.assertEqual(conflict.status_code, 409)
            self.assertEqual(conflict.json()["detail"]["code"], "project_version_conflict")
            injected = client.request(
                "PATCH", path + "/" + project_id,
                json={"config_version": 1, "video_width": 1920},
            )
            self.assertEqual(injected.status_code, 422)
            self.assertEqual(client.get("/api/v1/projects").json(), [])
            workspace = client.post("/api/v1/projects", json={"project_id": "workspace-one"})
            self.assertEqual(workspace.status_code, 201)
            listed = client.get("/api/v1/projects").json()
            self.assertEqual(listed[0]["project_id"], "workspace-one")
            self.assertEqual(set(listed[0]), {"project_id", "file_count", "workspace_status"})
        with ASGITestClient(self.app_for(student.user_id)) as client:
            detail = client.get(path + "/" + project_id)
            self.assertEqual(detail.status_code, 200)
            self.assertFalse(detail.json()["permissions"]["can_modify"])
            self.assertEqual(client.post(path, json=dict(payload, title="Student" )).status_code, 403)
        with ASGITestClient(self.app_for(assigned.user_id)) as client:
            self.assertEqual(client.request(
                "PATCH", path + "/" + project_id,
                json={"config_version": 1, "title": "Assigned"},
            ).status_code, 403)
        with ASGITestClient(self.app_for(admin.user_id)) as client:
            duplicate = client.post(path, json=payload)
            self.assertEqual(duplicate.status_code, 409)
            self.assertEqual(duplicate.json()["detail"]["code"], "project_conflict")
            multibyte = dict(payload, title="Too Large", starter_code="测" * 180000)
            self.assertEqual(client.post(path, json=multibyte).status_code, 422)


if __name__ == "__main__":
    unittest.main()
