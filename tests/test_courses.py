import os
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from alembic import command as alembic_command
from alembic.config import Config
from sqlalchemy import delete, func, inspect, select, update

from apps.api.database import create_postgres_engine, create_session_factory
from apps.api.main import create_app
from robotics.auth import AuthenticationService, PasswordHasher, PostgreSQLUserRepository
from robotics.courses import (
    Course,
    CourseService,
    Enrollment,
    PostgreSQLCourseRepository,
    TeachingClass,
    TeachingConflict,
    TeachingPermissionDenied,
    TeachingResourceNotFound,
    TeachingValidationError,
)
from robotics.organizations import (
    OrganizationRole,
    OrganizationService,
    PostgreSQLOrganizationRepository,
    permissions_for,
)
from robotics.runs import InMemoryRunRepository
from robotics.runs.db_models import (
    CourseRow,
    EnrollmentRow,
    OrganizationMembershipRow,
    OrganizationRow,
    TeachingClassRow,
    UserRow,
)
from tests.auth import StaticCurrentUserProvider
from tests.http_client import ASGITestClient
from tests.test_terminal_callback_api import SAFE_POSTGRES_URL


ROOT = Path(__file__).resolve().parents[1]


class CourseDomainTest(unittest.TestCase):
    def test_models_require_timezone_aware_datetimes(self):
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            Course("c", "o", "C", "Title", "", "u", True, datetime.now(), datetime.now())
        non_utc = datetime.now(timezone(timedelta(hours=8)))
        with self.assertRaisesRegex(ValueError, "must use UTC"):
            Course("c", "o", "C", "Title", "", "u", True, non_utc, non_utc)

    def test_permissions_match_implemented_course_and_class_capabilities(self):
        student = permissions_for(OrganizationRole.STUDENT)
        instructor = permissions_for(OrganizationRole.INSTRUCTOR)
        self.assertIn("course:read", student)
        self.assertIn("class:read", student)
        self.assertNotIn("enrollment:read", student)
        self.assertTrue({"course:manage", "class:manage", "enrollment:read", "enrollment:manage"}.issubset(instructor))


@unittest.skipUnless(SAFE_POSTGRES_URL is not None, "PostgreSQL test configuration is unavailable")
class CoursePostgreSQLTest(unittest.TestCase):
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
        self.prefix = "p73-%s" % uuid4().hex
        self.users = PostgreSQLUserRepository(self.factory)
        self.auth = AuthenticationService(self.users, PasswordHasher())
        self.organizations = OrganizationService(PostgreSQLOrganizationRepository(self.factory))
        self.repository = PostgreSQLCourseRepository(self.factory)
        self.service = CourseService(self.repository)
        self.user_ids = []
        self.organization_ids = []
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        with self.factory() as session:
            organization_ids = session.execute(
                select(OrganizationRow.organization_id).where(OrganizationRow.slug.like(self.prefix + "%"))
            ).scalars().all()
            course_ids = session.execute(
                select(CourseRow.course_id).where(CourseRow.organization_id.in_(organization_ids))
            ).scalars().all() if organization_ids else []
            class_ids = session.execute(
                select(TeachingClassRow.class_id).where(TeachingClassRow.course_id.in_(course_ids))
            ).scalars().all() if course_ids else []
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
        user = self.auth.create_user(self.prefix + suffix, "Phase 7.3 " + suffix, "safe-test-password")
        self.user_ids.append(user.user_id)
        return user

    def create_context(self, suffix=""):
        admin = self.create_user(suffix + "-admin")
        instructor = self.create_user(suffix + "-instructor")
        assigned = self.create_user(suffix + "-assigned")
        student = self.create_user(suffix + "-student")
        outsider = self.create_user(suffix + "-outsider")
        organization, _ = self.organizations.create_organization(
            self.prefix + suffix + "-org", "Phase 7.3 Organization", admin.user_id
        )
        self.organization_ids.append(organization.organization_id)
        self.organizations.add_membership(organization.organization_id, instructor.user_id, OrganizationRole.INSTRUCTOR)
        self.organizations.add_membership(organization.organization_id, assigned.user_id, OrganizationRole.INSTRUCTOR)
        self.organizations.add_membership(organization.organization_id, student.user_id, OrganizationRole.STUDENT)
        return organization, admin, instructor, assigned, student, outsider

    def create_course_and_class(self, organization, instructor, assigned, code="ROBOTICS"):
        course = self.service.create_course(
            organization.organization_id, self.prefix + "-" + code, "Robotics", "Teaching course", instructor.user_id, instructor.user_id
        )
        teaching_class = self.service.create_class(
            course.course_id, "A", "Class A", "2026-Fall", assigned.user_id, instructor.user_id
        )
        return course, teaching_class

    def app_for(self, user_id):
        root = Path(self.temp.name) / user_id
        return create_app(
            root / "runs",
            project_root=root / "projects",
            run_repository=InMemoryRunRepository(),
            current_user_provider=StaticCurrentUserProvider(user_id),
            organization_service=self.organizations,
            course_service=self.service,
        )

    def test_01_migration_round_trip_schema_constraints_indexes_and_utc(self):
        with self.factory() as session:
            for model in (EnrollmentRow, TeachingClassRow, CourseRow):
                self.assertEqual(session.scalar(select(func.count()).select_from(model)), 0)
        with patch.dict(os.environ, self.environment, clear=False):
            alembic_command.downgrade(self.config, "20260828_0005")
        self.assertFalse({"courses", "classes", "enrollments"} & set(inspect(self.engine).get_table_names()))
        with patch.dict(os.environ, self.environment, clear=False):
            alembic_command.upgrade(self.config, "head")
        inspector = inspect(self.engine)
        self.assertTrue({"courses", "classes", "enrollments"}.issubset(inspector.get_table_names()))
        self.assertIn({"organization_id", "code"}, [set(item["column_names"]) for item in inspector.get_unique_constraints("courses")])
        self.assertIn({"course_id", "term", "code"}, [set(item["column_names"]) for item in inspector.get_unique_constraints("classes")])
        self.assertIn({"class_id", "user_id"}, [set(item["column_names"]) for item in inspector.get_unique_constraints("enrollments")])
        self.assertGreaterEqual(len(inspector.get_indexes("courses")), 2)
        self.assertGreaterEqual(len(inspector.get_indexes("classes")), 2)
        self.assertGreaterEqual(len(inspector.get_indexes("enrollments")), 2)
        organization, _, instructor, assigned, student, _ = self.create_context("-schema")
        course, teaching_class = self.create_course_and_class(organization, instructor, assigned)
        enrollment = self.service.put_enrollment(teaching_class.class_id, student.user_id, True, assigned.user_id)
        for value in (course.created_at, teaching_class.updated_at, enrollment.created_at):
            self.assertIsNotNone(value.utcoffset())

    def test_course_class_uniques_reuse_and_atomic_rollback(self):
        organization, admin, instructor, assigned, _, _ = self.create_context("-one")
        course, teaching_class = self.create_course_and_class(organization, instructor, assigned)
        with self.assertRaises(TeachingConflict):
            self.service.create_course(organization.organization_id, course.code, "Duplicate", "", instructor.user_id, admin.user_id)
        with self.assertRaises(TeachingConflict):
            self.service.create_class(course.course_id, teaching_class.code, "Duplicate", teaching_class.term, assigned.user_id, instructor.user_id)
        other, _, other_instructor, other_assigned, _, _ = self.create_context("-two")
        reused = self.service.create_course(other.organization_id, course.code, "Reuse", "", other_instructor.user_id, other_instructor.user_id)
        self.assertEqual(reused.code, course.code)
        with self.assertRaises(TeachingResourceNotFound):
            self.service.create_class(course.course_id, "ROLLBACK", "Rollback", "2027", other_assigned.user_id, instructor.user_id)
        with self.factory() as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(TeachingClassRow).where(TeachingClassRow.code == "ROLLBACK")), 0)

    def test_visibility_matrix_cross_organization_and_safe_api(self):
        organization, admin, instructor, assigned, student, outsider = self.create_context("-matrix")
        course, teaching_class = self.create_course_and_class(organization, instructor, assigned)
        self.assertEqual(self.service.list_courses(organization.organization_id, student.user_id), [])
        self.service.put_enrollment(teaching_class.class_id, student.user_id, True, assigned.user_id)
        self.assertEqual([item.course_id for item in self.service.list_courses(organization.organization_id, student.user_id)], [course.course_id])
        self.assertEqual([item.course_id for item in self.service.list_courses(organization.organization_id, assigned.user_id)], [course.course_id])
        self.assertEqual([item.course_id for item in self.service.list_courses(organization.organization_id, admin.user_id)], [course.course_id])
        with self.assertRaises(TeachingResourceNotFound):
            self.service.list_courses(organization.organization_id, outsider.user_id)
        with ASGITestClient(self.app_for(student.user_id)) as client:
            listed = client.get("/api/v1/courses", params={"organization_id": organization.organization_id})
            self.assertEqual(listed.status_code, 200)
            self.assertNotIn("password_hash", listed.text)
            self.assertEqual(client.get("/api/v1/classes/%s/enrollments" % teaching_class.class_id).status_code, 403)
            self.assertEqual(client.get("/api/v1/auth/me").status_code, 200)
            permissions = client.get("/api/v1/me").json()["organizations"][0]["permissions"]
            self.assertIn("course:read", permissions)
            self.assertNotIn("enrollment:read", permissions)
        with ASGITestClient(self.app_for(outsider.user_id)) as client:
            self.assertEqual(client.get("/api/v1/courses", params={"organization_id": organization.organization_id}).status_code, 404)
            self.assertEqual(client.get("/api/v1/courses/%s" % course.course_id).status_code, 404)

    def test_enrollment_add_disable_enable_idempotency_and_target_validation(self):
        organization, _, instructor, assigned, student, outsider = self.create_context("-enroll")
        course, teaching_class = self.create_course_and_class(organization, instructor, assigned)
        first = self.service.put_enrollment(teaching_class.class_id, student.user_id, True, assigned.user_id)
        replay = self.service.put_enrollment(teaching_class.class_id, student.user_id, True, assigned.user_id)
        self.assertEqual(first.enrollment_id, replay.enrollment_id)
        self.assertEqual(first.updated_at, replay.updated_at)
        disabled = self.service.put_enrollment(teaching_class.class_id, student.user_id, False, instructor.user_id)
        self.assertFalse(disabled.enabled)
        enabled = self.service.put_enrollment(teaching_class.class_id, student.user_id, True, instructor.user_id)
        self.assertTrue(enabled.enabled)
        with self.assertRaises(TeachingValidationError):
            self.service.put_enrollment(teaching_class.class_id, assigned.user_id, True, instructor.user_id)
        with self.assertRaises(TeachingResourceNotFound):
            self.service.put_enrollment(teaching_class.class_id, outsider.user_id, True, instructor.user_id)
        roster = self.service.list_enrollments(teaching_class.class_id, assigned.user_id)
        self.assertEqual([item.user_id for item in roster], [student.user_id])

    def test_current_membership_and_disabled_resource_behavior(self):
        organization, _, instructor, assigned, student, _ = self.create_context("-state")
        course, teaching_class = self.create_course_and_class(organization, instructor, assigned)
        self.service.put_enrollment(teaching_class.class_id, student.user_id, True, assigned.user_id)
        self.organizations.set_membership(organization.organization_id, student.user_id, OrganizationRole.STUDENT, False)
        with self.assertRaises(TeachingResourceNotFound):
            self.service.get_course(course.course_id, student.user_id)
        self.organizations.set_membership(organization.organization_id, student.user_id, OrganizationRole.STUDENT, True)
        self.organizations.set_membership(organization.organization_id, assigned.user_id, OrganizationRole.INSTRUCTOR, False)
        with self.assertRaises(TeachingResourceNotFound):
            self.service.list_enrollments(teaching_class.class_id, assigned.user_id)
        changed = self.service.update_course(course.course_id, instructor.user_id, {"enabled": False})
        self.assertFalse(changed.enabled)
        self.assertEqual(self.service.list_courses(organization.organization_id, student.user_id), [])
        with self.assertRaises(TeachingValidationError):
            self.service.create_class(course.course_id, "B", "Class B", "2027", instructor.user_id, instructor.user_id)
        self.service.update_course(course.course_id, instructor.user_id, {"enabled": True})
        self.service.update_class(teaching_class.class_id, instructor.user_id, {"enabled": False})
        with self.assertRaises(TeachingValidationError):
            self.service.put_enrollment(teaching_class.class_id, student.user_id, True, instructor.user_id)
        self.service.put_enrollment(teaching_class.class_id, student.user_id, False, instructor.user_id)
        with self.factory() as session:
            session.execute(update(UserRow).where(UserRow.user_id == student.user_id).values(enabled=False))
            session.commit()
        with self.assertRaises(TeachingResourceNotFound):
            self.service.put_enrollment(teaching_class.class_id, student.user_id, True, instructor.user_id)
        with self.factory() as session:
            session.execute(update(UserRow).where(UserRow.user_id == student.user_id).values(enabled=True))
            session.execute(update(OrganizationRow).where(OrganizationRow.organization_id == organization.organization_id).values(enabled=False))
            session.commit()
        with self.assertRaises(TeachingResourceNotFound):
            self.service.get_course(course.course_id, instructor.user_id)

    def test_owner_transfer_class_adjustment_permissions_and_strict_patch(self):
        organization, admin, instructor, assigned, _, _ = self.create_context("-patch")
        course, teaching_class = self.create_course_and_class(organization, instructor, assigned)
        with self.assertRaises(TeachingPermissionDenied):
            self.service.update_course(course.course_id, instructor.user_id, {"owner_instructor_user_id": assigned.user_id})
        transferred = self.service.update_course(course.course_id, admin.user_id, {"owner_instructor_user_id": assigned.user_id})
        self.assertEqual(transferred.owner_instructor_user_id, assigned.user_id)
        with self.assertRaises(TeachingPermissionDenied):
            self.service.update_class(teaching_class.class_id, instructor.user_id, {"instructor_user_id": instructor.user_id})
        adjusted = self.service.update_class(teaching_class.class_id, assigned.user_id, {"instructor_user_id": instructor.user_id})
        self.assertEqual(adjusted.instructor_user_id, instructor.user_id)
        with ASGITestClient(self.app_for(admin.user_id)) as client:
            self.assertEqual(client.request("PATCH", "/api/v1/courses/%s" % course.course_id, json={"organization_id": organization.organization_id}).status_code, 422)
            self.assertEqual(client.request("PATCH", "/api/v1/courses/%s" % course.course_id, json={"created_at": "2026-01-01T00:00:00Z"}).status_code, 422)

    def test_api_create_conflicts_role_rules_and_app_rebuild_recovery(self):
        organization, admin, instructor, assigned, student, _ = self.create_context("-api")
        payload = {
            "organization_id": organization.organization_id,
            "code": self.prefix + "-API",
            "title": "API Course",
            "description": "",
            "owner_instructor_user_id": instructor.user_id,
        }
        with ASGITestClient(self.app_for(instructor.user_id)) as client:
            forbidden = dict(payload, owner_instructor_user_id=assigned.user_id)
            self.assertEqual(client.post("/api/v1/courses", json=forbidden).status_code, 403)
            payload["owner_instructor_user_id"] = instructor.user_id
            own = dict(payload, code=self.prefix + "-OWN")
            self.assertEqual(client.post("/api/v1/courses", json=own).status_code, 201)
        with ASGITestClient(self.app_for(admin.user_id)) as client:
            created = client.post("/api/v1/courses", json=payload)
            self.assertEqual(created.status_code, 201)
            self.assertEqual(client.post("/api/v1/courses", json=payload).status_code, 409)
            course_id = created.json()["course_id"]
            class_response = client.post(
                "/api/v1/courses/%s/classes" % course_id,
                json={"code": "A", "name": "API Class", "term": "2026", "instructor_user_id": assigned.user_id},
            )
            self.assertEqual(class_response.status_code, 201)
            class_id = class_response.json()["class_id"]
            added = client.put("/api/v1/classes/%s/enrollments/%s" % (class_id, student.user_id), json={"enabled": True})
            self.assertEqual(added.status_code, 200)
        rebuilt = CourseService(PostgreSQLCourseRepository(self.factory))
        self.assertEqual(rebuilt.get_course(course_id, admin.user_id).course_id, course_id)
        self.assertEqual(rebuilt.get_class(class_id, admin.user_id).class_id, class_id)
        self.assertEqual(rebuilt.list_enrollments(class_id, admin.user_id)[0].user_id, student.user_id)


if __name__ == "__main__":
    unittest.main()
