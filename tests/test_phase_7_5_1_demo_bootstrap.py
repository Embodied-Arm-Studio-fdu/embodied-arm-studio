import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from uuid import uuid4

from sqlalchemy import delete, select, update

from apps.api.database import create_postgres_engine, create_session_factory
from apps.api.main import create_app
from robotics.auth import AuthenticationService, PasswordHasher, PostgreSQLUserRepository
from robotics.projects import (
    ALGORITHM,
    ENVIRONMENT_TYPE,
    TASK_TYPE,
    TrainingProfile,
    evaluation_rubric,
    parameter_schema,
    profile_config,
)
from robotics.runs.db_models import (
    CourseRow,
    EnrollmentRow,
    OrganizationMembershipRow,
    OrganizationRow,
    TeachingClassRow,
    TeachingProjectRow,
    UserRow,
)
from scripts.bootstrap_midterm_demo import (
    BootstrapConflict,
    MidtermDemoBootstrap,
    _parser,
    demo_names,
)
from scripts.check_midterm_runtime import Reporter, check_demo_namespace
from tests.auth import StaticCurrentUserProvider
from tests.http_client import ASGITestClient
from tests.test_projects import SAFE_POSTGRES_URL


class MidtermDemoCliContractTest(unittest.TestCase):
    def test_default_mode_is_preview_and_apply_must_be_explicit(self):
        args = _parser().parse_args(["--namespace", "midterm-demo"])
        self.assertFalse(args.apply)
        self.assertFalse(args.dry_run)
        self.assertTrue(
            _parser().parse_args(
                ["--namespace", "midterm-demo", "--apply"]
            ).apply
        )

    def test_existing_standard_v3_project_is_promoted_to_video_v4(self):
        legacy_schema = parameter_schema()
        legacy_schema.pop("x-environment-type")
        legacy = SimpleNamespace(
            project_id="standard-project",
            title="Standard",
            description="Canonical S1 Push standard training profile.",
            instructions="Set only seed and inspect real training evidence.",
            starter_code="",
            training_profile=TrainingProfile.STANDARD,
            environment_type="s1_push",
            video_enabled=False,
            parameter_schema=legacy_schema,
            evaluation_rubric=evaluation_rubric(TrainingProfile.STANDARD, 3),
            config_version=3,
            enabled=True,
            created_by_user_id="owner",
        )
        upgraded = SimpleNamespace(
            **{
                **legacy.__dict__,
                "video_enabled": True,
                "parameter_schema": parameter_schema(),
                "evaluation_rubric": evaluation_rubric(
                    TrainingProfile.STANDARD, 4
                ),
                "config_version": 4,
            }
        )

        class ProjectService:
            def list_projects(self, _course_id, _actor_user_id):
                return [SimpleNamespace(project=legacy)]

            def update_project(
                self, course_id, project_id, actor_user_id, version, changes
            ):
                self.call = (
                    course_id, project_id, actor_user_id, version, changes
                )
                return SimpleNamespace(project=upgraded)

        bootstrap = object.__new__(MidtermDemoBootstrap)
        bootstrap.projects = ProjectService()
        bootstrap.actions = []
        bootstrap.output = io.StringIO()
        result = bootstrap._ensure_project(
            "course",
            "owner",
            "Standard",
            TrainingProfile.STANDARD,
            True,
        )
        self.assertIs(result, upgraded)
        self.assertEqual(
            bootstrap.projects.call,
            (
                "course", "standard-project", "owner", 3,
                {"training_profile": "standard", "video_enabled": True},
            ),
        )


@unittest.skipUnless(
    SAFE_POSTGRES_URL is not None,
    "PostgreSQL test configuration is unavailable",
)
class MidtermDemoBootstrapPostgreSQLTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "test" not in (SAFE_POSTGRES_URL.database or "").lower():
            raise AssertionError("bootstrap tests require a test-named database")
        cls.engine = create_postgres_engine(SAFE_POSTGRES_URL)
        cls.factory = create_session_factory(cls.engine)

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def setUp(self):
        self.namespace = "p751-%s" % uuid4().hex[:12]
        self.names = demo_names(self.namespace)
        self.clients = []

    def tearDown(self):
        for client in reversed(self.clients):
            client.close()
        self._cleanup_namespace()

    def _cleanup_namespace(self):
        usernames = (
            self.names.administrator_username,
            self.names.instructor_username,
            self.names.student_username,
        )
        with self.factory() as session:
            organization_ids = select(OrganizationRow.organization_id).where(
                OrganizationRow.slug == self.names.organization_slug
            )
            course_ids = select(CourseRow.course_id).where(
                CourseRow.organization_id.in_(organization_ids)
            )
            class_ids = select(TeachingClassRow.class_id).where(
                TeachingClassRow.course_id.in_(course_ids)
            )
            session.execute(
                delete(TeachingProjectRow).where(
                    TeachingProjectRow.course_id.in_(course_ids)
                )
            )
            session.execute(
                delete(EnrollmentRow).where(EnrollmentRow.class_id.in_(class_ids))
            )
            session.execute(
                delete(TeachingClassRow).where(
                    TeachingClassRow.course_id.in_(course_ids)
                )
            )
            session.execute(
                delete(CourseRow).where(
                    CourseRow.organization_id.in_(organization_ids)
                )
            )
            session.execute(
                delete(OrganizationMembershipRow).where(
                    OrganizationMembershipRow.organization_id.in_(organization_ids)
                )
            )
            session.execute(
                delete(OrganizationRow).where(
                    OrganizationRow.slug == self.names.organization_slug
                )
            )
            session.execute(delete(UserRow).where(UserRow.username.in_(usernames)))
            session.commit()

    @staticmethod
    def _password_reader(values):
        iterator = iter(values)
        return lambda _prompt: next(iterator)

    def _apply(self, output=None, password_values=None):
        stream = output or io.StringIO()
        values = password_values or [
            "administrator-test-password",
            "administrator-test-password",
            "instructor-test-password",
            "instructor-test-password",
            "student-test-password",
            "student-test-password",
        ]
        bootstrap = MidtermDemoBootstrap(
            self.factory,
            self.namespace,
            password_reader=self._password_reader(values),
            output=stream,
        )
        return bootstrap.run(apply=True), stream.getvalue()

    def test_default_preview_is_zero_write_and_never_prompts(self):
        output = io.StringIO()
        bootstrap = MidtermDemoBootstrap(
            self.factory,
            self.namespace,
            password_reader=lambda _prompt: self.fail("dry-run requested a password"),
            output=output,
        )
        summary = bootstrap.run(apply=False)
        with self.factory() as session:
            self.assertIsNone(
                session.scalar(
                    select(UserRow).where(
                        UserRow.username == self.names.administrator_username
                    )
                )
            )
            self.assertIsNone(
                session.scalar(
                    select(OrganizationRow).where(
                        OrganizationRow.slug == self.names.organization_slug
                    )
                )
            )
        self.assertEqual(
            [(item["profile"], item["version"]) for item in summary.project_facts],
            [("short", 1)] + [("standard", 4)] * 5,
        )
        self.assertIn("dry-run (zero writes)", output.getvalue())

    def test_apply_creates_complete_facts_then_reuses_without_password_reset(self):
        secret_values = [
            "administrator-test-password",
            "instructor-test-password",
            "student-test-password",
        ]
        preview = MidtermDemoBootstrap(
            self.factory,
            self.namespace,
            password_reader=lambda _prompt: self.fail("dry-run requested a password"),
            output=io.StringIO(),
        ).run(apply=False)
        summary, output = self._apply()
        self.assertEqual(preview.project_facts, summary.project_facts)
        self.assertEqual(len(summary.project_facts), 6)
        for secret in secret_values:
            self.assertNotIn(secret, output)
        for forbidden in ("password_hash", "database_url", "jwt", "secret"):
            self.assertNotIn(forbidden, output.lower())

        with self.factory() as session:
            users = {
                row.username: row
                for row in session.execute(
                    select(UserRow).where(
                        UserRow.username.in_(
                            [
                                self.names.administrator_username,
                                self.names.instructor_username,
                                self.names.student_username,
                            ]
                        )
                    )
                ).scalars()
            }
            hashes = {name: row.password_hash for name, row in users.items()}
            organization = session.scalar(
                select(OrganizationRow).where(
                    OrganizationRow.slug == self.names.organization_slug
                )
            )
            memberships = session.execute(
                select(OrganizationMembershipRow).where(
                    OrganizationMembershipRow.organization_id
                    == organization.organization_id
                )
            ).scalars().all()
            course = session.scalar(
                select(CourseRow).where(CourseRow.code == self.names.course_code)
            )
            teaching_class = session.scalar(
                select(TeachingClassRow).where(
                    TeachingClassRow.code == self.names.class_code
                )
            )
            enrollment = session.scalar(
                select(EnrollmentRow).where(
                    EnrollmentRow.class_id == teaching_class.class_id,
                    EnrollmentRow.user_id
                    == users[self.names.student_username].user_id,
                )
            )
            projects = session.execute(
                select(TeachingProjectRow).where(
                    TeachingProjectRow.course_id == course.course_id
                )
            ).scalars().all()
            self.assertEqual(
                {row.role for row in memberships},
                {"administrator", "instructor", "student"},
            )
            self.assertTrue(all(row.enabled for row in memberships))
            self.assertEqual(
                course.owner_instructor_user_id,
                users[self.names.instructor_username].user_id,
            )
            self.assertEqual(
                teaching_class.instructor_user_id,
                users[self.names.instructor_username].user_id,
            )
            self.assertTrue(enrollment.enabled)
            by_title = {row.title: row for row in projects}
            self.assertEqual(set(by_title), {
                self.names.short_project_title, self.names.standard_project_title,
                self.names.reach_project_title,
                self.names.grasp_project_title, self.names.lift_project_title,
                self.names.lower_release_project_title,
            })
            self.assertFalse(by_title[self.names.short_project_title].video_enabled)
            self.assertTrue(by_title[self.names.standard_project_title].video_enabled)
            self.assertTrue(by_title[self.names.reach_project_title].video_enabled)
            self.assertTrue(by_title[self.names.grasp_project_title].video_enabled)
            self.assertTrue(by_title[self.names.lift_project_title].video_enabled)
            self.assertTrue(by_title[self.names.lower_release_project_title].video_enabled)
            for expected_title, profile, environment_type in (
                (self.names.short_project_title, TrainingProfile.SHORT, "s1_push"),
                (self.names.standard_project_title, TrainingProfile.STANDARD, "s1_push"),
                (self.names.reach_project_title, TrainingProfile.STANDARD, "s1_reach"),
                (self.names.grasp_project_title, TrainingProfile.STANDARD, "s1_grasp"),
                (self.names.lift_project_title, TrainingProfile.STANDARD, "s1_lift"),
                (self.names.lower_release_project_title, TrainingProfile.STANDARD, "s1_lower_release"),
            ):
                row = by_title[expected_title]
                self.assertEqual(row.course_id, course.course_id)
                self.assertEqual(row.title, expected_title)
                expected_version = 1 if profile is TrainingProfile.SHORT else 4
                self.assertEqual(row.config_version, expected_version)
                self.assertTrue(row.enabled)
                self.assertEqual(
                    row.created_by_user_id,
                    users[self.names.administrator_username].user_id,
                )
                self.assertEqual(
                    profile_config(profile, expected_version)["expected_training_steps"],
                    32 if profile is TrainingProfile.SHORT else 20000,
                )
                self.assertEqual(row.parameter_schema, parameter_schema(environment_type))
                self.assertEqual(
                    row.evaluation_rubric,
                    evaluation_rubric(profile, expected_version, environment_type),
                )

        database_report = Reporter(io.StringIO())
        with self.engine.connect() as connection:
            self.assertTrue(
                check_demo_namespace(connection, self.namespace, database_report)
            )
        self.assertEqual(database_report.failures, 0)

        second = MidtermDemoBootstrap(
            self.factory,
            self.namespace,
            password_reader=lambda _prompt: self.fail("replay requested a password"),
            output=io.StringIO(),
        )
        second_summary = second.run(apply=True)
        self.assertEqual(len(second_summary.project_facts), 6)
        with self.factory() as session:
            self.assertEqual(
                session.scalar(
                    select(UserRow.password_hash).where(
                        UserRow.username == self.names.administrator_username
                    )
                ),
                hashes[self.names.administrator_username],
            )
            self.assertEqual(
                len(
                    session.execute(
                        select(TeachingProjectRow).where(
                            TeachingProjectRow.course_id == course.course_id
                        )
                    ).scalars().all()
                ),
                6,
            )

        root = Path("/tmp") / self.namespace
        app = create_app(
            root / "runs",
            project_root=root / "projects",
            run_repository_mode="postgresql",
            database_url=SAFE_POSTGRES_URL,
            current_user_provider=StaticCurrentUserProvider(
                users[self.names.student_username].user_id
            ),
        )
        client = ASGITestClient(app)
        self.clients.append(client)
        response = client.get(
            "/api/v1/courses/%s/projects" % course.course_id
        )
        self.assertEqual(response.status_code, 200, response.text)
        public_projects = {item["title"]: item for item in response.json()}
        self.assertEqual(set(public_projects), {
            self.names.short_project_title, self.names.standard_project_title,
            self.names.reach_project_title,
            self.names.grasp_project_title, self.names.lift_project_title,
            self.names.lower_release_project_title,
        })
        for title, profile, environment_type, video_enabled, expected_steps in (
            (self.names.short_project_title, TrainingProfile.SHORT, "s1_push", False, 32),
            (self.names.standard_project_title, TrainingProfile.STANDARD, "s1_push", True, 20000),
            (self.names.reach_project_title, TrainingProfile.STANDARD, "s1_reach", True, 20000),
            (self.names.grasp_project_title, TrainingProfile.STANDARD, "s1_grasp", True, 20000),
            (self.names.lift_project_title, TrainingProfile.STANDARD, "s1_lift", True, 20000),
            (self.names.lower_release_project_title, TrainingProfile.STANDARD, "s1_lower_release", True, 20000),
        ):
            item = public_projects[title]
            self.assertEqual(item["course_id"], course.course_id)
            self.assertEqual(item["task_type"], TASK_TYPE)
            self.assertEqual(item["environment_type"], environment_type)
            self.assertEqual(item["algorithm"], ALGORITHM)
            expected_version = 1 if profile is TrainingProfile.SHORT else 4
            self.assertEqual(item["config_version"], expected_version)
            self.assertEqual(item["video_enabled"], video_enabled)
            self.assertEqual(item["expected_training_steps"], expected_steps)
            self.assertEqual(item["parameter_schema"], parameter_schema(environment_type))
            self.assertEqual(
                item["evaluation_rubric"],
                evaluation_rubric(profile, expected_version, environment_type),
            )

    def test_database_preflight_rejects_tampered_project_json(self):
        self._apply()
        with self.factory() as session:
            session.execute(
                update(TeachingProjectRow)
                .where(
                    TeachingProjectRow.course_id
                    == select(CourseRow.course_id)
                    .where(CourseRow.code == self.names.course_code)
                    .scalar_subquery(),
                    TeachingProjectRow.training_profile == "short",
                )
                .values(parameter_schema={"type": "object"})
            )
            session.commit()

        reporter = Reporter(io.StringIO())
        with self.engine.connect() as connection:
            self.assertFalse(
                check_demo_namespace(connection, self.namespace, reporter)
            )
        self.assertGreater(reporter.failures, 0)

    def test_partial_user_is_reused_and_missing_facts_are_completed(self):
        repository = PostgreSQLUserRepository(self.factory)
        service = AuthenticationService(repository, PasswordHasher())
        expected_name = "Midterm Demo Administrator [%s]" % self.namespace
        existing = service.create_user(
            self.names.administrator_username,
            expected_name,
            "existing-password-must-remain",
        )
        original_hash = existing.password_hash
        values = [
            "instructor-test-password",
            "instructor-test-password",
            "student-test-password",
            "student-test-password",
        ]
        self._apply(password_values=values)
        self.assertEqual(
            repository.get_by_username(
                self.names.administrator_username
            ).password_hash,
            original_hash,
        )

    def test_conflicting_or_disabled_existing_fact_is_not_overwritten(self):
        repository = PostgreSQLUserRepository(self.factory)
        service = AuthenticationService(repository, PasswordHasher())
        existing = service.create_user(
            self.names.administrator_username,
            "Conflicting Display Name",
            "conflicting-password",
        )
        with self.factory() as session:
            session.execute(
                update(UserRow)
                .where(UserRow.user_id == existing.user_id)
                .values(enabled=False)
            )
            session.commit()
        bootstrap = MidtermDemoBootstrap(
            self.factory,
            self.namespace,
            password_reader=lambda _prompt: self.fail("conflict requested a password"),
            output=io.StringIO(),
        )
        with self.assertRaises(BootstrapConflict):
            bootstrap.run(apply=True)
        current = repository.get_by_username(self.names.administrator_username)
        self.assertFalse(current.enabled)
        self.assertEqual(current.display_name, "Conflicting Display Name")
        self.assertEqual(current.password_hash, existing.password_hash)


if __name__ == "__main__":
    unittest.main()
