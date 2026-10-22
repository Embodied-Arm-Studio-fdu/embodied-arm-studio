import os
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from alembic import command as alembic_command
from alembic.config import Config
from sqlalchemy import delete, func, inspect, select, update

from apps.api.database import create_postgres_engine, create_session_factory
from apps.api.main import create_app
from robotics.auth import (
    AuthenticationService,
    JWTConfiguration,
    PasswordHasher,
    PostgreSQLUserRepository,
    User,
)
from robotics.auth.models import utc_now as user_utc_now
from robotics.auth.tokens import JWTAccessTokenService
from robotics.organizations import (
    LastAdministratorError,
    MembershipNotFound,
    OrganizationConflict,
    OrganizationContext,
    OrganizationMembership,
    OrganizationNotFound,
    OrganizationRole,
    OrganizationService,
    PostgreSQLOrganizationRepository,
    permissions_for,
)
from robotics.runs import InMemoryRunRepository
from robotics.runs.db_models import (
    OrganizationMembershipRow,
    OrganizationRow,
    UserRow,
)
from scripts import create_organization, set_organization_membership
from tests.auth import StaticCurrentUserProvider
from tests.http_client import ASGITestClient
from tests.test_terminal_callback_api import SAFE_POSTGRES_URL


ROOT = Path(__file__).resolve().parents[1]
TEST_SECRET = "phase-7-2-test-secret-with-32-bytes-minimum"


class MemoryOrganizationRepository:
    def __init__(self):
        self.organizations = {}
        self.memberships = {}
        self.lock = threading.Lock()

    def create_with_administrator(self, organization, membership):
        with self.lock:
            if any(
                item.slug == organization.slug
                for item in self.organizations.values()
            ):
                raise OrganizationConflict("slug already exists")
            self.organizations[organization.organization_id] = organization
            self.memberships[
                (membership.organization_id, membership.user_id)
            ] = membership
        return organization, membership

    def get_by_id(self, organization_id):
        try:
            return self.organizations[organization_id]
        except KeyError as exc:
            raise OrganizationNotFound("organization not found") from exc

    def get_by_slug(self, slug):
        for organization in self.organizations.values():
            if organization.slug == slug:
                return organization
        raise OrganizationNotFound("organization not found")

    def find_by_slug(self, slug):
        try:
            return self.get_by_slug(slug)
        except OrganizationNotFound:
            return None

    def get_membership(self, organization_id, user_id):
        try:
            return self.memberships[(organization_id, user_id)]
        except KeyError as exc:
            raise MembershipNotFound("membership not found") from exc

    def list_enabled_for_user(self, user_id):
        contexts = []
        for (organization_id, member_user_id), membership in self.memberships.items():
            organization = self.organizations[organization_id]
            if (
                member_user_id == user_id
                and membership.enabled
                and organization.enabled
            ):
                contexts.append(OrganizationContext(organization, membership))
        return sorted(contexts, key=lambda item: item.organization.slug)

    def get_enabled_context(self, organization_id, user_id):
        organization = self.get_by_id(organization_id)
        try:
            membership = self.get_membership(organization_id, user_id)
        except MembershipNotFound as exc:
            raise OrganizationNotFound("organization not found") from exc
        if not organization.enabled or not membership.enabled:
            raise OrganizationNotFound("organization not found")
        return OrganizationContext(organization, membership)

    def add_membership(self, membership):
        key = (membership.organization_id, membership.user_id)
        current = self.memberships.get(key)
        if current is not None:
            if current.role == membership.role and current.enabled == membership.enabled:
                return current
            raise OrganizationConflict("different membership exists")
        self.get_by_id(membership.organization_id)
        self.memberships[key] = membership
        return membership

    def set_membership(self, organization_id, user_id, role, enabled):
        key = (organization_id, user_id)
        current = self.get_membership(organization_id, user_id)
        role = OrganizationRole(role)
        if current.role == role and current.enabled == enabled:
            return current
        if (
            current.enabled
            and current.role == OrganizationRole.ADMINISTRATOR
            and (not enabled or role != OrganizationRole.ADMINISTRATOR)
        ):
            administrators = sum(
                item.enabled
                and item.role == OrganizationRole.ADMINISTRATOR
                and item.organization_id == organization_id
                for item in self.memberships.values()
            )
            if administrators <= 1:
                raise LastAdministratorError("last administrator")
        changed = replace(
            current, role=role, enabled=enabled, updated_at=user_utc_now()
        )
        self.memberships[key] = changed
        return changed


class OrganizationUnitAndApiTest(unittest.TestCase):
    def setUp(self):
        self.repository = MemoryOrganizationRepository()
        self.service = OrganizationService(self.repository)
        self.organization, self.administrator = self.service.create_organization(
            "robotics-school", "Robotics School", "admin-user"
        )
        self.student = self.service.add_membership(
            self.organization.organization_id,
            "student-user",
            OrganizationRole.STUDENT,
        )
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()

    def app_for(self, user_id):
        root = Path(self.temp.name) / user_id
        return create_app(
            root / "runs",
            project_root=root / "projects",
            run_repository=InMemoryRunRepository(),
            current_user_provider=StaticCurrentUserProvider(user_id),
            organization_service=self.service,
        )

    def test_roles_serialize_stably_and_permissions_are_centralized(self):
        self.assertEqual(
            [role.value for role in OrganizationRole],
            ["student", "instructor", "administrator"],
        )
        self.assertIn("own_run:create", permissions_for(OrganizationRole.STUDENT))
        self.assertTrue(
            permissions_for(OrganizationRole.STUDENT).issubset(
                permissions_for(OrganizationRole.INSTRUCTOR)
            )
        )
        self.assertTrue(
            permissions_for(OrganizationRole.INSTRUCTOR).issubset(
                permissions_for(OrganizationRole.ADMINISTRATOR)
            )
        )

    def test_create_add_idempotency_conflict_and_multiple_organizations(self):
        same = self.service.create_organization(
            "robotics-school", "Robotics School", "admin-user"
        )
        self.assertEqual(same[0], self.organization)
        with self.assertRaises(OrganizationConflict):
            self.service.create_organization(
                "robotics-school", "Changed Name", "admin-user"
            )
        repeated = self.service.add_membership(
            self.organization.organization_id,
            "student-user",
            OrganizationRole.STUDENT,
        )
        self.assertEqual(repeated.membership_id, self.student.membership_id)
        with self.assertRaises(OrganizationConflict):
            self.service.add_membership(
                self.organization.organization_id,
                "student-user",
                OrganizationRole.INSTRUCTOR,
            )
        second, _ = self.service.create_organization(
            "ai-school", "AI School", "second-admin"
        )
        self.service.add_membership(
            second.organization_id, "student-user", OrganizationRole.INSTRUCTOR
        )
        self.assertEqual(
            len(self.service.list_enabled_for_user("student-user")), 2
        )

    def test_membership_enable_disable_role_and_last_administrator(self):
        disabled = self.service.set_membership(
            self.organization.organization_id,
            "student-user",
            OrganizationRole.STUDENT,
            False,
        )
        self.assertFalse(disabled.enabled)
        enabled = self.service.set_membership(
            self.organization.organization_id,
            "student-user",
            OrganizationRole.INSTRUCTOR,
            True,
        )
        self.assertEqual(enabled.role, OrganizationRole.INSTRUCTOR)
        with self.assertRaises(LastAdministratorError):
            self.service.set_membership(
                self.organization.organization_id,
                "admin-user",
                OrganizationRole.STUDENT,
                True,
            )
        with self.assertRaises(LastAdministratorError):
            self.service.set_membership(
                self.organization.organization_id,
                "admin-user",
                OrganizationRole.ADMINISTRATOR,
                False,
            )

    def test_me_and_organizations_scope_compatibility_and_sensitive_fields(self):
        with ASGITestClient(self.app_for("student-user")) as client:
            me = client.get("/api/v1/me")
            self.assertEqual(me.status_code, 200)
            body = me.json()
            self.assertEqual(body["user_id"], "student-user")
            self.assertEqual(len(body["organizations"]), 1)
            context = body["organizations"][0]
            self.assertEqual(context["role"], "student")
            self.assertIn("own_run:create", context["permissions"])
            self.assertNotIn("password_hash", str(body))
            self.assertNotIn("JWT", str(body))
            listed = client.get("/api/v1/organizations")
            self.assertEqual(listed.json(), body["organizations"])
            detail = client.get(
                "/api/v1/organizations/%s"
                % self.organization.organization_id
            )
            self.assertEqual(detail.json(), context)
            legacy = client.get("/api/v1/auth/me")
            self.assertEqual(
                set(legacy.json()),
                {
                    "user_id",
                    "username",
                    "display_name",
                    "enabled",
                    "created_at",
                    "updated_at",
                },
            )

    def test_empty_and_unknown_cross_user_disabled_are_hidden(self):
        with ASGITestClient(self.app_for("outsider")) as client:
            self.assertEqual(client.get("/api/v1/organizations").json(), [])
            self.assertEqual(client.get("/api/v1/me").json()["organizations"], [])
            self.assertEqual(
                client.get(
                    "/api/v1/organizations/%s"
                    % self.organization.organization_id
                ).status_code,
                404,
            )
            self.assertEqual(
                client.get("/api/v1/organizations/unknown").status_code, 404
            )
        self.repository.organizations[self.organization.organization_id] = replace(
            self.organization, enabled=False, updated_at=user_utc_now()
        )
        with ASGITestClient(self.app_for("student-user")) as client:
            self.assertEqual(client.get("/api/v1/organizations").json(), [])
            self.assertEqual(
                client.get(
                    "/api/v1/organizations/%s"
                    % self.organization.organization_id
                ).status_code,
                404,
            )

    def test_missing_or_illegal_jwt_and_disabled_user_fail_closed(self):
        with ASGITestClient(self.app_for("student-user")) as client:
            for header in ("", "Bearer illegal"):
                response = client.get(
                    "/api/v1/me", headers={"Authorization": header}
                )
                self.assertEqual(response.status_code, 401)
        provider = StaticCurrentUserProvider("student-user", enabled=False)
        root = Path(self.temp.name) / "disabled"
        app = create_app(
            root / "runs",
            project_root=root / "projects",
            run_repository=InMemoryRunRepository(),
            current_user_provider=provider,
            organization_service=self.service,
        )
        with ASGITestClient(app) as client:
            self.assertEqual(client.get("/api/v1/me").status_code, 401)


@unittest.skipUnless(
    SAFE_POSTGRES_URL is not None,
    "PostgreSQL test configuration is unavailable",
)
class OrganizationPostgreSQLTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Config(str(ROOT / "alembic.ini"))
        cls.config.set_main_option("script_location", str(ROOT / "migrations"))
        cls.environment = {
            "DATABASE_URL": SAFE_POSTGRES_URL.render_as_string(False)
        }
        with patch.dict(os.environ, cls.environment, clear=False):
            alembic_command.upgrade(cls.config, "head")
        cls.engine = create_postgres_engine(SAFE_POSTGRES_URL)
        cls.factory = create_session_factory(cls.engine)

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def setUp(self):
        self.prefix = "p72-%s" % uuid4().hex
        self.users = PostgreSQLUserRepository(self.factory)
        self.auth = AuthenticationService(self.users, PasswordHasher())
        self.repository = PostgreSQLOrganizationRepository(self.factory)
        self.service = OrganizationService(self.repository)
        self.user_ids = []
        self.organization_ids = []

    def tearDown(self):
        with self.factory() as session:
            prefixed_organizations = session.execute(
                select(OrganizationRow.organization_id).where(
                    OrganizationRow.slug.like(self.prefix + "%")
                )
            ).scalars().all()
            organization_ids = list(
                set(self.organization_ids + prefixed_organizations)
            )
            if organization_ids:
                session.execute(
                    delete(OrganizationMembershipRow).where(
                        OrganizationMembershipRow.organization_id.in_(
                            organization_ids
                        )
                    )
                )
                session.execute(
                    delete(OrganizationRow).where(
                        OrganizationRow.organization_id.in_(organization_ids)
                    )
                )
            if self.user_ids:
                session.execute(
                    delete(UserRow).where(UserRow.user_id.in_(self.user_ids))
                )
            session.execute(
                delete(UserRow).where(UserRow.username.like(self.prefix + "%"))
            )
            session.commit()

    def create_user(self, suffix):
        user = self.auth.create_user(
            self.prefix + suffix, "Phase 7.2 " + suffix, "safe-test-password"
        )
        self.user_ids.append(user.user_id)
        return user

    def create_organization(self, suffix="-org", administrator=None):
        administrator = administrator or self.create_user("-admin")
        organization, membership = self.service.create_organization(
            self.prefix + suffix,
            "Phase 7.2 Organization",
            administrator.user_id,
        )
        self.organization_ids.append(organization.organization_id)
        return organization, membership, administrator

    def test_migration_0004_round_trip_schema_constraints_and_utc(self):
        with self.factory() as session:
            self.assertEqual(
                session.scalar(select(func.count()).select_from(OrganizationRow)),
                0,
            )
            self.assertEqual(
                session.scalar(
                    select(func.count()).select_from(OrganizationMembershipRow)
                ),
                0,
            )
        with patch.dict(os.environ, self.environment, clear=False):
            alembic_command.downgrade(self.config, "20260828_0004")
        self.assertNotIn("organizations", inspect(self.engine).get_table_names())
        with patch.dict(os.environ, self.environment, clear=False):
            alembic_command.upgrade(self.config, "head")
        inspector = inspect(self.engine)
        self.assertIn("organizations", inspector.get_table_names())
        self.assertIn("organization_memberships", inspector.get_table_names())
        self.assertEqual(
            inspector.get_foreign_keys("organization_memberships")[0][
                "referred_table"
            ]
            in {"organizations", "users"},
            True,
        )
        organization, membership, _ = self.create_organization()
        self.assertIsNotNone(organization.created_at.utcoffset())
        self.assertIsNotNone(membership.updated_at.utcoffset())

    def test_atomic_create_rollback_uniques_and_multi_organization(self):
        admin = self.create_user("-admin")
        organization, membership, _ = self.create_organization(
            "-one", admin
        )
        self.assertEqual(membership.role, OrganizationRole.ADMINISTRATOR)
        duplicate_id = str(uuid4())
        with self.assertRaises(OrganizationConflict):
            self.repository.create_with_administrator(
                replace(organization, organization_id=duplicate_id),
                replace(
                    membership,
                    membership_id=str(uuid4()),
                    organization_id=duplicate_id,
                ),
            )
        invalid_id = str(uuid4())
        invalid_org = replace(
            organization,
            organization_id=invalid_id,
            slug=self.prefix + "-rollback",
        )
        invalid_member = replace(
            membership,
            membership_id=str(uuid4()),
            organization_id=invalid_id,
            user_id="missing-user",
        )
        with self.assertRaises(OrganizationConflict):
            self.repository.create_with_administrator(
                invalid_org, invalid_member
            )
        with self.assertRaises(OrganizationNotFound):
            self.repository.get_by_id(invalid_id)
        second, _, _ = self.create_organization("-two", admin)
        self.assertEqual(
            len(self.service.list_enabled_for_user(admin.user_id)), 2
        )
        student = self.create_user("-student")
        added = self.service.add_membership(
            organization.organization_id,
            student.user_id,
            OrganizationRole.STUDENT,
        )
        self.assertEqual(
            self.service.add_membership(
                organization.organization_id,
                student.user_id,
                OrganizationRole.STUDENT,
            ).membership_id,
            added.membership_id,
        )
        with self.assertRaises(OrganizationConflict):
            self.service.add_membership(
                organization.organization_id,
                student.user_id,
                OrganizationRole.INSTRUCTOR,
            )
        self.assertNotEqual(organization.organization_id, second.organization_id)

    def test_last_administrator_and_concurrent_changes(self):
        organization, _, first = self.create_organization()
        second = self.create_user("-admin-two")
        self.service.add_membership(
            organization.organization_id,
            second.user_id,
            OrganizationRole.ADMINISTRATOR,
        )
        barrier = threading.Barrier(2)
        results = []

        def disable(user_id):
            barrier.wait()
            try:
                self.service.set_membership(
                    organization.organization_id,
                    user_id,
                    OrganizationRole.ADMINISTRATOR,
                    False,
                )
                results.append("disabled")
            except LastAdministratorError:
                results.append("protected")

        threads = [
            threading.Thread(target=disable, args=(first.user_id,)),
            threading.Thread(target=disable, args=(second.user_id,)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        self.assertEqual(sorted(results), ["disabled", "protected"])
        with self.factory() as session:
            count = session.scalar(
                select(func.count())
                .select_from(OrganizationMembershipRow)
                .where(
                    OrganizationMembershipRow.organization_id
                    == organization.organization_id,
                    OrganizationMembershipRow.enabled.is_(True),
                    OrganizationMembershipRow.role == "administrator",
                )
            )
        self.assertEqual(count, 1)

    def test_app_rebuild_context_and_cli_success_conflict_and_safety(self):
        first = create_app(
            Path(tempfile.mkdtemp()) / "runs",
            run_repository_mode="postgresql",
            database_url=SAFE_POSTGRES_URL,
            jwt_secret=TEST_SECRET,
        )
        created = first.state.authentication_service.create_user(
            self.prefix + "-persist",
            "Persistent Organization User",
            "persistent-password",
        )
        self.user_ids.append(created.user_id)
        organization, _ = first.state.organization_service.create_organization(
            self.prefix + "-persist-org",
            "Persistent Organization",
            created.user_id,
        )
        self.organization_ids.append(organization.organization_id)
        first.state.engine.dispose()
        second = create_app(
            Path(tempfile.mkdtemp()) / "runs",
            run_repository_mode="postgresql",
            database_url=SAFE_POSTGRES_URL,
            jwt_secret=TEST_SECRET,
        )
        try:
            with ASGITestClient(second) as client:
                login = client.post(
                    "/api/v1/auth/login",
                    json={
                        "username": created.username,
                        "password": "persistent-password",
                    },
                )
                token = login.json()["access_token"]
                headers = {"Authorization": "Bearer " + token}
                self.assertEqual(
                    client.get("/api/v1/me", headers=headers).json()[
                        "organizations"
                    ][0]["organization_id"],
                    organization.organization_id,
                )
        finally:
            second.state.engine.dispose()

        cli_admin = self.create_user("-cli-admin")
        cli_member = self.create_user("-cli-member")
        stdout = StringIO()
        with patch.dict(os.environ, self.environment, clear=False), patch(
            "sys.argv",
            [
                "create_organization.py",
                self.prefix + "-cli-org",
                "--name",
                "CLI Organization",
                "--administrator-username",
                cli_admin.username,
            ],
        ), redirect_stdout(stdout):
            self.assertEqual(create_organization.main(), 0)
        cli_org = self.service.get_by_slug(self.prefix + "-cli-org")
        self.organization_ids.append(cli_org.organization_id)
        conflict_error = StringIO()
        with patch.dict(os.environ, self.environment, clear=False), patch(
            "sys.argv",
            [
                "create_organization.py",
                self.prefix + "-cli-org",
                "--name",
                "Conflicting CLI Organization",
                "--administrator-username",
                cli_admin.username,
            ],
        ), redirect_stderr(conflict_error):
            self.assertEqual(create_organization.main(), 1)
        self.assertIn("different facts", conflict_error.getvalue())
        with patch.dict(os.environ, self.environment, clear=False), patch(
            "sys.argv",
            [
                "set_organization_membership.py",
                cli_org.slug,
                cli_member.username,
                "--role",
                "student",
                "--enable",
            ],
        ):
            self.assertEqual(set_organization_membership.main(), 0)
        stderr = StringIO()
        with patch.dict(os.environ, self.environment, clear=False), patch(
            "sys.argv",
            [
                "set_organization_membership.py",
                cli_org.slug,
                cli_admin.username,
                "--role",
                "student",
                "--enable",
            ],
        ), redirect_stderr(stderr):
            self.assertEqual(set_organization_membership.main(), 1)
        self.assertIn("last administrator", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
