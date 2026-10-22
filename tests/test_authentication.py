import os
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import jwt
from alembic import command as alembic_command
from alembic.config import Config
from sqlalchemy import delete, inspect, update

from apps.api.database import create_postgres_engine, create_session_factory
from apps.api.dependencies.auth import CurrentUserProvider
from apps.api.main import create_app
from robotics.auth import (
    AuthenticationFailed,
    AuthenticationService,
    DuplicateUsername,
    JWTConfiguration,
    JWTConfigurationError,
    JWTTokenError,
    PasswordHasher,
    PostgreSQLUserRepository,
    User,
    UserNotFound,
)
from robotics.auth.models import utc_now
from robotics.auth.tokens import JWTAccessTokenService
from robotics.runs import ExecutionMode, InMemoryRunRepository, RunKind
from robotics.runs.db_models import RunRow, UserRow
from tests.http_client import ASGITestClient
from tests.test_terminal_callback_api import SAFE_POSTGRES_URL


ROOT = Path(__file__).resolve().parents[1]
TEST_SECRET = "phase-7-1-test-secret-with-32-bytes-minimum"
ISSUER = "phase-7-1-test-issuer"
AUDIENCE = "phase-7-1-test-audience"


class MemoryUserRepository:
    def __init__(self):
        self.users = {}

    def create(self, user):
        if any(item.username == user.username for item in self.users.values()):
            raise DuplicateUsername("username already exists")
        self.users[user.user_id] = user
        return user

    def get_by_id(self, user_id):
        try:
            return self.users[user_id]
        except KeyError as exc:
            raise UserNotFound("user not found") from exc

    def get_by_username(self, username):
        for user in self.users.values():
            if user.username == username:
                return user
        raise UserNotFound("user not found")

    def find_by_username(self, username):
        try:
            return self.get_by_username(username)
        except UserNotFound:
            return None

    def update_password_hash(self, user_id, password_hash):
        user = self.get_by_id(user_id)
        updated = replace(user, password_hash=password_hash, updated_at=utc_now())
        self.users[user_id] = updated
        return updated


def token_service(expires_seconds=900):
    return JWTAccessTokenService(
        JWTConfiguration(
            secret=TEST_SECRET,
            issuer=ISSUER,
            audience=AUDIENCE,
            expires_seconds=expires_seconds,
        )
    )


class AuthenticationUnitAndApiTest(unittest.TestCase):
    def setUp(self):
        self.repository = MemoryUserRepository()
        self.passwords = PasswordHasher()
        self.tokens = token_service()
        self.service = AuthenticationService(
            self.repository, self.passwords, self.tokens
        )
        self.user = self.service.create_user(
            "student-1", "Student One", "correct-password"
        )
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.app = create_app(
            root / "runs",
            project_root=root / "projects",
            run_repository=InMemoryRunRepository(),
            authentication_service=self.service,
        )
        self.client = ASGITestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def login(self, password="correct-password", username="student-1"):
        return self.client.post(
            "/api/v1/auth/login",
            json={"username": username, "password": password},
        )

    def test_user_creation_hashes_argon2id_and_duplicate_fails(self):
        self.assertTrue(self.user.password_hash.startswith("$argon2id$"))
        self.assertNotIn("correct-password", self.user.password_hash)
        self.assertTrue(
            self.passwords.verify(self.user.password_hash, "correct-password")
        )
        self.assertFalse(self.passwords.verify(self.user.password_hash, "wrong"))
        self.assertIsNotNone(self.user.created_at.utcoffset())
        with self.assertRaises(DuplicateUsername):
            self.service.create_user(
                "student-1", "Replacement", "replacement-password"
            )

    def test_login_success_me_and_generic_failure(self):
        logged_in = self.login()
        self.assertEqual(logged_in.status_code, 200)
        body = logged_in.json()
        self.assertEqual(body["token_type"], "bearer")
        self.assertEqual(body["expires_in"], 900)
        self.assertNotIn("password_hash", body["user"])
        me = self.client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer " + body["access_token"]},
        )
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["user_id"], self.user.user_id)
        wrong = self.login(password="wrong")
        unknown = self.login(username="unknown")
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(wrong.json(), unknown.json())
        self.assertEqual(wrong.headers["www-authenticate"], "Bearer")

    def test_change_password_requires_current_password_and_rotates_hash(self):
        original_hash = self.user.password_hash
        with self.assertRaises(AuthenticationFailed):
            self.service.change_password(
                "student-1", "wrong-password", "replacement-password"
            )
        with self.assertRaisesRegex(ValueError, "at least 12 characters"):
            self.service.change_password("student-1", "correct-password", "short")

        updated = self.service.change_password(
            "student-1", "correct-password", "replacement-password"
        )

        self.assertNotEqual(updated.password_hash, original_hash)
        self.assertEqual(self.login().status_code, 401)
        self.assertEqual(self.login(password="replacement-password").status_code, 200)

    def test_disabled_user_cannot_login_or_continue_using_token(self):
        token = self.service.issue_access_token(self.user)
        self.repository.users[self.user.user_id] = replace(
            self.user, enabled=False, updated_at=utc_now()
        )
        self.assertEqual(self.login().status_code, 401)
        response = self.client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer " + token},
        )
        self.assertEqual(response.status_code, 401)

    def test_missing_tampered_expired_issuer_and_audience_are_401(self):
        self.assertEqual(self.client.get("/api/v1/auth/me").status_code, 401)
        valid = self.service.issue_access_token(self.user)
        tampered = valid[:-1] + ("a" if valid[-1] != "a" else "b")
        now = datetime.now(timezone.utc)
        base = {
            "sub": self.user.user_id,
            "iss": ISSUER,
            "aud": AUDIENCE,
            "iat": now - timedelta(minutes=2),
            "nbf": now - timedelta(minutes=2),
            "exp": now - timedelta(minutes=1),
            "jti": str(uuid4()),
        }
        variants = [tampered]
        for changes in (
            {},
            {"exp": now + timedelta(minutes=1), "iss": "wrong"},
            {"exp": now + timedelta(minutes=1), "aud": "wrong"},
        ):
            claims = dict(base)
            claims.update(changes)
            variants.append(jwt.encode(claims, TEST_SECRET, algorithm="HS256"))
        for token in variants:
            response = self.client.get(
                "/api/v1/auth/me",
                headers={"Authorization": "Bearer " + token},
            )
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.headers["www-authenticate"], "Bearer")

    def test_owner_isolation_ignores_client_identity_headers(self):
        own = self.app.state.run_service.create_run(
            user_id=self.user.user_id,
            project_id="project-own",
            execution_mode=ExecutionMode.MOCK,
            run_kind=RunKind.TRAIN,
            parameters={},
            project_config_snapshot={},
            code="",
        )
        other = self.app.state.run_service.create_run(
            user_id="other-user",
            project_id="project-other",
            execution_mode=ExecutionMode.MOCK,
            run_kind=RunKind.TRAIN,
            parameters={},
            project_config_snapshot={},
            code="",
        )
        token = self.service.issue_access_token(self.user)
        headers = {
            "Authorization": "Bearer " + token,
            "X-User-Id": "other-user",
        }
        listed = self.client.get("/api/v1/runs", headers=headers)
        self.assertEqual([item["run_id"] for item in listed.json()], [own.run_id])
        self.assertEqual(
            self.client.get(
                "/api/v1/runs/%s" % other.run_id,
                headers=headers,
            ).status_code,
            404,
        )

    def test_configuration_requires_strong_secret_and_postgres_mode_has_no_default(self):
        with self.assertRaises(JWTConfigurationError):
            JWTConfiguration(secret="short", issuer=ISSUER, audience=AUDIENCE)
        root = Path(self.temp.name)
        with self.assertRaisesRegex(RuntimeError, "USER_JWT_SECRET"):
            create_app(
                root / "formal-runs",
                project_root=root / "formal-projects",
                run_repository_mode="postgresql",
                database_url="postgresql+psycopg://ignored@localhost/auth_test",
                database_environment={},
            )


@unittest.skipUnless(
    SAFE_POSTGRES_URL is not None,
    "PostgreSQL test configuration is unavailable",
)
class AuthenticationPostgreSQLTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Config(str(ROOT / "alembic.ini"))
        cls.config.set_main_option(
            "script_location", str(ROOT / "migrations")
        )
        cls.database_environment = {
            "DATABASE_URL": SAFE_POSTGRES_URL.render_as_string(
                hide_password=False
            )
        }
        with patch.dict(os.environ, cls.database_environment, clear=False):
            alembic_command.upgrade(cls.config, "head")
        cls.engine = create_postgres_engine(SAFE_POSTGRES_URL)
        cls.factory = create_session_factory(cls.engine)

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def setUp(self):
        self.prefix = "p71-%s" % uuid4().hex

    def tearDown(self):
        with self.factory() as session:
            session.execute(delete(RunRow).where(RunRow.user_id == self.prefix))
            session.execute(delete(UserRow).where(UserRow.user_id == self.prefix))
            session.execute(delete(UserRow).where(UserRow.username == self.prefix))
            session.commit()

    def test_migration_upgrade_downgrade_upgrade(self):
        self.assertIn("users", inspect(self.engine).get_table_names())
        with patch.dict(os.environ, self.database_environment, clear=False):
            alembic_command.downgrade(self.config, "20260827_0003")
        self.assertNotIn("users", inspect(self.engine).get_table_names())
        with patch.dict(os.environ, self.database_environment, clear=False):
            alembic_command.upgrade(self.config, "head")
        self.assertIn("users", inspect(self.engine).get_table_names())

    def test_repository_unique_constraint_and_app_rebuild_login(self):
        first = create_app(
            Path(tempfile.mkdtemp()) / "runs",
            run_repository_mode="postgresql",
            database_url=SAFE_POSTGRES_URL,
            jwt_secret=TEST_SECRET,
            jwt_issuer=ISSUER,
            jwt_audience=AUDIENCE,
        )
        created = first.state.authentication_service.create_user(
            self.prefix, "Persistent User", "persistent-password"
        )
        self.prefix = created.user_id
        run = first.state.run_service.create_run(
            user_id=created.user_id,
            project_id="persistent-project",
            execution_mode=ExecutionMode.MOCK,
            run_kind=RunKind.TRAIN,
            parameters={},
            project_config_snapshot={},
            code="",
        )
        with self.assertRaises(DuplicateUsername):
            first.state.authentication_service.create_user(
                created.username, "Duplicate", "duplicate-password"
            )
        first.state.engine.dispose()
        second = create_app(
            Path(tempfile.mkdtemp()) / "runs",
            run_repository_mode="postgresql",
            database_url=SAFE_POSTGRES_URL,
            jwt_secret=TEST_SECRET,
            jwt_issuer=ISSUER,
            jwt_audience=AUDIENCE,
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
                self.assertEqual(login.status_code, 200)
                listed = client.get(
                    "/api/v1/runs",
                    headers={
                        "Authorization": "Bearer "
                        + login.json()["access_token"]
                    },
                )
                self.assertEqual([item["run_id"] for item in listed.json()], [run.run_id])
        finally:
            second.state.engine.dispose()

    def test_repository_password_change_persists(self):
        repository = PostgreSQLUserRepository(self.factory)
        service = AuthenticationService(repository, PasswordHasher())
        created = service.create_user(
            self.prefix, "Password Change User", "original-password"
        )
        self.prefix = created.user_id

        updated = service.change_password(
            created.username, "original-password", "replacement-password"
        )
        restored = repository.get_by_id(created.user_id)

        self.assertGreater(updated.updated_at, created.updated_at)
        self.assertEqual(restored.password_hash, updated.password_hash)
        with self.assertRaises(AuthenticationFailed):
            service.authenticate(created.username, "original-password")
        self.assertEqual(
            service.authenticate(created.username, "replacement-password").user_id,
            created.user_id,
        )
