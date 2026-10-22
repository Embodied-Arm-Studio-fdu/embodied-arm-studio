import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from alembic import command as alembic_command
from alembic.config import Config
from sqlalchemy import delete, event, func, select
from sqlalchemy.exc import SQLAlchemyError

from apps.api.database import (
    create_postgres_engine,
    create_session_factory,
)
from robotics.runs import (
    ExecutionMode,
    IdempotencyConflict,
    IdempotentRunCreateCommand,
    PostgreSQLIdempotentRunCreationService,
    PostgreSQLRunRepository,
    RunKind,
    RunService,
    RunStatus,
)
from robotics.runs.db_models import RunIdempotencyKeyRow, RunRow
from tests.postgres_safety import safe_test_database_url


ROOT = Path(__file__).resolve().parents[1]
UTC_TIME = datetime(2026, 8, 25, 12, 30, tzinfo=timezone.utc)


SAFE_TEST_DATABASE_URL = safe_test_database_url()


@unittest.skipUnless(
    SAFE_TEST_DATABASE_URL is not None,
    "TEST_DATABASE_URL is unset; real PostgreSQL idempotency tests were not run",
)
class PostgreSQLRunIdempotencyIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = create_postgres_engine(SAFE_TEST_DATABASE_URL)
        cls.factory = create_session_factory(cls.engine)
        config = Config(str(ROOT / "alembic.ini"))
        with patch.dict(
            os.environ,
            {"DATABASE_URL": SAFE_TEST_DATABASE_URL.render_as_string(False)},
            clear=False,
        ):
            alembic_command.upgrade(config, "head")

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def setUp(self):
        self.prefix = "p42-%s-" % uuid4().hex
        self.repository = PostgreSQLRunRepository(self.factory)
        self.service = RunService(
            self.repository,
            clock=lambda: UTC_TIME,
            run_id_factory=lambda: self.prefix + uuid4().hex,
        )
        self.idempotency = PostgreSQLIdempotentRunCreationService(
            self.service
        )

    def tearDown(self):
        with self.factory() as session:
            session.execute(
                delete(RunIdempotencyKeyRow).where(
                    RunIdempotencyKeyRow.user_id.like(self.prefix + "%")
                )
            )
            session.execute(
                delete(RunRow).where(RunRow.user_id.like(self.prefix + "%"))
            )
            session.commit()

    def command(self, **changes):
        values = {
            "user_id": self.prefix + "user-1",
            "idempotency_key": "create-run",
            "project_id": self.prefix + "project-1",
            "execution_mode": ExecutionMode.SIMULATION,
            "parameters": {"seed": 42, "nested": {"gain": 0.25}},
            "project_config_snapshot": {
                "scene": {"name": "push"},
                "config_version": 4,
            },
            "code": "print('train')\n",
            "run_kind": RunKind.PLAY,
        }
        values.update(changes)
        return IdempotentRunCreateCommand(**values)

    def counts(self):
        with self.factory() as session:
            run_count = session.scalar(
                select(func.count())
                .select_from(RunRow)
                .where(RunRow.user_id.like(self.prefix + "%"))
            )
            key_count = session.scalar(
                select(func.count())
                .select_from(RunIdempotencyKeyRow)
                .where(RunIdempotencyKeyRow.user_id.like(self.prefix + "%"))
            )
            return run_count, key_count

    def test_first_create_replay_counts_and_conflict(self):
        first = self.idempotency.create_run(self.command())
        replay = self.idempotency.create_run(self.command())

        self.assertFalse(first.replayed)
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.run.run_id, first.run.run_id)
        self.assertEqual(self.counts(), (1, 1))

        with self.assertRaises(IdempotencyConflict):
            self.idempotency.create_run(
                self.command(parameters={"seed": 99})
            )
        self.assertEqual(self.counts(), (1, 1))

    def test_different_users_can_reuse_the_same_key(self):
        first = self.idempotency.create_run(self.command())
        second = self.idempotency.create_run(
            self.command(user_id=self.prefix + "user-2")
        )
        self.assertFalse(first.replayed)
        self.assertFalse(second.replayed)
        self.assertNotEqual(first.run.run_id, second.run.run_id)
        self.assertEqual(self.counts(), (2, 2))

    def test_concurrent_identical_request_creates_exactly_one_run(self):
        command = self.command()
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(self.idempotency.create_run, [command, command])
            )
        self.assertEqual(
            sorted(result.replayed for result in results), [False, True]
        )
        self.assertEqual(len({result.run.run_id for result in results}), 1)
        self.assertEqual(self.counts(), (1, 1))

    def test_run_insert_database_failure_rolls_back_both_rows(self):
        def invalidate_run_hash(mapper, connection, target):
            if target.user_id.startswith(self.prefix):
                target.code_sha256 = "x" * 65

        event.listen(RunRow, "before_insert", invalidate_run_hash)
        try:
            with self.assertRaises(SQLAlchemyError):
                self.idempotency.create_run(self.command())
        finally:
            event.remove(RunRow, "before_insert", invalidate_run_hash)
        self.assertEqual(self.counts(), (0, 0))

    def test_idempotency_insert_database_failure_rolls_back_run(self):
        def invalidate_request_hash(mapper, connection, target):
            if target.user_id.startswith(self.prefix):
                target.request_hash = "x" * 65

        event.listen(
            RunIdempotencyKeyRow, "before_insert", invalidate_request_hash
        )
        try:
            with self.assertRaises(SQLAlchemyError):
                self.idempotency.create_run(self.command())
        finally:
            event.remove(
                RunIdempotencyKeyRow,
                "before_insert",
                invalidate_request_hash,
            )
        self.assertEqual(self.counts(), (0, 0))

    def test_service_rebuild_replays_and_restores_canonical_types(self):
        created = self.idempotency.create_run(self.command())
        rebuilt = PostgreSQLIdempotentRunCreationService(
            RunService(
                PostgreSQLRunRepository(self.factory),
                clock=lambda: datetime(2030, 1, 1, tzinfo=timezone.utc),
                run_id_factory=lambda: self.prefix + "must-not-be-created",
            )
        )
        replay = rebuilt.create_run(self.command())

        self.assertTrue(replay.replayed)
        self.assertEqual(replay.run, created.run)
        self.assertEqual(replay.run.status, RunStatus.QUEUED)
        self.assertEqual(replay.run.execution_mode, ExecutionMode.SIMULATION)
        self.assertEqual(replay.run.run_kind, RunKind.PLAY)
        self.assertEqual(replay.run.created_at, UTC_TIME)
        self.assertEqual(replay.run.updated_at, UTC_TIME)
        self.assertEqual(replay.run.created_at.utcoffset().total_seconds(), 0)
        self.assertEqual(
            replay.run.parameters, {"seed": 42, "nested": {"gain": 0.25}}
        )
        self.assertEqual(
            replay.run.project_config_snapshot,
            {"scene": {"name": "push"}, "config_version": 4},
        )
        self.assertEqual(self.counts(), (1, 1))


if __name__ == "__main__":
    unittest.main()
