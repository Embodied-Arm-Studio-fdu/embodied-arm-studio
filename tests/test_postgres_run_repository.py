import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from alembic import command
from alembic.config import Config
from sqlalchemy import delete, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError

from apps.api.database import (
    create_postgres_engine,
    create_session_factory,
)
from robotics.runs import (
    AppendEventResult,
    EventConflict,
    EventSequenceConflict,
    ExecutionMode,
    PostgreSQLRunRepository,
    RepositoryDataError,
    RepositoryIntegrityError,
    Run,
    RunAlreadyExists,
    RunError,
    RunEvent,
    RunEventType,
    RunKind,
    RunNotFound,
    RunStatus,
    TaskIdConflict,
)
from robotics.runs.db_models import RunEventRow, RunIdempotencyKeyRow, RunRow
from robotics.runs.postgres_repository import (
    event_from_row,
    event_to_row,
    run_from_row,
    run_to_row,
)
from robotics.runs.repository import RunRepository
from tests.postgres_safety import safe_test_database_url


ROOT = Path(__file__).resolve().parents[1]
UTC_TIME = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)


def make_run(run_id="run-1", created_at=UTC_TIME, task_id=None):
    return Run(
        run_id=run_id,
        user_id="user-1",
        project_id="project-1",
        status=RunStatus.QUEUED,
        execution_mode=ExecutionMode.MOCK,
        run_kind=RunKind.TRAIN,
        parameters={"seed": 7, "nested": {"value": 1}},
        project_config_snapshot={"scene": "tabletop"},
        code_sha256="a" * 64,
        simulation_task_id=task_id,
        metrics_summary={"reward": 1.25},
        error=RunError(code="example", message="example error", retryable=True),
        created_at=created_at,
        updated_at=created_at,
    )


def make_event(run_id="run-1", sequence=1, event_id=None):
    return RunEvent(
        event_id=event_id or "event-%s-%s" % (run_id, sequence),
        run_id=run_id,
        sequence=sequence,
        type=RunEventType.METRIC,
        occurred_at=UTC_TIME,
        payload={"step": sequence, "metrics": {"average_reward": 1.5}},
    )


class PostgreSQLRunRepositoryUnitTest(unittest.TestCase):
    def test_class_implements_every_run_repository_operation(self):
        methods = {
            name
            for name, value in RunRepository.__dict__.items()
            if callable(value) and not name.startswith("_")
        }
        self.assertEqual(
            methods,
            {
                "create_run",
                "get_run",
                "list_runs",
                "save_run",
                "bind_task_id",
                "append_event",
                "record_dispatch_error",
                "list_events",
            },
        )
        for method in methods:
            self.assertTrue(callable(getattr(PostgreSQLRunRepository, method)))

    def test_run_row_round_trip_restores_enums_error_and_utc(self):
        expected = make_run(task_id="task-1")
        restored = run_from_row(run_to_row(expected))
        self.assertEqual(restored, expected)
        self.assertIsInstance(restored.status, RunStatus)
        self.assertIsInstance(restored.execution_mode, ExecutionMode)
        self.assertIsInstance(restored.run_kind, RunKind)
        self.assertEqual(restored.created_at.tzinfo, timezone.utc)
        self.assertEqual(restored.error.retryable, True)

    def test_run_json_fields_are_deep_copied_in_both_directions(self):
        source = make_run()
        row = run_to_row(source)
        source.parameters["nested"]["value"] = 2
        self.assertEqual(row.parameters["nested"]["value"], 1)
        restored = run_from_row(row)
        row.parameters["nested"]["value"] = 3
        row.metrics_summary["reward"] = 9
        self.assertEqual(restored.parameters["nested"]["value"], 1)
        self.assertEqual(restored.metrics_summary["reward"], 1.25)

    def test_event_row_round_trip_copies_payload_and_returns_canonical_event(self):
        event = make_event()
        row = event_to_row(event)
        restored = event_from_row(row)
        row.payload["metrics"]["average_reward"] = 8
        self.assertEqual(restored, event)
        self.assertEqual(restored.payload["metrics"]["average_reward"], 1.5)
        self.assertNotIsInstance(restored, RunEventRow)

    def test_unknown_enums_and_malformed_json_raise_repository_data_error(self):
        row = run_to_row(make_run())
        row.status = "unknown"
        with self.assertRaisesRegex(RepositoryDataError, "Invalid persisted Run"):
            run_from_row(row)

        row = run_to_row(make_run())
        row.metrics_summary = ["not", "an", "object"]
        with self.assertRaisesRegex(RepositoryDataError, "metrics_summary"):
            run_from_row(row)

        event_row = event_to_row(make_event())
        event_row.type = "unknown"
        with self.assertRaisesRegex(RepositoryDataError, "RunEvent"):
            event_from_row(event_row)

    def test_naive_or_non_utc_database_time_is_rejected(self):
        naive = run_to_row(make_run())
        naive.created_at = UTC_TIME.replace(tzinfo=None)
        with self.assertRaisesRegex(RepositoryDataError, "timezone-aware UTC"):
            run_from_row(naive)

        non_utc = run_to_row(make_run())
        non_utc.created_at = UTC_TIME.astimezone(
            timezone(timedelta(hours=1))
        )
        with self.assertRaisesRegex(RepositoryDataError, "must be UTC"):
            run_from_row(non_utc)

    def test_repository_statements_compile_with_postgresql_dialect(self):
        statements = (
            select(RunRow)
            .where(RunRow.run_id == "run-1")
            .with_for_update(),
            select(RunRow).order_by(
                RunRow.created_at.desc(), RunRow.run_id.asc()
            ),
            select(RunEventRow).where(
                RunEventRow.run_id == "run-1", RunEventRow.sequence > 4
            ),
        )
        compiled = [
            str(statement.compile(dialect=postgresql.dialect()))
            for statement in statements
        ]
        self.assertIn("FOR UPDATE", compiled[0])
        self.assertIn("ORDER BY runs.created_at DESC, runs.run_id ASC", compiled[1])
        self.assertIn("run_events.sequence >", compiled[2])

    def test_successful_operation_commits_closes_and_returns_no_orm_row(self):
        session = _FakeSession(get_values={(RunRow, "run-1"): run_to_row(make_run())})
        repository = PostgreSQLRunRepository(_SessionFactory(session))
        returned = repository.get_run("run-1")
        self.assertEqual(returned, make_run())
        self.assertIsInstance(returned, Run)
        self.assertNotIsInstance(returned, RunRow)
        self.assertEqual(session.calls, ["commit", "close"])

    def test_failed_operation_rolls_back_closes_and_preserves_error(self):
        failure = RuntimeError("database unavailable")
        session = _FakeSession(get_error=failure)
        repository = PostgreSQLRunRepository(_SessionFactory(session))
        with self.assertRaisesRegex(RuntimeError, "database unavailable"):
            repository.get_run("run-1")
        self.assertEqual(session.calls, ["rollback", "close"])

    def test_create_duplicate_maps_integrity_error_after_rollback(self):
        integrity = IntegrityError("INSERT", {}, RuntimeError("unique violation"))
        writing = _FakeSession(flush_error=integrity)
        classifying = _FakeSession(
            get_values={(RunRow, "run-1"): run_to_row(make_run())}
        )
        repository = PostgreSQLRunRepository(
            _SessionFactory(writing, classifying)
        )
        with self.assertRaises(RunAlreadyExists):
            repository.create_run(make_run())
        self.assertEqual(writing.calls, ["add", "flush", "rollback", "close"])
        self.assertEqual(classifying.calls, ["rollback", "close"])

    def test_unmapped_integrity_error_keeps_cause_without_leaking_sqlalchemy_error(self):
        integrity = IntegrityError("INSERT", {}, RuntimeError("constraint"))
        writing = _FakeSession(flush_error=integrity)
        classifying = _FakeSession(get_values={(RunRow, "run-1"): None})
        repository = PostgreSQLRunRepository(
            _SessionFactory(writing, classifying)
        )
        with self.assertRaises(RepositoryIntegrityError) as raised:
            repository.create_run(make_run())
        self.assertIs(raised.exception.__cause__, integrity)
        self.assertEqual(writing.calls[-2:], ["rollback", "close"])
        self.assertEqual(classifying.calls, ["rollback", "close"])

    def test_cross_run_task_unique_failure_maps_to_task_conflict(self):
        run = make_run()
        integrity = IntegrityError("UPDATE", {}, RuntimeError("unique violation"))
        writing = _FakeSession(
            execute_results=[_FakeResult(scalar=run_to_row(run))],
            flush_error=integrity,
        )
        classifying = _FakeSession(
            execute_results=[_FakeResult(scalar="owner-run")]
        )
        repository = PostgreSQLRunRepository(
            _SessionFactory(writing, classifying)
        )
        with self.assertRaises(TaskIdConflict) as raised:
            repository.bind_task_id(run.run_id, "task-shared")
        self.assertEqual(raised.exception.owner_run_id, "owner-run")
        self.assertIs(raised.exception.__cause__, integrity)
        self.assertEqual(writing.calls[-2:], ["rollback", "close"])

    def test_event_integrity_failure_is_rechecked_for_idempotency(self):
        event = make_event()
        integrity = IntegrityError("INSERT", {}, RuntimeError("unique violation"))
        writing = _FakeSession(
            get_values={(RunRow, event.run_id): run_to_row(make_run())},
            flush_error=integrity,
        )
        classifying = _FakeSession(
            get_values={
                (RunRow, event.run_id): run_to_row(make_run()),
                (RunEventRow, event.event_id): event_to_row(event),
            }
        )
        repository = PostgreSQLRunRepository(
            _SessionFactory(writing, classifying)
        )
        self.assertEqual(
            repository.append_event(event), AppendEventResult.DUPLICATE
        )
        self.assertEqual(writing.calls[-2:], ["rollback", "close"])
        self.assertEqual(classifying.calls, ["commit", "close"])

    def test_event_integrity_failure_rechecks_sequence_after_event_id(self):
        event = make_event()
        integrity = IntegrityError("INSERT", {}, RuntimeError("unique violation"))
        writing = _FakeSession(
            get_values={(RunRow, event.run_id): run_to_row(make_run())},
            flush_error=integrity,
        )
        classifying = _FakeSession(
            get_values={(RunRow, event.run_id): run_to_row(make_run())},
            execute_results=[_FakeResult(scalar=event_to_row(event))],
        )
        repository = PostgreSQLRunRepository(
            _SessionFactory(writing, classifying)
        )
        with self.assertRaises(EventSequenceConflict) as raised:
            repository.append_event(replace(event, event_id="event-other"))
        self.assertIs(raised.exception.__cause__, integrity)
        self.assertEqual(classifying.calls, ["rollback", "close"])


class _FakeResult:
    def __init__(self, scalar=None, rows=None):
        self.scalar = scalar
        self.rows = [] if rows is None else rows

    def scalar_one_or_none(self):
        return self.scalar

    def scalars(self):
        return self

    def all(self):
        return self.rows


class _FakeSession:
    def __init__(
        self,
        *,
        get_values=None,
        get_error=None,
        execute_results=None,
        flush_error=None,
    ):
        self.get_values = {} if get_values is None else get_values
        self.get_error = get_error
        self.execute_results = [] if execute_results is None else execute_results
        self.flush_error = flush_error
        self.calls = []

    def get(self, model, key):
        if self.get_error is not None:
            raise self.get_error
        return self.get_values.get((model, key))

    def add(self, _row):
        self.calls.append("add")

    def execute(self, _statement):
        if self.execute_results:
            return self.execute_results.pop(0)
        return _FakeResult()

    def flush(self):
        self.calls.append("flush")
        if self.flush_error is not None:
            raise self.flush_error

    def commit(self):
        self.calls.append("commit")

    def rollback(self):
        self.calls.append("rollback")

    def close(self):
        self.calls.append("close")


class _SessionFactory:
    def __init__(self, *sessions):
        self.sessions = list(sessions)

    def __call__(self):
        if not self.sessions:
            raise AssertionError("unexpected Session creation")
        return self.sessions.pop(0)


SAFE_TEST_DATABASE_URL = safe_test_database_url()


@unittest.skipUnless(
    SAFE_TEST_DATABASE_URL is not None,
    "TEST_DATABASE_URL is unset; real PostgreSQL repository tests were not run",
)
class PostgreSQLRunRepositoryIntegrationTest(unittest.TestCase):
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
            command.upgrade(config, "head")

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def setUp(self):
        self.prefix = "p33-%s-" % uuid4().hex
        self.repository = PostgreSQLRunRepository(self.factory)

    def tearDown(self):
        with self.factory() as session:
            session.execute(
                delete(RunIdempotencyKeyRow).where(
                    RunIdempotencyKeyRow.run_id.like(self.prefix + "%")
                )
            )
            session.execute(
                delete(RunEventRow).where(
                    RunEventRow.run_id.like(self.prefix + "%")
                )
            )
            session.execute(
                delete(RunRow).where(RunRow.run_id.like(self.prefix + "%"))
            )
            session.commit()

    def _run(self, suffix, created_at=UTC_TIME):
        return make_run(self.prefix + suffix, created_at=created_at)

    def test_create_get_list_save_and_new_repository_recovery(self):
        older = self._run("older")
        newer = self._run("newer", UTC_TIME + timedelta(seconds=1))
        self.repository.create_run(older)
        self.repository.create_run(newer)
        saved = replace(older, status=RunStatus.INITIALIZING, progress=0.2)
        self.repository.save_run(saved)
        reopened = PostgreSQLRunRepository(self.factory)
        self.assertEqual(reopened.get_run(older.run_id), saved)
        ids = [run.run_id for run in reopened.list_runs()]
        self.assertLess(ids.index(newer.run_id), ids.index(older.run_id))

    def test_task_binding_idempotency_and_cross_run_conflict(self):
        first = self._run("first")
        second = self._run("second")
        self.repository.create_run(first)
        self.repository.create_run(second)
        self.assertEqual(
            self.repository.bind_task_id(first.run_id, "task-" + self.prefix),
            self.repository.bind_task_id(first.run_id, "task-" + self.prefix),
        )
        with self.assertRaises(TaskIdConflict):
            self.repository.bind_task_id(second.run_id, "task-" + self.prefix)

    def test_event_sort_filter_idempotency_and_conflicts(self):
        run = self._run("events")
        self.repository.create_run(run)
        events = [make_event(run.run_id, sequence) for sequence in (3, 1, 5)]
        for event in events:
            self.assertEqual(
                self.repository.append_event(event), AppendEventResult.STORED
            )
        self.assertEqual(
            [event.sequence for event in self.repository.list_events(run.run_id)],
            [1, 3, 5],
        )
        self.assertEqual(
            [
                event.sequence
                for event in self.repository.list_events(run.run_id, 3)
            ],
            [5],
        )
        self.assertEqual(
            self.repository.append_event(events[0]), AppendEventResult.DUPLICATE
        )
        with self.assertRaises(EventConflict):
            self.repository.append_event(
                replace(events[0], sequence=7, payload={"different": True})
            )
        with self.assertRaises(EventSequenceConflict):
            self.repository.append_event(
                replace(events[0], event_id="other-" + events[0].event_id)
            )

    def test_reconcile_queued_unbound_writes_run_and_event_atomically(self):
        expected = replace(
            self._run("administrative"), execution_mode=ExecutionMode.SIMULATION
        )
        self.repository.create_run(expected)
        event = RunEvent(
            event_id=self.prefix + "administrative-event",
            run_id=expected.run_id,
            sequence=1,
            type=RunEventType.STATE,
            occurred_at=UTC_TIME + timedelta(seconds=1),
            payload={
                "previous_status": "queued",
                "status": "stopped",
                "administrative_reconciliation": True,
                "reason": "stale unbound Run",
            },
        )
        updated = replace(
            expected,
            status=RunStatus.STOPPED,
            stop_reason="administrative_reconciliation",
            last_sequence=1,
            ended_at=UTC_TIME + timedelta(seconds=1),
            updated_at=UTC_TIME + timedelta(seconds=1),
        )
        self.assertEqual(
            self.repository.reconcile_queued_unbound(expected, updated, event),
            updated,
        )
        self.assertEqual(self.repository.get_run(expected.run_id), updated)
        self.assertEqual(self.repository.list_events(expected.run_id), [event])
        with self.assertRaises(RepositoryIntegrityError):
            self.repository.reconcile_queued_unbound(expected, updated, event)

    def test_unknown_event_run_and_database_foreign_key(self):
        unknown_id = self.prefix + "missing"
        with self.assertRaises(RunNotFound):
            self.repository.append_event(make_event(unknown_id))
        with self.factory() as session:
            session.add(event_to_row(make_event(unknown_id, event_id=self.prefix + "fk")))
            with self.assertRaises(IntegrityError):
                session.commit()
            session.rollback()

    def test_concurrent_task_binding_has_one_winner(self):
        run = self._run("task-race")
        self.repository.create_run(run)

        def bind(task_id):
            try:
                return self.repository.bind_task_id(run.run_id, task_id).simulation_task_id
            except TaskIdConflict:
                return "conflict"

        task_ids = [self.prefix + "task-a", self.prefix + "task-b"]
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(bind, task_ids))
        self.assertEqual(outcomes.count("conflict"), 1)
        self.assertIn(self.repository.get_run(run.run_id).simulation_task_id, task_ids)

    def test_concurrent_identical_event_is_stored_once(self):
        run = self._run("event-race")
        self.repository.create_run(run)
        event = make_event(run.run_id)
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(self.repository.append_event, [event, event]))
        self.assertEqual(
            sorted(outcome.value for outcome in outcomes), ["duplicate", "stored"]
        )
        self.assertEqual(self.repository.list_events(run.run_id), [event])

    def test_failed_duplicate_transaction_rolls_back(self):
        run = self._run("rollback")
        self.repository.create_run(run)
        with self.assertRaises(RunAlreadyExists):
            self.repository.create_run(run)
        self.assertEqual(self.repository.get_run(run.run_id), run)


if __name__ == "__main__":
    unittest.main()
