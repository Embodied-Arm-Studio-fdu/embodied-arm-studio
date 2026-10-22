import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from robotics.runs import (
    AppendEventResult,
    EventConflict,
    EventSequenceConflict,
    ExecutionMode,
    RepositoryDataError,
    Run,
    RunAlreadyExists,
    RunError,
    RunEvent,
    RunEventType,
    RunKind,
    RunNotFound,
    RunStatus,
    TaskIdAlreadyBound,
    TaskIdConflict,
)
from robotics.runs.sqlite_repository import SQLiteRunRepository


UTC_TIME = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)


def make_run(
    run_id="run-1", created_at=UTC_TIME, simulation_task_id=None
):
    return Run(
        run_id=run_id,
        user_id="user-1",
        project_id="project-1",
        status=RunStatus.QUEUED,
        execution_mode=ExecutionMode.MOCK,
        run_kind=RunKind.TRAIN,
        parameters={"seed": 7},
        project_config_snapshot={"scene": "tabletop"},
        code_sha256="a" * 64,
        simulation_task_id=simulation_task_id,
        created_at=created_at,
        updated_at=created_at,
    )


def make_event(run_id="run-1", sequence=1, event_id=None, payload=None):
    return RunEvent(
        event_id=event_id or "event-%s-%s" % (run_id, sequence),
        run_id=run_id,
        sequence=sequence,
        type=RunEventType.METRIC,
        occurred_at=UTC_TIME + timedelta(seconds=sequence),
        payload=payload if payload is not None else {"step": sequence},
    )


class SQLiteRunRepositoryTest(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self._temporary_directory.name) / "runs.sqlite3"
        self.repository = SQLiteRunRepository(self.database_path)

    def tearDown(self):
        self._temporary_directory.cleanup()

    def test_create_and_get_run(self):
        run = make_run()
        self.assertEqual(self.repository.create_run(run), run)
        self.assertEqual(self.repository.get_run(run.run_id), run)

    def test_list_runs_uses_in_memory_sort_order(self):
        older = make_run("older", UTC_TIME)
        newer = make_run("newer", UTC_TIME + timedelta(seconds=1))
        self.repository.create_run(older)
        self.repository.create_run(newer)
        self.assertEqual(
            [run.run_id for run in self.repository.list_runs()], ["newer", "older"]
        )

    def test_save_status_metrics_error_and_utc_times(self):
        self.repository.create_run(make_run())
        saved_at = UTC_TIME + timedelta(seconds=10)
        expected = replace(
            make_run(),
            status=RunStatus.FAILED,
            progress=0.75,
            metrics_summary={"reward": 4.25},
            error=RunError(code="mock_failed", message="failed", retryable=True),
            last_sequence=3,
            started_at=UTC_TIME,
            ended_at=saved_at,
            updated_at=saved_at,
        )
        self.assertEqual(self.repository.save_run(expected), expected)
        restored = self.repository.get_run(expected.run_id)
        self.assertEqual(restored, expected)
        self.assertEqual(restored.created_at.tzinfo, timezone.utc)
        self.assertEqual(restored.error.retryable, True)

    def test_duplicate_run_id_raises(self):
        self.repository.create_run(make_run())
        with self.assertRaises(RunAlreadyExists):
            self.repository.create_run(make_run())

    def test_unknown_run_raises_for_get_save_and_events(self):
        with self.assertRaises(RunNotFound):
            self.repository.get_run("missing")
        with self.assertRaises(RunNotFound):
            self.repository.save_run(make_run("missing"))
        with self.assertRaises(RunNotFound):
            self.repository.list_events("missing")
        with self.assertRaises(RunNotFound):
            self.repository.append_event(make_event("missing"))

    def test_bind_task_id_first_and_same_are_idempotent(self):
        self.repository.create_run(make_run())
        first = self.repository.bind_task_id("run-1", "task-1")
        second = self.repository.bind_task_id("run-1", "task-1")
        self.assertEqual(first.simulation_task_id, "task-1")
        self.assertEqual(second, first)

    def test_bind_different_task_id_conflicts(self):
        self.repository.create_run(make_run(simulation_task_id="task-1"))
        with self.assertRaises(TaskIdConflict):
            self.repository.bind_task_id("run-1", "task-2")

    def test_task_id_is_unique_across_runs(self):
        self.repository.create_run(make_run("run-1", simulation_task_id="task-1"))
        self.repository.create_run(make_run("run-2"))
        with self.assertRaises(TaskIdAlreadyBound):
            self.repository.bind_task_id("run-2", "task-1")
        self.assertIsNone(self.repository.get_run("run-2").simulation_task_id)

    def test_events_sort_and_allow_sequence_gaps(self):
        self.repository.create_run(make_run())
        self.repository.append_event(make_event(sequence=3))
        self.repository.append_event(make_event(sequence=1))
        self.assertEqual(
            [event.sequence for event in self.repository.list_events("run-1")],
            [1, 3],
        )

    def test_after_sequence_returns_only_greater_events(self):
        self.repository.create_run(make_run())
        for sequence in (1, 2, 4):
            self.repository.append_event(make_event(sequence=sequence))
        self.assertEqual(
            [event.sequence for event in self.repository.list_events("run-1", 2)],
            [4],
        )

    def test_identical_event_replay_is_idempotent(self):
        self.repository.create_run(make_run())
        event = make_event()
        self.assertEqual(
            self.repository.append_event(event), AppendEventResult.STORED
        )
        self.assertEqual(
            self.repository.append_event(event), AppendEventResult.DUPLICATE
        )

    def test_event_id_with_different_content_conflicts(self):
        self.repository.create_run(make_run())
        self.repository.append_event(make_event(event_id="same"))
        with self.assertRaises(EventConflict):
            self.repository.append_event(
                make_event(sequence=2, event_id="same", payload={"other": True})
            )

    def test_sequence_with_different_event_conflicts(self):
        self.repository.create_run(make_run())
        self.repository.append_event(make_event(sequence=1, event_id="first"))
        with self.assertRaises(EventSequenceConflict):
            self.repository.append_event(make_event(sequence=1, event_id="second"))

    def test_multiple_runs_keep_events_isolated(self):
        self.repository.create_run(make_run("run-1"))
        self.repository.create_run(make_run("run-2"))
        self.repository.append_event(make_event("run-1"))
        self.repository.append_event(make_event("run-2"))
        self.assertEqual(self.repository.list_events("run-1")[0].run_id, "run-1")
        self.assertEqual(self.repository.list_events("run-2")[0].run_id, "run-2")

    def test_reconstructing_repository_restores_runs_and_events(self):
        run = make_run()
        event = make_event()
        self.repository.create_run(run)
        self.repository.append_event(event)
        reopened = SQLiteRunRepository(self.database_path)
        self.assertEqual(reopened.get_run(run.run_id), run)
        self.assertEqual(reopened.list_events(run.run_id), [event])

    def test_failed_write_rolls_back_entire_transaction(self):
        self.repository.create_run(make_run("run-1", simulation_task_id="task-1"))
        with self.assertRaises(TaskIdAlreadyBound):
            self.repository.create_run(
                make_run("run-2", simulation_task_id="task-1")
            )
        with self.assertRaises(RunNotFound):
            self.repository.get_run("run-2")
        self.assertEqual([run.run_id for run in self.repository.list_runs()], ["run-1"])

    def test_basic_multithreaded_access_does_not_corrupt_database(self):
        self.repository.create_run(make_run())

        def append(sequence):
            return self.repository.append_event(make_event(sequence=sequence))

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(append, range(1, 81)))
        self.assertEqual(results, [AppendEventResult.STORED] * 80)
        self.assertEqual(
            [event.sequence for event in self.repository.list_events("run-1")],
            list(range(1, 81)),
        )
        self.assertEqual(SQLiteRunRepository(self.database_path).get_run("run-1"), make_run())

    def test_schema_enables_constraints_and_wal(self):
        with sqlite3.connect(str(self.database_path)) as connection:
            mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            foreign_keys = connection.execute(
                "PRAGMA foreign_key_list(run_events)"
            ).fetchall()
        self.assertEqual(mode.lower(), "wal")
        self.assertEqual(version, 1)
        self.assertTrue(foreign_keys)

    def test_foreign_key_rejects_event_for_unknown_run(self):
        with sqlite3.connect(str(self.database_path)) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    """
                    INSERT INTO run_events (
                        event_id, run_id, sequence, type, task_id, occurred_at,
                        schema_version, payload
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "event-unknown",
                        "missing",
                        1,
                        "metric",
                        None,
                        UTC_TIME.isoformat(),
                        "1.0",
                        "{}",
                    ),
                )

    def test_corrupt_json_raises_clear_repository_error(self):
        self.repository.create_run(make_run())
        with sqlite3.connect(str(self.database_path)) as connection:
            connection.execute(
                "UPDATE runs SET metrics_summary = ? WHERE run_id = ?",
                ("not-json", "run-1"),
            )
        with self.assertRaisesRegex(RepositoryDataError, "metrics_summary"):
            self.repository.get_run("run-1")

    def test_corrupt_enum_raises_clear_repository_error(self):
        self.repository.create_run(make_run())
        with sqlite3.connect(str(self.database_path)) as connection:
            connection.execute(
                "UPDATE runs SET status = ? WHERE run_id = ?", ("bogus", "run-1")
            )
        with self.assertRaisesRegex(RepositoryDataError, "Invalid persisted Run"):
            self.repository.get_run("run-1")


if __name__ == "__main__":
    unittest.main()
