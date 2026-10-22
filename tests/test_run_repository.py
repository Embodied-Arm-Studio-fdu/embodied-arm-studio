import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from robotics.runs import (
    AppendEventResult,
    EventConflict,
    EventSequenceConflict,
    ExecutionMode,
    InMemoryRunRepository,
    Run,
    RunAlreadyExists,
    RunEvent,
    RunEventType,
    RunNotFound,
    RunStatus,
    TaskIdConflict,
)


UTC_TIME = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)


def make_run(run_id="run-1", created_at=UTC_TIME):
    return Run(
        run_id=run_id,
        user_id="user-1",
        project_id="project-1",
        status=RunStatus.QUEUED,
        execution_mode=ExecutionMode.MOCK,
        code_sha256="a" * 64,
        parameters={"seed": 7},
        created_at=created_at,
        updated_at=created_at,
    )


def make_event(run_id="run-1", sequence=1, event_id=None, payload=None):
    return RunEvent(
        event_id=event_id or "event-%s-%s" % (run_id, sequence),
        sequence=sequence,
        type=RunEventType.METRIC,
        run_id=run_id,
        occurred_at=UTC_TIME,
        payload=payload if payload is not None else {"step": sequence},
    )


class InMemoryRunRepositoryTest(unittest.TestCase):
    def setUp(self):
        self.repository = InMemoryRunRepository()

    def test_create_get_and_list_newest_first(self):
        older = make_run("older")
        newer = make_run("newer", UTC_TIME + timedelta(seconds=1))
        self.repository.create_run(older)
        self.repository.create_run(newer)
        self.assertEqual(self.repository.get_run("older"), older)
        self.assertEqual(
            [run.run_id for run in self.repository.list_runs()], ["newer", "older"]
        )

    def test_duplicate_run_id_fails(self):
        self.repository.create_run(make_run())
        with self.assertRaises(RunAlreadyExists):
            self.repository.create_run(make_run())

    def test_unknown_run_fails(self):
        with self.assertRaises(RunNotFound):
            self.repository.get_run("missing")
        with self.assertRaises(RunNotFound):
            self.repository.save_run(make_run("missing"))

    def test_save_updates_content(self):
        original = self.repository.create_run(make_run())
        updated = replace(original, status=RunStatus.INITIALIZING, progress=0.2)
        self.repository.save_run(updated)
        self.assertEqual(self.repository.get_run("run-1"), updated)

    def test_save_cannot_change_run_id(self):
        original = self.repository.create_run(make_run())
        with self.assertRaises(RunNotFound):
            self.repository.save_run(replace(original, run_id="different"))
        self.assertEqual(self.repository.get_run("run-1").run_id, "run-1")

    def test_returned_values_do_not_mutate_storage(self):
        source = make_run()
        created = self.repository.create_run(source)
        source.parameters["seed"] = 99
        created.parameters["seed"] = 88
        listed = self.repository.list_runs()[0]
        listed.parameters["seed"] = 77
        self.assertEqual(self.repository.get_run("run-1").parameters["seed"], 7)

        event = make_event(payload={"nested": {"value": 1}})
        self.repository.append_event(event)
        event.payload["nested"]["value"] = 2
        returned = self.repository.list_events("run-1")[0]
        returned.payload["nested"]["value"] = 3
        self.assertEqual(
            self.repository.list_events("run-1")[0].payload["nested"]["value"], 1
        )

    def test_task_id_first_binding_and_same_id_are_idempotent(self):
        self.repository.create_run(make_run())
        first = self.repository.bind_task_id("run-1", "task-1")
        second = self.repository.bind_task_id("run-1", "task-1")
        self.assertEqual(first.simulation_task_id, "task-1")
        self.assertEqual(second, first)

    def test_different_task_id_conflicts(self):
        self.repository.create_run(make_run())
        self.repository.bind_task_id("run-1", "task-1")
        with self.assertRaises(TaskIdConflict):
            self.repository.bind_task_id("run-1", "task-2")

    def test_event_first_write_and_identical_replay(self):
        self.repository.create_run(make_run())
        event = make_event()
        self.assertEqual(
            self.repository.append_event(event), AppendEventResult.STORED
        )
        self.assertEqual(
            self.repository.append_event(event), AppendEventResult.DUPLICATE
        )

    def test_same_event_id_with_different_content_conflicts(self):
        self.repository.create_run(make_run())
        self.repository.append_event(make_event(event_id="shared"))
        with self.assertRaises(EventConflict):
            self.repository.append_event(
                make_event(sequence=2, event_id="shared", payload={"step": 2})
            )

    def test_same_run_sequence_conflicts(self):
        self.repository.create_run(make_run())
        self.repository.append_event(make_event(sequence=3, event_id="first"))
        with self.assertRaises(EventSequenceConflict):
            self.repository.append_event(make_event(sequence=3, event_id="second"))

    def test_events_are_sorted_and_after_sequence_filters(self):
        self.repository.create_run(make_run())
        for sequence in (7, 1, 4):
            self.repository.append_event(make_event(sequence=sequence))
        self.assertEqual(
            [event.sequence for event in self.repository.list_events("run-1")],
            [1, 4, 7],
        )
        self.assertEqual(
            [
                event.sequence
                for event in self.repository.list_events("run-1", after_sequence=4)
            ],
            [7],
        )

    def test_unknown_run_cannot_receive_event(self):
        with self.assertRaises(RunNotFound):
            self.repository.append_event(make_event(run_id="missing"))

    def test_runs_have_isolated_event_streams(self):
        self.repository.create_run(make_run("run-1"))
        self.repository.create_run(make_run("run-2"))
        self.repository.append_event(make_event("run-1", 1))
        self.repository.append_event(make_event("run-2", 1))
        self.assertEqual(
            [event.run_id for event in self.repository.list_events("run-1")],
            ["run-1"],
        )
        self.assertEqual(
            [event.run_id for event in self.repository.list_events("run-2")],
            ["run-2"],
        )

    def test_concurrent_reads_and_writes_are_consistent(self):
        self.repository.create_run(make_run())

        def write(sequence):
            return self.repository.append_event(make_event(sequence=sequence))

        def read(_):
            self.repository.get_run("run-1")
            return self.repository.list_events("run-1")

        with ThreadPoolExecutor(max_workers=8) as pool:
            writes = [pool.submit(write, sequence) for sequence in range(1, 101)]
            reads = [pool.submit(read, index) for index in range(40)]
            self.assertTrue(
                all(future.result() == AppendEventResult.STORED for future in writes)
            )
            for future in reads:
                future.result()
        self.assertEqual(
            [event.sequence for event in self.repository.list_events("run-1")],
            list(range(1, 101)),
        )


if __name__ == "__main__":
    unittest.main()
