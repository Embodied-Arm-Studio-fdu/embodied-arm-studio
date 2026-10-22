import tempfile
import unittest
from pathlib import Path
from uuid import UUID

from simulation_core import (
    ArtifactPathError,
    ArtifactPaths,
    FakeBackendClient,
    FakeScheduler,
    InMemoryTaskRepository,
    InvalidStateTransition,
    SimulationService,
    TaskStatus,
)


class ServiceTestCase(unittest.TestCase):
    def setUp(self):
        self.repository = InMemoryTaskRepository()
        self.scheduler = FakeScheduler()
        self.backend = FakeBackendClient()
        self.ids = iter(["task-1", "task-2", "task-3"])
        self.service = SimulationService(
            self.repository,
            self.scheduler,
            self.backend,
            task_id_factory=lambda: next(self.ids),
        )

    def create(self, run_id="run-1", project_id="project-1"):
        return self.service.create_task(
            run_id, project_id, {"environment": {"id": "future-push"}, "seed": 7}
        )

    def advance_to_running(self, task_id):
        self.service.transition_task_status(task_id, TaskStatus.INITIALIZING)
        return self.service.transition_task_status(task_id, TaskStatus.RUNNING)

    def test_first_create_generates_task_and_submits_once(self):
        task = self.create()
        self.assertEqual("task-1", task.task_id)
        self.assertEqual(TaskStatus.QUEUED, task.status)
        self.assertEqual([task.task_id], self.scheduler.submitted_task_ids)

    def test_default_task_id_is_a_uuid(self):
        service = SimulationService(
            InMemoryTaskRepository(), FakeScheduler(), FakeBackendClient()
        )
        task = service.create_task("run-uuid", "project-1")
        self.assertEqual(task.task_id, str(UUID(task.task_id)))

    def test_create_is_idempotent_by_run_id(self):
        first = self.create()
        second = self.service.create_task("run-1", "different-project", {"seed": 99})
        self.assertEqual(first.task_id, second.task_id)
        self.assertEqual(1, len(self.repository))
        self.assertEqual([first.task_id], self.scheduler.submitted_task_ids)
        self.assertEqual(1, len(self.backend.reported_tasks))

    def test_different_runs_get_different_task_ids(self):
        first = self.create("run-1")
        second = self.create("run-2")
        self.assertNotEqual(first.task_id, second.task_id)

    def test_repository_reads_by_run_and_task_id(self):
        task = self.create()
        self.assertEqual(task, self.repository.get_by_task_id(task.task_id))
        self.assertEqual(task, self.repository.get_by_run_id(task.run_id))
        self.assertTrue(self.repository.has_run_id(task.run_id))
        self.assertFalse(self.repository.has_run_id("missing"))

    def test_normal_state_chain(self):
        task = self.create()
        for status in (
            TaskStatus.INITIALIZING,
            TaskStatus.RUNNING,
            TaskStatus.FINALIZING,
            TaskStatus.COMPLETED,
        ):
            task = self.service.transition_task_status(task.task_id, status)
        self.assertEqual(TaskStatus.COMPLETED, task.status)

    def test_completed_cannot_return_to_running(self):
        task = self.create()
        self.advance_to_running(task.task_id)
        self.service.transition_task_status(task.task_id, TaskStatus.FINALIZING)
        self.service.transition_task_status(task.task_id, TaskStatus.COMPLETED)
        with self.assertRaises(InvalidStateTransition):
            self.service.transition_task_status(task.task_id, TaskStatus.RUNNING)

    def test_failed_cannot_return_to_running(self):
        task = self.create()
        self.service.transition_task_status(task.task_id, TaskStatus.INITIALIZING)
        self.service.transition_task_status(task.task_id, TaskStatus.FAILED)
        with self.assertRaises(InvalidStateTransition):
            self.service.transition_task_status(task.task_id, TaskStatus.RUNNING)

    def test_stopped_cannot_return_to_running(self):
        task = self.create()
        self.advance_to_running(task.task_id)
        self.service.request_stop(task.task_id)
        self.service.transition_task_status(task.task_id, TaskStatus.STOPPED)
        with self.assertRaises(InvalidStateTransition):
            self.service.transition_task_status(task.task_id, TaskStatus.RUNNING)

    def test_illegal_transition_is_rejected(self):
        task = self.create()
        with self.assertRaisesRegex(
            InvalidStateTransition, "queued -> completed"
        ):
            self.service.transition_task_status(task.task_id, TaskStatus.COMPLETED)

    def test_running_stop_enters_stopping(self):
        task = self.create()
        self.advance_to_running(task.task_id)
        stopped = self.service.request_stop(task.task_id)
        self.assertEqual(TaskStatus.STOPPING, stopped.status)

    def test_repeated_stop_is_idempotent_for_stopping_and_stopped(self):
        task = self.create()
        self.advance_to_running(task.task_id)
        stopping = self.service.request_stop(task.task_id)
        self.assertEqual(stopping, self.service.request_stop(task.task_id))
        stopped = self.service.transition_task_status(task.task_id, TaskStatus.STOPPED)
        self.assertEqual(stopped, self.service.request_stop(task.task_id))
        self.assertEqual(1, len(self.repository))

    def test_stop_preserves_completed_and_failed_terminal_states(self):
        completed = self.create("run-completed")
        self.advance_to_running(completed.task_id)
        self.service.transition_task_status(completed.task_id, TaskStatus.FINALIZING)
        completed = self.service.transition_task_status(
            completed.task_id, TaskStatus.COMPLETED
        )
        self.assertEqual(completed, self.service.request_stop(completed.task_id))

        failed = self.create("run-failed")
        self.service.transition_task_status(failed.task_id, TaskStatus.INITIALIZING)
        failed = self.service.transition_task_status(failed.task_id, TaskStatus.FAILED)
        self.assertEqual(failed, self.service.request_stop(failed.task_id))

    def test_queued_stop_cancels_without_entering_stopping(self):
        task = self.create()
        stopped = self.service.request_stop(task.task_id)
        self.assertEqual(TaskStatus.STOPPED, stopped.status)
        self.assertEqual([task.task_id], self.scheduler.cancelled_task_ids)

    def test_backend_boundary_receives_created_and_transitioned_statuses(self):
        task = self.create()
        self.service.transition_task_status(task.task_id, TaskStatus.INITIALIZING)
        self.assertEqual(
            [TaskStatus.QUEUED, TaskStatus.INITIALIZING],
            [reported.status for reported in self.backend.reported_tasks],
        )

    def test_configuration_is_snapshotted(self):
        source = {"nested": {"seed": 7}}
        task = self.service.create_task("run-snapshot", "project-1", source)
        source["nested"]["seed"] = 99
        self.assertEqual(7, task.config.values["nested"]["seed"])


class ArtifactPathsTestCase(unittest.TestCase):
    def test_project_and_run_paths_are_isolated_without_creating_them(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = ArtifactPaths(Path(directory) / "artifacts")
            first = paths.run_root("project-a", "run-a")
            second = paths.run_root("project-b", "run-b")
            self.assertNotEqual(first, second)
            self.assertEqual(
                Path(directory).resolve() / "artifacts" / "project-a" / "run-a",
                first,
            )
            self.assertFalse(first.exists())

    def test_dangerous_components_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = ArtifactPaths(Path(directory) / "artifacts")
            for project_id, run_id in (
                ("..", "run"),
                ("../outside", "run"),
                ("project", "../../outside"),
                ("/absolute", "run"),
                ("project\\outside", "run"),
                ("", "run"),
            ):
                with self.subTest(project_id=project_id, run_id=run_id):
                    with self.assertRaises(ArtifactPathError):
                        paths.run_root(project_id, run_id)


if __name__ == "__main__":
    unittest.main()
