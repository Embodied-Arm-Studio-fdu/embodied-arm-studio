import hashlib
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from robotics.runs import (
    ApplyEventResult,
    ExecutionMode,
    InMemoryRunRepository,
    InvalidRunCreateRequest,
    InvalidRunEvent,
    InvalidRunStateTransition,
    RunEvent,
    RunEventType,
    RunNotFound,
    RunService,
    RunStatus,
    RunStopNotAllowed,
    RunTaskMismatch,
    TaskIdConflict,
)
from robotics.runs.service import (
    calculate_run_progress,
    expected_training_steps,
)


UTC_TIME = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)


class AdvancingClock:
    def __init__(self, start=UTC_TIME):
        self.start = start
        self.calls = 0

    def __call__(self):
        value = self.start + timedelta(seconds=self.calls)
        self.calls += 1
        return value


class RunServiceTest(unittest.TestCase):
    def setUp(self):
        self.repository = InMemoryRunRepository()
        self.clock = AdvancingClock()
        self.next_run_number = 1

        def run_id_factory():
            value = "run-%s" % self.next_run_number
            self.next_run_number += 1
            return value

        self.service = RunService(
            self.repository,
            clock=self.clock,
            run_id_factory=run_id_factory,
        )

    def create_run(self, execution_mode=ExecutionMode.MOCK, **changes):
        values = {
            "user_id": "user-1",
            "project_id": "project-1",
            "execution_mode": execution_mode,
            "parameters": {"seed": 7},
            "project_config_snapshot": {"scene": {"name": "push"}},
            "code": "print('hello')\n",
        }
        values.update(changes)
        return self.service.create_run(**values)

    def event(
        self,
        run_id,
        sequence,
        event_type,
        payload,
        task_id=None,
        event_id=None,
    ):
        return RunEvent(
            event_id=event_id or "event-%s-%s" % (run_id, sequence),
            sequence=sequence,
            type=event_type,
            run_id=run_id,
            task_id=task_id,
            occurred_at=UTC_TIME,
            payload=payload,
        )

    def advance_to_running(self, run_id):
        self.service.transition_status(run_id, RunStatus.INITIALIZING)
        return self.service.transition_status(run_id, RunStatus.RUNNING)

    def advance_to_completed(self, run_id):
        self.advance_to_running(run_id)
        self.service.transition_status(run_id, RunStatus.FINALIZING)
        return self.service.transition_status(run_id, RunStatus.COMPLETED)

    def test_create_run_has_canonical_initial_state(self):
        run = self.create_run()
        self.assertEqual(run.run_id, "run-1")
        self.assertEqual(run.status, RunStatus.QUEUED)
        self.assertEqual(run.progress, 0)
        self.assertEqual(run.last_sequence, 0)
        self.assertIsNone(run.simulation_task_id)
        self.assertEqual(run.metrics_summary, {})
        self.assertIsNone(run.error)
        self.assertIsNone(run.started_at)
        self.assertIsNone(run.ended_at)
        self.assertEqual(run.created_at, UTC_TIME)
        self.assertEqual(run.updated_at, UTC_TIME)
        self.assertEqual(self.service.get_run(run.run_id), run)
        self.assertEqual(self.service.list_runs(), [run])

    def test_fixed_clock_and_run_id_are_injected(self):
        fixed = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
        service = RunService(
            InMemoryRunRepository(),
            clock=lambda: fixed,
            run_id_factory=lambda: "fixed-run",
        )
        run = service.create_run(
            user_id="user",
            project_id="project",
            execution_mode=ExecutionMode.MOCK,
            parameters={},
            project_config_snapshot={},
            code="",
        )
        self.assertEqual(run.run_id, "fixed-run")
        self.assertEqual(run.created_at, fixed)
        self.assertEqual(run.updated_at, fixed)

    def test_code_sha256_is_computed_from_utf8_content(self):
        code = "机械臂 = 'push'\n"
        run = self.create_run(code=code)
        self.assertEqual(
            run.code_sha256, hashlib.sha256(code.encode("utf-8")).hexdigest()
        )

    def test_create_copies_mutable_input(self):
        parameters = {"nested": {"seed": 7}}
        snapshot = {"scene": {"name": "push"}}
        run = self.create_run(
            parameters=parameters, project_config_snapshot=snapshot
        )
        parameters["nested"]["seed"] = 99
        snapshot["scene"]["name"] = "changed"
        saved = self.service.get_run(run.run_id)
        self.assertEqual(saved.parameters["nested"]["seed"], 7)
        self.assertEqual(saved.project_config_snapshot["scene"]["name"], "push")

    def test_invalid_create_request_uses_service_error(self):
        with self.assertRaises(InvalidRunCreateRequest):
            self.create_run(user_id="")
        with self.assertRaises(InvalidRunCreateRequest):
            self.create_run(execution_mode="mock")

    def test_simulation_task_first_binding(self):
        run = self.create_run(ExecutionMode.SIMULATION)
        bound = self.service.bind_simulation_task(run.run_id, "task-1")
        self.assertEqual(bound.simulation_task_id, "task-1")
        self.assertGreater(bound.updated_at, run.updated_at)

    def test_same_simulation_task_binding_is_idempotent(self):
        run = self.create_run(ExecutionMode.SIMULATION)
        first = self.service.bind_simulation_task(run.run_id, "task-1")
        second = self.service.bind_simulation_task(run.run_id, "task-1")
        self.assertEqual(second, first)

    def test_different_simulation_task_binding_conflicts(self):
        run = self.create_run(ExecutionMode.SIMULATION)
        self.service.bind_simulation_task(run.run_id, "task-1")
        with self.assertRaises(TaskIdConflict):
            self.service.bind_simulation_task(run.run_id, "task-2")

    def test_mock_run_rejects_simulation_task_binding(self):
        run = self.create_run()
        with self.assertRaises(RunTaskMismatch):
            self.service.bind_simulation_task(run.run_id, "task-1")

    def test_empty_task_id_is_rejected(self):
        run = self.create_run(ExecutionMode.SIMULATION)
        with self.assertRaises(RunTaskMismatch):
            self.service.bind_simulation_task(run.run_id, " ")

    def test_complete_status_chain_and_timestamps(self):
        run = self.create_run()
        initializing = self.service.transition_status(
            run.run_id, RunStatus.INITIALIZING
        )
        running = self.service.transition_status(run.run_id, RunStatus.RUNNING)
        finalizing = self.service.transition_status(
            run.run_id, RunStatus.FINALIZING
        )
        completed = self.service.transition_status(
            run.run_id, RunStatus.COMPLETED
        )
        self.assertEqual(
            [
                run.status,
                initializing.status,
                running.status,
                finalizing.status,
                completed.status,
            ],
            [
                RunStatus.QUEUED,
                RunStatus.INITIALIZING,
                RunStatus.RUNNING,
                RunStatus.FINALIZING,
                RunStatus.COMPLETED,
            ],
        )
        self.assertEqual(running.started_at, initializing.started_at)
        self.assertEqual(finalizing.started_at, initializing.started_at)
        self.assertIsNone(initializing.ended_at)
        self.assertIsNone(running.ended_at)
        self.assertIsNone(finalizing.ended_at)
        self.assertIsNotNone(completed.ended_at)
        self.assertEqual(completed.progress, 1)

    def test_idempotent_status_write_does_not_replace_timestamps(self):
        run = self.create_run()
        initializing = self.service.transition_status(
            run.run_id, RunStatus.INITIALIZING
        )
        repeated = self.service.transition_status(
            run.run_id, RunStatus.INITIALIZING
        )
        self.assertEqual(repeated, initializing)
        completed = self.advance_to_completed(run.run_id)
        repeated_terminal = self.service.transition_status(
            run.run_id, RunStatus.COMPLETED
        )
        self.assertEqual(repeated_terminal.ended_at, completed.ended_at)
        self.assertEqual(repeated_terminal.updated_at, completed.updated_at)

    def test_invalid_status_transition_fails(self):
        run = self.create_run()
        with self.assertRaises(InvalidRunStateTransition):
            self.service.transition_status(run.run_id, RunStatus.RUNNING)
        self.assertEqual(self.service.get_run(run.run_id).status, RunStatus.QUEUED)

    def test_state_event_applies_legal_transition(self):
        run = self.create_run()
        event = self.event(
            run.run_id,
            1,
            RunEventType.STATE,
            {"previous_status": "queued", "status": "initializing"},
        )
        self.assertEqual(self.service.apply_event(event), ApplyEventResult.APPLIED)
        saved = self.service.get_run(run.run_id)
        self.assertEqual(saved.status, RunStatus.INITIALIZING)
        self.assertEqual(saved.last_sequence, 1)
        self.assertIsNotNone(saved.started_at)

    def test_unknown_state_event_is_a_domain_error(self):
        run = self.create_run()
        event = self.event(
            run.run_id,
            1,
            RunEventType.STATE,
            {"previous_status": "queued", "status": "mystery"},
        )
        with self.assertRaises(InvalidRunEvent):
            self.service.apply_event(event)
        self.assertEqual(self.service.get_run(run.run_id).status, RunStatus.QUEUED)

    def test_state_event_cannot_bypass_state_machine(self):
        run = self.create_run()
        event = self.event(
            run.run_id,
            1,
            RunEventType.STATE,
            {"previous_status": "queued", "status": "completed"},
        )
        with self.assertRaises(InvalidRunStateTransition):
            self.service.apply_event(event)
        self.assertEqual(self.service.get_run(run.run_id).status, RunStatus.QUEUED)

    def test_metric_event_updates_summary_and_progress(self):
        run = self.create_run()
        event = self.event(
            run.run_id,
            1,
            RunEventType.METRIC,
            {
                "step": 80,
                "metrics": {"average_reward": 0.82, "policy_loss": 0.12},
                "progress": 0.4,
            },
        )
        self.assertEqual(self.service.apply_event(event), ApplyEventResult.APPLIED)
        saved = self.service.get_run(run.run_id)
        self.assertEqual(
            saved.metrics_summary,
            {"average_reward": 0.82, "policy_loss": 0.12, "step": 80},
        )
        self.assertEqual(saved.progress, 0.4)
        self.assertEqual(saved.training_steps, 80)

    def test_metric_progress_cannot_regress(self):
        run = self.create_run()
        for sequence, progress in ((1, 0.8), (2, 0.2)):
            self.service.apply_event(
                self.event(
                    run.run_id,
                    sequence,
                    RunEventType.METRIC,
                    {
                        "step": sequence,
                        "metrics": {"average_reward": progress},
                        "progress": progress,
                    },
                )
            )
        self.assertEqual(self.service.get_run(run.run_id).progress, 0.8)

    def test_heartbeat_advances_training_steps_without_overwriting_metrics(self):
        run = self.create_run()
        before = self.service.get_run(run.run_id)
        result = self.service.apply_event(
            self.event(
                run.run_id,
                1,
                RunEventType.HEARTBEAT,
                {"status": "running", "training_steps": 10},
            )
        )
        after = self.service.get_run(run.run_id)
        self.assertEqual(result, ApplyEventResult.APPLIED)
        self.assertEqual(after.status, before.status)
        self.assertEqual(after.progress, before.progress)
        self.assertEqual(after.training_steps, 10)
        self.assertEqual(after.metrics_summary, before.metrics_summary)
        self.assertEqual(after.last_sequence, 1)
        self.assertGreater(after.updated_at, before.updated_at)

    def test_frozen_expected_steps_and_progress_rules_are_pure(self):
        run = self.create_run(
            ExecutionMode.SIMULATION,
            project_config_snapshot={"expected_training_steps": 100},
        )
        self.assertEqual(expected_training_steps(run.project_config_snapshot), 100)
        self.assertEqual(
            calculate_run_progress(
                run, status=RunStatus.RUNNING, training_steps=40
            ),
            0.4,
        )
        self.assertEqual(
            calculate_run_progress(
                run, status=RunStatus.RUNNING, training_steps=100
            ),
            0.99,
        )
        self.assertEqual(
            calculate_run_progress(
                run, status=RunStatus.COMPLETED, training_steps=0
            ),
            1.0,
        )
        known = replace(run, progress=0.3)
        self.assertEqual(
            calculate_run_progress(
                known, status=RunStatus.INITIALIZING, training_steps=80
            ),
            0.3,
        )
        for invalid in (None, True, False, 0, -1, 100.0, "100"):
            with self.subTest(invalid=invalid):
                snapshot = {"expected_training_steps": invalid}
                self.assertIsNone(expected_training_steps(snapshot))
                historical = self.create_run(
                    ExecutionMode.SIMULATION,
                    project_config_snapshot=snapshot,
                )
                historical = replace(historical, progress=0.3)
                self.assertEqual(
                    calculate_run_progress(
                        historical,
                        status=RunStatus.RUNNING,
                        training_steps=80,
                    ),
                    0.3,
                )

    def test_formal_metric_and_heartbeat_progress_are_monotonic(self):
        run = self.create_run(
            ExecutionMode.SIMULATION,
            parameters={"max_steps": 10},
            project_config_snapshot={"expected_training_steps": 100},
        )
        self.service.bind_simulation_task(run.run_id, "task-progress")
        self.service.transition_status(run.run_id, RunStatus.INITIALIZING)
        self.service.transition_status(run.run_id, RunStatus.RUNNING)
        events = (
            self.event(
                run.run_id,
                1,
                RunEventType.METRIC,
                {
                    "step": 40,
                    "metrics": {"update_index": 2, "average_reward": 0.4},
                },
                task_id="task-progress",
            ),
            self.event(
                run.run_id,
                2,
                RunEventType.HEARTBEAT,
                {"status": "running", "training_steps": 60},
                task_id="task-progress",
            ),
            self.event(
                run.run_id,
                3,
                RunEventType.METRIC,
                {
                    "step": 20,
                    "metrics": {"update_index": 1, "average_reward": 0.2},
                },
                task_id="task-progress",
            ),
        )
        for event in events:
            self.service.apply_simulation_event(event)
        saved = self.service.get_run(run.run_id)
        self.assertEqual(saved.training_steps, 60)
        self.assertEqual(saved.progress, 0.6)
        self.assertEqual(saved.metrics_summary["step"], 40)
        self.assertEqual(saved.metrics_summary["update_index"], 2)
        self.assertEqual(saved.metrics_summary["average_reward"], 0.2)

    def test_historical_simulation_does_not_use_parameters_max_steps(self):
        run = self.create_run(
            ExecutionMode.SIMULATION,
            parameters={"max_steps": 10},
            project_config_snapshot={},
        )
        self.service.bind_simulation_task(run.run_id, "task-history")
        self.service.transition_status(run.run_id, RunStatus.INITIALIZING)
        self.service.transition_status(run.run_id, RunStatus.RUNNING)
        self.service.apply_simulation_event(
            self.event(
                run.run_id,
                1,
                RunEventType.METRIC,
                {"step": 10, "metrics": {"average_reward": 0.1}},
                task_id="task-history",
            )
        )
        saved = self.service.get_run(run.run_id)
        self.assertEqual(saved.training_steps, 10)
        self.assertEqual(saved.progress, 0.0)

    def test_metric_update_index_validation_rejects_invalid_values(self):
        for invalid in (0, -1, 1.0, "1", True):
            run = self.create_run()
            with self.subTest(invalid=invalid):
                with self.assertRaises(InvalidRunEvent):
                    self.service.apply_event(
                        self.event(
                            run.run_id,
                            1,
                            RunEventType.METRIC,
                            {
                                "step": 1,
                                "metrics": {"update_index": invalid},
                            },
                        )
                    )

    def test_metric_extension_values_remain_named_and_finite(self):
        for metrics in (
            {" ": 1.0},
            {"custom": float("nan")},
            {"custom": float("inf")},
        ):
            run = self.create_run()
            with self.subTest(metrics=metrics):
                with self.assertRaises(InvalidRunEvent):
                    self.service.apply_event(
                        self.event(
                            run.run_id,
                            1,
                            RunEventType.METRIC,
                            {"step": 1, "metrics": metrics},
                        )
                    )

    def test_error_event_records_bounded_structured_error_without_failing(self):
        run = self.create_run()
        event = self.event(
            run.run_id,
            1,
            RunEventType.ERROR,
            {"code": "WORKER_FAILED", "message": "x" * 1200, "retryable": False},
        )
        self.service.apply_event(event)
        saved = self.service.get_run(run.run_id)
        self.assertEqual(saved.error.code, "WORKER_FAILED")
        self.assertEqual(len(saved.error.message), 1000)
        self.assertFalse(saved.error.retryable)
        self.assertEqual(saved.status, RunStatus.QUEUED)

    def test_identical_event_replay_is_idempotent(self):
        run = self.create_run()
        event = self.event(
            run.run_id,
            1,
            RunEventType.METRIC,
            {"step": 1, "metrics": {"average_reward": 0.5}},
        )
        self.assertEqual(self.service.apply_event(event), ApplyEventResult.APPLIED)
        first = self.service.get_run(run.run_id)
        self.assertEqual(self.service.apply_event(event), ApplyEventResult.DUPLICATE)
        self.assertEqual(self.service.get_run(run.run_id), first)

    def test_late_event_is_stored_without_overwriting_snapshot(self):
        run = self.create_run()
        newer = self.event(
            run.run_id,
            2,
            RunEventType.METRIC,
            {
                "step": 20,
                "metrics": {"average_reward": 0.9},
                "progress": 0.8,
            },
        )
        older = self.event(
            run.run_id,
            1,
            RunEventType.METRIC,
            {
                "step": 10,
                "metrics": {"average_reward": 0.1},
                "progress": 0.1,
            },
        )
        self.service.apply_event(newer)
        snapshot = self.service.get_run(run.run_id)
        self.assertEqual(self.service.apply_event(older), ApplyEventResult.LATE)
        self.assertEqual(self.service.get_run(run.run_id), snapshot)
        self.assertEqual(
            [item.sequence for item in self.repository.list_events(run.run_id)],
            [1, 2],
        )

    def test_simulation_event_task_must_match_binding(self):
        run = self.create_run(ExecutionMode.SIMULATION)
        self.service.bind_simulation_task(run.run_id, "task-1")
        event = self.event(
            run.run_id,
            1,
            RunEventType.HEARTBEAT,
            {"status": "running", "training_steps": 0},
            task_id="task-other",
        )
        with self.assertRaises(RunTaskMismatch):
            self.service.apply_event(event)

    def test_simulation_event_requires_task_id(self):
        run = self.create_run(ExecutionMode.SIMULATION)
        self.service.bind_simulation_task(run.run_id, "task-1")
        event = self.event(
            run.run_id,
            1,
            RunEventType.HEARTBEAT,
            {"status": "running", "training_steps": 0},
        )
        with self.assertRaises(RunTaskMismatch):
            self.service.apply_event(event)

    def test_terminal_run_is_not_rolled_back_by_late_running_event(self):
        run = self.create_run()
        self.service.apply_event(
            self.event(
                run.run_id,
                2,
                RunEventType.METRIC,
                {"step": 2, "metrics": {"average_reward": 1.0}},
            )
        )
        completed = self.advance_to_completed(run.run_id)
        late = self.event(
            run.run_id,
            1,
            RunEventType.STATE,
            {"previous_status": "queued", "status": "running"},
        )
        self.assertEqual(self.service.apply_event(late), ApplyEventResult.LATE)
        saved = self.service.get_run(run.run_id)
        self.assertEqual(saved.status, RunStatus.COMPLETED)
        self.assertEqual(saved.ended_at, completed.ended_at)

    def test_queued_stop_enters_stopped_and_repeats_idempotently(self):
        run = self.create_run()
        stopped = self.service.request_stop(run.run_id)
        repeated = self.service.request_stop(run.run_id)
        self.assertEqual(stopped.status, RunStatus.STOPPED)
        self.assertIsNotNone(stopped.ended_at)
        self.assertEqual(repeated, stopped)

    def test_bound_simulation_queued_stop_enters_stopping(self):
        run = self.create_run(ExecutionMode.SIMULATION)
        bound = self.service.bind_simulation_task(run.run_id, "task-queued")
        stopping = self.service.request_stop(bound.run_id)
        self.assertEqual(stopping.status, RunStatus.STOPPING)
        self.assertIsNone(stopping.ended_at)

    def test_administrative_reconciliation_is_atomic_and_append_only(self):
        run = self.create_run(ExecutionMode.SIMULATION)
        stopped = self.service.administratively_reconcile_queued_unbound(
            run.run_id,
            "stale before-run recovery target with no Simulation task",
            event_id="administrative-event-1",
        )
        self.assertEqual(stopped.status, RunStatus.STOPPED)
        self.assertEqual(stopped.stop_reason, "administrative_reconciliation")
        self.assertEqual(stopped.last_sequence, 1)
        event = self.service.list_events(run.run_id)[0]
        self.assertEqual(event.event_id, "administrative-event-1")
        self.assertIsNone(event.task_id)
        self.assertEqual(
            event.payload,
            {
                "previous_status": "queued",
                "status": "stopped",
                "administrative_reconciliation": True,
                "reason": "stale before-run recovery target with no Simulation task",
            },
        )
        with self.assertRaises(RunStopNotAllowed):
            self.service.administratively_reconcile_queued_unbound(
                run.run_id, "must not replay a terminal Run"
            )

    def test_administrative_reconciliation_rejects_bound_or_mock_run(self):
        mock = self.create_run()
        with self.assertRaises(RunStopNotAllowed):
            self.service.administratively_reconcile_queued_unbound(
                mock.run_id, "not simulation"
            )
        simulation = self.create_run(ExecutionMode.SIMULATION)
        self.service.bind_simulation_task(simulation.run_id, "task-bound")
        with self.assertRaises(RunStopNotAllowed):
            self.service.administratively_reconcile_queued_unbound(
                simulation.run_id, "already bound"
            )

    def test_running_stop_enters_stopping_and_repeats_idempotently(self):
        run = self.create_run()
        self.advance_to_running(run.run_id)
        stopping = self.service.request_stop(run.run_id)
        repeated = self.service.request_stop(run.run_id)
        self.assertEqual(stopping.status, RunStatus.STOPPING)
        self.assertIsNone(stopping.ended_at)
        self.assertEqual(repeated, stopping)

    def test_terminal_stop_requests_are_idempotent(self):
        completed_run = self.create_run()
        completed = self.advance_to_completed(completed_run.run_id)
        self.assertEqual(self.service.request_stop(completed.run_id), completed)

        failed_run = self.create_run()
        self.service.transition_status(failed_run.run_id, RunStatus.INITIALIZING)
        failed = self.service.transition_status(failed_run.run_id, RunStatus.FAILED)
        self.assertEqual(self.service.request_stop(failed.run_id), failed)

        stopped_run = self.create_run()
        stopped = self.service.request_stop(stopped_run.run_id)
        self.assertEqual(self.service.request_stop(stopped.run_id), stopped)

    def test_initializing_and_finalizing_stop_are_rejected(self):
        initializing = self.create_run()
        self.service.transition_status(initializing.run_id, RunStatus.INITIALIZING)
        with self.assertRaises(RunStopNotAllowed):
            self.service.request_stop(initializing.run_id)

        finalizing = self.create_run()
        self.advance_to_running(finalizing.run_id)
        self.service.transition_status(finalizing.run_id, RunStatus.FINALIZING)
        with self.assertRaises(RunStopNotAllowed):
            self.service.request_stop(finalizing.run_id)

    def test_unknown_run_operations_fail(self):
        missing_event = self.event(
            "missing",
            1,
            RunEventType.HEARTBEAT,
            {"status": "running", "training_steps": 0},
        )
        operations = (
            lambda: self.service.get_run("missing"),
            lambda: self.service.bind_simulation_task("missing", "task"),
            lambda: self.service.transition_status("missing", RunStatus.STOPPED),
            lambda: self.service.apply_event(missing_event),
            lambda: self.service.request_stop("missing"),
        )
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaises(RunNotFound):
                    operation()


if __name__ == "__main__":
    unittest.main()
