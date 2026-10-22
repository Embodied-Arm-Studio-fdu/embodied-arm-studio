"""Dispatch rejection convergence without a GPU, network or production database."""
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from robotics.runs import (ExecutionMode, InMemoryRunRepository, Run, RunError,
                           RunEvent, RunEventType, RunService, RunStatus)
from robotics.runs.repository import validate_run_deletion
from robotics.runs.sqlite_repository import SQLiteRunRepository
from robotics.simulation import (SimulationDispatchService, SimulationRejectedError,
    SimulationTimeoutError, SimulationSnapshotMismatch, SimulationRecoveryService)

NOW = datetime(2026, 10, 14, tzinfo=timezone.utc)


def make_run(run_id='orphan', **changes):
    values = dict(run_id=run_id, user_id='owner', project_id='project',
                  status=RunStatus.QUEUED, execution_mode=ExecutionMode.SIMULATION,
                  code_sha256='a'*64, created_at=NOW-timedelta(hours=63), updated_at=NOW)
    values.update(changes)
    return Run(**values)


class RepositoryCases:
    def setUp(self):
        self.repository = self.make_repository()
        self.service = RunService(self.repository, clock=lambda: NOW)
        self.repository.create_run(make_run())

    def test_definitive_rejection_is_terminal_audited_and_deletable(self):
        result = self.service.record_dispatch_error('orphan', 'DISPATCH_REJECTED', 'Invalid config', definitive=True)
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(result.ended_at, NOW)
        self.assertIsNone(result.started_at)
        self.assertFalse(result.error.retryable)
        self.assertEqual(result.last_sequence, 1)
        self.assertEqual(self.repository.list_events('orphan')[0].payload['status'], 'failed')
        validate_run_deletion([result], ['orphan'], 'owner')
        # Reconciliation is idempotent and cannot rewrite the terminal reason.
        again = self.service.record_dispatch_error('orphan', 'DISPATCH_OTHER', 'Other', definitive=True)
        self.assertEqual(again, result)
        self.assertEqual(len(self.repository.list_events('orphan')), 1)

    def test_timeout_preserves_event_sequence_and_clears_on_binding(self):
        queued = self.service.record_dispatch_error('orphan', 'DISPATCH_UNCONFIRMED', 'Unknown', definitive=False)
        self.assertEqual(queued.status, RunStatus.QUEUED)
        self.assertEqual(queued.last_sequence, 0)
        self.assertIsNone(queued.ended_at)
        self.assertTrue(queued.error.retryable)
        self.assertEqual(self.repository.list_events('orphan'), [])
        bound = self.service.bind_simulation_task('orphan', 'accepted-task')
        self.assertIsNone(bound.error)
        self.service.apply_simulation_event(RunEvent(
            event_id='worker-first', run_id='orphan', task_id='accepted-task',
            sequence=1, type=RunEventType.STATE, occurred_at=NOW,
            payload={'previous_status': 'queued', 'status': 'initializing'},
        ))
        self.assertEqual(self.repository.get_run('orphan').status, RunStatus.INITIALIZING)

    def test_bound_or_stopped_run_cannot_be_overwritten(self):
        for index, changes in enumerate((dict(simulation_task_id='task'), dict(status=RunStatus.STOPPED))):
            original = make_run('protected-'+str(index), **changes)
            self.repository.create_run(original)
            actual = self.service.record_dispatch_error(original.run_id, 'DISPATCH_REJECTED', 'Rejected', definitive=True)
            self.assertEqual(actual, original)

    def test_existing_execution_event_prevents_convergence(self):
        self.repository.append_event(RunEvent(
            event_id='existing', run_id='orphan', sequence=1,
            type=RunEventType.STATE, occurred_at=NOW, payload={'status': 'initializing'}))
        actual = self.service.record_dispatch_error('orphan', 'DISPATCH_REJECTED', 'Rejected', definitive=True)
        self.assertEqual(actual.status, RunStatus.QUEUED)
        self.assertIsNone(actual.error)


class MemoryDispatchFailureTest(RepositoryCases, unittest.TestCase):
    def make_repository(self): return InMemoryRunRepository()


class SQLiteDispatchFailureTest(RepositoryCases, unittest.TestCase):
    def make_repository(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        return SQLiteRunRepository(Path(self.directory.name)/'runs.sqlite3')

    def test_event_write_failure_rolls_back_run_status(self):
        with self.repository._connection() as connection:
            connection.execute("CREATE TRIGGER reject_event BEFORE INSERT ON run_events BEGIN SELECT RAISE(ABORT, 'injected'); END")
        with self.assertRaises(Exception):
            self.service.record_dispatch_error('orphan', 'DISPATCH_REJECTED', 'Rejected', definitive=True)
        self.assertEqual(self.repository.get_run('orphan').status, RunStatus.QUEUED)
        self.assertIsNone(self.repository.get_run('orphan').error)
        self.assertEqual(self.repository.list_events('orphan'), [])


class DispatcherClassificationTest(unittest.TestCase):
    def setUp(self):
        self.repository=InMemoryRunRepository(); self.repository.create_run(make_run())
        self.service=RunService(self.repository, clock=lambda: NOW)
        self.client=Mock()
        self.dispatch=SimulationDispatchService(self.service, self.client)
        self.config=patch('robotics.simulation.dispatch.build_simulation_config', return_value={})
        self.config.start(); self.addCleanup(self.config.stop)

    def test_documented_config_rejection_fails_without_leaking_upstream_details(self):
        self.client.create_task.side_effect=SimulationRejectedError(422, code='invalid_task_config', upstream_message='secret-token')
        with self.assertRaises(SimulationRejectedError): self.dispatch.dispatch('orphan')
        result=self.repository.get_run('orphan')
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertNotIn('secret-token', result.error.message)
        self.dispatch.dispatch('orphan')
        self.assertEqual(self.client.create_task.call_count, 1)

    def test_local_invalid_snapshot_fails_before_any_request(self):
        with patch('robotics.simulation.dispatch.build_simulation_config', side_effect=SimulationSnapshotMismatch('invalid')):
            with self.assertRaises(SimulationSnapshotMismatch): self.dispatch.dispatch('orphan')
        self.client.create_task.assert_not_called()
        self.assertEqual(self.repository.get_run('orphan').status, RunStatus.FAILED)

    def test_local_value_error_uses_existing_snapshot_error_contract(self):
        with patch('robotics.simulation.dispatch.build_simulation_config', side_effect=ValueError('invalid')):
            with self.assertRaises(SimulationSnapshotMismatch): self.dispatch.dispatch('orphan')
        self.assertEqual(self.repository.get_run('orphan').status, RunStatus.FAILED)
        self.client.create_task.assert_not_called()

    def test_ambiguous_errors_never_claim_no_task_exists(self):
        for error in (SimulationTimeoutError('timeout'), SimulationRejectedError(401),
                      SimulationRejectedError(422, code='unknown_contract'), SimulationRejectedError(409)):
            self.client.create_task.side_effect=error
            with self.assertRaises(type(error)): self.dispatch.dispatch('orphan')
            result=self.repository.get_run('orphan')
            self.assertEqual(result.status, RunStatus.QUEUED)
            self.assertTrue(result.error.retryable)
            self.assertEqual(self.repository.list_events('orphan'), [])

    def test_stopped_run_is_not_relaunched(self):
        self.service.request_stop('orphan')
        result=self.dispatch.dispatch('orphan')
        self.assertEqual(result.status, RunStatus.STOPPED)
        self.client.create_task.assert_not_called()

    def test_stale_valid_record_gets_diagnostic_without_starting_gpu(self):
        actual=self.dispatch.inspect_stale_dispatch('orphan')
        self.assertEqual(actual.status, RunStatus.QUEUED)
        self.assertEqual(actual.error.code, 'DISPATCH_UNCONFIRMED')
        self.client.create_task.assert_not_called()

    def test_stale_invalid_record_fails_without_starting_gpu(self):
        with patch('robotics.simulation.dispatch.build_simulation_config', side_effect=SimulationSnapshotMismatch('old-invalid')):
            actual=self.dispatch.inspect_stale_dispatch('orphan')
        self.assertEqual(actual.status, RunStatus.FAILED)
        self.client.create_task.assert_not_called()

    def test_terminal_history_cannot_hide_old_unbound_run(self):
        for i in range(120): self.repository.create_run(make_run('completed-'+str(i), status=RunStatus.COMPLETED, created_at=NOW))
        dispatch=Mock(); stop=Mock()
        result=SimulationRecoveryService(self.service, dispatch, stop, max_runs=1).reconcile_simulation_runs()
        dispatch.inspect_stale_dispatch.assert_called_once_with('orphan')
        dispatch.dispatch.assert_not_called()
        self.assertEqual(result.examined, 1)

if __name__ == '__main__': unittest.main()
