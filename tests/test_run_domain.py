import unittest

from robotics.runs.domain import (
    TERMINAL_STATUSES,
    InvalidRunStateTransition,
    RunStatus,
    can_transition,
    is_terminal,
    validate_transition,
)


class RunStateMachineTest(unittest.TestCase):
    def assert_path_is_valid(self, *statuses):
        for current, target in zip(statuses, statuses[1:]):
            self.assertTrue(can_transition(current, target))
            self.assertIsNone(validate_transition(current, target))

    def test_success_path(self):
        self.assert_path_is_valid(
            RunStatus.QUEUED,
            RunStatus.INITIALIZING,
            RunStatus.RUNNING,
            RunStatus.FINALIZING,
            RunStatus.COMPLETED,
        )

    def test_queued_can_stop(self):
        self.assert_path_is_valid(RunStatus.QUEUED, RunStatus.STOPPED)

    def test_queued_bound_task_can_stop_cooperatively(self):
        self.assert_path_is_valid(
            RunStatus.QUEUED, RunStatus.STOPPING, RunStatus.STOPPED
        )

    def test_initializing_can_fail(self):
        self.assert_path_is_valid(RunStatus.INITIALIZING, RunStatus.FAILED)

    def test_running_can_stop_cooperatively(self):
        self.assert_path_is_valid(
            RunStatus.RUNNING, RunStatus.STOPPING, RunStatus.STOPPED
        )

    def test_running_can_fail(self):
        self.assert_path_is_valid(RunStatus.RUNNING, RunStatus.FAILED)

    def test_same_status_is_idempotent(self):
        for status in RunStatus:
            with self.subTest(status=status):
                self.assertTrue(can_transition(status, status))
                self.assertIsNone(validate_transition(status, status))

    def test_terminal_statuses(self):
        self.assertEqual(
            TERMINAL_STATUSES,
            frozenset(
                {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.STOPPED}
            ),
        )
        for status in RunStatus:
            with self.subTest(status=status):
                self.assertEqual(is_terminal(status), status in TERMINAL_STATUSES)

    def test_invalid_transitions_raise_domain_error(self):
        invalid_transitions = (
            (RunStatus.COMPLETED, RunStatus.RUNNING),
            (RunStatus.FAILED, RunStatus.QUEUED),
            (RunStatus.STOPPED, RunStatus.RUNNING),
            (RunStatus.QUEUED, RunStatus.COMPLETED),
        )
        for current, target in invalid_transitions:
            with self.subTest(current=current, target=target):
                self.assertFalse(can_transition(current, target))
                with self.assertRaises(InvalidRunStateTransition) as raised:
                    validate_transition(current, target)
                self.assertEqual(raised.exception.current, current)
                self.assertEqual(raised.exception.target, target)
                self.assertIn(current.value, str(raised.exception))
                self.assertIn(target.value, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
