import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from simulation_core.resource_tracker_cleanup import cleanup_owned_resource_tracker
from simulation_core import rq_jobs


class FakeTracker:
    def __init__(self, pid=None, fd=None, error=None):
        self._pid = pid
        self._fd = fd
        self.error = error
        self.stops = 0

    def _stop(self):
        self.stops += 1
        if self.error is not None:
            raise self.error


class ResourceTrackerCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.proc = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def process(self, pid, ppid, cmdline=b"multiprocessing.resource_tracker\0"):
        root = self.proc / str(pid)
        root.mkdir()
        (root / "status").write_text(
            "Name:\tpython3\nPPid:\t%d\n" % ppid, encoding="utf-8"
        )
        (root / "cmdline").write_bytes(cmdline)

    def cleanup(self, tracker, current_pid=100):
        return cleanup_owned_resource_tracker(
            tracker, proc_root=self.proc, current_pid=current_pid,
            logger=logging.getLogger("resource-tracker-test"),
        )

    def test_tracker_missing_and_pid_none_are_safe_noops(self):
        self.assertFalse(self.cleanup(None))
        tracker = FakeTracker(pid=None, fd=4)
        self.assertFalse(self.cleanup(tracker))
        self.assertEqual(0, tracker.stops)

    def test_exited_tracker_is_safe_noop(self):
        tracker = FakeTracker(pid=201, fd=4)
        self.assertFalse(self.cleanup(tracker))
        self.assertEqual(0, tracker.stops)

    def test_invalid_fd_is_safe_noop(self):
        tracker = FakeTracker(pid=202, fd=None)
        self.process(202, 100)
        self.assertFalse(self.cleanup(tracker))
        self.assertEqual(0, tracker.stops)

    def test_non_owned_tracker_is_not_touched(self):
        tracker = FakeTracker(pid=203, fd=4)
        self.process(203, 999)
        self.assertFalse(self.cleanup(tracker))
        self.assertEqual(0, tracker.stops)

    def test_unrecognized_cmdline_is_not_touched(self):
        tracker = FakeTracker(pid=204, fd=4)
        self.process(204, 100, b"python3\0training.py\0")
        self.assertFalse(self.cleanup(tracker))
        self.assertEqual(0, tracker.stops)

    def test_owned_tracker_is_stopped_and_reaped_by_private_api(self):
        tracker = FakeTracker(pid=205, fd=4)
        self.process(205, 100)
        self.assertTrue(self.cleanup(tracker))
        self.assertEqual(1, tracker.stops)

    def test_cleanup_exception_is_logged_and_suppressed(self):
        tracker = FakeTracker(pid=206, fd=4, error=OSError("wait failed"))
        self.process(206, 100)
        with self.assertLogs("resource-tracker-test", level="ERROR"):
            self.assertFalse(self.cleanup(tracker))
        self.assertEqual(1, tracker.stops)


class RQJobCleanupIsolationTests(unittest.TestCase):
    def run_job(self, outcome):
        repository = object()
        cancellation = SimpleNamespace(request_ttl=60)
        service = object()
        with patch.object(rq_jobs.RedisTaskRepository, "from_url",
                          return_value=repository), \
             patch.object(rq_jobs.RedisCancellationStore, "from_url",
                          return_value=cancellation), \
             patch.object(rq_jobs.HttpBackendClient, "from_env",
                          return_value=object()), \
             patch.object(rq_jobs, "SimulationService", return_value=service), \
             patch.object(rq_jobs, "execute_training_task", side_effect=outcome), \
             patch.object(rq_jobs, "cleanup_owned_resource_tracker",
                          side_effect=RuntimeError("cleanup failed")), \
             patch.object(rq_jobs.LOGGER, "exception"):
            return rq_jobs.run_training_task("task", "redis://test", "/tmp/a")

    def test_completed_stopped_and_failed_results_survive_cleanup_failure(self):
        for status in ("completed", "stopped", "failed"):
            with self.subTest(status=status):
                result = {"task_id": "task", "status": status}
                self.assertEqual(result, self.run_job(lambda *args, **kwargs: result))

    def test_original_exception_survives_cleanup_failure(self):
        original = ValueError("initialization failed")
        with self.assertRaises(ValueError) as raised:
            self.run_job(original)
        self.assertIs(original, raised.exception)


if __name__ == "__main__":
    unittest.main()
