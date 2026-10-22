"""Bounded ownership guard for CPython's private multiprocessing tracker."""

import logging
import os
from pathlib import Path


LOGGER = logging.getLogger(__name__)
_TRACKER_MARKER = b"multiprocessing.resource_tracker"
_UNSET = object()


def cleanup_owned_resource_tracker(
    tracker=_UNSET,
    *,
    proc_root=Path("/proc"),
    current_pid=None,
    logger=None,
):
    """Stop/reap only a resource tracker directly owned by this process.

    This helper deliberately contains all access to CPython's private tracker
    object.  It is best-effort: cleanup diagnostics must never replace a task
    result or its original failure.
    """
    log = logger or LOGGER
    try:
        if tracker is _UNSET:
            try:
                from multiprocessing import resource_tracker
            except (ImportError, AttributeError):
                return False
            tracker = getattr(resource_tracker, "_resource_tracker", None)
        if tracker is None:
            return False

        tracker_pid = getattr(tracker, "_pid", None)
        tracker_fd = getattr(tracker, "_fd", None)
        stop = getattr(tracker, "_stop", None)
        if (
            not isinstance(tracker_pid, int)
            or tracker_pid <= 0
            or not isinstance(tracker_fd, int)
            or tracker_fd < 0
            or not callable(stop)
        ):
            return False

        owner_pid = os.getpid() if current_pid is None else int(current_pid)
        process_path = Path(proc_root) / str(tracker_pid)
        try:
            status = (process_path / "status").read_text(encoding="utf-8")
            cmdline = (process_path / "cmdline").read_bytes()
        except FileNotFoundError:
            # It exited between tracker inspection and /proc validation.
            return False

        parent_pid = None
        for line in status.splitlines():
            if line.startswith("PPid:"):
                try:
                    parent_pid = int(line.split(":", 1)[1].strip())
                except ValueError:
                    return False
                break
        if parent_pid != owner_pid or _TRACKER_MARKER not in cmdline:
            return False

        stop()  # CPython 3.8: close alive fd, then waitpid(tracker pid).
        return True
    except Exception:
        log.exception("owned multiprocessing resource_tracker cleanup failed")
        return False
