"""Pure domain rules for the platform Run state machine."""

from enum import Enum
from typing import FrozenSet, Mapping


class RunStatus(str, Enum):
    QUEUED = "queued"
    INITIALIZING = "initializing"
    RUNNING = "running"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


TERMINAL_STATUSES: FrozenSet[RunStatus] = frozenset(
    {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.STOPPED}
)

ALLOWED_TRANSITIONS: Mapping[RunStatus, FrozenSet[RunStatus]] = {
    RunStatus.QUEUED: frozenset(
        {RunStatus.INITIALIZING, RunStatus.STOPPING, RunStatus.STOPPED}
    ),
    RunStatus.INITIALIZING: frozenset({RunStatus.RUNNING, RunStatus.FAILED}),
    RunStatus.RUNNING: frozenset(
        {RunStatus.FINALIZING, RunStatus.STOPPING, RunStatus.FAILED}
    ),
    RunStatus.FINALIZING: frozenset({RunStatus.COMPLETED, RunStatus.FAILED}),
    RunStatus.STOPPING: frozenset({RunStatus.STOPPED}),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.STOPPED: frozenset(),
}


class InvalidRunStateTransition(ValueError):
    """Raised when a Run transition is outside the domain rules."""

    def __init__(self, current: RunStatus, target: RunStatus) -> None:
        super().__init__(
            "invalid Run status transition: %s -> %s"
            % (current.value, target.value)
        )
        self.current = current
        self.target = target


class TerminalRunStatusConflict(ValueError):
    """Raised when a trusted terminal fact conflicts with an existing terminal."""

    def __init__(self, current: RunStatus, target: RunStatus) -> None:
        super().__init__(
            "terminal Run status conflict: %s -> %s"
            % (current.value, target.value)
        )
        self.current = current
        self.target = target


def is_terminal(status: RunStatus) -> bool:
    """Return whether ``status`` is final and cannot transition further."""

    return status in TERMINAL_STATUSES


def can_transition(current: RunStatus, target: RunStatus) -> bool:
    """Return whether the transition is legal, including idempotent writes."""

    return current == target or target in ALLOWED_TRANSITIONS[current]


def validate_transition(current: RunStatus, target: RunStatus) -> None:
    """Raise a domain error unless the transition is legal or idempotent."""

    if not can_transition(current, target):
        raise InvalidRunStateTransition(current, target)


def validate_terminal_convergence(
    current: RunStatus, target: RunStatus
) -> None:
    """Allow a trusted Callback to close any nonterminal Run exactly once."""

    if target not in TERMINAL_STATUSES:
        raise ValueError("terminal convergence target must be terminal")
    if current in TERMINAL_STATUSES and current != target:
        raise TerminalRunStatusConflict(current, target)
