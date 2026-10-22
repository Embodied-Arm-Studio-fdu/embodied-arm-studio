"""Bounded polling for browser-facing canonical Run Event streams."""

from dataclasses import dataclass
from typing import List

from robotics.runs.domain import TERMINAL_STATUSES, RunStatus
from robotics.runs.models import Run, RunEvent, RunEventType
from robotics.runs.service import RunService


@dataclass(frozen=True)
class RunEventStreamConfig:
    poll_interval_seconds: float = 0.15
    batch_size: int = 100

    def __post_init__(self) -> None:
        if (
            isinstance(self.poll_interval_seconds, bool)
            or not isinstance(self.poll_interval_seconds, (int, float))
            or not 0.01 <= self.poll_interval_seconds <= 10
        ):
            raise ValueError(
                "poll_interval_seconds must be between 0.01 and 10 seconds"
            )
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or not 1 <= self.batch_size <= 1000
        ):
            raise ValueError("batch_size must be between 1 and 1000")


class RunEventStream:
    """Read canonical Events from the Run repository in bounded batches."""

    def __init__(
        self,
        run_service: RunService,
        config: RunEventStreamConfig = RunEventStreamConfig(),
    ) -> None:
        self._run_service = run_service
        self.config = config

    def get_run(self, run_id: str) -> Run:
        return self._run_service.get_run(run_id)

    def list_batch(self, run_id: str, after_sequence: int) -> List[RunEvent]:
        return self._run_service.list_events(
            run_id,
            after_sequence=after_sequence,
            limit=self.config.batch_size,
        )

    @staticmethod
    def run_is_terminal(run: Run) -> bool:
        return run.status in TERMINAL_STATUSES

    @staticmethod
    def event_is_terminal(event: RunEvent) -> bool:
        if event.type != RunEventType.STATE:
            return False
        try:
            status = RunStatus(event.payload.get("status"))
        except (TypeError, ValueError):
            return False
        return status in TERMINAL_STATUSES
