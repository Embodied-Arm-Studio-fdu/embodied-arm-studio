"""Browser stream-session control plane over the Simulation HTTP boundary."""

from typing import Any, Optional

from robotics.runs import ExecutionMode, RunService, RunStatus

from .contracts import CreateStreamSessionRequest, StreamDescriptor


class StreamSessionNotAllowed(ValueError):
    """Raised when persisted Run facts do not permit a video session."""


class StreamSessionUnavailable(RuntimeError):
    """Raised when the Simulation stream boundary is not configured."""


class StreamDescriptorMismatch(ValueError):
    """Raised when Simulation returns a descriptor for another resource."""


class StreamSessionService:
    def __init__(self, run_service: RunService, client: Optional[Any]) -> None:
        self._run_service = run_service
        self._client = client

    def get_stream(
        self, run_id: str, camera_id: str = "overview"
    ) -> StreamDescriptor:
        try:
            safe_camera_id = CreateStreamSessionRequest(camera_id).camera_id
        except ValueError as exc:
            raise StreamSessionNotAllowed(str(exc)) from exc
        run = self._run_service.get_run(run_id)
        if run.execution_mode != ExecutionMode.SIMULATION:
            raise StreamSessionNotAllowed(
                "Only simulation Runs can have video streams"
            )
        if run.simulation_task_id is None:
            raise StreamSessionNotAllowed(
                "Simulation Run has no bound task"
            )
        if run.status != RunStatus.RUNNING:
            raise StreamSessionNotAllowed(
                "Video stream is available only while the Run is running"
            )
        method = getattr(self._client, "create_stream_session", None)
        if not callable(method):
            raise StreamSessionUnavailable(
                "Simulation stream-session is not configured"
            )
        descriptor = method(run.simulation_task_id, safe_camera_id)
        if not isinstance(descriptor, StreamDescriptor):
            raise StreamDescriptorMismatch(
                "Simulation returned an invalid StreamDescriptor"
            )
        if (
            descriptor.run_id != run.run_id
            or descriptor.task_id != run.simulation_task_id
            or descriptor.camera_id != safe_camera_id
        ):
            raise StreamDescriptorMismatch(
                "Simulation StreamDescriptor identity does not match the Run"
            )
        return descriptor
