"""Lightweight HTTP boundary for the existing :class:`SimulationService`."""

from __future__ import annotations

import hmac
import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .adapters import NullBackendClient
from .cancellation import RedisCancellationStore
from .domain import InvalidStateTransition, SimulationTask, TaskNotFound, TaskStatus
from .http_backend import HttpBackendClient
from .input_events import (
    InputEvent,
    InputEventConflict,
    RecordingInputEventSink,
)
from .queueing import RQTaskQueue, TaskQueueError
from .redis_repository import RedisRepositoryError, RedisTaskRepository
from .service import SimulationService, StopNotAllowed
from .stream_session import StreamDescriptor, StreamSessionStatus
from .task_catalog import task_catalog_document
from .task_registry import PRODUCTION_TASK_REGISTRY, UnknownTaskDefinition


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class HttpAdapterConfigurationError(RuntimeError):
    """Raised when the process cannot be configured safely from its environment."""


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class CreateTaskBody(BaseModel):
    run_id: str = Field(..., min_length=1, max_length=128)
    project_id: str = Field(..., min_length=1, max_length=128)
    config: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "forbid"


class CreateStreamSessionBody(BaseModel):
    camera_id: str = Field("overview", min_length=1, max_length=128)

    class Config:
        extra = "forbid"


class InputEventBody(BaseModel):
    event_id: Any
    run_id: Any
    task_id: Any
    type: Any
    occurred_at: Any
    key: Any = None
    action: Any = None
    x: Any = None
    y: Any = None
    button: Any = None

    class Config:
        extra = "forbid"


def _validate_identifier(name: str, value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ApiError(422, "invalid_request", "%s must be a safe identifier" % name)
    return value


def _task_payload(task: SimulationTask) -> Dict[str, Any]:
    return {
        "task_id": task.task_id,
        "run_id": task.run_id,
        "project_id": task.project_id,
        "status": task.status.value,
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
        "config": dict(task.config.values),
    }


def create_http_app(
    service: SimulationService,
    service_token: str,
    stream_manager=None,
    stream_gateway=None,
    input_sink=None,
    task_registry=PRODUCTION_TASK_REGISTRY,
) -> FastAPI:
    """Build an injectable HTTP app; task behavior stays in ``SimulationService``."""
    if not isinstance(service_token, str) or not service_token.strip():
        raise HttpAdapterConfigurationError("SIMULATION_SERVICE_TOKEN is required")

    app = FastAPI(title="Embodied Arm Simulation Service", version="1.0")
    app.state.simulation_service = service
    app.state.input_event_sink = (
        input_sink if input_sink is not None else RecordingInputEventSink()
    )

    async def authorize(authorization: Optional[str] = Header(None)) -> None:
        expected = "Bearer " + service_token
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise ApiError(401, "unauthorized", "valid Bearer service token required")

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message},
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        status_code = 400 if request.url.path.endswith(
            ("/stream-session", "/input-events")
        ) else 422
        return JSONResponse(
            status_code=status_code,
            content={"code": "invalid_request", "message": "request validation failed"},
        )

    @app.exception_handler(TaskNotFound)
    async def not_found_handler(request: Request, exc: TaskNotFound) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"code": "task_not_found", "message": "simulation task not found"},
        )

    @app.exception_handler(StopNotAllowed)
    @app.exception_handler(InvalidStateTransition)
    async def conflict_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"code": "stop_not_allowed", "message": "task cannot be stopped in its current state"},
        )

    @app.exception_handler(Exception)
    async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"code": "internal_error", "message": "simulation service internal error"},
        )

    @app.post("/simulation/tasks", status_code=202, dependencies=[Depends(authorize)])
    async def create_task(body: CreateTaskBody) -> Mapping[str, Any]:
        run_id = _validate_identifier("run_id", body.run_id)
        project_id = _validate_identifier("project_id", body.project_id)
        try:
            config = task_registry.parse_config(body.config)
        except (TypeError, ValueError, UnknownTaskDefinition) as exc:
            raise ApiError(422, "invalid_task_config", str(exc)) from exc
        return _task_payload(
            service.create_task(run_id, project_id, config.to_dict())
        )

    @app.get("/simulation/task-catalog", dependencies=[Depends(authorize)])
    async def get_task_catalog() -> Mapping[str, Any]:
        return task_catalog_document(task_registry)

    @app.get("/simulation/tasks/{task_id}", dependencies=[Depends(authorize)])
    async def get_task(task_id: str) -> Mapping[str, Any]:
        return _task_payload(service.get_task(_validate_identifier("task_id", task_id)))

    @app.post("/simulation/tasks/{task_id}/stop", dependencies=[Depends(authorize)])
    async def stop_task(task_id: str) -> Mapping[str, Any]:
        return _task_payload(service.request_stop(_validate_identifier("task_id", task_id)))

    @app.post(
        "/simulation/tasks/{task_id}/input-events",
        status_code=202,
        dependencies=[Depends(authorize)],
    )
    async def record_input_event(
        task_id: str, body: InputEventBody
    ) -> Mapping[str, Any]:
        try:
            path_task_id = _validate_identifier("task_id", task_id)
        except ApiError as exc:
            raise ApiError(400, "invalid_request", "request validation failed") from exc
        try:
            values = (
                body.model_dump(exclude_unset=True)
                if hasattr(body, "model_dump")
                else body.dict(exclude_unset=True)
            )
            event = InputEvent.from_mapping(values)
        except (TypeError, ValueError) as exc:
            raise ApiError(400, "invalid_request", "request validation failed") from exc
        if path_task_id != event.task_id:
            raise ApiError(400, "invalid_request", "path and body task_id must match")

        task = service.get_task(path_task_id)
        if task.run_id != event.run_id:
            raise ApiError(400, "invalid_request", "run_id does not match task")
        if task.status != TaskStatus.RUNNING:
            raise ApiError(
                409,
                "input_not_allowed",
                "input recording requires a running task",
            )
        try:
            result = app.state.input_event_sink.record(event)
        except InputEventConflict as exc:
            raise ApiError(
                409, "event_conflict", "event_id conflicts with retained event"
            ) from exc
        except Exception as exc:
            raise ApiError(
                503, "input_sink_unavailable", "input recording is unavailable"
            ) from exc
        return {
            "accepted": result.accepted,
            "event_id": event.event_id,
            "duplicate": result.duplicate,
        }

    @app.post(
        "/simulation/tasks/{task_id}/stream-session",
        dependencies=[Depends(authorize)],
    )
    async def create_stream_session(
        task_id: str, body: Optional[CreateStreamSessionBody] = None
    ) -> Mapping[str, Any]:
        task = service.get_task(_validate_identifier("task_id", task_id))
        camera_id = _validate_identifier(
            "camera_id", body.camera_id if body is not None else "overview"
        )
        if camera_id != "overview":
            raise ApiError(400, "invalid_camera_id", "only overview camera is supported")
        if task.status != TaskStatus.RUNNING:
            raise ApiError(
                409,
                "stream_session_not_allowed",
                "stream session requires a running task",
            )

        if stream_manager is not None:
            config = task.config.values
            session = stream_manager.create_session(
                task.run_id,
                task.task_id,
                camera_id,
                int(config.get("video_width", 640)),
                int(config.get("video_height", 480)),
                float(config.get("video_fps", 15.0)),
            )
            session = stream_manager.start_session(session.stream_id)
            if session.status == StreamSessionStatus.ERROR:
                raise ApiError(503, "stream_not_ready", "video stream is not ready")
            return StreamDescriptor.from_session(session).to_dict()

        record = (
            stream_gateway.get_for_run(task.run_id)
            if stream_gateway is not None else None
        )
        if record is None or record.get("task_id") != task.task_id:
            raise ApiError(503, "stream_not_ready", "video stream is not ready")
        if record.get("camera_id") != camera_id:
            raise ApiError(400, "invalid_camera_id", "camera is not available")
        if record.get("status") not in {
            StreamSessionStatus.STARTING.value,
            StreamSessionStatus.READY.value,
        }:
            raise ApiError(503, "stream_not_ready", "video stream is not ready")
        return StreamDescriptor(**record).to_dict()

    return app


def build_service_from_env(environ: Optional[Mapping[str, str]] = None) -> SimulationService:
    """Assemble the existing Redis/RQ-backed service for an HTTP process."""
    values = environ if environ is not None else os.environ
    redis_url = values.get("SIMULATION_REDIS_URL", "redis://127.0.0.1:6379/0")
    namespace = values.get("SIMULATION_REDIS_NAMESPACE", "simulation")
    queue_name = values.get("SIMULATION_RQ_QUEUE", "simulation-gpu")
    artifacts_root = Path(values.get("SIMULATION_ARTIFACTS_ROOT", "artifacts"))
    repository = RedisTaskRepository.from_url(redis_url, namespace=namespace)
    queue = RQTaskQueue.from_url(
        redis_url,
        artifacts_root=str(artifacts_root),
        queue_name=queue_name,
        namespace=namespace,
    )
    cancellation = RedisCancellationStore.from_url(redis_url, namespace=namespace)
    backend = HttpBackendClient.from_env(required=False, environ=values) or NullBackendClient()
    return SimulationService(
        repository,
        queue,
        backend,
        cancellation_store=cancellation,
        artifacts_root=artifacts_root,
    )


def create_app_from_env() -> FastAPI:
    """Uvicorn factory entry point."""
    token = os.environ.get("SIMULATION_SERVICE_TOKEN", "")
    service = build_service_from_env()
    from .stream_session import RedisRecordingStreamGateway
    stream_gateway = RedisRecordingStreamGateway(
        service._repository.client, namespace=service._repository.namespace
    )
    return create_http_app(service, token, stream_gateway=stream_gateway)
