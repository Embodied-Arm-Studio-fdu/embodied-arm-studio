import asyncio
from contextlib import suppress
import logging
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse
from sqlalchemy.exc import SQLAlchemyError
from robotics.runs.repository import RunDeletionNotAllowed

from apps.api.dependencies.runs import RunCreationService
from apps.api.dependencies.websocket_auth import authenticate_websocket
from apps.api.schemas.runs import (
    CompareRequest,
    ExportRequest,
    RunCreate,
    V1RunCreateRequest,
    V1RunBatchRequest,
    V1RunDeleteRequest,
    V1RunComparisonResponse,
    V1RunExportRequest,
    V1RunArtifactResponse,
    V1RunEventResponse,
    V1RunResponse,
    V1StreamDescriptorResponse,
)
from robotics.adapters.mock import MockSimulatorAdapter
from robotics.artifacts import (
    ArtifactIntegrityError,
    ArtifactNotFound,
    ArtifactServiceUnavailable,
)
from robotics.contracts import RunRequest, RunStatus
from robotics.evaluation import EvaluationService
from robotics.evaluations import (
    CanonicalRunFactsNotFound,
    CanonicalRunFactsUnavailable,
)
from robotics.runs import (
    IdempotencyConflict,
    IdempotentRunCreateCommand,
    InvalidIdempotentRunCreateRequest,
    ProjectRunCreationNotFound,
    ProjectRunCreationPermissionDenied,
    ProjectRunVersionConflict,
    SceneRunVersionConflict,
    RunNotFound,
    RepositoryError,
    RunStopNotAllowed,
    RunTaskMismatch,
    TaskIdAlreadyBound,
    TaskIdConflict,
    RunStatus as CanonicalRunStatus,
    RunEventStream,
)
from robotics.run_access import (
    RunAccessForbidden,
    RunAccessNotFound,
    RunAccessUnavailable,
)
from robotics.simulation import (
    SimulationConfigurationError,
    SimulationNetworkError,
    SimulationRejectedError,
    SimulationResponseError,
    SimulationServerError,
    SimulationSnapshotMismatch,
    SimulationTimeoutError,
    SimulationStopUnavailable,
    StreamDescriptorMismatch,
    StreamSessionNotAllowed,
    StreamSessionUnavailable,
)


logger = logging.getLogger(__name__)


def create_runs_router(
    adapter: MockSimulatorAdapter,
    evaluation_service: EvaluationService,
    run_creation_service: RunCreationService = None,
    artifact_service: Optional[Any] = None,
    artifact_download_service: Optional[Any] = None,
    simulation_dispatch_service: Optional[Any] = None,
    simulation_stop_service: Optional[Any] = None,
    stream_session_service: Optional[Any] = None,
    run_event_stream: Optional[RunEventStream] = None,
    comparison_export_service: Optional[Any] = None,
    run_access_service: Optional[Any] = None,
    current_user_provider: Optional[Any] = None,
    run_deletion_repository: Optional[Any] = None,
    websocket_auth_timeout_seconds: float = 5.0,
) -> APIRouter:
    if current_user_provider is None:
        raise ValueError("current_user_provider is required")
    router = APIRouter()

    @router.get("/api/v1/runs/{run_id}/charts/{metric}.png")
    def download_metric_chart(run_id: str, metric: str, request: Request):
        get_accessible_run(run_access_service, run_id, request)
        allowed = {"average_reward", "episode_length", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "explained_variance", "learning_rate", "steps_per_second", "success_rate", "timeout_rate", "failure_rate", "final_distance", "minimum_distance", "object_displacement", "candidate_rate", "mean_maximum_cube_rise", "mean_hold_duration", "descended_rate", "detached_rate", "mean_final_xy_error", "verified_success_rate", "settled_1s_rate"}
        if metric not in allowed:
            raise HTTPException(422, "Unsupported metric")
        try:
            from robotics.artifacts import metric_chart_png
            content = metric_chart_png(request.app.state.run_service.list_events(run_id), metric)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        except RepositoryError as exc:
            raise HTTPException(503, "Metric history is temporarily unavailable") from exc
        return Response(content=content, media_type="image/png", headers={"Content-Disposition": f'attachment; filename="{metric}.png"', "X-Content-Type-Options": "nosniff"})

    @router.get("/api/runs")
    def runs() -> List[Dict[str, Any]]:
        return [record.to_dict() for record in adapter.list_runs()]

    @router.post("/api/runs/play", status_code=202)
    def launch_play(payload: RunCreate) -> Dict[str, Any]:
        return _launch(adapter.launch_play, payload)

    @router.post("/api/runs/train", status_code=202)
    def launch_train(payload: RunCreate) -> Dict[str, Any]:
        return _launch(adapter.launch_training, payload)

    @router.post("/api/runs/compare")
    def compare(payload: CompareRequest) -> Dict[str, Any]:
        compared = []
        for run_id in payload.run_ids:
            record = get_run_or_404(adapter, run_id)
            metric_events = [
                event
                for event in adapter.stream_events(run_id)
                if event.type == "metric"
            ]
            final_metrics = (
                metric_events[-1].payload["metrics"] if metric_events else {}
            )
            compared.append(
                {"run": record.to_dict(), "final_metrics": final_metrics}
            )
        return {"runs": compared}

    @router.get("/api/runs/{run_id}")
    def run(run_id: str) -> Dict[str, Any]:
        return get_run_or_404(adapter, run_id).to_dict()

    @router.post("/api/runs/{run_id}/stop")
    def stop(run_id: str) -> Dict[str, Any]:
        try:
            return adapter.stop_run(run_id).to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/api/runs/{run_id}/events")
    def events(run_id: str, after_event_id: str = "") -> List[Dict[str, Any]]:
        try:
            return [
                event.to_dict()
                for event in adapter.stream_events(run_id, after_event_id)
            ]
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/api/runs/{run_id}/artifacts")
    def artifacts(run_id: str) -> List[Dict[str, Any]]:
        get_run_or_404(adapter, run_id)
        return adapter.list_artifacts(run_id)

    @router.post("/api/runs/{run_id}/export")
    def export(run_id: str, payload: ExportRequest) -> Dict[str, Any]:
        try:
            gate = evaluation_service.export_gate(
                run_id, payload.evaluation_report_id
            )
            if payload.format != "manifest" and not gate["eligible"]:
                raise ValueError(
                    "导出门禁未满足: " + "; ".join(gate["reasons"])
                )
            return adapter.export_policy(run_id, payload.format, gate)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="运行或评估报告不存在"
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/api/v1/runs", response_model=List[V1RunResponse])
    def v1_runs(request: Request) -> List[V1RunResponse]:
        try:
            runs = run_access_service.list_own_runs(
                request.state.current_user.user_id
            )
        except RunAccessNotFound as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        except RunAccessUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail="Run lookup is temporarily unavailable",
            ) from exc
        return [V1RunResponse.model_validate(run) for run in runs]

    @router.post("/api/v1/runs/delete")
    def v1_delete_runs(payload: V1RunDeleteRequest, request: Request) -> Dict[str, Any]:
        delete_batch = getattr(run_deletion_repository, "delete_owned_terminal_runs", None)
        if delete_batch is None:
            raise HTTPException(503, "Run deletion is temporarily unavailable")
        actor = request.state.current_user.user_id
        try:
            for run_id in payload.run_ids:
                run_access_service.control(run_id, actor)
            count = delete_batch(payload.run_ids, actor)
        except (RunAccessNotFound, RunAccessForbidden, RunNotFound) as exc:
            raise HTTPException(404, "Run not found") from exc
        except RunDeletionNotAllowed as exc:
            raise HTTPException(409, str(exc)) from exc
        except (RunAccessUnavailable, RepositoryError, SQLAlchemyError) as exc:
            raise HTTPException(503, "Run deletion is temporarily unavailable") from exc
        return {"deleted_count": count, "deleted_run_ids": payload.run_ids}

    @router.get(
        "/api/v1/classes/{class_id}/runs",
        response_model=List[V1RunResponse],
    )
    def v1_class_runs(
        class_id: str,
        request: Request,
        project_id: Optional[str] = Query(None, min_length=1, max_length=255),
        status: Optional[CanonicalRunStatus] = Query(None),
    ) -> List[V1RunResponse]:
        try:
            runs = run_access_service.list_class_runs(
                class_id,
                request.state.current_user.user_id,
                project_id=project_id,
                status=status.value if status is not None else None,
            )
        except RunAccessNotFound as exc:
            raise HTTPException(status_code=404, detail="Class not found") from exc
        except RunAccessUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail="Run lookup is temporarily unavailable",
            ) from exc
        return [V1RunResponse.model_validate(run) for run in runs]

    @router.post(
        "/api/v1/runs",
        response_model=V1RunResponse,
        status_code=202,
    )
    def v1_create_run(
        payload: V1RunCreateRequest,
        response: Response,
        request: Request,
        idempotency_key: str = Header(..., alias="Idempotency-Key"),
    ) -> V1RunResponse:
        if run_creation_service is None:
            raise HTTPException(
                status_code=503,
                detail="Canonical Run creation service is not configured",
            )
        normalized_key = idempotency_key.strip()
        if not normalized_key:
            raise HTTPException(
                status_code=422,
                detail="Idempotency-Key must not be blank",
            )
        if len(normalized_key) > 255:
            raise HTTPException(
                status_code=422,
                detail="Idempotency-Key must be at most 255 characters",
            )
        try:
            result = run_creation_service.create_run(
                IdempotentRunCreateCommand(
                    user_id=request.state.current_user.user_id,
                    idempotency_key=normalized_key,
                    **payload.model_dump(),
                )
            )
        except IdempotencyConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except InvalidIdempotentRunCreateRequest as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (
            SimulationConfigurationError,
            SimulationTimeoutError,
            SimulationNetworkError,
            SimulationServerError,
        ) as exc:
            raise HTTPException(
                status_code=503,
                detail="Simulation task catalog is temporarily unavailable",
            ) from exc
        except SimulationResponseError as exc:
            raise HTTPException(
                status_code=502,
                detail="Simulation Service returned an invalid task catalog",
            ) from exc
        except SimulationRejectedError as exc:
            status_code = 503 if exc.status_code in {401, 500} else 502
            raise HTTPException(
                status_code=status_code,
                detail="Simulation Service rejected the task catalog request",
            ) from exc
        except ProjectRunCreationNotFound as exc:
            raise HTTPException(
                status_code=404, detail="Teaching resource not found"
            ) from exc
        except ProjectRunCreationPermissionDenied as exc:
            raise HTTPException(
                status_code=403, detail="Run creation is forbidden"
            ) from exc
        except ProjectRunVersionConflict as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": "project_version_conflict"},
            ) from exc
        except SceneRunVersionConflict as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": str(exc)},
            ) from exc
        response.headers["Idempotency-Replayed"] = str(
            result.replayed
        ).lower()
        run = result.run
        if run.execution_mode.value == "simulation":
            if run.simulation_task_id is not None:
                return V1RunResponse.model_validate(run)
            if simulation_dispatch_service is None:
                raise HTTPException(
                    status_code=503,
                    detail="Simulation dispatch is not configured",
                )
            try:
                run = simulation_dispatch_service.dispatch(run.run_id)
            except ProjectRunCreationNotFound as exc:
                raise HTTPException(
                    status_code=404, detail="Teaching resource not found"
                ) from exc
            except ProjectRunCreationPermissionDenied as exc:
                raise HTTPException(
                    status_code=403, detail="Run creation is forbidden"
                ) from exc
            except (SimulationTimeoutError, SimulationNetworkError, SimulationServerError) as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Simulation Service is temporarily unavailable",
                ) from exc
            except SimulationRejectedError as exc:
                if exc.status_code in {401, 500}:
                    status_code = 503
                    detail = "Simulation Service is temporarily unavailable"
                elif exc.status_code == 409:
                    status_code = 409
                    detail = "Simulation Service rejected the Run state"
                else:
                    status_code = 502
                    detail = "Simulation Service rejected the task request"
                raise HTTPException(status_code=status_code, detail=detail) from exc
            except (
                SimulationConfigurationError,
                SimulationResponseError,
                SimulationSnapshotMismatch,
            ) as exc:
                raise HTTPException(
                    status_code=502,
                    detail="Simulation Service returned an invalid task snapshot",
                ) from exc
            except (TaskIdConflict, TaskIdAlreadyBound, RunTaskMismatch) as exc:
                raise HTTPException(
                    status_code=409,
                    detail="Simulation task binding conflicts with the persisted Run",
                ) from exc
            except RepositoryError as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Run task binding is temporarily unavailable",
                ) from exc
        return V1RunResponse.model_validate(run)

    @router.post(
        "/api/v1/runs/compare",
        response_model=V1RunComparisonResponse,
    )
    def v1_compare_runs(
        payload: V1RunBatchRequest, request: Request
    ) -> V1RunComparisonResponse:
        if comparison_export_service is None:
            raise HTTPException(
                status_code=503,
                detail="Canonical Run comparison is not configured",
            )
        try:
            comparison = comparison_export_service.compare(
                payload.run_ids, request.state.current_user.user_id
            )
        except CanonicalRunFactsNotFound as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        except CanonicalRunFactsUnavailable as exc:
            logger.error("Canonical Run comparison failed")
            raise HTTPException(
                status_code=503,
                detail="Canonical Run comparison is temporarily unavailable",
            ) from exc
        return V1RunComparisonResponse.model_validate(comparison)

    @router.post("/api/v1/runs/export")
    def v1_export_runs(payload: V1RunExportRequest, request: Request) -> Response:
        if comparison_export_service is None:
            raise HTTPException(
                status_code=503,
                detail="Canonical Run export is not configured",
            )
        try:
            exported = comparison_export_service.export_json(
                payload.run_ids, request.state.current_user.user_id
            )
        except CanonicalRunFactsNotFound as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        except CanonicalRunFactsUnavailable as exc:
            logger.error("Canonical Run export failed")
            raise HTTPException(
                status_code=503,
                detail="Canonical Run export is temporarily unavailable",
            ) from exc
        return Response(
            content=exported.content,
            media_type="application/json",
            headers={
                "Content-Disposition": 'attachment; filename="%s"'
                % exported.filename,
                "X-Content-Type-Options": "nosniff",
                "ETag": '"%s"' % exported.etag,
            },
        )

    @router.get("/api/v1/runs/{run_id}", response_model=V1RunResponse)
    def v1_run(run_id: str, request: Request) -> V1RunResponse:
        return V1RunResponse.model_validate(
            get_accessible_run(run_access_service, run_id, request)
        )

    @router.get(
        "/api/v1/runs/{run_id}/events",
        response_model=List[V1RunEventResponse],
    )
    def v1_events(
        run_id: str,
        request: Request,
        after_sequence: int = Query(0, ge=0),
    ) -> List[V1RunEventResponse]:
        get_accessible_run(run_access_service, run_id, request)
        events = adapter.run_service.list_events(
            run_id, after_sequence=after_sequence
        )
        return [
            V1RunEventResponse.model_validate(event)
            for event in events
        ]

    @router.get(
        "/api/v1/runs/{run_id}/artifacts",
        response_model=List[V1RunArtifactResponse],
    )
    def v1_artifacts(
        run_id: str, request: Request
    ) -> List[V1RunArtifactResponse]:
        get_accessible_run(run_access_service, run_id, request)
        if artifact_service is None:
            return []
        try:
            return [
                V1RunArtifactResponse.model_validate(artifact)
                for artifact in artifact_service.list_artifacts(run_id)
            ]
        except RunNotFound as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        except RepositoryError as exc:
            raise HTTPException(
                status_code=503,
                detail="Artifact metadata is temporarily unavailable",
            ) from exc

    @router.get("/api/v1/runs/{run_id}/artifacts/{artifact_id}/download")
    def v1_artifact_download(
        run_id: str, artifact_id: str, request: Request
    ) -> StreamingResponse:
        if artifact_download_service is None:
            raise HTTPException(
                status_code=503,
                detail="Artifact download is not configured",
            )
        try:
            download = artifact_download_service.prepare_download(
                run_id, artifact_id, request.state.current_user.user_id
            )
        except ArtifactNotFound as exc:
            raise HTTPException(status_code=404, detail="Artifact not found") from exc
        except ArtifactIntegrityError as exc:
            raise HTTPException(
                status_code=409,
                detail="Artifact integrity check failed",
            ) from exc
        except ArtifactServiceUnavailable as exc:
            logger.error("Artifact download service failed")
            raise HTTPException(
                status_code=503,
                detail="Artifact download is temporarily unavailable",
            ) from exc
        return StreamingResponse(
            download.chunks,
            media_type=download.media_type,
            headers=download.headers,
        )

    @router.get(
        "/api/v1/runs/{run_id}/stream",
        response_model=V1StreamDescriptorResponse,
    )
    def v1_stream(
        run_id: str,
        request: Request,
        camera_id: str = Query(
            "overview",
            min_length=1,
            max_length=8,
            pattern=r"^overview$",
        ),
    ) -> V1StreamDescriptorResponse:
        get_accessible_run(run_access_service, run_id, request)
        if stream_session_service is None:
            raise HTTPException(
                status_code=503,
                detail="Simulation stream-session is unavailable",
            )
        try:
            descriptor = stream_session_service.get_stream(run_id, camera_id)
        except RunNotFound as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        except StreamSessionNotAllowed as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except StreamSessionUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail="Simulation stream-session is unavailable",
            ) from exc
        except (
            SimulationTimeoutError,
            SimulationNetworkError,
        ) as exc:
            raise HTTPException(
                status_code=503,
                detail="Simulation Service is temporarily unavailable",
            ) from exc
        except SimulationServerError as exc:
            detail = (
                "Simulation video is not ready; retry later"
                if exc.status_code == 503 and exc.code == "stream_not_ready"
                else "Simulation Service is temporarily unavailable"
            )
            raise HTTPException(status_code=503, detail=detail) from exc
        except SimulationRejectedError as exc:
            if exc.status_code == 401 and exc.code == "unauthorized":
                status_code = 503
                detail = "Simulation Service is temporarily unavailable"
            elif (
                exc.status_code == 409
                and exc.code == "stream_session_not_allowed"
            ):
                status_code = 409
                detail = "Simulation video is not available for the task state"
            else:
                status_code = 502
                detail = "Simulation Service rejected the stream request"
            raise HTTPException(status_code=status_code, detail=detail) from exc
        except (
            SimulationConfigurationError,
            SimulationResponseError,
            StreamDescriptorMismatch,
        ) as exc:
            raise HTTPException(
                status_code=502,
                detail="Simulation Service returned an invalid stream descriptor",
            ) from exc
        except RepositoryError as exc:
            raise HTTPException(
                status_code=503,
                detail="Run lookup is temporarily unavailable",
            ) from exc
        return V1StreamDescriptorResponse.model_validate(descriptor)

    @router.post("/api/v1/runs/{run_id}/stop", response_model=V1RunResponse)
    def v1_stop(run_id: str, request: Request) -> V1RunResponse:
        run = get_controllable_run(run_access_service, run_id, request)
        if run.execution_mode.value == "simulation":
            if simulation_stop_service is None:
                raise HTTPException(
                    status_code=503, detail="Simulation stop is not configured"
                )
            try:
                stopped = simulation_stop_service.stop(run_id)
            except RunStopNotAllowed as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except SimulationStopUnavailable as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Simulation stop is temporarily unavailable",
                ) from exc
            except (
                SimulationTimeoutError,
                SimulationNetworkError,
                SimulationServerError,
            ) as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Simulation Service is temporarily unavailable",
                ) from exc
            except SimulationRejectedError as exc:
                if exc.status_code == 401:
                    status_code = 503
                    detail = "Simulation Service is temporarily unavailable"
                elif exc.status_code == 409:
                    status_code = 409
                    detail = "Simulation task cannot be stopped in its current state"
                elif exc.status_code == 404:
                    status_code = 502
                    detail = "Simulation task was not found"
                else:
                    status_code = 502
                    detail = "Simulation Service rejected the stop request"
                raise HTTPException(status_code=status_code, detail=detail) from exc
            except (SimulationResponseError, SimulationSnapshotMismatch) as exc:
                raise HTTPException(
                    status_code=502,
                    detail="Simulation Service returned an invalid stop snapshot",
                ) from exc
            except RepositoryError as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Run stop persistence is temporarily unavailable",
                ) from exc
            return V1RunResponse.model_validate(stopped)
        if run.status in {
            CanonicalRunStatus.INITIALIZING,
            CanonicalRunStatus.FINALIZING,
        }:
            raise HTTPException(
                status_code=409,
                detail="Run %s cannot be stopped while %s"
                % (run_id, run.status.value),
            )
        try:
            adapter.stop_canonical_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return V1RunResponse.model_validate(
            get_canonical_run_or_404(
                adapter, run_id, request.state.current_user.user_id
            )
        )

    @router.websocket("/api/v1/runs/{run_id}/events")
    async def v1_run_events(websocket: WebSocket, run_id: str) -> None:
        authentication = await authenticate_websocket(
            websocket,
            current_user_provider,
            timeout_seconds=websocket_auth_timeout_seconds,
        )
        if authentication is None:
            return
        current_user = authentication.user
        after_sequence = _websocket_after_sequence(websocket)
        if after_sequence is None:
            await websocket.close(code=4400)
            return
        if run_event_stream is None:
            await websocket.close(code=1011)
            return

        loop = asyncio.get_running_loop()
        try:
            run = await loop.run_in_executor(
                None,
                run_access_service.read,
                run_id,
                current_user.user_id,
            )
        except RunAccessNotFound:
            await websocket.close(code=4404)
            return
        except RunAccessUnavailable:
            logger.error("Canonical Run Event stream repository lookup failed")
            await websocket.close(code=1011)
            return

        if not authentication.accepted:
            await websocket.accept()
        disconnected = asyncio.create_task(_wait_for_disconnect(websocket))
        cursor = after_sequence
        terminal_at_connect = run_event_stream.run_is_terminal(run)
        poll = None
        delay = None
        try:
            while True:
                poll = asyncio.ensure_future(
                    loop.run_in_executor(
                        None, run_event_stream.list_batch, run_id, cursor
                    )
                )
                done, _ = await asyncio.wait(
                    {poll, disconnected},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if disconnected in done:
                    poll.cancel()
                    return
                try:
                    events = poll.result()
                except Exception:
                    logger.error(
                        "Canonical Run Event stream repository poll failed"
                    )
                    await websocket.close(code=1011)
                    return

                if not events:
                    if terminal_at_connect:
                        await websocket.close(code=1000)
                        return
                    delay = asyncio.create_task(
                        asyncio.sleep(
                            run_event_stream.config.poll_interval_seconds
                        )
                    )
                    done, _ = await asyncio.wait(
                        {delay, disconnected},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if disconnected in done:
                        delay.cancel()
                        return
                    continue

                batch_event_ids = set()
                for event in events:
                    if event.sequence <= cursor or event.event_id in batch_event_ids:
                        continue
                    message = V1RunEventResponse.model_validate(event).model_dump(
                        mode="json"
                    )
                    await websocket.send_json(message)
                    batch_event_ids.add(event.event_id)
                    cursor = event.sequence
                    if run_event_stream.event_is_terminal(event):
                        await websocket.close(code=1000)
                        return
        except (WebSocketDisconnect, RuntimeError):
            return
        finally:
            if poll is not None:
                poll.cancel()
            if delay is not None:
                delay.cancel()
            disconnected.cancel()
            with suppress(asyncio.CancelledError):
                await disconnected

    @router.websocket("/ws/runs/{run_id}")
    async def run_events(websocket: WebSocket, run_id: str) -> None:
        await websocket.accept()
        try:
            adapter.get_run(run_id)
            sent = set()
            while True:
                for event in adapter.stream_events(run_id):
                    if event.event_id not in sent:
                        await websocket.send_json(event.to_dict())
                        sent.add(event.event_id)
                if adapter.get_run(run_id).status not in (
                    RunStatus.QUEUED,
                    RunStatus.RUNNING,
                ):
                    return
                await asyncio.sleep(0.02)
        except (KeyError, WebSocketDisconnect):
            return

    return router


def get_run_or_404(adapter: MockSimulatorAdapter, run_id: str):
    try:
        return adapter.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def get_canonical_run_or_404(
    adapter: MockSimulatorAdapter, run_id: str, user_id: Optional[str] = None
):
    try:
        run = adapter.run_service.get_run(run_id)
    except RunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if user_id is not None and run.user_id != user_id:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


def get_accessible_run(run_access_service: Any, run_id: str, request: Request):
    try:
        return run_access_service.read(
            run_id, request.state.current_user.user_id
        )
    except RunAccessNotFound as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    except RunAccessUnavailable as exc:
        raise HTTPException(
            status_code=503, detail="Run lookup is temporarily unavailable"
        ) from exc


def get_controllable_run(run_access_service: Any, run_id: str, request: Request):
    try:
        return run_access_service.control(
            run_id, request.state.current_user.user_id
        )
    except RunAccessNotFound as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    except RunAccessForbidden as exc:
        raise HTTPException(status_code=403, detail="Run control is forbidden") from exc
    except RunAccessUnavailable as exc:
        raise HTTPException(
            status_code=503, detail="Run lookup is temporarily unavailable"
        ) from exc


def _websocket_after_sequence(websocket: WebSocket) -> Optional[int]:
    values = websocket.query_params.getlist("after_sequence")
    if not values:
        return 0
    if len(values) != 1 or not values[0] or not values[0].isdigit():
        return None
    try:
        return int(values[0])
    except ValueError:
        return None


async def _wait_for_disconnect(websocket: WebSocket) -> None:
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return


def _launch(launcher: Any, payload: RunCreate) -> Dict[str, Any]:
    try:
        record = launcher(RunRequest(**payload.model_dump()))
        return record.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

