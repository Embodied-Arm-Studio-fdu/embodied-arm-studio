from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from sqlalchemy.engine import URL

from agent.context import AgentContextService
from agent.policy import ASSISTANCE_CAPABILITIES, ASSISTANCE_LEVELS, StudentDesign, TeachingPolicy
from agent.provider import (
    PROVIDERS,
    provider_config,
    provider_status,
    validate_runtime_provider_config,
)
from agent.workspace import WorkspaceService
from apps.api.database import check_postgres_readiness
from apps.api.dependencies.auth import (
    CurrentUserProvider,
    UnavailableCurrentUserProvider,
)
from apps.api.dependencies.runs import RunCreationService, build_run_assembly
from apps.api.dependencies.projects import build_teaching_project_service
from apps.api.dependencies.simulation import build_simulation_assembly
from apps.api.routers.internal_runs import create_internal_runs_router
from apps.api.routers.auth import create_auth_router
from apps.api.routers.courses import create_courses_router
from apps.api.routers.organizations import create_organizations_router
from apps.api.routers.projects import create_teaching_projects_router
from apps.api.routers.evaluations import create_evaluations_router
from apps.api.routers.input import InputWebSocketConfig, create_input_router
from apps.api.routers.runs import create_runs_router, get_run_or_404
from apps.api.routers.simulation import create_simulation_router
from robotics.contracts import RunStatus
from robotics.auth import (
    AuthenticationService,
    JWTConfiguration,
    PasswordHasher,
    PostgreSQLUserRepository,
)
from robotics.auth.tokens import JWTAccessTokenService
from robotics.organizations import (
    OrganizationService,
    PostgreSQLOrganizationRepository,
)
from robotics.courses import CourseService, PostgreSQLCourseRepository
from robotics.projects import TeachingProjectService
from robotics.run_access import (
    OwnerOnlyRunAccessRepository,
    PostgreSQLRunAccessRepository,
    RunAccessService,
)
from robotics.artifacts import ArtifactService, LocalArtifactStorage
from robotics.evaluation import EvaluationService
from robotics.evaluations import (
    CanonicalRunComparisonExportService,
    EvaluationService as CanonicalEvaluationService,
    EvaluationNotFound,
    EvaluationUnavailable,
    PostgreSQLEvaluationRepository,
)
from robotics.pipelines import PipelineRepository
from robotics.runs import RunEventStream, RunEventStreamConfig, RunRepository
from robotics.runs.postgres_callback import PostgreSQLTerminalCallbackService
from robotics.sandbox import ContainerBackend, RemoteWorkerBackend, SandboxRunner
from robotics.scenes import SceneRepository, scene_preflight
from robotics.strategies import STRATEGIES
from robotics.training_code import TrainingCodeRepository, validate_training_code


class ProjectCreate(BaseModel):
    project_id: Optional[str] = None


class WorkspaceFileWrite(BaseModel):
    content: str = Field(..., max_length=524288)


class WorkspaceDirectoryCreate(BaseModel):
    path: str = Field(..., min_length=1, max_length=240)


class WorkspaceMove(BaseModel):
    source: str = Field(..., min_length=1, max_length=240)
    destination: str = Field(..., min_length=1, max_length=240)


class AgentSessionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    provider: str = "mock"
    assistance_level: int = Field(4, ge=1, le=5)
    locale: str = Field("zh-CN", pattern="^(zh-CN|en)$")


class AgentProviderConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_key: SecretStr
    model: str = Field(..., min_length=1, max_length=160)
    base_url: str = Field("", max_length=500)


class AgentPageContext(BaseModel):
    path: str = Field(..., min_length=1, max_length=240)
    project_id: Optional[str] = Field(None, max_length=160)
    run_id: Optional[str] = Field(None, max_length=160)
    scene_id: Optional[str] = Field(None, max_length=160)
    evaluation_id: Optional[str] = Field(None, max_length=255)


class AgentMessage(BaseModel):
    content: str = Field(..., min_length=1, max_length=4000)
    context: Optional[AgentPageContext] = None


class StudentDesignCreate(BaseModel):
    observation: str = Field(..., min_length=1, max_length=2000)
    action: str = Field(..., min_length=1, max_length=2000)
    reward: str = Field(..., min_length=1, max_length=2000)
    termination: str = Field(..., min_length=1, max_length=2000)
    expected_result: str = Field("", max_length=2000)


class PatchCreate(BaseModel):
    session_id: str
    project_id: str
    path: str
    diff: str


class PatchApproval(BaseModel):
    confirm_high_risk: bool = False


class RollbackRequest(BaseModel):
    snapshot_id: str


class PipelineDocument(BaseModel):
    document: Dict[str, Any]


class EvaluationCreate(BaseModel):
    run_id: Optional[str] = None
    run_ids: List[str] = Field(default_factory=list, max_length=8)


class EvaluationCompare(BaseModel):
    before_report_id: str
    after_report_id: str


class SceneDocument(BaseModel):
    document: Dict[str, Any]


class TrainingCodeDocument(BaseModel):
    document: Dict[str, Any]


class TrainingCodeValidation(BaseModel):
    content: str


class SandboxJobCreate(BaseModel):
    template_id: str = Field(..., min_length=1, max_length=80)

    class Config:
        extra = "forbid"


def create_app(
    storage_root: Optional[Path] = None,
    tick_seconds: Optional[float] = None,
    project_root: Optional[Path] = None,
    sandbox_backend: Optional[ContainerBackend] = None,
    run_repository: Optional[RunRepository] = None,
    run_repository_mode: Optional[str] = None,
    database_url: Optional[Union[str, URL]] = None,
    database_environment: Optional[Mapping[str, str]] = None,
    run_creation_service: Optional[RunCreationService] = None,
    current_user_provider: Optional[Any] = None,
    authentication_service: Optional[AuthenticationService] = None,
    organization_service: Optional[OrganizationService] = None,
    course_service: Optional[CourseService] = None,
    teaching_project_service: Optional[TeachingProjectService] = None,
    jwt_secret: Optional[str] = None,
    jwt_issuer: Optional[str] = None,
    jwt_audience: Optional[str] = None,
    jwt_expires_seconds: Optional[int] = None,
    internal_run_event_service_token: Optional[str] = None,
    simulation_client: Optional[Any] = None,
    simulation_environment: Optional[Mapping[str, str]] = None,
    run_event_poll_interval_seconds: Optional[float] = None,
    run_event_batch_size: Optional[int] = None,
    websocket_auth_timeout_seconds: float = 5.0,
    input_ws_max_message_bytes: int = 8192,
    input_ws_messages_per_second: float = 60.0,
    input_ws_burst: int = 120,
    input_ws_rate_limit_close_after: int = 3,
    artifact_storage_root: Optional[Path] = None,
    artifact_metadata_service: Optional[Any] = None,
    artifact_download_service: Optional[Any] = None,
    canonical_evaluation_service: Optional[Any] = None,
) -> FastAPI:
    root = storage_root or Path(os.environ.get("STUDIO_RUNS_DIR", "var/runs"))
    tick = tick_seconds if tick_seconds is not None else float(os.environ.get("STUDIO_MOCK_TICK", "0.03"))
    scene_repository = SceneRepository(root / "scenes")
    run_assembly = build_run_assembly(
        storage_root=root,
        tick_seconds=tick,
        repository=run_repository,
        repository_mode=run_repository_mode,
        database_url=database_url,
        environment=database_environment,
        creation_service=run_creation_service,
        scene_repository=scene_repository,
    )
    adapter = run_assembly.adapter
    run_source = (
        os.environ if database_environment is None else database_environment
    )
    resolved_authentication_service = authentication_service
    resolved_current_user_provider = current_user_provider
    if (
        resolved_authentication_service is None
        and run_assembly.session_factory is not None
        and resolved_current_user_provider is None
    ):
        resolved_secret = jwt_secret or run_source.get("USER_JWT_SECRET")
        if not resolved_secret:
            raise RuntimeError(
                "USER_JWT_SECRET is required for PostgreSQL application mode"
            )
        jwt_config = JWTConfiguration(
            secret=resolved_secret,
            issuer=jwt_issuer
            or run_source.get("USER_JWT_ISSUER", "embodied-arm-studio"),
            audience=jwt_audience
            or run_source.get(
                "USER_JWT_AUDIENCE", "embodied-arm-studio-api"
            ),
            expires_seconds=(
                jwt_expires_seconds
                if jwt_expires_seconds is not None
                else int(run_source.get("USER_JWT_EXPIRES_SECONDS", "900"))
            ),
        )
        resolved_authentication_service = AuthenticationService(
            PostgreSQLUserRepository(run_assembly.session_factory),
            PasswordHasher(),
            JWTAccessTokenService(jwt_config),
        )
    if resolved_current_user_provider is None:
        if resolved_authentication_service is None:
            resolved_current_user_provider = UnavailableCurrentUserProvider()
        else:
            resolved_current_user_provider = CurrentUserProvider(
                resolved_authentication_service
            )
    resolved_organization_service = organization_service
    if (
        resolved_organization_service is None
        and run_assembly.session_factory is not None
    ):
        resolved_organization_service = OrganizationService(
            PostgreSQLOrganizationRepository(run_assembly.session_factory)
        )
    resolved_course_service = course_service
    if resolved_course_service is None and run_assembly.session_factory is not None:
        resolved_course_service = CourseService(
            PostgreSQLCourseRepository(run_assembly.session_factory)
        )
    run_access_service = RunAccessService(
        PostgreSQLRunAccessRepository(run_assembly.session_factory)
        if run_assembly.session_factory is not None
        else OwnerOnlyRunAccessRepository(run_assembly.service)
    )
    resolved_teaching_project_service = teaching_project_service
    if resolved_teaching_project_service is None:
        resolved_teaching_project_service = build_teaching_project_service(
            run_assembly.session_factory
        )
    stream_config = RunEventStreamConfig(
        poll_interval_seconds=(
            run_event_poll_interval_seconds
            if run_event_poll_interval_seconds is not None
            else float(
                run_source.get("RUN_EVENT_WS_POLL_INTERVAL_SECONDS", "0.15")
            )
        ),
        batch_size=(
            run_event_batch_size
            if run_event_batch_size is not None
            else int(run_source.get("RUN_EVENT_WS_BATCH_SIZE", "100"))
        ),
    )
    run_event_stream = RunEventStream(run_assembly.service, stream_config)
    input_websocket_config = InputWebSocketConfig(
        auth_timeout_seconds=websocket_auth_timeout_seconds,
        max_message_bytes=input_ws_max_message_bytes,
        messages_per_second=input_ws_messages_per_second,
        burst=input_ws_burst,
        rate_limit_close_after=input_ws_rate_limit_close_after,
    )
    simulation_assembly = build_simulation_assembly(
        run_assembly.service,
        client=simulation_client,
        environment=simulation_environment,
        dispatch_authorizer=run_assembly.creation_service,
    )
    configure_catalog = getattr(
        run_assembly.creation_service,
        "configure_simulation_catalog_client",
        None,
    )
    if callable(configure_catalog) and simulation_assembly.client is not None:
        configure_catalog(simulation_assembly.client)
    repository_root = Path(__file__).resolve().parents[2]
    workspace_service = WorkspaceService(
        project_root or Path(os.environ.get("STUDIO_PROJECTS_DIR", "var/projects")),
        repository_root / "templates",
    )
    pipeline_repository = PipelineRepository(root.parent / "pipelines")
    evaluation_service = EvaluationService(root.parent / "evaluations")
    training_code_repository = TrainingCodeRepository(root / "training-code")
    resolved_sandbox_backend = sandbox_backend or RemoteWorkerBackend.from_environment()
    sandbox_runner = SandboxRunner(root.parent / "sandbox", resolved_sandbox_backend)
    teaching_policy = TeachingPolicy(repository_root / "agent" / "policy" / "blocked_intents.yaml")
    agent_sessions: Dict[str, Dict[str, Any]] = {}
    agent_participation: List[Dict[str, Any]] = []
    agent_provider_configs: Dict[str, Dict[str, str]] = {}
    agent_provider_checks: Dict[str, Dict[str, Any]] = {}
    app = FastAPI(title="Embodied Arm Studio API", version="0.1.0")
    from apps.api.routers.workbench import create_workbench_router
    app.include_router(create_workbench_router(root / "experiment-drafts", repository_root / "simulation" / "src" / "simulation_core"))
    terminal_callback_service = (
        PostgreSQLTerminalCallbackService(run_assembly.session_factory)
        if run_assembly.session_factory is not None
        else None
    )
    resolved_artifact_metadata_service = (
        artifact_metadata_service or terminal_callback_service
    )
    resolved_artifact_download_service = artifact_download_service
    if (
        resolved_artifact_download_service is None
        and resolved_artifact_metadata_service is not None
    ):
        configured_artifact_root = artifact_storage_root
        if configured_artifact_root is None:
            artifact_root_value = run_source.get("ARTIFACT_STORAGE_ROOT")
            if artifact_root_value:
                configured_artifact_root = Path(artifact_root_value)
        if configured_artifact_root is not None:
            # Fresh named volumes have no artifacts directory. Create it before
            # LocalArtifactStorage resolves it strictly; never alter existing data
            # or permissions. The runtime UID must own a writable volume parent.
            configured_artifact_root = Path(configured_artifact_root)
            if (configured_artifact_root.is_absolute()
                    and configured_artifact_root != Path("/")):
                configured_artifact_root.mkdir(mode=0o750, parents=True, exist_ok=True)
            resolved_artifact_download_service = ArtifactService(
                run_assembly.service,
                resolved_artifact_metadata_service,
                LocalArtifactStorage(configured_artifact_root),
                run_access_service=run_access_service,
            )
    resolved_canonical_evaluation_service = canonical_evaluation_service
    if (
        resolved_canonical_evaluation_service is None
        and run_assembly.session_factory is not None
        and resolved_artifact_metadata_service is not None
    ):
        resolved_canonical_evaluation_service = CanonicalEvaluationService(
            run_assembly.service,
            resolved_artifact_metadata_service,
            PostgreSQLEvaluationRepository(run_assembly.session_factory),
            run_access_service=run_access_service,
        )
    agent_context_service = AgentContextService(
        workspace_service,
        scene_repository,
        run_assembly.service,
        resolved_canonical_evaluation_service,
    )
    canonical_run_comparison_export_service = None
    if (
        run_assembly.session_factory is not None
        and resolved_artifact_metadata_service is not None
    ):
        canonical_run_comparison_export_service = (
            CanonicalRunComparisonExportService(
                run_assembly.service,
                resolved_artifact_metadata_service,
                PostgreSQLEvaluationRepository(run_assembly.session_factory),
                run_access_service=run_access_service,
            )
        )
    app.state.engine = run_assembly.engine
    app.state.session_factory = run_assembly.session_factory
    app.state.repository = run_assembly.repository
    app.state.run_service = run_assembly.service
    app.state.run_creation_service = run_assembly.creation_service
    app.state.run_access_service = run_access_service
    app.state.terminal_callback_service = terminal_callback_service
    app.state.artifact_metadata_service = resolved_artifact_metadata_service
    app.state.artifact_download_service = resolved_artifact_download_service
    app.state.canonical_evaluation_service = (
        resolved_canonical_evaluation_service
    )
    app.state.canonical_run_comparison_export_service = (
        canonical_run_comparison_export_service
    )
    app.state.simulation_client = simulation_assembly.client
    app.state.simulation_dispatch_service = simulation_assembly.dispatch_service
    app.state.simulation_stop_service = simulation_assembly.stop_service
    app.state.simulation_recovery_service = simulation_assembly.recovery_service
    app.state.stream_session_service = simulation_assembly.stream_service
    app.state.simulation_recovery_result = None
    app.state.adapter = adapter
    app.state.run_repository_mode = run_assembly.repository_mode
    app.state.authentication_service = resolved_authentication_service
    app.state.current_user_provider = resolved_current_user_provider
    app.state.organization_service = resolved_organization_service
    app.state.course_service = resolved_course_service
    app.state.teaching_project_service = resolved_teaching_project_service
    app.state.run_event_stream = run_event_stream
    app.state.input_websocket_config = input_websocket_config
    app.state.workspace_service = workspace_service
    app.state.pipeline_repository = pipeline_repository
    app.state.evaluation_service = evaluation_service
    app.state.scene_repository = scene_repository
    app.state.training_code_repository = training_code_repository
    app.state.sandbox_runner = sandbox_runner
    app.state.agent_sessions = agent_sessions
    app.state.agent_participation = agent_participation
    app.state.agent_provider_configs = agent_provider_configs
    app.state.agent_provider_checks = agent_provider_checks
    app.state.agent_context_service = agent_context_service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    @app.middleware("http")
    async def authenticate_canonical_api(request: Request, call_next):
        path = request.url.path
        if (
            request.method != "OPTIONS"
            and (path.startswith("/api/v1/") or path.startswith("/api/agent/"))
            and path != "/api/v1/auth/login"
        ):
            try:
                request.state.current_user = (
                    resolved_current_user_provider.from_authorization(
                        request.headers.get("authorization", "")
                    )
                )
            except HTTPException:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing bearer credentials"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
        return await call_next(request)

    app.include_router(
        create_auth_router(
            resolved_authentication_service,
            resolved_current_user_provider,
        )
    )
    app.include_router(
        create_organizations_router(
            resolved_organization_service,
            resolved_current_user_provider,
        )
    )
    app.include_router(
        create_courses_router(
            resolved_course_service,
            resolved_current_user_provider,
        )
    )
    app.include_router(
        create_teaching_projects_router(
            resolved_teaching_project_service,
            resolved_current_user_provider,
        )
    )
    app.include_router(
        create_runs_router(
            adapter,
            evaluation_service,
            run_creation_service=run_assembly.creation_service,
            run_deletion_repository=run_assembly.repository,
            artifact_service=resolved_artifact_metadata_service,
            artifact_download_service=resolved_artifact_download_service,
            simulation_dispatch_service=simulation_assembly.dispatch_service,
            simulation_stop_service=simulation_assembly.stop_service,
            stream_session_service=simulation_assembly.stream_service,
            run_event_stream=run_event_stream,
            comparison_export_service=canonical_run_comparison_export_service,
            run_access_service=run_access_service,
            current_user_provider=resolved_current_user_provider,
            websocket_auth_timeout_seconds=websocket_auth_timeout_seconds,
        )
    )
    app.include_router(
        create_input_router(
            run_assembly.service,
            simulation_assembly.client,
            resolved_current_user_provider,
            input_websocket_config,
            run_access_service,
        )
    )
    app.include_router(
        create_simulation_router(
            simulation_assembly.client,
            resolved_current_user_provider,
        )
    )
    app.include_router(
        create_evaluations_router(
            resolved_canonical_evaluation_service,
            run_access_service,
        )
    )
    resolved_internal_token = internal_run_event_service_token
    if resolved_internal_token is None:
        resolved_internal_token = (
            os.environ if database_environment is None else database_environment
        ).get("INTERNAL_RUN_EVENT_SERVICE_TOKEN")
    app.include_router(
        create_internal_runs_router(
            run_assembly.service,
            resolved_internal_token,
            callback_service=terminal_callback_service,
        )
    )

    if run_assembly.engine is not None:
        @app.on_event("startup")
        def check_run_database() -> None:
            try:
                check_postgres_readiness(run_assembly.engine)
            except Exception:
                run_assembly.engine.dispose()
                raise

        @app.on_event("shutdown")
        def dispose_run_database() -> None:
            run_assembly.engine.dispose()

    if simulation_assembly.client is not None:
        @app.on_event("startup")
        def reconcile_simulation_runs() -> None:
            if simulation_assembly.recovery_service is None:
                return
            try:
                app.state.simulation_recovery_result = (
                    simulation_assembly.recovery_service.reconcile_simulation_runs()
                )
            except Exception:
                # Recovery is best-effort and must never block read-only startup.
                app.state.simulation_recovery_result = None

        @app.on_event("shutdown")
        def close_simulation_client() -> None:
            simulation_assembly.client.close()

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return adapter.health()

    @app.get("/api/ready")
    def readiness() -> JSONResponse:
        """Report deploy-time capability readiness without exposing configuration."""
        database_ready = False
        if app.state.engine is not None:
            try:
                check_postgres_readiness(app.state.engine)
                database_ready = True
            except Exception:
                database_ready = False
        payload = {
            "status": "ready" if database_ready else "not_ready",
            "database_ready": database_ready,
            "simulation_configured": app.state.simulation_client is not None,
            "artifact_download_configured": (
                app.state.artifact_download_service is not None
            ),
        }
        return JSONResponse(
            status_code=200 if database_ready else 503,
            content=payload,
        )

    @app.get("/api/environments")
    def environments() -> List[Dict[str, Any]]:
        return [asdict(environment) for environment in adapter.list_environments()]

    @app.get("/api/scenes")
    def scenes() -> List[Dict[str, Any]]:
        return scene_repository.list_scenes()

    @app.post("/api/scenes", status_code=201)
    def save_scene(payload: SceneDocument) -> Dict[str, Any]:
        try:
            return scene_repository.save(payload.document)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/scenes/{scene_id}/export")
    def export_scene(scene_id: str) -> Dict[str, Any]:
        try:
            scene_data = scene_repository.get(scene_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="场景不存在") from exc
        return {"format": "embodied-arm-scene", "scene": scene_data, "real_robot_control_enabled": False, "sim2real": "blocked"}

    @app.get("/api/scenes/{scene_id}")
    def scene(scene_id: str) -> Dict[str, Any]:
        try:
            return scene_repository.get(scene_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="场景不存在") from exc

    @app.get("/api/scenes/{scene_id}/preflight")
    def scene_preflight_route(scene_id: str) -> Dict[str, Any]:
        try:
            return scene_preflight(scene_repository.get(scene_id))
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="场景不存在") from exc

    @app.get("/api/training-code")
    def training_code_list() -> List[Dict[str, Any]]:
        return training_code_repository.list()

    @app.get("/api/training-code/{code_id}")
    def training_code(code_id: str) -> Dict[str, Any]:
        try:
            return training_code_repository.get(code_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="训练代码不存在") from exc

    @app.post("/api/training-code/validate")
    def validate_code(payload: TrainingCodeValidation) -> Dict[str, Any]:
        return validate_training_code(payload.content)

    @app.post("/api/training-code", status_code=201)
    def save_training_code(payload: TrainingCodeDocument) -> Dict[str, Any]:
        try:
            return training_code_repository.save(payload.document)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/environments/{environment_id}/schema")
    def environment_schema(environment_id: str) -> Dict[str, Any]:
        try:
            return adapter.get_environment_schema(environment_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/strategies")
    def strategies() -> List[Dict[str, Any]]:
        return [asdict(strategy.info) for strategy in STRATEGIES.values()]

    @app.get("/api/pipelines")
    def pipelines() -> List[Dict[str, Any]]:
        return pipeline_repository.list_pipelines()

    @app.get("/api/pipelines/{pipeline_id}")
    def pipeline(pipeline_id: str, revision: Optional[int] = None) -> Dict[str, Any]:
        try:
            return pipeline_repository.get(pipeline_id, revision)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/pipelines/validate")
    def validate_pipeline(payload: PipelineDocument) -> Dict[str, Any]:
        return pipeline_repository.validate(payload.document)

    @app.post("/api/pipelines", status_code=201)
    def save_pipeline(payload: PipelineDocument) -> Dict[str, Any]:
        try:
            return pipeline_repository.save_revision(payload.document)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/courses")
    def courses() -> List[Dict[str, Any]]:
        return [_load_course(repository_root / "courses" / "quick_start.json")]

    @app.get("/api/courses/{course_id}")
    def course(course_id: str) -> Dict[str, Any]:
        course_data = _load_course(repository_root / "courses" / "quick_start.json")
        if course_data["id"] != course_id:
            raise HTTPException(status_code=404, detail="未知课程")
        return course_data

    @app.post("/api/projects", status_code=201)
    def create_project(payload: ProjectCreate) -> Dict[str, str]:
        try:
            return workspace_service.create_project(payload.project_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/v1/projects")
    def v1_projects() -> List[Dict[str, Any]]:
        return workspace_service.list_projects()

    @app.post("/api/v1/projects", status_code=201)
    def v1_create_project(payload: ProjectCreate) -> Dict[str, str]:
        try:
            return workspace_service.create_project(payload.project_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/v1/workspaces/{project_id}")
    def v1_workspace(project_id: str) -> Dict[str, Any]:
        try:
            return {"project_id": project_id, "tree": workspace_service.file_tree(project_id)}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/workspaces/{project_id}/files/{path:path}")
    def v1_workspace_file(project_id: str, path: str) -> Dict[str, str]:
        try:
            return {"path": path, "content": workspace_service.read_file(project_id, path)}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.put("/api/v1/workspaces/{project_id}/files/{path:path}")
    def v1_write_workspace_file(project_id: str, path: str, payload: WorkspaceFileWrite) -> Dict[str, Any]:
        try:
            return workspace_service.write_file(project_id, path, payload.content)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/workspaces/{project_id}/directories", status_code=201)
    def v1_create_workspace_directory(project_id: str, payload: WorkspaceDirectoryCreate) -> Dict[str, str]:
        try:
            return workspace_service.create_directory(project_id, payload.path)
        except (FileExistsError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/workspaces/{project_id}/move")
    def v1_move_workspace_path(project_id: str, payload: WorkspaceMove) -> Dict[str, str]:
        try:
            return workspace_service.move(project_id, payload.source, payload.destination)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/api/v1/workspaces/{project_id}/files/{path:path}")
    def v1_delete_workspace_path(project_id: str, path: str) -> None:
        try:
            workspace_service.delete(project_id, path)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/v1/workspaces/{project_id}/search")
    def v1_search_workspace(project_id: str, q: str) -> List[Dict[str, Any]]:
        try:
            return workspace_service.search(project_id, q)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/v1/workspaces/{project_id}/history/{path:path}")
    def v1_workspace_history(project_id: str, path: str) -> List[Dict[str, str]]:
        try:
            return workspace_service.file_history(project_id, path)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/v1/workspaces/{project_id}/history/{path:path}/versions/{version_id}/restore")
    def v1_restore_workspace_version(project_id: str, path: str, version_id: str) -> Dict[str, Any]:
        try:
            return workspace_service.restore_version(project_id, path, version_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/v1/workspaces/{project_id}/diff")
    def v1_workspace_diff(project_id: str, path: str) -> Dict[str, str]:
        try:
            return workspace_service.starter_diff(project_id, path)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/v1/workspaces/{project_id}/starter/restore")
    def v1_restore_workspace_starter(project_id: str) -> Dict[str, str]:
        try:
            return workspace_service.restore_starter(project_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/workspaces/{project_id}/snapshots", status_code=201)
    def v1_snapshot_workspace(project_id: str) -> Dict[str, str]:
        try:
            return workspace_service.create_snapshot(project_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/workspaces/{project_id}/snapshots/{snapshot_id}/restore")
    def v1_restore_workspace(project_id: str, snapshot_id: str) -> Dict[str, str]:
        try:
            workspace_service.rollback(project_id, snapshot_id)
            return {"status": "restored", "snapshot_id": snapshot_id}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/projects/{project_id}/files")
    def project_files(project_id: str) -> List[str]:
        try:
            return workspace_service.list_files(project_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/projects/{project_id}/files/{path:path}")
    def project_file(project_id: str, path: str) -> Dict[str, str]:
        try:
            return {"path": path, "content": workspace_service.read_file(project_id, path)}
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.post("/api/projects/{project_id}/rollback")
    def rollback_project(project_id: str, payload: RollbackRequest) -> Dict[str, str]:
        try:
            workspace_service.rollback(project_id, payload.snapshot_id)
            return {"status": "rolled_back"}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/agent/providers")
    def providers() -> List[Dict[str, Any]]:
        return [provider_status(provider_id, agent_provider_configs, agent_provider_checks) for provider_id in PROVIDERS]

    def revoke_provider_sessions(provider_id: str) -> None:
        revoked_session_ids = [
            session_id
            for session_id, session in agent_sessions.items()
            if session["provider"] == provider_id
        ]
        for session_id in revoked_session_ids:
            agent_sessions.pop(session_id, None)

    @app.put("/api/agent/providers/{provider_id}/configuration")
    def configure_provider(provider_id: str, payload: AgentProviderConfiguration) -> Dict[str, Any]:
        if provider_id not in PROVIDERS:
            raise HTTPException(status_code=404, detail="未知模型供应商")
        try:
            base_url = validate_runtime_provider_config(provider_id, payload.model, payload.base_url)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not payload.api_key.get_secret_value().strip():
            raise HTTPException(422, "API Key 不能为空")
        agent_provider_configs[provider_id] = {
            "api_key": payload.api_key.get_secret_value().strip(),
            "model": payload.model.strip(),
            "base_url": base_url,
        }
        agent_provider_checks.pop(provider_id, None)
        revoke_provider_sessions(provider_id)
        return provider_status(provider_id, agent_provider_configs, agent_provider_checks)

    @app.post("/api/agent/providers/{provider_id}/check")
    def check_provider_connection(provider_id: str) -> Dict[str, Any]:
        if provider_id not in PROVIDERS:
            raise HTTPException(404, "Unknown Provider")
        try:
            config = provider_config(provider_id, agent_provider_configs)
        except RuntimeError as exc:
            raise HTTPException(422, str(exc)) from exc
        # Probe with a fixed message, never with learner/page/session data.
        try:
            PROVIDERS[provider_id].complete([{"role": "user", "content": "Reply with OK."}], config)
            result = {"status": "healthy", "message": "Connection verified"}
        except RuntimeError as exc:
            result = {"status": "degraded", "message": str(exc)}
        from datetime import datetime, timezone
        result["checked_at"] = datetime.now(timezone.utc).isoformat()
        agent_provider_checks[provider_id] = result
        return provider_status(provider_id, agent_provider_configs, agent_provider_checks)

    @app.delete("/api/agent/providers/{provider_id}/configuration")
    def clear_provider_configuration(provider_id: str) -> Dict[str, Any]:
        if provider_id not in PROVIDERS or provider_id == "mock":
            raise HTTPException(status_code=404, detail="该 Provider 没有网页端配置")
        agent_provider_configs.pop(provider_id, None)
        agent_provider_checks.pop(provider_id, None)
        revoke_provider_sessions(provider_id)
        return provider_status(provider_id, agent_provider_configs, agent_provider_checks)

    @app.get("/api/agent/assistance-levels")
    def assistance_levels() -> List[Dict[str, Any]]:
        return [
            {
                "level": level,
                "id": name,
                "capabilities": sorted(ASSISTANCE_CAPABILITIES[level]),
                "requires_teacher_authorization": False,
            }
            for level, name in ASSISTANCE_LEVELS.items()
        ]

    @app.post("/api/agent/sessions", status_code=201)
    def create_agent_session(payload: AgentSessionCreate) -> Dict[str, str]:
        if payload.provider not in PROVIDERS:
            raise HTTPException(status_code=422, detail="未知模型供应商")
        try:
            config = provider_config(payload.provider, agent_provider_configs)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        try:
            workspace_service.list_files(payload.project_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        session_id = "agent-" + uuid4().hex[:12]
        agent_sessions[session_id] = {
            "project_id": payload.project_id,
            "provider": payload.provider,
            "locale": payload.locale,
            "config": config,
            "assistance_level": payload.assistance_level,
            "capabilities": frozenset(ASSISTANCE_CAPABILITIES[payload.assistance_level]),
            "design": None,
            "messages": [],
        }
        return {"session_id": session_id}

    @app.put("/api/agent/sessions/{session_id}/student-design")
    def save_student_design(session_id: str, payload: StudentDesignCreate) -> Dict[str, Any]:
        session = agent_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="未知 Agent 会话")
        design = StudentDesign(**payload.model_dump())
        session["design"] = design
        agent_participation.append({
            "session_id": session_id,
            "project_id": session["project_id"],
            "event": "student_design_submitted",
            "assistance_level": session["assistance_level"],
        })
        return {"status": "saved", "complete": design.is_complete()}

    @app.get("/api/agent/sessions/{session_id}/participation")
    def participation(session_id: str) -> List[Dict[str, Any]]:
        if session_id not in agent_sessions:
            raise HTTPException(status_code=404, detail="未知 Agent 会话")
        return [entry for entry in agent_participation if entry["session_id"] == session_id]

    @app.post("/api/agent/sessions/{session_id}/messages")
    def send_agent_message(
        session_id: str,
        payload: AgentMessage,
        request: Request,
    ) -> Dict[str, Any]:
        session = agent_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="未知 Agent 会话")
        decision = teaching_policy.evaluate_message(
            payload.content,
            session["assistance_level"],
            session["design"],
        )
        agent_participation.append({
            "session_id": session_id,
            "project_id": session["project_id"],
            "event": "message_policy_decision",
            "intent": decision.intent,
            "outcome": decision.outcome,
            "assistance_level": decision.assistance_level,
        })
        if decision.outcome != "allow":
            raise HTTPException(status_code=403, detail={"outcome": decision.outcome, "intent": decision.intent, "reason": decision.reason})
        prompt_path = repository_root / "agent" / "prompts" / "teaching_agent_system.md"
        messages = [{"role": "system", "content": prompt_path.read_text(encoding="utf-8") + ("\nReply in English." if session.get("locale") == "en" else "\n回复使用中文。")}]
        try:
            trusted_context, tools_used = agent_context_service.build(
                set(session["capabilities"]),
                session["project_id"],
                payload.context,
                request.state.current_user.user_id,
            )
        except (KeyError, ValueError, EvaluationNotFound) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except EvaluationUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if trusted_context:
            messages.append({
                "role": "system",
                "content": agent_context_service.prompt_document(trusted_context),
            })
        messages.extend(session["messages"])
        messages.append({"role": "user", "content": payload.content})
        try:
            reply = PROVIDERS[session["config"].provider].complete(messages, session["config"])
            agent_provider_checks[session["provider"]] = {"status": "healthy", "message": "Chat request succeeded"}
        except RuntimeError as exc:
            agent_provider_checks[session["provider"]] = {"status": "degraded", "message": str(exc)}
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        session["messages"].extend([{"role": "user", "content": payload.content}, {"role": "assistant", "content": reply}])
        return {"content": reply, "tools_used": tools_used}

    @app.post("/api/agent/patches", status_code=201)
    def propose_patch(payload: PatchCreate) -> Dict[str, Any]:
        session = agent_sessions.get(payload.session_id)
        if not session or session["project_id"] != payload.project_id:
            raise HTTPException(status_code=403, detail="Patch 必须关联当前项目的有效 Agent 会话")
        if "propose_patch" not in session["capabilities"]:
            raise HTTPException(status_code=403, detail="当前 Assistance Level 不允许提出 Patch")
        if session["design"] is None or not session["design"].is_complete():
            raise HTTPException(status_code=403, detail="提出 Patch 前需提交完整的学生设计草案")
        try:
            result = asdict(workspace_service.propose_patch(payload.project_id, payload.path, payload.diff))
            agent_participation.append({
                "session_id": payload.session_id,
                "project_id": payload.project_id,
                "event": "patch_proposed",
                "patch_id": result["patch_id"],
                "assistance_level": session["assistance_level"],
            })
            return result
        except (ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/agent/patches/{patch_id}/approve")
    def approve_patch(patch_id: str, payload: PatchApproval) -> Dict[str, Any]:
        try:
            return asdict(workspace_service.approve(patch_id, payload.confirm_high_risk))
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/agent/patches/{patch_id}/reject")
    def reject_patch(patch_id: str) -> Dict[str, Any]:
        try:
            return asdict(workspace_service.reject(patch_id))
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/sandbox/templates")
    def sandbox_templates() -> List[Dict[str, object]]:
        return sandbox_runner.list_templates()

    @app.post("/api/sandbox/jobs", status_code=202)
    def submit_sandbox_job(payload: SandboxJobCreate) -> Dict[str, object]:
        try:
            return sandbox_runner.submit(payload.template_id).to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/sandbox/jobs")
    def sandbox_jobs() -> List[Dict[str, object]]:
        return sandbox_runner.list_jobs()

    @app.get("/api/sandbox/jobs/{job_id}")
    def sandbox_job(job_id: str) -> Dict[str, object]:
        try:
            return sandbox_runner.get(job_id).to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/sandbox/jobs/{job_id}/logs")
    def sandbox_logs(job_id: str) -> Dict[str, str]:
        try:
            return sandbox_runner.logs(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/sandbox/jobs/{job_id}/artifacts")
    def sandbox_artifacts(job_id: str) -> List[Dict[str, object]]:
        try:
            return sandbox_runner.artifacts(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/sandbox/jobs/{job_id}/artifacts/{name}")
    def download_sandbox_artifact(job_id: str, name: str) -> FileResponse:
        try:
            path = sandbox_runner.artifact_path(job_id, name)
            return FileResponse(path, media_type="application/json", filename=name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/sandbox/jobs/{job_id}/audit")
    def sandbox_audit(job_id: str) -> Dict[str, object]:
        try:
            return sandbox_runner.audit(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/sandbox/jobs/{job_id}/stop")
    def stop_sandbox_job(job_id: str) -> Dict[str, object]:
        try:
            return sandbox_runner.stop(job_id).to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/evaluations", status_code=201)
    def create_evaluation(payload: EvaluationCreate) -> Dict[str, Any]:
        run_ids = list(dict.fromkeys(payload.run_ids or ([payload.run_id] if payload.run_id else [])))
        if not run_ids:
            raise HTTPException(status_code=422, detail="至少选择一个运行")
        samples = []
        for run_id in run_ids:
            record = get_run_or_404(adapter, run_id)
            if record.status in (RunStatus.QUEUED, RunStatus.RUNNING):
                raise HTTPException(status_code=409, detail="运行结束后才能生成评估报告")
            samples.append((record, adapter.stream_events(run_id), adapter.list_artifacts(run_id)))
        try:
            return evaluation_service.evaluate_cohort(samples)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/evaluations")
    def list_evaluations() -> List[Dict[str, Any]]:
        return evaluation_service.list_reports()

    @app.post("/api/evaluations/compare")
    def compare_evaluations(payload: EvaluationCompare) -> Dict[str, Any]:
        try:
            return evaluation_service.compare_reports(payload.before_report_id, payload.after_report_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="评估报告不存在") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/evaluations/{report_id}")
    def get_evaluation(report_id: str) -> Dict[str, Any]:
        try:
            return evaluation_service.get_report(report_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="评估报告不存在") from exc

    return app


app = create_app()


def _load_course(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
