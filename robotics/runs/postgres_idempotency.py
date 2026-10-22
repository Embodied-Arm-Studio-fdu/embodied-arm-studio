"""Atomic PostgreSQL implementations of idempotent canonical Run creation."""

from copy import deepcopy
from typing import Any, Callable, Mapping, Optional

from sqlalchemy import exists, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from robotics.organizations import OrganizationRole
from robotics.projects.models import (
    ENVIRONMENT_TYPE,
    SUPPORTED_ENVIRONMENT_TYPES,
    TrainingProfile,
    worker_config,
)
from robotics.runs.db_models import (
    CourseRow,
    EnrollmentRow,
    OrganizationMembershipRow,
    OrganizationRow,
    RunIdempotencyKeyRow,
    RunRow,
    TeachingClassRow,
    TeachingProjectRow,
    UserRow,
)
from robotics.runs.idempotency import (
    IdempotencyConflict,
    IdempotentRunCreateCommand,
    IdempotentRunCreateResult,
    InvalidIdempotentRunCreateRequest,
    fingerprint_run_create,
)
from robotics.runs.postgres_repository import (
    PostgreSQLRunRepository,
    run_from_row,
    run_to_row,
)
from robotics.runs.repository import RepositoryDataError, RepositoryIntegrityError
from robotics.runs.service import RunService
from robotics.runs.models import ExecutionMode, RunKind
from robotics.scenes import SceneRepository, scene_preflight


class ProjectRunCreationNotFound(LookupError):
    """A teaching resource is unknown, disabled, or outside the actor's organization."""


class ProjectRunCreationPermissionDenied(PermissionError):
    """The actor can see the resource context but cannot create this Run."""


class ProjectRunVersionConflict(ValueError):
    """The requested Project version is no longer current."""


class SceneRunVersionConflict(ValueError):
    """The requested scene revision is no longer current."""


def _project_environment_type(project) -> str:
    value = getattr(project, "environment_type", None)
    if value is None:
        schema = getattr(project, "parameter_schema", None)
        if not isinstance(schema, Mapping):
            raise InvalidIdempotentRunCreateRequest(
                "Project parameter_schema is invalid"
            )
        value = schema.get("x-environment-type", ENVIRONMENT_TYPE)
    if value not in SUPPORTED_ENVIRONMENT_TYPES:
        raise InvalidIdempotentRunCreateRequest(
            "Project environment_type is unsupported"
        )
    return value


def validate_project_task_compatibility(project, run_kind, environment_type):
    """Reject cross-task Project use before any Run row is inserted."""
    project_environment = _project_environment_type(project)
    requested = environment_type or project_environment
    if run_kind is RunKind.TRAIN and requested != project_environment:
        raise InvalidIdempotentRunCreateRequest(
            "Project environment_type=%s is incompatible with task=%s"
            % (project_environment, requested)
        )
    if run_kind is RunKind.PLAY:
        if TrainingProfile(project.training_profile) is not TrainingProfile.SHORT:
            raise InvalidIdempotentRunCreateRequest(
                "Only the Short Project accepts Demo tasks"
            )
        return requested
    if run_kind is not RunKind.TRAIN:
        raise InvalidIdempotentRunCreateRequest(
            "Training Projects only accept training tasks"
        )
    return requested


class PostgreSQLIdempotentRunCreationService:
    """Persist a Run and its idempotency key in one SQLAlchemy transaction."""

    def __init__(
        self,
        run_service: RunService,
        session_factory: Optional[Callable[[], Session]] = None,
        scene_repository: Optional[SceneRepository] = None,
    ) -> None:
        if not isinstance(run_service, RunService):
            raise TypeError("run_service must be a RunService")
        repository = run_service.repository
        if not isinstance(repository, PostgreSQLRunRepository):
            raise TypeError("run_service must use PostgreSQLRunRepository")
        resolved_factory = repository.session_factory
        if session_factory is not None and session_factory is not resolved_factory:
            raise ValueError(
                "session_factory must be the RunService repository factory"
            )
        self._run_service = run_service
        self._session_factory = resolved_factory
        self._scene_repository = scene_repository

    def create_run(
        self, command: IdempotentRunCreateCommand
    ) -> IdempotentRunCreateResult:
        """Create once, replay matching requests, and reject changed requests."""

        if not isinstance(command, IdempotentRunCreateCommand):
            raise TypeError("command must be an IdempotentRunCreateCommand")
        request_hash = fingerprint_run_create(command)
        session = self._session_factory()
        try:
            existing = self._find_key(session, command)
            if existing is not None:
                result = self._resolve_existing(
                    session, existing, command, request_hash
                )
                session.rollback()
                return result

            run = self._run_service.build_run(**command.create_kwargs())
            run_row = run_to_row(run)
            session.add(run_row)
            session.flush()
            session.add(
                RunIdempotencyKeyRow(
                    user_id=command.user_id,
                    idempotency_key=command.idempotency_key,
                    request_hash=request_hash,
                    run_id=run.run_id,
                    created_at=run.created_at,
                )
            )
            session.flush()
            persisted = run_from_row(run_row)
            session.commit()
            return IdempotentRunCreateResult(run=persisted, replayed=False)
        except IntegrityError as exc:
            session.rollback()
            existing = self._find_key(session, command)
            if existing is None:
                session.rollback()
                raise RepositoryIntegrityError(
                    "idempotent_create_run"
                ) from exc
            try:
                result = self._resolve_existing(
                    session, existing, command, request_hash
                )
            finally:
                session.rollback()
            return result
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _find_key(
        session: Session, command: IdempotentRunCreateCommand
    ) -> Optional[RunIdempotencyKeyRow]:
        return session.execute(
            select(RunIdempotencyKeyRow).where(
                RunIdempotencyKeyRow.user_id == command.user_id,
                RunIdempotencyKeyRow.idempotency_key
                == command.idempotency_key,
            )
        ).scalar_one_or_none()

    @staticmethod
    def _resolve_existing(
        session: Session,
        existing: RunIdempotencyKeyRow,
        command: IdempotentRunCreateCommand,
        request_hash: str,
    ) -> IdempotentRunCreateResult:
        if existing.request_hash != request_hash:
            raise IdempotencyConflict(
                command.user_id, command.idempotency_key
            )
        run_row = session.get(RunRow, existing.run_id)
        if run_row is None:
            raise RepositoryDataError(
                "Idempotency record references a missing Run"
            )
        return IdempotentRunCreateResult(
            run=run_from_row(run_row), replayed=True
        )


class PostgreSQLProjectRunCreationService(
    PostgreSQLIdempotentRunCreationService
):
    """Create formal teaching Runs from locked, authoritative Project facts."""

    def configure_simulation_catalog_client(self, client: Any) -> None:
        if not callable(getattr(client, "get_task_catalog", None)):
            raise TypeError("catalog client must implement get_task_catalog")
        self._catalog_client = client

    def create_run(
        self, command: IdempotentRunCreateCommand
    ) -> IdempotentRunCreateResult:
        try:
            self._require_formal_command(command)
        except ValueError as exc:
            if isinstance(exc, InvalidIdempotentRunCreateRequest):
                raise
            raise InvalidIdempotentRunCreateRequest(str(exc)) from exc
        catalog = None
        if command.environment_type is not None:
            client = getattr(self, "_catalog_client", None)
            if client is None:
                from robotics.simulation.client import SimulationConfigurationError

                raise SimulationConfigurationError(
                    "Simulation task catalog is not configured"
                )
            catalog = client.get_task_catalog()
        request_hash = fingerprint_run_create(command)
        session = self._session_factory()
        try:
            existing = self._find_key(session, command)
            if existing is not None:
                result = self._resolve_existing(
                    session, existing, command, request_hash
                )
                session.rollback()
                return result

            project, teaching_class = self._authorize(
                session,
                command.user_id,
                command.project_id,
                command.class_id,
                expected_version=None,
                lock_project=True,
            )
            # Another identical creator may have committed while this
            # transaction waited for the Project row lock.  Re-check the
            # idempotency scope before consulting mutable Project facts so a
            # later Project PATCH cannot turn a valid replay into a stale
            # version conflict.
            existing = self._find_key(session, command)
            if existing is not None:
                result = self._resolve_existing(
                    session, existing, command, request_hash
                )
                session.rollback()
                return result
            if project.config_version != command.project_config_version:
                raise ProjectRunVersionConflict("project_version_conflict")
            requested_environment = validate_project_task_compatibility(
                project, command.run_kind, command.environment_type
            )
            if command.environment_type is None:
                seed = command.parameters["seed"]
                training_config = worker_config(
                    TrainingProfile(project.training_profile),
                    project.video_enabled,
                    seed,
                    config_version=project.config_version,
                    environment_type=requested_environment,
                )
                scene_snapshot = self._scene_snapshot(command)
                snapshot = self._snapshot(
                    project, teaching_class, seed, training_config, scene_snapshot
                )
                run_parameters = {"seed": seed}
            else:
                try:
                    run_parameters, snapshot = self._catalog_snapshot(
                        command, catalog, project, teaching_class
                    )
                except SceneRunVersionConflict:
                    raise
                except ValueError as exc:
                    raise InvalidIdempotentRunCreateRequest(str(exc)) from exc
            if command.reward_extension is not None:
                from robotics.experiment_workbench import preflight
                result = preflight(requested_environment, command.reward_extension["source"])
                if not result["valid"]:
                    raise InvalidIdempotentRunCreateRequest(result["errors"][0]["message"])
                snapshot["reward_extension"] = deepcopy(command.reward_extension)
            run = self._run_service.build_run(
                user_id=command.user_id,
                project_id=command.project_id,
                class_id=command.class_id,
                execution_mode=command.execution_mode,
                parameters=run_parameters,
                project_config_snapshot=snapshot,
                code=command.code,
                run_kind=command.run_kind,
            )
            run_row = run_to_row(run)
            session.add(run_row)
            session.flush()
            session.add(
                RunIdempotencyKeyRow(
                    user_id=command.user_id,
                    idempotency_key=command.idempotency_key,
                    request_hash=request_hash,
                    run_id=run.run_id,
                    created_at=run.created_at,
                )
            )
            session.flush()
            persisted = run_from_row(run_row)
            session.commit()
            return IdempotentRunCreateResult(run=persisted, replayed=False)
        except IntegrityError as exc:
            session.rollback()
            existing = self._find_key(session, command)
            if existing is None:
                session.rollback()
                raise RepositoryIntegrityError(
                    "project_idempotent_create_run"
                ) from exc
            try:
                result = self._resolve_existing(
                    session, existing, command, request_hash
                )
            finally:
                session.rollback()
            return result
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    def authorize_dispatch(self, run) -> dict:
        """Recheck current authority before an unbound queued Run causes I/O."""

        if run.class_id is None:
            raise ProjectRunCreationNotFound("teaching Run context not found")
        session = self._session_factory()
        try:
            project, teaching_class = self._authorize(
                session,
                run.user_id,
                run.project_id,
                run.class_id,
                expected_version=None,
                lock_project=False,
            )
            return {
                "project_id": project.project_id,
                "course_id": project.course_id,
                "class_id": teaching_class.class_id,
            }
        finally:
            session.rollback()
            session.close()

    @staticmethod
    def _require_formal_command(command: IdempotentRunCreateCommand) -> None:
        if not isinstance(command, IdempotentRunCreateCommand):
            raise TypeError("command must be an IdempotentRunCreateCommand")
        if command.class_id is None or command.project_config_version is None:
            raise ValueError("class_id and project_config_version are required")
        if command.execution_mode is not ExecutionMode.SIMULATION:
            raise ValueError("execution_mode must be simulation")
        if command.run_kind not in {RunKind.TRAIN, RunKind.PLAY}:
            raise ValueError("run_kind must be train or play")
        if command.environment_type is None:
            seed = command.parameters.get("seed")
            if command.run_kind is not RunKind.TRAIN:
                raise ValueError("Demo Runs require environment_type")
            if (
                set(command.parameters) != {"seed"}
                or isinstance(seed, bool)
                or not isinstance(seed, int)
                or not 0 <= seed <= 2147483647
            ):
                raise ValueError("parameters must contain only seed in range")
        elif command.run_kind is RunKind.PLAY and command.scene_ref is not None:
            raise ValueError("Demo Runs do not accept scene_ref")

    @staticmethod
    def _authorize(
        session: Session,
        user_id: str,
        project_id: str,
        class_id: str,
        *,
        expected_version: Optional[int],
        lock_project: bool,
    ):
        project_query = select(TeachingProjectRow).where(
            TeachingProjectRow.project_id == project_id
        )
        if lock_project:
            project_query = project_query.with_for_update()
        project = session.execute(project_query).scalar_one_or_none()
        if project is None or not project.enabled:
            raise ProjectRunCreationNotFound("teaching resource not found")
        result = session.execute(
            select(TeachingClassRow, CourseRow, OrganizationRow)
            .join(CourseRow, CourseRow.course_id == TeachingClassRow.course_id)
            .join(
                OrganizationRow,
                OrganizationRow.organization_id == CourseRow.organization_id,
            )
            .where(TeachingClassRow.class_id == class_id)
        ).one_or_none()
        if result is None:
            raise ProjectRunCreationNotFound("teaching resource not found")
        teaching_class, course, organization = result
        if (
            not teaching_class.enabled
            or not course.enabled
            or not organization.enabled
            or teaching_class.course_id != project.course_id
        ):
            raise ProjectRunCreationNotFound("teaching resource not found")
        actor = session.execute(
            select(UserRow, OrganizationMembershipRow)
            .join(
                OrganizationMembershipRow,
                OrganizationMembershipRow.user_id == UserRow.user_id,
            )
            .where(
                UserRow.user_id == user_id,
                OrganizationMembershipRow.organization_id
                == course.organization_id,
            )
        ).one_or_none()
        if actor is None:
            raise ProjectRunCreationNotFound("teaching resource not found")
        user, membership = actor
        if not user.enabled or not membership.enabled:
            raise ProjectRunCreationNotFound("teaching resource not found")
        role = OrganizationRole(membership.role)
        allowed = role is OrganizationRole.ADMINISTRATOR
        if role is OrganizationRole.INSTRUCTOR:
            allowed = (
                course.owner_instructor_user_id == user_id
                or teaching_class.instructor_user_id == user_id
            )
        elif role is OrganizationRole.STUDENT:
            allowed = bool(
                session.execute(
                    select(
                        exists().where(
                            EnrollmentRow.class_id == class_id,
                            EnrollmentRow.user_id == user_id,
                            EnrollmentRow.enabled.is_(True),
                        )
                    )
                ).scalar()
            )
        if not allowed:
            raise ProjectRunCreationPermissionDenied(
                "teaching Run creation is not allowed"
            )
        if expected_version is not None and project.config_version != expected_version:
            raise ProjectRunVersionConflict("project_version_conflict")
        return project, teaching_class

    def _scene_snapshot(self, command):
        if command.scene_ref is None:
            return None
        if self._scene_repository is None:
            raise ProjectRunCreationNotFound("scene resource not found")
        try:
            scene = self._scene_repository.get(command.scene_ref["scene_id"])
        except (KeyError, ValueError) as exc:
            raise ProjectRunCreationNotFound("scene resource not found") from exc
        if scene.get("revision") != command.scene_ref["revision"]:
            raise SceneRunVersionConflict("scene_revision_conflict")
        if not scene_preflight(scene)["trainable_push"]:
            raise InvalidIdempotentRunCreateRequest("scene_preflight_failed")
        return {
            "scene_ref": deepcopy(command.scene_ref),
            "document": deepcopy(scene),
        }

    def _catalog_snapshot(
        self,
        command: IdempotentRunCreateCommand,
        catalog,
        project,
        teaching_class,
    ):
        from robotics.simulation.client import SimulationConfigurationError
        from robotics.simulation.contracts import (
            SimulationTaskCatalog,
            resolve_catalog_task_parameters,
        )

        if not isinstance(catalog, SimulationTaskCatalog):
            raise SimulationConfigurationError(
                "Simulation task catalog did not satisfy the parsed contract"
            )
        expected_type = (
            "training" if command.run_kind is RunKind.TRAIN else "demo"
        )
        requested_environment = (
            command.environment_type or _project_environment_type(project)
        )
        baseline = {}
        if command.run_kind is RunKind.TRAIN:
            seed = command.parameters.get("seed", 7)
            profile_values = worker_config(
                TrainingProfile(project.training_profile),
                True,
                seed,
                config_version=project.config_version,
                environment_type=requested_environment,
            )
            baseline = {
                name: value
                for name, value in profile_values.items()
                if name not in {"task_type", "environment_type", "device"}
            }
            baseline.setdefault("run_mode", "smoke")
        task, resolved = resolve_catalog_task_parameters(
            catalog,
            requested_environment,
            expected_type,
            command.parameters,
            baseline=baseline,
        )
        task_config = {
            "task_type": expected_type,
            "environment_type": task.id,
            **deepcopy(resolved),
        }
        if command.run_kind is RunKind.TRAIN:
            task_config["device"] = "cuda:0"
        scene_snapshot = None
        if command.scene_ref is not None:
            if task.id != "s1_push":
                raise ValueError("scene_ref is only supported by s1_push")
            scene_snapshot = self._scene_snapshot(command)
        snapshot = {
            "schema_version": "1.0",
            "project_id": project.project_id,
            "course_id": project.course_id,
            "class_id": teaching_class.class_id,
            "title": project.title,
            "description": project.description,
            "instructions": project.instructions,
            "task_type": expected_type,
            "environment_type": task.id,
            "catalog_task": task.to_dict(),
            "resolved_parameters": deepcopy(resolved),
            "task_config": deepcopy(task_config),
            "config_version": project.config_version,
            "project_enabled_at_creation": project.enabled,
            "starter_code_sha256": project.starter_code_sha256,
        }
        if command.run_kind is RunKind.TRAIN:
            snapshot.update(
                {
                    "algorithm": "PPO",
                    "training_profile": project.training_profile,
                    "expected_training_steps": (
                        resolved["rollout_steps"] * resolved["num_updates"]
                    ),
                    "evaluation_rubric": self._training_rubric(resolved),
                    "scene": scene_snapshot,
                    "video_enabled": resolved.get("video_enabled", False),
                }
            )
        return deepcopy(resolved), deepcopy(snapshot)

    @staticmethod
    def _training_rubric(parameters: Mapping[str, Any]):
        return {
            "version": "catalog-training-evidence-v1",
            "criteria": [
                {
                    "criterion_id": "terminal_completed",
                    "evidence_path": "run.status",
                    "operator": "eq",
                    "target": "completed",
                    "weight": 40,
                },
                {
                    "criterion_id": "training_steps",
                    "evidence_path": "run.training_steps",
                    "operator": "gte",
                    "target": parameters["rollout_steps"] * parameters["num_updates"],
                    "weight": 40,
                },
                {
                    "criterion_id": "update_steps",
                    "evidence_path": "run.update_steps",
                    "operator": "gte",
                    "target": parameters["num_updates"],
                    "weight": 20,
                },
            ],
            "grade_bands": [
                {"minimum_score": 100, "grade": "complete"},
                {"minimum_score": 0, "grade": "incomplete"},
            ],
        }

    @staticmethod
    def _snapshot(
        project, teaching_class, seed, training_config, scene_snapshot
    ):
        return deepcopy(
            {
                "schema_version": "1.0",
                "project_id": project.project_id,
                "course_id": project.course_id,
                "class_id": teaching_class.class_id,
                "title": project.title,
                "description": project.description,
                "instructions": project.instructions,
                "task_type": "training",
                "environment_type": "s1_push",
                "algorithm": "PPO",
                "success_condition": {
                    "metric": "object_to_goal_distance",
                    "operator": "lte",
                    "value": 0.05,
                    "unit": "m",
                },
                "training_profile": project.training_profile,
                "expected_training_steps": (
                    training_config["rollout_steps"]
                    * training_config["num_updates"]
                ),
                "video_enabled": project.video_enabled,
                "parameter_schema": deepcopy(project.parameter_schema),
                "resolved_parameters": {"seed": seed},
                "evaluation_rubric": deepcopy(project.evaluation_rubric),
                "config_version": project.config_version,
                "enabled": project.enabled,
                "project_enabled_at_creation": project.enabled,
                "starter_code_sha256": project.starter_code_sha256,
                "training_config": deepcopy(training_config),
                "scene": scene_snapshot,
            }
        )
