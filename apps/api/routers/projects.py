"""Canonical Course-nested teaching Project HTTP API."""

from copy import deepcopy
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException

from apps.api.schemas.projects import (
    TeachingProjectCreateRequest,
    TeachingProjectPatchRequest,
    TeachingProjectPermissionsResponse,
    TeachingProjectResponse,
)
from robotics.auth import User
from robotics.projects import (
    ALGORITHM,
    ENVIRONMENT_TYPE,
    TASK_TYPE,
    TeachingProjectAccess,
    TeachingProjectConflict,
    TeachingProjectNotFound,
    TeachingProjectPermissionDenied,
    TeachingProjectRepositoryError,
    TeachingProjectService,
    TeachingProjectVersionConflict,
    success_condition,
)


def create_teaching_projects_router(
    project_service: Optional[TeachingProjectService],
    current_user_provider: Any,
) -> APIRouter:
    router = APIRouter(tags=["teaching-projects"])

    def service() -> TeachingProjectService:
        if project_service is None:
            raise HTTPException(
                status_code=503, detail="Teaching projects are unavailable"
            )
        return project_service

    @router.get(
        "/api/v1/courses/{course_id}/projects",
        response_model=List[TeachingProjectResponse],
    )
    def list_projects(
        course_id: str, user: User = Depends(current_user_provider)
    ) -> List[TeachingProjectResponse]:
        try:
            return [
                _response(value)
                for value in service().list_projects(course_id, user.user_id)
            ]
        except Exception as exc:
            raise _http_error(exc)

    @router.post(
        "/api/v1/courses/{course_id}/projects",
        response_model=TeachingProjectResponse,
        status_code=201,
    )
    def create_project(
        course_id: str,
        payload: TeachingProjectCreateRequest,
        user: User = Depends(current_user_provider),
    ) -> TeachingProjectResponse:
        try:
            return _response(
                service().create_project(
                    course_id,
                    user.user_id,
                    **payload.model_dump(),
                )
            )
        except Exception as exc:
            raise _http_error(exc)

    @router.get(
        "/api/v1/courses/{course_id}/projects/{project_id}",
        response_model=TeachingProjectResponse,
    )
    def get_project(
        course_id: str,
        project_id: str,
        user: User = Depends(current_user_provider),
    ) -> TeachingProjectResponse:
        try:
            return _response(
                service().get_project(course_id, project_id, user.user_id)
            )
        except Exception as exc:
            raise _http_error(exc)

    @router.patch(
        "/api/v1/courses/{course_id}/projects/{project_id}",
        response_model=TeachingProjectResponse,
    )
    def patch_project(
        course_id: str,
        project_id: str,
        payload: TeachingProjectPatchRequest,
        user: User = Depends(current_user_provider),
    ) -> TeachingProjectResponse:
        changes = payload.model_dump(
            exclude_unset=True, exclude={"config_version"}
        )
        if not changes:
            raise HTTPException(
                status_code=422,
                detail="At least one Project field is required",
            )
        try:
            return _response(
                service().update_project(
                    course_id,
                    project_id,
                    user.user_id,
                    payload.config_version,
                    changes,
                )
            )
        except Exception as exc:
            raise _http_error(exc)

    return router


def _response(value: TeachingProjectAccess) -> TeachingProjectResponse:
    project = value.project
    return TeachingProjectResponse(
        project_id=project.project_id,
        course_id=project.course_id,
        title=project.title,
        description=project.description,
        instructions=project.instructions,
        starter_code=project.starter_code,
        starter_code_sha256=project.starter_code_sha256,
        task_type=TASK_TYPE,
        environment_type=project.environment_type,
        algorithm=ALGORITHM,
        success_condition=success_condition(project.environment_type),
        training_profile=project.training_profile,
        expected_training_steps=project.expected_training_steps,
        video_enabled=project.video_enabled,
        parameter_schema=deepcopy(project.parameter_schema),
        evaluation_rubric=deepcopy(project.evaluation_rubric),
        config_version=project.config_version,
        enabled=project.enabled,
        created_by_user_id=project.created_by_user_id,
        created_at=project.created_at,
        updated_at=project.updated_at,
        permissions=TeachingProjectPermissionsResponse(
            can_view=value.permissions.can_view,
            can_use=value.permissions.can_use,
            can_modify=value.permissions.can_modify,
        ),
    )


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, TeachingProjectNotFound):
        return HTTPException(status_code=404, detail="Teaching project not found")
    if isinstance(exc, TeachingProjectPermissionDenied):
        return HTTPException(status_code=403, detail="Project operation is not allowed")
    if isinstance(exc, TeachingProjectVersionConflict):
        return HTTPException(
            status_code=409,
            detail={
                "code": "project_version_conflict",
                "message": "Project config version does not match",
            },
        )
    if isinstance(exc, TeachingProjectConflict):
        return HTTPException(
            status_code=409,
            detail={
                "code": "project_conflict",
                "message": "Teaching project conflicts with existing data",
            },
        )
    if isinstance(exc, (ValueError, UnicodeError)):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, TeachingProjectRepositoryError):
        return HTTPException(
            status_code=503, detail="Teaching projects are unavailable"
        )
    raise exc

