"""Canonical course, class, and enrollment HTTP API."""

from dataclasses import asdict
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from apps.api.schemas.courses import (
    CourseCreateRequest,
    CoursePatchRequest,
    CourseResponse,
    EnrollmentPutRequest,
    EnrollmentResponse,
    TeachingClassCreateRequest,
    TeachingClassPatchRequest,
    TeachingClassResponse,
)
from robotics.auth import User
from robotics.courses import (
    CourseService,
    TeachingConflict,
    TeachingPermissionDenied,
    TeachingRepositoryError,
    TeachingResourceNotFound,
    TeachingValidationError,
)


def create_courses_router(
    course_service: Optional[CourseService], current_user_provider: Any
) -> APIRouter:
    router = APIRouter(tags=["courses"])

    def service() -> CourseService:
        if course_service is None:
            raise HTTPException(status_code=503, detail="Teaching resources are unavailable")
        return course_service

    @router.post("/api/v1/courses", response_model=CourseResponse, status_code=201)
    def create_course(
        payload: CourseCreateRequest,
        user: User = Depends(current_user_provider),
    ) -> CourseResponse:
        owner = payload.owner_instructor_user_id or user.user_id
        try:
            value = service().create_course(
                payload.organization_id,
                payload.code,
                payload.title,
                payload.description,
                owner,
                user.user_id,
            )
            return CourseResponse(**asdict(value))
        except Exception as exc:
            raise _http_error(exc)

    @router.get("/api/v1/courses", response_model=List[CourseResponse])
    def list_courses(
        organization_id: str = Query(..., min_length=1, max_length=255),
        user: User = Depends(current_user_provider),
    ) -> List[CourseResponse]:
        try:
            return [
                CourseResponse(**asdict(value))
                for value in service().list_courses(organization_id, user.user_id)
            ]
        except Exception as exc:
            raise _http_error(exc)

    @router.get("/api/v1/courses/{course_id}", response_model=CourseResponse)
    def get_course(
        course_id: str, user: User = Depends(current_user_provider)
    ) -> CourseResponse:
        try:
            return CourseResponse(
                **asdict(service().get_course(course_id, user.user_id))
            )
        except Exception as exc:
            raise _http_error(exc)

    @router.patch("/api/v1/courses/{course_id}", response_model=CourseResponse)
    def patch_course(
        course_id: str,
        payload: CoursePatchRequest,
        user: User = Depends(current_user_provider),
    ) -> CourseResponse:
        changes = payload.model_dump(exclude_unset=True)
        if not changes:
            raise HTTPException(status_code=422, detail="At least one field is required")
        try:
            return CourseResponse(
                **asdict(
                    service().update_course(course_id, user.user_id, changes)
                )
            )
        except Exception as exc:
            raise _http_error(exc)

    @router.post(
        "/api/v1/courses/{course_id}/classes",
        response_model=TeachingClassResponse,
        status_code=201,
    )
    def create_class(
        course_id: str,
        payload: TeachingClassCreateRequest,
        user: User = Depends(current_user_provider),
    ) -> TeachingClassResponse:
        try:
            return TeachingClassResponse(
                **asdict(
                    service().create_class(
                        course_id,
                        payload.code,
                        payload.name,
                        payload.term,
                        payload.instructor_user_id,
                        user.user_id,
                    )
                )
            )
        except Exception as exc:
            raise _http_error(exc)

    @router.get(
        "/api/v1/courses/{course_id}/classes",
        response_model=List[TeachingClassResponse],
    )
    def list_classes(
        course_id: str, user: User = Depends(current_user_provider)
    ) -> List[TeachingClassResponse]:
        try:
            return [
                TeachingClassResponse(**asdict(value))
                for value in service().list_classes(course_id, user.user_id)
            ]
        except Exception as exc:
            raise _http_error(exc)

    @router.get("/api/v1/classes/{class_id}", response_model=TeachingClassResponse)
    def get_class(
        class_id: str, user: User = Depends(current_user_provider)
    ) -> TeachingClassResponse:
        try:
            return TeachingClassResponse(
                **asdict(service().get_class(class_id, user.user_id))
            )
        except Exception as exc:
            raise _http_error(exc)

    @router.patch("/api/v1/classes/{class_id}", response_model=TeachingClassResponse)
    def patch_class(
        class_id: str,
        payload: TeachingClassPatchRequest,
        user: User = Depends(current_user_provider),
    ) -> TeachingClassResponse:
        changes = payload.model_dump(exclude_unset=True)
        if not changes:
            raise HTTPException(status_code=422, detail="At least one field is required")
        try:
            return TeachingClassResponse(
                **asdict(service().update_class(class_id, user.user_id, changes))
            )
        except Exception as exc:
            raise _http_error(exc)

    @router.get(
        "/api/v1/classes/{class_id}/enrollments",
        response_model=List[EnrollmentResponse],
    )
    def list_enrollments(
        class_id: str, user: User = Depends(current_user_provider)
    ) -> List[EnrollmentResponse]:
        try:
            return [
                EnrollmentResponse(**asdict(value))
                for value in service().list_enrollments(class_id, user.user_id)
            ]
        except Exception as exc:
            raise _http_error(exc)

    @router.put(
        "/api/v1/classes/{class_id}/enrollments/{user_id}",
        response_model=EnrollmentResponse,
    )
    def put_enrollment(
        class_id: str,
        user_id: str,
        payload: EnrollmentPutRequest,
        user: User = Depends(current_user_provider),
    ) -> EnrollmentResponse:
        try:
            return EnrollmentResponse(
                **asdict(
                    service().put_enrollment(
                        class_id, user_id, payload.enabled, user.user_id
                    )
                )
            )
        except Exception as exc:
            raise _http_error(exc)

    return router


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, TeachingResourceNotFound):
        return HTTPException(status_code=404, detail="Teaching resource not found")
    if isinstance(exc, TeachingPermissionDenied):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, TeachingConflict):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, (TeachingValidationError, ValueError)):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, TeachingRepositoryError):
        return HTTPException(status_code=503, detail="Teaching resources are unavailable")
    raise exc
