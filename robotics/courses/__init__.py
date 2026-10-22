"""Canonical teaching-resource domain."""

from .models import Course, Enrollment, TeachingClass
from .repository import (
    PostgreSQLCourseRepository,
    TeachingConflict,
    TeachingPermissionDenied,
    TeachingRepositoryError,
    TeachingResourceNotFound,
    TeachingValidationError,
)
from .service import CourseService

__all__ = [
    "Course",
    "CourseService",
    "Enrollment",
    "PostgreSQLCourseRepository",
    "TeachingClass",
    "TeachingConflict",
    "TeachingPermissionDenied",
    "TeachingRepositoryError",
    "TeachingResourceNotFound",
    "TeachingValidationError",
]
