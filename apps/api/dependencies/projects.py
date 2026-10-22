"""Composition helper for canonical teaching Projects."""

from typing import Callable, Optional

from sqlalchemy.orm import Session

from robotics.projects import (
    PostgreSQLTeachingProjectRepository,
    TeachingProjectService,
)


def build_teaching_project_service(
    session_factory: Optional[Callable[[], Session]],
) -> Optional[TeachingProjectService]:
    if session_factory is None:
        return None
    return TeachingProjectService(
        PostgreSQLTeachingProjectRepository(session_factory)
    )
