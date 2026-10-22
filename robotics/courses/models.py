"""Canonical course, teaching-class, and enrollment domain models."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("%s must be timezone-aware" % name)
    if value.utcoffset() != timedelta(0):
        raise ValueError("%s must use UTC" % name)


@dataclass(frozen=True)
class Course:
    course_id: str
    organization_id: str
    code: str
    title: str
    description: str
    owner_instructor_user_id: str
    enabled: bool
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        _require_aware(self.created_at, "created_at")
        _require_aware(self.updated_at, "updated_at")


@dataclass(frozen=True)
class TeachingClass:
    class_id: str
    course_id: str
    code: str
    name: str
    term: str
    instructor_user_id: str
    enabled: bool
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        _require_aware(self.created_at, "created_at")
        _require_aware(self.updated_at, "updated_at")


@dataclass(frozen=True)
class Enrollment:
    enrollment_id: str
    class_id: str
    user_id: str
    enabled: bool
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        _require_aware(self.created_at, "created_at")
        _require_aware(self.updated_at, "updated_at")
