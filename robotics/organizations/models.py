"""Organization and membership domain models."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import FrozenSet


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class OrganizationRole(str, Enum):
    STUDENT = "student"
    INSTRUCTOR = "instructor"
    ADMINISTRATOR = "administrator"


STUDENT_PERMISSIONS = frozenset(
    {
        "organization:read",
        "course:read",
        "class:read",
        "own_project:read",
        "own_run:read",
        "own_run:create",
    }
)
INSTRUCTOR_PERMISSIONS = STUDENT_PERMISSIONS | frozenset(
    {
        "course:manage",
        "class:manage",
        "enrollment:read",
        "enrollment:manage",
        "project_template:manage",
        "assigned_class_run:read",
    }
)
ADMINISTRATOR_PERMISSIONS = INSTRUCTOR_PERMISSIONS | frozenset(
    {
        "organization_user:manage",
        "platform_teaching_data:manage",
    }
)

ROLE_PERMISSIONS = {
    OrganizationRole.STUDENT: STUDENT_PERMISSIONS,
    OrganizationRole.INSTRUCTOR: INSTRUCTOR_PERMISSIONS,
    OrganizationRole.ADMINISTRATOR: ADMINISTRATOR_PERMISSIONS,
}


def permissions_for(role: OrganizationRole) -> FrozenSet[str]:
    """Return authorization capability identifiers, not implemented routes."""

    return ROLE_PERMISSIONS[OrganizationRole(role)]


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("%s must be timezone-aware" % name)


@dataclass(frozen=True)
class Organization:
    organization_id: str
    slug: str
    name: str
    enabled: bool
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        _require_aware(self.created_at, "created_at")
        _require_aware(self.updated_at, "updated_at")


@dataclass(frozen=True)
class OrganizationMembership:
    membership_id: str
    organization_id: str
    user_id: str
    role: OrganizationRole
    enabled: bool
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", OrganizationRole(self.role))
        _require_aware(self.created_at, "created_at")
        _require_aware(self.updated_at, "updated_at")


@dataclass(frozen=True)
class OrganizationContext:
    organization: Organization
    membership: OrganizationMembership
