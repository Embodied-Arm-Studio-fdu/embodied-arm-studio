"""Organization application service and validation boundary."""

import re
from typing import List, Tuple
from uuid import uuid4

from .models import (
    Organization,
    OrganizationContext,
    OrganizationMembership,
    OrganizationRole,
    utc_now,
)
from .repository import (
    MembershipNotFound,
    OrganizationConflict,
    PostgreSQLOrganizationRepository,
)


_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class OrganizationService:
    def __init__(self, repository: PostgreSQLOrganizationRepository) -> None:
        self.repository = repository

    def create_organization(
        self, slug: str, name: str, administrator_user_id: str
    ) -> Tuple[Organization, OrganizationMembership]:
        normalized_slug = _normalize_slug(slug)
        normalized_name = _normalize_name(name)
        existing = self.repository.find_by_slug(normalized_slug)
        if existing is not None:
            try:
                membership = self.repository.get_membership(
                    existing.organization_id, administrator_user_id
                )
            except MembershipNotFound as exc:
                raise OrganizationConflict(
                    "slug exists with a different administrator"
                ) from exc
            if (
                existing.name == normalized_name
                and existing.enabled
                and membership.enabled
                and membership.role is OrganizationRole.ADMINISTRATOR
            ):
                return existing, membership
            raise OrganizationConflict(
                "organization exists with different facts"
            )
        now = utc_now()
        organization = Organization(
            organization_id=str(uuid4()),
            slug=normalized_slug,
            name=normalized_name,
            enabled=True,
            created_at=now,
            updated_at=now,
        )
        membership = OrganizationMembership(
            membership_id=str(uuid4()),
            organization_id=organization.organization_id,
            user_id=administrator_user_id,
            role=OrganizationRole.ADMINISTRATOR,
            enabled=True,
            created_at=now,
            updated_at=now,
        )
        return self.repository.create_with_administrator(
            organization, membership
        )

    def get_by_id(self, organization_id: str) -> Organization:
        return self.repository.get_by_id(organization_id)

    def get_by_slug(self, slug: str) -> Organization:
        return self.repository.get_by_slug(_normalize_slug(slug))

    def list_enabled_for_user(
        self, user_id: str
    ) -> List[OrganizationContext]:
        return self.repository.list_enabled_for_user(user_id)

    def get_enabled_context(
        self, organization_id: str, user_id: str
    ) -> OrganizationContext:
        return self.repository.get_enabled_context(organization_id, user_id)

    def get_membership(
        self, organization_id: str, user_id: str
    ) -> OrganizationMembership:
        return self.repository.get_membership(organization_id, user_id)

    def add_membership(
        self,
        organization_id: str,
        user_id: str,
        role: OrganizationRole,
    ) -> OrganizationMembership:
        now = utc_now()
        return self.repository.add_membership(
            OrganizationMembership(
                membership_id=str(uuid4()),
                organization_id=organization_id,
                user_id=user_id,
                role=OrganizationRole(role),
                enabled=True,
                created_at=now,
                updated_at=now,
            )
        )

    def set_membership(
        self,
        organization_id: str,
        user_id: str,
        role: OrganizationRole,
        enabled: bool,
    ) -> OrganizationMembership:
        return self.repository.set_membership(
            organization_id,
            user_id,
            OrganizationRole(role),
            bool(enabled),
        )


def _normalize_slug(slug: str) -> str:
    if not isinstance(slug, str):
        raise ValueError("slug must be a string")
    value = slug.strip().lower()
    if len(value) > 255 or _SLUG.fullmatch(value) is None:
        raise ValueError(
            "slug must contain lowercase letters, digits, and single hyphens"
        )
    return value


def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("name must be a string")
    value = name.strip()
    if not value or len(value) > 255:
        raise ValueError("name must contain 1-255 characters")
    return value
