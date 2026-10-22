"""Read-only browser organization and current-context routes."""

from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException

from apps.api.schemas.organizations import (
    CurrentUserContextResponse,
    OrganizationContextResponse,
)
from robotics.auth import User
from robotics.organizations import (
    OrganizationContext,
    OrganizationNotFound,
    OrganizationRepositoryError,
    OrganizationService,
    permissions_for,
)


def create_organizations_router(
    organization_service: Optional[OrganizationService],
    current_user_provider: Any,
) -> APIRouter:
    router = APIRouter(tags=["organizations"])

    def contexts_for(user_id: str) -> List[OrganizationContext]:
        if organization_service is None:
            return []
        try:
            return organization_service.list_enabled_for_user(user_id)
        except OrganizationRepositoryError as exc:
            raise HTTPException(
                status_code=503, detail="Organization context is unavailable"
            ) from exc

    @router.get("/api/v1/me", response_model=CurrentUserContextResponse)
    def me(
        user: User = Depends(current_user_provider),
    ) -> CurrentUserContextResponse:
        public = user.public_dict()
        public["organizations"] = [
            _context_response(context) for context in contexts_for(user.user_id)
        ]
        return CurrentUserContextResponse.model_validate(public)

    @router.get(
        "/api/v1/organizations",
        response_model=List[OrganizationContextResponse],
    )
    def list_organizations(
        user: User = Depends(current_user_provider),
    ) -> List[OrganizationContextResponse]:
        return [
            _context_response(context) for context in contexts_for(user.user_id)
        ]

    @router.get(
        "/api/v1/organizations/{organization_id}",
        response_model=OrganizationContextResponse,
    )
    def get_organization(
        organization_id: str,
        user: User = Depends(current_user_provider),
    ) -> OrganizationContextResponse:
        if organization_service is None:
            raise _not_found()
        try:
            context = organization_service.get_enabled_context(
                organization_id, user.user_id
            )
        except OrganizationNotFound:
            raise _not_found()
        except OrganizationRepositoryError as exc:
            raise HTTPException(
                status_code=503, detail="Organization context is unavailable"
            ) from exc
        return _context_response(context)

    return router


def _context_response(
    context: OrganizationContext,
) -> OrganizationContextResponse:
    organization = context.organization
    return OrganizationContextResponse(
        organization_id=organization.organization_id,
        slug=organization.slug,
        name=organization.name,
        enabled=organization.enabled,
        created_at=organization.created_at,
        updated_at=organization.updated_at,
        role=context.membership.role.value,
        permissions=sorted(permissions_for(context.membership.role)),
    )


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Organization not found")
