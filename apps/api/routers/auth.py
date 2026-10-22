"""Canonical login and current-user routes."""

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException

from apps.api.dependencies.auth import unauthorized
from apps.api.schemas.auth import LoginRequest, LoginResponse, PublicUserResponse
from robotics.auth import AuthenticationFailed, AuthenticationService, User


def create_auth_router(
    authentication_service: Optional[AuthenticationService],
    current_user_provider: Any,
) -> APIRouter:
    router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])

    @router.post("/login", response_model=LoginResponse)
    def login(payload: LoginRequest) -> LoginResponse:
        if authentication_service is None:
            raise HTTPException(
                status_code=503, detail="Authentication is not configured"
            )
        try:
            user = authentication_service.authenticate(
                payload.username, payload.password
            )
            token = authentication_service.issue_access_token(user)
        except (AuthenticationFailed, ValueError):
            raise unauthorized()
        return LoginResponse(
            access_token=token,
            expires_in=authentication_service.tokens.config.expires_seconds,
            user=PublicUserResponse.model_validate(user.public_dict()),
        )

    @router.get("/me", response_model=PublicUserResponse)
    def me(user: User = Depends(current_user_provider)) -> PublicUserResponse:
        return PublicUserResponse.model_validate(user.public_dict())

    return router
