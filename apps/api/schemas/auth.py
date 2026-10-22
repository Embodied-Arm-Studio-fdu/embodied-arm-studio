"""Public authentication request and response schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(..., min_length=1, max_length=255)
    # Keep password validation inside the authentication service so framework
    # validation errors never echo the submitted secret.
    password: str


class PublicUserResponse(BaseModel):
    user_id: str
    username: str
    display_name: str
    enabled: bool
    created_at: datetime
    updated_at: datetime


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: PublicUserResponse
