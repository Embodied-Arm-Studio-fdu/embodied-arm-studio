"""Public current-user organization context schemas."""

from datetime import datetime
from typing import List

from pydantic import BaseModel


class OrganizationContextResponse(BaseModel):
    organization_id: str
    slug: str
    name: str
    enabled: bool
    created_at: datetime
    updated_at: datetime
    role: str
    permissions: List[str]


class CurrentUserContextResponse(BaseModel):
    user_id: str
    username: str
    display_name: str
    enabled: bool
    created_at: datetime
    updated_at: datetime
    organizations: List[OrganizationContextResponse]
