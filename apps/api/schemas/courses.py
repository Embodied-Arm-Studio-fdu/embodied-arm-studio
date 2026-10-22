"""Strict canonical teaching-resource request and response schemas."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class _StrictRequest(BaseModel):
    class Config:
        extra = "forbid"


class CourseCreateRequest(_StrictRequest):
    organization_id: str = Field(..., min_length=1, max_length=255)
    code: str = Field(..., min_length=1, max_length=255)
    title: str = Field(..., min_length=1, max_length=255)
    description: str = Field("", max_length=4000)
    owner_instructor_user_id: Optional[str] = Field(None, min_length=1, max_length=255)


class CoursePatchRequest(_StrictRequest):
    code: Optional[str] = Field(None, min_length=1, max_length=255)
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=4000)
    owner_instructor_user_id: Optional[str] = Field(None, min_length=1, max_length=255)
    enabled: Optional[bool] = None


class CourseResponse(BaseModel):
    course_id: str
    organization_id: str
    code: str
    title: str
    description: str
    owner_instructor_user_id: str
    enabled: bool
    created_at: datetime
    updated_at: datetime


class TeachingClassCreateRequest(_StrictRequest):
    code: str = Field(..., min_length=1, max_length=255)
    name: str = Field(..., min_length=1, max_length=255)
    term: str = Field(..., min_length=1, max_length=255)
    instructor_user_id: str = Field(..., min_length=1, max_length=255)


class TeachingClassPatchRequest(_StrictRequest):
    code: Optional[str] = Field(None, min_length=1, max_length=255)
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    term: Optional[str] = Field(None, min_length=1, max_length=255)
    instructor_user_id: Optional[str] = Field(None, min_length=1, max_length=255)
    enabled: Optional[bool] = None


class TeachingClassResponse(BaseModel):
    class_id: str
    course_id: str
    code: str
    name: str
    term: str
    instructor_user_id: str
    enabled: bool
    created_at: datetime
    updated_at: datetime


class EnrollmentPutRequest(_StrictRequest):
    enabled: bool


class EnrollmentResponse(BaseModel):
    enrollment_id: str
    class_id: str
    user_id: str
    enabled: bool
    created_at: datetime
    updated_at: datetime
