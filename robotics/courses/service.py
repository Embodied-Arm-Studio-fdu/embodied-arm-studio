"""Course application service and input-normalization boundary."""

from typing import Dict, List
from uuid import uuid4

from .models import Course, Enrollment, TeachingClass, utc_now
from .repository import PostgreSQLCourseRepository


class CourseService:
    def __init__(self, repository: PostgreSQLCourseRepository) -> None:
        self.repository = repository

    def list_courses(self, organization_id: str, actor_user_id: str) -> List[Course]:
        return self.repository.list_courses(_identifier(organization_id), actor_user_id)

    def get_course(self, course_id: str, actor_user_id: str) -> Course:
        return self.repository.get_course(_identifier(course_id), actor_user_id)

    def create_course(
        self,
        organization_id: str,
        code: str,
        title: str,
        description: str,
        owner_instructor_user_id: str,
        actor_user_id: str,
    ) -> Course:
        now = utc_now()
        return self.repository.create_course(
            Course(
                course_id=str(uuid4()),
                organization_id=_identifier(organization_id),
                code=_required(code, "code", 255),
                title=_required(title, "title", 255),
                description=_optional(description, "description", 4000),
                owner_instructor_user_id=_identifier(owner_instructor_user_id),
                enabled=True,
                created_at=now,
                updated_at=now,
            ),
            actor_user_id,
        )

    def update_course(self, course_id: str, actor_user_id: str, changes: Dict[str, object]) -> Course:
        normalized = _normalize_changes(
            changes,
            required={"code": 255, "title": 255},
            optional={"description": 4000},
            identifiers={"owner_instructor_user_id"},
            booleans={"enabled"},
        )
        return self.repository.update_course(_identifier(course_id), actor_user_id, normalized)

    def list_classes(self, course_id: str, actor_user_id: str) -> List[TeachingClass]:
        return self.repository.list_classes(_identifier(course_id), actor_user_id)

    def get_class(self, class_id: str, actor_user_id: str) -> TeachingClass:
        return self.repository.get_class(_identifier(class_id), actor_user_id)

    def create_class(
        self,
        course_id: str,
        code: str,
        name: str,
        term: str,
        instructor_user_id: str,
        actor_user_id: str,
    ) -> TeachingClass:
        now = utc_now()
        return self.repository.create_class(
            TeachingClass(
                class_id=str(uuid4()),
                course_id=_identifier(course_id),
                code=_required(code, "code", 255),
                name=_required(name, "name", 255),
                term=_required(term, "term", 255),
                instructor_user_id=_identifier(instructor_user_id),
                enabled=True,
                created_at=now,
                updated_at=now,
            ),
            actor_user_id,
        )

    def update_class(self, class_id: str, actor_user_id: str, changes: Dict[str, object]) -> TeachingClass:
        normalized = _normalize_changes(
            changes,
            required={"code": 255, "name": 255, "term": 255},
            optional={},
            identifiers={"instructor_user_id"},
            booleans={"enabled"},
        )
        return self.repository.update_class(_identifier(class_id), actor_user_id, normalized)

    def list_enrollments(self, class_id: str, actor_user_id: str) -> List[Enrollment]:
        return self.repository.list_enrollments(_identifier(class_id), actor_user_id)

    def put_enrollment(
        self, class_id: str, user_id: str, enabled: bool, actor_user_id: str
    ) -> Enrollment:
        now = utc_now()
        return self.repository.put_enrollment(
            Enrollment(
                enrollment_id=str(uuid4()),
                class_id=_identifier(class_id),
                user_id=_identifier(user_id),
                enabled=bool(enabled),
                created_at=now,
                updated_at=now,
            ),
            actor_user_id,
        )


def _identifier(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 255:
        raise ValueError("identifier must contain 1-255 characters")
    return value.strip()


def _required(value: object, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum:
        raise ValueError("%s must contain 1-%d characters" % (name, maximum))
    return value.strip()


def _optional(value: object, name: str, maximum: int) -> str:
    if not isinstance(value, str) or len(value.strip()) > maximum:
        raise ValueError("%s must contain at most %d characters" % (name, maximum))
    return value.strip()


def _normalize_changes(changes, required, optional, identifiers, booleans):
    result = {}
    allowed = set(required) | set(optional) | set(identifiers) | set(booleans)
    for name, value in changes.items():
        if name not in allowed:
            raise ValueError("field cannot be changed")
        if value is None:
            raise ValueError("%s cannot be null" % name)
        if name in required:
            value = _required(value, name, required[name])
        elif name in optional:
            value = _optional(value, name, optional[name])
        elif name in identifiers:
            value = _identifier(value)
        elif not isinstance(value, bool):
            raise ValueError("%s must be boolean" % name)
        result[name] = value
    return result
