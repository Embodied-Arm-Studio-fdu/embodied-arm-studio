"""Teaching Project application service and strict input boundary."""

from typing import Dict, List
from uuid import uuid4

from .models import (
    TeachingProject,
    TeachingProjectAccess,
    TrainingProfile,
    current_config_version,
    evaluation_rubric,
    parameter_schema,
    starter_code_digest,
    utc_now,
)
from .repository import PostgreSQLTeachingProjectRepository


class TeachingProjectService:
    def __init__(self, repository: PostgreSQLTeachingProjectRepository) -> None:
        self.repository = repository

    def list_projects(
        self, course_id: str, actor_user_id: str
    ) -> List[TeachingProjectAccess]:
        return self.repository.list_projects(
            _identifier(course_id), _identifier(actor_user_id)
        )

    def get_project(
        self, course_id: str, project_id: str, actor_user_id: str
    ) -> TeachingProjectAccess:
        return self.repository.get_project(
            _identifier(course_id),
            _identifier(project_id),
            _identifier(actor_user_id),
        )

    def create_project(
        self,
        course_id: str,
        actor_user_id: str,
        *,
        title: str,
        description: str,
        instructions: str,
        starter_code: str,
        training_profile: TrainingProfile,
        video_enabled: bool,
        enabled: bool,
        environment_type: str = "s1_push",
    ) -> TeachingProjectAccess:
        actor = _identifier(actor_user_id)
        profile = _profile(training_profile)
        config_version = current_config_version(profile)
        code = _starter_code(starter_code)
        now = utc_now()
        project = TeachingProject(
            project_id=str(uuid4()),
            course_id=_identifier(course_id),
            title=_required_text(title, "title", 255),
            description=_optional_text(description, "description", 4000),
            instructions=_optional_text(instructions, "instructions", 16000),
            starter_code=code,
            starter_code_sha256=starter_code_digest(code),
            training_profile=profile,
            video_enabled=_boolean(video_enabled, "video_enabled"),
            parameter_schema=parameter_schema(environment_type),
            evaluation_rubric=evaluation_rubric(
                profile, config_version, environment_type
            ),
            config_version=config_version,
            enabled=_boolean(enabled, "enabled"),
            created_by_user_id=actor,
            created_at=now,
            updated_at=now,
        )
        return self.repository.create_project(project, actor)

    def update_project(
        self,
        course_id: str,
        project_id: str,
        actor_user_id: str,
        config_version: int,
        changes: Dict[str, object],
    ) -> TeachingProjectAccess:
        if isinstance(config_version, bool) or not isinstance(config_version, int):
            raise ValueError("config_version must be an integer")
        if config_version < 1:
            raise ValueError("config_version must be positive")
        current = self.repository.get_project(
            _identifier(course_id), _identifier(project_id), _identifier(actor_user_id)
        ).project
        normalized = _normalize_changes(
            changes, config_version + 1, current.environment_type
        )
        return self.repository.update_project(
            _identifier(course_id),
            _identifier(project_id),
            _identifier(actor_user_id),
            config_version,
            normalized,
        )


def _normalize_changes(
    changes: Dict[str, object], next_config_version: int,
    environment_type: str = "s1_push",
) -> Dict[str, object]:
    if not isinstance(changes, dict):
        raise ValueError("changes must be an object")
    allowed = {
        "title",
        "description",
        "instructions",
        "starter_code",
        "training_profile",
        "video_enabled",
        "enabled",
    }
    if set(changes).difference(allowed):
        raise ValueError("field cannot be changed")
    normalized: Dict[str, object] = {}
    for name, value in changes.items():
        if value is None:
            raise ValueError("%s cannot be null" % name)
        if name == "title":
            normalized[name] = _required_text(value, name, 255)
        elif name == "description":
            normalized[name] = _optional_text(value, name, 4000)
        elif name == "instructions":
            normalized[name] = _optional_text(value, name, 16000)
        elif name == "starter_code":
            code = _starter_code(value)
            normalized[name] = code
            normalized["starter_code_sha256"] = starter_code_digest(code)
        elif name == "training_profile":
            profile = _profile(value)
            normalized[name] = profile
            normalized["parameter_schema"] = parameter_schema(environment_type)
            normalized["evaluation_rubric"] = evaluation_rubric(
                profile, next_config_version, environment_type
            )
        else:
            normalized[name] = _boolean(value, name)
    return normalized


def _identifier(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 255:
        raise ValueError("identifier must contain 1-255 characters")
    return value.strip()


def _required_text(value: object, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum:
        raise ValueError("%s must contain 1-%d characters" % (name, maximum))
    return value.strip()


def _optional_text(value: object, name: str, maximum: int) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise ValueError("%s must contain at most %d characters" % (name, maximum))
    return value


def _starter_code(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("starter_code must be a string")
    starter_code_digest(value)
    return value


def _profile(value: object) -> TrainingProfile:
    try:
        return TrainingProfile(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("training_profile must be short or standard") from exc


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError("%s must be boolean" % name)
    return value

