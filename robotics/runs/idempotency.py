"""Domain inputs and request fingerprints for idempotent Run creation."""

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Dict, Optional

from robotics.runs.models import ExecutionMode, Run, RunKind


class InvalidIdempotentRunCreateRequest(ValueError):
    """Raised when an idempotent creation input cannot be normalized safely."""


class IdempotencyConflict(ValueError):
    """Raised when a user reuses a key for a different Run creation request."""

    def __init__(self, user_id: str, idempotency_key: str) -> None:
        super().__init__("Idempotency key was already used for a different request")
        self.user_id = user_id
        self.idempotency_key = idempotency_key


@dataclass(frozen=True)
class IdempotentRunCreateCommand:
    """Deep-copied business input for one idempotent Run creation attempt."""

    user_id: str
    idempotency_key: str
    project_id: str
    execution_mode: ExecutionMode
    class_id: Optional[str] = None
    project_config_version: Optional[int] = None
    environment_type: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    project_config_snapshot: Dict[str, Any] = field(default_factory=dict)
    scene_ref: Optional[Dict[str, Any]] = None
    code: str = ""
    run_kind: RunKind = RunKind.TRAIN
    reward_extension: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        try:
            _required_database_text(self.user_id, "user_id")
            _required_database_text(self.idempotency_key, "idempotency_key")
            if not isinstance(self.project_id, str) or not self.project_id.strip():
                raise ValueError("project_id must be a non-empty string")
            if not isinstance(self.execution_mode, ExecutionMode):
                raise ValueError("execution_mode must be an ExecutionMode")
            if not isinstance(self.run_kind, RunKind):
                raise ValueError("run_kind must be a RunKind")
            if not isinstance(self.parameters, dict):
                raise ValueError("parameters must be a dict")
            if self.class_id is not None:
                _required_database_text(self.class_id, "class_id")
            if self.project_config_version is not None and (
                isinstance(self.project_config_version, bool)
                or not isinstance(self.project_config_version, int)
                or self.project_config_version < 1
            ):
                raise ValueError("project_config_version must be a positive integer")
            if self.environment_type is not None:
                _required_database_text(self.environment_type, "environment_type")
            if not isinstance(self.project_config_snapshot, dict):
                raise ValueError("project_config_snapshot must be a dict")
            if self.scene_ref is not None:
                if not isinstance(self.scene_ref, dict):
                    raise ValueError("scene_ref must be a dict")
                if set(self.scene_ref) != {"scene_id", "revision"}:
                    raise ValueError("scene_ref must contain only scene_id and revision")
                _required_database_text(self.scene_ref.get("scene_id"), "scene_ref.scene_id")
                revision = self.scene_ref.get("revision")
                if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
                    raise ValueError("scene_ref.revision must be a positive integer")
            if not isinstance(self.code, str):
                raise ValueError("code must be a string")
            object.__setattr__(self, "parameters", deepcopy(self.parameters))
            object.__setattr__(
                self,
                "project_config_snapshot",
                deepcopy(self.project_config_snapshot),
            )
            object.__setattr__(self, "scene_ref", deepcopy(self.scene_ref))
            if self.reward_extension is not None:
                if self.run_kind is not RunKind.TRAIN:
                    raise ValueError("Reward extensions require a training Run")
                from simulation.src.simulation_core.reward_extension import validate_extension
                validate_extension(self.environment_type, self.reward_extension)
                if self.code != self.reward_extension["source"]:
                    raise ValueError("Submitted code must match the reward extension source")
                object.__setattr__(self, "reward_extension", deepcopy(self.reward_extension))
            canonical_run_create_json(self)
        except (TypeError, ValueError, UnicodeError) as exc:
            raise InvalidIdempotentRunCreateRequest(str(exc)) from exc

    def create_kwargs(self) -> Dict[str, Any]:
        """Return isolated arguments accepted by ``RunService.build_run``."""
        snapshot = deepcopy(self.project_config_snapshot)
        if self.reward_extension is not None:
            snapshot["reward_extension"] = deepcopy(self.reward_extension)
        return {
            "user_id": self.user_id,
            "project_id": self.project_id,
            "execution_mode": self.execution_mode,
            "parameters": deepcopy(self.parameters),
            "project_config_snapshot": snapshot,
            "code": self.code,
            "run_kind": self.run_kind,
            "class_id": self.class_id,
        }


@dataclass(frozen=True)
class IdempotentRunCreateResult:
    run: Run
    replayed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.run, Run):
            raise TypeError("run must be a Run")
        if not isinstance(self.replayed, bool):
            raise TypeError("replayed must be a boolean")


def canonical_run_create_json(command: IdempotentRunCreateCommand) -> str:
    """Serialize only Run-affecting business fields as stable canonical JSON."""

    if (
        command.class_id is not None
        and command.project_config_version is not None
        and command.environment_type is None
    ):
        seed = command.parameters.get("seed")
        if set(command.parameters) != {"seed"} or isinstance(seed, bool) or not isinstance(seed, int):
            raise InvalidIdempotentRunCreateRequest(
                "parameters must contain only an integer seed"
            )
        payload = {
            "class_id": command.class_id,
            "code_sha256": hashlib.sha256(command.code.encode("utf-8")).hexdigest(),
            "execution_mode": command.execution_mode.value,
            "project_config_version": command.project_config_version,
            "project_id": command.project_id,
            "run_kind": command.run_kind.value,
            "seed": seed,
            "scene_ref": deepcopy(command.scene_ref),
            "user_id": command.user_id,
        }
    else:
        payload = {
            "class_id": command.class_id,
            "code_sha256": hashlib.sha256(command.code.encode("utf-8")).hexdigest(),
            "environment_type": command.environment_type,
            "execution_mode": command.execution_mode.value,
            "parameters": deepcopy(command.parameters),
            "project_config_snapshot": deepcopy(command.project_config_snapshot),
            "project_config_version": command.project_config_version,
            "project_id": command.project_id,
            "run_kind": command.run_kind.value,
            "scene_ref": deepcopy(command.scene_ref),
            "user_id": command.user_id,
        }
    try:
        if command.reward_extension is not None:
            payload["reward_extension"] = deepcopy(command.reward_extension)
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, UnicodeError) as exc:
        raise InvalidIdempotentRunCreateRequest(
            "Run creation fields must be canonical JSON values"
        ) from exc


def fingerprint_run_create(command: IdempotentRunCreateCommand) -> str:
    """Return a stable SHA-256 fingerprint without Python's process-local hash."""

    encoded = canonical_run_create_json(command).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_database_text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-empty string" % name)
    if len(value) > 255:
        raise ValueError("%s must be at most 255 characters" % name)

