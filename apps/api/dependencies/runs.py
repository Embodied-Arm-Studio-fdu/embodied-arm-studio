"""Composition root for canonical Run persistence and Mock execution."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Protocol, Union

from sqlalchemy import Engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker

from apps.api.database import create_postgres_engine, create_session_factory
from robotics.adapters.mock import MockSimulatorAdapter
from robotics.runs import (
    IdempotentRunCreateCommand,
    IdempotentRunCreateResult,
    InMemoryRunRepository,
    PostgreSQLIdempotentRunCreationService,
    PostgreSQLProjectRunCreationService,
    PostgreSQLRunRepository,
    RunRepository,
    RunService,
)
from robotics.scenes import SceneRepository


class RunRepositoryConfigurationError(RuntimeError):
    """Raised when the configured authoritative Run repository is invalid."""


class RunCreationService(Protocol):
    """Boundary used by the HTTP Router for idempotent Run creation."""

    def create_run(
        self, command: IdempotentRunCreateCommand
    ) -> IdempotentRunCreateResult:
        ...


@dataclass(frozen=True)
class RunAssembly:
    """One app instance's single authoritative Run dependency graph."""

    engine: Optional[Engine]
    session_factory: Optional[sessionmaker]
    repository: RunRepository
    service: RunService
    creation_service: Optional[RunCreationService]
    adapter: MockSimulatorAdapter
    repository_mode: str


def build_run_assembly(
    *,
    storage_root: Path,
    tick_seconds: float,
    repository: Optional[RunRepository] = None,
    repository_mode: Optional[str] = None,
    database_url: Optional[Union[str, URL]] = None,
    environment: Optional[Mapping[str, str]] = None,
    creation_service: Optional[RunCreationService] = None,
    scene_repository: Optional[SceneRepository] = None,
) -> RunAssembly:
    """Build exactly one Repository, Service, and Adapter for an app instance."""

    source = os.environ if environment is None else environment
    if repository is not None:
        if repository_mode is not None:
            raise RunRepositoryConfigurationError(
                "repository_mode cannot be combined with an injected repository"
            )
        if database_url is not None:
            raise RunRepositoryConfigurationError(
                "database_url cannot be combined with an injected repository"
            )
        mode = "memory"
        engine = None
        factory = None
        resolved_repository = repository
    else:
        raw_mode = (
            repository_mode
            if repository_mode is not None
            else source.get("RUN_REPOSITORY_MODE", "postgresql")
        )
        mode = raw_mode.strip().lower() if isinstance(raw_mode, str) else ""
        if mode == "postgresql":
            if creation_service is not None:
                raise RunRepositoryConfigurationError(
                    "PostgreSQL mode must use its real idempotent creation service"
                )
            engine = create_postgres_engine(database_url, source)
            factory = create_session_factory(engine)
            resolved_repository = PostgreSQLRunRepository(factory)
        elif mode == "memory":
            engine = None
            factory = None
            resolved_repository = InMemoryRunRepository()
        else:
            raise RunRepositoryConfigurationError(
                "RUN_REPOSITORY_MODE must be 'postgresql' or 'memory'"
            )

    service = RunService(resolved_repository)
    if mode == "postgresql":
        resolved_creation_service = PostgreSQLProjectRunCreationService(
            service, scene_repository=scene_repository
        )
    else:
        resolved_creation_service = creation_service
        if resolved_creation_service is not None and not callable(
            getattr(resolved_creation_service, "create_run", None)
        ):
            raise RunRepositoryConfigurationError(
                "creation_service must implement create_run"
            )
    adapter = MockSimulatorAdapter(
        storage_root,
        tick_seconds=tick_seconds,
        run_service=service,
    )
    return RunAssembly(
        engine=engine,
        session_factory=factory,
        repository=resolved_repository,
        service=service,
        creation_service=resolved_creation_service,
        adapter=adapter,
        repository_mode=mode,
    )
