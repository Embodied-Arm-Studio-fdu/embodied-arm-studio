"""Synchronous PostgreSQL engine and SQLAlchemy Session lifecycle helpers."""

import os
from contextlib import contextmanager
from typing import Callable, Iterator, Mapping, Optional, Union

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker


class DatabaseConfigurationError(RuntimeError):
    """Raised when the PostgreSQL connection configuration is unavailable or invalid."""


class DatabaseReadinessError(RuntimeError):
    """Raised when PostgreSQL or the migrated Run schema is unavailable."""


CURRENT_SCHEMA_REVISION = "20260903_0008"


def resolve_database_url(
    database_url: Optional[Union[str, URL]] = None,
    environment: Optional[Mapping[str, str]] = None,
) -> URL:
    """Resolve an injected URL or DATABASE_URL without logging its contents."""

    source = os.environ if environment is None else environment
    candidate = database_url if database_url is not None else source.get("DATABASE_URL")
    if candidate is None or (isinstance(candidate, str) and not candidate.strip()):
        raise DatabaseConfigurationError(
            "DATABASE_URL is required and must point to PostgreSQL"
        )
    try:
        url = candidate if isinstance(candidate, URL) else make_url(candidate)
    except (TypeError, ValueError) as exc:
        raise DatabaseConfigurationError("DATABASE_URL is not a valid database URL") from exc
    if url.get_backend_name() != "postgresql":
        raise DatabaseConfigurationError("DATABASE_URL must use PostgreSQL")
    if url.drivername == "postgresql":
        url = url.set(drivername="postgresql+psycopg")
    if url.drivername != "postgresql+psycopg":
        raise DatabaseConfigurationError(
            "DATABASE_URL must use the PostgreSQL psycopg driver"
        )
    return url


def create_postgres_engine(
    database_url: Optional[Union[str, URL]] = None,
    environment: Optional[Mapping[str, str]] = None,
) -> Engine:
    """Create a synchronous PostgreSQL Engine without opening a connection."""

    return create_engine(
        resolve_database_url(database_url, environment),
        future=True,
        hide_parameters=True,
        pool_pre_ping=True,
    )


def create_session_factory(engine: Engine) -> sessionmaker:
    """Create the injectable Session factory used by repositories and services."""

    return sessionmaker(
        bind=engine,
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
        future=True,
    )


def check_postgres_readiness(engine: Engine) -> None:
    """Check connectivity and the migrated runs table without modifying data."""

    try:
        with engine.connect() as connection:
            revision = connection.exec_driver_sql(
                "SELECT version_num FROM alembic_version"
            ).scalar_one()
            if revision != CURRENT_SCHEMA_REVISION:
                raise DatabaseReadinessError(
                    "PostgreSQL is unavailable or the Run schema is not migrated"
                )
            connection.exec_driver_sql(
                "SELECT run_id, class_id, training_steps, update_steps, stop_reason "
                "FROM runs LIMIT 1"
            )
            connection.exec_driver_sql(
                "SELECT artifact_id FROM run_artifacts LIMIT 1"
            )
            connection.exec_driver_sql(
                "SELECT evaluation_id FROM evaluation_reports LIMIT 1"
            )
            connection.exec_driver_sql(
                "SELECT user_id, username, enabled FROM users LIMIT 1"
            )
            connection.exec_driver_sql(
                "SELECT organization_id, slug, enabled "
                "FROM organizations LIMIT 1"
            )
            connection.exec_driver_sql(
                "SELECT membership_id, organization_id, user_id, role, enabled "
                "FROM organization_memberships LIMIT 1"
            )
            connection.exec_driver_sql(
                "SELECT course_id, organization_id, owner_instructor_user_id, enabled "
                "FROM courses LIMIT 1"
            )
            connection.exec_driver_sql(
                "SELECT class_id, course_id, instructor_user_id, enabled "
                "FROM classes LIMIT 1"
            )
            connection.exec_driver_sql(
                "SELECT enrollment_id, class_id, user_id, enabled "
                "FROM enrollments LIMIT 1"
            )
            connection.exec_driver_sql(
                "SELECT project_id, course_id, config_version, enabled "
                "FROM teaching_projects LIMIT 1"
            )
    except SQLAlchemyError as exc:
        raise DatabaseReadinessError(
            "PostgreSQL is unavailable or the Run schema is not migrated"
        ) from exc


@contextmanager
def session_scope(factory: Callable[[], Session]) -> Iterator[Session]:
    """Commit successful work and always roll back failures and close the Session."""

    session = factory()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


def create_session_dependency(
    factory: Callable[[], Session],
) -> Callable[[], Iterator[Session]]:
    """Build a FastAPI-compatible dependency without wiring it into current routes."""

    def dependency() -> Iterator[Session]:
        with session_scope(factory) as session:
            yield session

    return dependency
