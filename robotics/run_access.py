"""Central browser authorization for canonical teaching Runs."""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Protocol, Sequence, Tuple

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from robotics.organizations import OrganizationRole
from robotics.runs.db_models import (
    CourseRow,
    EnrollmentRow,
    OrganizationMembershipRow,
    OrganizationRow,
    RunRow,
    TeachingClassRow,
    TeachingProjectRow,
    UserRow,
)
from robotics.runs.models import Run
from robotics.runs.postgres_repository import run_from_row
from robotics.runs.repository import RepositoryError, RunNotFound


class RunAccessNotFound(LookupError):
    """The Run or teaching scope is unknown, disabled, or invisible."""


class RunAccessForbidden(PermissionError):
    """The Run is readable but the requested operation is forbidden."""


class RunAccessUnavailable(RuntimeError):
    """The authoritative access facts could not be read."""


@dataclass(frozen=True)
class RunAuthorization:
    run: Run
    can_control: bool
    can_create_evaluation: bool


class RunAccessRepository(Protocol):
    def get_authorization(self, run_id: str, actor_user_id: str) -> RunAuthorization:
        ...

    def list_owned_runs(self, actor_user_id: str) -> List[Run]:
        ...

    def list_class_runs(
        self,
        class_id: str,
        actor_user_id: str,
        project_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Run]:
        ...


class RunAccessService:
    """Expose operation-specific access without leaking persistence rows."""

    def __init__(self, repository: RunAccessRepository) -> None:
        self._repository = repository

    def read(self, run_id: str, actor_user_id: str) -> Run:
        return self._repository.get_authorization(run_id, actor_user_id).run

    def control(self, run_id: str, actor_user_id: str) -> Run:
        authorization = self._repository.get_authorization(run_id, actor_user_id)
        if not authorization.can_control:
            raise RunAccessForbidden("Run control is forbidden")
        return authorization.run

    def create_evaluation(self, run_id: str, actor_user_id: str) -> Run:
        authorization = self._repository.get_authorization(run_id, actor_user_id)
        if not authorization.can_create_evaluation:
            raise RunAccessForbidden("Evaluation creation is forbidden")
        return authorization.run

    def list_own_runs(self, actor_user_id: str) -> List[Run]:
        return self._repository.list_owned_runs(actor_user_id)

    def list_class_runs(
        self,
        class_id: str,
        actor_user_id: str,
        project_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Run]:
        return self._repository.list_class_runs(
            class_id, actor_user_id, project_id=project_id, status=status
        )


class OwnerOnlyRunAccessRepository:
    """Explicit Memory-mode compatibility policy for isolated tests."""

    def __init__(self, run_service: Any) -> None:
        self._run_service = run_service

    def get_authorization(self, run_id: str, actor_user_id: str) -> RunAuthorization:
        try:
            run = self._run_service.get_run(run_id)
        except RunNotFound as exc:
            raise RunAccessNotFound("Run not found") from exc
        except RepositoryError as exc:
            raise RunAccessUnavailable("Run access is unavailable") from exc
        if run.user_id != actor_user_id:
            raise RunAccessNotFound("Run not found")
        return RunAuthorization(run, can_control=True, can_create_evaluation=True)

    def list_owned_runs(self, actor_user_id: str) -> List[Run]:
        try:
            return [
                run for run in self._run_service.list_runs()
                if run.user_id == actor_user_id
            ]
        except RepositoryError as exc:
            raise RunAccessUnavailable("Run access is unavailable") from exc

    def list_class_runs(
        self,
        class_id: str,
        actor_user_id: str,
        project_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Run]:
        # Memory mode has no authoritative teaching tables and deliberately
        # does not emulate or broaden formal Class visibility.
        raise RunAccessNotFound("Class not found")


class PostgreSQLRunAccessRepository:
    """Resolve browser Run access from current PostgreSQL teaching facts."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def get_authorization(self, run_id: str, actor_user_id: str) -> RunAuthorization:
        session = self._session_factory()
        try:
            run_row = session.get(RunRow, run_id)
            actor = session.get(UserRow, actor_user_id)
            if run_row is None or actor is None or not actor.enabled:
                raise RunAccessNotFound("Run not found")
            run = run_from_row(run_row)
            if run.class_id is None:
                if run.user_id != actor_user_id:
                    raise RunAccessNotFound("Run not found")
                return RunAuthorization(run, True, True)
            context = self._class_context(session, run.class_id, actor_user_id)
            can_read, is_valid = self._class_capabilities(
                session, run_row, actor_user_id, context
            )
            if not is_valid or not can_read:
                raise RunAccessNotFound("Run not found")
            owns_run = run.user_id == actor_user_id
            return RunAuthorization(
                run,
                can_control=owns_run,
                can_create_evaluation=True,
            )
        except RunAccessNotFound:
            raise
        except (SQLAlchemyError, RepositoryError, ValueError, TypeError) as exc:
            raise RunAccessUnavailable("Run access is unavailable") from exc
        finally:
            session.rollback()
            session.close()

    def list_owned_runs(self, actor_user_id: str) -> List[Run]:
        session = self._session_factory()
        try:
            actor = session.get(UserRow, actor_user_id)
            if actor is None or not actor.enabled:
                raise RunAccessNotFound("Run not found")
            rows = session.execute(
                select(RunRow)
                .where(RunRow.user_id == actor_user_id)
                .order_by(RunRow.created_at.desc(), RunRow.run_id.asc())
            ).scalars().all()
            class_ids = {row.class_id for row in rows if row.class_id is not None}
            project_ids = {row.project_id for row in rows if row.class_id is not None}
            context_rows = session.execute(
                select(
                    TeachingClassRow,
                    CourseRow,
                    OrganizationRow,
                    OrganizationMembershipRow,
                    EnrollmentRow,
                    TeachingProjectRow,
                )
                .join(CourseRow, CourseRow.course_id == TeachingClassRow.course_id)
                .join(
                    OrganizationRow,
                    OrganizationRow.organization_id == CourseRow.organization_id,
                )
                .join(
                    OrganizationMembershipRow,
                    (OrganizationMembershipRow.organization_id == OrganizationRow.organization_id)
                    & (OrganizationMembershipRow.user_id == actor_user_id),
                )
                .outerjoin(
                    EnrollmentRow,
                    (EnrollmentRow.class_id == TeachingClassRow.class_id)
                    & (EnrollmentRow.user_id == actor_user_id),
                )
                .join(
                    TeachingProjectRow,
                    TeachingProjectRow.course_id == CourseRow.course_id,
                )
                .where(
                    TeachingClassRow.class_id.in_(class_ids),
                    TeachingProjectRow.project_id.in_(project_ids),
                )
            ).all() if class_ids and project_ids else []
            contexts = {
                (item[0].class_id, item[5].project_id): item
                for item in context_rows
            }
            visible = []
            for row in rows:
                if row.class_id is None:
                    visible.append(run_from_row(row))
                    continue
                context = contexts.get((row.class_id, row.project_id))
                if context is not None and self._loaded_context_allows_owner(
                    row, actor_user_id, context
                ):
                    visible.append(run_from_row(row))
            return visible
        except RunAccessNotFound:
            raise
        except (SQLAlchemyError, RepositoryError, ValueError, TypeError) as exc:
            raise RunAccessUnavailable("Run access is unavailable") from exc
        finally:
            session.rollback()
            session.close()

    def list_class_runs(
        self,
        class_id: str,
        actor_user_id: str,
        project_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Run]:
        session = self._session_factory()
        try:
            actor = session.get(UserRow, actor_user_id)
            if actor is None or not actor.enabled:
                raise RunAccessNotFound("Class not found")
            context = self._class_context(session, class_id, actor_user_id)
            teaching_class, course, organization, membership, enrollment = context
            role = OrganizationRole(membership.role)
            is_student = role is OrganizationRole.STUDENT
            is_instructor = (
                role is OrganizationRole.INSTRUCTOR
                and (
                    teaching_class.instructor_user_id == actor_user_id
                    or course.owner_instructor_user_id == actor_user_id
                )
            )
            is_admin = role is OrganizationRole.ADMINISTRATOR
            if is_student and (enrollment is None or not enrollment.enabled):
                raise RunAccessNotFound("Class not found")
            if not (is_student or is_instructor or is_admin):
                raise RunAccessNotFound("Class not found")
            query = (
                select(RunRow)
                .join(
                    TeachingProjectRow,
                    TeachingProjectRow.project_id == RunRow.project_id,
                )
                .where(
                    RunRow.class_id == class_id,
                    TeachingProjectRow.course_id == course.course_id,
                    TeachingProjectRow.enabled.is_(True),
                )
            )
            if is_student:
                query = query.where(RunRow.user_id == actor_user_id)
            if project_id is not None:
                query = query.where(RunRow.project_id == project_id)
            if status is not None:
                query = query.where(RunRow.status == status)
            rows = session.execute(
                query.order_by(RunRow.created_at.desc(), RunRow.run_id.asc())
            ).scalars().all()
            return [
                run_from_row(row)
                for row in rows
                if self._snapshot_matches(row, course.course_id, class_id)
            ]
        except RunAccessNotFound:
            raise
        except (SQLAlchemyError, RepositoryError, ValueError, TypeError) as exc:
            raise RunAccessUnavailable("Run access is unavailable") from exc
        finally:
            session.rollback()
            session.close()

    @staticmethod
    def _class_context(
        session: Session, class_id: str, actor_user_id: str
    ) -> Tuple[Any, Any, Any, Any, Any]:
        result = session.execute(
            select(
                TeachingClassRow,
                CourseRow,
                OrganizationRow,
                OrganizationMembershipRow,
                EnrollmentRow,
            )
            .join(CourseRow, CourseRow.course_id == TeachingClassRow.course_id)
            .join(
                OrganizationRow,
                OrganizationRow.organization_id == CourseRow.organization_id,
            )
            .join(
                OrganizationMembershipRow,
                (OrganizationMembershipRow.organization_id == OrganizationRow.organization_id)
                & (OrganizationMembershipRow.user_id == actor_user_id),
            )
            .outerjoin(
                EnrollmentRow,
                (EnrollmentRow.class_id == TeachingClassRow.class_id)
                & (EnrollmentRow.user_id == actor_user_id),
            )
            .where(TeachingClassRow.class_id == class_id)
        ).one_or_none()
        if result is None:
            raise RunAccessNotFound("Class not found")
        teaching_class, course, organization, membership, enrollment = result
        if not (
            teaching_class.enabled
            and course.enabled
            and organization.enabled
            and membership.enabled
        ):
            raise RunAccessNotFound("Class not found")
        return teaching_class, course, organization, membership, enrollment

    def _class_capabilities(
        self,
        session: Session,
        run_row: RunRow,
        actor_user_id: str,
        context: Sequence[Any],
    ) -> Tuple[bool, bool]:
        teaching_class, course, _, membership, enrollment = context
        project = session.get(TeachingProjectRow, run_row.project_id)
        valid = (
            project is not None
            and project.enabled
            and project.course_id == course.course_id
            and self._snapshot_matches(
                run_row, course.course_id, teaching_class.class_id
            )
        )
        if not valid:
            return False, False
        role = OrganizationRole(membership.role)
        if role is OrganizationRole.ADMINISTRATOR:
            return True, True
        if role is OrganizationRole.INSTRUCTOR:
            return (
                teaching_class.instructor_user_id == actor_user_id
                or course.owner_instructor_user_id == actor_user_id,
                True,
            )
        return (
            run_row.user_id == actor_user_id
            and enrollment is not None
            and enrollment.enabled,
            True,
        )

    @classmethod
    def _loaded_context_allows_owner(
        cls, run_row: RunRow, actor_user_id: str, context: Sequence[Any]
    ) -> bool:
        teaching_class, course, organization, membership, enrollment, project = context
        if not (
            teaching_class.enabled
            and course.enabled
            and organization.enabled
            and membership.enabled
            and project.enabled
            and project.course_id == course.course_id
            and cls._snapshot_matches(
                run_row, course.course_id, teaching_class.class_id
            )
        ):
            return False
        role = OrganizationRole(membership.role)
        if role is OrganizationRole.STUDENT:
            return enrollment is not None and enrollment.enabled
        if role is OrganizationRole.INSTRUCTOR:
            return (
                teaching_class.instructor_user_id == actor_user_id
                or course.owner_instructor_user_id == actor_user_id
            )
        return role is OrganizationRole.ADMINISTRATOR

    @staticmethod
    def _snapshot_matches(
        run_row: RunRow, course_id: str, class_id: str
    ) -> bool:
        snapshot = run_row.project_config_snapshot
        return (
            isinstance(snapshot, dict)
            and snapshot.get("project_id") == run_row.project_id
            and snapshot.get("course_id") == course_id
            and snapshot.get("class_id") == class_id
        )
