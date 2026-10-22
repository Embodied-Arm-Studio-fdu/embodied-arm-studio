"""Transactional PostgreSQL repository for canonical teaching Projects."""

from copy import deepcopy
from typing import Callable, Dict, List, Tuple

from sqlalchemy import exists, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from robotics.organizations import OrganizationRole
from robotics.runs.db_models import (
    CourseRow,
    EnrollmentRow,
    OrganizationMembershipRow,
    OrganizationRow,
    TeachingClassRow,
    TeachingProjectRow,
    UserRow,
)

from .models import (
    ProjectPermissions,
    TeachingProject,
    TeachingProjectAccess,
    TrainingProfile,
    utc_now,
)


class TeachingProjectRepositoryError(RuntimeError):
    pass


class TeachingProjectNotFound(TeachingProjectRepositoryError, LookupError):
    pass


class TeachingProjectPermissionDenied(TeachingProjectRepositoryError):
    pass


class TeachingProjectConflict(TeachingProjectRepositoryError):
    pass


class TeachingProjectVersionConflict(TeachingProjectConflict):
    pass


class PostgreSQLTeachingProjectRepository:
    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def list_projects(
        self, course_id: str, actor_user_id: str
    ) -> List[TeachingProjectAccess]:
        session = self._session_factory()
        try:
            _, _, can_modify = self._course_access(
                session, course_id, actor_user_id
            )
            query = select(TeachingProjectRow).where(
                TeachingProjectRow.course_id == course_id
            )
            if not can_modify:
                query = query.where(TeachingProjectRow.enabled.is_(True))
            rows = session.execute(
                query.order_by(TeachingProjectRow.title, TeachingProjectRow.project_id)
            ).scalars().all()
            return [self._access(row, can_modify) for row in rows]
        except TeachingProjectRepositoryError:
            raise
        except SQLAlchemyError as exc:
            raise TeachingProjectRepositoryError(
                "could not list teaching projects"
            ) from exc
        finally:
            session.close()

    def get_project(
        self, course_id: str, project_id: str, actor_user_id: str
    ) -> TeachingProjectAccess:
        session = self._session_factory()
        try:
            _, _, can_modify = self._course_access(
                session, course_id, actor_user_id
            )
            row = self._project_row(session, course_id, project_id)
            if not row.enabled and not can_modify:
                raise TeachingProjectNotFound("teaching project not found")
            return self._access(row, can_modify)
        except TeachingProjectRepositoryError:
            raise
        except SQLAlchemyError as exc:
            raise TeachingProjectRepositoryError(
                "could not read teaching project"
            ) from exc
        finally:
            session.close()

    def create_project(
        self, project: TeachingProject, actor_user_id: str
    ) -> TeachingProjectAccess:
        session = self._session_factory()
        try:
            _, _, can_modify = self._course_access(
                session, project.course_id, actor_user_id, lock=True
            )
            if not can_modify:
                raise TeachingProjectPermissionDenied(
                    "teaching project management is not allowed"
                )
            if project.created_by_user_id != actor_user_id:
                raise TeachingProjectPermissionDenied(
                    "created_by_user_id must be the authenticated actor"
                )
            row = _project_to_row(project)
            session.add(row)
            session.commit()
            return self._access(row, True)
        except TeachingProjectRepositoryError:
            session.rollback()
            raise
        except IntegrityError as exc:
            session.rollback()
            raise TeachingProjectConflict(
                "teaching project title already exists in the course"
            ) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise TeachingProjectRepositoryError(
                "could not create teaching project"
            ) from exc
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    def update_project(
        self,
        course_id: str,
        project_id: str,
        actor_user_id: str,
        expected_config_version: int,
        changes: Dict[str, object],
    ) -> TeachingProjectAccess:
        session = self._session_factory()
        try:
            _, _, can_modify = self._course_access(
                session, course_id, actor_user_id, lock=True
            )
            row = self._project_row(session, course_id, project_id, lock=True)
            if not row.enabled and not can_modify:
                raise TeachingProjectNotFound("teaching project not found")
            if not can_modify:
                raise TeachingProjectPermissionDenied(
                    "teaching project management is not allowed"
                )
            facts_match = _facts_match(row, changes)
            if expected_config_version != row.config_version:
                if expected_config_version < row.config_version and facts_match:
                    session.rollback()
                    return self._access(row, True)
                raise TeachingProjectVersionConflict(
                    "project config version does not match"
                )
            if facts_match:
                session.rollback()
                return self._access(row, True)
            for name, value in changes.items():
                if name == "training_profile":
                    value = TrainingProfile(value).value
                elif name in {"parameter_schema", "evaluation_rubric"}:
                    value = deepcopy(value)
                setattr(row, name, value)
            row.config_version += 1
            row.updated_at = utc_now()
            session.commit()
            return self._access(row, True)
        except TeachingProjectRepositoryError:
            session.rollback()
            raise
        except IntegrityError as exc:
            session.rollback()
            raise TeachingProjectConflict(
                "teaching project update conflicts with existing data"
            ) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise TeachingProjectRepositoryError(
                "could not update teaching project"
            ) from exc
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _actor_role(
        session: Session, organization_id: str, actor_user_id: str
    ) -> OrganizationRole:
        role = session.execute(
            select(OrganizationMembershipRow.role)
            .join(
                OrganizationRow,
                OrganizationRow.organization_id
                == OrganizationMembershipRow.organization_id,
            )
            .join(UserRow, UserRow.user_id == OrganizationMembershipRow.user_id)
            .where(
                OrganizationMembershipRow.organization_id == organization_id,
                OrganizationMembershipRow.user_id == actor_user_id,
                OrganizationMembershipRow.enabled.is_(True),
                OrganizationRow.enabled.is_(True),
                UserRow.enabled.is_(True),
            )
        ).scalar_one_or_none()
        if role is None:
            raise TeachingProjectNotFound("teaching project not found")
        return OrganizationRole(role)

    def _course_access(
        self,
        session: Session,
        course_id: str,
        actor_user_id: str,
        lock: bool = False,
    ) -> Tuple[CourseRow, OrganizationRole, bool]:
        query = select(CourseRow).where(CourseRow.course_id == course_id)
        if lock:
            query = query.with_for_update()
        course = session.execute(query).scalar_one_or_none()
        if course is None or not course.enabled:
            raise TeachingProjectNotFound("teaching project not found")
        role = self._actor_role(session, course.organization_id, actor_user_id)
        if role is OrganizationRole.ADMINISTRATOR:
            return course, role, True
        if role is OrganizationRole.INSTRUCTOR:
            if course.owner_instructor_user_id == actor_user_id:
                return course, role, True
            assigned = session.execute(
                select(
                    exists().where(
                        TeachingClassRow.course_id == course_id,
                        TeachingClassRow.instructor_user_id == actor_user_id,
                        TeachingClassRow.enabled.is_(True),
                    )
                )
            ).scalar()
            if assigned:
                return course, role, False
            raise TeachingProjectNotFound("teaching project not found")
        enrolled = session.execute(
            select(
                exists().where(
                    TeachingClassRow.course_id == course_id,
                    TeachingClassRow.enabled.is_(True),
                    EnrollmentRow.class_id == TeachingClassRow.class_id,
                    EnrollmentRow.user_id == actor_user_id,
                    EnrollmentRow.enabled.is_(True),
                )
            )
        ).scalar()
        if not enrolled:
            raise TeachingProjectNotFound("teaching project not found")
        return course, role, False

    @staticmethod
    def _project_row(
        session: Session,
        course_id: str,
        project_id: str,
        lock: bool = False,
    ) -> TeachingProjectRow:
        query = select(TeachingProjectRow).where(
            TeachingProjectRow.project_id == project_id,
            TeachingProjectRow.course_id == course_id,
        )
        if lock:
            query = query.with_for_update()
        row = session.execute(query).scalar_one_or_none()
        if row is None:
            raise TeachingProjectNotFound("teaching project not found")
        return row

    @staticmethod
    def _access(
        row: TeachingProjectRow, can_modify: bool
    ) -> TeachingProjectAccess:
        project = _project_from_row(row)
        return TeachingProjectAccess(
            project=project,
            permissions=ProjectPermissions(
                can_view=True,
                can_use=project.enabled,
                can_modify=can_modify,
            ),
        )


def _facts_match(row: TeachingProjectRow, changes: Dict[str, object]) -> bool:
    for name, expected in changes.items():
        actual = getattr(row, name)
        if name == "training_profile":
            expected = TrainingProfile(expected).value
        elif name in {"parameter_schema", "evaluation_rubric"}:
            expected = deepcopy(expected)
        if actual != expected:
            return False
    return True


def _project_to_row(project: TeachingProject) -> TeachingProjectRow:
    return TeachingProjectRow(
        project_id=project.project_id,
        course_id=project.course_id,
        title=project.title,
        description=project.description,
        instructions=project.instructions,
        starter_code=project.starter_code,
        starter_code_sha256=project.starter_code_sha256,
        training_profile=project.training_profile.value,
        video_enabled=project.video_enabled,
        parameter_schema=deepcopy(project.parameter_schema),
        evaluation_rubric=deepcopy(project.evaluation_rubric),
        config_version=project.config_version,
        enabled=project.enabled,
        created_by_user_id=project.created_by_user_id,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


def _project_from_row(row: TeachingProjectRow) -> TeachingProject:
    return TeachingProject(
        project_id=row.project_id,
        course_id=row.course_id,
        title=row.title,
        description=row.description,
        instructions=row.instructions,
        starter_code=row.starter_code,
        starter_code_sha256=row.starter_code_sha256,
        training_profile=TrainingProfile(row.training_profile),
        video_enabled=row.video_enabled,
        parameter_schema=deepcopy(row.parameter_schema),
        evaluation_rubric=deepcopy(row.evaluation_rubric),
        config_version=row.config_version,
        enabled=row.enabled,
        created_by_user_id=row.created_by_user_id,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )

