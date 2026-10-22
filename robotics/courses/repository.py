"""Transactional PostgreSQL repository for canonical teaching resources."""

from typing import Callable, Dict, List

from sqlalchemy import exists, or_, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from robotics.organizations import OrganizationRole
from robotics.runs.db_models import (
    CourseRow,
    EnrollmentRow,
    OrganizationMembershipRow,
    OrganizationRow,
    TeachingClassRow,
    UserRow,
)

from .models import Course, Enrollment, TeachingClass, utc_now


class TeachingRepositoryError(RuntimeError):
    pass


class TeachingResourceNotFound(TeachingRepositoryError, LookupError):
    pass


class TeachingPermissionDenied(TeachingRepositoryError):
    pass


class TeachingConflict(TeachingRepositoryError):
    pass


class TeachingValidationError(TeachingRepositoryError, ValueError):
    pass


class PostgreSQLCourseRepository:
    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def list_courses(self, organization_id: str, actor_user_id: str) -> List[Course]:
        session = self._session_factory()
        try:
            role = self._actor_role(session, organization_id, actor_user_id)
            query = select(CourseRow).where(CourseRow.organization_id == organization_id)
            if role is OrganizationRole.STUDENT:
                enrolled = exists().where(
                    TeachingClassRow.course_id == CourseRow.course_id,
                    TeachingClassRow.enabled.is_(True),
                    EnrollmentRow.class_id == TeachingClassRow.class_id,
                    EnrollmentRow.user_id == actor_user_id,
                    EnrollmentRow.enabled.is_(True),
                )
                query = query.where(CourseRow.enabled.is_(True), enrolled)
            elif role is OrganizationRole.INSTRUCTOR:
                assigned = exists().where(
                    TeachingClassRow.course_id == CourseRow.course_id,
                    TeachingClassRow.instructor_user_id == actor_user_id,
                )
                query = query.where(
                    or_(CourseRow.owner_instructor_user_id == actor_user_id, assigned)
                )
            rows = session.execute(
                query.order_by(CourseRow.title, CourseRow.course_id)
            ).scalars().all()
            return [_course_from_row(row) for row in rows]
        except TeachingRepositoryError:
            raise
        except SQLAlchemyError as exc:
            raise TeachingRepositoryError("could not list courses") from exc
        finally:
            session.close()

    def get_course(self, course_id: str, actor_user_id: str) -> Course:
        session = self._session_factory()
        try:
            row = self._course_row(session, course_id)
            role = self._actor_role(session, row.organization_id, actor_user_id)
            if not self._can_view_course(session, row, actor_user_id, role):
                raise TeachingResourceNotFound("course not found")
            return _course_from_row(row)
        except TeachingRepositoryError:
            raise
        except SQLAlchemyError as exc:
            raise TeachingRepositoryError("could not read course") from exc
        finally:
            session.close()

    def create_course(self, course: Course, actor_user_id: str) -> Course:
        session = self._session_factory()
        try:
            role = self._actor_role(session, course.organization_id, actor_user_id)
            if role is OrganizationRole.STUDENT:
                raise TeachingPermissionDenied("course management is not allowed")
            if (
                role is OrganizationRole.INSTRUCTOR
                and course.owner_instructor_user_id != actor_user_id
            ):
                raise TeachingPermissionDenied("instructors can only own their course")
            self._require_staff(
                session, course.organization_id, course.owner_instructor_user_id
            )
            session.add(_course_to_row(course))
            session.commit()
            return course
        except (TeachingRepositoryError, ValueError):
            session.rollback()
            raise
        except IntegrityError as exc:
            session.rollback()
            raise TeachingConflict("course code already exists") from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise TeachingRepositoryError("could not create course") from exc
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    def update_course(
        self, course_id: str, actor_user_id: str, changes: Dict[str, object]
    ) -> Course:
        session = self._session_factory()
        try:
            row = self._course_row(session, course_id, lock=True)
            role = self._actor_role(session, row.organization_id, actor_user_id)
            is_owner = row.owner_instructor_user_id == actor_user_id
            if role is not OrganizationRole.ADMINISTRATOR and not (
                role is OrganizationRole.INSTRUCTOR and is_owner
            ):
                raise TeachingPermissionDenied("course management is not allowed")
            new_owner = changes.get("owner_instructor_user_id")
            if new_owner is not None and new_owner != row.owner_instructor_user_id:
                if role is not OrganizationRole.ADMINISTRATOR:
                    raise TeachingPermissionDenied("only administrators can transfer course ownership")
                self._require_staff(session, row.organization_id, str(new_owner))
            for name, value in changes.items():
                setattr(row, name, value)
            row.updated_at = utc_now()
            session.commit()
            return _course_from_row(row)
        except TeachingRepositoryError:
            session.rollback()
            raise
        except IntegrityError as exc:
            session.rollback()
            raise TeachingConflict("course code already exists") from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise TeachingRepositoryError("could not update course") from exc
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    def list_classes(self, course_id: str, actor_user_id: str) -> List[TeachingClass]:
        session = self._session_factory()
        try:
            course = self._course_row(session, course_id)
            role = self._actor_role(session, course.organization_id, actor_user_id)
            if not self._can_view_course(session, course, actor_user_id, role):
                raise TeachingResourceNotFound("course not found")
            query = select(TeachingClassRow).where(TeachingClassRow.course_id == course_id)
            if role is OrganizationRole.STUDENT:
                query = query.join(
                    EnrollmentRow, EnrollmentRow.class_id == TeachingClassRow.class_id
                ).where(
                    TeachingClassRow.enabled.is_(True),
                    EnrollmentRow.user_id == actor_user_id,
                    EnrollmentRow.enabled.is_(True),
                )
            elif (
                role is OrganizationRole.INSTRUCTOR
                and course.owner_instructor_user_id != actor_user_id
            ):
                query = query.where(TeachingClassRow.instructor_user_id == actor_user_id)
            rows = session.execute(
                query.order_by(TeachingClassRow.term, TeachingClassRow.code, TeachingClassRow.class_id)
            ).scalars().all()
            return [_class_from_row(row) for row in rows]
        except TeachingRepositoryError:
            raise
        except SQLAlchemyError as exc:
            raise TeachingRepositoryError("could not list classes") from exc
        finally:
            session.close()

    def get_class(self, class_id: str, actor_user_id: str) -> TeachingClass:
        session = self._session_factory()
        try:
            row, course = self._class_and_course(session, class_id)
            role = self._actor_role(session, course.organization_id, actor_user_id)
            if not self._can_view_class(session, row, course, actor_user_id, role):
                raise TeachingResourceNotFound("class not found")
            return _class_from_row(row)
        except TeachingRepositoryError:
            raise
        except SQLAlchemyError as exc:
            raise TeachingRepositoryError("could not read class") from exc
        finally:
            session.close()

    def create_class(
        self, teaching_class: TeachingClass, actor_user_id: str
    ) -> TeachingClass:
        session = self._session_factory()
        try:
            course = self._course_row(session, teaching_class.course_id, lock=True)
            role = self._actor_role(session, course.organization_id, actor_user_id)
            if not course.enabled:
                raise TeachingValidationError("disabled course cannot accept new classes")
            if role is not OrganizationRole.ADMINISTRATOR and not (
                role is OrganizationRole.INSTRUCTOR
                and course.owner_instructor_user_id == actor_user_id
            ):
                raise TeachingPermissionDenied("class creation is not allowed")
            self._require_staff(
                session, course.organization_id, teaching_class.instructor_user_id
            )
            session.add(_class_to_row(teaching_class))
            session.commit()
            return teaching_class
        except TeachingRepositoryError:
            session.rollback()
            raise
        except IntegrityError as exc:
            session.rollback()
            raise TeachingConflict("class term and code already exist") from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise TeachingRepositoryError("could not create class") from exc
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    def update_class(
        self, class_id: str, actor_user_id: str, changes: Dict[str, object]
    ) -> TeachingClass:
        session = self._session_factory()
        try:
            row, course = self._class_and_course(session, class_id, lock=True)
            role = self._actor_role(session, course.organization_id, actor_user_id)
            is_owner = course.owner_instructor_user_id == actor_user_id
            is_instructor = row.instructor_user_id == actor_user_id
            if role is not OrganizationRole.ADMINISTRATOR and not (
                role is OrganizationRole.INSTRUCTOR and (is_owner or is_instructor)
            ):
                raise TeachingPermissionDenied("class management is not allowed")
            new_instructor = changes.get("instructor_user_id")
            if new_instructor is not None and new_instructor != row.instructor_user_id:
                if role is not OrganizationRole.ADMINISTRATOR and not is_owner:
                    raise TeachingPermissionDenied(
                        "only the course owner or administrator can change the class instructor"
                    )
                self._require_staff(session, course.organization_id, str(new_instructor))
            for name, value in changes.items():
                setattr(row, name, value)
            row.updated_at = utc_now()
            session.commit()
            return _class_from_row(row)
        except TeachingRepositoryError:
            session.rollback()
            raise
        except IntegrityError as exc:
            session.rollback()
            raise TeachingConflict("class term and code already exist") from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise TeachingRepositoryError("could not update class") from exc
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    def list_enrollments(self, class_id: str, actor_user_id: str) -> List[Enrollment]:
        session = self._session_factory()
        try:
            teaching_class, course = self._class_and_course(session, class_id)
            role = self._actor_role(session, course.organization_id, actor_user_id)
            self._require_roster_access(teaching_class, course, actor_user_id, role)
            rows = session.execute(
                select(EnrollmentRow)
                .where(EnrollmentRow.class_id == class_id)
                .order_by(EnrollmentRow.user_id, EnrollmentRow.enrollment_id)
            ).scalars().all()
            return [_enrollment_from_row(row) for row in rows]
        except TeachingRepositoryError:
            raise
        except SQLAlchemyError as exc:
            raise TeachingRepositoryError("could not list enrollments") from exc
        finally:
            session.close()

    def put_enrollment(
        self,
        enrollment: Enrollment,
        actor_user_id: str,
    ) -> Enrollment:
        session = self._session_factory()
        try:
            teaching_class, course = self._class_and_course(
                session, enrollment.class_id, lock=True
            )
            role = self._actor_role(session, course.organization_id, actor_user_id)
            self._require_roster_access(teaching_class, course, actor_user_id, role)
            self._require_student(
                session,
                course.organization_id,
                enrollment.user_id,
                require_enabled=enrollment.enabled,
            )
            if enrollment.enabled and (not course.enabled or not teaching_class.enabled):
                raise TeachingValidationError(
                    "disabled course or class cannot accept enrollments"
                )
            row = session.execute(
                select(EnrollmentRow)
                .where(
                    EnrollmentRow.class_id == enrollment.class_id,
                    EnrollmentRow.user_id == enrollment.user_id,
                )
                .with_for_update()
            ).scalar_one_or_none()
            if row is None:
                session.add(_enrollment_to_row(enrollment))
                session.commit()
                return enrollment
            if row.enabled == enrollment.enabled:
                session.rollback()
                return _enrollment_from_row(row)
            row.enabled = enrollment.enabled
            row.updated_at = utc_now()
            session.commit()
            return _enrollment_from_row(row)
        except TeachingRepositoryError:
            session.rollback()
            raise
        except IntegrityError as exc:
            session.rollback()
            raise TeachingConflict("enrollment write conflicted") from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise TeachingRepositoryError("could not write enrollment") from exc
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _actor_role(session: Session, organization_id: str, user_id: str) -> OrganizationRole:
        row = session.execute(
            select(OrganizationMembershipRow)
            .join(OrganizationRow, OrganizationRow.organization_id == OrganizationMembershipRow.organization_id)
            .join(UserRow, UserRow.user_id == OrganizationMembershipRow.user_id)
            .where(
                OrganizationMembershipRow.organization_id == organization_id,
                OrganizationMembershipRow.user_id == user_id,
                OrganizationMembershipRow.enabled.is_(True),
                OrganizationRow.enabled.is_(True),
                UserRow.enabled.is_(True),
            )
        ).scalar_one_or_none()
        if row is None:
            raise TeachingResourceNotFound("organization context not found")
        return OrganizationRole(row.role)

    @staticmethod
    def _course_row(session: Session, course_id: str, lock: bool = False) -> CourseRow:
        query = select(CourseRow).where(CourseRow.course_id == course_id)
        if lock:
            query = query.with_for_update()
        row = session.execute(query).scalar_one_or_none()
        if row is None:
            raise TeachingResourceNotFound("course not found")
        return row

    @staticmethod
    def _class_and_course(
        session: Session, class_id: str, lock: bool = False
    ):
        query = (
            select(TeachingClassRow, CourseRow)
            .join(CourseRow, CourseRow.course_id == TeachingClassRow.course_id)
            .where(TeachingClassRow.class_id == class_id)
        )
        if lock:
            query = query.with_for_update()
        result = session.execute(query).one_or_none()
        if result is None:
            raise TeachingResourceNotFound("class not found")
        return result[0], result[1]

    @staticmethod
    def _require_staff(session: Session, organization_id: str, user_id: str) -> None:
        result = session.execute(
            select(OrganizationMembershipRow.role)
            .join(UserRow, UserRow.user_id == OrganizationMembershipRow.user_id)
            .where(
                OrganizationMembershipRow.organization_id == organization_id,
                OrganizationMembershipRow.user_id == user_id,
                OrganizationMembershipRow.enabled.is_(True),
                UserRow.enabled.is_(True),
            )
        ).scalar_one_or_none()
        if result is None:
            raise TeachingResourceNotFound("instructor membership not found")
        if result not in {
            OrganizationRole.INSTRUCTOR.value,
            OrganizationRole.ADMINISTRATOR.value,
        }:
            raise TeachingValidationError("course or class instructor must be staff")

    @staticmethod
    def _require_student(
        session: Session,
        organization_id: str,
        user_id: str,
        require_enabled: bool,
    ) -> None:
        query = (
            select(OrganizationMembershipRow.role, OrganizationMembershipRow.enabled, UserRow.enabled)
            .join(UserRow, UserRow.user_id == OrganizationMembershipRow.user_id)
            .where(
                OrganizationMembershipRow.organization_id == organization_id,
                OrganizationMembershipRow.user_id == user_id,
            )
        )
        result = session.execute(query).one_or_none()
        if result is None:
            raise TeachingResourceNotFound("student membership not found")
        if result[0] != OrganizationRole.STUDENT.value:
            raise TeachingValidationError("enrollment target must be a student")
        if require_enabled and (not result[1] or not result[2]):
            raise TeachingResourceNotFound("student membership not found")

    @staticmethod
    def _can_view_course(
        session: Session,
        course: CourseRow,
        actor_user_id: str,
        role: OrganizationRole,
    ) -> bool:
        if role is OrganizationRole.ADMINISTRATOR:
            return True
        if role is OrganizationRole.INSTRUCTOR:
            if course.owner_instructor_user_id == actor_user_id:
                return True
            return bool(
                session.execute(
                    select(
                        exists().where(
                            TeachingClassRow.course_id == course.course_id,
                            TeachingClassRow.instructor_user_id == actor_user_id,
                        )
                    )
                ).scalar()
            )
        if not course.enabled:
            return False
        return bool(
            session.execute(
                select(
                    exists().where(
                        TeachingClassRow.course_id == course.course_id,
                        TeachingClassRow.enabled.is_(True),
                        EnrollmentRow.class_id == TeachingClassRow.class_id,
                        EnrollmentRow.user_id == actor_user_id,
                        EnrollmentRow.enabled.is_(True),
                    )
                )
            ).scalar()
        )

    def _can_view_class(
        self,
        session: Session,
        teaching_class: TeachingClassRow,
        course: CourseRow,
        actor_user_id: str,
        role: OrganizationRole,
    ) -> bool:
        if role is OrganizationRole.ADMINISTRATOR:
            return True
        if role is OrganizationRole.INSTRUCTOR:
            return (
                course.owner_instructor_user_id == actor_user_id
                or teaching_class.instructor_user_id == actor_user_id
            )
        if not course.enabled or not teaching_class.enabled:
            return False
        return bool(
            session.execute(
                select(
                    exists().where(
                        EnrollmentRow.class_id == teaching_class.class_id,
                        EnrollmentRow.user_id == actor_user_id,
                        EnrollmentRow.enabled.is_(True),
                    )
                )
            ).scalar()
        )

    @staticmethod
    def _require_roster_access(
        teaching_class: TeachingClassRow,
        course: CourseRow,
        actor_user_id: str,
        role: OrganizationRole,
    ) -> None:
        if role is OrganizationRole.ADMINISTRATOR:
            return
        if role is OrganizationRole.INSTRUCTOR and (
            course.owner_instructor_user_id == actor_user_id
            or teaching_class.instructor_user_id == actor_user_id
        ):
            return
        raise TeachingPermissionDenied("roster access is not allowed")


def _course_to_row(value: Course) -> CourseRow:
    return CourseRow(**value.__dict__)


def _course_from_row(row: CourseRow) -> Course:
    return Course(
        course_id=row.course_id,
        organization_id=row.organization_id,
        code=row.code,
        title=row.title,
        description=row.description,
        owner_instructor_user_id=row.owner_instructor_user_id,
        enabled=row.enabled,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _class_to_row(value: TeachingClass) -> TeachingClassRow:
    return TeachingClassRow(**value.__dict__)


def _class_from_row(row: TeachingClassRow) -> TeachingClass:
    return TeachingClass(
        class_id=row.class_id,
        course_id=row.course_id,
        code=row.code,
        name=row.name,
        term=row.term,
        instructor_user_id=row.instructor_user_id,
        enabled=row.enabled,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _enrollment_to_row(value: Enrollment) -> EnrollmentRow:
    return EnrollmentRow(**value.__dict__)


def _enrollment_from_row(row: EnrollmentRow) -> Enrollment:
    return Enrollment(
        enrollment_id=row.enrollment_id,
        class_id=row.class_id,
        user_id=row.user_id,
        enabled=row.enabled,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
