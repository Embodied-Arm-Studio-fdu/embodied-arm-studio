"""PostgreSQL repository for organizations and memberships."""

from typing import Callable, List, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from robotics.runs.db_models import (
    OrganizationMembershipRow,
    OrganizationRow,
    UserRow,
)

from .models import (
    Organization,
    OrganizationContext,
    OrganizationMembership,
    OrganizationRole,
    utc_now,
)


class OrganizationRepositoryError(RuntimeError):
    pass


class OrganizationNotFound(OrganizationRepositoryError, LookupError):
    pass


class MembershipNotFound(OrganizationRepositoryError, LookupError):
    pass


class OrganizationConflict(OrganizationRepositoryError):
    pass


class LastAdministratorError(OrganizationConflict):
    pass


class PostgreSQLOrganizationRepository:
    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def create_with_administrator(
        self,
        organization: Organization,
        membership: OrganizationMembership,
    ) -> Tuple[Organization, OrganizationMembership]:
        session = self._session_factory()
        try:
            if membership.organization_id != organization.organization_id:
                raise ValueError("membership organization does not match")
            if membership.role is not OrganizationRole.ADMINISTRATOR:
                raise ValueError("initial membership must be administrator")
            session.add(_organization_to_row(organization))
            # The rows intentionally have no ORM relationship. Flush the
            # parent first so PostgreSQL can enforce both foreign keys while
            # keeping the entire operation inside this transaction.
            session.flush()
            session.add(_membership_to_row(membership))
            session.commit()
            return organization, membership
        except IntegrityError as exc:
            session.rollback()
            raise OrganizationConflict(
                "organization or membership already exists"
            ) from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise OrganizationRepositoryError(
                "could not create organization"
            ) from exc
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    def get_by_id(self, organization_id: str) -> Organization:
        return self._get_organization(
            OrganizationRow.organization_id == organization_id
        )

    def get_by_slug(self, slug: str) -> Organization:
        return self._get_organization(OrganizationRow.slug == slug)

    def find_by_slug(self, slug: str) -> Optional[Organization]:
        try:
            return self.get_by_slug(slug)
        except OrganizationNotFound:
            return None

    def _get_organization(self, condition) -> Organization:
        session = self._session_factory()
        try:
            row = session.execute(
                select(OrganizationRow).where(condition)
            ).scalar_one_or_none()
            if row is None:
                raise OrganizationNotFound("organization not found")
            return _organization_from_row(row)
        except OrganizationNotFound:
            raise
        except SQLAlchemyError as exc:
            raise OrganizationRepositoryError(
                "could not read organization"
            ) from exc
        finally:
            session.close()

    def list_enabled_for_user(
        self, user_id: str
    ) -> List[OrganizationContext]:
        session = self._session_factory()
        try:
            rows = session.execute(
                select(OrganizationRow, OrganizationMembershipRow)
                .join(
                    OrganizationMembershipRow,
                    OrganizationMembershipRow.organization_id
                    == OrganizationRow.organization_id,
                )
                .where(
                    OrganizationMembershipRow.user_id == user_id,
                    OrganizationMembershipRow.enabled.is_(True),
                    OrganizationRow.enabled.is_(True),
                )
                .order_by(OrganizationRow.slug)
            ).all()
            return [
                OrganizationContext(
                    _organization_from_row(organization),
                    _membership_from_row(membership),
                )
                for organization, membership in rows
            ]
        except SQLAlchemyError as exc:
            raise OrganizationRepositoryError(
                "could not list memberships"
            ) from exc
        finally:
            session.close()

    def get_enabled_context(
        self, organization_id: str, user_id: str
    ) -> OrganizationContext:
        session = self._session_factory()
        try:
            row = session.execute(
                select(OrganizationRow, OrganizationMembershipRow)
                .join(
                    OrganizationMembershipRow,
                    OrganizationMembershipRow.organization_id
                    == OrganizationRow.organization_id,
                )
                .where(
                    OrganizationRow.organization_id == organization_id,
                    OrganizationRow.enabled.is_(True),
                    OrganizationMembershipRow.user_id == user_id,
                    OrganizationMembershipRow.enabled.is_(True),
                )
            ).one_or_none()
            if row is None:
                raise OrganizationNotFound("organization not found")
            return OrganizationContext(
                _organization_from_row(row[0]),
                _membership_from_row(row[1]),
            )
        except OrganizationNotFound:
            raise
        except SQLAlchemyError as exc:
            raise OrganizationRepositoryError(
                "could not read organization context"
            ) from exc
        finally:
            session.close()

    def get_membership(
        self, organization_id: str, user_id: str
    ) -> OrganizationMembership:
        session = self._session_factory()
        try:
            row = session.execute(
                select(OrganizationMembershipRow).where(
                    OrganizationMembershipRow.organization_id
                    == organization_id,
                    OrganizationMembershipRow.user_id == user_id,
                )
            ).scalar_one_or_none()
            if row is None:
                raise MembershipNotFound("membership not found")
            return _membership_from_row(row)
        except MembershipNotFound:
            raise
        except SQLAlchemyError as exc:
            raise OrganizationRepositoryError(
                "could not read membership"
            ) from exc
        finally:
            session.close()

    def add_membership(
        self, membership: OrganizationMembership
    ) -> OrganizationMembership:
        session = self._session_factory()
        try:
            self._lock_organization(session, membership.organization_id)
            existing = self._membership_row(
                session, membership.organization_id, membership.user_id
            )
            if existing is not None:
                current = _membership_from_row(existing)
                if (
                    current.role is membership.role
                    and current.enabled == membership.enabled
                ):
                    session.rollback()
                    return current
                raise OrganizationConflict(
                    "membership exists with different facts"
                )
            session.add(_membership_to_row(membership))
            session.commit()
            return membership
        except (OrganizationNotFound, OrganizationConflict):
            session.rollback()
            raise
        except IntegrityError as exc:
            session.rollback()
            raise OrganizationConflict("membership already exists") from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise OrganizationRepositoryError(
                "could not add membership"
            ) from exc
        finally:
            session.close()

    def set_membership(
        self,
        organization_id: str,
        user_id: str,
        role: OrganizationRole,
        enabled: bool,
    ) -> OrganizationMembership:
        session = self._session_factory()
        try:
            self._lock_organization(session, organization_id)
            row = self._membership_row(session, organization_id, user_id)
            if row is None:
                raise MembershipNotFound("membership not found")
            role = OrganizationRole(role)
            if row.role == role.value and row.enabled == enabled:
                session.rollback()
                return _membership_from_row(row)
            if (
                row.enabled
                and row.role == OrganizationRole.ADMINISTRATOR.value
                and (
                    not enabled
                    or role is not OrganizationRole.ADMINISTRATOR
                )
                and self._enabled_administrator_count(
                    session, organization_id
                ) <= 1
            ):
                raise LastAdministratorError(
                    "cannot disable or demote the last administrator"
                )
            row.role = role.value
            row.enabled = enabled
            row.updated_at = utc_now()
            session.commit()
            return _membership_from_row(row)
        except (
            OrganizationNotFound,
            MembershipNotFound,
            LastAdministratorError,
        ):
            session.rollback()
            raise
        except SQLAlchemyError as exc:
            session.rollback()
            raise OrganizationRepositoryError(
                "could not update membership"
            ) from exc
        finally:
            session.close()

    @staticmethod
    def _lock_organization(session: Session, organization_id: str) -> None:
        row = session.execute(
            select(OrganizationRow)
            .where(OrganizationRow.organization_id == organization_id)
            .with_for_update()
        ).scalar_one_or_none()
        if row is None:
            raise OrganizationNotFound("organization not found")

    @staticmethod
    def _membership_row(
        session: Session, organization_id: str, user_id: str
    ) -> Optional[OrganizationMembershipRow]:
        return session.execute(
            select(OrganizationMembershipRow).where(
                OrganizationMembershipRow.organization_id
                == organization_id,
                OrganizationMembershipRow.user_id == user_id,
            )
        ).scalar_one_or_none()

    @staticmethod
    def _enabled_administrator_count(
        session: Session, organization_id: str
    ) -> int:
        return int(
            session.execute(
                select(func.count())
                .select_from(OrganizationMembershipRow)
                .where(
                    OrganizationMembershipRow.organization_id
                    == organization_id,
                    OrganizationMembershipRow.enabled.is_(True),
                    OrganizationMembershipRow.role
                    == OrganizationRole.ADMINISTRATOR.value,
                )
            ).scalar_one()
        )


def _organization_to_row(organization: Organization) -> OrganizationRow:
    return OrganizationRow(
        organization_id=organization.organization_id,
        slug=organization.slug,
        name=organization.name,
        enabled=organization.enabled,
        created_at=organization.created_at,
        updated_at=organization.updated_at,
    )


def _organization_from_row(row: OrganizationRow) -> Organization:
    return Organization(
        organization_id=row.organization_id,
        slug=row.slug,
        name=row.name,
        enabled=row.enabled,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _membership_to_row(
    membership: OrganizationMembership,
) -> OrganizationMembershipRow:
    return OrganizationMembershipRow(
        membership_id=membership.membership_id,
        organization_id=membership.organization_id,
        user_id=membership.user_id,
        role=membership.role.value,
        enabled=membership.enabled,
        created_at=membership.created_at,
        updated_at=membership.updated_at,
    )


def _membership_from_row(
    row: OrganizationMembershipRow,
) -> OrganizationMembership:
    return OrganizationMembership(
        membership_id=row.membership_id,
        organization_id=row.organization_id,
        user_id=row.user_id,
        role=OrganizationRole(row.role),
        enabled=row.enabled,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
