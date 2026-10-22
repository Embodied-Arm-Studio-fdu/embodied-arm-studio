"""PostgreSQL repository for canonical users."""

from typing import Callable, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from robotics.runs.db_models import UserRow

from .models import User, utc_now


class UserRepositoryError(RuntimeError):
    pass


class UserNotFound(UserRepositoryError, LookupError):
    pass


class DuplicateUsername(UserRepositoryError):
    pass


class PostgreSQLUserRepository:
    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def create(self, user: User) -> User:
        session = self._session_factory()
        try:
            session.add(_to_row(user))
            session.commit()
            return user
        except IntegrityError as exc:
            session.rollback()
            raise DuplicateUsername("username already exists") from exc
        except SQLAlchemyError as exc:
            session.rollback()
            raise UserRepositoryError("could not create user") from exc
        finally:
            session.close()

    def get_by_id(self, user_id: str) -> User:
        return self._get(UserRow.user_id == user_id)

    def get_by_username(self, username: str) -> User:
        return self._get(UserRow.username == username)

    def find_by_username(self, username: str) -> Optional[User]:
        try:
            return self.get_by_username(username)
        except UserNotFound:
            return None

    def update_password_hash(self, user_id: str, password_hash: str) -> User:
        session = self._session_factory()
        try:
            row = session.get(UserRow, user_id)
            if row is None:
                raise UserNotFound("user not found")
            row.password_hash = password_hash
            row.updated_at = utc_now()
            session.commit()
            session.refresh(row)
            return _from_row(row)
        except UserNotFound:
            raise
        except SQLAlchemyError as exc:
            session.rollback()
            raise UserRepositoryError("could not update user password") from exc
        finally:
            session.close()

    def _get(self, condition) -> User:
        session = self._session_factory()
        try:
            row = session.execute(select(UserRow).where(condition)).scalar_one_or_none()
            if row is None:
                raise UserNotFound("user not found")
            return _from_row(row)
        except UserNotFound:
            raise
        except SQLAlchemyError as exc:
            raise UserRepositoryError("could not read user") from exc
        finally:
            session.close()


def _to_row(user: User) -> UserRow:
    return UserRow(
        user_id=user.user_id,
        username=user.username,
        display_name=user.display_name,
        password_hash=user.password_hash,
        enabled=user.enabled,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


def _from_row(row: UserRow) -> User:
    return User(
        user_id=row.user_id,
        username=row.username,
        display_name=row.display_name,
        password_hash=row.password_hash,
        enabled=row.enabled,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
