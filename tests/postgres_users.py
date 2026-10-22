"""Scoped canonical User fixtures for PostgreSQL API tests."""

from datetime import datetime, timezone

from robotics.runs.db_models import UserRow


def seed_test_user(factory, user_id):
    with factory() as session:
        if session.get(UserRow, user_id) is None:
            now = datetime.now(timezone.utc)
            session.add(
                UserRow(
                    user_id=user_id,
                    username=user_id,
                    display_name=user_id,
                    password_hash="test-only",
                    enabled=True,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.commit()
            return True
    return False


def delete_test_user(factory, user_id):
    with factory() as session:
        row = session.get(UserRow, user_id)
        if row is not None:
            session.delete(row)
            session.commit()
