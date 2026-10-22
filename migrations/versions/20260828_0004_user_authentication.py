"""add canonical users

Revision ID: 20260828_0004
Revises: 20260827_0003
Create Date: 2026-08-28
"""

from alembic import op
import sqlalchemy as sa


revision = "20260828_0004"
down_revision = "20260827_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("username", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.String(length=1024), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("user_id", name=op.f("pk_users")),
        sa.UniqueConstraint("username", name="uq_users_username"),
    )
    op.create_index("ix_users_created_at", "users", ["created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_users_created_at", table_name="users")
    op.drop_table("users")
