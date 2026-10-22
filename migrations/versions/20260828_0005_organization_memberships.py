"""add organizations and memberships

Revision ID: 20260828_0005
Revises: 20260828_0004
Create Date: 2026-08-28
"""

from alembic import op
import sqlalchemy as sa


revision = "20260828_0005"
down_revision = "20260828_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "organizations",
        sa.Column("organization_id", sa.String(length=255), nullable=False),
        sa.Column("slug", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("organization_id", name=op.f("pk_organizations")),
        sa.UniqueConstraint("slug", name="uq_organizations_slug"),
    )
    op.create_index(
        "ix_organizations_enabled", "organizations", ["enabled"], unique=False
    )
    op.create_index(
        "ix_organizations_created_at",
        "organizations",
        ["created_at"],
        unique=False,
    )
    op.create_table(
        "organization_memberships",
        sa.Column("membership_id", sa.String(length=255), nullable=False),
        sa.Column("organization_id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "role IN ('student', 'instructor', 'administrator')",
            name=op.f("ck_organization_memberships_role"),
        ),
        sa.ForeignKeyConstraint(
            ["organization_id"],
            ["organizations.organization_id"],
            name=(
                "fk_organization_memberships_organization_id_organizations"
            ),
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.user_id"],
            name="fk_organization_memberships_user_id_users",
        ),
        sa.PrimaryKeyConstraint(
            "membership_id", name=op.f("pk_organization_memberships")
        ),
        sa.UniqueConstraint(
            "organization_id",
            "user_id",
            name="uq_organization_memberships_organization_user",
        ),
    )
    op.create_index(
        "ix_organization_memberships_user_id",
        "organization_memberships",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        "ix_organization_memberships_organization_enabled",
        "organization_memberships",
        ["organization_id", "enabled"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_organization_memberships_organization_enabled",
        table_name="organization_memberships",
    )
    op.drop_index(
        "ix_organization_memberships_user_id",
        table_name="organization_memberships",
    )
    op.drop_table("organization_memberships")
    op.drop_index("ix_organizations_created_at", table_name="organizations")
    op.drop_index("ix_organizations_enabled", table_name="organizations")
    op.drop_table("organizations")
