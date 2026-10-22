"""Create canonical S1 Push teaching Projects.

Revision ID: 20260831_0007
Revises: 20260831_0006
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260831_0007"
down_revision = "20260831_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "teaching_projects",
        sa.Column("project_id", sa.String(length=255), nullable=False),
        sa.Column("course_id", sa.String(length=255), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("description", sa.String(length=4000), nullable=False),
        sa.Column("instructions", sa.String(length=16000), nullable=False),
        sa.Column("starter_code", sa.Text(), nullable=False),
        sa.Column("starter_code_sha256", sa.String(length=64), nullable=False),
        sa.Column("training_profile", sa.String(length=32), nullable=False),
        sa.Column("video_enabled", sa.Boolean(), nullable=False),
        sa.Column("parameter_schema", postgresql.JSONB(), nullable=False),
        sa.Column("evaluation_rubric", postgresql.JSONB(), nullable=False),
        sa.Column("config_version", sa.BigInteger(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("created_by_user_id", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "training_profile IN ('short', 'standard')",
            name="training_profile",
        ),
        sa.CheckConstraint(
            "config_version >= 1",
            name="config_version_positive",
        ),
        sa.CheckConstraint(
            "octet_length(starter_code) <= 524288",
            name="starter_code_size",
        ),
        sa.CheckConstraint(
            "starter_code_sha256 ~ '^[0-9a-f]{64}$'",
            name="starter_code_sha256",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(parameter_schema) = 'object'",
            name="parameter_schema_object",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(evaluation_rubric) = 'object'",
            name="evaluation_rubric_object",
        ),
        sa.ForeignKeyConstraint(
            ["course_id"],
            ["courses.course_id"],
            name="fk_teaching_projects_course_id_courses",
        ),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"],
            ["users.user_id"],
            name="fk_teaching_projects_created_by_user_id_users",
        ),
        sa.PrimaryKeyConstraint("project_id", name="pk_teaching_projects"),
        sa.UniqueConstraint(
            "course_id", "title", name="uq_teaching_projects_course_title"
        ),
    )
    op.create_index(
        "ix_teaching_projects_course_enabled",
        "teaching_projects",
        ["course_id", "enabled"],
    )
    op.create_index(
        "ix_teaching_projects_created_by_user_id",
        "teaching_projects",
        ["created_by_user_id"],
    )
    op.create_index(
        "ix_teaching_projects_course_updated_at",
        "teaching_projects",
        ["course_id", "updated_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_teaching_projects_course_updated_at",
        table_name="teaching_projects",
    )
    op.drop_index(
        "ix_teaching_projects_created_by_user_id",
        table_name="teaching_projects",
    )
    op.drop_index(
        "ix_teaching_projects_course_enabled",
        table_name="teaching_projects",
    )
    op.drop_table("teaching_projects")
