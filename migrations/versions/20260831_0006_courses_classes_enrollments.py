"""Create courses, teaching classes, and enrollments.

Revision ID: 20260831_0006
Revises: 20260828_0005
"""

from alembic import op
import sqlalchemy as sa


revision = "20260831_0006"
down_revision = "20260828_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "courses",
        sa.Column("course_id", sa.String(length=255), nullable=False),
        sa.Column("organization_id", sa.String(length=255), nullable=False),
        sa.Column("code", sa.String(length=255), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("description", sa.String(length=4000), nullable=False),
        sa.Column("owner_instructor_user_id", sa.String(length=255), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.organization_id"], name="fk_courses_organization_id_organizations"),
        sa.ForeignKeyConstraint(["owner_instructor_user_id"], ["users.user_id"], name="fk_courses_owner_instructor_user_id_users"),
        sa.PrimaryKeyConstraint("course_id", name="pk_courses"),
        sa.UniqueConstraint("organization_id", "code", name="uq_courses_organization_code"),
    )
    op.create_index("ix_courses_organization_enabled", "courses", ["organization_id", "enabled"])
    op.create_index("ix_courses_owner_instructor_user_id", "courses", ["owner_instructor_user_id"])
    op.create_table(
        "classes",
        sa.Column("class_id", sa.String(length=255), nullable=False),
        sa.Column("course_id", sa.String(length=255), nullable=False),
        sa.Column("code", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("term", sa.String(length=255), nullable=False),
        sa.Column("instructor_user_id", sa.String(length=255), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["course_id"], ["courses.course_id"], name="fk_classes_course_id_courses"),
        sa.ForeignKeyConstraint(["instructor_user_id"], ["users.user_id"], name="fk_classes_instructor_user_id_users"),
        sa.PrimaryKeyConstraint("class_id", name="pk_classes"),
        sa.UniqueConstraint("course_id", "term", "code", name="uq_classes_course_term_code"),
    )
    op.create_index("ix_classes_course_enabled", "classes", ["course_id", "enabled"])
    op.create_index("ix_classes_instructor_user_id", "classes", ["instructor_user_id"])
    op.create_table(
        "enrollments",
        sa.Column("enrollment_id", sa.String(length=255), nullable=False),
        sa.Column("class_id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["class_id"], ["classes.class_id"], name="fk_enrollments_class_id_classes"),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], name="fk_enrollments_user_id_users"),
        sa.PrimaryKeyConstraint("enrollment_id", name="pk_enrollments"),
        sa.UniqueConstraint("class_id", "user_id", name="uq_enrollments_class_user"),
    )
    op.create_index("ix_enrollments_class_enabled", "enrollments", ["class_id", "enabled"])
    op.create_index("ix_enrollments_user_enabled", "enrollments", ["user_id", "enabled"])


def downgrade() -> None:
    op.drop_index("ix_enrollments_user_enabled", table_name="enrollments")
    op.drop_index("ix_enrollments_class_enabled", table_name="enrollments")
    op.drop_table("enrollments")
    op.drop_index("ix_classes_instructor_user_id", table_name="classes")
    op.drop_index("ix_classes_course_enabled", table_name="classes")
    op.drop_table("classes")
    op.drop_index("ix_courses_owner_instructor_user_id", table_name="courses")
    op.drop_index("ix_courses_organization_enabled", table_name="courses")
    op.drop_table("courses")
