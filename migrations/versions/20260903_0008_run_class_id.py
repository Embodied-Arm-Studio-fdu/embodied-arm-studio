"""Associate new teaching Runs with a Class while preserving legacy Runs.

Revision ID: 20260903_0008
Revises: 20260831_0007
"""

from alembic import op
import sqlalchemy as sa


revision = "20260903_0008"
down_revision = "20260831_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "runs", sa.Column("class_id", sa.String(length=255), nullable=True)
    )
    op.create_foreign_key(
        "fk_runs_class_id_classes",
        "runs",
        "classes",
        ["class_id"],
        ["class_id"],
    )
    op.create_index("ix_runs_class_id", "runs", ["class_id"])


def downgrade() -> None:
    op.drop_index("ix_runs_class_id", table_name="runs")
    op.drop_constraint(
        "fk_runs_class_id_classes", "runs", type_="foreignkey"
    )
    op.drop_column("runs", "class_id")
