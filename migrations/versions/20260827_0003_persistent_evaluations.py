"""Add immutable canonical Evaluation reports."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260827_0003"
down_revision = "20260827_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "evaluation_reports",
        sa.Column("evaluation_id", sa.String(length=255), nullable=False),
        sa.Column("run_id", sa.String(length=255), nullable=False),
        sa.Column("requested_by_user_id", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("grade", sa.String(length=64), nullable=False),
        sa.Column("rubric_version", sa.String(length=255), nullable=False),
        sa.Column("rubric_snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("evidence", postgresql.JSONB(), nullable=False),
        sa.Column("evidence_sha256", sa.String(length=64), nullable=False),
        sa.Column("summary", postgresql.JSONB(), nullable=False),
        sa.Column("limitations", postgresql.JSONB(), nullable=False),
        sa.Column("evaluator_version", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.run_id"],
            name="fk_evaluation_reports_run_id_runs",
        ),
        sa.PrimaryKeyConstraint(
            "evaluation_id", name="pk_evaluation_reports"
        ),
        sa.UniqueConstraint(
            "run_id",
            "rubric_version",
            "evaluator_version",
            "evidence_sha256",
            name="uq_evaluation_reports_fingerprint",
        ),
    )
    op.create_index(
        "ix_evaluation_reports_run_id",
        "evaluation_reports",
        ["run_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_evaluation_reports_run_id", table_name="evaluation_reports"
    )
    op.drop_table("evaluation_reports")
