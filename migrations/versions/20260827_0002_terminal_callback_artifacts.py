"""Add terminal Callback facts and canonical Run Artifact metadata."""

from alembic import op
import sqlalchemy as sa


revision = "20260827_0002"
down_revision = "20260824_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column(
            "training_steps",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )
    op.add_column(
        "runs",
        sa.Column(
            "update_steps",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )
    op.add_column(
        "runs", sa.Column("stop_reason", sa.String(length=255), nullable=True)
    )
    op.alter_column("runs", "training_steps", server_default=None)
    op.alter_column("runs", "update_steps", server_default=None)

    op.create_table(
        "run_artifacts",
        sa.Column("artifact_id", sa.String(length=255), nullable=False),
        sa.Column("run_id", sa.String(length=255), nullable=False),
        sa.Column("artifact_type", sa.String(length=255), nullable=False),
        sa.Column("path", sa.String(length=1024), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("is_primary_model", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.run_id"], name="fk_run_artifacts_run_id_runs"
        ),
        sa.PrimaryKeyConstraint("artifact_id", name="pk_run_artifacts"),
        sa.UniqueConstraint(
            "run_id", "path", name="uq_run_artifacts_run_id_path"
        ),
    )
    op.create_index(
        "ix_run_artifacts_run_id", "run_artifacts", ["run_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_run_artifacts_run_id", table_name="run_artifacts")
    op.drop_table("run_artifacts")
    op.drop_column("runs", "stop_reason")
    op.drop_column("runs", "update_steps")
    op.drop_column("runs", "training_steps")
