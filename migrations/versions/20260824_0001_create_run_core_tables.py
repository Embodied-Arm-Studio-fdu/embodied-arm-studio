"""Create the canonical Run, Event, and idempotency tables."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260824_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "runs",
        sa.Column("run_id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("project_id", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("execution_mode", sa.String(length=32), nullable=False),
        sa.Column("run_kind", sa.String(length=32), nullable=False),
        sa.Column("parameters", postgresql.JSONB(), nullable=False),
        sa.Column("project_config_snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("code_sha256", sa.String(length=64), nullable=False),
        sa.Column("simulation_task_id", sa.String(length=255), nullable=True),
        sa.Column("progress", sa.Float(), nullable=False),
        sa.Column("metrics_summary", postgresql.JSONB(), nullable=False),
        sa.Column("error", postgresql.JSONB(), nullable=True),
        sa.Column("last_sequence", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("run_id", name="pk_runs"),
        sa.UniqueConstraint(
            "simulation_task_id", name="uq_runs_simulation_task_id"
        ),
    )
    op.create_index("ix_runs_created_at", "runs", ["created_at"], unique=False)

    op.create_table(
        "run_events",
        sa.Column("event_id", sa.String(length=255), nullable=False),
        sa.Column("run_id", sa.String(length=255), nullable=False),
        sa.Column("sequence", sa.BigInteger(), nullable=False),
        sa.Column("type", sa.String(length=32), nullable=False),
        sa.Column("task_id", sa.String(length=255), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("schema_version", sa.String(length=32), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"], ["runs.run_id"], name="fk_run_events_run_id_runs"
        ),
        sa.PrimaryKeyConstraint("event_id", name="pk_run_events"),
        sa.UniqueConstraint(
            "run_id", "sequence", name="uq_run_events_run_id_sequence"
        ),
    )
    op.create_index(
        "ix_run_events_run_id", "run_events", ["run_id"], unique=False
    )

    op.create_table(
        "run_idempotency_keys",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.run_id"],
            name="fk_run_idempotency_keys_run_id_runs",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_run_idempotency_keys"),
        sa.UniqueConstraint(
            "user_id",
            "idempotency_key",
            name="uq_run_idempotency_keys_user_key",
        ),
    )
    op.create_index(
        "ix_run_idempotency_keys_run_id",
        "run_idempotency_keys",
        ["run_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_run_idempotency_keys_run_id", table_name="run_idempotency_keys"
    )
    op.drop_table("run_idempotency_keys")
    op.drop_index("ix_run_events_run_id", table_name="run_events")
    op.drop_table("run_events")
    op.drop_index("ix_runs_created_at", table_name="runs")
    op.drop_table("runs")
