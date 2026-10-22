"""SQLAlchemy table models for the PostgreSQL canonical Run store."""

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    MetaData,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base


NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

Base = declarative_base(metadata=MetaData(naming_convention=NAMING_CONVENTION))


class RunRow(Base):
    __tablename__ = "runs"
    __table_args__ = (
        UniqueConstraint(
            "simulation_task_id", name="uq_runs_simulation_task_id"
        ),
        Index("ix_runs_created_at", "created_at"),
        Index("ix_runs_class_id", "class_id"),
    )

    run_id = Column(String(255), primary_key=True)
    user_id = Column(String(255), nullable=False)
    project_id = Column(String(255), nullable=False)
    class_id = Column(
        String(255),
        ForeignKey("classes.class_id", name="fk_runs_class_id_classes"),
        nullable=True,
    )
    status = Column(String(32), nullable=False)
    execution_mode = Column(String(32), nullable=False)
    run_kind = Column(String(32), nullable=False)
    parameters = Column(JSONB, nullable=False)
    project_config_snapshot = Column(JSONB, nullable=False)
    code_sha256 = Column(String(64), nullable=False)
    simulation_task_id = Column(String(255), nullable=True)
    progress = Column(Float, nullable=False)
    metrics_summary = Column(JSONB, nullable=False)
    error = Column(JSONB, nullable=True)
    last_sequence = Column(BigInteger, nullable=False)
    training_steps = Column(BigInteger, nullable=False)
    update_steps = Column(BigInteger, nullable=False)
    stop_reason = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False)


class RunEventRow(Base):
    __tablename__ = "run_events"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "sequence", name="uq_run_events_run_id_sequence"
        ),
        Index("ix_run_events_run_id", "run_id"),
    )

    event_id = Column(String(255), primary_key=True)
    run_id = Column(
        String(255),
        ForeignKey("runs.run_id", name="fk_run_events_run_id_runs"),
        nullable=False,
    )
    sequence = Column(BigInteger, nullable=False)
    type = Column(String(32), nullable=False)
    task_id = Column(String(255), nullable=True)
    occurred_at = Column(DateTime(timezone=True), nullable=False)
    schema_version = Column(String(32), nullable=False)
    payload = Column(JSONB, nullable=False)


class RunIdempotencyKeyRow(Base):
    __tablename__ = "run_idempotency_keys"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "idempotency_key",
            name="uq_run_idempotency_keys_user_key",
        ),
        Index("ix_run_idempotency_keys_run_id", "run_id"),
    )

    id = Column(BigInteger, Identity(), primary_key=True)
    user_id = Column(String(255), nullable=False)
    idempotency_key = Column(String(255), nullable=False)
    request_hash = Column(String(64), nullable=False)
    run_id = Column(
        String(255),
        ForeignKey("runs.run_id", name="fk_run_idempotency_keys_run_id_runs"),
        nullable=False,
    )
    created_at = Column(DateTime(timezone=True), nullable=False)


class RunArtifactRow(Base):
    __tablename__ = "run_artifacts"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "path", name="uq_run_artifacts_run_id_path"
        ),
        Index("ix_run_artifacts_run_id", "run_id"),
    )

    artifact_id = Column(String(255), primary_key=True)
    run_id = Column(
        String(255),
        ForeignKey("runs.run_id", name="fk_run_artifacts_run_id_runs"),
        nullable=False,
    )
    artifact_type = Column(String(255), nullable=False)
    path = Column(String(1024), nullable=False)
    size_bytes = Column(BigInteger, nullable=True)
    sha256 = Column(String(64), nullable=True)
    is_primary_model = Column(Boolean, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)


class EvaluationReportRow(Base):
    __tablename__ = "evaluation_reports"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "rubric_version",
            "evaluator_version",
            "evidence_sha256",
            name="uq_evaluation_reports_fingerprint",
        ),
        Index("ix_evaluation_reports_run_id", "run_id"),
    )

    evaluation_id = Column(String(255), primary_key=True)
    run_id = Column(
        String(255),
        ForeignKey("runs.run_id", name="fk_evaluation_reports_run_id_runs"),
        nullable=False,
    )
    requested_by_user_id = Column(String(255), nullable=False)
    status = Column(String(32), nullable=False)
    score = Column(Float, nullable=True)
    grade = Column(String(64), nullable=False)
    rubric_version = Column(String(255), nullable=False)
    rubric_snapshot = Column(JSONB, nullable=False)
    evidence = Column(JSONB, nullable=False)
    evidence_sha256 = Column(String(64), nullable=False)
    summary = Column(JSONB, nullable=False)
    limitations = Column(JSONB, nullable=False)
    evaluator_version = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)


class UserRow(Base):
    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("username", name="uq_users_username"),
        Index("ix_users_created_at", "created_at"),
    )

    user_id = Column(String(255), primary_key=True)
    username = Column(String(255), nullable=False)
    display_name = Column(String(255), nullable=False)
    password_hash = Column(String(1024), nullable=False)
    enabled = Column(Boolean, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)


class OrganizationRow(Base):
    __tablename__ = "organizations"
    __table_args__ = (
        UniqueConstraint("slug", name="uq_organizations_slug"),
        Index("ix_organizations_enabled", "enabled"),
        Index("ix_organizations_created_at", "created_at"),
    )

    organization_id = Column(String(255), primary_key=True)
    slug = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    enabled = Column(Boolean, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)


class OrganizationMembershipRow(Base):
    __tablename__ = "organization_memberships"
    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "user_id",
            name="uq_organization_memberships_organization_user",
        ),
        CheckConstraint(
            "role IN ('student', 'instructor', 'administrator')",
            name="role",
        ),
        Index("ix_organization_memberships_user_id", "user_id"),
        Index(
            "ix_organization_memberships_organization_enabled",
            "organization_id",
            "enabled",
        ),
    )

    membership_id = Column(String(255), primary_key=True)
    organization_id = Column(
        String(255),
        ForeignKey(
            "organizations.organization_id",
            name=(
                "fk_organization_memberships_organization_id_organizations"
            ),
        ),
        nullable=False,
    )
    user_id = Column(
        String(255),
        ForeignKey(
            "users.user_id",
            name="fk_organization_memberships_user_id_users",
        ),
        nullable=False,
    )
    role = Column(String(32), nullable=False)
    enabled = Column(Boolean, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)


class CourseRow(Base):
    __tablename__ = "courses"
    __table_args__ = (
        UniqueConstraint(
            "organization_id", "code", name="uq_courses_organization_code"
        ),
        Index("ix_courses_organization_enabled", "organization_id", "enabled"),
        Index("ix_courses_owner_instructor_user_id", "owner_instructor_user_id"),
    )

    course_id = Column(String(255), primary_key=True)
    organization_id = Column(
        String(255),
        ForeignKey("organizations.organization_id", name="fk_courses_organization_id_organizations"),
        nullable=False,
    )
    code = Column(String(255), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(String(4000), nullable=False)
    owner_instructor_user_id = Column(
        String(255),
        ForeignKey("users.user_id", name="fk_courses_owner_instructor_user_id_users"),
        nullable=False,
    )
    enabled = Column(Boolean, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)


class TeachingClassRow(Base):
    __tablename__ = "classes"
    __table_args__ = (
        UniqueConstraint(
            "course_id", "term", "code", name="uq_classes_course_term_code"
        ),
        Index("ix_classes_course_enabled", "course_id", "enabled"),
        Index("ix_classes_instructor_user_id", "instructor_user_id"),
    )

    class_id = Column(String(255), primary_key=True)
    course_id = Column(
        String(255),
        ForeignKey("courses.course_id", name="fk_classes_course_id_courses"),
        nullable=False,
    )
    code = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    term = Column(String(255), nullable=False)
    instructor_user_id = Column(
        String(255),
        ForeignKey("users.user_id", name="fk_classes_instructor_user_id_users"),
        nullable=False,
    )
    enabled = Column(Boolean, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)


class EnrollmentRow(Base):
    __tablename__ = "enrollments"
    __table_args__ = (
        UniqueConstraint("class_id", "user_id", name="uq_enrollments_class_user"),
        Index("ix_enrollments_class_enabled", "class_id", "enabled"),
        Index("ix_enrollments_user_enabled", "user_id", "enabled"),
    )

    enrollment_id = Column(String(255), primary_key=True)
    class_id = Column(
        String(255),
        ForeignKey("classes.class_id", name="fk_enrollments_class_id_classes"),
        nullable=False,
    )
    user_id = Column(
        String(255),
        ForeignKey("users.user_id", name="fk_enrollments_user_id_users"),
        nullable=False,
    )
    enabled = Column(Boolean, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)


class TeachingProjectRow(Base):
    __tablename__ = "teaching_projects"
    __table_args__ = (
        UniqueConstraint(
            "course_id", "title", name="uq_teaching_projects_course_title"
        ),
        CheckConstraint(
            "training_profile IN ('short', 'standard')",
            name="training_profile",
        ),
        CheckConstraint("config_version >= 1", name="config_version_positive"),
        CheckConstraint(
            "octet_length(starter_code) <= 524288",
            name="starter_code_size",
        ),
        CheckConstraint(
            "starter_code_sha256 ~ '^[0-9a-f]{64}$'",
            name="starter_code_sha256",
        ),
        CheckConstraint(
            "jsonb_typeof(parameter_schema) = 'object'",
            name="parameter_schema_object",
        ),
        CheckConstraint(
            "jsonb_typeof(evaluation_rubric) = 'object'",
            name="evaluation_rubric_object",
        ),
        Index(
            "ix_teaching_projects_course_enabled", "course_id", "enabled"
        ),
        Index(
            "ix_teaching_projects_created_by_user_id", "created_by_user_id"
        ),
        Index(
            "ix_teaching_projects_course_updated_at", "course_id", "updated_at"
        ),
    )

    project_id = Column(String(255), primary_key=True)
    course_id = Column(
        String(255),
        ForeignKey(
            "courses.course_id",
            name="fk_teaching_projects_course_id_courses",
        ),
        nullable=False,
    )
    title = Column(String(255), nullable=False)
    description = Column(String(4000), nullable=False)
    instructions = Column(String(16000), nullable=False)
    starter_code = Column(Text, nullable=False)
    starter_code_sha256 = Column(String(64), nullable=False)
    training_profile = Column(String(32), nullable=False)
    video_enabled = Column(Boolean, nullable=False)
    parameter_schema = Column(JSONB, nullable=False)
    evaluation_rubric = Column(JSONB, nullable=False)
    config_version = Column(BigInteger, nullable=False)
    enabled = Column(Boolean, nullable=False)
    created_by_user_id = Column(
        String(255),
        ForeignKey(
            "users.user_id",
            name="fk_teaching_projects_created_by_user_id_users",
        ),
        nullable=False,
    )
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)
