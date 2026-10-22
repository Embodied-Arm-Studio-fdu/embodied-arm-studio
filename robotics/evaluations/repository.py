"""PostgreSQL persistence for immutable Evaluation reports."""

from copy import deepcopy
from datetime import timezone
from typing import Callable, List, Protocol, Tuple

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from robotics.evaluations.models import EvaluationReport, EvaluationStatus
from robotics.runs.db_models import EvaluationReportRow


class EvaluationRepositoryError(RuntimeError):
    """A redacted Evaluation persistence failure."""


class EvaluationReportNotFound(EvaluationRepositoryError, LookupError):
    """The requested Evaluation does not exist."""


class EvaluationRepository(Protocol):
    def create_or_get(self, report: EvaluationReport) -> Tuple[EvaluationReport, bool]:
        ...

    def get(self, evaluation_id: str) -> EvaluationReport:
        ...

    def list_for_run(self, run_id: str) -> List[EvaluationReport]:
        ...


class PostgreSQLEvaluationRepository:
    """Store append-only reports and resolve concurrent idempotent creates."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def create_or_get(
        self, report: EvaluationReport
    ) -> Tuple[EvaluationReport, bool]:
        session = self._session_factory()
        try:
            with session.begin():
                existing = self._find_fingerprint(session, report)
                if existing is not None:
                    return self._from_row(existing), True
                session.add(self._to_row(report))
                session.flush()
            return report, False
        except IntegrityError:
            session.rollback()
            return self._get_after_conflict(report)
        except (SQLAlchemyError, ValueError) as exc:
            raise EvaluationRepositoryError(
                "Evaluation persistence is unavailable"
            ) from exc
        finally:
            session.close()

    def get(self, evaluation_id: str) -> EvaluationReport:
        session = self._session_factory()
        try:
            row = session.get(EvaluationReportRow, evaluation_id)
            if row is None:
                raise EvaluationReportNotFound()
            return self._from_row(row)
        except EvaluationReportNotFound:
            raise
        except (SQLAlchemyError, ValueError) as exc:
            raise EvaluationRepositoryError(
                "Evaluation lookup is unavailable"
            ) from exc
        finally:
            session.close()

    def list_for_run(self, run_id: str) -> List[EvaluationReport]:
        """Return immutable reports newest-first with a stable tie-breaker."""
        session = self._session_factory()
        try:
            rows = session.execute(
                select(EvaluationReportRow)
                .where(EvaluationReportRow.run_id == run_id)
                .order_by(
                    EvaluationReportRow.created_at.desc(),
                    EvaluationReportRow.evaluation_id.desc(),
                )
            ).scalars().all()
            return [self._from_row(row) for row in rows]
        except (SQLAlchemyError, ValueError) as exc:
            raise EvaluationRepositoryError(
                "Evaluation lookup is unavailable"
            ) from exc
        finally:
            session.close()

    def _get_after_conflict(
        self, report: EvaluationReport
    ) -> Tuple[EvaluationReport, bool]:
        session = self._session_factory()
        try:
            row = self._find_fingerprint(session, report)
            if row is None:
                raise EvaluationRepositoryError(
                    "Evaluation persistence is unavailable"
                )
            return self._from_row(row), True
        except EvaluationRepositoryError:
            raise
        except (SQLAlchemyError, ValueError) as exc:
            raise EvaluationRepositoryError(
                "Evaluation persistence is unavailable"
            ) from exc
        finally:
            session.close()

    @staticmethod
    def _find_fingerprint(session: Session, report: EvaluationReport):
        return session.execute(
            select(EvaluationReportRow).where(
                EvaluationReportRow.run_id == report.run_id,
                EvaluationReportRow.rubric_version == report.rubric_version,
                EvaluationReportRow.evaluator_version == report.evaluator_version,
                EvaluationReportRow.evidence_sha256 == report.evidence_sha256,
            )
        ).scalar_one_or_none()

    @staticmethod
    def _to_row(report: EvaluationReport) -> EvaluationReportRow:
        return EvaluationReportRow(
            evaluation_id=report.evaluation_id,
            run_id=report.run_id,
            requested_by_user_id=report.requested_by_user_id,
            status=report.status.value,
            score=report.score,
            grade=report.grade,
            rubric_version=report.rubric_version,
            rubric_snapshot=deepcopy(report.rubric_snapshot),
            evidence=deepcopy(report.evidence),
            evidence_sha256=report.evidence_sha256,
            summary=deepcopy(report.summary),
            limitations=deepcopy(report.limitations),
            evaluator_version=report.evaluator_version,
            created_at=report.created_at,
        )

    @staticmethod
    def _from_row(row: EvaluationReportRow) -> EvaluationReport:
        created_at = row.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        else:
            created_at = created_at.astimezone(timezone.utc)
        return EvaluationReport(
            evaluation_id=row.evaluation_id,
            run_id=row.run_id,
            requested_by_user_id=row.requested_by_user_id,
            status=EvaluationStatus(row.status),
            score=row.score,
            grade=row.grade,
            rubric_version=row.rubric_version,
            rubric_snapshot=deepcopy(row.rubric_snapshot),
            evidence=deepcopy(row.evidence),
            evidence_sha256=row.evidence_sha256,
            summary=deepcopy(row.summary),
            limitations=deepcopy(row.limitations),
            evaluator_version=row.evaluator_version,
            created_at=created_at,
        )
