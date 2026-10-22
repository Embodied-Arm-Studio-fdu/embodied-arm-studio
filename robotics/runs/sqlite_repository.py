"""SQLite-backed implementation of the canonical Run repository."""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from robotics.runs.domain import RunStatus
from robotics.runs.models import (
    ExecutionMode,
    Run,
    RunError,
    RunEvent,
    RunEventType,
    RunKind,
)
from robotics.runs.repository import (
    AppendEventResult,
    EventConflict,
    EventSequenceConflict,
    RepositoryDataError,
    RepositoryError,
    RunAlreadyExists,
    RunNotFound,
    TaskIdAlreadyBound,
    TaskIdConflict,
    dispatch_error_update,
)


_SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    status TEXT NOT NULL,
    execution_mode TEXT NOT NULL,
    run_kind TEXT NOT NULL,
    parameters TEXT NOT NULL,
    project_config_snapshot TEXT NOT NULL,
    code_sha256 TEXT NOT NULL,
    simulation_task_id TEXT UNIQUE,
    progress REAL NOT NULL,
    metrics_summary TEXT NOT NULL,
    error TEXT,
    last_sequence INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    ended_at TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    type TEXT NOT NULL,
    task_id TEXT,
    occurred_at TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    payload TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    UNIQUE (run_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_run_events_run_sequence
    ON run_events (run_id, sequence);
"""


class SQLiteRunRepository:
    """Single-file persistent repository using short-lived SQLite connections."""

    def __init__(
        self,
        database_path: Union[str, os.PathLike],
        busy_timeout_ms: int = 5000,
    ) -> None:
        self._database_path = os.fspath(database_path)
        if self._database_path == ":memory:":
            raise ValueError("SQLiteRunRepository requires a persistent file path")
        if not isinstance(busy_timeout_ms, int) or busy_timeout_ms <= 0:
            raise ValueError("busy_timeout_ms must be a positive integer")
        self._busy_timeout_ms = busy_timeout_ms
        self._initialize()

    @property
    def database_path(self) -> Path:
        return Path(self._database_path)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self._database_path,
            timeout=self._busy_timeout_ms / 1000.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = %d" % self._busy_timeout_ms)
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    def _initialize(self) -> None:
        with self._connection() as connection:
            mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if str(mode).lower() != "wal":
                raise RepositoryError("SQLite WAL mode could not be enabled")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in (0, _SCHEMA_VERSION):
                raise RepositoryDataError(
                    "Unsupported Run repository schema version: %s" % version
                )
            connection.executescript(_SCHEMA)
            if version == 0:
                connection.execute("PRAGMA user_version = %d" % _SCHEMA_VERSION)

    def create_run(self, run: Run) -> Run:
        with self._transaction() as connection:
            if self._select_run(connection, run.run_id) is not None:
                raise RunAlreadyExists(run.run_id)
            try:
                connection.execute(
                    """
                    INSERT INTO runs (
                        run_id, user_id, project_id, status, execution_mode,
                        run_kind, parameters, project_config_snapshot,
                        code_sha256, simulation_task_id, progress,
                        metrics_summary, error, last_sequence, created_at,
                        started_at, ended_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    self._run_values(run),
                )
            except sqlite3.IntegrityError as exc:
                self._raise_run_integrity_error(exc, run)
            return self._row_to_run(self._select_run(connection, run.run_id))

    def get_run(self, run_id: str) -> Run:
        with self._connection() as connection:
            row = self._select_run(connection, run_id)
            if row is None:
                raise RunNotFound(run_id)
            return self._row_to_run(row)

    def list_runs(self) -> List[Run]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC, rowid ASC"
            ).fetchall()
            return [self._row_to_run(row) for row in rows]

    def save_run(self, run: Run) -> Run:
        with self._transaction() as connection:
            if self._select_run(connection, run.run_id) is None:
                raise RunNotFound(run.run_id)
            try:
                connection.execute(
                    """
                    UPDATE runs SET
                        user_id = ?, project_id = ?, status = ?,
                        execution_mode = ?, run_kind = ?, parameters = ?,
                        project_config_snapshot = ?, code_sha256 = ?,
                        simulation_task_id = ?, progress = ?, metrics_summary = ?,
                        error = ?, last_sequence = ?, created_at = ?,
                        started_at = ?, ended_at = ?, updated_at = ?
                    WHERE run_id = ?
                    """,
                    self._run_values(run)[1:] + (run.run_id,),
                )
            except sqlite3.IntegrityError as exc:
                self._raise_run_integrity_error(exc, run)
            return self._row_to_run(self._select_run(connection, run.run_id))

    def record_dispatch_error(self, run_id, error, now, definitive):
        with self._transaction() as connection:
            row = self._select_run(connection, run_id)
            if row is None:
                raise RunNotFound(run_id)
            current = self._row_to_run(row)
            if connection.execute("SELECT 1 FROM run_events WHERE run_id = ? LIMIT 1",
                                  (run_id,)).fetchone() is not None:
                return current
            updated, event = dispatch_error_update(current, error, now, definitive)
            connection.execute(
                "UPDATE runs SET status = ?, error = ?, ended_at = ?, "
                "updated_at = ?, last_sequence = ? WHERE run_id = ?",
                (updated.status.value, self._encode_error(updated.error),
                 self._encode_optional_datetime(updated.ended_at),
                 self._encode_datetime(updated.updated_at), updated.last_sequence, run_id),
            )
            if event is not None:
                connection.execute(
                    "INSERT INTO run_events (event_id, run_id, sequence, type, task_id, "
                    "occurred_at, schema_version, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (event.event_id, run_id, event.sequence, event.type.value, None,
                     self._encode_datetime(event.occurred_at), event.schema_version,
                     self._encode_json(event.payload)),
                )
            return self._row_to_run(self._select_run(connection, run_id))

    def bind_task_id(self, run_id: str, task_id: str) -> Run:
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        with self._transaction() as connection:
            row = self._select_run(connection, run_id)
            if row is None:
                raise RunNotFound(run_id)
            current = row["simulation_task_id"]
            if current is not None:
                if current != task_id:
                    raise TaskIdConflict(run_id, current, task_id)
                return self._row_to_run(row)
            error = self._row_to_run(row).error
            if error and error.code.startswith("DISPATCH_"):
                error = None
            try:
                connection.execute(
                    "UPDATE runs SET simulation_task_id = ?, error = ? WHERE run_id = ?",
                    (task_id, self._encode_error(error), run_id),
                )
            except sqlite3.IntegrityError as exc:
                if "simulation_task_id" in str(exc):
                    raise TaskIdAlreadyBound(task_id) from exc
                raise
            return self._row_to_run(self._select_run(connection, run_id))

    def append_event(self, event: RunEvent) -> AppendEventResult:
        with self._transaction() as connection:
            if self._select_run(connection, event.run_id) is None:
                raise RunNotFound(event.run_id)
            existing = connection.execute(
                "SELECT * FROM run_events WHERE event_id = ?", (event.event_id,)
            ).fetchone()
            if existing is not None:
                if self._row_to_event(existing) == event:
                    return AppendEventResult.DUPLICATE
                raise EventConflict(event.event_id)
            occupied = connection.execute(
                "SELECT event_id FROM run_events WHERE run_id = ? AND sequence = ?",
                (event.run_id, event.sequence),
            ).fetchone()
            if occupied is not None:
                raise EventSequenceConflict(event.run_id, event.sequence)
            connection.execute(
                """
                INSERT INTO run_events (
                    event_id, run_id, sequence, type, task_id, occurred_at,
                    schema_version, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.run_id,
                    event.sequence,
                    event.type.value,
                    event.task_id,
                    self._encode_datetime(event.occurred_at),
                    event.schema_version,
                    self._encode_json(event.payload),
                ),
            )
            return AppendEventResult.STORED

    def list_events(
        self,
        run_id: str,
        after_sequence: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[RunEvent]:
        threshold = after_sequence if after_sequence is not None else 0
        with self._connection() as connection:
            if self._select_run(connection, run_id) is None:
                raise RunNotFound(run_id)
            query = """
                SELECT * FROM run_events
                WHERE run_id = ? AND sequence > ?
                ORDER BY sequence ASC
            """
            parameters = [run_id, threshold]
            if limit is not None:
                query += " LIMIT ?"
                parameters.append(limit)
            rows = connection.execute(query, parameters).fetchall()
            return [self._row_to_event(row) for row in rows]

    @staticmethod
    def _select_run(
        connection: sqlite3.Connection, run_id: str
    ) -> Optional[sqlite3.Row]:
        return connection.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()

    def _run_values(self, run: Run) -> tuple:
        return (
            run.run_id,
            run.user_id,
            run.project_id,
            run.status.value,
            run.execution_mode.value,
            run.run_kind.value,
            self._encode_json(run.parameters),
            self._encode_json(run.project_config_snapshot),
            run.code_sha256,
            run.simulation_task_id,
            run.progress,
            self._encode_json(run.metrics_summary),
            self._encode_error(run.error),
            run.last_sequence,
            self._encode_datetime(run.created_at),
            self._encode_optional_datetime(run.started_at),
            self._encode_optional_datetime(run.ended_at),
            self._encode_datetime(run.updated_at),
        )

    @staticmethod
    def _encode_json(value: Dict[str, Any]) -> str:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _encode_error(self, error: Optional[RunError]) -> Optional[str]:
        if error is None:
            return None
        return self._encode_json(
            {
                "code": error.code,
                "message": error.message,
                "retryable": error.retryable,
            }
        )

    @staticmethod
    def _encode_datetime(value: datetime) -> str:
        return value.astimezone(timezone.utc).isoformat()

    def _encode_optional_datetime(self, value: Optional[datetime]) -> Optional[str]:
        return None if value is None else self._encode_datetime(value)

    def _row_to_run(self, row: sqlite3.Row) -> Run:
        run_id = row["run_id"] if row is not None else "<missing>"
        try:
            raw_error = self._decode_optional_json(row["error"], "error")
            error = RunError(**raw_error) if raw_error is not None else None
            return Run(
                run_id=run_id,
                user_id=row["user_id"],
                project_id=row["project_id"],
                status=RunStatus(row["status"]),
                execution_mode=ExecutionMode(row["execution_mode"]),
                run_kind=RunKind(row["run_kind"]),
                parameters=self._decode_json(row["parameters"], "parameters"),
                project_config_snapshot=self._decode_json(
                    row["project_config_snapshot"], "project_config_snapshot"
                ),
                code_sha256=row["code_sha256"],
                simulation_task_id=row["simulation_task_id"],
                progress=row["progress"],
                metrics_summary=self._decode_json(
                    row["metrics_summary"], "metrics_summary"
                ),
                error=error,
                last_sequence=row["last_sequence"],
                created_at=self._decode_datetime(row["created_at"], "created_at"),
                started_at=self._decode_optional_datetime(
                    row["started_at"], "started_at"
                ),
                ended_at=self._decode_optional_datetime(row["ended_at"], "ended_at"),
                updated_at=self._decode_datetime(row["updated_at"], "updated_at"),
            )
        except RepositoryDataError:
            raise
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise RepositoryDataError(
                "Invalid persisted Run %s: %s" % (run_id, exc)
            ) from exc

    def _row_to_event(self, row: sqlite3.Row) -> RunEvent:
        event_id = row["event_id"]
        try:
            return RunEvent(
                event_id=event_id,
                run_id=row["run_id"],
                sequence=row["sequence"],
                type=RunEventType(row["type"]),
                task_id=row["task_id"],
                occurred_at=self._decode_datetime(row["occurred_at"], "occurred_at"),
                schema_version=row["schema_version"],
                payload=self._decode_json(row["payload"], "payload"),
            )
        except RepositoryDataError:
            raise
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise RepositoryDataError(
                "Invalid persisted RunEvent %s: %s" % (event_id, exc)
            ) from exc

    @staticmethod
    def _decode_json(raw: str, field: str) -> Dict[str, Any]:
        try:
            value = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RepositoryDataError("Invalid JSON in %s" % field) from exc
        if not isinstance(value, dict):
            raise RepositoryDataError("Persisted %s must be a JSON object" % field)
        return value

    def _decode_optional_json(
        self, raw: Optional[str], field: str
    ) -> Optional[Dict[str, Any]]:
        return None if raw is None else self._decode_json(raw, field)

    @staticmethod
    def _decode_datetime(raw: str, field: str) -> datetime:
        try:
            value = datetime.fromisoformat(raw)
        except (TypeError, ValueError) as exc:
            raise RepositoryDataError("Invalid datetime in %s" % field) from exc
        if value.tzinfo is None or value.utcoffset() is None:
            raise RepositoryDataError("Persisted %s must be timezone-aware" % field)
        if value.utcoffset() != timezone.utc.utcoffset(value):
            raise RepositoryDataError("Persisted %s must be UTC" % field)
        return value.astimezone(timezone.utc)

    def _decode_optional_datetime(
        self, raw: Optional[str], field: str
    ) -> Optional[datetime]:
        return None if raw is None else self._decode_datetime(raw, field)

    @staticmethod
    def _raise_run_integrity_error(
        exc: sqlite3.IntegrityError, run: Run
    ) -> None:
        message = str(exc)
        if "runs.run_id" in message:
            raise RunAlreadyExists(run.run_id) from exc
        if "runs.simulation_task_id" in message and run.simulation_task_id is not None:
            raise TaskIdAlreadyBound(run.simulation_task_id) from exc
        raise exc
