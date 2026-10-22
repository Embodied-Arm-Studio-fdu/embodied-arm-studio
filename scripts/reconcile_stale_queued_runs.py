#!/usr/bin/env python3
"""One-time, fail-closed reconciliation for explicitly named stale queued Runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Iterable, Mapping, Optional, Set

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apps.api.database import resolve_database_url
from robotics.runs import (
    ExecutionMode,
    PostgreSQLRunRepository,
    RunService,
    RunStatus,
)
from scripts.check_midterm_runtime import RedisProbe, parse_env_file


DEFAULT_RUNTIME_CONFIG = Path(
    "/home/user/.config/embodied-arm-studio/midterm-runtime.env"
)
DEFAULT_DATABASE_CONFIG = Path("/home/user/project/embodied-arm-studio/.env")


class ReconciliationPreconditionError(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--protected-run-id", action="append", default=[])
    parser.add_argument("--reason", required=True)
    parser.add_argument("--minimum-age-minutes", type=int, default=60)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--database-config", type=Path, default=DEFAULT_DATABASE_CONFIG)
    parser.add_argument("--database-host-override")
    parser.add_argument("--apply", action="store_true")
    return parser


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _scan_string_values(probe: RedisProbe, pattern: str) -> Iterable[str]:
    cursor = "0"
    while True:
        reply = probe.command("SCAN", cursor, "MATCH", pattern, "COUNT", "100")
        cursor = _text(reply[0])
        for key in reply[1]:
            encoded = probe.command("GET", _text(key))
            if encoded is not None:
                yield _text(encoded)
        if cursor == "0":
            return


def _require_runtime_idle(
    values: Mapping[str, str], run_id: str, protected: Set[str]
) -> None:
    if run_id in protected:
        raise ReconciliationPreconditionError("Run is protected acceptance evidence")
    namespace = values["SIMULATION_REDIS_NAMESPACE"]
    queue = values["SIMULATION_RQ_QUEUE"]
    probe = RedisProbe(values["SIMULATION_REDIS_URL"])
    if probe.command("GET", "%s:run:%s" % (namespace, run_id)) is not None:
        raise ReconciliationPreconditionError("Simulation run index exists")
    if any(run_id in value for value in _scan_string_values(
        probe, "%s:task:*" % namespace
    )):
        raise ReconciliationPreconditionError("Simulation task payload exists")
    if int(probe.command("LLEN", "rq:queue:%s" % queue) or 0) != 0:
        raise ReconciliationPreconditionError("RQ queue is not empty")
    for prefix in ("rq:wip:", "rq:deferred:", "rq:scheduled:"):
        try:
            count = int(probe.command("ZCARD", prefix + queue) or 0)
        except RuntimeError:
            count = 0
        if count:
            raise ReconciliationPreconditionError("RQ active registry is not empty")
    workers = probe.command("SMEMBERS", "rq:workers:%s" % queue) or []
    selected = []
    for worker in workers:
        worker_key = _text(worker)
        queues = probe.command("HGET", worker_key, "queues")
        if queues is not None and queue in _text(queues).split(","):
            selected.append(worker_key)
    if len(selected) != 1:
        raise ReconciliationPreconditionError("expected exactly one RQ Worker")
    state = probe.command("HGET", selected[0], "state")
    current_job = probe.command("HGET", selected[0], "current_job")
    if _text(state) != "idle" or current_job not in {None, b"", ""}:
        raise ReconciliationPreconditionError("RQ Worker is not idle")
    callback_ids = probe.command(
        "SMEMBERS", "%s:callback_outbox:index" % namespace
    ) or []
    for event_id in callback_ids:
        encoded = probe.command(
            "GET",
            "%s:callback_outbox:event:%s" % (namespace, _text(event_id)),
        )
        if encoded is not None:
            try:
                record = json.loads(_text(encoded))
            except (TypeError, ValueError):
                raise ReconciliationPreconditionError("callback outbox is malformed")
            if run_id in json.dumps(record, sort_keys=True):
                raise ReconciliationPreconditionError("callback outbox references Run")


def reconcile(args: argparse.Namespace) -> int:
    if args.minimum_age_minutes < 1:
        raise ValueError("minimum age must be positive")
    if len(args.run_id) != len(set(args.run_id)):
        raise ValueError("run ids must be unique")
    runtime_values = parse_env_file(args.runtime_config.expanduser())
    database_values = parse_env_file(args.database_config.expanduser())
    url = resolve_database_url(database_values["DATABASE_URL"])
    if args.database_host_override:
        url = url.set(host=args.database_host_override)
    engine = create_engine(url, future=True, hide_parameters=True, pool_pre_ping=True)
    repository = PostgreSQLRunRepository(
        sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    )
    service = RunService(repository)
    protected = set(args.protected_run_id)
    try:
        for run_id in args.run_id:
            run = service.get_run(run_id)
            if (
                run.status is not RunStatus.QUEUED
                or run.execution_mode is not ExecutionMode.SIMULATION
                or run.simulation_task_id is not None
                or service.list_events(run_id)
            ):
                raise ReconciliationPreconditionError(
                    "%s is not a queued, unbound, event-free simulation Run" % run_id
                )
            if datetime.now(timezone.utc) - run.created_at < timedelta(
                minutes=args.minimum_age_minutes
            ):
                raise ReconciliationPreconditionError(
                    "%s is inside the normal startup window" % run_id
                )
            _require_runtime_idle(runtime_values, run_id, protected)
            if args.apply:
                stopped = service.administratively_reconcile_queued_unbound(
                    run_id, args.reason
                )
                print(
                    "RECONCILED run_id=%s status=%s event_sequence=%d"
                    % (stopped.run_id, stopped.status.value, stopped.last_sequence)
                )
            else:
                print("READY run_id=%s" % run_id)
        return 0
    finally:
        engine.dispose()


def main(argv: Optional[Iterable[str]] = None) -> int:
    try:
        return reconcile(_parser().parse_args(argv))
    except (KeyError, OSError, ValueError, ReconciliationPreconditionError) as exc:
        print("STOP: %s" % exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
