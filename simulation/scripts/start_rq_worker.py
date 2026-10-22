#!/usr/bin/env python3
"""Start the configured RQ worker only when its queue has no live worker."""

import argparse
from datetime import datetime, timezone
import json
import os
import sys


ALIVE_STATES = frozenset(("started", "busy", "idle", "suspended"))


def worker_is_alive(worker, connection, now=None):
    """Use RQ heartbeat metadata, not process command text, as liveness proof."""
    worker.refresh()
    heartbeat = worker.last_heartbeat
    if heartbeat is None or worker.state not in ALIVE_STATES:
        return False
    ttl = int(connection.ttl(worker.key))
    if ttl <= 0:
        return False
    if heartbeat.tzinfo is None:
        heartbeat = heartbeat.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    maximum_age = float(worker.worker_ttl + 60)
    return 0.0 <= (current - heartbeat).total_seconds() <= maximum_age


def live_workers(redis_url, queue_name, redis_factory=None, queue_factory=None,
                 worker_class=None, now=None):
    """Return live RQ workers registered for one queue on one Redis instance."""
    if redis_factory is None:
        from redis import Redis
        redis_factory = Redis.from_url
    if queue_factory is None or worker_class is None:
        from rq import Queue, Worker
        queue_factory = queue_factory or Queue
        worker_class = worker_class or Worker
    connection = redis_factory(redis_url)
    queue = queue_factory(queue_name, connection=connection)
    return [
        worker for worker in worker_class.all(queue=queue)
        if worker_is_alive(worker, connection, now=now)
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only", action="store_true",
        help="report liveness without starting a missing worker",
    )
    args = parser.parse_args(argv)
    redis_url = os.environ.get("SIMULATION_REDIS_URL")
    queue_name = os.environ.get("SIMULATION_RQ_QUEUE")
    if not redis_url or not queue_name:
        parser.error("SIMULATION_REDIS_URL and SIMULATION_RQ_QUEUE are required")

    workers = live_workers(redis_url, queue_name)
    result = {
        "queue": queue_name,
        "alive_worker_count": len(workers),
        "alive_workers": [
            {
                "name": worker.name,
                "state": worker.state,
                "pid": worker.pid,
                "last_heartbeat": worker.last_heartbeat.isoformat(),
            }
            for worker in workers
        ],
        "action": "skip_start" if workers else (
            "check_only" if args.check_only else "start_worker"
        ),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    if workers or args.check_only:
        return 0

    os.execvp(
        "rq",
        ("rq", "worker", "-u", redis_url, queue_name),
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
