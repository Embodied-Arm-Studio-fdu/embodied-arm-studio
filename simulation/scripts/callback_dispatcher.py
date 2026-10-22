#!/usr/bin/env python3
"""Poll a Redis callback outbox and deliver due records over HTTP."""

import argparse
import os
from pathlib import Path
import stat

from simulation_core import (
    CallbackDispatcher, HttpBackendClient, RedisCallbackOutbox,
    run_polling_loop,
)


TOKEN_ENVIRONMENT_VARIABLE = "SIMULATION_BACKEND_SERVICE_TOKEN"


def _read_secret_file(path_text):
    path = Path(path_text)
    try:
        descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        raise ValueError("service token secret file cannot be read") from None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("service token secret file must be a regular file")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise ValueError("service token secret file must have mode 0600")
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = None
            token = stream.read().strip()
    except (OSError, UnicodeError):
        raise ValueError("service token secret file cannot be read") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not token:
        raise ValueError("service token secret file is empty")
    return token


def resolve_service_token(args, environ=None):
    values = os.environ if environ is None else environ
    sources = []
    if args.service_token is not None:
        sources.append(("legacy --service-token", args.service_token))
    environment_token = values.get(TOKEN_ENVIRONMENT_VARIABLE)
    if environment_token is not None:
        sources.append((TOKEN_ENVIRONMENT_VARIABLE, environment_token))
    if args.service_token_file is not None:
        sources.append(("--service-token-file", None))
    if len(sources) > 1:
        raise ValueError("service token sources are mutually exclusive")
    if not sources:
        return None
    source, token = sources[0]
    if source == "--service-token-file":
        return _read_secret_file(args.service_token_file)
    token = token.strip()
    if not token:
        raise ValueError("configured service token is empty")
    return token


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Recommended token source: environment variable %s; alternatively "
            "use --service-token-file with mode 0600."
            % TOKEN_ENVIRONMENT_VARIABLE
        ),
    )
    parser.add_argument("--redis-url", default="redis://127.0.0.1:6379/0")
    parser.add_argument("--namespace", default="simulation")
    parser.add_argument("--backend-base-url", required=True)
    parser.add_argument("--request-timeout", type=float, default=5.0)
    parser.add_argument(
        "--service-token", help="legacy/insecure: raw token is exposed in argv"
    )
    parser.add_argument(
        "--service-token-file",
        help="read the service token from a regular file with exact mode 0600",
    )
    parser.add_argument("--polling-interval", type=float, default=1.0)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--base-delay", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        service_token = resolve_service_token(args)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        import redis
    except ImportError as exc:
        raise SystemExit("Python package 'redis' is required: %s" % exc)
    client = redis.Redis.from_url(args.redis_url, decode_responses=False)
    client.ping()
    outbox = RedisCallbackOutbox(client, args.namespace)
    backend = HttpBackendClient(
        args.backend_base_url, args.request_timeout, service_token
    )
    dispatcher = CallbackDispatcher(
        outbox, backend, args.max_attempts, args.base_delay
    )
    if args.once:
        dispatcher.dispatch_once()
        return
    print("callback dispatcher started; Ctrl-C to stop", flush=True)
    try:
        run_polling_loop(dispatcher, args.polling_interval)
    except KeyboardInterrupt:
        print("callback dispatcher stopped", flush=True)


if __name__ == "__main__":
    main()
