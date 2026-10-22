#!/usr/bin/env python3
"""Preview or safely create one private midterm runtime environment file."""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import secrets
import shlex
import stat
import sys
import tempfile
from typing import Callable, Dict, Iterable, Mapping, Optional, TextIO
from urllib.parse import urlsplit

from sqlalchemy.engine import URL

from apps.api.database import resolve_database_url


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("/home/user/.config/embodied-arm-studio/midterm-runtime.env")
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "var" / "midterm-demo-artifacts"
DEFAULT_DATABASE_HOST = "127.0.0.1"
DEFAULT_DATABASE_PORT = 5433
DEFAULT_BACKEND_ORIGIN = "http://127.0.0.1:8000"
DEFAULT_SIMULATION_ORIGIN = "http://127.0.0.1:8090"
DEFAULT_REDIS_URL = "redis://127.0.0.1:6379/0"
DEFAULT_REDIS_NAMESPACE = "midterm-demo"
DEFAULT_RQ_QUEUE = "midterm-demo-gpu"
DEFAULT_STREAM_RTSP_BASE = "rtsp://127.0.0.1:18554"
DEFAULT_MEDIAMTX_CONTROL_URL = "http://127.0.0.1:19997"
DEFAULT_MEDIAMTX_CONFIG_PATH = PROJECT_ROOT / "simulation" / "config" / "mediamtx_midterm.yml"
DEFAULT_MEDIAMTX_BINARY_PATH = Path("/opt/mediamtx/mediamtx")
DEFAULT_FFMPEG_PATH = Path("/opt/ffmpeg/bin/ffmpeg")
SECRET_KEYS = frozenset(
    {
        "DATABASE_URL",
        "USER_JWT_SECRET",
        "SIMULATION_SERVICE_TOKEN",
        "INTERNAL_RUN_EVENT_SERVICE_TOKEN",
        "SIMULATION_BACKEND_SERVICE_TOKEN",
    }
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the private file")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace an existing file after an interactive confirmation",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--database-host")
    parser.add_argument("--database-port", type=int)
    parser.add_argument("--database-name")
    parser.add_argument("--database-user")
    parser.add_argument("--backend-origin")
    parser.add_argument("--simulation-origin")
    parser.add_argument("--redis-url")
    parser.add_argument("--redis-namespace")
    parser.add_argument("--rq-queue")
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument(
        "--video-only", action="store_true",
        help="append or verify only public media settings in an existing 0600 file",
    )
    parser.add_argument("--stream-rtsp-base", default=DEFAULT_STREAM_RTSP_BASE)
    parser.add_argument("--stream-public-base")
    parser.add_argument("--mediamtx-control-url", default=DEFAULT_MEDIAMTX_CONTROL_URL)
    parser.add_argument("--mediamtx-ice-udp-port", type=int, default=18189)
    parser.add_argument("--mediamtx-ice-additional-host")
    parser.add_argument("--mediamtx-allowed-origin")
    parser.add_argument("--mediamtx-config-path", type=Path, default=DEFAULT_MEDIAMTX_CONFIG_PATH)
    parser.add_argument("--mediamtx-binary-path", type=Path, default=DEFAULT_MEDIAMTX_BINARY_PATH)
    parser.add_argument("--simulation-ffmpeg-path", type=Path, default=DEFAULT_FFMPEG_PATH)
    return parser


def _prompt(
    supplied: Optional[str],
    label: str,
    reader: Callable[[str], str],
) -> str:
    value = supplied if supplied is not None else reader(label + ": ")
    value = value.strip()
    if not value:
        raise ValueError("%s must not be empty" % label)
    return value


def _prompt_default(
    supplied: Optional[str],
    label: str,
    default: str,
    reader: Callable[[str], str],
) -> str:
    if supplied is not None:
        value = str(supplied).strip()
    else:
        value = reader("%s [%s]: " % (label, default)).strip() or default
    if not value:
        raise ValueError("%s must not be empty" % label)
    return value


def _origin(value: str, label: str, required_port: int) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("%s must be an HTTP origin" % label)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("%s must not contain credentials, query, or fragment" % label)
    if parsed.path not in {"", "/"}:
        raise ValueError("%s must not contain a path" % label)
    if parsed.port != required_port:
        raise ValueError("%s must use port %d" % (label, required_port))
    return "%s://%s:%d" % (parsed.scheme, parsed.hostname, parsed.port)


def _redis_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"redis", "rediss"} or not parsed.hostname:
        raise ValueError("Redis URL must use redis or rediss and include a host")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Redis URL must not contain credentials, query, or fragment")
    database = parsed.path.lstrip("/")
    if not database.isdigit() or "/" in database:
        raise ValueError("Redis URL must select one numeric database")
    return value


def _media_origin(value: str, label: str, scheme: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != scheme or not parsed.hostname or parsed.port is None
        or parsed.username or parsed.password or parsed.query or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("%s must be a credential-free %s origin with an explicit port" % (label, scheme))
    return "%s://%s:%d" % (parsed.scheme, parsed.hostname, parsed.port)


def build_video_values(args: argparse.Namespace) -> Dict[str, str]:
    if not args.stream_public_base:
        raise ValueError("--stream-public-base is required with --video-only")
    if not args.mediamtx_ice_additional_host:
        raise ValueError("--mediamtx-ice-additional-host is required with --video-only")
    if not args.mediamtx_allowed_origin:
        raise ValueError("--mediamtx-allowed-origin is required with --video-only")
    rtsp = _media_origin(args.stream_rtsp_base, "internal RTSP base", "rtsp")
    public = _media_origin(args.stream_public_base, "public playback base", "http")
    control = _media_origin(args.mediamtx_control_url, "MediaMTX control URL", "http")
    public_parts = urlsplit(public)
    if public_parts.hostname in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("public playback base must not use loopback")
    if urlsplit(rtsp).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("internal RTSP base must use loopback")
    if args.mediamtx_ice_additional_host != public_parts.hostname:
        raise ValueError("ICE additional host must MATCH public playback host")
    allowed_origin = _origin(args.mediamtx_allowed_origin, "MediaMTX allowed origin", 5173)
    if not 1 <= args.mediamtx_ice_udp_port <= 65535:
        raise ValueError("MediaMTX ICE UDP port must be in 1..65535")
    paths = {
        "MEDIAMTX_CONFIG_PATH": args.mediamtx_config_path.expanduser().resolve(),
        "MEDIAMTX_BINARY_PATH": args.mediamtx_binary_path.expanduser().resolve(),
        "SIMULATION_FFMPEG_PATH": args.simulation_ffmpeg_path.expanduser().resolve(),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise ValueError("%s must name an existing file" % label)
    return {
        "SIMULATION_STREAM_RTSP_BASE": rtsp,
        "SIMULATION_STREAM_PUBLIC_BASE_URL": public,
        "SIMULATION_FFMPEG_PATH": str(paths["SIMULATION_FFMPEG_PATH"]),
        "MEDIAMTX_CONTROL_URL": control,
        "MEDIAMTX_ICE_UDP_PORT": str(args.mediamtx_ice_udp_port),
        "MEDIAMTX_ICE_ADDITIONAL_HOST": args.mediamtx_ice_additional_host,
        "MEDIAMTX_WEBRTC_ALLOWED_ORIGIN": allowed_origin,
        "MEDIAMTX_CONFIG_PATH": str(paths["MEDIAMTX_CONFIG_PATH"]),
        "MEDIAMTX_BINARY_PATH": str(paths["MEDIAMTX_BINARY_PATH"]),
    }


def _existing_assignments(content: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for line_number, raw_line in enumerate(content.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError("invalid existing config line %d" % line_number)
        key, raw_value = line.split("=", 1)
        tokens = shlex.split(raw_value, posix=True)
        if len(tokens) != 1:
            raise ValueError("invalid existing config value on line %d" % line_number)
        values[key.strip()] = tokens[0]
    return values


def _run_video_only(args: argparse.Namespace, output: TextIO, error: TextIO) -> int:
    try:
        additions = build_video_values(args)
    except (OSError, ValueError) as exc:
        print("ERROR: %s" % exc, file=error)
        return 2
    print("Mode: %s" % ("apply video-only" if args.apply else "preview video-only (zero writes)"), file=output)
    print("Target: %s" % args.config, file=output)
    print("Public media: host=%s port=%d" % (
        urlsplit(additions["SIMULATION_STREAM_PUBLIC_BASE_URL"]).hostname,
        urlsplit(additions["SIMULATION_STREAM_PUBLIC_BASE_URL"]).port,
    ), file=output)
    print("Variables: %s" % ",".join(sorted(additions)), file=output)
    print("Secrets: existing values are preserved and never displayed", file=output)
    if not args.apply:
        return 0
    target = args.config.expanduser().resolve()
    try:
        if not target.is_file():
            raise ValueError("video-only apply requires an existing private config")
        if stat.S_IMODE(target.stat().st_mode) != 0o600:
            raise ValueError("existing private config must have mode 0600")
        original = target.read_text(encoding="utf-8")
        existing = _existing_assignments(original)
        conflicts = sorted(key for key, value in additions.items() if key in existing and existing[key] != value)
        if conflicts:
            raise ValueError("video configuration conflict: %s" % ",".join(conflicts))
        missing = [key for key in additions if key not in existing]
        if missing:
            suffix = "\n# Phase 7.7.1 MediaMTX and Simulation video.\n"
            suffix += "\n".join("%s=%s" % (key, _shell_quote(additions[key])) for key in missing) + "\n"
            _atomic_write(target, original.rstrip("\n") + suffix)
        if stat.S_IMODE(target.stat().st_mode) != 0o600:
            raise ValueError("private config mode changed unexpectedly")
    except (OSError, ValueError) as exc:
        print("ERROR: %s" % exc, file=error)
        return 2
    print("Video configuration MATCH; private file remains 0600", file=output)
    return 0


def _safe_name(value: str, label: str) -> str:
    if not value or len(value) > 80 or any(
        character.isspace() or character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in value
    ):
        raise ValueError("%s must be a 1-80 character safe name" % label)
    return value


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _redacted_database(url: URL) -> str:
    return "%s://%s:%s/%s" % (
        url.drivername,
        url.host or "<host>",
        url.port or "<default>",
        url.database or "<database>",
    )


def build_values(
    args: argparse.Namespace,
    database_password: str,
    *,
    token_factory: Callable[[int], str] = secrets.token_urlsafe,
) -> Dict[str, str]:
    if not database_password:
        raise ValueError("database password must not be empty")
    database_name = args.database_name
    database_user = args.database_user
    if not database_name or not database_user:
        raise ValueError("database name and user are required")
    database_url = URL.create(
        "postgresql+psycopg",
        username=database_user,
        password=database_password,
        host=args.database_host,
        port=args.database_port,
        database=database_name,
    )
    resolve_database_url(database_url)
    rendered_database_url = database_url.render_as_string(hide_password=False)
    backend_origin = _origin(args.backend_origin, "Backend origin", 8000)
    simulation_origin = _origin(args.simulation_origin, "Simulation origin", 8090)
    redis_url = _redis_url(args.redis_url)
    namespace = _safe_name(args.redis_namespace, "Redis namespace")
    queue = _safe_name(args.rq_queue, "RQ queue")
    artifact_root = args.artifact_root.expanduser().resolve()
    if not artifact_root.is_absolute():
        raise ValueError("Artifact root must be absolute")
    simulation_parts = urlsplit(simulation_origin)
    user_jwt = token_factory(48)
    adapter_token = token_factory(48)
    callback_token = token_factory(48)
    return {
        "DATABASE_URL": rendered_database_url,
        "RUN_REPOSITORY_MODE": "postgresql",
        "USER_JWT_SECRET": user_jwt,
        "USER_JWT_ISSUER": "embodied-arm-studio",
        "USER_JWT_AUDIENCE": "embodied-arm-studio-api",
        "USER_JWT_EXPIRES_SECONDS": "900",
        "OPENAI_API_KEY": "",
        "OPENAI_MODEL": "",
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "DEEPSEEK_API_KEY": "",
        "DEEPSEEK_MODEL": "",
        "ANTHROPIC_API_KEY": "",
        "ANTHROPIC_MODEL": "",
        "FRONTEND_ORIGIN": "http://127.0.0.1:5173",
        "VITE_API_PROXY_TARGET": backend_origin,
        "BACKEND_ORIGIN": backend_origin,
        "SIMULATION_BASE_URL": simulation_origin,
        "SIMULATION_SERVICE_TOKEN": adapter_token,
        "SIMULATION_TIMEOUT_SECONDS": "5",
        "SIMULATION_RECOVERY_MAX_RUNS": "100",
        "INTERNAL_RUN_EVENT_SERVICE_TOKEN": callback_token,
        "ARTIFACT_STORAGE_ROOT": str(artifact_root),
        "SIMULATION_HTTP_HOST": simulation_parts.hostname or "127.0.0.1",
        "SIMULATION_HTTP_PORT": str(simulation_parts.port or 8090),
        "SIMULATION_REDIS_URL": redis_url,
        "SIMULATION_REDIS_NAMESPACE": namespace,
        "SIMULATION_RQ_QUEUE": queue,
        "SIMULATION_ARTIFACTS_ROOT": str(artifact_root),
        "SIMULATION_BACKEND_BASE_URL": backend_origin,
        "SIMULATION_BACKEND_REQUEST_TIMEOUT": "5",
        "SIMULATION_BACKEND_SERVICE_TOKEN": callback_token,
        "SIMULATION_HEARTBEAT_INTERVAL": "5",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    }


def render_file(values: Mapping[str, str]) -> str:
    groups: Iterable[Iterable[str]] = (
        (
            "DATABASE_URL", "RUN_REPOSITORY_MODE", "USER_JWT_SECRET",
            "USER_JWT_ISSUER", "USER_JWT_AUDIENCE", "USER_JWT_EXPIRES_SECONDS",
        ),
        (
            "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL",
            "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL",
            "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
        ),
        (
            "FRONTEND_ORIGIN", "VITE_API_PROXY_TARGET", "BACKEND_ORIGIN",
            "SIMULATION_BASE_URL",
            "SIMULATION_SERVICE_TOKEN", "SIMULATION_TIMEOUT_SECONDS",
            "SIMULATION_RECOVERY_MAX_RUNS", "INTERNAL_RUN_EVENT_SERVICE_TOKEN",
            "ARTIFACT_STORAGE_ROOT",
        ),
        (
            "SIMULATION_HTTP_HOST", "SIMULATION_HTTP_PORT", "SIMULATION_REDIS_URL",
            "SIMULATION_REDIS_NAMESPACE", "SIMULATION_RQ_QUEUE",
            "SIMULATION_ARTIFACTS_ROOT", "SIMULATION_BACKEND_BASE_URL",
            "SIMULATION_BACKEND_REQUEST_TIMEOUT", "SIMULATION_BACKEND_SERVICE_TOKEN",
            "SIMULATION_HEARTBEAT_INTERVAL",
        ),
        ("NO_PROXY", "no_proxy"),
    )
    lines = ["# Generated by configure_midterm_runtime.py; keep private."]
    for group in groups:
        lines.append("")
        lines.extend("%s=%s" % (key, _shell_quote(values[key])) for key in group)
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".midterm-runtime.", dir=str(path.parent))
    try:
        os.fchmod(descriptor, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        os.chmod(path, 0o600)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _preview(args: argparse.Namespace, output: TextIO) -> None:
    database_name = args.database_name or "<interactive-database>"
    database_user = args.database_user or "<interactive-user>"
    database_host = args.database_host or DEFAULT_DATABASE_HOST
    database_port = args.database_port or DEFAULT_DATABASE_PORT
    backend_origin = args.backend_origin or DEFAULT_BACKEND_ORIGIN
    simulation_origin = args.simulation_origin or DEFAULT_SIMULATION_ORIGIN
    redis_url = args.redis_url or DEFAULT_REDIS_URL
    redis_namespace = args.redis_namespace or DEFAULT_REDIS_NAMESPACE
    rq_queue = args.rq_queue or DEFAULT_RQ_QUEUE
    artifact_root = args.artifact_root or DEFAULT_ARTIFACT_ROOT
    safe_url = URL.create(
        "postgresql+psycopg",
        username=database_user,
        host=database_host,
        port=database_port,
        database=database_name,
    )
    print("Mode: preview (zero writes)", file=output)
    print("Target: %s" % args.config, file=output)
    print("Database: %s" % _redacted_database(safe_url), file=output)
    print("Backend: %s" % backend_origin, file=output)
    print("Simulation: %s" % simulation_origin, file=output)
    print("Redis: %s namespace=%s queue=%s" % (redis_url, redis_namespace, rq_queue), file=output)
    print("Artifact root: %s" % Path(artifact_root).expanduser().resolve(), file=output)
    print("Variables: %s" % ",".join(sorted(build_variable_names())), file=output)
    print("Secrets: generated only during --apply and never displayed", file=output)


def build_variable_names() -> Iterable[str]:
    return (
        "DATABASE_URL", "RUN_REPOSITORY_MODE", "USER_JWT_SECRET",
        "USER_JWT_ISSUER", "USER_JWT_AUDIENCE", "USER_JWT_EXPIRES_SECONDS",
        "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL",
        "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
        "FRONTEND_ORIGIN", "VITE_API_PROXY_TARGET", "BACKEND_ORIGIN",
        "SIMULATION_BASE_URL",
        "SIMULATION_SERVICE_TOKEN", "SIMULATION_TIMEOUT_SECONDS",
        "SIMULATION_RECOVERY_MAX_RUNS", "INTERNAL_RUN_EVENT_SERVICE_TOKEN",
        "ARTIFACT_STORAGE_ROOT", "SIMULATION_HTTP_HOST", "SIMULATION_HTTP_PORT",
        "SIMULATION_REDIS_URL", "SIMULATION_REDIS_NAMESPACE", "SIMULATION_RQ_QUEUE",
        "SIMULATION_ARTIFACTS_ROOT", "SIMULATION_BACKEND_BASE_URL",
        "SIMULATION_BACKEND_REQUEST_TIMEOUT", "SIMULATION_BACKEND_SERVICE_TOKEN",
        "SIMULATION_HEARTBEAT_INTERVAL", "NO_PROXY", "no_proxy",
    )


def run(
    args: argparse.Namespace,
    *,
    input_reader: Callable[[str], str] = input,
    password_reader: Callable[[str], str] = getpass.getpass,
    output: TextIO = sys.stdout,
    error: TextIO = sys.stderr,
) -> int:
    if args.video_only:
        if args.replace:
            print("ERROR: --replace is not allowed with --video-only", file=error)
            return 2
        return _run_video_only(args, output, error)
    if args.replace and not args.apply:
        print("ERROR: --replace requires --apply", file=error)
        return 2
    _preview(args, output)
    if not args.apply:
        return 0
    target = args.config.expanduser().resolve()
    if target.exists() and not args.replace:
        print("ERROR: target already exists; use --apply --replace", file=error)
        return 2
    if target.exists():
        confirmation = input_reader("Type REPLACE to overwrite the private file: ")
        if confirmation != "REPLACE":
            print("ERROR: replacement cancelled", file=error)
            return 2
    try:
        args.database_host = _prompt_default(
            args.database_host, "PostgreSQL host", DEFAULT_DATABASE_HOST, input_reader
        )
        args.database_port = int(_prompt_default(
            str(args.database_port) if args.database_port is not None else None,
            "PostgreSQL port", str(DEFAULT_DATABASE_PORT), input_reader,
        ))
        database_name = _prompt(args.database_name, "PostgreSQL database name", input_reader)
        database_user = _prompt(args.database_user, "PostgreSQL database user", input_reader)
        args.backend_origin = _prompt_default(
            args.backend_origin, "Backend origin", DEFAULT_BACKEND_ORIGIN, input_reader
        )
        args.simulation_origin = _prompt_default(
            args.simulation_origin, "Simulation origin", DEFAULT_SIMULATION_ORIGIN, input_reader
        )
        args.redis_url = _prompt_default(
            args.redis_url, "Redis URL", DEFAULT_REDIS_URL, input_reader
        )
        args.redis_namespace = _prompt_default(
            args.redis_namespace, "Redis namespace", DEFAULT_REDIS_NAMESPACE, input_reader
        )
        args.rq_queue = _prompt_default(
            args.rq_queue, "RQ queue", DEFAULT_RQ_QUEUE, input_reader
        )
        args.artifact_root = Path(_prompt_default(
            str(args.artifact_root) if args.artifact_root is not None else None,
            "Artifact root", str(DEFAULT_ARTIFACT_ROOT), input_reader,
        ))
        first = password_reader("PostgreSQL database password: ")
        second = password_reader("Confirm PostgreSQL database password: ")
        if first != second:
            raise ValueError("database password confirmation does not match")
        args.database_name = database_name
        args.database_user = database_user
        values = build_values(args, first)
        _atomic_write(target, render_file(values))
    except (OSError, ValueError) as exc:
        print("ERROR: %s" % exc, file=error)
        return 2
    print("Created private runtime configuration with mode 0600: %s" % target, file=output)
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    return run(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
