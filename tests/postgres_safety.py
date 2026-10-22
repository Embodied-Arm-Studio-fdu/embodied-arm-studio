"""Fail-closed PostgreSQL test target validation without network access."""

import os
from typing import Mapping, Optional, Tuple

from sqlalchemy.engine import URL, make_url


OPT_IN_ENV = "POSTGRES_TEST_OPT_IN"
TEST_URL_ENV = "TEST_DATABASE_URL"
RUNTIME_FINGERPRINT_ENV = "MIDTERM_RUNTIME_DATABASE_FINGERPRINT"
KNOWN_RUNTIME_DATABASES = frozenset({"embodied_arm_test"})


def database_fingerprint(value) -> Tuple[str, str, int, str, str]:
    """Return a password-free, comparison-safe PostgreSQL endpoint identity."""
    url = value if isinstance(value, URL) else make_url(value)
    backend = url.get_backend_name().lower()
    if backend != "postgresql":
        raise RuntimeError("PostgreSQL tests require a PostgreSQL URL")
    host = (url.host or "").strip().lower()
    if host in {"localhost", "127.0.0.1", "::1"}:
        host = "loopback"
    database = (url.database or "").strip()
    if not database:
        raise RuntimeError("PostgreSQL test URL must name a database")
    return backend, host, url.port or 5432, url.username or "", database


def fingerprint_text(value) -> str:
    """Serialize a non-secret fingerprint for explicit runtime comparison."""
    return "|".join(str(part) for part in database_fingerprint(value))


def safe_test_database_url(
    environment: Optional[Mapping[str, str]] = None,
) -> Optional[URL]:
    """Validate explicit opt-in and reject known/current runtime identities.

    This function reads environment variables only. It never reads a runtime or
    test credential file and never opens a database connection.
    """
    source = os.environ if environment is None else environment
    raw = source.get(TEST_URL_ENV, "").strip()
    opted_in = source.get(OPT_IN_ENV) == "1"

    if not raw and not opted_in:
        return None
    if not raw:
        raise RuntimeError("POSTGRES_TEST_OPT_IN=1 requires TEST_DATABASE_URL")
    if not opted_in:
        raise RuntimeError("TEST_DATABASE_URL requires POSTGRES_TEST_OPT_IN=1")

    url = make_url(raw)
    fingerprint = database_fingerprint(url)
    database = fingerprint[-1]
    if "test" not in database.lower():
        raise RuntimeError("PostgreSQL test database name must contain 'test'")
    if database.lower() in KNOWN_RUNTIME_DATABASES:
        raise RuntimeError("refusing known midterm runtime database")

    runtime_url = source.get("DATABASE_URL", "").strip()
    if runtime_url and fingerprint == database_fingerprint(runtime_url):
        raise RuntimeError("test database fingerprint matches runtime database")

    runtime_fingerprint = source.get(RUNTIME_FINGERPRINT_ENV, "").strip()
    if runtime_fingerprint and fingerprint_text(url) == runtime_fingerprint:
        raise RuntimeError("test database fingerprint matches midterm runtime")
    return url
