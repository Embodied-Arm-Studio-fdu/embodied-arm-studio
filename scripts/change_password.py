#!/usr/bin/env python3
"""Change a canonical user's password without exposing secrets in argv."""

import argparse
import getpass
import os
import sys

from apps.api.database import create_postgres_engine, create_session_factory
from robotics.auth import (
    AuthenticationFailed,
    AuthenticationService,
    PasswordHasher,
    PostgreSQLUserRepository,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Change one local studio password")
    parser.add_argument("username")
    args = parser.parse_args()

    current_password = getpass.getpass("Current password: ")
    new_password = getpass.getpass("New password: ")
    confirmation = getpass.getpass("Confirm new password: ")
    if new_password != confirmation:
        print("Passwords do not match", file=sys.stderr)
        return 2

    engine = create_postgres_engine(environment=os.environ)
    try:
        repository = PostgreSQLUserRepository(create_session_factory(engine))
        service = AuthenticationService(repository, PasswordHasher())
        try:
            service.change_password(args.username, current_password, new_password)
        except AuthenticationFailed:
            print("Current credentials are invalid", file=sys.stderr)
            return 1
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print("Changed password for %s" % args.username)
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())