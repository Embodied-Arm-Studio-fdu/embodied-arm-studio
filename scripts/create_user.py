#!/usr/bin/env python3
"""Create a local canonical user without exposing the password in argv."""

import argparse
import getpass
import os
import sys

from apps.api.database import create_postgres_engine, create_session_factory
from robotics.auth import (
    AuthenticationService,
    DuplicateUsername,
    PasswordHasher,
    PostgreSQLUserRepository,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create one local studio user")
    parser.add_argument("username")
    parser.add_argument("--display-name", required=True)
    args = parser.parse_args()
    password = getpass.getpass("Password: ")
    confirmation = getpass.getpass("Confirm password: ")
    if password != confirmation:
        print("Passwords do not match", file=sys.stderr)
        return 2
    engine = create_postgres_engine(environment=os.environ)
    try:
        repository = PostgreSQLUserRepository(create_session_factory(engine))
        service = AuthenticationService(repository, PasswordHasher())
        try:
            user = service.create_user(args.username, args.display_name, password)
        except DuplicateUsername:
            print("Username already exists; no user was changed", file=sys.stderr)
            return 1
        print("Created user %s" % user.username)
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
