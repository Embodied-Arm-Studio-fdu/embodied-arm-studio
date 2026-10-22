#!/usr/bin/env python3
"""Create an organization and its first administrator atomically."""

import argparse
import os
import sys

from apps.api.database import create_postgres_engine, create_session_factory
from robotics.auth import PostgreSQLUserRepository, UserNotFound
from robotics.organizations import (
    OrganizationConflict,
    OrganizationService,
    PostgreSQLOrganizationRepository,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create one studio organization"
    )
    parser.add_argument("slug")
    parser.add_argument("--name", required=True)
    parser.add_argument("--administrator-username", required=True)
    args = parser.parse_args()
    engine = create_postgres_engine(environment=os.environ)
    try:
        factory = create_session_factory(engine)
        users = PostgreSQLUserRepository(factory)
        organizations = OrganizationService(
            PostgreSQLOrganizationRepository(factory)
        )
        try:
            administrator = users.get_by_username(
                args.administrator_username
            )
            organization, membership = organizations.create_organization(
                args.slug, args.name, administrator.user_id
            )
        except UserNotFound:
            print(
                "Administrator username does not exist; no data was changed",
                file=sys.stderr,
            )
            return 1
        except (OrganizationConflict, ValueError) as exc:
            print("Organization was not changed: %s" % exc, file=sys.stderr)
            return 1
        print(
            "Organization %s is ready with administrator %s"
            % (organization.slug, administrator.username)
        )
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
