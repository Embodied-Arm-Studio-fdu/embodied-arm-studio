#!/usr/bin/env python3
"""Add or update one existing user's organization membership."""

import argparse
import os
import sys

from apps.api.database import create_postgres_engine, create_session_factory
from robotics.auth import PostgreSQLUserRepository, UserNotFound
from robotics.organizations import (
    LastAdministratorError,
    MembershipNotFound,
    OrganizationConflict,
    OrganizationNotFound,
    OrganizationRole,
    OrganizationService,
    PostgreSQLOrganizationRepository,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set one studio organization membership"
    )
    parser.add_argument("organization_slug")
    parser.add_argument("username")
    parser.add_argument(
        "--role",
        required=True,
        choices=[role.value for role in OrganizationRole],
    )
    status = parser.add_mutually_exclusive_group(required=True)
    status.add_argument("--enable", action="store_true")
    status.add_argument("--disable", action="store_true")
    args = parser.parse_args()
    engine = create_postgres_engine(environment=os.environ)
    try:
        factory = create_session_factory(engine)
        users = PostgreSQLUserRepository(factory)
        organizations = OrganizationService(
            PostgreSQLOrganizationRepository(factory)
        )
        try:
            user = users.get_by_username(args.username)
            organization = organizations.get_by_slug(
                args.organization_slug
            )
            role = OrganizationRole(args.role)
            try:
                organizations.get_membership(
                    organization.organization_id, user.user_id
                )
            except MembershipNotFound:
                if args.disable:
                    raise OrganizationConflict(
                        "cannot create a disabled membership"
                    )
                membership = organizations.add_membership(
                    organization.organization_id, user.user_id, role
                )
            else:
                membership = organizations.set_membership(
                    organization.organization_id,
                    user.user_id,
                    role,
                    args.enable,
                )
        except (UserNotFound, OrganizationNotFound):
            print(
                "User or organization does not exist; no data was changed",
                file=sys.stderr,
            )
            return 1
        except (
            LastAdministratorError,
            OrganizationConflict,
            ValueError,
        ) as exc:
            print("Membership was not changed: %s" % exc, file=sys.stderr)
            return 1
        print(
            "Membership %s/%s is %s as %s"
            % (
                organization.slug,
                user.username,
                "enabled" if membership.enabled else "disabled",
                membership.role.value,
            )
        )
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
