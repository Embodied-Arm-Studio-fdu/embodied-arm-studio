"""Embodied Arm Studio test suite."""

import os


# Ordinary tests are isolated from PostgreSQL unless a test explicitly injects
# its repository and passes the PostgreSQL safety gate.
os.environ.setdefault("RUN_REPOSITORY_MODE", "memory")
