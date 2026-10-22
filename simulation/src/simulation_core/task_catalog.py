"""Versioned, canonical read-only projection of executable task definitions."""

import hashlib
import json
from typing import Any, Mapping

from .task_registry import PRODUCTION_TASK_REGISTRY

CATALOG_VERSION = "task-catalog/v2"


def task_catalog(registry=PRODUCTION_TASK_REGISTRY):
    tasks = [
        definition.catalog_entry() for definition in registry.definitions()
        if definition.catalog_visible
    ]
    return sorted(tasks, key=lambda task: task["id"])


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the exact cross-language bytes used by all catalog digests."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_digest(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def task_catalog_document(registry=PRODUCTION_TASK_REGISTRY):
    digest_input = {
        "catalog_version": CATALOG_VERSION,
        "tasks": task_catalog(registry),
    }
    return {
        "catalog_version": CATALOG_VERSION,
        "catalog_digest": sha256_digest(digest_input),
        "tasks": digest_input["tasks"],
    }
