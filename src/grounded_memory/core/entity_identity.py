"""Deterministic entity identity helpers for cross-run idempotency."""

from __future__ import annotations

import re
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from grounded_memory.core.models import EntityType

_WS_RE = re.compile(r"\s+")


def _normalize_text(value: str) -> str:
    return _WS_RE.sub(" ", value.strip().lower())


def _resolve_scope_id(attributes: dict[str, Any] | None) -> str | None:
    attrs = attributes or {}
    explicit_scope = attrs.get("scope_id")
    if isinstance(explicit_scope, str) and explicit_scope.strip():
        return explicit_scope.strip()

    tenant_id = attrs.get("tenant_id")
    app_id = attrs.get("app_id")
    user_id = attrs.get("user_id")
    if all(isinstance(value, str) and value.strip() for value in (tenant_id, app_id, user_id)):
        return f"{tenant_id.strip()}:{app_id.strip()}:{user_id.strip()}"

    return None


def build_entity_uniqueness_key(
    *,
    name: str,
    entity_type: EntityType | str,
    attributes: dict[str, Any] | None = None,
    canonical_id: str | None = None,
    uniqueness_key: str | None = None,
) -> str:
    """Build a stable semantic key for entity identity across process restarts."""
    scope_id = _resolve_scope_id(attributes)
    scope_segment = f"scope:{scope_id}" if scope_id else "scope:global"

    if isinstance(entity_type, EntityType):
        entity_type_value = entity_type.value
    else:
        entity_type_value = str(entity_type)

    if uniqueness_key is not None and str(uniqueness_key).strip():
        base = f"uk:{_normalize_text(str(uniqueness_key))}"
    elif canonical_id is not None and str(canonical_id).strip():
        base = f"canonical:{_normalize_text(str(canonical_id))}"
    else:
        base = f"name:{_normalize_text(name)}"

    return f"{scope_segment}|type:{_normalize_text(entity_type_value)}|{base}"


def stable_entity_id(uniqueness_key: str) -> str:
    """Derive a deterministic UUID from a semantic uniqueness key."""
    return str(uuid5(NAMESPACE_URL, f"gmem:entity:{uniqueness_key}"))
