"""Shared entity identity resolution helpers for adapters."""

from __future__ import annotations

import inspect
from typing import Any

from grounded_memory.core.entity_identity import build_entity_uniqueness_key, stable_entity_id


async def maybe_await(value: Any) -> Any:
    """Await values that may be either sync or async."""
    if inspect.isawaitable(value):
        return await value
    return value


async def find_entity_by_canonical_id(
    *,
    store: Any,
    entity_type: Any,
    canonical_id: str,
) -> Any | None:
    """Find entity by canonical id/identifier when store supports type scans."""
    normalized = canonical_id.strip().lower()
    if not normalized:
        return None

    getter = getattr(store, "get_entities_by_type", None)
    if not callable(getter):
        return None

    entities = await maybe_await(getter(entity_type))
    for entity in entities or []:
        existing_canonical = (getattr(entity, "canonical_id", None) or "").strip().lower()
        existing_identifier = (
            str(getattr(entity, "attributes", {}).get("identifier", "")).strip().lower()
        )
        if existing_canonical == normalized or existing_identifier == normalized:
            return entity

    return None


async def get_or_create_entity(
    *,
    store: Any,
    entity_cls: Any,
    name: str,
    entity_type: Any,
    attributes: dict[str, Any] | None = None,
    canonical_id: str | None = None,
    strict_identity: bool = False,
) -> Any:
    """Resolve entity identity and create/update entity records in a store-agnostic way."""
    normalized_attributes = {k: v for k, v in (attributes or {}).items() if v is not None}

    if strict_identity and canonical_id:
        existing_by_id = await find_entity_by_canonical_id(
            store=store,
            entity_type=entity_type,
            canonical_id=canonical_id,
        )
        if existing_by_id:
            for key, value in normalized_attributes.items():
                if key not in existing_by_id.attributes:
                    existing_by_id.attributes[key] = value
            await maybe_await(store.add_entity(existing_by_id))
            return existing_by_id

        uniqueness_key = build_entity_uniqueness_key(
            name=name,
            entity_type=entity_type,
            attributes=normalized_attributes,
            canonical_id=canonical_id,
        )
        entity = entity_cls(
            id=stable_entity_id(uniqueness_key),
            entity_type=entity_type,
            name=name,
            canonical_id=canonical_id,
            attributes=normalized_attributes,
        )
        await maybe_await(store.add_entity(entity))
        return entity

    existing = await maybe_await(store.find_entity_by_name(name, entity_type))
    if existing:
        if normalized_attributes:
            for key, value in normalized_attributes.items():
                if value is not None and key not in existing.attributes:
                    existing.attributes[key] = value
            if canonical_id and not existing.canonical_id:
                existing.canonical_id = canonical_id
            await maybe_await(store.add_entity(existing))
        return existing

    entity = entity_cls(
        entity_type=entity_type,
        name=name,
        canonical_id=canonical_id,
        attributes=normalized_attributes,
    )
    await maybe_await(store.add_entity(entity))
    return entity
