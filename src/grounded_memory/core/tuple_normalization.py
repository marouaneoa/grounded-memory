"""Utilities for canonical tuple normalization and semantic-key derivation."""

from __future__ import annotations

import re
from typing import Any

_WS_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[\s\.,;:!?]+$")
_KEYED_VALUE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_\- ./]{0,63})\s*[:=]\s*(.+?)\s*$")
_KEY_NON_ALNUM_RE = re.compile(r"[^a-z0-9_]+")
_MULTI_UNDERSCORE_RE = re.compile(r"_+")

ATTRIBUTE_KEY_ALIASES: dict[str, str] = {
    "preference": "prefers",
    "preferences": "prefers",
    "preferred": "prefers",
    "prefer": "prefers",
    "like": "likes",
    "loc": "location",
    "lives_in": "location",
    "located_in": "location",
    "work_on": "works_on",
    "working_on": "works_on",
}


def _relation_value(relation: Any) -> str:
    value = getattr(relation, "value", relation)
    return str(value).strip()


def sanitize_fact_value(value: str | None) -> str | None:
    """Trim and normalize spacing while preserving value casing."""
    if value is None:
        return None
    text = _WS_RE.sub(" ", str(value).strip())
    text = _TRAILING_PUNCT_RE.sub("", text)
    return text or None


def normalize_attribute_key(key: str | None) -> str | None:
    """Canonicalize attribute key strings for stable matching."""
    if key is None:
        return None

    candidate = str(key).strip().lower()
    if not candidate:
        return None

    candidate = _KEY_NON_ALNUM_RE.sub("_", candidate)
    candidate = _MULTI_UNDERSCORE_RE.sub("_", candidate).strip("_")
    if not candidate:
        return None

    return ATTRIBUTE_KEY_ALIASES.get(candidate, candidate)


def parse_keyed_value(value: str | None) -> tuple[str | None, str | None]:
    """Parse values like 'key=value' or 'key: value'."""
    sanitized = sanitize_fact_value(value)
    if sanitized is None:
        return None, None

    match = _KEYED_VALUE_RE.match(sanitized)
    if not match:
        return None, sanitized

    key = normalize_attribute_key(match.group(1))
    parsed_value = sanitize_fact_value(match.group(2))
    if key is None:
        return None, sanitized

    return key, parsed_value


def resolve_attribute_key(
    value: str | None,
    attributes: dict[str, Any] | None = None,
) -> str | None:
    """Resolve tuple slot key from explicit attributes or keyed value payload."""
    attrs = attributes or {}
    explicit_key = normalize_attribute_key(attrs.get("key"))
    if explicit_key is not None:
        return explicit_key

    key, _ = parse_keyed_value(value)
    return key


def normalize_fact_attributes(
    value: str | None,
    attributes: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a normalized attribute dictionary with canonical key if available."""
    normalized = dict(attributes or {})
    key = resolve_attribute_key(value, normalized)
    if key is not None:
        normalized["key"] = key
    return normalized


def normalize_fact_value_for_match(value: str | None) -> str | None:
    """Canonical value representation used for duplicate and retire matching."""
    sanitized = sanitize_fact_value(value)
    if sanitized is None:
        return None

    key, parsed_value = parse_keyed_value(sanitized)
    if key is not None and parsed_value is not None:
        return f"{key}={parsed_value.lower()}"

    return sanitized.lower()


def fact_values_equal(left: str | None, right: str | None) -> bool:
    """Compare fact values under canonical normalization."""
    return normalize_fact_value_for_match(left) == normalize_fact_value_for_match(right)


def should_materialize_attribute_object(value: str | None) -> bool:
    """Decide whether a plain string value is specific enough to become an entity node."""
    normalized = sanitize_fact_value(value)
    if normalized is None:
        return False
    if len(normalized) > 80:
        return False
    if len(normalized.split()) > 8:
        return False
    lowered = normalized.lower()
    if lowered.startswith(("the user ", "assistant ", "summary", "message")):
        return False
    return not any(ch in normalized for ch in ["\n", "\r", "\t", "{", "}", "[", "]"])


def build_fact_semantic_key(
    *,
    subject_id: str,
    relation: Any,
    object_id: str | None,
    value: str | None,
    attributes: dict[str, Any] | None,
    include_subject: bool = True,
) -> str | None:
    """Build semantic slot key for tuple lineage and diversity grouping."""
    relation_value = _relation_value(relation)
    if not relation_value:
        return None

    parts: list[str] = []
    if include_subject:
        subject = str(subject_id).strip()
        if not subject:
            return None
        parts.append(subject)

    parts.append(relation_value)

    slot_key = resolve_attribute_key(value, attributes)
    if slot_key is not None:
        parts.append(f"k:{slot_key}")
        return "|".join(parts)

    object_part = str(object_id or "").strip()
    if object_part:
        parts.append(f"o:{object_part}")
        return "|".join(parts)

    normalized_value = normalize_fact_value_for_match(value)
    if normalized_value is None:
        return None

    parts.append(f"v:{normalized_value}")
    return "|".join(parts)
