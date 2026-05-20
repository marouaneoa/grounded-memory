"""Healthcare-specific medication lifecycle helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from grounded_memory.adapters.healthcare.knowledge import normalize_drug_name
from grounded_memory.core.grounding import GroundingResult
from grounded_memory.core.models import CandidateFact, RelationType, ValidatedFact


def fact_medication_names(fact: ValidatedFact, store: Any) -> set[str]:
    """Collect medication names from a prescription/discontinuation fact."""
    names: set[str] = set()

    medication_name = (fact.attributes or {}).get("medication_name")
    if isinstance(medication_name, str) and medication_name.strip():
        names.add(medication_name.strip())

    normalized_name = (fact.attributes or {}).get("normalized_name")
    if isinstance(normalized_name, str) and normalized_name.strip():
        names.add(normalized_name.strip())

    if fact.value and str(fact.value).strip():
        # Values often contain dose/frequency, so keep this as a weak signal.
        names.add(str(fact.value).strip())

    if fact.object_id:
        entity = store.get_entity(fact.object_id) if hasattr(store, "get_entity") else None
        if entity is not None and getattr(entity, "name", None):
            names.add(str(entity.name).strip())

    return {name for name in names if name}


def candidate_medication_names(candidate: CandidateFact, store: Any) -> set[str]:
    """Collect medication names from a medication lifecycle candidate."""
    names: set[str] = set()

    medication_name = (candidate.attributes or {}).get("medication_name")
    if isinstance(medication_name, str) and medication_name.strip():
        names.add(medication_name.strip())

    normalized_name = (candidate.attributes or {}).get("normalized_name")
    if isinstance(normalized_name, str) and normalized_name.strip():
        names.add(normalized_name.strip())

    if candidate.value and str(candidate.value).strip():
        names.add(str(candidate.value).strip())

    if candidate.object_entity_id:
        entity = (
            store.get_entity(candidate.object_entity_id) if hasattr(store, "get_entity") else None
        )
        if entity is not None and getattr(entity, "name", None):
            names.add(str(entity.name).strip())

    return {name for name in names if name}


def normalized_medication_terms(names: set[str]) -> set[str]:
    return {normalize_drug_name(name) for name in names if name and str(name).strip()}


def medication_facts_match(left: ValidatedFact, right: CandidateFact, store: Any) -> bool:
    """Return True when an active prescription matches a discontinuation candidate."""
    if left.object_id and right.object_entity_id and left.object_id == right.object_entity_id:
        return True

    left_terms = normalized_medication_terms(fact_medication_names(left, store))
    right_terms = normalized_medication_terms(candidate_medication_names(right, store))
    return bool(left_terms and right_terms and left_terms & right_terms)


def close_active_prescriptions_for_discontinuation(
    *,
    store: Any,
    candidate: CandidateFact,
    discontinuation_fact: ValidatedFact,
    valid_to: datetime | None = None,
) -> list[ValidatedFact]:
    """Close active PRESCRIBED facts matching an approved DISCONTINUED candidate."""
    if candidate.relation != RelationType.DISCONTINUED:
        return []

    getter = getattr(store, "get_facts_by_relation", None)
    supersede = getattr(store, "supersede_fact", None)
    if not callable(getter) or not callable(supersede):
        return []

    active_prescriptions = getter(
        entity_id=candidate.subject_entity_id,
        relation=RelationType.PRESCRIBED,
        as_subject=True,
    )

    closed: list[ValidatedFact] = []
    close_time = valid_to or discontinuation_fact.valid_from
    for fact in active_prescriptions or []:
        if not getattr(fact, "is_active", False):
            continue
        if not medication_facts_match(fact, candidate, store):
            continue

        supersede(
            fact_id=fact.id,
            superseded_by=discontinuation_fact.id,
            valid_to=close_time,
        )
        updated = store.get_fact(fact.id) if hasattr(store, "get_fact") else None
        closed.append(updated or fact)

    return closed


def apply_medication_lifecycle_after_grounding(
    *,
    store: Any,
    result: GroundingResult,
) -> list[ValidatedFact]:
    """Apply healthcare lifecycle side effects after a successful grounding result."""
    if not result.is_success or result.validated_fact is None:
        return []

    candidate = result.candidate_fact
    action = str((candidate.attributes or {}).get("action", "")).strip().lower()
    if candidate.relation != RelationType.DISCONTINUED and action not in {"discontinue", "hold"}:
        return []

    closed = close_active_prescriptions_for_discontinuation(
        store=store,
        candidate=candidate,
        discontinuation_fact=result.validated_fact,
    )
    if closed:
        result.validated_fact.source_metadata["closed_prescription_fact_ids"] = [
            fact.id for fact in closed
        ]
    return closed
