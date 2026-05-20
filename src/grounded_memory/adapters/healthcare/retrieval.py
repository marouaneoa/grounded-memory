"""Healthcare retrieval planning and clinical context construction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from dateutil import parser as date_parser
from pydantic import BaseModel, Field

from grounded_memory.adapters.healthcare.knowledge import expand_drug_terms, normalize_drug_name
from grounded_memory.adapters.healthcare.lifecycle import (
    fact_medication_names,
    normalized_medication_terms,
)
from grounded_memory.core.models import (
    AnswerContext,
    Entity,
    EntityType,
    RelationType,
    ValidatedFact,
    datetime_after,
)
from grounded_memory.retrieval import GraphRetriever, RetrievalStrategy, select_seed_entities

HEALTHCARE_QUERY_PLANNER_PROMPT = """You extract retrieval intent for a medication reconciliation memory system.
Return only JSON matching the provided schema.
Identify patient names or identifiers, requested clinical categories, medication/allergy names, whether safety facts are requested, and any as-of timestamp.
Use null when the query does not provide a value. Do not invent patient identifiers or dates."""


class HealthcareQueryPlan(BaseModel):
    """Structured retrieval intent for healthcare questions."""

    patient_name: str | None = Field(
        default=None,
        description="Patient name mentioned in the query, if any.",
    )
    patient_identifier: str | None = Field(
        default=None,
        description="Patient identifier such as MRN or patient ID, if mentioned.",
    )
    requested_categories: list[str] = Field(
        default_factory=list,
        description="Requested categories such as current_medications, allergies, safety_alerts, history.",
    )
    medication_names: list[str] = Field(
        default_factory=list,
        description="Medication names mentioned in the query.",
    )
    allergy_names: list[str] = Field(
        default_factory=list,
        description="Allergy or allergen names mentioned in the query.",
    )
    safety_focus: bool = Field(
        default=False,
        description="True when the user asks about allergies, contraindications, interactions, warnings, or safety.",
    )
    as_of: datetime | None = Field(
        default=None,
        description="Point in time for historical retrieval, if explicitly requested.",
    )
    raw_time_expression: str | None = Field(
        default=None,
        description="Original time phrase, if any.",
    )
    ambiguous: bool = Field(
        default=False,
        description="True if more patient/entity information is needed.",
    )
    ambiguity_reason: str | None = Field(default=None)


@dataclass
class HealthcareClinicalContext:
    """Post-processed clinical retrieval view for medication reconciliation."""

    query: str
    plan: HealthcareQueryPlan
    answer_context: AnswerContext
    seed_entities: list[str]
    current_medications: list[dict[str, Any]] = field(default_factory=list)
    allergies: list[dict[str, Any]] = field(default_factory=list)
    safety_alerts: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "plan": self.plan.model_dump(mode="json"),
            "seed_entities": self.seed_entities,
            "timestamp": self.answer_context.timestamp.isoformat(),
            "current_medications": self.current_medications,
            "allergies": self.allergies,
            "safety_alerts": self.safety_alerts,
            "history": self.history,
            "retrieval_metadata": self.answer_context.retrieval_metadata,
            "raw_facts": [
                {
                    "id": fact.id,
                    "relation": fact.relation.value,
                    "subject_id": fact.subject_id,
                    "object_id": fact.object_id,
                    "value": fact.value,
                    "valid_from": fact.valid_from.isoformat(),
                    "valid_to": fact.valid_to.isoformat() if fact.valid_to else None,
                    "attributes": fact.attributes,
                }
                for fact in self.answer_context.facts
            ],
        }


class HealthcareRetrievalPlanner:
    """LLM-backed retrieval planner with deterministic lexical fallbacks."""

    def __init__(
        self,
        memory_store: Any,
        llm_client: Any | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.llm_client = llm_client

    def plan(self, query: str) -> HealthcareQueryPlan:
        fallback = self._fallback_plan(query)
        if self.llm_client is None:
            return fallback

        try:
            llm_plan = self.llm_client.extract(
                text=query,
                output_model=HealthcareQueryPlan,
                system_prompt=HEALTHCARE_QUERY_PLANNER_PROMPT,
            )
        except Exception:
            return fallback

        if llm_plan.as_of is None:
            llm_plan.as_of = fallback.as_of
            llm_plan.raw_time_expression = (
                llm_plan.raw_time_expression or fallback.raw_time_expression
            )
        if not llm_plan.requested_categories:
            llm_plan.requested_categories = fallback.requested_categories
        return llm_plan

    def resolve_seed_entities(
        self,
        *,
        query: str,
        plan: HealthcareQueryPlan,
        scope: dict[str, str] | None = None,
        max_seeds: int = 6,
    ) -> list[str]:
        seed_ids: list[str] = []

        if plan.patient_identifier:
            for entity in self._entities_by_type(EntityType.PATIENT):
                if not self._entity_matches_scope(entity, scope):
                    continue
                if self._entity_identifier_matches(entity, plan.patient_identifier):
                    self._append_unique(seed_ids, entity.id)

        if plan.patient_name:
            for entity in self._entities_by_type(EntityType.PATIENT):
                if not self._entity_matches_scope(entity, scope):
                    continue
                if entity.name.strip().lower() == plan.patient_name.strip().lower():
                    self._append_unique(seed_ids, entity.id)

        for clinical_name, entity_type in [
            *[(name, EntityType.MEDICATION) for name in plan.medication_names],
            *[(name, EntityType.ALLERGY) for name in plan.allergy_names],
        ]:
            for entity in self._entities_by_type(entity_type):
                if not self._entity_matches_scope(entity, scope):
                    continue
                if entity.name.strip().lower() == clinical_name.strip().lower():
                    self._append_unique(seed_ids, entity.id)

        # KB-aware expansion: cross-reactive drugs / therapeutic classes
        for clinical_name, entity_type in [
            *[(name, EntityType.MEDICATION) for name in plan.medication_names],
            *[(name, EntityType.ALLERGY) for name in plan.allergy_names],
        ]:
            expanded_terms = expand_drug_terms(clinical_name)
            for entity in self._entities_by_type(entity_type):
                if not self._entity_matches_scope(entity, scope):
                    continue
                entity_terms = expand_drug_terms(entity.name)
                if entity_terms & expanded_terms:
                    self._append_unique(seed_ids, entity.id)

        if len(seed_ids) < max_seeds and plan.patient_name:
            scored = self._score_patient_name(plan.patient_name, scope=scope)
            for _, entity_id in scored:
                self._append_unique(seed_ids, entity_id)
                if len(seed_ids) >= max_seeds:
                    break

        if len(seed_ids) < max_seeds:
            for entity_id in select_seed_entities(query, self.memory_store, max_seeds=max_seeds):
                entity = self.memory_store.get_entity(entity_id)
                if entity is None or not self._entity_matches_scope(entity, scope):
                    continue
                self._append_unique(seed_ids, entity_id)
                if len(seed_ids) >= max_seeds:
                    break

        return seed_ids[:max_seeds]

    def _fallback_plan(self, query: str) -> HealthcareQueryPlan:
        tokens = _tokens(query)
        categories: list[str] = []
        if tokens & {"medication", "medications", "meds", "prescribed", "prescriptions", "taking"}:
            categories.append("current_medications")
        if tokens & {"allergy", "allergies", "allergic", "allergen"}:
            categories.append("allergies")
        if tokens & {"history", "historical", "previous", "prior", "before", "old"}:
            categories.append("history")
        if tokens & {
            "safety",
            "warning",
            "warnings",
            "interaction",
            "interactions",
            "contraindicated",
        }:
            categories.append("safety_alerts")
        if not categories:
            categories = ["current_medications", "allergies"]

        patient_identifier = _extract_identifier(query)
        patient_name = _extract_patient_name(query)
        as_of, raw_time_expression = _extract_as_of(query)

        return HealthcareQueryPlan(
            patient_name=patient_name,
            patient_identifier=patient_identifier,
            requested_categories=categories,
            safety_focus="safety_alerts" in categories or "allergies" in categories,
            as_of=as_of,
            raw_time_expression=raw_time_expression,
        )

    def _entities_by_type(self, entity_type: EntityType) -> list[Entity]:
        getter = getattr(self.memory_store, "get_entities_by_type", None)
        if callable(getter):
            return list(getter(entity_type) or [])
        entities_getter = getattr(self.memory_store, "get_all_entities", None)
        if callable(entities_getter):
            return [entity for entity in entities_getter() if entity.entity_type == entity_type]
        return []

    @staticmethod
    def _append_unique(values: list[str], value: str) -> None:
        if value not in values:
            values.append(value)

    @staticmethod
    def _entity_identifier_matches(entity: Entity, identifier: str) -> bool:
        expected = _normalize_identifier(identifier)
        candidates = [
            entity.canonical_id,
            entity.attributes.get("identifier"),
            entity.attributes.get("mrn"),
            entity.attributes.get("patient_id"),
        ]
        return any(_normalize_identifier(value) == expected for value in candidates if value)

    @staticmethod
    def _entity_matches_scope(entity: Entity, scope: dict[str, str] | None) -> bool:
        if not scope:
            return True

        attrs = entity.attributes or {}
        expected_scope = scope.get("scope_id")
        actual_scope = attrs.get("scope_id")
        if expected_scope and actual_scope:
            return actual_scope == expected_scope

        for key in ("tenant_id", "app_id", "user_id"):
            expected = scope.get(key)
            actual = attrs.get(key)
            if expected and actual and expected != actual:
                return False
        return True

    def _score_patient_name(
        self,
        patient_name: str,
        *,
        scope: dict[str, str] | None,
    ) -> list[tuple[float, str]]:
        query_tokens = _tokens(patient_name)
        scored: list[tuple[float, str]] = []
        for entity in self._entities_by_type(EntityType.PATIENT):
            if not self._entity_matches_scope(entity, scope):
                continue
            name_tokens = _tokens(entity.name)
            if not name_tokens:
                continue
            overlap = len(query_tokens & name_tokens)
            if overlap:
                scored.append((overlap / len(name_tokens), entity.id))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored


class HealthcareRetrievalService:
    """High-level healthcare retrieval service.

    Encapsulates query planning, graph retrieval, clinical context building,
    and common cross-patient / shared-entity queries so that demos and
    applications never re-implement retrieval orchestration.
    """

    def __init__(
        self,
        memory_store: Any,
        retriever: GraphRetriever,
        llm_client: Any | None = None,
    ) -> None:
        planner = HealthcareRetrievalPlanner(
            memory_store=memory_store,
            llm_client=llm_client,
        )
        self._builder = HealthcareContextBuilder(
            memory_store=memory_store,
            retriever=retriever,
            planner=planner,
        )
        self._store = memory_store

    # ------------------------------------------------------------------
    # Single-patient queries
    # ------------------------------------------------------------------

    def retrieve_current_state(
        self,
        query: str,
        scope: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> HealthcareClinicalContext:
        """Retrieve the current clinical state for a patient query."""
        return self._builder.build(
            query,
            scope=scope,
            strategy=kwargs.pop("strategy", RetrievalStrategy.SAFETY_PRIORITY),
            **kwargs,
        )

    def retrieve_historical_state(
        self,
        query: str,
        as_of: datetime,
        scope: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> HealthcareClinicalContext:
        """Retrieve the clinical state at a specific point in time."""
        # Inject the as_of timestamp into the query so the planner picks it up
        query = f"{query} (as of {as_of.isoformat()})"
        return self._builder.build(
            query,
            scope=scope,
            strategy=kwargs.pop("strategy", RetrievalStrategy.SAFETY_PRIORITY),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Cross-patient / graph-level queries
    # ------------------------------------------------------------------

    def check_cross_patient_isolation(
        self,
        query: str,
        scope: dict[str, str] | None = None,
        forbidden_medication_names: set[str] | None = None,
        **kwargs: Any,
    ) -> tuple[bool, set[str]]:
        """Return (is_isolated, actual_medications) for a patient query.

        ``is_isolated`` is ``True`` when none of the *forbidden* medication
        names appear in the retrieved current medications.
        """
        ctx = self.retrieve_current_state(query, scope=scope, **kwargs)
        actual = {m["medication_name"] for m in ctx.current_medications}
        forbidden = forbidden_medication_names or set()
        return not (actual & forbidden), actual

    def find_patients_by_medication(self, medication_name: str) -> list[str]:
        """Return patient names currently prescribed a given medication.

        This demonstrates entity deduplication: all patients link to the
        same medication node in the graph.
        """
        return self.find_patients_by_shared_entity(medication_name, RelationType.PRESCRIBED)

    def find_patients_by_allergy(self, allergen_name: str) -> list[str]:
        """Return patient names with a documented allergy to a given substance.

        This demonstrates entity deduplication: all patients link to the
        same allergy node in the graph.
        """
        return self.find_patients_by_shared_entity(allergen_name, RelationType.HAS_ALLERGY)

    def find_patients_by_shared_entity(
        self,
        entity_name: str,
        relation: RelationType,
    ) -> list[str]:
        """Return patient names linked to a given entity by a specific relation.

        This verifies that the knowledge graph correctly deduplicates entities
        (e.g. the same ``Penicillin`` allergy node is shared by multiple
        patients, or the same ``Metformin`` medication node is prescribed to
        multiple patients).
        """
        entities = self._store.get_all_entities()
        target = next(
            (e for e in entities if e.name.lower() == entity_name.lower()),
            None,
        )
        if target is None:
            return []

        facts = self._store.get_all_validated_facts()
        patient_ids: set[str] = {
            f.subject_id
            for f in facts
            if f.object_id == target.id and f.relation == relation and f.is_active
        }
        return sorted(
            {
                self._store.get_entity(pid).name
                for pid in patient_ids
                if self._store.get_entity(pid) is not None
            }
        )

    # ------------------------------------------------------------------
    # Grounded answer generation
    # ------------------------------------------------------------------

    def generate_grounded_answer(
        self,
        query: str,
        scope: dict[str, str] | None,
        llm_client: Any,
        **kwargs: Any,
    ) -> str:
        """Generate a strictly-grounded natural-language answer.

        The LLM receives only the structured ``HealthcareClinicalContext`` JSON
        and is instructed never to hallucinate outside of it.
        """
        ctx = self.retrieve_current_state(query, scope=scope, **kwargs)
        answer_prompt = (
            "Answer the clinician strictly from this JSON context. "
            "Mention active medications, allergies, rejected safety alerts, and relevant history. "
            "If something is absent from context, say it is not present in grounded memory.\n\n"
            f"{json.dumps(ctx.to_dict(), indent=2, default=str)}"
        )
        return llm_client.complete(
            prompt=query,
            system_prompt=answer_prompt,
            temperature=kwargs.pop("temperature", 0.1),
            **kwargs,
        )


class HealthcareContextBuilder:
    """Build healthcare-specific clinical views from graph retrieval output."""

    def __init__(
        self,
        memory_store: Any,
        retriever: GraphRetriever,
        planner: HealthcareRetrievalPlanner,
    ) -> None:
        self.memory_store = memory_store
        self.retriever = retriever
        self.planner = planner

    def build(
        self,
        query: str,
        *,
        scope: dict[str, str] | None = None,
        max_seeds: int = 6,
        max_hops: int = 2,
        max_facts: int = 30,
        strategy: RetrievalStrategy = RetrievalStrategy.SAFETY_PRIORITY,
    ) -> HealthcareClinicalContext:
        plan = self.planner.plan(query)
        seed_entities = self.planner.resolve_seed_entities(
            query=query,
            plan=plan,
            scope=scope,
            max_seeds=max_seeds,
        )

        if seed_entities:
            answer_context = self.retriever.retrieve(
                query=query,
                seed_entities=seed_entities,
                max_hops=max_hops,
                max_facts=max_facts,
                strategy=strategy,
                at_time=plan.as_of,
                scope=scope,
            )
        else:
            answer_context = AnswerContext(
                query=query,
                seed_entities=[],
                timestamp=plan.as_of or datetime.now(timezone.utc),
                retrieval_metadata={
                    "strategy": strategy.value,
                    "reason": "no_seed_entities_resolved",
                    "scope": scope or {},
                },
            )

        patient_ids = self._patient_seed_ids(seed_entities)
        facts = answer_context.facts

        return HealthcareClinicalContext(
            query=query,
            plan=plan,
            answer_context=answer_context,
            seed_entities=seed_entities,
            current_medications=self._current_medications(
                facts=facts,
                patient_ids=patient_ids,
                at_time=answer_context.timestamp,
                requested_medications=plan.medication_names,
            ),
            allergies=self._allergies(facts=facts, patient_ids=patient_ids),
            safety_alerts=self._safety_alerts(patient_ids=patient_ids),
            history=self._history(patient_ids=patient_ids, medication_names=plan.medication_names),
        )

    def _patient_seed_ids(self, seed_entities: list[str]) -> set[str]:
        patient_ids: set[str] = set()
        for entity_id in seed_entities:
            entity = self.memory_store.get_entity(entity_id)
            if entity is not None and entity.entity_type == EntityType.PATIENT:
                patient_ids.add(entity.id)
        return patient_ids

    def _current_medications(
        self,
        *,
        facts: list[ValidatedFact],
        patient_ids: set[str],
        at_time: datetime,
        requested_medications: list[str],
    ) -> list[dict[str, Any]]:
        prescribed = [
            fact
            for fact in facts
            if fact.relation == RelationType.PRESCRIBED
            and (not patient_ids or fact.subject_id in patient_ids)
            and fact.is_active_at(at_time)
        ]
        discontinued = [
            fact
            for fact in facts
            if fact.relation == RelationType.DISCONTINUED
            and (not patient_ids or fact.subject_id in patient_ids)
            and fact.is_active_at(at_time)
        ]
        requested_terms = normalized_medication_terms(set(requested_medications))

        active_items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for fact in prescribed:
            if self._is_discontinued(fact, discontinued, at_time=at_time):
                continue
            med_terms = normalized_medication_terms(fact_medication_names(fact, self.memory_store))
            if requested_terms and not (requested_terms & med_terms):
                continue
            if fact.id in seen:
                continue
            seen.add(fact.id)
            active_items.append(self._medication_item(fact))

        active_items.sort(key=lambda item: item["valid_from"], reverse=True)
        return active_items

    def _is_discontinued(
        self,
        prescription: ValidatedFact,
        discontinued: list[ValidatedFact],
        *,
        at_time: datetime,
    ) -> bool:
        prescription_terms = normalized_medication_terms(
            fact_medication_names(prescription, self.memory_store)
        )
        for fact in discontinued:
            if datetime_after(fact.valid_from, at_time):
                continue
            if prescription.object_id and fact.object_id == prescription.object_id:
                return True
            discontinued_terms = normalized_medication_terms(
                fact_medication_names(fact, self.memory_store)
            )
            if (
                prescription_terms
                and discontinued_terms
                and prescription_terms & discontinued_terms
            ):
                return True
        return False

    def _allergies(
        self,
        *,
        facts: list[ValidatedFact],
        patient_ids: set[str],
    ) -> list[dict[str, Any]]:
        allergies: list[dict[str, Any]] = []
        seen: set[str] = set()
        for fact in facts:
            if fact.relation != RelationType.HAS_ALLERGY:
                continue
            if patient_ids and fact.subject_id not in patient_ids:
                continue
            if fact.id in seen:
                continue
            seen.add(fact.id)
            entity = self.memory_store.get_entity(fact.object_id) if fact.object_id else None
            allergies.append(
                {
                    "fact_id": fact.id,
                    "allergen": (
                        fact.attributes.get("allergen_name")
                        or (entity.name if entity else None)
                        or fact.value
                    ),
                    "reaction": fact.attributes.get("reaction"),
                    "severity": fact.attributes.get("severity"),
                    "valid_from": fact.valid_from.isoformat(),
                    "source_text": fact.source_text or fact.attributes.get("source_text"),
                }
            )
        allergies.sort(key=lambda item: item["valid_from"], reverse=True)
        return allergies

    def _safety_alerts(self, *, patient_ids: set[str]) -> list[dict[str, Any]]:
        getter = getattr(self.memory_store, "get_all_rejections", None)
        if not callable(getter):
            return []

        alerts: list[dict[str, Any]] = []
        for rejection in getter():
            # Scope isolation: only include rejections for the target patients.
            if rejection.subject_entity_id not in patient_ids:
                continue
            alerts.append(
                {
                    "rejection_id": rejection.id,
                    "candidate_fact_id": rejection.candidate_fact_id,
                    "constraint_id": rejection.constraint_id,
                    "constraint_name": rejection.constraint_name,
                    "reason": rejection.reason,
                    "domain_reasoning": rejection.domain_reasoning,
                    "alternatives": rejection.alternatives,
                    "severity": rejection.severity,
                    "rejected_at": rejection.rejected_at.isoformat(),
                }
            )
        return alerts

    def _history(
        self,
        *,
        patient_ids: set[str],
        medication_names: list[str],
    ) -> list[dict[str, Any]]:
        if not patient_ids:
            return []

        requested_terms = normalized_medication_terms(set(medication_names))
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for patient_id in patient_ids:
            facts = self.memory_store.get_facts_for_entity(
                patient_id,
                include_superseded=True,
            )
            for fact in facts:
                if fact.relation not in {RelationType.PRESCRIBED, RelationType.DISCONTINUED}:
                    continue
                med_terms = normalized_medication_terms(
                    fact_medication_names(fact, self.memory_store)
                )
                if requested_terms and not (requested_terms & med_terms):
                    continue
                if fact.id in seen:
                    continue
                seen.add(fact.id)
                rows.append(self._medication_item(fact))

        rows.sort(key=lambda item: item["valid_from"], reverse=True)
        return rows

    def _medication_item(self, fact: ValidatedFact) -> dict[str, Any]:
        entity = self.memory_store.get_entity(fact.object_id) if fact.object_id else None
        medication_name = (
            fact.attributes.get("medication_name")
            or (entity.name if entity is not None else None)
            or fact.value
        )
        return {
            "fact_id": fact.id,
            "relation": fact.relation.value,
            "medication_id": fact.object_id,
            "medication_name": medication_name,
            "normalized_name": fact.attributes.get("normalized_name")
            or (normalize_drug_name(str(medication_name)) if medication_name else None),
            "dosage": fact.attributes.get("dosage"),
            "frequency": fact.attributes.get("frequency"),
            "route": fact.attributes.get("route"),
            "action": fact.attributes.get("action"),
            "order_status": fact.attributes.get("order_status"),
            "value": fact.value,
            "valid_from": fact.valid_from.isoformat(),
            "valid_to": fact.valid_to.isoformat() if fact.valid_to else None,
            "active": fact.is_active,
            "source_text": fact.source_text or fact.attributes.get("source_text"),
        }


def _tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) >= 2}


def _normalize_identifier(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _extract_identifier(query: str) -> str | None:
    patterns = [
        r"\b(?:mrn|patient\s*id|identifier)\s*[:#]?\s*([A-Za-z0-9][A-Za-z0-9_-]{1,40})",
        r"\b([A-Z]{2,}-\d{2,})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _extract_patient_name(query: str) -> str | None:
    patterns = [
        r"\b[Pp]atient\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
        r"\b[Ff]or\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
        r"\b[Aa]bout\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})['\u2019]s\s+(?:medication|medications|meds|allergy|allergies|prescription|prescriptions)",
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s+(?:is|are|was|were)\s+(?:on|taking|prescribed|allergic)",
        r"\b[Ww]hich\s+(?:medication|medications|meds|allergy|allergies)\s+(?:is|are|was|were)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
    ]
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            return match.group(1).strip()
    return None


def _extract_as_of(query: str) -> tuple[datetime | None, str | None]:
    iso_match = re.search(
        r"\b(20\d{2}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?)\b",
        query,
    )
    if iso_match:
        raw = iso_match.group(1)
        try:
            return date_parser.parse(raw), raw
        except (ValueError, TypeError):
            pass

    as_of_match = re.search(r"\b(?:as of|at|on|before)\s+([^?.!,;]+)", query, flags=re.IGNORECASE)
    if as_of_match:
        raw = as_of_match.group(1).strip()
        try:
            return date_parser.parse(raw, fuzzy=True), raw
        except (ValueError, TypeError, OverflowError):
            return None, raw

    return None, None
