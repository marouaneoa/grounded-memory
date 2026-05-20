"""
Healthcare LLM Extractor

Uses Pydantic AI and an LLM to extract structured clinical facts from text.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING

from grounded_memory.adapters.healthcare.knowledge import normalize_drug_name
from grounded_memory.adapters.healthcare.models import (
    ClinicalExtractionResult,
    ExtractedMedicationLLM,
)
from grounded_memory.core.entity_identity import build_entity_uniqueness_key, stable_entity_id
from grounded_memory.llm.client import LLMConfig, SyncLLMClient
from grounded_memory.llm.prompts import (
    CLINICAL_EXTRACTION_SYSTEM_PROMPT,
    CONNECTIVITY_TEST_SYSTEM_PROMPT,
    build_clinical_extraction_user_prompt,
)

if TYPE_CHECKING:
    from grounded_memory.core.models import (
        CandidateFact,
        Entity,
        EntityType,
        Interaction,
        RelationType,
    )


@dataclass
class HealthcareLLMExtractor:
    """
    LLM-powered fact extractor for clinical text.
    """

    config: LLMConfig = None
    client: SyncLLMClient = None

    def __post_init__(self):
        if self.config is None:
            self.config = LLMConfig.from_env()
        if self.client is None:
            self.client = SyncLLMClient(self.config)

    def extract(
        self,
        text: str,
        include_context: str | None = None,
    ) -> ClinicalExtractionResult:
        prompt = self._build_extraction_prompt(text, include_context)

        result = self.client.extract(
            text=prompt,
            output_model=ClinicalExtractionResult,
            system_prompt=CLINICAL_EXTRACTION_SYSTEM_PROMPT,
        )
        return result

    def extract_medications_only(self, text: str) -> list[ExtractedMedicationLLM]:
        result = self.extract(text)
        return result.medications

    def _build_extraction_prompt(self, text: str, context: str | None = None) -> str:
        return build_clinical_extraction_user_prompt(input_text=text, context_text=context)

    def test_connection(self) -> bool:
        try:
            response = self.client.complete(
                "Respond with exactly: OK",
                system_prompt=CONNECTIVITY_TEST_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=10,
            )
            return "OK" in response
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False


class HealthcareDatabaseExtractor:
    """
    LLM-powered extractor integrated with the database for the clinical domain.
    """

    def __init__(self, store, llm_extractor: HealthcareLLMExtractor | None = None):
        self.store = store
        self.llm_extractor = llm_extractor or HealthcareLLMExtractor()

    async def process_interaction(
        self,
        raw_text: str,
        user_id: str | None = None,
        session_id: str | None = None,
        actor: str = "user",
        metadata: dict | None = None,
    ) -> ExtractionPipelineResult:
        from grounded_memory.core.models import (
            ActorType,
            CandidateFact,
            EntityType,
            Interaction,
            RelationType,
        )

        metadata = dict(metadata or {})
        scope_attributes = self._scope_attributes(metadata, user_id=user_id, session_id=session_id)

        interaction = Interaction(
            tenant_id=scope_attributes.get("tenant_id"),
            app_id=scope_attributes.get("app_id"),
            user_id=user_id,
            agent_id=scope_attributes.get("agent_id"),
            run_id=scope_attributes.get("run_id"),
            session_id=session_id,
            space_type=scope_attributes.get("space_type"),
            actor=ActorType(actor),
            raw_text=raw_text,
            metadata={**metadata, **scope_attributes},
        )

        await self._maybe_await(self.store.add_interaction(interaction))

        extraction_result = self.llm_extractor.extract(raw_text)

        entities: list[Entity] = []
        candidate_facts: list[CandidateFact] = []

        patient_entity = None
        if extraction_result.patient:
            patient_identifier = (
                extraction_result.patient.identifier
                or (metadata or {}).get("patient_id")
                or (metadata or {}).get("mrn")
            )
            patient_entity = await self._get_or_create_entity(
                name=extraction_result.patient.name,
                entity_type=EntityType.PATIENT,
                canonical_id=patient_identifier,
                strict_identity=bool(patient_identifier),
                attributes={
                    **scope_attributes,
                    "identifier": patient_identifier,
                    "age": extraction_result.patient.age,
                    "gender": extraction_result.patient.gender,
                },
            )
            entities.append(patient_entity)

        for med in extraction_result.medications:
            normalized_name = normalize_drug_name(med.name)
            relation = self._action_to_relation(med.action)
            order_status = self._action_to_order_status(med.action)
            med_entity = await self._get_or_create_entity(
                name=med.name,
                entity_type=EntityType.MEDICATION,
                attributes={
                    "normalized_name": normalized_name,
                },
            )
            entities.append(med_entity)

            if patient_entity:
                candidate_fact = CandidateFact(
                    source_interaction_id=interaction.id,
                    subject_entity_id=patient_entity.id,
                    relation=relation,
                    object_entity_id=med_entity.id,
                    value=f"{med.dosage or ''} {med.frequency or ''}".strip() or None,
                    confidence=med.confidence,
                    attributes={
                        **scope_attributes,
                        "medication_name": med.name,
                        "normalized_name": normalized_name,
                        "action": med.action,
                        "order_status": order_status,
                        "dosage": med.dosage,
                        "frequency": med.frequency,
                        "route": med.route,
                        "duration": med.duration,
                        "source_text": raw_text,
                    },
                )
                candidate_facts.append(candidate_fact)

        for condition in extraction_result.conditions:
            cond_entity = await self._get_or_create_entity(
                name=condition.name,
                entity_type=EntityType.CONDITION,
                attributes={
                    **scope_attributes,
                    "status": condition.status,
                    "diagnosed_date": condition.diagnosed_date,
                },
            )
            entities.append(cond_entity)

            if patient_entity:
                candidate_fact = CandidateFact(
                    source_interaction_id=interaction.id,
                    subject_entity_id=patient_entity.id,
                    relation=RelationType.HAS_CONDITION,
                    object_entity_id=cond_entity.id,
                    confidence=0.9,
                    attributes={
                        **scope_attributes,
                        "status": condition.status,
                        "source_text": raw_text,
                    },
                )
                candidate_facts.append(candidate_fact)

        for allergy in extraction_result.allergies:
            normalized_allergen = normalize_drug_name(allergy.allergen)
            allergy_entity = await self._get_or_create_entity(
                name=allergy.allergen,
                entity_type=EntityType.ALLERGY,
                attributes={
                    "normalized_name": normalized_allergen,
                    "reaction": allergy.reaction,
                    "severity": allergy.severity,
                },
            )
            entities.append(allergy_entity)

            if patient_entity:
                candidate_fact = CandidateFact(
                    source_interaction_id=interaction.id,
                    subject_entity_id=patient_entity.id,
                    relation=RelationType.HAS_ALLERGY,
                    object_entity_id=allergy_entity.id,
                    confidence=0.95,
                    attributes={
                        **scope_attributes,
                        "allergen_name": allergy.allergen,
                        "normalized_name": normalized_allergen,
                        "reaction": allergy.reaction,
                        "severity": allergy.severity,
                        "source_text": raw_text,
                    },
                )
                candidate_facts.append(candidate_fact)

        if candidate_facts:
            await self._maybe_await(self.store.add_candidate_facts(candidate_facts))

        return ExtractionPipelineResult(
            interaction=interaction,
            extraction_result=extraction_result,
            entities=entities,
            candidate_facts=candidate_facts,
        )

    async def _get_or_create_entity(
        self,
        name: str,
        entity_type: EntityType,
        attributes: dict | None = None,
        canonical_id: str | None = None,
        strict_identity: bool = False,
    ) -> Entity:
        from grounded_memory.core.models import Entity

        normalized_attributes = {k: v for k, v in (attributes or {}).items() if v is not None}

        if strict_identity and canonical_id:
            existing_by_id = await self._find_entity_by_canonical_id(entity_type, canonical_id)
            if existing_by_id:
                for key, value in normalized_attributes.items():
                    if key not in existing_by_id.attributes:
                        existing_by_id.attributes[key] = value
                await self._maybe_await(self.store.add_entity(existing_by_id))
                return existing_by_id

            uniqueness_key = build_entity_uniqueness_key(
                name=name,
                entity_type=entity_type,
                attributes=normalized_attributes,
                canonical_id=canonical_id,
            )
            entity = Entity(
                id=stable_entity_id(uniqueness_key),
                entity_type=entity_type,
                name=name,
                canonical_id=canonical_id,
                attributes=normalized_attributes,
            )
            await self._maybe_await(self.store.add_entity(entity))
            return entity

        existing = await self._maybe_await(self.store.find_entity_by_name(name, entity_type))

        if existing:
            if normalized_attributes:
                for key, value in normalized_attributes.items():
                    if value is not None and key not in existing.attributes:
                        existing.attributes[key] = value
                if canonical_id and not existing.canonical_id:
                    existing.canonical_id = canonical_id
                await self._maybe_await(self.store.add_entity(existing))
            return existing

        entity = Entity(
            entity_type=entity_type,
            name=name,
            canonical_id=canonical_id,
            attributes=normalized_attributes,
        )
        await self._maybe_await(self.store.add_entity(entity))
        return entity

    async def _find_entity_by_canonical_id(
        self,
        entity_type: EntityType,
        canonical_id: str,
    ) -> Entity | None:
        normalized = canonical_id.strip().lower()
        if not normalized:
            return None

        getter = getattr(self.store, "get_entities_by_type", None)
        if not callable(getter):
            return None

        entities = await self._maybe_await(getter(entity_type))
        for entity in entities or []:
            existing_canonical = (getattr(entity, "canonical_id", None) or "").strip().lower()
            existing_identifier = (
                str(getattr(entity, "attributes", {}).get("identifier", "")).strip().lower()
            )
            if existing_canonical == normalized or existing_identifier == normalized:
                return entity

        return None

    async def _maybe_await(self, value):
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    def _scope_attributes(
        metadata: dict,
        *,
        user_id: str | None,
        session_id: str | None,
    ) -> dict[str, str]:
        scope: dict[str, str] = {}
        for key in (
            "tenant_id",
            "app_id",
            "user_id",
            "agent_id",
            "run_id",
            "space_type",
            "scope_id",
        ):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                scope[key] = value.strip()

        if user_id and "user_id" not in scope:
            scope["user_id"] = user_id
        if session_id and "run_id" not in scope:
            scope["run_id"] = session_id

        if (
            "scope_id" not in scope
            and scope.get("tenant_id")
            and scope.get("app_id")
            and scope.get("user_id")
        ):
            scope["scope_id"] = f"{scope['tenant_id']}:{scope['app_id']}:{scope['user_id']}"

        return scope

    def _action_to_relation(self, action: str) -> RelationType:
        from grounded_memory.core.models import RelationType

        action_map = {
            "prescribe": RelationType.PRESCRIBED,
            "discontinue": RelationType.DISCONTINUED,
            "continue": RelationType.PRESCRIBED,
            "adjust": RelationType.PRESCRIBED,
            "hold": RelationType.DISCONTINUED,
        }
        return action_map.get(action.lower(), RelationType.PRESCRIBED)

    def _action_to_order_status(self, action: str) -> str:
        normalized = (action or "").strip().lower()
        if normalized in {"discontinue", "hold"}:
            return "discontinued"
        if normalized == "adjust":
            return "adjusted"
        return "active"


@dataclass
class ExtractionPipelineResult:
    """Result from the healthcare extraction pipeline."""

    interaction: Interaction
    extraction_result: ClinicalExtractionResult
    entities: list[Entity]
    candidate_facts: list[CandidateFact]

    @property
    def patient_entity(self) -> Entity | None:
        from grounded_memory.core.models import EntityType

        for entity in self.entities:
            if entity.entity_type == EntityType.PATIENT:
                return entity
        return None

    @property
    def medication_entities(self) -> list[Entity]:
        from grounded_memory.core.models import EntityType

        return [e for e in self.entities if e.entity_type == EntityType.MEDICATION]

    @property
    def has_candidate_facts(self) -> bool:
        return len(self.candidate_facts) > 0
