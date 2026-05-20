"""Built-in generic LLM-backed agent for open-domain memory formation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from grounded_memory.core.entity_identity import (
    build_entity_uniqueness_key,
    stable_entity_id,
)
from grounded_memory.core.grounding import GroundingDecision, GroundingResult
from grounded_memory.core.models import (
    ActorType,
    CandidateFact,
    Entity,
    EntityType,
    Interaction,
    MemoryDisposition,
    RelationType,
    ValidatedFact,
)
from grounded_memory.core.tuple_normalization import (
    normalize_attribute_key,
    normalize_fact_attributes,
    normalize_fact_value_for_match,
    parse_keyed_value,
    resolve_attribute_key,
    sanitize_fact_value,
    should_materialize_attribute_object,
)
from grounded_memory.llm.client import LLMConfig, SyncLLMClient
from grounded_memory.llm.prompts import (
    GENERIC_TUPLE_EXTRACTION_SYSTEM_PROMPT,
    build_generic_tuple_extraction_user_prompt,
)


class GenericExtractedFact(BaseModel):
    """Open-domain fact extracted from natural language."""

    subject_name: str = Field(..., description="Canonical subject mention.")
    subject_type: EntityType = Field(..., description="Type of the subject entity.")
    relation: RelationType = Field(
        default=RelationType.HAS_ATTRIBUTE, description="Type of relationship."
    )
    object_name: str | None = Field(default=None)
    object_type: EntityType | None = Field(
        default=None, description="Type of the object entity, if applicable."
    )
    value: str | None = Field(default=None)
    disposition: MemoryDisposition = Field(default=MemoryDisposition.CAPTURE)
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    attributes: dict[str, Any] = Field(default_factory=dict)


class GenericExtractionResult(BaseModel):
    """Structured generic extraction payload."""

    facts: list[GenericExtractedFact] = Field(default_factory=list)


GENERIC_EXTRACTION_SYSTEM_PROMPT = GENERIC_TUPLE_EXTRACTION_SYSTEM_PROMPT


@dataclass
class GenericProcessingResult:
    """Processing result for the built-in generic agent."""

    interaction_id: str
    grounding_results: list[GroundingResult] = field(default_factory=list)
    approved_facts: list[ValidatedFact] = field(default_factory=list)
    rejected_facts: list[GroundingResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dispositions: list[dict[str, Any]] = field(default_factory=list)


class GenericMemoryAgent:
    """LLM-backed generic agent for open-domain memory writes."""

    def __init__(
        self,
        *,
        memory_store: Any,
        grounding_operator: Any,
        llm_config: LLMConfig | None = None,
        adapter_key: str = "generic",
        domain_profile: str | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.grounding_operator = grounding_operator
        resolved_adapter = (domain_profile or adapter_key).strip().lower()
        self.adapter_key = resolved_adapter
        self.domain_profile = resolved_adapter  # Backward-compatible alias
        self.llm_config = llm_config or LLMConfig.from_env()
        self.client = SyncLLMClient(self.llm_config)

    def process(
        self,
        input_text: str,
        source: str = "user",
        *,
        tenant_id: str | None = None,
        app_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        space_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        fact: dict[str, Any] | None = None,
        **_: Any,
    ) -> GenericProcessingResult:
        normalized_source = source.strip().lower()
        resolved_run_id = run_id or session_id
        scope_metadata = {
            "tenant_id": tenant_id,
            "app_id": app_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "run_id": resolved_run_id,
            "space_type": space_type,
        }
        if tenant_id and app_id and user_id:
            scope_metadata["scope_id"] = f"{tenant_id}:{app_id}:{user_id}"

        merged_metadata = {
            "adapter": self.adapter_key,
            "domain_profile": self.domain_profile,
        }
        if metadata:
            merged_metadata.update(metadata)
        merged_metadata.update({k: v for k, v in scope_metadata.items() if v is not None})

        interaction = Interaction(
            actor=self._coerce_actor(normalized_source),
            raw_text=input_text,
            tenant_id=tenant_id,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            run_id=resolved_run_id,
            session_id=session_id or resolved_run_id,
            space_type=space_type,
            metadata=merged_metadata,
        )
        self.memory_store.add_interaction(interaction)

        result = GenericProcessingResult(interaction_id=interaction.id)

        if fact is not None:
            self._ground_structured_fact(interaction.id, fact, result)
            return result

        if normalized_source in {"assistant", "agent"}:
            result.warnings.append(
                "generic agent skipped assistant/agent free text to prevent self-referential memory noise"
            )
            result.dispositions.append(
                self._disposition_event(
                    MemoryDisposition.PASS,
                    reason="assistant_or_agent_source",
                )
            )
            return result

        if self._should_skip_unstructured_text(input_text, source=normalized_source):
            result.warnings.append("generic agent skipped non-durable or assistant-style text")
            result.dispositions.append(
                self._disposition_event(
                    MemoryDisposition.PASS, reason="non_durable_or_assistant_text"
                )
            )
            return result

        extracted = self._extract_facts(input_text, user_id=user_id, source=normalized_source)
        for item in extracted.facts:
            disposition = self._coerce_disposition(item.disposition)

            if disposition == MemoryDisposition.PASS:
                result.dispositions.append(
                    self._disposition_event(
                        MemoryDisposition.PASS,
                        reason="llm_marked_pass",
                        payload=item.model_dump(mode="json"),
                    )
                )
                continue

            if disposition == MemoryDisposition.RETIRE:
                retired = self._retire_matching_facts(
                    item,
                    scope={k: v for k, v in scope_metadata.items() if v is not None},
                )
                result.dispositions.append(
                    self._disposition_event(
                        MemoryDisposition.RETIRE,
                        reason="retired_matching_facts"
                        if retired
                        else "no_matching_fact_to_retire",
                        fact_ids=retired,
                        payload=item.model_dump(mode="json"),
                    )
                )
                continue

            try:
                candidate = self._candidate_from_extracted(
                    item,
                    interaction.id,
                    user_id=user_id,
                    scope={k: v for k, v in scope_metadata.items() if v is not None},
                )
            except ValueError as exc:
                result.warnings.append(str(exc))
                continue

            grounding = self.grounding_operator.ground(candidate)
            result.grounding_results.append(grounding)

            if grounding.is_success and grounding.validated_fact is not None:
                result.approved_facts.append(grounding.validated_fact)
                result.dispositions.append(self._disposition_from_grounding(grounding))
            else:
                result.rejected_facts.append(grounding)
                result.dispositions.append(
                    self._disposition_event(
                        MemoryDisposition.PASS,
                        reason=f"grounding_{grounding.decision.value}",
                        payload=item.model_dump(mode="json"),
                    )
                )

        return result

    def _ground_structured_fact(
        self,
        interaction_id: str,
        fact: dict[str, Any],
        result: GenericProcessingResult,
    ) -> None:
        relation = self._coerce_relation(str(fact["relation"]))
        normalized_value = sanitize_fact_value(fact.get("value"))
        attributes = normalize_fact_attributes(normalized_value, dict(fact.get("attributes", {})))
        object_entity_id = fact.get("object_entity_id")

        if relation == RelationType.HAS_ATTRIBUTE:
            parsed_key, parsed_tail = parse_keyed_value(normalized_value)
            resolved_key = normalize_attribute_key(attributes.get("key")) or parsed_key
            if resolved_key is not None:
                attributes["key"] = resolved_key
            if parsed_tail is not None:
                normalized_value = parsed_tail
            if object_entity_id is None and should_materialize_attribute_object(normalized_value):
                scope_attrs = {
                    key: value
                    for key, value in attributes.items()
                    if key
                    in {
                        "tenant_id",
                        "app_id",
                        "user_id",
                        "agent_id",
                        "run_id",
                        "space_type",
                        "scope_id",
                    }
                }
                object_entity = self._get_or_create_entity(
                    normalized_value,
                    EntityType.FACILITY,
                    attributes=scope_attrs,
                )
                object_entity_id = object_entity.id

        candidate = CandidateFact(
            source_interaction_id=interaction_id,
            subject_entity_id=str(fact["subject_entity_id"]),
            relation=relation,
            object_entity_id=object_entity_id,
            value=normalized_value,
            confidence=float(fact.get("confidence", 0.9)),
            attributes=attributes,
        )
        grounding = self.grounding_operator.ground(candidate)
        result.grounding_results.append(grounding)

        if grounding.is_success and grounding.validated_fact is not None:
            result.approved_facts.append(grounding.validated_fact)
            result.dispositions.append(self._disposition_from_grounding(grounding))
        else:
            result.rejected_facts.append(grounding)
            result.dispositions.append(
                self._disposition_event(
                    MemoryDisposition.PASS,
                    reason=f"grounding_{grounding.decision.value}",
                    payload=fact,
                )
            )

    def _extract_facts(
        self,
        text: str,
        *,
        user_id: str | None,
        source: str,
    ) -> GenericExtractionResult:
        prompt = self._build_extraction_prompt(text, user_id=user_id, source=source)
        try:
            extracted = self.client.extract(
                prompt,
                GenericExtractionResult,
                system_prompt=GENERIC_EXTRACTION_SYSTEM_PROMPT,
            )
            return self._dedupe_and_validate(extracted)
        except Exception:
            return self._heuristic_extract(text, user_id=user_id, source=source)

    def _build_extraction_prompt(self, text: str, *, user_id: str | None, source: str) -> str:
        return build_generic_tuple_extraction_user_prompt(
            input_text=text,
            source_actor=source,
            user_identifier=user_id,
        )

    def _heuristic_extract(
        self,
        text: str,
        *,
        user_id: str | None,
        source: str,
    ) -> GenericExtractionResult:
        if source == "assistant":
            return GenericExtractionResult(facts=[])

        subject_name = f"user:{user_id}" if user_id else "speaker"
        normalized = text.strip()
        facts: list[GenericExtractedFact] = []

        patterns = [
            (r"\bI no longer use\s+([^.!?]+)", "uses", MemoryDisposition.RETIRE),
            (r"\bI stopped using\s+([^.!?]+)", "uses", MemoryDisposition.RETIRE),
            (r"\bdelete my preference for\s+([^.!?]+)", "prefers", MemoryDisposition.RETIRE),
            (r"\bremove my preference for\s+([^.!?]+)", "prefers", MemoryDisposition.RETIRE),
            (r"\bI use\s+([^.!?]+)", "uses"),
            (r"\bI like\s+([^.!?]+)", "likes"),
            (r"\bI prefer\s+([^.!?]+)", "prefers"),
            (r"\bI am\s+([A-Za-z][^.!?]*)", "identity"),
            (r"\bmy name is\s+([^.!?]+)", "name"),
            (r"\bI work on\s+([^.!?]+)", "works_on"),
            (r"\bI live in\s+([^.!?]+)", "location"),
            (r"\bI am from\s+([^.!?]+)", "origin"),
        ]

        for entry in patterns:
            pattern = entry[0]
            key = entry[1]
            disposition = entry[2] if len(entry) > 2 else MemoryDisposition.CAPTURE
            match = re.search(pattern, normalized, re.IGNORECASE)
            if not match:
                continue
            value = self._normalize_text_fragment(match.group(1))
            if self._looks_like_summary(value or ""):
                continue
            if value:
                facts.append(
                    GenericExtractedFact(
                        subject_name=subject_name,
                        subject_type=EntityType.PERSON,
                        relation=RelationType.HAS_ATTRIBUTE,
                        value=f"{key}={value}",
                        disposition=disposition,
                        confidence=0.8,
                        attributes={"key": key},
                    )
                )

        key_value_match = re.search(
            r"^\s*([A-Za-z_][A-Za-z0-9_\- ]{1,40})\s*(?:=|:)\s*([^\n.!?]{1,120})\s*$",
            normalized,
        )
        if key_value_match:
            key = key_value_match.group(1).strip().lower().replace(" ", "_")
            value = self._normalize_text_fragment(key_value_match.group(2))
            if self._looks_like_summary(value or ""):
                return self._dedupe_and_validate(GenericExtractionResult(facts=facts))
            if value:
                facts.append(
                    GenericExtractedFact(
                        subject_name=subject_name,
                        subject_type=EntityType.PERSON,
                        relation=RelationType.HAS_ATTRIBUTE,
                        value=f"{key}={value}",
                        disposition=MemoryDisposition.CAPTURE,
                        confidence=0.7,
                        attributes={"key": key, "source": "heuristic_key_value"},
                    )
                )

        return self._dedupe_and_validate(GenericExtractionResult(facts=facts))

    def _dedupe_and_validate(self, extracted: GenericExtractionResult) -> GenericExtractionResult:
        seen: set[tuple[str, str, str, str]] = set()
        facts: list[GenericExtractedFact] = []

        for fact in extracted.facts:
            fact.subject_name = self._normalize_text_fragment(fact.subject_name) or ""
            fact.object_name = self._normalize_text_fragment(fact.object_name)
            fact.value = sanitize_fact_value(fact.value)
            fact.attributes = dict(fact.attributes or {})

            if isinstance(fact.relation, RelationType):
                relation_value = fact.relation.value
            else:
                relation_value = str(fact.relation).strip().upper()
                try:
                    fact.relation = RelationType(relation_value)
                except ValueError:
                    # Dynamic fallback based on presence of object_name
                    fact.relation = (
                        RelationType.RELATED_TO if fact.object_name else RelationType.HAS_ATTRIBUTE
                    )
                    relation_value = fact.relation.value

            if relation_value == RelationType.HAS_ATTRIBUTE.value:
                parsed_key, parsed_tail = parse_keyed_value(fact.value)
                resolved_key = normalize_attribute_key(fact.attributes.get("key")) or parsed_key
                if resolved_key is not None:
                    fact.attributes["key"] = resolved_key
                if parsed_tail is not None:
                    fact.value = parsed_tail

            subject_name = fact.subject_name.strip()
            object_name = (fact.object_name or "").strip()
            value = (fact.value or "").strip()

            if not subject_name:
                continue
            if not object_name and not value:
                continue
            if self._looks_like_summary(value):
                continue
            if self._looks_like_summary(object_name):
                continue

            if (
                relation_value == RelationType.HAS_ATTRIBUTE.value
                and not object_name
                and resolve_attribute_key(value, fact.attributes) is None
            ):
                continue

            key = (
                subject_name.lower(),
                relation_value,
                object_name.lower(),
                normalize_fact_value_for_match(value) or "",
            )
            if key in seen:
                continue
            seen.add(key)
            facts.append(fact)

        return GenericExtractionResult(facts=facts)

    @staticmethod
    def _looks_like_summary(value: str) -> bool:
        lowered = value.strip().lower()
        if not lowered:
            return False

        if "```" in lowered or '{"' in lowered or '"facts"' in lowered:
            return True

        bad_prefixes = (
            "noted",
            "understood",
            "the user said",
            "the assistant said",
            "summary=",
            "message=",
            "response=",
        )
        if lowered.startswith(bad_prefixes):
            return True
        return len(lowered.split()) > 16 and "=" not in lowered

    @staticmethod
    def _normalize_text_fragment(value: str | None) -> str | None:
        if value is None:
            return None
        collapsed = re.sub(r"\s+", " ", str(value)).strip().strip("\"'")
        collapsed = collapsed.rstrip(".!")
        return collapsed or None

    @staticmethod
    def _should_skip_unstructured_text(text: str, *, source: str) -> bool:
        lowered = text.strip().lower()
        if not lowered:
            return True
        if source == "assistant":
            return lowered.startswith(
                ("noted", "understood", "got it", "okay", "sure", "i understand")
            )
        return lowered in {"ok", "okay", "thanks", "thank you"}

    def _candidate_from_extracted(
        self,
        item: GenericExtractedFact,
        interaction_id: str,
        *,
        user_id: str | None,
        scope: dict[str, str],
    ) -> CandidateFact:
        entity_scope_attrs = {
            key: value
            for key, value in scope.items()
            if key
            in {"tenant_id", "app_id", "user_id", "agent_id", "run_id", "space_type", "scope_id"}
        }
        # Smart coercion: if subject_name starts with "user:" or "speaker", it is definitively a PERSON.
        is_user_centric = item.subject_name.startswith("user:") or item.subject_name == "speaker"
        resolved_subject_type = (
            EntityType.PERSON
            if is_user_centric
            else self._coerce_entity_type(item.subject_type, default=EntityType.CONCEPT)
        )

        subject = self._get_or_create_entity(
            item.subject_name,
            resolved_subject_type,
            attributes={
                **(
                    {"user_id": user_id}
                    if user_id and item.subject_name.startswith("user:")
                    else {}
                ),
                **entity_scope_attrs,
            },
        )

        object_entity_id: str | None = None
        if item.object_name:
            object_entity = self._get_or_create_entity(
                item.object_name,
                self._coerce_entity_type(item.object_type, default=EntityType.CONCEPT),
                attributes=entity_scope_attrs,
            )
            object_entity_id = object_entity.id

        relation = self._coerce_relation_dynamic(item.relation, has_object=bool(object_entity_id))
        normalized_value = sanitize_fact_value(item.value)
        attributes = normalize_fact_attributes(normalized_value, dict(item.attributes))

        if relation == RelationType.HAS_ATTRIBUTE:
            parsed_key, parsed_tail = parse_keyed_value(normalized_value)
            resolved_key = normalize_attribute_key(attributes.get("key")) or parsed_key
            if resolved_key is not None:
                attributes["key"] = resolved_key
            if parsed_tail is not None:
                normalized_value = parsed_tail

            if object_entity_id is None and should_materialize_attribute_object(normalized_value):
                object_entity = self._get_or_create_entity(
                    normalized_value,
                    EntityType.FACILITY,
                    attributes=entity_scope_attrs,
                )
                object_entity_id = object_entity.id

        if object_entity_id is None and (
            normalized_value is None or not str(normalized_value).strip()
        ):
            raise ValueError("generic extracted fact requires object_name or value")

        return CandidateFact(
            source_interaction_id=interaction_id,
            subject_entity_id=subject.id,
            relation=relation,
            object_entity_id=object_entity_id,
            value=normalized_value,
            confidence=item.confidence,
            attributes={
                **attributes,
                **entity_scope_attrs,
                "disposition": self._coerce_disposition(item.disposition).value,
            },
        )

    def _retire_matching_facts(
        self, item: GenericExtractedFact, *, scope: dict[str, str]
    ) -> list[str]:
        user_id = scope.get("user_id")
        entity_scope_attrs = {
            key: value
            for key, value in scope.items()
            if key
            in {"tenant_id", "app_id", "user_id", "agent_id", "run_id", "space_type", "scope_id"}
        }
        is_user_centric = item.subject_name.startswith("user:") or item.subject_name == "speaker"
        resolved_subject_type = (
            EntityType.PERSON
            if is_user_centric
            else self._coerce_entity_type(item.subject_type, default=EntityType.CONCEPT)
        )

        subject = self._get_or_create_entity(
            item.subject_name,
            resolved_subject_type,
            attributes={
                **(
                    {"user_id": user_id}
                    if user_id and item.subject_name.startswith("user:")
                    else {}
                ),
                **entity_scope_attrs,
            },
        )
        relation = self._coerce_relation_dynamic(item.relation, has_object=bool(item.object_name))
        active = self.memory_store.get_facts_by_relation(subject.id, relation, as_subject=True)
        retire_key = self._infer_attribute_key(item.value, dict(item.attributes))

        retired: list[str] = []
        for fact in active:
            if not fact.is_active:
                continue
            if scope and not self._fact_matches_scope(fact, scope=scope):
                continue
            if not self._fact_matches_retire_target(fact, item, retire_key=retire_key):
                continue
            if self.memory_store.supersede_fact(
                fact_id=fact.id,
                superseded_by=f"retire:{uuid4()}",
                valid_to=datetime.now(timezone.utc),
            ):
                retired.append(fact.id)
        return retired

    @staticmethod
    def _fact_matches_retire_target(
        fact: ValidatedFact,
        item: GenericExtractedFact,
        *,
        retire_key: str | None,
    ) -> bool:
        if item.object_name:
            return False

        target_value = normalize_fact_value_for_match(item.value)
        fact_value = normalize_fact_value_for_match(fact.value)
        if target_value and fact_value == target_value:
            return True

        item_key = normalize_attribute_key(retire_key or item.attributes.get("key"))
        fact_key = resolve_attribute_key(fact.value, fact.attributes)

        if item_key and fact_key and item_key != fact_key:
            return False

        if item_key and fact_key == item_key:
            target_key, target_tail = parse_keyed_value(item.value)
            _, fact_tail = parse_keyed_value(fact.value)

            if target_key is None and not target_value:
                return True
            if target_tail is None:
                return True

            target_tail_norm = normalize_fact_value_for_match(target_tail)
            fact_tail_norm = normalize_fact_value_for_match(fact_tail)
            if target_tail_norm and fact_tail_norm:
                return target_tail_norm == fact_tail_norm

            # Fact can carry key in attributes while storing plain value.
            if target_tail_norm and fact_value:
                return target_tail_norm == fact_value

        return False

    def _fact_matches_scope(self, fact: ValidatedFact, *, scope: dict[str, str]) -> bool:
        interaction = self.memory_store.get_interaction(fact.source_interaction_id)
        if interaction is None:
            return False

        metadata = interaction.metadata or {}
        for key, expected in scope.items():
            if expected is None:
                continue
            if key == "run_id":
                actual = (
                    getattr(interaction, "run_id", None)
                    or getattr(interaction, "session_id", None)
                    or metadata.get("run_id")
                    or metadata.get("session_id")
                )
            else:
                actual = getattr(interaction, key, None) or metadata.get(key)
            if actual != expected:
                return False

        return True

    @staticmethod
    def _infer_attribute_key(value: str | None, attributes: dict[str, Any]) -> str | None:
        return resolve_attribute_key(value, attributes)

    def _disposition_from_grounding(self, grounding: GroundingResult) -> dict[str, Any]:
        if grounding.decision == GroundingDecision.SUPERSEDED:
            disposition = MemoryDisposition.REFINE
        elif grounding.decision == GroundingDecision.APPROVED:
            disposition = MemoryDisposition.CAPTURE
        elif grounding.decision == GroundingDecision.DUPLICATE:
            disposition = MemoryDisposition.PASS
        else:
            disposition = MemoryDisposition.PASS

        fact_ids = [grounding.validated_fact.id] if grounding.validated_fact is not None else []
        superseded_fact_ids = [fact.id for fact in grounding.superseded_facts]
        return self._disposition_event(
            disposition,
            reason=f"grounding_{grounding.decision.value}",
            fact_ids=fact_ids,
            superseded_fact_ids=superseded_fact_ids,
        )

    @staticmethod
    def _disposition_event(
        disposition: MemoryDisposition,
        *,
        reason: str,
        fact_ids: list[str] | None = None,
        superseded_fact_ids: list[str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "disposition": disposition.value,
            "reason": reason,
        }
        if fact_ids:
            event["fact_ids"] = fact_ids
        if superseded_fact_ids:
            event["superseded_fact_ids"] = superseded_fact_ids
        if payload is not None:
            event["payload"] = payload
        return event

    @staticmethod
    def _coerce_disposition(value: str | None) -> MemoryDisposition:
        if isinstance(value, MemoryDisposition):
            return value
        if not value:
            return MemoryDisposition.CAPTURE
        try:
            return MemoryDisposition(str(value).strip().lower())
        except ValueError:
            return MemoryDisposition.CAPTURE

    def _get_or_create_entity(
        self,
        name: str,
        entity_type: EntityType,
        attributes: dict[str, Any] | None = None,
    ) -> Entity:
        resolved_attributes = dict(attributes or {})
        resolved_uniqueness_key = build_entity_uniqueness_key(
            name=name,
            entity_type=entity_type,
            attributes=resolved_attributes,
        )
        resolved_entity_id = stable_entity_id(resolved_uniqueness_key)

        entity, _ = self.memory_store.find_or_create_entity(
            name=name,
            entity_type=entity_type,
            uniqueness_key=resolved_uniqueness_key,
            create_fn=lambda: Entity(
                id=resolved_entity_id,
                entity_type=entity_type,
                name=name,
                attributes=resolved_attributes,
            ),
        )
        return entity

    @staticmethod
    def _coerce_actor(source: str) -> ActorType:
        value = source.strip().lower()
        if value in {"assistant", "agent"}:
            return ActorType.AGENT
        if value == "tool":
            return ActorType.TOOL
        if value == "system":
            return ActorType.SYSTEM
        return ActorType.USER

    @staticmethod
    def _coerce_relation(relation: Any) -> RelationType:
        if isinstance(relation, RelationType):
            return relation
        try:
            return RelationType(str(relation))
        except ValueError as exc:
            allowed = ", ".join(sorted(r.value for r in RelationType))
            raise ValueError(f"Unknown relation '{relation}'. Allowed: {allowed}") from exc

    @staticmethod
    def _coerce_relation_dynamic(relation: Any, has_object: bool) -> RelationType:
        if isinstance(relation, RelationType):
            return relation
        try:
            return RelationType(str(relation))
        except ValueError:
            return RelationType.RELATED_TO if has_object else RelationType.HAS_ATTRIBUTE

    @staticmethod
    def _coerce_entity_type(value: Any, *, default: EntityType) -> EntityType:
        if isinstance(value, EntityType):
            return value
        if not value:
            return default
        try:
            return EntityType(str(value))
        except ValueError:
            return default
