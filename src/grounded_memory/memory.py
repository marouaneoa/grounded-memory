"""High-level SDK facade for Grounded Memory."""

from __future__ import annotations

import os
import re
from contextlib import suppress
from dataclasses import dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    import yaml
except ImportError:  # pragma: no cover - optional runtime dependency safety
    yaml = None

from grounded_memory.adapters.discovery import (
    ConstraintSeedDiscoverer,
    DiscoveredConstraintSeed,
)
from grounded_memory.adapters.registry import list_registered_adapters, list_supported_profiles
from grounded_memory.adapters.seeds import (
    CardinalitySeedConstraintEvaluator,
    SeedConstraintEvaluator,
    TemporalCardinalitySeedConstraintEvaluator,
)
from grounded_memory.core.constraints import (
    ConstraintLifecycleStatus,
    DynamicConstraintScope,
)
from grounded_memory.core.entity_identity import (
    build_entity_uniqueness_key,
    stable_entity_id,
)
from grounded_memory.core.intent import (
    BaseIntentRouter,
    UserIntent,
)
from grounded_memory.core.models import (
    ActorType,
    AnswerContext,
    CandidateFact,
    Entity,
    EntityType,
    Interaction,
    RelationType,
    ValidatedFact,
)
from grounded_memory.core.system import Neo4jConfig, StorageBackend
from grounded_memory.core.tuple_normalization import (
    build_fact_semantic_key,
    fact_values_equal,
    normalize_attribute_key,
    normalize_fact_attributes,
    parse_keyed_value,
    sanitize_fact_value,
    should_materialize_attribute_object,
)
from grounded_memory.llm.client import LLMConfig
from grounded_memory.retrieval import GraphRetriever, RelationshipPreset, RetrievalStrategy
from grounded_memory.system import GroundedMemorySystem


@dataclass
class SearchResult:
    """Serializable search result item returned by Memory.search()."""

    fact_id: str
    relation: str
    subject_id: str
    subject_name: str
    object_id: str | None
    object_name: str | None
    value: str | None
    confidence: float
    valid_from: str
    valid_to: str | None
    score: float | None = None
    signals: dict[str, Any] = field(default_factory=dict)


class OptimizationProfile(str, Enum):
    """Named optimization presets for SDK defaults."""

    LATENCY = "latency"
    BALANCED = "balanced"
    RECALL = "recall"


@dataclass
class OptimizationSettings:
    """Resolved retrieval defaults for a Memory instance."""

    max_seeds: int
    max_hops: int
    max_facts: int
    strategy: RetrievalStrategy


@dataclass(frozen=True)
class ScopeContext:
    """Normalized scope envelope used for write/read filtering."""

    tenant_id: str | None = None
    app_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    space_type: str | None = None

    @property
    def scope_id(self) -> str | None:
        if self.tenant_id and self.app_id and self.user_id:
            return f"{self.tenant_id}:{self.app_id}:{self.user_id}"
        return None

    def as_dict(self) -> dict[str, str]:
        payload: dict[str, str] = {}
        if self.tenant_id is not None:
            payload["tenant_id"] = self.tenant_id
        if self.app_id is not None:
            payload["app_id"] = self.app_id
        if self.user_id is not None:
            payload["user_id"] = self.user_id
        if self.agent_id is not None:
            payload["agent_id"] = self.agent_id
        if self.run_id is not None:
            payload["run_id"] = self.run_id
        if self.space_type is not None:
            payload["space_type"] = self.space_type
        if self.scope_id is not None:
            payload["scope_id"] = self.scope_id
        return payload


DEFAULT_OPTIMIZATION_SETTINGS: dict[OptimizationProfile, OptimizationSettings] = {
    OptimizationProfile.LATENCY: OptimizationSettings(
        max_seeds=3,
        max_hops=1,
        max_facts=12,
        strategy=RetrievalStrategy.WEIGHTED,
    ),
    OptimizationProfile.BALANCED: OptimizationSettings(
        max_seeds=6,
        max_hops=2,
        max_facts=30,
        strategy=RetrievalStrategy.SAFETY_PRIORITY,
    ),
    OptimizationProfile.RECALL: OptimizationSettings(
        max_seeds=10,
        max_hops=3,
        max_facts=60,
        strategy=RetrievalStrategy.BREADTH_FIRST,
    ),
}


class Memory:
    """
    Simple top-level memory API.

    Example:
        from grounded_memory import Memory

        memory = Memory()
        memory.add("Service Atlas owner is platform-team")
        results = memory.search("What allergies does Alice have?")

    The ``adapter`` parameter is the canonical way to select a behavior profile
    (``generic``, ``healthcare``, …). ``domain_profile`` is a deprecated alias
    kept for backward compatibility; new code should use ``adapter``.
    """

    def __init__(
        self,
        adapter: str | None = None,
        domain_profile: str = "generic",
        *,
        storage_backend: StorageBackend | str | None = None,
        neo4j_config: Neo4jConfig | None = None,
        use_llm: bool = True,
        llm_config: LLMConfig | None = None,
        configure_validator: Any | None = None,
        agent_factory: Any | None = None,
        agent: Any | None = None,
        intent_router: BaseIntentRouter | None = None,
        optimization_profile: OptimizationProfile | str = OptimizationProfile.BALANCED,
        default_max_seeds: int | None = None,
        default_max_hops: int | None = None,
        default_max_facts: int | None = None,
        default_strategy: RetrievalStrategy | str | None = None,
        relationship_preset: RelationshipPreset | str | None = None,
        require_scope: bool | None = None,
        default_tenant_id: str | None = None,
        default_app_id: str | None = None,
        default_user_id: str | None = None,
        default_agent_id: str | None = None,
        default_space_type: str | None = None,
    ):
        resolved_adapter = (adapter or domain_profile).strip().lower()
        self.system = GroundedMemorySystem(
            neo4j_config=neo4j_config,
            storage_backend=storage_backend,
            adapter=resolved_adapter,
            domain_profile=domain_profile,
            configure_validator=configure_validator,
            agent_factory=agent_factory,
        )
        self.adapter = self.system.adapter
        self.domain_profile = self.system.domain_profile  # Backward-compatible alias
        self.llm_config = llm_config
        self.agent = agent

        if not use_llm and self.agent is None:
            raise RuntimeError(
                "Memory requires LLM mode. "
                "Set use_llm=True and configure LLM_MODEL/LLM_BASE_URL/LLM_API_KEY, "
                "or provide an agent configured with an LLM backend."
            )

        if self.agent is None:
            resolved_llm_config = self.llm_config or LLMConfig.from_env()
            self.llm_config = resolved_llm_config
            self.agent = self.system.create_agent(
                use_llm=True,
                llm_config=resolved_llm_config,
            )

        spec = self.system.adapter_spec

        if relationship_preset is None:
            if spec is not None and spec.relationship_preset is not None:
                relationship_preset = RelationshipPreset(spec.relationship_preset)
            else:
                relationship_preset = RelationshipPreset.GENERIC

        profile = self._coerce_optimization_profile(optimization_profile)
        profile_settings = DEFAULT_OPTIMIZATION_SETTINGS[profile]
        self.optimization_profile = profile
        self.optimization_settings = OptimizationSettings(
            max_seeds=(
                default_max_seeds if default_max_seeds is not None else profile_settings.max_seeds
            ),
            max_hops=(
                default_max_hops if default_max_hops is not None else profile_settings.max_hops
            ),
            max_facts=(
                default_max_facts if default_max_facts is not None else profile_settings.max_facts
            ),
            strategy=(
                self._coerce_retrieval_strategy(default_strategy)
                if default_strategy is not None
                else profile_settings.strategy
            ),
        )

        self.relationship_preset = RelationshipPreset(relationship_preset)
        hint_registry = None
        if spec is not None and spec.query_hints:
            from grounded_memory.retrieval.graph import QueryHintRegistry

            hint_registry = QueryHintRegistry()
            for category, terms in spec.query_hints.items():
                hint_registry.register(category, terms)
        self.retriever = GraphRetriever(
            self.system.memory_store,
            relationship_preset=self.relationship_preset,
            hint_registry=hint_registry,
        )
        self._apply_adapter_retrieval_weights()

        if require_scope is None:
            require_scope = os.getenv("GM_REQUIRE_SCOPE", "0") == "1"
        self.require_scope = bool(require_scope)

        self.default_scope = ScopeContext(
            tenant_id=default_tenant_id or os.getenv("GM_SCOPE_TENANT_ID"),
            app_id=default_app_id or os.getenv("GM_SCOPE_APP_ID"),
            user_id=default_user_id or os.getenv("GM_SCOPE_USER_ID"),
            agent_id=default_agent_id or os.getenv("GM_SCOPE_AGENT_ID"),
            run_id=None,
            space_type=default_space_type or os.getenv("GM_SCOPE_SPACE_TYPE"),
        )

        self.intent_router = intent_router or self._build_default_intent_router()

    def _build_default_intent_router(self) -> BaseIntentRouter:
        """Build a keyword router with optional adapter-specific patterns."""
        from grounded_memory.core.intent import KeywordIntentRouter

        router = KeywordIntentRouter()
        spec = self.system.adapter_spec
        if spec is not None and spec.intent_router_patterns:
            for category, patterns in spec.intent_router_patterns.items():
                with suppress(ValueError):
                    router.register_patterns(category, list(patterns))
        return router

    def route(self, query: str) -> UserIntent:
        """Classify user intent and return a structured routing decision."""
        return self.intent_router.route(query)

    def process(
        self,
        text: str,
        *,
        source: str = "user",
        tenant_id: str | None = None,
        app_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        space_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Auto-route user input based on inferred intent.

        - REMEMBER -> add(text) (write path)
        - RECALL / FIND_RELATED / EXPLAIN -> search(text) (read path)
        - UNKNOWN -> attempt read, then attempt write if empty
        """
        intent = self.route(text)

        if intent.is_write():
            return {
                "intent": intent.model_dump(),
                "results": self.add(
                    text,
                    source=source,
                    tenant_id=tenant_id,
                    app_id=app_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    run_id=run_id,
                    session_id=session_id,
                    space_type=space_type,
                    metadata=metadata,
                    **kwargs,
                ),
            }

        if intent.is_read():
            return {
                "intent": intent.model_dump(),
                "results": self.search(
                    text,
                    tenant_id=tenant_id,
                    app_id=app_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    run_id=run_id,
                    space_type=space_type,
                    **kwargs,
                ),
            }

        # UNKNOWN: try read first, then write if no results
        read_results = self.search(
            text,
            tenant_id=tenant_id,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            space_type=space_type,
            **kwargs,
        )
        if read_results:
            return {"intent": intent.model_dump(), "results": read_results}

        return {
            "intent": intent.model_dump(),
            "results": self.add(
                text,
                source=source,
                tenant_id=tenant_id,
                app_id=app_id,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                session_id=session_id,
                space_type=space_type,
                metadata=metadata,
                **kwargs,
            ),
        }

    def add(
        self,
        text: str | list[dict[str, Any]],
        *,
        source: str = "user",
        tenant_id: str | None = None,
        app_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        space_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Ingest interaction text (or compatibility message list) via the LLM-backed agent."""
        scope = self._resolve_scope(
            tenant_id=tenant_id,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id or session_id,
            space_type=space_type,
            require_scope=self.require_scope,
        )

        merged_metadata = self._merge_scope_metadata(metadata=metadata, scope=scope)

        if isinstance(text, list):
            return self.add_many(
                text,
                source=source,
                tenant_id=scope.tenant_id,
                app_id=scope.app_id,
                user_id=scope.user_id,
                agent_id=scope.agent_id,
                run_id=scope.run_id,
                session_id=scope.run_id,
                space_type=scope.space_type,
                metadata=merged_metadata,
                continue_on_error=bool(kwargs.pop("continue_on_error", False)),
            )

        if self.agent is None:
            raise RuntimeError(
                "No agent available for add()/remember(). "
                "Memory is LLM-required; configure an LLM-backed agent or domain adapter."
            )
        kwargs["tenant_id"] = scope.tenant_id
        kwargs["app_id"] = scope.app_id
        kwargs["user_id"] = scope.user_id
        kwargs["agent_id"] = scope.agent_id
        kwargs["run_id"] = scope.run_id
        kwargs["session_id"] = scope.run_id
        kwargs["space_type"] = scope.space_type
        kwargs["metadata"] = self._merge_scope_metadata(
            metadata=kwargs.pop("metadata", None) or merged_metadata,
            scope=scope,
        )
        return self.agent.process(text, source=source, **kwargs)

    def remember(
        self,
        text: str | list[dict[str, Any]],
        *,
        source: str = "user",
        tenant_id: str | None = None,
        app_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        space_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Alias for add(), mirroring compatibility terminology."""
        return self.add(
            text,
            source=source,
            tenant_id=tenant_id,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            session_id=session_id,
            space_type=space_type,
            metadata=metadata,
            **kwargs,
        )

    def add_many(
        self,
        messages: list[str | dict[str, Any]],
        *,
        source: str = "user",
        tenant_id: str | None = None,
        app_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        space_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        continue_on_error: bool = False,
    ) -> dict[str, Any]:
        """Batch-ingest messages and return per-item results with summary stats."""
        ingested = 0
        failed = 0
        results: list[dict[str, Any]] = []

        for index, item in enumerate(messages):
            try:
                if isinstance(item, str):
                    out = self.add(
                        item,
                        source=source,
                        tenant_id=tenant_id,
                        app_id=app_id,
                        user_id=user_id,
                        agent_id=agent_id,
                        run_id=run_id,
                        session_id=session_id,
                        space_type=space_type,
                        metadata=metadata,
                    )
                elif isinstance(item, dict):
                    text = str(item.get("text") or item.get("content") or "").strip()
                    if not text:
                        raise ValueError("message payload must include 'text' or 'content'")
                    item_source = str(item.get("source") or item.get("role") or source)
                    item_tenant_id = item.get("tenant_id", tenant_id)
                    item_app_id = item.get("app_id", app_id)
                    item_user_id = item.get("user_id", user_id)
                    item_agent_id = item.get("agent_id", agent_id)
                    item_run_id = item.get("run_id", run_id or session_id)
                    item_session_id = item.get("session_id", item_run_id)
                    item_space_type = item.get("space_type", space_type)
                    item_metadata = item.get("metadata", metadata)
                    call_kwargs = {
                        key: value
                        for key, value in item.items()
                        if key
                        not in {
                            "text",
                            "content",
                            "source",
                            "role",
                            "tenant_id",
                            "app_id",
                            "user_id",
                            "agent_id",
                            "run_id",
                            "session_id",
                            "space_type",
                            "metadata",
                        }
                    }
                    out = self.add(
                        text,
                        source=item_source,
                        tenant_id=item_tenant_id,
                        app_id=item_app_id,
                        user_id=item_user_id,
                        agent_id=item_agent_id,
                        run_id=item_run_id,
                        session_id=item_session_id,
                        space_type=item_space_type,
                        metadata=item_metadata,
                        **call_kwargs,
                    )
                else:
                    raise ValueError("each message must be a string or dict payload")

                ingested += 1
                results.append({"index": index, "ok": True, "result": self._to_serializable(out)})
            except Exception as exc:
                failed += 1
                results.append({"index": index, "ok": False, "error": str(exc)})
                if not continue_on_error:
                    break

        return {
            "ingested": ingested,
            "failed": failed,
            "total": len(messages),
            "results": results,
        }

    @classmethod
    def _to_serializable(cls, value: Any) -> Any:
        """Convert nested runtime objects into JSON-serializable structures."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, Enum):
            return value.value

        if isinstance(value, dict):
            return {str(key): cls._to_serializable(item) for key, item in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [cls._to_serializable(item) for item in value]

        if hasattr(value, "model_dump") and callable(value.model_dump):
            try:
                return cls._to_serializable(value.model_dump(mode="json"))
            except TypeError:
                return cls._to_serializable(value.model_dump())

        if is_dataclass(value):
            return {
                field_name: cls._to_serializable(getattr(value, field_name))
                for field_name in value.__dataclass_fields__
            }

        return str(value)

    def add_entity(
        self,
        name: str,
        *,
        entity_type: EntityType | str = EntityType.FACILITY,
        attributes: dict[str, Any] | None = None,
        canonical_id: str | None = None,
        uniqueness_key: str | None = None,
        entity_id: str | None = None,
    ) -> dict[str, Any]:
        """Create or reuse an entity and return its serialized representation."""
        resolved_type = self._coerce_entity_type(entity_type)
        resolved_attributes = dict(attributes or {})
        resolved_uniqueness_key = build_entity_uniqueness_key(
            name=name,
            entity_type=resolved_type,
            attributes=resolved_attributes,
            canonical_id=canonical_id,
            uniqueness_key=uniqueness_key,
        )
        resolved_entity_id = entity_id or stable_entity_id(resolved_uniqueness_key)

        entity, created = self.system.memory_store.find_or_create_entity(
            name=name,
            entity_type=resolved_type,
            uniqueness_key=resolved_uniqueness_key,
            create_fn=lambda: Entity(
                id=resolved_entity_id,
                entity_type=resolved_type,
                name=name,
                canonical_id=canonical_id,
                attributes=resolved_attributes,
            ),
        )
        return {
            "created": created,
            "entity": entity.model_dump(),
        }

    def add_fact(
        self,
        *,
        subject_id: str,
        relation: RelationType | str,
        object_id: str | None = None,
        value: str | None = None,
        confidence: float = 0.9,
        attributes: dict[str, Any] | None = None,
        source: str = "system",
        tenant_id: str | None = None,
        app_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        space_type: str | None = None,
        source_interaction_id: str | None = None,
    ) -> dict[str, Any]:
        """Write a structured fact through the grounding pipeline."""
        normalized_value = sanitize_fact_value(value)
        if object_id is None and normalized_value is None:
            raise ValueError("add_fact requires object_id or value")

        relation_type = self._coerce_relation_type(relation)

        scope = self._resolve_scope(
            tenant_id=tenant_id,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id or session_id,
            space_type=space_type,
            require_scope=self.require_scope,
        )
        normalized_attributes = normalize_fact_attributes(normalized_value, attributes)
        normalized_attributes.update(scope.as_dict())

        resolved_object_id = object_id
        if relation_type == RelationType.HAS_ATTRIBUTE:
            parsed_key, parsed_tail = parse_keyed_value(normalized_value)
            resolved_key = normalize_attribute_key(normalized_attributes.get("key")) or parsed_key
            if resolved_key is not None:
                normalized_attributes["key"] = resolved_key
            if parsed_tail is not None:
                normalized_value = parsed_tail
            if resolved_object_id is None and should_materialize_attribute_object(normalized_value):
                resolved_object_id = self._materialize_attribute_value_entity(
                    normalized_value,
                    scope=scope,
                )

        interaction_id = source_interaction_id or self._ensure_interaction_for_scope(
            raw_text=f"[sdk:add_fact] {subject_id} {relation} {object_id or value}",
            actor=self._coerce_actor(source),
            scope=scope,
        )

        candidate = CandidateFact(
            source_interaction_id=interaction_id,
            subject_entity_id=subject_id,
            relation=relation_type,
            object_entity_id=resolved_object_id,
            value=normalized_value,
            confidence=confidence,
            attributes=normalized_attributes,
        )

        result = self.system.grounding_operator.ground(candidate)
        return {
            "decision": result.decision.value,
            "fact": result.validated_fact.model_dump() if result.validated_fact else None,
            "rejection": result.rejection_record.model_dump() if result.rejection_record else None,
            "superseded_facts": [fact.model_dump() for fact in result.superseded_facts],
        }

    def build_context(
        self,
        query: str,
        *,
        tenant_id: str | None = None,
        app_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        space_type: str | None = None,
        at_time: datetime | None = None,
        lookback_days: int | None = None,
        max_seeds: int | None = None,
        max_hops: int | None = None,
        max_facts: int | None = None,
        strategy: RetrievalStrategy | str | None = None,
    ) -> AnswerContext:
        """Retrieve structured AnswerContext for a query."""
        scope = self._resolve_scope(
            tenant_id=tenant_id,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            space_type=space_type,
            require_scope=self.require_scope,
        )

        resolved_max_seeds = (
            max_seeds if max_seeds is not None else self.optimization_settings.max_seeds
        )
        resolved_max_hops = (
            max_hops if max_hops is not None else self.optimization_settings.max_hops
        )
        resolved_max_facts = (
            max_facts if max_facts is not None else self.optimization_settings.max_facts
        )
        resolved_strategy = (
            self._coerce_retrieval_strategy(strategy)
            if strategy is not None
            else self.optimization_settings.strategy
        )

        seed_entities = self.retriever.select_seed_entities(query, max_seeds=resolved_max_seeds)
        if not seed_entities:
            # Fallback: score entities by active-fact density for relevance
            all_entities = self.system.memory_store.get_all_entities()
            scored: list[tuple[int, str]] = []
            for entity in all_entities:
                fact_count = len(self.system.memory_store.get_active_facts_for_entity(entity.id))
                if fact_count > 0:
                    scored.append((fact_count, entity.id))
            scored.sort(key=lambda item: item[0], reverse=True)
            seed_entities = [entity_id for _, entity_id in scored[:resolved_max_seeds]]

        if not seed_entities:
            return AnswerContext(query=query)

        return self.retriever.retrieve(
            query=query,
            seed_entities=seed_entities,
            max_hops=resolved_max_hops,
            max_facts=resolved_max_facts,
            strategy=resolved_strategy,
            at_time=at_time,
            lookback_days=lookback_days,
            user_id=scope.user_id,
            scope=scope.as_dict(),
        )

    def search(
        self,
        query: str,
        *,
        tenant_id: str | None = None,
        app_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        space_type: str | None = None,
        at_time: datetime | None = None,
        lookback_days: int | None = None,
        limit: int = 10,
        threshold: float | None = None,
        rerank_debug: bool = False,
        max_hops: int | None = None,
        max_seeds: int | None = None,
        strategy: RetrievalStrategy | str | None = None,
    ) -> list[dict[str, Any]]:
        """Search memory and return serializable fact records."""
        scope = self._resolve_scope(
            tenant_id=tenant_id,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            space_type=space_type,
            require_scope=self.require_scope,
        )

        context = self.build_context(
            query,
            tenant_id=scope.tenant_id,
            app_id=scope.app_id,
            user_id=scope.user_id,
            agent_id=scope.agent_id,
            run_id=scope.run_id,
            space_type=scope.space_type,
            at_time=at_time,
            lookback_days=lookback_days,
            max_seeds=max_seeds,
            max_hops=max_hops,
            max_facts=limit,
            strategy=strategy,
        )
        items = [
            self._fact_to_result(
                fact,
                context.entities,
                score_payload=(context.retrieval_metadata.get("scores") or {}).get(fact.id),
            ).__dict__
            for fact in context.facts[:limit]
        ]
        if scope.as_dict():
            items = self._filter_results_by_scope(items, scope=scope)
        items = self._apply_result_threshold(items, threshold=threshold)
        if not rerank_debug:
            items = [self._strip_debug_signals(item) for item in items]
        if items:
            return items[:limit]

        lexical_items = self._lexical_fact_fallback(query, limit=limit, scope=scope)
        lexical_items = self._apply_result_threshold(lexical_items, threshold=threshold)
        if not rerank_debug:
            lexical_items = [self._strip_debug_signals(item) for item in lexical_items]
        return lexical_items

    def build_memory_prompt(
        self,
        query: str,
        *,
        tenant_id: str | None = None,
        app_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        space_type: str | None = None,
        at_time: datetime | None = None,
        lookback_days: int | None = None,
        limit: int = 5,
        threshold: float | None = None,
        max_hops: int | None = None,
        max_seeds: int | None = None,
        strategy: RetrievalStrategy | str | None = None,
        empty_text: str = "- (no prior memories)",
    ) -> str:
        """Retrieve memories for a query and render them as prompt-ready bullet lines."""
        items = self.retrieve(
            query,
            tenant_id=tenant_id,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            space_type=space_type,
            at_time=at_time,
            lookback_days=lookback_days,
            limit=limit,
            threshold=threshold,
            max_hops=max_hops,
            max_seeds=max_seeds,
            strategy=strategy,
        )
        return self.render_memories(items, empty_text=empty_text)

    def render_memories(
        self,
        items: list[dict[str, Any]],
        *,
        empty_text: str = "- (no prior memories)",
    ) -> str:
        """Render retrieved memory rows into a compact prompt block."""
        if not items:
            return empty_text
        return "\n".join(self._memory_line_from_result(item) for item in items)

    def query(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Alias for search()."""
        return self.search(query, **kwargs)

    def retrieve(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Alias for search(), mirroring compatibility terminology."""
        return self.search(query, **kwargs)

    def configure_optimization(
        self,
        *,
        profile: OptimizationProfile | str | None = None,
        max_seeds: int | None = None,
        max_hops: int | None = None,
        max_facts: int | None = None,
        strategy: RetrievalStrategy | str | None = None,
    ) -> dict[str, Any]:
        """Update runtime retrieval defaults for this Memory instance."""
        if profile is not None:
            resolved_profile = self._coerce_optimization_profile(profile)
            defaults = DEFAULT_OPTIMIZATION_SETTINGS[resolved_profile]
            self.optimization_profile = resolved_profile
            self.optimization_settings = OptimizationSettings(
                max_seeds=defaults.max_seeds,
                max_hops=defaults.max_hops,
                max_facts=defaults.max_facts,
                strategy=defaults.strategy,
            )

        if max_seeds is not None:
            self.optimization_settings.max_seeds = max_seeds
        if max_hops is not None:
            self.optimization_settings.max_hops = max_hops
        if max_facts is not None:
            self.optimization_settings.max_facts = max_facts
        if strategy is not None:
            self.optimization_settings.strategy = self._coerce_retrieval_strategy(strategy)

        return {
            "optimization_profile": self.optimization_profile.value,
            "max_seeds": self.optimization_settings.max_seeds,
            "max_hops": self.optimization_settings.max_hops,
            "max_facts": self.optimization_settings.max_facts,
            "strategy": self.optimization_settings.strategy.value,
        }

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """Get a memory object by ID across entities/facts/interactions/rejections."""
        entity = self.system.memory_store.get_entity(memory_id)
        if entity is not None:
            return {"kind": "entity", "data": entity.model_dump()}

        fact = self.system.memory_store.get_fact(memory_id)
        if fact is not None:
            return {"kind": "fact", "data": fact.model_dump()}

        interaction = self.system.memory_store.get_interaction(memory_id)
        if interaction is not None:
            return {"kind": "interaction", "data": interaction.model_dump()}

        rejection = self.system.memory_store.get_rejection(memory_id)
        if rejection is not None:
            return {"kind": "rejection", "data": rejection.model_dump()}

        return None

    def update_fact(
        self,
        fact_id: str,
        *,
        relation: RelationType | str | None = None,
        object_id: str | None = None,
        value: str | None = None,
        confidence: float | None = None,
        attributes: dict[str, Any] | None = None,
        source: str = "system",
        tenant_id: str | None = None,
        app_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        space_type: str | None = None,
    ) -> dict[str, Any]:
        """Update a fact by writing a new candidate and allowing supersession."""
        current = self.system.memory_store.get_fact(fact_id)
        if current is None:
            raise ValueError(f"Fact not found: {fact_id}")

        next_relation = self._coerce_relation_type(relation) if relation else current.relation
        next_object_id = object_id if object_id is not None else current.object_id
        next_value = value if value is not None else current.value
        normalized_value = sanitize_fact_value(next_value)

        next_attributes = dict(current.attributes)
        if attributes:
            next_attributes.update(attributes)

        scope = self._resolve_scope(
            tenant_id=tenant_id,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id or session_id,
            space_type=space_type,
            require_scope=self.require_scope,
        )
        next_attributes.update(scope.as_dict())
        next_attributes = normalize_fact_attributes(normalized_value, next_attributes)

        if next_relation == RelationType.HAS_ATTRIBUTE:
            parsed_key, parsed_tail = parse_keyed_value(normalized_value)
            resolved_key = normalize_attribute_key(next_attributes.get("key")) or parsed_key
            if resolved_key is not None:
                next_attributes["key"] = resolved_key
            if parsed_tail is not None:
                normalized_value = parsed_tail
            if next_object_id is None and should_materialize_attribute_object(normalized_value):
                next_object_id = self._materialize_attribute_value_entity(
                    normalized_value,
                    scope=scope,
                )

        if next_object_id is None and normalized_value is None:
            raise ValueError("update_fact requires object_id or value")

        interaction_id = self._ensure_interaction_for_scope(
            raw_text=f"[sdk:update_fact] {fact_id}",
            actor=self._coerce_actor(source),
            scope=scope,
        )

        candidate = CandidateFact(
            source_interaction_id=interaction_id,
            subject_entity_id=current.subject_id,
            relation=next_relation,
            object_entity_id=next_object_id,
            value=normalized_value,
            confidence=confidence if confidence is not None else current.confidence,
            attributes=next_attributes,
        )

        result = self.system.grounding_operator.ground(candidate)
        return {
            "decision": result.decision.value,
            "fact": result.validated_fact.model_dump() if result.validated_fact else None,
            "rejection": result.rejection_record.model_dump() if result.rejection_record else None,
            "superseded_facts": [fact.model_dump() for fact in result.superseded_facts],
        }

    def delete_fact(
        self,
        fact_id: str,
        *,
        reason: str = "deleted via sdk",
    ) -> dict[str, Any]:
        """Soft-delete a fact by closing its temporal validity window."""
        fact = self.system.memory_store.get_fact(fact_id)
        if fact is None:
            raise ValueError(f"Fact not found: {fact_id}")

        if not fact.is_active:
            return {
                "deleted": False,
                "reason": "fact_already_inactive",
                "fact": fact.model_dump(),
            }

        superseded_marker = f"sdk-delete:{uuid4()}:{reason}"
        self.system.memory_store.supersede_fact(
            fact_id=fact_id,
            superseded_by=superseded_marker,
            valid_to=datetime.now(timezone.utc),
        )
        updated = self.system.memory_store.get_fact(fact_id)

        return {
            "deleted": True,
            "fact": updated.model_dump() if updated else None,
        }

    def history(
        self,
        *,
        fact_id: str | None = None,
        entity_id: str | None = None,
        memory_id: str | None = None,
        tenant_id: str | None = None,
        app_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        space_type: str | None = None,
        relation: RelationType | str | None = None,
        include_inactive: bool = True,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return temporal fact history with optional fact/entity/relation filters."""
        target_fact_id = fact_id or memory_id

        if target_fact_id and entity_id:
            raise ValueError("history accepts either fact_id/memory_id or entity_id, not both")

        if target_fact_id:
            root_fact = self.system.memory_store.get_fact(target_fact_id)
            if root_fact is None:
                raise ValueError(f"Fact not found: {target_fact_id}")
            facts = self._history_for_fact_lineage(root_fact)
        elif entity_id:
            facts = self.system.memory_store.get_facts_for_entity(
                entity_id,
                include_superseded=True,
            )
        else:
            facts = self.system.memory_store.get_all_validated_facts()

        if relation is not None:
            relation_type = self._coerce_relation_type(relation)
            facts = [fact for fact in facts if fact.relation == relation_type]

        scope = self._resolve_scope(
            tenant_id=tenant_id,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            space_type=space_type,
            require_scope=False,
        )
        if scope.as_dict():
            facts = [fact for fact in facts if self._fact_matches_scope(fact, scope=scope)]

        if not include_inactive:
            facts = [fact for fact in facts if fact.is_active]

        facts.sort(key=lambda fact: fact.valid_from, reverse=True)
        return [fact.model_dump() for fact in facts[:limit]]

    def _history_for_fact_lineage(self, root_fact: ValidatedFact) -> list[ValidatedFact]:
        """Collect the temporal lineage for a fact across refinements/retirements."""
        all_facts = self.system.memory_store.get_all_validated_facts()
        root_semantic_key = self._fact_semantic_key(root_fact)

        # TODO: Lineage robustness
        # - Current lineage relies on exact semantic_key or exact value/object matches.
        # - Consider fuzzy matching, canonicalization, and timestamp-aware ordering
        #   to better capture refinements that slightly change wording (e.g. "Python" vs "python").
        # - Also consider exposing a configurable similarity threshold and clearer
        #   provenance linking (e.g., explicit 'refines' relation when produced by the agent).

        lineage: list[ValidatedFact] = []
        for fact in all_facts:
            if fact.subject_id != root_fact.subject_id:
                continue
            if fact.relation != root_fact.relation:
                continue

            if fact.id == root_fact.id:
                lineage.append(fact)
                continue

            if root_semantic_key and self._fact_semantic_key(fact) == root_semantic_key:
                lineage.append(fact)
                continue

            if self._fact_exact_match(fact, root_fact):
                lineage.append(fact)

        # Deduplicate while preserving objects.
        unique: dict[str, ValidatedFact] = {fact.id: fact for fact in lineage}
        return list(unique.values())

    def list_entities(
        self,
        *,
        entity_type: EntityType | str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List entities, optionally filtered by entity type."""
        entities = self.system.memory_store.get_all_entities()
        if entity_type is not None:
            resolved = self._coerce_entity_type(entity_type)
            entities = [entity for entity in entities if entity.entity_type == resolved]

        return [entity.model_dump() for entity in entities[:limit]]

    def list_facts(
        self,
        *,
        tenant_id: str | None = None,
        app_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        space_type: str | None = None,
        active_only: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List facts, optionally restricted to active facts."""
        scope = self._resolve_scope(
            tenant_id=tenant_id,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            space_type=space_type,
            require_scope=self.require_scope,
        )
        facts = self.system.memory_store.get_all_validated_facts()
        if scope.as_dict():
            facts = [fact for fact in facts if self._fact_matches_scope(fact, scope=scope)]
        if active_only:
            facts = [fact for fact in facts if fact.is_active]
        facts.sort(key=lambda fact: fact.valid_from, reverse=True)
        return [fact.model_dump() for fact in facts[:limit]]

    def list_interactions(
        self,
        *,
        tenant_id: str | None = None,
        app_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        space_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List recent interactions."""
        scope = self._resolve_scope(
            tenant_id=tenant_id,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            space_type=space_type,
            require_scope=self.require_scope,
        )
        interactions = self.system.memory_store.get_interactions(limit=limit)
        if scope.as_dict():
            interactions = [
                interaction
                for interaction in interactions
                if self._interaction_matches_scope(interaction, scope=scope)
            ]
        return [interaction.model_dump() for interaction in interactions]

    def add_constraint_seed(
        self,
        *,
        constraint_id: str,
        name: str,
        description: str,
        relation_types: list[RelationType | str] | None = None,
        lifecycle: ConstraintLifecycleStatus | str = ConstraintLifecycleStatus.PROPOSED,
        priority: int = 50,
        severity: str = "error",
        required_attributes: dict[str, Any] | None = None,
        required_attribute_keys: list[str] | None = None,
        forbidden_attributes: list[str] | None = None,
        require_object: bool = False,
        require_value: bool = False,
        value_regex: str | None = None,
        required_context: dict[str, Any] | None = None,
        min_candidate_confidence: float = 0.0,
        form_id: str | None = None,
        form_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register a user-defined dynamic constraint seed."""
        lifecycle_status = self._coerce_lifecycle(lifecycle)
        resolved_relations = [self._coerce_relation_type(r) for r in (relation_types or [])]

        evaluator = SeedConstraintEvaluator(
            constraint_id=constraint_id,
            constraint_name=name,
            description=description,
            applies_to_relations=resolved_relations,
            severity=severity,
            required_attributes=required_attributes,
            required_attribute_keys=required_attribute_keys,
            forbidden_attributes=forbidden_attributes,
            require_object=require_object,
            require_value=require_value,
            value_regex=value_regex,
        )

        scope = DynamicConstraintScope(
            relation_types=resolved_relations,
            required_context=required_context or {},
            min_candidate_confidence=min_candidate_confidence,
        )

        self.system.validator.register_dynamic(
            evaluator,
            lifecycle=lifecycle_status,
            priority=priority,
            scope=scope,
            form_id=form_id,
            form_metadata=form_metadata,
        )

        managed = self.system.validator.get_managed_constraint(constraint_id)
        return {
            "constraint_id": constraint_id,
            "name": name,
            "lifecycle": managed.lifecycle.value if managed else lifecycle_status.value,
            "priority": managed.priority if managed else priority,
            "source": managed.source.value if managed else "agent",
        }

    def discover_constraint_seeds(
        self,
        *,
        signal_limit: int = 1000,
        min_samples_per_relation: int = 20,
        min_rejections_per_relation: int = 6,
        min_gap: float = 0.35,
        min_gap_mode: str = "fixed",
        min_gap_floor: float = 0.15,
        min_gap_ceiling: float = 0.60,
        target_false_block_rate: float = 0.10,
        max_suggestions: int = 20,
    ) -> list[dict[str, Any]]:
        """Mine validation signals and synthesize candidate dynamic seeds."""
        discoverer = ConstraintSeedDiscoverer(
            min_samples_per_relation=min_samples_per_relation,
            min_rejections_per_relation=min_rejections_per_relation,
            min_gap=min_gap,
            min_gap_mode=min_gap_mode,
            min_gap_floor=min_gap_floor,
            min_gap_ceiling=min_gap_ceiling,
            target_false_block_rate=target_false_block_rate,
            max_suggestions=max_suggestions,
        )

        existing_ids = {
            managed.constraint_id for managed in self.system.validator.list_managed_constraints()
        }
        signals = self.system.validator.list_validation_signals(limit=signal_limit)
        discovered = discoverer.discover(
            validation_signals=signals,
            existing_constraint_ids=existing_ids,
        )
        return [seed.as_dict() for seed in discovered]

    def register_discovered_constraint_seeds(
        self,
        seeds: list[dict[str, Any] | DiscoveredConstraintSeed],
        *,
        lifecycle: ConstraintLifecycleStatus | str = ConstraintLifecycleStatus.SHADOW,
        priority: int = 45,
        continue_on_error: bool = True,
    ) -> dict[str, Any]:
        """Register discovered seeds into managed constraints (default shadow mode)."""
        lifecycle_status = self._coerce_lifecycle(lifecycle)

        registered: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []

        for item in seeds:
            payload = item.as_dict() if isinstance(item, DiscoveredConstraintSeed) else dict(item)
            try:
                response = self.add_constraint_seed(
                    constraint_id=str(payload["constraint_id"]),
                    name=str(payload["name"]),
                    description=str(payload["description"]),
                    relation_types=list(payload.get("relation_types", [])),
                    lifecycle=lifecycle_status,
                    priority=priority,
                    required_attribute_keys=list(payload.get("required_attribute_keys", [])),
                    require_value=bool(payload.get("require_value", False)),
                    form_metadata={
                        "discovery": {
                            "confidence": payload.get("confidence"),
                            "evidence_count": payload.get("evidence_count"),
                            "mining_rule": payload.get("mining_rule"),
                        }
                    },
                )
                registered.append(response)
            except Exception as exc:
                failed.append(
                    {
                        "constraint_id": payload.get("constraint_id", "unknown"),
                        "error": str(exc),
                    }
                )
                if not continue_on_error:
                    break

        return {
            "registered": registered,
            "failed": failed,
            "lifecycle": lifecycle_status.value,
        }

    def discover_and_register_constraint_seeds(
        self,
        *,
        signal_limit: int = 1000,
        min_samples_per_relation: int = 20,
        min_rejections_per_relation: int = 6,
        min_gap: float = 0.35,
        min_gap_mode: str = "fixed",
        min_gap_floor: float = 0.15,
        min_gap_ceiling: float = 0.60,
        target_false_block_rate: float = 0.10,
        max_suggestions: int = 20,
        lifecycle: ConstraintLifecycleStatus | str = ConstraintLifecycleStatus.SHADOW,
        priority: int = 45,
    ) -> dict[str, Any]:
        """Run autonomous mining and immediately register synthesized seeds."""
        discovered = self.discover_constraint_seeds(
            signal_limit=signal_limit,
            min_samples_per_relation=min_samples_per_relation,
            min_rejections_per_relation=min_rejections_per_relation,
            min_gap=min_gap,
            min_gap_mode=min_gap_mode,
            min_gap_floor=min_gap_floor,
            min_gap_ceiling=min_gap_ceiling,
            target_false_block_rate=target_false_block_rate,
            max_suggestions=max_suggestions,
        )

        registration = self.register_discovered_constraint_seeds(
            discovered,
            lifecycle=lifecycle,
            priority=priority,
            continue_on_error=True,
        )
        return {
            "discovered": discovered,
            "registration": registration,
        }

    def add_cardinality_constraint_seed(
        self,
        *,
        constraint_id: str,
        name: str,
        description: str,
        relation: RelationType | str,
        max_count: int,
        lifecycle: ConstraintLifecycleStatus | str = ConstraintLifecycleStatus.PROPOSED,
        priority: int = 50,
        severity: str = "error",
        require_same_subject: bool = True,
        required_context: dict[str, Any] | None = None,
        min_candidate_confidence: float = 0.0,
        form_id: str | None = None,
        form_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register a count/threshold-based dynamic constraint seed."""
        if max_count < 0:
            raise ValueError("max_count must be >= 0")

        lifecycle_status = self._coerce_lifecycle(lifecycle)
        relation_type = self._coerce_relation_type(relation)

        evaluator = CardinalitySeedConstraintEvaluator(
            constraint_id=constraint_id,
            constraint_name=name,
            description=description,
            relation=relation_type,
            max_count=max_count,
            severity=severity,
            require_same_subject=require_same_subject,
        )

        scope = DynamicConstraintScope(
            relation_types=[relation_type],
            required_context=required_context or {},
            min_candidate_confidence=min_candidate_confidence,
        )

        self.system.validator.register_dynamic(
            evaluator,
            lifecycle=lifecycle_status,
            priority=priority,
            scope=scope,
            form_id=form_id,
            form_metadata=form_metadata,
        )

        managed = self.system.validator.get_managed_constraint(constraint_id)
        return {
            "constraint_id": constraint_id,
            "name": name,
            "relation": relation_type.value,
            "max_count": max_count,
            "lifecycle": managed.lifecycle.value if managed else lifecycle_status.value,
            "priority": managed.priority if managed else priority,
            "source": managed.source.value if managed else "agent",
        }

    def add_temporal_cardinality_constraint_seed(
        self,
        *,
        constraint_id: str,
        name: str,
        description: str,
        relation: RelationType | str,
        max_count: int,
        window_seconds: int,
        lifecycle: ConstraintLifecycleStatus | str = ConstraintLifecycleStatus.PROPOSED,
        priority: int = 50,
        severity: str = "error",
        require_same_subject: bool = True,
        required_context: dict[str, Any] | None = None,
        min_candidate_confidence: float = 0.0,
        form_id: str | None = None,
        form_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register a rolling-window count seed (e.g., max N writes per 24h)."""
        if max_count < 0:
            raise ValueError("max_count must be >= 0")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")

        lifecycle_status = self._coerce_lifecycle(lifecycle)
        relation_type = self._coerce_relation_type(relation)

        evaluator = TemporalCardinalitySeedConstraintEvaluator(
            constraint_id=constraint_id,
            constraint_name=name,
            description=description,
            relation=relation_type,
            max_count=max_count,
            window_seconds=window_seconds,
            severity=severity,
            require_same_subject=require_same_subject,
        )

        scope = DynamicConstraintScope(
            relation_types=[relation_type],
            required_context=required_context or {},
            min_candidate_confidence=min_candidate_confidence,
        )

        self.system.validator.register_dynamic(
            evaluator,
            lifecycle=lifecycle_status,
            priority=priority,
            scope=scope,
            form_id=form_id,
            form_metadata=form_metadata,
        )

        managed = self.system.validator.get_managed_constraint(constraint_id)
        return {
            "constraint_id": constraint_id,
            "name": name,
            "relation": relation_type.value,
            "max_count": max_count,
            "window_seconds": window_seconds,
            "lifecycle": managed.lifecycle.value if managed else lifecycle_status.value,
            "priority": managed.priority if managed else priority,
            "source": managed.source.value if managed else "agent",
        }

    def list_constraint_seeds(self) -> list[dict[str, Any]]:
        """List managed constraints and their lifecycle metadata."""
        rows: list[dict[str, Any]] = []
        for managed in self.system.validator.list_managed_constraints():
            rows.append(
                {
                    "constraint_id": managed.constraint_id,
                    "name": managed.evaluator.constraint_name,
                    "description": managed.evaluator.description,
                    "lifecycle": managed.lifecycle.value,
                    "source": managed.source.value,
                    "priority": managed.priority,
                    "form_id": managed.form_id,
                    "shadow_hits": managed.shadow_hits,
                    "shadow_violations": managed.shadow_violations,
                }
            )
        rows.sort(key=lambda row: row["priority"], reverse=True)
        return rows

    def set_constraint_seed_lifecycle(
        self,
        constraint_id: str,
        lifecycle: ConstraintLifecycleStatus | str,
    ) -> bool:
        """Change lifecycle for a managed seed constraint."""
        lifecycle_status = self._coerce_lifecycle(lifecycle)
        return self.system.validator.set_lifecycle(constraint_id, lifecycle_status)

    def replay_constraint_seeds(
        self,
        candidates: list[CandidateFact],
    ) -> dict[str, dict[str, Any]]:
        """Replay dynamic seeds over candidate facts and return metrics."""
        metrics = self.system.validator.replay_dynamic_constraints(
            candidates=candidates,
            knowledge_state=self.system.memory_store,
        )
        return {
            cid: {
                "lifecycle": m.lifecycle.value,
                "evaluated_candidates": m.evaluated_candidates,
                "violations": m.violations,
                "incremental_blocks": m.incremental_blocks,
                "covered_existing_blocks": m.covered_existing_blocks,
                "trigger_rate": m.trigger_rate,
                "projected_false_block_rate": m.projected_false_block_rate,
                "projected_miss_coverage": m.projected_miss_coverage,
            }
            for cid, m in metrics.items()
        }

    def promote_constraint_seeds(
        self,
        replay_metrics: dict[str, dict[str, Any]],
        *,
        min_trigger_rate: float = 0.01,
        max_projected_false_block_rate: float = 0.02,
        min_candidates: int = 100,
    ) -> list[str]:
        """Promote dynamic seeds to active using replay metric dictionaries."""
        from grounded_memory.core.constraints import ConstraintReplayMetrics

        metrics: dict[str, ConstraintReplayMetrics] = {}
        for cid, payload in replay_metrics.items():
            lifecycle = self._coerce_lifecycle(payload.get("lifecycle", "proposed"))
            metrics[cid] = ConstraintReplayMetrics(
                constraint_id=cid,
                lifecycle=lifecycle,
                evaluated_candidates=int(payload.get("evaluated_candidates", 0)),
                violations=int(payload.get("violations", 0)),
                incremental_blocks=int(payload.get("incremental_blocks", 0)),
                covered_existing_blocks=int(payload.get("covered_existing_blocks", 0)),
            )

        return self.system.validator.promote_dynamic_constraints(
            replay_metrics=metrics,
            min_trigger_rate=min_trigger_rate,
            max_projected_false_block_rate=max_projected_false_block_rate,
            min_candidates=min_candidates,
        )

    def get_all(
        self,
        *,
        tenant_id: str | None = None,
        app_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        space_type: str | None = None,
    ) -> dict[str, Any]:
        """Return all entities and active facts from the backing store."""
        scope = self._resolve_scope(
            tenant_id=tenant_id,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            space_type=space_type,
            require_scope=self.require_scope,
        )
        entities = self.system.memory_store.get_all_entities()
        facts = self.system.memory_store.get_all_validated_facts()
        interactions = self.system.memory_store.get_interactions(limit=1000)

        if scope.as_dict():
            interactions = [
                interaction
                for interaction in interactions
                if self._interaction_matches_scope(interaction, scope=scope)
            ]
            facts = [fact for fact in facts if self._fact_matches_scope(fact, scope=scope)]
            scoped_entity_ids = {
                entity_id
                for fact in facts
                for entity_id in [fact.subject_id, fact.object_id]
                if entity_id is not None
            }
            entities = [
                entity
                for entity in entities
                if entity.id in scoped_entity_ids or self._entity_matches_scope(entity, scope=scope)
            ]

        return {
            "entities": [entity.model_dump() for entity in entities],
            "facts": [fact.model_dump() for fact in facts],
            "interactions": [interaction.model_dump() for interaction in interactions],
            "statistics": self.system.get_statistics(),
            "optimization": {
                "profile": self.optimization_profile.value,
                "settings": {
                    "max_seeds": self.optimization_settings.max_seeds,
                    "max_hops": self.optimization_settings.max_hops,
                    "max_facts": self.optimization_settings.max_facts,
                    "strategy": self.optimization_settings.strategy.value,
                },
            },
        }

    def runtime_status(self) -> dict[str, Any]:
        """Return operational metadata for health checks and service introspection."""
        llm_provider = None
        llm_model = None
        llm_base_url = None
        llm_timeout = None

        if self.llm_config is not None:
            llm_provider = self.llm_config.provider
            llm_model = self.llm_config.model
            llm_base_url = self.llm_config.base_url
            llm_timeout = self.llm_config.timeout

        return {
            "adapter": self.adapter,
            "registered_adapters": list_registered_adapters(),
            "domain_profile": self.domain_profile,
            "supported_profiles": list_supported_profiles(),
            "agent": {
                "configured": self.agent is not None,
                "type": type(self.agent).__name__ if self.agent is not None else None,
            },
            "llm": {
                "provider": llm_provider,
                "model": llm_model,
                "base_url": llm_base_url,
                "timeout_seconds": llm_timeout,
            },
            "storage": {
                "backend": getattr(self.system, "storage_backend", "memory"),
                "neo4j_enabled": self.system.has_neo4j,
                "store_type": type(self.system.memory_store).__name__,
            },
            "scope": {
                "require_scope": self.require_scope,
                "defaults": self.default_scope.as_dict(),
            },
            "statistics": self.system.get_statistics(),
            "optimization": {
                "profile": self.optimization_profile.value,
                "settings": {
                    "max_seeds": self.optimization_settings.max_seeds,
                    "max_hops": self.optimization_settings.max_hops,
                    "max_facts": self.optimization_settings.max_facts,
                    "strategy": self.optimization_settings.strategy.value,
                },
            },
        }

    def healthcheck(self) -> dict[str, Any]:
        """Return a lightweight readiness payload for service integrations."""
        status = self.runtime_status()
        return {
            "ok": bool(status["agent"]["configured"]),
            "adapter": status["adapter"],
            "domain_profile": status["domain_profile"],
            "agent_type": status["agent"]["type"],
            "neo4j_enabled": status["storage"]["neo4j_enabled"],
            "statistics": status["statistics"],
        }

    def close(self) -> None:
        """Close memory resources."""
        self.system.close()

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @staticmethod
    def _fact_to_result(
        fact: ValidatedFact,
        entities: dict[str, Any],
        score_payload: dict[str, Any] | None = None,
    ) -> SearchResult:
        subject = entities.get(fact.subject_id)
        obj = entities.get(fact.object_id) if fact.object_id else None

        return SearchResult(
            fact_id=fact.id,
            relation=fact.relation.value,
            subject_id=fact.subject_id,
            subject_name=subject.name if subject is not None else fact.subject_id,
            object_id=fact.object_id,
            object_name=obj.name if obj is not None else fact.object_id,
            value=fact.value,
            confidence=fact.confidence,
            valid_from=fact.valid_from.isoformat(),
            valid_to=fact.valid_to.isoformat() if fact.valid_to else None,
            score=(
                float(score_payload.get("final_score", score_payload.get("score")))
                if score_payload
                and score_payload.get("final_score", score_payload.get("score")) is not None
                else None
            ),
            signals=dict(score_payload or {}),
        )

    def _ensure_interaction(
        self,
        *,
        raw_text: str,
        actor: ActorType,
        tenant_id: str | None = None,
        app_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        space_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        scope = ScopeContext(
            tenant_id=tenant_id,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id or session_id,
            space_type=space_type,
        )
        return self._ensure_interaction_for_scope(
            raw_text=raw_text,
            actor=actor,
            scope=scope,
            metadata=metadata,
        )

    def _ensure_interaction_for_scope(
        self,
        *,
        raw_text: str,
        actor: ActorType,
        scope: ScopeContext,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persist an Interaction stamped with the provided scope envelope."""
        interaction = Interaction(
            raw_text=raw_text,
            actor=actor,
            tenant_id=scope.tenant_id,
            app_id=scope.app_id,
            user_id=scope.user_id,
            agent_id=scope.agent_id,
            run_id=scope.run_id,
            session_id=scope.run_id,
            space_type=scope.space_type,
            metadata=self._merge_scope_metadata(metadata=metadata, scope=scope),
        )
        self.system.memory_store.add_interaction(interaction)
        return interaction.id

    def _resolve_scope(
        self,
        *,
        tenant_id: str | None,
        app_id: str | None,
        user_id: str | None,
        agent_id: str | None,
        run_id: str | None,
        space_type: str | None,
        require_scope: bool,
    ) -> ScopeContext:
        resolved = ScopeContext(
            tenant_id=tenant_id or self.default_scope.tenant_id,
            app_id=app_id or self.default_scope.app_id,
            user_id=user_id or self.default_scope.user_id,
            agent_id=agent_id or self.default_scope.agent_id,
            run_id=run_id,
            space_type=space_type or self.default_scope.space_type,
        )

        has_scope_signal = any(
            value is not None
            for value in [
                resolved.tenant_id,
                resolved.app_id,
                resolved.user_id,
                resolved.agent_id,
                resolved.run_id,
                resolved.space_type,
            ]
        )

        if resolved.tenant_id and not resolved.app_id:
            raise ValueError("scope requires app_id when tenant_id is provided")
        if resolved.app_id and not resolved.tenant_id:
            raise ValueError("scope requires tenant_id when app_id is provided")
        if (resolved.tenant_id or resolved.app_id) and not resolved.user_id:
            raise ValueError("scope requires user_id when tenant_id/app_id are provided")

        if require_scope and not (resolved.tenant_id and resolved.app_id and resolved.user_id):
            raise ValueError(
                "scope is required: provide tenant_id, app_id, and user_id, or configure GM_SCOPE_* defaults"
            )

        if has_scope_signal and not resolved.space_type:
            resolved = ScopeContext(
                tenant_id=resolved.tenant_id,
                app_id=resolved.app_id,
                user_id=resolved.user_id,
                agent_id=resolved.agent_id,
                run_id=resolved.run_id,
                space_type="user",
            )

        return resolved

    @staticmethod
    def _merge_scope_metadata(
        *,
        metadata: dict[str, Any] | None,
        scope: ScopeContext,
    ) -> dict[str, Any]:
        merged = dict(metadata or {})
        merged.update(scope.as_dict())
        return merged

    @staticmethod
    def _coerce_actor(source: str) -> ActorType:
        normalized = source.strip().lower()
        if normalized in {"assistant", "agent"}:
            return ActorType.AGENT
        if normalized == "tool":
            return ActorType.TOOL
        if normalized == "system":
            return ActorType.SYSTEM
        return ActorType.USER

    @staticmethod
    def _coerce_entity_type(entity_type: EntityType | str) -> EntityType:
        if isinstance(entity_type, EntityType):
            return entity_type
        try:
            return EntityType(entity_type)
        except ValueError as exc:
            valid = ", ".join(sorted(member.value for member in EntityType))
            raise ValueError(f"Unknown entity_type '{entity_type}'. Valid types: {valid}") from exc

    @staticmethod
    def _coerce_relation_type(relation: RelationType | str) -> RelationType:
        if isinstance(relation, RelationType):
            return relation
        try:
            return RelationType(relation)
        except ValueError as exc:
            valid = ", ".join(sorted(member.value for member in RelationType))
            raise ValueError(f"Unknown relation '{relation}'. Valid relations: {valid}") from exc

    @staticmethod
    def _coerce_lifecycle(
        lifecycle: ConstraintLifecycleStatus | str,
    ) -> ConstraintLifecycleStatus:
        if isinstance(lifecycle, ConstraintLifecycleStatus):
            return lifecycle
        try:
            return ConstraintLifecycleStatus(str(lifecycle).lower())
        except ValueError as exc:
            valid = ", ".join(status.value for status in ConstraintLifecycleStatus)
            raise ValueError(f"Unknown lifecycle '{lifecycle}'. Valid values: {valid}") from exc

    @staticmethod
    def _coerce_retrieval_strategy(
        strategy: RetrievalStrategy | str,
    ) -> RetrievalStrategy:
        if isinstance(strategy, RetrievalStrategy):
            return strategy
        try:
            return RetrievalStrategy(str(strategy).lower())
        except ValueError as exc:
            valid = ", ".join(item.value for item in RetrievalStrategy)
            raise ValueError(
                f"Unknown retrieval strategy '{strategy}'. Valid values: {valid}"
            ) from exc

    @staticmethod
    def _coerce_optimization_profile(
        profile: OptimizationProfile | str,
    ) -> OptimizationProfile:
        if isinstance(profile, OptimizationProfile):
            return profile
        try:
            return OptimizationProfile(str(profile).lower())
        except ValueError as exc:
            valid = ", ".join(item.value for item in OptimizationProfile)
            raise ValueError(
                f"Unknown optimization profile '{profile}'. Valid values: {valid}"
            ) from exc

    def _apply_adapter_retrieval_weights(self) -> None:
        config = self._load_adapter_constraint_config(self.adapter)
        if not isinstance(config, dict):
            return

        retrieval_weights = config.get("retrieval_weights")
        if not isinstance(retrieval_weights, dict):
            return

        updates: dict[RelationType, dict[str, Any]] = {}
        for relation_name, payload in retrieval_weights.items():
            if not isinstance(payload, dict):
                continue
            if "weight" not in payload:
                continue

            try:
                relation = RelationType(str(relation_name).strip())
            except ValueError:
                continue

            try:
                weight = float(payload["weight"])
            except (TypeError, ValueError):
                continue

            update: dict[str, Any] = {
                "weight": weight,
            }
            if "is_safety_critical" in payload:
                update["is_safety_critical"] = bool(payload["is_safety_critical"])
            if "decay_per_hop" in payload:
                with suppress(TypeError, ValueError):
                    update["decay_per_hop"] = float(payload["decay_per_hop"])

            updates[relation] = update

        if updates:
            self.retriever.bulk_update_weights(updates)

    @staticmethod
    def _load_adapter_constraint_config(adapter_key: str) -> dict[str, Any] | None:
        if yaml is None:
            return None

        normalized = str(adapter_key or "").strip().lower()
        if not normalized:
            return None

        config_path = Path(__file__).resolve().parent / "configs" / f"{normalized}_constraints.yaml"
        if not config_path.exists():
            return None

        try:
            with config_path.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle)
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def _materialize_attribute_value_entity(
        self,
        value: str,
        *,
        scope: ScopeContext,
    ) -> str:
        scope_attrs = {
            key: val
            for key, val in scope.as_dict().items()
            if key
            in {"tenant_id", "app_id", "user_id", "agent_id", "run_id", "space_type", "scope_id"}
        }
        uniqueness_key = build_entity_uniqueness_key(
            name=value,
            entity_type=EntityType.FACILITY,
            attributes=scope_attrs,
        )
        entity_id = stable_entity_id(uniqueness_key)
        entity, _ = self.system.memory_store.find_or_create_entity(
            name=value,
            entity_type=EntityType.FACILITY,
            uniqueness_key=uniqueness_key,
            create_fn=lambda: Entity(
                id=entity_id,
                entity_type=EntityType.FACILITY,
                name=value,
                attributes=scope_attrs,
            ),
        )
        return entity.id

    def _lexical_fact_fallback(
        self,
        query: str,
        *,
        limit: int,
        scope: ScopeContext,
    ) -> list[dict[str, Any]]:
        """Fallback lexical scan for value-centric queries when seed selection has no match."""
        query_tokens = {
            token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) >= 2
        }
        if not query_tokens:
            return []

        entities = {entity.id: entity for entity in self.system.memory_store.get_all_entities()}
        candidates: list[tuple[float, ValidatedFact]] = []

        for fact in self.system.memory_store.get_all_validated_facts():
            if scope.as_dict() and not self._fact_matches_scope(fact, scope=scope):
                continue

            fact_text_parts = [
                fact.relation.value,
                fact.value or "",
            ]

            subject = entities.get(fact.subject_id)
            if subject is not None:
                fact_text_parts.append(subject.name)

            if fact.object_id:
                obj = entities.get(fact.object_id)
                fact_text_parts.append(obj.name if obj is not None else fact.object_id)

            fact_tokens = {
                token
                for token in re.findall(r"[a-z0-9]+", " ".join(fact_text_parts).lower())
                if len(token) >= 2
            }
            overlap = len(query_tokens & fact_tokens)
            if overlap <= 0:
                continue

            score = overlap / len(query_tokens)
            if fact.is_active:
                score += 0.1
            candidates.append((score, fact))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [
            self._fact_to_result(
                fact,
                entities,
                score_payload={"score": round(score, 6), "final_score": round(score, 6)},
            ).__dict__
            for score, fact in candidates[:limit]
        ]

    def _filter_results_by_scope(
        self,
        items: list[dict[str, Any]],
        *,
        scope: ScopeContext,
    ) -> list[dict[str, Any]]:
        """Filter search results to facts whose provenance interaction matches scope."""
        scoped: list[dict[str, Any]] = []
        for item in items:
            fact_id = str(item.get("fact_id") or "").strip()
            if not fact_id:
                continue
            fact = self.system.memory_store.get_fact(fact_id)
            if fact is not None and self._fact_matches_scope(fact, scope=scope):
                scoped.append(item)
        return scoped

    @staticmethod
    def _apply_result_threshold(
        items: list[dict[str, Any]],
        *,
        threshold: float | None,
    ) -> list[dict[str, Any]]:
        """Drop results below the requested score threshold."""
        if threshold is None:
            return items
        return [
            item
            for item in items
            if isinstance(item.get("score"), (int, float)) and float(item["score"]) >= threshold
        ]

    @staticmethod
    def _strip_debug_signals(item: dict[str, Any]) -> dict[str, Any]:
        """Hide internal reranking signals from default SDK responses."""
        cleaned = dict(item)
        cleaned.pop("signals", None)
        return cleaned

    def _fact_matches_scope(self, fact: ValidatedFact, *, scope: ScopeContext) -> bool:
        """Check whether a fact's provenance interaction matches the requested scope."""
        expected = scope.as_dict()
        if not expected:
            return True

        interaction = self.system.memory_store.get_interaction(fact.source_interaction_id)
        if interaction is not None:
            return self._interaction_matches_scope(interaction, scope=scope)

        # Fallback for stores without interaction lookup: use fact attributes.
        attrs = fact.attributes or {}
        for key, value in expected.items():
            if value is None:
                continue
            if attrs.get(key) != value:
                return False
        return True

    @staticmethod
    def _interaction_matches_scope(interaction: Interaction, *, scope: ScopeContext) -> bool:
        expected = scope.as_dict()
        if not expected:
            return True

        metadata = interaction.metadata or {}
        for key, value in expected.items():
            if value is None:
                continue

            if key == "scope_id":
                actual_scope_id = metadata.get("scope_id")
                if actual_scope_id is None:
                    tenant = getattr(interaction, "tenant_id", None) or metadata.get("tenant_id")
                    app = getattr(interaction, "app_id", None) or metadata.get("app_id")
                    user = getattr(interaction, "user_id", None) or metadata.get("user_id")
                    actual_scope_id = f"{tenant}:{app}:{user}" if tenant and app and user else None
                if actual_scope_id != value:
                    return False
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

            if actual != value:
                return False

        return True

    @staticmethod
    def _entity_matches_scope(entity: Entity, *, scope: ScopeContext) -> bool:
        expected = scope.as_dict()
        if not expected:
            return True
        attrs = entity.attributes or {}
        for key, value in expected.items():
            if value is None:
                continue
            if attrs.get(key) != value:
                return False
        return True

    @staticmethod
    def _fact_semantic_key(fact: ValidatedFact) -> str | None:
        return build_fact_semantic_key(
            subject_id="memory",
            relation=fact.relation,
            object_id=fact.object_id,
            value=fact.value,
            attributes=fact.attributes,
            include_subject=False,
        )

    @staticmethod
    def _fact_exact_match(left: ValidatedFact, right: ValidatedFact) -> bool:
        return left.object_id == right.object_id and fact_values_equal(left.value, right.value)

    @staticmethod
    def _memory_line_from_result(item: dict[str, Any]) -> str:
        """Render one retrieved result into a compact, prompt-friendly bullet line."""
        value = str(item.get("value") or "").strip()
        subject = str(item.get("subject_name") or item.get("subject_id") or "something").strip()
        relation = str(item.get("relation") or "RELATED_TO").strip()
        object_name = str(item.get("object_name") or item.get("object_id") or "").strip()
        score = item.get("score")
        score_suffix = f" [score={float(score):.2f}]" if isinstance(score, (int, float)) else ""

        if value:
            return f"- {subject}: {value}{score_suffix}"
        if object_name:
            return f"- {subject} [{relation}] {object_name}{score_suffix}"
        return f"- {subject} [{relation}]{score_suffix}"
