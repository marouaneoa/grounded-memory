"""
Graph-Based Retrieval

This module implements graph-based retrieval strategies for the memory system.
The knowledge is stored as a temporal property graph where:
- Entities are nodes
- ValidatedFacts are edges (with temporal boundaries)

Hybrid architecture support:
- When a HybridMemoryStore with Neo4j is provided, graph queries are
  delegated to Neo4j for native graph traversal performance.
- For temporal (point-in-time) queries, or when Neo4j is unavailable,
  falls back to the in-memory NetworkX approach.

Retrieval strategies:
1. Entity-centric seed identification
2. Multi-hop expansion with relationship weighting
3. Safety-priority weighting (high-risk info first)
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Protocol

import networkx as nx

from grounded_memory.core.models import (
    AnswerContext,
    Entity,
    RelationType,
    ValidatedFact,
    as_utc_datetime,
    datetime_on_or_after,
)
from grounded_memory.core.tuple_normalization import (
    build_fact_semantic_key,
    resolve_attribute_key,
)

# Neo4j / hybrid support (optional)
try:
    from grounded_memory.core.hybrid_store import HybridMemoryStore

    HAS_HYBRID = True
except ImportError:
    HAS_HYBRID = False
    HybridMemoryStore = None

logger = logging.getLogger(__name__)


# =============================================================================
# Retrieval Configuration
# =============================================================================


class RetrievalStrategy(str, Enum):
    """Strategy for retrieving facts from the knowledge graph."""

    BREADTH_FIRST = "breadth_first"  # Standard BFS expansion
    WEIGHTED = "weighted"  # Weight-based expansion
    SAFETY_PRIORITY = "safety_priority"  # Prioritize safety-critical facts


class RelationshipPreset(str, Enum):
    """Named default relationship-weight presets."""

    GENERIC = "generic"
    SAFETY = "safety"


class GraphMemoryStore(Protocol):
    """Protocol for stores used by graph retrieval."""

    def get_entity(self, entity_id: str) -> Entity | None: ...

    def get_active_facts_for_entity(
        self,
        entity_id: str,
        at_time: datetime | None = None,
    ) -> list[ValidatedFact]: ...

    def get_connected_entities(
        self,
        entity_id: str,
        max_hops: int = 1,
        at_time: datetime | None = None,
    ) -> dict[str, int]: ...

    def get_subgraph(
        self,
        entity_ids: list[str],
        at_time: datetime | None = None,
    ) -> tuple[dict[str, Entity], list[ValidatedFact]]: ...


@dataclass
class RelationshipWeight:
    """Weight configuration for relationship types during retrieval."""

    relation: RelationType
    weight: float = 1.0
    is_safety_critical: bool = False
    decay_per_hop: float = 0.2  # Decay ratio in [0, 1] applied for next hop carry


@dataclass(frozen=True)
class QueryProfile:
    """Simple intent profile used to rebalance retrieval signals."""

    prefers_recency: bool = False
    prefers_profile_facts: bool = False
    prefers_safety: bool = False
    prefers_relational_context: bool = False


@dataclass(frozen=True)
class RetrievalSignalWeights:
    """Resolved signal weights for a query."""

    relation: float
    relevance: float
    recency: float
    safety: float
    profile_match: float


def _clone_relationship_weights(
    weights: dict[RelationType, RelationshipWeight],
) -> dict[RelationType, RelationshipWeight]:
    """Clone weight map to avoid mutating shared defaults."""
    return {
        relation: RelationshipWeight(
            relation=weight.relation,
            weight=weight.weight,
            is_safety_critical=weight.is_safety_critical,
            decay_per_hop=weight.decay_per_hop,
        )
        for relation, weight in weights.items()
    }


def _resolve_relation_type(relation: RelationType | str) -> RelationType:
    """Normalize relation type from enum or string value."""
    if isinstance(relation, RelationType):
        return relation
    try:
        return RelationType(relation)
    except ValueError as exc:
        allowed = ", ".join(sorted(r.value for r in RelationType))
        raise ValueError(
            f"Unknown relation type '{relation}'. Allowed relations: {allowed}"
        ) from exc


# Generic defaults (domain-agnostic)
DEFAULT_GENERIC_WEIGHTS: dict[RelationType, RelationshipWeight] = {
    RelationType.RELATED_TO: RelationshipWeight(
        relation=RelationType.RELATED_TO,
        weight=1.0,
    ),
}


# Safety-oriented defaults
DEFAULT_SAFETY_WEIGHTS: dict[RelationType, RelationshipWeight] = {
    # Safety-critical relationships (highest weight)
    RelationType.HAS_ALLERGY: RelationshipWeight(
        relation=RelationType.HAS_ALLERGY,
        weight=10.0,
        is_safety_critical=True,
    ),
    RelationType.CONTRAINDICATED_WITH: RelationshipWeight(
        relation=RelationType.CONTRAINDICATED_WITH,
        weight=10.0,
        is_safety_critical=True,
    ),
    # High-importance relationships
    RelationType.PRESCRIBED: RelationshipWeight(
        relation=RelationType.PRESCRIBED,
        weight=5.0,
        is_safety_critical=True,
    ),
    RelationType.HAS_CONDITION: RelationshipWeight(
        relation=RelationType.HAS_CONDITION,
        weight=4.0,
    ),
    # Medium-importance relationships
    RelationType.CONTAINS_INGREDIENT: RelationshipWeight(
        relation=RelationType.CONTAINS_INGREDIENT,
        weight=3.0,
    ),
    RelationType.SAME_THERAPEUTIC_CLASS: RelationshipWeight(
        relation=RelationType.SAME_THERAPEUTIC_CLASS,
        weight=3.0,
    ),
    RelationType.TREATS: RelationshipWeight(
        relation=RelationType.TREATS,
        weight=3.0,
    ),
    # Lower-importance (but still relevant)
    RelationType.DISCONTINUED: RelationshipWeight(
        relation=RelationType.DISCONTINUED,
        weight=2.0,
    ),
    # Default for other relationships
    RelationType.RELATED_TO: RelationshipWeight(
        relation=RelationType.RELATED_TO,
        weight=1.0,
    ),
}


def _relationship_weights_for_preset(
    preset: RelationshipPreset,
) -> dict[RelationType, RelationshipWeight]:
    if preset == RelationshipPreset.GENERIC:
        return _clone_relationship_weights(DEFAULT_GENERIC_WEIGHTS)
    if preset == RelationshipPreset.SAFETY:
        return _clone_relationship_weights(DEFAULT_SAFETY_WEIGHTS)
    raise ValueError(f"Unsupported relationship preset: {preset}")


def _iter_entities(store: GraphMemoryStore) -> Iterable[Entity]:
    if hasattr(store, "iter_entities"):
        return store.iter_entities()  # type: ignore[attr-defined]
    if hasattr(store, "get_all_entities"):
        return store.get_all_entities()  # type: ignore[attr-defined]
    if hasattr(store, "_entities"):
        entities = store._entities  # type: ignore[attr-defined]
        if isinstance(entities, dict):
            return entities.values()
    raise RuntimeError(
        "Memory store does not expose entity iteration. "
        "Implement iter_entities() or get_all_entities()."
    )


def _iter_active_facts(store: GraphMemoryStore, at_time: datetime) -> Iterable[ValidatedFact]:
    if hasattr(store, "iter_active_facts"):
        return store.iter_active_facts(at_time)  # type: ignore[attr-defined]
    if hasattr(store, "get_all_validated_facts"):
        return [
            fact
            for fact in store.get_all_validated_facts()  # type: ignore[attr-defined]
            if fact.is_active_at(at_time)
        ]
    if hasattr(store, "_facts"):
        facts = store._facts  # type: ignore[attr-defined]
        if isinstance(facts, dict):
            return [fact for fact in facts.values() if fact.is_active_at(at_time)]
    raise RuntimeError(
        "Memory store does not expose fact iteration. "
        "Implement iter_active_facts() or get_all_validated_facts()."
    )


def _find_seed_entity_ids(query: str, store: GraphMemoryStore) -> list[str]:
    if hasattr(store, "find_entity_ids_by_name_fragment"):
        return list(store.find_entity_ids_by_name_fragment(query))  # type: ignore[attr-defined]

    query_lower = query.lower()
    matched: set[str] = set()
    for entity in _iter_entities(store):
        if entity.name.lower() in query_lower:
            matched.add(entity.id)
    return list(matched)


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) >= 2}


@dataclass
class QueryHintRegistry:
    """Pluggable keyword hints for query-profile inference.

    Domain adapters can instantiate this with their own hint sets or
    register additional hints at runtime.
    """

    temporal: set[str] = field(
        default_factory=lambda: {
            "latest",
            "recent",
            "recently",
            "current",
            "currently",
            "now",
            "new",
            "updated",
            "changed",
            "last",
        }
    )
    profile: set[str] = field(
        default_factory=lambda: {
            "prefer",
            "preference",
            "preferences",
            "favorite",
            "likes",
            "like",
            "dislike",
            "dislikes",
            "hobby",
            "habit",
            "bio",
            "profile",
            "who",
        }
    )
    safety: set[str] = field(
        default_factory=lambda: {
            "avoid",
            "warning",
            "warnings",
            "unsafe",
            "risk",
            "critical",
        }
    )
    relational: set[str] = field(
        default_factory=lambda: {
            "relationship",
            "related",
            "connected",
            "connection",
            "between",
            "with",
            "through",
            "network",
        }
    )

    def register(self, category: str, hints: set[str]) -> None:
        """Add extra hints to a category."""
        attr = category if category != "profile" else "profile"
        if not hasattr(self, attr):
            raise ValueError(f"Unknown hint category: {category}")
        getattr(self, attr).update(hints)


def _infer_query_profile(query: str, registry: QueryHintRegistry | None = None) -> QueryProfile:
    """Infer coarse retrieval intent from lexical hints."""
    reg = registry or QueryHintRegistry()
    tokens = _tokenize(query)
    return QueryProfile(
        prefers_recency=bool(tokens & reg.temporal),
        prefers_profile_facts=bool(tokens & reg.profile),
        prefers_safety=bool(tokens & reg.safety),
        prefers_relational_context=bool(tokens & reg.relational),
    )


def _resolve_signal_weights(profile: QueryProfile) -> RetrievalSignalWeights:
    """Map query intent to ranking weights."""
    relation = 0.24
    relevance = 0.44
    recency = 0.16
    safety = 0.08
    profile_match = 0.08

    if profile.prefers_recency:
        recency += 0.10
        relation -= 0.04
        relevance -= 0.03
        profile_match -= 0.03
    if profile.prefers_profile_facts:
        profile_match += 0.10
        relation -= 0.04
        safety -= 0.02
        recency -= 0.04
    if profile.prefers_safety:
        safety += 0.16
        relevance -= 0.05
        relation -= 0.05
        profile_match -= 0.03
        recency -= 0.03
    if profile.prefers_relational_context:
        relation += 0.10
        relevance -= 0.04
        profile_match -= 0.03
        recency -= 0.03

    total = relation + relevance + recency + safety + profile_match
    return RetrievalSignalWeights(
        relation=relation / total,
        relevance=relevance / total,
        recency=recency / total,
        safety=safety / total,
        profile_match=profile_match / total,
    )


def _fact_semantic_key(fact: ValidatedFact) -> str:
    semantic_key = build_fact_semantic_key(
        subject_id=fact.subject_id,
        relation=fact.relation,
        object_id=fact.object_id,
        value=fact.value,
        attributes=fact.attributes,
        include_subject=True,
    )
    return semantic_key or f"{fact.subject_id}|{fact.relation.value}|v:"


def _fact_profile_match_signal(fact: ValidatedFact, profile: QueryProfile) -> float:
    if not profile.prefers_profile_facts:
        return 0.0
    attribute_key = resolve_attribute_key(fact.value, fact.attributes) or ""
    if fact.relation == RelationType.HAS_ATTRIBUTE and attribute_key:
        return 1.0
    if fact.relation == RelationType.HAS_ATTRIBUTE and fact.value:
        return 0.6
    return 0.0


def _entity_query_score(entity: Entity, query_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0

    name_tokens = _tokenize(entity.name)
    attr_tokens = _tokenize(" ".join(str(v) for v in entity.attributes.values()))

    overlap_name = len(name_tokens & query_tokens)
    overlap_attr = len(attr_tokens & query_tokens)

    score = 0.0
    score += 1.0 * overlap_name
    score += 0.35 * overlap_attr

    if entity.name.lower() in " ".join(sorted(query_tokens)):
        score += 0.5

    return score


def select_seed_entities(
    query: str,
    memory_store: GraphMemoryStore,
    max_seeds: int = 6,
) -> list[str]:
    """Select seed entities using lexical matching and query-token relevance scoring."""
    direct_matches = set(_find_seed_entity_ids(query, memory_store))

    query_tokens = _tokenize(query)
    scored_entities: list[tuple[float, str]] = []
    for entity in _iter_entities(memory_store):
        score = _entity_query_score(entity, query_tokens)
        if score > 0:
            scored_entities.append((score, entity.id))

    scored_entities.sort(key=lambda item: item[0], reverse=True)

    ranked_ids: list[str] = []
    for _, entity_id in scored_entities:
        if entity_id not in ranked_ids:
            ranked_ids.append(entity_id)

    for entity_id in direct_matches:
        if entity_id in ranked_ids:
            ranked_ids.remove(entity_id)
        ranked_ids.insert(0, entity_id)

    return ranked_ids[:max_seeds]


# =============================================================================
# Graph Retriever
# =============================================================================


class GraphRetriever:
    """
    Graph-based retrieval for the knowledge graph.

    This implements multi-hop expansion from seed entities with
    configurable weighting strategies. Relations marked as
    safety-critical in the active weight profile are prioritized
    during retrieval.

    When backed by a HybridMemoryStore with Neo4j, current-time queries
    are delegated to Neo4j for native graph traversal. Temporal queries
    and fallback use the NetworkX-based approach on the MemoryStore.

    Since we validate at write-time, we don't need to re-check
    constraints during retrieval. However, retrieval order
    matters for:
    1. Explanation generation (show safety info first)
    2. Context window optimization (most important facts first)
    3. Cascade detection for critical relation neighborhoods
    """

    def __init__(
        self,
        memory_store: GraphMemoryStore,
        relationship_weights: dict[RelationType, RelationshipWeight] | None = None,
        relationship_preset: RelationshipPreset | str = RelationshipPreset.GENERIC,
        default_decay_per_hop: float = 0.2,
        hint_registry: QueryHintRegistry | None = None,
    ):
        """
        Initialize the retriever.

        Args:
            memory_store: The memory store to retrieve from
                          (can be MemoryStore or HybridMemoryStore)
            relationship_weights: Custom relationship weights
            default_decay_per_hop: Default decay ratio in [0, 1] used when a
                relation does not define its own decay value
            hint_registry: Optional query-hint registry for domain-specific
                lexical cues (e.g., healthcare safety terms).
        """
        self.memory_store = memory_store
        self.default_decay_per_hop = max(0.0, min(default_decay_per_hop, 1.0))
        self.relationship_preset = RelationshipPreset(relationship_preset)
        if relationship_weights is None:
            self.weights = _relationship_weights_for_preset(self.relationship_preset)
        else:
            self.weights = _clone_relationship_weights(relationship_weights)

        self.hint_registry = hint_registry or QueryHintRegistry()

        # Detect if we have Neo4j-backed hybrid store
        self._neo4j_store = None
        if HAS_HYBRID and isinstance(memory_store, HybridMemoryStore) and memory_store.has_neo4j:
            self._neo4j_store = memory_store.neo4j
            logger.info("GraphRetriever using Neo4j for graph queries")

    def get_weight(self, relation: RelationType | str) -> RelationshipWeight:
        """Get relation weight configuration, creating a neutral profile if missing."""
        relation_type = _resolve_relation_type(relation)
        if relation_type not in self.weights:
            self.weights[relation_type] = RelationshipWeight(
                relation=relation_type,
                weight=1.0,
                is_safety_critical=False,
                decay_per_hop=self.default_decay_per_hop,
            )
        return self.weights[relation_type]

    def set_weight(
        self,
        relation: RelationType | str,
        *,
        weight: float,
        is_safety_critical: bool | None = None,
        decay_per_hop: float | None = None,
    ) -> RelationshipWeight:
        """
        Set or update relation retrieval weight settings at runtime.

        Returns:
            The updated RelationshipWeight profile.
        """
        if weight <= 0:
            raise ValueError("weight must be greater than 0")

        relation_type = _resolve_relation_type(relation)
        existing = self.get_weight(relation_type)
        existing.weight = weight
        if is_safety_critical is not None:
            existing.is_safety_critical = is_safety_critical
        if decay_per_hop is not None:
            existing.decay_per_hop = max(0.0, min(decay_per_hop, 1.0))
        self.weights[relation_type] = existing
        return existing

    def bulk_update_weights(
        self,
        updates: dict[RelationType | str, RelationshipWeight | dict[str, Any]],
    ) -> None:
        """Apply multiple weight updates in one call."""
        for relation, config in updates.items():
            relation_type = _resolve_relation_type(relation)

            if isinstance(config, RelationshipWeight):
                self.set_weight(
                    relation_type,
                    weight=config.weight,
                    is_safety_critical=config.is_safety_critical,
                    decay_per_hop=config.decay_per_hop,
                )
                continue

            if not isinstance(config, dict):
                raise ValueError(
                    f"Invalid weight config for {relation_type.value}; expected dict or RelationshipWeight"
                )

            if "weight" not in config:
                raise ValueError(
                    f"Missing 'weight' for relation {relation_type.value} in bulk update"
                )

            self.set_weight(
                relation_type,
                weight=float(config["weight"]),
                is_safety_critical=config.get("is_safety_critical"),
                decay_per_hop=config.get("decay_per_hop"),
            )

    def load_weight_config(self, config: dict[str, Any]) -> None:
        """
        Load relationship weights from a config dictionary.

        Expected shape:
            {
              "default_decay_per_hop": 0.2,
              "relations": {
                "HAS_ALLERGY": {"weight": 12.0, "is_safety_critical": true, "decay_per_hop": 0.1}
              }
            }
        """
        if "default_decay_per_hop" in config:
            self.default_decay_per_hop = max(0.0, min(float(config["default_decay_per_hop"]), 1.0))

        relation_config = config.get("relations", {})
        if not isinstance(relation_config, dict):
            raise ValueError("weight config 'relations' must be a dictionary")

        self.bulk_update_weights(relation_config)

    def export_weight_config(self) -> dict[str, Any]:
        """Export the active relationship weight configuration."""
        return {
            "default_decay_per_hop": self.default_decay_per_hop,
            "relationship_preset": self.relationship_preset.value,
            "relations": {
                relation.value: {
                    "weight": cfg.weight,
                    "is_safety_critical": cfg.is_safety_critical,
                    "decay_per_hop": cfg.decay_per_hop,
                }
                for relation, cfg in sorted(self.weights.items(), key=lambda item: item[0].value)
            },
        }

    @property
    def has_neo4j(self) -> bool:
        """Check if Neo4j is available for graph queries."""
        return self._neo4j_store is not None

    def _is_current_time(self, at_time: datetime) -> bool:
        """Check if the query is for approximately the current time."""
        delta = abs(
            (as_utc_datetime(datetime.now(timezone.utc)) - as_utc_datetime(at_time)).total_seconds()
        )
        return delta < 5  # Within 5 seconds = current

    def retrieve(
        self,
        query: str,
        seed_entities: list[str],
        max_hops: int = 2,
        max_facts: int = 50,
        strategy: RetrievalStrategy = RetrievalStrategy.SAFETY_PRIORITY,
        at_time: datetime | None = None,
        lookback_days: int | None = None,
        user_id: str | None = None,
        scope: dict[str, str] | None = None,
    ) -> AnswerContext:
        """
        Retrieve relevant facts for answering a query.

        Args:
            query: The question or context for retrieval
            seed_entities: Starting entity IDs
            max_hops: Maximum graph traversal depth
            max_facts: Maximum facts to return
            strategy: Retrieval strategy to use
            at_time: Point-in-time for fact validity

        Returns:
            AnswerContext with retrieved facts
        """
        at_time = at_time or datetime.now(timezone.utc)

        use_neo4j_path = self.has_neo4j and self._is_current_time(at_time)
        facts: list[ValidatedFact] = []
        entities: dict[str, Entity] = {}
        if use_neo4j_path:
            try:
                facts, entities = self._retrieve_neo4j(
                    seed_entities,
                    max_hops,
                    max_facts,
                    strategy,
                    user_id=user_id,
                    scope=scope,
                )
            except Exception as e:
                logger.warning("Neo4j retrieval failed, falling back to in-memory: %s", e)
                use_neo4j_path = False
        if not use_neo4j_path:
            facts, entities = self._retrieve_fallback(
                seed_entities,
                max_hops,
                max_facts,
                at_time,
                strategy,
                user_id=user_id,
                scope=scope,
                query=query,
            )

        facts = self._apply_temporal_context(
            facts=facts,
            at_time=at_time,
            lookback_days=lookback_days,
        )

        facts, score_map = self._rerank_facts(
            query=query,
            facts=facts,
            entities=entities,
            at_time=at_time,
            max_facts=max_facts,
        )

        return AnswerContext(
            query=query,
            timestamp=at_time,
            seed_entities=seed_entities,
            facts=facts,
            entities=entities,
            retrieval_metadata={
                "strategy": strategy.value,
                "max_hops": max_hops,
                "facts_retrieved": len(facts),
                "entities_retrieved": len(entities),
                "user_scope": user_id,
                "scope": scope or {},
                "temporal_context": {
                    "at_time": at_time.isoformat(),
                    "lookback_days": lookback_days,
                },
                "scores": score_map,
            },
        )

    def _apply_temporal_context(
        self,
        *,
        facts: list[ValidatedFact],
        at_time: datetime,
        lookback_days: int | None,
    ) -> list[ValidatedFact]:
        """Filter retrieved facts using optional temporal context controls."""
        if lookback_days is None:
            return facts
        if lookback_days <= 0:
            raise ValueError("lookback_days must be greater than 0")

        cutoff = at_time - timedelta(days=lookback_days)
        return [fact for fact in facts if datetime_on_or_after(fact.valid_from, cutoff)]

    def _rerank_facts(
        self,
        query: str,
        facts: list[ValidatedFact],
        entities: dict[str, Entity],
        at_time: datetime,
        max_facts: int,
    ) -> tuple[list[ValidatedFact], dict[str, dict[str, float | str]]]:
        """Rerank facts with calibrated relevance-gated scoring and content-aware diversity."""
        if not facts:
            return facts, {}

        query_tokens = _tokenize(query)
        query_profile = _infer_query_profile(query, registry=self.hint_registry)
        signal_weights = _resolve_signal_weights(query_profile)

        # Calibrate relation signal against the configured max weight for better dynamic range
        max_relation_weight = max((w.weight for w in self.weights.values()), default=1.0)

        scored: list[tuple[float, ValidatedFact]] = []
        score_map: dict[str, dict[str, float | str]] = {}
        for fact in facts:
            relation_profile = self.get_weight(fact.relation)
            relation_weight = max(relation_profile.weight, 0.001)

            # Calibrated relation signal: normalize by max configured weight
            relation_signal = relation_weight / max(max_relation_weight, 1.0)

            subject = entities.get(fact.subject_id)
            obj = entities.get(fact.object_id) if fact.object_id is not None else None

            fact_text_parts = [
                fact.relation.value,
                subject.name if subject is not None else fact.subject_id,
                obj.name if obj is not None else "",
                fact.value or "",
            ]
            fact_tokens = _tokenize(" ".join(part for part in fact_text_parts if part))
            relevance_signal = (
                len(query_tokens & fact_tokens) / max(len(query_tokens), 1) if query_tokens else 0.0
            )

            age_seconds = max(
                (as_utc_datetime(at_time) - as_utc_datetime(fact.valid_from)).total_seconds(),
                0.0,
            )
            recency_signal = 1.0 / (1.0 + age_seconds / 86400.0)

            safety_signal = 1.0 if relation_profile.is_safety_critical else 0.0
            profile_match_signal = _fact_profile_match_signal(fact, query_profile)

            # Relevance-gated scoring: relevance modulates structural signals,
            # but safety gets a direct contribution and relevance itself contributes
            # directly so that highly relevant facts are not under-weighted.
            structural_score = (
                signal_weights.relation * relation_signal
                + signal_weights.recency * recency_signal
                + signal_weights.profile_match * profile_match_signal
            )

            combined_score = (
                relevance_signal * structural_score  # gated structural
                + signal_weights.safety * safety_signal  # safety bypass
                + signal_weights.relevance * relevance_signal  # direct relevance
            )

            scored.append((combined_score, fact))
            score_map[fact.id] = {
                "score": round(combined_score, 6),
                "relation": round(relation_signal, 6),
                "relevance": round(relevance_signal, 6),
                "recency": round(recency_signal, 6),
                "safety": round(safety_signal, 6),
                "profile_match": round(profile_match_signal, 6),
                "semantic_key": _fact_semantic_key(fact),
            }

        scored.sort(key=lambda item: item[0], reverse=True)

        seen_ids: set[str] = set()
        semantic_key_counts: dict[str, int] = {}
        ranked: list[ValidatedFact] = []
        reranked_score_map: dict[str, dict[str, float | str]] = {}
        # Content-aware diversity: track token sets of selected facts
        selected_fact_tokens: dict[str, set[str]] = {}

        for base_score, fact in scored:
            if fact.id in seen_ids:
                continue

            semantic_key = _fact_semantic_key(fact)

            # Compute token overlap redundancy with already-selected facts
            subject = entities.get(fact.subject_id)
            obj = entities.get(fact.object_id) if fact.object_id is not None else None
            fact_text_parts = [
                fact.relation.value,
                subject.name if subject is not None else fact.subject_id,
                obj.name if obj is not None else "",
                fact.value or "",
            ]
            candidate_tokens = _tokenize(" ".join(part for part in fact_text_parts if part))

            max_overlap_ratio = 0.0
            for selected_tokens in selected_fact_tokens.values():
                if selected_tokens and candidate_tokens:
                    overlap = len(candidate_tokens & selected_tokens)
                    max_overlap_ratio = max(
                        max_overlap_ratio,
                        overlap / max(len(candidate_tokens), len(selected_tokens), 1),
                    )

            semantic_penalty = 0.08 * semantic_key_counts.get(semantic_key, 0)
            redundancy_penalty = 0.10 * max_overlap_ratio
            diversity_penalty = semantic_penalty + redundancy_penalty

            final_score = max(base_score - diversity_penalty, 0.0)
            score_map[fact.id]["diversity_penalty"] = round(diversity_penalty, 6)
            score_map[fact.id]["final_score"] = round(final_score, 6)

            seen_ids.add(fact.id)
            ranked.append(fact)
            semantic_key_counts[semantic_key] = semantic_key_counts.get(semantic_key, 0) + 1
            selected_fact_tokens[fact.id] = candidate_tokens
            reranked_score_map[fact.id] = score_map[fact.id]
            if len(ranked) >= max_facts:
                break

        ranked.sort(
            key=lambda fact: float(reranked_score_map.get(fact.id, {}).get("final_score", 0.0)),
            reverse=True,
        )
        return ranked, reranked_score_map

    def select_seed_entities(
        self,
        query: str,
        max_seeds: int = 6,
    ) -> list[str]:
        """Select seed entities using retrieval-aware lexical relevance."""
        return select_seed_entities(query, self.memory_store, max_seeds=max_seeds)

    # =========================================================================
    # Neo4j-Backed Retrieval (primary path when Neo4j is available)
    # =========================================================================

    def _retrieve_neo4j(
        self,
        seed_entities: list[str],
        max_hops: int,
        max_facts: int,
        strategy: RetrievalStrategy,
        user_id: str | None = None,
        scope: dict[str, str] | None = None,
    ) -> tuple[list[ValidatedFact], dict[str, Entity]]:
        """
        Retrieve facts using Neo4j native graph traversal.

        This replaces the NetworkX-based approach for current-time queries.
        """
        all_entities: dict[str, Entity] = {}
        all_facts: list[ValidatedFact] = []
        seen_fact_ids: set[str] = set()

        if strategy == RetrievalStrategy.SAFETY_PRIORITY:
            safety_relations = [
                relation for relation, weight in self.weights.items() if weight.is_safety_critical
            ]

            # Phase 1: Safety-critical facts first
            if safety_relations:
                for seed in seed_entities:
                    safety_facts = self._neo4j_store.get_safety_critical_facts(
                        seed,
                        max_hops=max_hops,
                        relation_types=safety_relations,
                        scope_id=(scope or {}).get("scope_id"),
                    )
                    for fact in safety_facts:
                        if not self._fact_matches_scope(fact, user_id=user_id, scope=scope):
                            continue
                        if fact.id not in seen_fact_ids:
                            seen_fact_ids.add(fact.id)
                            all_facts.append(fact)

            # Phase 2: Fill with remaining facts up to max_facts
            remaining = max_facts - len(all_facts)
            if remaining > 0:
                for seed in seed_entities:
                    entities, facts = self._neo4j_store.get_neighbors(
                        seed,
                        max_hops=max_hops,
                        scope_id=(scope or {}).get("scope_id"),
                    )
                    all_entities.update(entities)
                    for fact in facts:
                        if not self._fact_matches_scope(fact, user_id=user_id, scope=scope):
                            continue
                        if fact.id not in seen_fact_ids and len(all_facts) < max_facts:
                            seen_fact_ids.add(fact.id)
                            all_facts.append(fact)
        else:
            # BFS or WEIGHTED: use get_neighbors for all seeds
            for seed in seed_entities:
                entities, facts = self._neo4j_store.get_neighbors(
                    seed,
                    max_hops=max_hops,
                    scope_id=(scope or {}).get("scope_id"),
                )
                all_entities.update(entities)
                for fact in facts:
                    if not self._fact_matches_scope(fact, user_id=user_id, scope=scope):
                        continue
                    if fact.id not in seen_fact_ids and len(all_facts) < max_facts:
                        seen_fact_ids.add(fact.id)
                        all_facts.append(fact)

        # If using weighted strategy, sort by weight
        if strategy == RetrievalStrategy.WEIGHTED:
            all_facts.sort(
                key=lambda f: (
                    self.weights.get(
                        f.relation, RelationshipWeight(relation=f.relation, weight=1.0)
                    ).weight
                ),
                reverse=True,
            )
            all_facts = all_facts[:max_facts]

        # Ensure all fact endpoints are in entities dict
        for fact in all_facts:
            for eid in (fact.subject_id, fact.object_id):
                if eid is None:
                    continue
                if eid not in all_entities:
                    entity = self.memory_store.get_entity(eid)
                    if entity:
                        all_entities[eid] = entity

        return all_facts, all_entities

    # =========================================================================
    # Query-Aware Helpers
    # =========================================================================

    def _fact_local_relevance_score(
        self,
        fact: ValidatedFact,
        query_tokens: set[str],
        entities: dict[str, Entity],
    ) -> float:
        """Compute a quick token-overlap relevance score between a fact and the query."""
        if not query_tokens:
            return 1.0

        subject = entities.get(fact.subject_id)
        if subject is None:
            subject = self.memory_store.get_entity(fact.subject_id)
            if subject is not None:
                entities[fact.subject_id] = subject

        obj = None
        if fact.object_id:
            obj = entities.get(fact.object_id)
            if obj is None:
                obj = self.memory_store.get_entity(fact.object_id)
                if obj is not None:
                    entities[fact.object_id] = obj

        fact_text_parts = [
            fact.relation.value,
            subject.name if subject else fact.subject_id,
            obj.name if obj else "",
            fact.value or "",
        ]
        fact_tokens = _tokenize(" ".join(part for part in fact_text_parts if part))

        if not fact_tokens:
            return 0.0

        overlap = len(query_tokens & fact_tokens)
        return overlap / max(len(query_tokens), 1)

    # =========================================================================
    # Fallback: NetworkX/In-Memory Retrieval (temporal or when Neo4j unavailable)
    # =========================================================================

    def _retrieve_fallback(
        self,
        seed_entities: list[str],
        max_hops: int,
        max_facts: int,
        at_time: datetime,
        strategy: RetrievalStrategy,
        user_id: str | None = None,
        scope: dict[str, str] | None = None,
        query: str = "",
    ) -> tuple[list[ValidatedFact], dict[str, Entity]]:
        """Dispatch to the original NetworkX-based retrieval methods."""
        if strategy == RetrievalStrategy.BREADTH_FIRST:
            return self._retrieve_bfs(
                seed_entities,
                max_hops,
                max_facts,
                at_time,
                user_id=user_id,
                scope=scope,
                query=query,
            )
        elif strategy == RetrievalStrategy.WEIGHTED:
            return self._retrieve_weighted(
                seed_entities,
                max_hops,
                max_facts,
                at_time,
                user_id=user_id,
                scope=scope,
                query=query,
            )
        else:
            return self._retrieve_safety_priority(
                seed_entities,
                max_hops,
                max_facts,
                at_time,
                user_id=user_id,
                scope=scope,
                query=query,
            )

    # =========================================================================
    # Original NetworkX-Based Retrieval (used as fallback / for temporal queries)
    # =========================================================================

    def _retrieve_bfs(
        self,
        seed_entities: list[str],
        max_hops: int,
        max_facts: int,
        at_time: datetime,
        user_id: str | None = None,
        scope: dict[str, str] | None = None,
        query: str = "",
    ) -> tuple[list[ValidatedFact], dict[str, Entity]]:
        """Standard breadth-first retrieval with query-aware sorting and pruning."""
        visited_entities: set[str] = set()
        visited_facts: set[str] = set()
        facts: list[ValidatedFact] = []
        entities: dict[str, Entity] = {}

        current_level = set(seed_entities)
        query_tokens = _tokenize(query)

        for hop in range(max_hops):
            if len(facts) >= max_facts:
                break

            next_level: set[str] = set()

            for entity_id in current_level:
                if entity_id in visited_entities:
                    continue

                visited_entities.add(entity_id)

                # Get entity
                entity = self.memory_store.get_entity(entity_id)
                if entity:
                    entities[entity_id] = entity

                # Get facts for this entity
                entity_facts = self.memory_store.get_active_facts_for_entity(entity_id, at_time)

                # Score and sort by local query relevance for better early collection
                scored_facts: list[tuple[float, ValidatedFact]] = []
                for fact in entity_facts:
                    if not self._fact_matches_scope(fact, user_id=user_id, scope=scope):
                        continue
                    if fact.id in visited_facts:
                        continue
                    local_score = self._fact_local_relevance_score(fact, query_tokens, entities)
                    scored_facts.append((local_score, fact))

                scored_facts.sort(key=lambda item: item[0], reverse=True)

                for local_score, fact in scored_facts:
                    if len(facts) >= max_facts:
                        break

                    visited_facts.add(fact.id)
                    facts.append(fact)

                    # Query-aware pruning: only expand through relevant or safety-critical edges
                    # beyond the first hop. First hop always expands to ensure coverage.
                    relation_profile = self.get_weight(fact.relation)
                    should_expand = (
                        hop == 0 or local_score > 0 or relation_profile.is_safety_critical
                    )

                    if should_expand:
                        # Add connected entities to next level
                        if fact.subject_id != entity_id:
                            next_level.add(fact.subject_id)
                        if fact.object_id is not None and fact.object_id != entity_id:
                            next_level.add(fact.object_id)

            current_level = next_level - visited_entities

        return facts, entities

    def _retrieve_weighted(
        self,
        seed_entities: list[str],
        max_hops: int,
        max_facts: int,
        at_time: datetime,
        user_id: str | None = None,
        scope: dict[str, str] | None = None,
        query: str = "",
    ) -> tuple[list[ValidatedFact], dict[str, Entity]]:
        """Weighted retrieval based on relationship importance with query-aware tiebreaking."""
        # Build a weighted graph
        G = self._build_weighted_graph(at_time, user_id=user_id, scope=scope)

        # Find all facts reachable within max_hops
        reachable_facts: list[tuple[float, ValidatedFact]] = []
        visited_entities: set[str] = set()
        visited_facts: set[str] = set()
        entities: dict[str, Entity] = {}
        query_tokens = _tokenize(query)

        def explore(entity_id: str, hop: int, cumulative_weight: float):
            if hop > max_hops or entity_id in visited_entities:
                return

            visited_entities.add(entity_id)
            entity = self.memory_store.get_entity(entity_id)
            if entity:
                entities[entity_id] = entity

            # Get outgoing edges (facts)
            if entity_id in G:
                for neighbor, edge_data in G[entity_id].items():
                    for _edge_key, data in edge_data.items():
                        fact = data["fact"]
                        if fact.id in visited_facts:
                            continue
                        visited_facts.add(fact.id)
                        weight = data["weight"]
                        decay_per_hop = data["decay_per_hop"]

                        # Score current edge and decay only for recursive carry.
                        effective_weight = cumulative_weight * weight
                        reachable_facts.append((effective_weight, fact))

                        # Continue exploration
                        carry_multiplier = 1.0 - max(0.0, min(decay_per_hop, 1.0))
                        explore(neighbor, hop + 1, effective_weight * carry_multiplier)

        # Start from each seed entity
        for seed in seed_entities:
            explore(seed, 0, 1.0)

        # Sort by weight descending; use query relevance as tiebreaker
        def _sort_key(item: tuple[float, ValidatedFact]) -> tuple[float, float]:
            weight, fact = item
            local_score = self._fact_local_relevance_score(fact, query_tokens, entities)
            return (weight, local_score)

        reachable_facts.sort(key=_sort_key, reverse=True)
        facts = []
        seen_fact_ids: set[str] = set()

        for _, fact in reachable_facts:
            if fact.id not in seen_fact_ids and len(facts) < max_facts:
                seen_fact_ids.add(fact.id)
                facts.append(fact)

        return facts, entities

    def _retrieve_safety_priority(
        self,
        seed_entities: list[str],
        max_hops: int,
        max_facts: int,
        at_time: datetime,
        user_id: str | None = None,
        scope: dict[str, str] | None = None,
        query: str = "",
    ) -> tuple[list[ValidatedFact], dict[str, Entity]]:
        """
        Safety-priority retrieval for regulated or high-risk domains.

        Always retrieves safety-critical facts first (allergies,
        contraindications), then fills remaining context with
        other relevant facts. Applies query-aware sorting and pruning
        so that irrelevant non-safety facts at deeper hops are skipped.
        """
        entities: dict[str, Entity] = {}
        safety_facts: list[tuple[float, ValidatedFact]] = []
        other_facts: list[tuple[float, ValidatedFact]] = []
        seen_fact_ids: set[str] = set()

        # First pass: Get all safety-critical facts
        safety_relations = {
            rel for rel, weight in self.weights.items() if weight.is_safety_critical
        }

        visited_entities: set[str] = set()
        current_level = set(seed_entities)
        query_tokens = _tokenize(query)

        for hop in range(max_hops):
            next_level: set[str] = set()

            for entity_id in current_level:
                if entity_id in visited_entities:
                    continue

                visited_entities.add(entity_id)
                entity = self.memory_store.get_entity(entity_id)
                if entity:
                    entities[entity_id] = entity

                # Get facts for this entity
                entity_facts = self.memory_store.get_active_facts_for_entity(entity_id, at_time)

                for fact in entity_facts:
                    if not self._fact_matches_scope(fact, user_id=user_id, scope=scope):
                        continue
                    if fact.id in seen_fact_ids:
                        continue

                    seen_fact_ids.add(fact.id)
                    local_score = self._fact_local_relevance_score(fact, query_tokens, entities)

                    if fact.relation in safety_relations:
                        safety_facts.append((local_score, fact))
                    else:
                        # Query-aware pruning: skip irrelevant non-safety facts beyond first hop
                        if hop > 0 and local_score == 0:
                            continue
                        other_facts.append((local_score, fact))

                    # Query-aware expansion
                    relation_profile = self.get_weight(fact.relation)
                    should_expand = (
                        hop == 0 or local_score > 0 or relation_profile.is_safety_critical
                    )
                    if should_expand:
                        # Add connected entities
                        if fact.subject_id != entity_id:
                            next_level.add(fact.subject_id)
                        if fact.object_id is not None and fact.object_id != entity_id:
                            next_level.add(fact.object_id)

            current_level = next_level - visited_entities

        # Sort by local relevance within each group so most relevant safety info comes first
        safety_facts.sort(key=lambda item: item[0], reverse=True)
        other_facts.sort(key=lambda item: item[0], reverse=True)

        combined = [fact for _, fact in safety_facts] + [fact for _, fact in other_facts]
        return combined[:max_facts], entities

    def _build_weighted_graph(
        self,
        at_time: datetime,
        user_id: str | None = None,
        scope: dict[str, str] | None = None,
    ) -> nx.MultiDiGraph:
        """Build a NetworkX graph from the memory store."""
        G = nx.MultiDiGraph()

        # Add all entities as nodes
        for entity in _iter_entities(self.memory_store):
            G.add_node(entity.id, entity=entity)

        # Add facts as edges
        for fact in _iter_active_facts(self.memory_store, at_time):
            if not self._fact_matches_scope(fact, user_id=user_id, scope=scope):
                continue
            if fact.object_id is None:
                continue

            # Get weight for this relation
            weight_config = self.get_weight(fact.relation)

            G.add_edge(
                fact.subject_id,
                fact.object_id,
                key=fact.id,
                fact=fact,
                weight=weight_config.weight,
                is_safety_critical=weight_config.is_safety_critical,
                decay_per_hop=weight_config.decay_per_hop,
            )

        return G

    def _fact_matches_scope(
        self,
        fact: ValidatedFact,
        *,
        user_id: str | None = None,
        scope: dict[str, str] | None = None,
    ) -> bool:
        """Filter facts by provenance scope when requested.

        Scope boundaries are defined by tenant, app, user, and agent.
        ``run_id`` and ``session_id`` are ephemeral session identifiers
        and are intentionally NOT enforced as scope boundaries so that
        data written in one run remains queryable from another run
        within the same tenant/app/user scope.
        """
        expected = dict(scope or {})
        if user_id is not None and "user_id" not in expected:
            expected["user_id"] = user_id

        # Remove ephemeral session identifiers from scope checks.
        expected.pop("run_id", None)
        expected.pop("session_id", None)

        if not expected:
            return True

        interaction = None
        if hasattr(self.memory_store, "get_interaction"):
            interaction = self.memory_store.get_interaction(fact.source_interaction_id)  # type: ignore[attr-defined]

        if interaction is not None:
            metadata = getattr(interaction, "metadata", {}) or {}
            for key, value in expected.items():
                if value is None:
                    continue

                if key == "scope_id":
                    actual_scope_id = metadata.get("scope_id")
                    if actual_scope_id is None:
                        tenant = getattr(interaction, "tenant_id", None) or metadata.get(
                            "tenant_id"
                        )
                        app = getattr(interaction, "app_id", None) or metadata.get("app_id")
                        scoped_user = getattr(interaction, "user_id", None) or metadata.get(
                            "user_id"
                        )
                        actual_scope_id = (
                            f"{tenant}:{app}:{scoped_user}"
                            if tenant and app and scoped_user
                            else None
                        )
                    if actual_scope_id != value:
                        return False
                    continue

                actual = getattr(interaction, key, None) or metadata.get(key)
                if actual != value:
                    return False

            return True

        # Fallback: use fact attributes when interaction provenance is unavailable.
        attrs = fact.attributes or {}
        for key, value in expected.items():
            if value is None:
                continue
            if attrs.get(key) != value:
                return False
        return True

    def find_paths(
        self,
        source_entity: str,
        target_entity: str,
        max_length: int = 3,
        at_time: datetime | None = None,
    ) -> list[list[ValidatedFact]]:
        """
        Find all paths between two entities.

        Uses Neo4j when available (native shortestPath), otherwise
        falls back to NetworkX.
        """
        at_time = at_time or datetime.now(timezone.utc)

        # Use Neo4j for current-time path queries
        if self.has_neo4j and self._is_current_time(at_time):
            try:
                return self._neo4j_store.find_paths(source_entity, target_entity, max_length)
            except Exception as e:
                logger.warning("Neo4j find_paths failed, falling back: %s", e)

        # Fallback: NetworkX
        G = self._build_weighted_graph(at_time)

        paths = []
        try:
            for path in nx.all_simple_paths(G, source_entity, target_entity, cutoff=max_length):
                fact_path = []
                for i in range(len(path) - 1):
                    edge_data = G.get_edge_data(path[i], path[i + 1])
                    if edge_data:
                        first_key = list(edge_data.keys())[0]
                        fact_path.append(edge_data[first_key]["fact"])

                if fact_path:
                    paths.append(fact_path)
        except nx.NetworkXNoPath:
            pass

        return paths

    def get_entity_neighborhood(
        self,
        entity_id: str,
        radius: int = 1,
        at_time: datetime | None = None,
    ) -> tuple[dict[str, Entity], list[ValidatedFact]]:
        """
        Get the immediate neighborhood of an entity.

        Uses Neo4j's native get_neighbors when available.
        Returns all entities and facts within 'radius' hops.
        """
        at_time = at_time or datetime.now(timezone.utc)

        # Use Neo4j for current-time neighborhood queries
        if self.has_neo4j and self._is_current_time(at_time):
            try:
                return self._neo4j_store.get_neighbors(entity_id, max_hops=radius)
            except Exception as e:
                logger.warning("Neo4j neighborhood query failed, falling back: %s", e)

        # Fallback: in-memory traversal
        entity_ids = list(
            self.memory_store.get_connected_entities(
                entity_id, max_hops=radius, at_time=at_time
            ).keys()
        )

        return self.memory_store.get_subgraph(entity_ids, at_time)


# =============================================================================
# Query Understanding (Future: LLM-based)
# =============================================================================


def identify_seed_entities(
    query: str,
    memory_store: GraphMemoryStore,
) -> list[str]:
    """
    Identify seed entities from a natural language query.

    This is a simple implementation - in production would use
    LLM-based entity recognition.
    """
    return select_seed_entities(query, memory_store)
