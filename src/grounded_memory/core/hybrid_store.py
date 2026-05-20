"""
Hybrid Memory Store (PostgreSQL + Neo4j)

This module bridges the in-memory/PostgreSQL store (source of truth) with
the Neo4j active graph store. It implements the same interface as MemoryStore
so it's a drop-in replacement, while transparently syncing to Neo4j.

Architecture:
    ┌─────────────────────────────────────────────┐
    │              HybridMemoryStore              │
    │                                             │
    │  ┌─────────────┐     ┌──────────────────┐   │
    │  │ MemoryStore  │     │   Neo4jStore     │   │
    │  │ (source of   │────▶│  (active graph   │   │
    │  │  truth)      │sync │   projection)    │   │
    │  └─────────────┘     └──────────────────┘   │
    │                                             │
    │  Write: MemoryStore first, then sync Neo4j  │
    │  Read:  Neo4j for graph queries             │
    │         MemoryStore for temporal/audit       │
    └─────────────────────────────────────────────┘

All mutations go through MemoryStore first. On success, the change is
replicated to Neo4j. If Neo4j sync fails, data is still safe in the
primary store — Neo4j can be rebuilt from MemoryStore at any time.

For the PostgreSQL variant, swap MemoryStore with PostgresStore and
make the sync calls async.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from grounded_memory.core.models import (
    Entity,
    ValidatedFact,
)
from grounded_memory.core.store import MemoryStore

logger = logging.getLogger(__name__)

# Neo4j dependency
try:
    from grounded_memory.core.neo4j_store import Neo4jConfig, Neo4jStore

    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

if TYPE_CHECKING:
    from grounded_memory.core.neo4j_store import Neo4jConfig, Neo4jStore


class HybridMemoryStore(MemoryStore):
    """
    Hybrid store: MemoryStore (source of truth) + Neo4j (active graph).

    Inherits from MemoryStore so all existing code works unchanged.
    Transparently syncs entities and active facts to Neo4j.

    Neo4j is mandatory for this hybrid store; initialization and sync failures
    are treated as runtime errors.

    Usage:
        # With Neo4j
        store = HybridMemoryStore(neo4j_config=Neo4jConfig.from_env())
        store.initialize_neo4j()

        # Same API as MemoryStore
        store.add_entity(entity)
        store.add_validated_fact(fact)
    """

    def __init__(
        self,
        neo4j_config: Neo4jConfig | None = None,
        neo4j_store: Neo4jStore | None = None,
        sync_enabled: bool = True,
    ):
        """
        Initialize the hybrid store.

        Args:
            neo4j_config: Neo4j config (creates store automatically)
            neo4j_store: Pre-initialized Neo4j store (takes precedence over config)
            sync_enabled: Whether to sync to Neo4j (can be disabled for testing)
        """
        super().__init__()

        self._neo4j_store: Neo4jStore | None = neo4j_store
        self._neo4j_config = neo4j_config
        if sync_enabled and not HAS_NEO4J:
            raise RuntimeError("HybridMemoryStore requires Neo4j dependencies when sync is enabled")
        self._sync_enabled = sync_enabled
        self._neo4j_initialized = neo4j_store is not None

    def initialize_neo4j(self, create_schema: bool = True) -> bool:
        """
        Initialize the Neo4j connection.

        Returns True if successful. Raises on failure.
        """
        if not HAS_NEO4J:
            raise RuntimeError("Neo4j driver not installed")

        if self._neo4j_initialized:
            return True

        try:
            config = self._neo4j_config or Neo4jConfig.from_env()
            self._neo4j_store = Neo4jStore(config)
            self._neo4j_store.initialize(create_schema=create_schema)
            self._neo4j_initialized = True
            logger.info("Neo4j initialized for hybrid store")
            return True
        except Exception as e:
            self._neo4j_store = None
            self._neo4j_initialized = False
            raise RuntimeError(f"Failed to initialize Neo4j: {e}") from e

    @property
    def neo4j(self) -> Neo4jStore | None:
        """Access the underlying Neo4j store (may be None)."""
        return self._neo4j_store

    @property
    def has_neo4j(self) -> bool:
        """Check if Neo4j is available and initialized."""
        return self._neo4j_initialized and self._neo4j_store is not None

    def _sync_to_neo4j(self, operation: str, func, *args, **kwargs) -> None:
        """
        Safely sync an operation to Neo4j.

        Neo4j sync must succeed for hybrid-store consistency.
        """
        if not self._sync_enabled or not self.has_neo4j:
            return

        try:
            func(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Neo4j sync failed for {operation}: {e}") from e

    # =========================================================================
    # Entity Operations (override MemoryStore)
    # =========================================================================

    def add_entity(self, entity: Entity) -> None:
        """Add entity to primary store and sync to Neo4j."""
        super().add_entity(entity)
        if self.has_neo4j:
            self._sync_to_neo4j(
                "add_entity",
                self._neo4j_store.upsert_entity,
                entity,
            )

    # get_entity, get_entities_by_type, find_entity_by_name, find_or_create_entity
    # all inherited from MemoryStore — they read from the primary store

    # =========================================================================
    # Fact Operations (override MemoryStore)
    # =========================================================================

    def add_validated_fact(self, fact: ValidatedFact) -> None:
        """Add fact to primary store and sync active facts to Neo4j."""
        super().add_validated_fact(fact)

        # Only sync active facts to Neo4j
        if fact.is_active and self.has_neo4j and fact.object_id is not None:
            self._sync_to_neo4j(
                "add_fact",
                self._neo4j_store.add_fact,
                fact,
            )

    def supersede_fact(
        self,
        fact_id: str,
        superseded_by: str,
        valid_to: datetime | None = None,
    ) -> None:
        """Supersede fact in primary store and remove from Neo4j."""
        # Call parent (MemoryStore has two supersede_fact definitions;
        # the second one is the actual implementation)
        fact = self._facts.get(fact_id)
        if fact:
            fact.superseded_by = superseded_by
            fact.valid_to = valid_to or datetime.now(timezone.utc)
            self._facts[fact_id] = fact

            # Remove the now-superseded relationship from Neo4j
            if self.has_neo4j:
                self._sync_to_neo4j(
                    "remove_fact (supersede)",
                    self._neo4j_store.remove_fact,
                    fact_id,
                )

    # =========================================================================
    # Graph Operations (delegate to Neo4j when available)
    # =========================================================================

    def get_connected_entities(
        self,
        entity_id: str,
        max_hops: int = 2,
        at_time: datetime | None = None,
    ) -> dict[str, Entity]:
        """
        Get connected entities.

        For current-time queries, Neo4j is faster (native graph traversal).
        For point-in-time queries, use MemoryStore.
        """
        # Point-in-time queries always go to primary store
        if at_time is not None:
            return super().get_connected_entities(entity_id, max_hops, at_time)

        # Current-time: use Neo4j if available
        if self.has_neo4j:
            try:
                entities, _ = self._neo4j_store.get_neighbors(entity_id, max_hops=max_hops)
                return entities
            except Exception as e:
                raise RuntimeError(f"Neo4j query failed for get_connected_entities: {e}") from e

        return super().get_connected_entities(entity_id, max_hops, at_time)

    def get_subgraph(
        self,
        entity_ids: list[str],
        at_time: datetime | None = None,
    ) -> tuple[dict[str, Entity], list[ValidatedFact]]:
        """
        Get subgraph — uses Neo4j for current state, MemoryStore for temporal.
        """
        # Point-in-time: primary store
        if at_time is not None:
            return super().get_subgraph(entity_ids, at_time)

        # For current state with Neo4j, gather neighbors for each entity
        if self.has_neo4j:
            try:
                all_entities: dict[str, Entity] = {}
                all_facts: list[ValidatedFact] = []
                seen_fact_ids: set[str] = set()

                for eid in entity_ids:
                    entities, facts = self._neo4j_store.get_neighbors(eid, max_hops=1)
                    all_entities.update(entities)
                    for f in facts:
                        if f.id not in seen_fact_ids and (
                            f.subject_id in entity_ids or f.object_id in entity_ids
                        ):
                            seen_fact_ids.add(f.id)
                            all_facts.append(f)

                return all_entities, all_facts
            except Exception as e:
                raise RuntimeError(f"Neo4j query failed for get_subgraph: {e}") from e

        return super().get_subgraph(entity_ids, at_time)

    # =========================================================================
    # Bulk Sync
    # =========================================================================

    def rebuild_neo4j(self) -> dict[str, int]:
        """
        Rebuild the Neo4j graph from the primary MemoryStore.

        Use this to recover Neo4j after failures, or to initially populate
        it from an existing MemoryStore with data.

        Returns:
            Dict with entity_count and fact_count synced
        """
        if not self.has_neo4j:
            raise RuntimeError("Neo4j is not initialized")

        # Clear Neo4j
        self._neo4j_store.clear()

        # Sync all entities
        entities = self.get_all_entities()
        entity_count = self._neo4j_store.sync_all_entities(entities)

        # Sync only active facts
        all_facts = self.get_all_validated_facts()
        active_facts = [f for f in all_facts if f.is_active]
        fact_count = self._neo4j_store.sync_all_active_facts(active_facts)

        logger.info(
            "Neo4j rebuilt: %d entities, %d active facts",
            entity_count,
            fact_count,
        )

        return {"entity_count": entity_count, "fact_count": fact_count}

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def close(self) -> None:
        """Close Neo4j connection."""
        if self._neo4j_store:
            self._neo4j_store.close()
            self._neo4j_store = None
            self._neo4j_initialized = False

    def clear(self) -> None:
        """Clear all data from both stores."""
        super().clear()
        if self.has_neo4j:
            self._neo4j_store.clear()

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics from both stores."""
        stats = super().get_statistics()

        if self.has_neo4j:
            neo4j_stats = self._neo4j_store.get_statistics()
            stats["neo4j"] = neo4j_stats
            stats["neo4j_available"] = True
        else:
            stats["neo4j_available"] = False

        return stats
