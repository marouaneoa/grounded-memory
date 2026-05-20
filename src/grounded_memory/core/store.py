"""
Memory Store with Bitemporal Fact Management

This module implements the persistent storage layer for the Grounded Memory System.
It stores:
- Entities (nodes in the knowledge graph)
- ValidatedFacts (edges in the knowledge graph with valid-time boundaries)
- Interactions (immutable event log)
- RejectionRecords (audit trail of rejected facts)

Key features:
- Bitemporal semantics: valid-time fact boundaries plus record-time provenance
- Supersession versioning: Facts are superseded, not deleted
- Graph-based access: Efficient traversal of entity relationships
- Point-in-time queries: Answer questions about past states
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from grounded_memory.core.models import (
    Entity,
    EntityType,
    Interaction,
    RejectionRecord,
    RelationType,
    ValidatedFact,
)

# =============================================================================
# In-Memory Store Implementation
# =============================================================================


class MemoryStore:
    """
    In-memory implementation of the knowledge store.

    This store implements the KnowledgeState protocol required by
    the ConstraintValidator. It provides efficient access patterns for:
    - Entity lookup by ID or type
    - Fact retrieval by entity, relation, or time
    - Temporal queries (point-in-time state)

    For production, this would be backed by PostgreSQL with bitemporal semantics.
    """

    def __init__(self):
        # Primary storage
        self._entities: dict[str, Entity] = {}
        self._facts: dict[str, ValidatedFact] = {}
        self._interactions: dict[str, Interaction] = {}
        self._rejections: dict[str, RejectionRecord] = {}

        # Indices for efficient querying
        self._entities_by_type: dict[EntityType, set[str]] = defaultdict(set)
        self._facts_by_subject: dict[str, set[str]] = defaultdict(set)
        self._facts_by_object: dict[str, set[str]] = defaultdict(set)
        self._facts_by_relation: dict[RelationType, set[str]] = defaultdict(set)
        self._entity_name_index: dict[str, set[str]] = defaultdict(
            set
        )  # lowercase name -> entity IDs

    # =========================================================================
    # Entity Operations
    # =========================================================================

    def add_entity(self, entity: Entity) -> None:
        """Add or update an entity."""
        self._entities[entity.id] = entity
        self._entities_by_type[entity.entity_type].add(entity.id)
        self._entity_name_index[entity.name.lower()].add(entity.id)

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        return self._entities.get(entity_id)

    def get_entities_by_type(self, entity_type: EntityType) -> list[Entity]:
        """Get all entities of a specific type."""
        return [
            self._entities[eid]
            for eid in self._entities_by_type[entity_type]
            if eid in self._entities
        ]

    def find_entity_by_name(
        self,
        name: str,
        entity_type: EntityType | None = None,
    ) -> Entity | None:
        """Find an entity by name (case-insensitive)."""
        entity_ids = self._entity_name_index.get(name.lower(), set())

        for eid in entity_ids:
            entity = self._entities.get(eid)
            if entity and (entity_type is None or entity.entity_type == entity_type):
                return entity

        return None

    def find_or_create_entity(
        self,
        name: str,
        entity_type: EntityType,
        create_fn: callable,
        uniqueness_key: str | None = None,
    ) -> tuple[Entity, bool]:
        """
        Find an existing entity or create a new one.

        This ensures uniqueness for entities based on name (and optionally
        additional attributes via uniqueness_key).

        Args:
            name: Entity name to search for
            entity_type: Type of entity
            create_fn: Callable that creates the entity if not found
            uniqueness_key: Optional additional key for uniqueness
                           (e.g., "name|dosage" for medications)

        Returns:
            Tuple of (entity, created) where created is True if newly created
        """
        scope_prefix = None
        if uniqueness_key and "|type:" in uniqueness_key:
            scope_prefix = uniqueness_key.split("|type:")[0]

        # First try to find by exact name match across ALL types
        existing_ids = self._entity_name_index.get(name.lower(), set())

        for eid in existing_ids:
            entity = self._entities.get(eid)
            if entity:
                # If we have a scope prefix, check if this entity is in the same scope
                if scope_prefix:
                    existing_key = entity.attributes.get("uniqueness_key", "")
                    if existing_key.startswith(scope_prefix + "|"):
                        # Found a cross-type match in the exact same scope!
                        # Upgrade type if current is FACILITY and new is not
                        if (
                            entity.entity_type == EntityType.FACILITY
                            and entity_type != EntityType.FACILITY
                        ):
                            self._entities_by_type[entity.entity_type].remove(entity.id)
                            entity.entity_type = entity_type
                            self._entities_by_type[entity_type].add(entity.id)

                            # Also update the type in the uniqueness key if it exists
                            if existing_key and "|type:facility|" in existing_key:
                                entity.attributes["uniqueness_key"] = existing_key.replace(
                                    "|type:facility|", f"|type:{entity_type.value}|"
                                )

                        return entity, False
                else:
                    # If no uniqueness key provided, just return the first name match
                    # Optionally upgrade type here too
                    if (
                        entity.entity_type == EntityType.FACILITY
                        and entity_type != EntityType.FACILITY
                    ):
                        self._entities_by_type[entity.entity_type].remove(entity.id)
                        entity.entity_type = entity_type
                        self._entities_by_type[entity_type].add(entity.id)
                    return entity, False

        # Not found, create new entity
        new_entity = create_fn()

        # Store the uniqueness key if provided
        if uniqueness_key:
            new_entity.attributes["uniqueness_key"] = uniqueness_key

        self.add_entity(new_entity)
        return new_entity, True

    def search_entities(
        self,
        query: str,
        entity_type: EntityType | None = None,
        limit: int = 10,
    ) -> list[Entity]:
        """Search entities by name prefix."""
        query_lower = query.lower()
        results = []

        for name, entity_ids in self._entity_name_index.items():
            if name.startswith(query_lower):
                for eid in entity_ids:
                    entity = self._entities.get(eid)
                    if entity and (entity_type is None or entity.entity_type == entity_type):
                        results.append(entity)
                        if len(results) >= limit:
                            return results

        return results

    def iter_entities(self) -> list[Entity]:
        """Return all entities for backend-agnostic graph traversal."""
        return list(self._entities.values())

    def find_entity_ids_by_name_fragment(self, text: str) -> list[str]:
        """Find entity IDs whose normalized name is contained in the provided text."""
        if not text:
            return []

        text_lower = text.lower()
        matched: set[str] = set()
        for name, entity_ids in self._entity_name_index.items():
            if name in text_lower:
                matched.update(entity_ids)
        return list(matched)

    # =========================================================================
    # Fact Operations
    # =========================================================================

    def add_validated_fact(self, fact: ValidatedFact) -> None:
        """Add a validated fact to the store."""
        self._facts[fact.id] = fact
        self._facts_by_subject[fact.subject_id].add(fact.id)
        if fact.object_id is not None:
            self._facts_by_object[fact.object_id].add(fact.id)
        self._facts_by_relation[fact.relation].add(fact.id)

    def get_fact(self, fact_id: str) -> ValidatedFact | None:
        """Get a fact by ID."""
        return self._facts.get(fact_id)

    def get_active_facts_for_entity(
        self,
        entity_id: str,
        at_time: datetime | None = None,
    ) -> list[ValidatedFact]:
        """
        Get all active facts where entity is subject or object.

        Args:
            entity_id: The entity to query
            at_time: Point in time (None = current time)
        """
        at_time = at_time or datetime.now(timezone.utc)

        # Get fact IDs where entity is subject or object
        fact_ids = self._facts_by_subject.get(entity_id, set()) | self._facts_by_object.get(
            entity_id, set()
        )

        # Filter to active facts at the given time
        return [
            self._facts[fid]
            for fid in fact_ids
            if fid in self._facts and self._facts[fid].is_active_at(at_time)
        ]

    def get_facts_by_relation(
        self,
        entity_id: str,
        relation: RelationType,
        as_subject: bool = True,
        at_time: datetime | None = None,
    ) -> list[ValidatedFact]:
        """
        Get facts with a specific relation for an entity.

        Args:
            entity_id: The entity to query
            relation: Type of relation to filter by
            as_subject: If True, entity must be subject; if False, entity must be object
            at_time: Point in time (None = current time)
        """
        at_time = at_time or datetime.now(timezone.utc)

        # Get facts by entity role
        if as_subject:
            fact_ids = self._facts_by_subject.get(entity_id, set())
        else:
            fact_ids = self._facts_by_object.get(entity_id, set())

        # Filter by relation and time
        result = []
        for fid in fact_ids:
            fact = self._facts.get(fid)
            if fact and fact.relation == relation and fact.is_active_at(at_time):
                result.append(fact)

        return result

    def get_all_facts_by_relation(
        self,
        relation: RelationType,
        at_time: datetime | None = None,
    ) -> list[ValidatedFact]:
        """Get all facts of a specific relation type."""
        at_time = at_time or datetime.now(timezone.utc)

        return [
            self._facts[fid]
            for fid in self._facts_by_relation.get(relation, set())
            if fid in self._facts and self._facts[fid].is_active_at(at_time)
        ]

    def get_all_validated_facts(self) -> list[ValidatedFact]:
        """Get all validated facts (both active and superseded)."""
        return list(self._facts.values())

    def get_validated_fact(self, fact_id: str) -> ValidatedFact | None:
        """Get a single validated fact by its ID."""
        return self._facts.get(fact_id)

    def iter_active_facts(self, at_time: datetime | None = None) -> list[ValidatedFact]:
        """Return active facts at a point in time for backend-agnostic retrieval."""
        at_time = at_time or datetime.now(timezone.utc)
        return [fact for fact in self._facts.values() if fact.is_active_at(at_time)]

    def get_all_entities(self) -> list[Entity]:
        """Get all entities in the store."""
        return list(self._entities.values())

    def get_facts_for_entity(
        self,
        entity_id: str,
        include_superseded: bool = False,
    ) -> list[ValidatedFact]:
        """
        Get all facts where entity is subject or object.

        Args:
            entity_id: The entity to query
            include_superseded: If True, include superseded facts
        """
        fact_ids = self._facts_by_subject.get(entity_id, set()) | self._facts_by_object.get(
            entity_id, set()
        )

        facts = [self._facts[fid] for fid in fact_ids if fid in self._facts]

        if not include_superseded:
            facts = [f for f in facts if f.is_active]

        # TODO: Consider adding an index for superseded_by and a fast flag
        # to query superseded facts efficiently. Current approach iterates
        # over candidate IDs and filters in memory which may be slow for
        # large datasets. Also consider returning facts ordered by
        # `valid_from`/`valid_to` for clearer temporal semantics.

        return facts

    # =========================================================================
    # Interaction Operations
    # =========================================================================

    def add_interaction(self, interaction: Interaction) -> None:
        """Add an interaction to the log."""
        self._interactions[interaction.id] = interaction

    def get_interaction(self, interaction_id: str) -> Interaction | None:
        """Get an interaction by ID."""
        return self._interactions.get(interaction_id)

    def get_interactions(
        self,
        limit: int = 100,
        before: datetime | None = None,
    ) -> list[Interaction]:
        """Get recent interactions."""
        interactions = list(self._interactions.values())

        if before:
            interactions = [i for i in interactions if i.timestamp < before]

        # Sort by timestamp descending
        interactions.sort(key=lambda i: i.timestamp, reverse=True)

        return interactions[:limit]

    # =========================================================================
    # Rejection Operations
    # =========================================================================

    def add_rejection(self, rejection: RejectionRecord) -> None:
        """Add a rejection record."""
        self._rejections[rejection.id] = rejection

    def get_rejection(self, rejection_id: str) -> RejectionRecord | None:
        """Get a rejection record by ID."""
        return self._rejections.get(rejection_id)

    def get_all_rejections(self) -> list[RejectionRecord]:
        """Get all rejection records sorted newest first."""
        rejections = list(self._rejections.values())
        rejections.sort(key=lambda r: r.rejected_at, reverse=True)
        return rejections

    def get_rejections_for_constraint(
        self,
        constraint_id: str,
        limit: int = 100,
    ) -> list[RejectionRecord]:
        """Get rejections caused by a specific constraint."""
        rejections = [r for r in self._rejections.values() if r.constraint_id == constraint_id]
        rejections.sort(key=lambda r: r.rejected_at, reverse=True)
        return rejections[:limit]

    # =========================================================================
    # Graph Operations (for retrieval)
    # =========================================================================

    def get_connected_entities(
        self,
        entity_id: str,
        max_hops: int = 2,
        at_time: datetime | None = None,
    ) -> dict[str, Entity]:
        """
        Get all entities connected to a seed entity within N hops.

        This is the foundation for graph-based retrieval.
        """
        at_time = at_time or datetime.now(timezone.utc)

        visited: set[str] = set()
        to_visit = {entity_id}
        result: dict[str, Entity] = {}

        for _ in range(max_hops):
            if not to_visit:
                break

            next_level: set[str] = set()

            for eid in to_visit:
                if eid in visited:
                    continue

                visited.add(eid)
                entity = self._entities.get(eid)
                if entity:
                    result[eid] = entity

                # Get connected entities through facts
                facts = self.get_active_facts_for_entity(eid, at_time)
                for fact in facts:
                    if fact.subject_id == eid:
                        if fact.object_id is not None:
                            next_level.add(fact.object_id)
                    else:
                        next_level.add(fact.subject_id)

            to_visit = next_level - visited

        return result

    def get_subgraph(
        self,
        entity_ids: list[str],
        at_time: datetime | None = None,
    ) -> tuple[dict[str, Entity], list[ValidatedFact]]:
        """
        Get a subgraph containing specified entities and facts between them.

        Returns:
            (entities dict, list of facts)
        """
        at_time = at_time or datetime.now(timezone.utc)
        entity_set = set(entity_ids)

        entities = {eid: self._entities[eid] for eid in entity_ids if eid in self._entities}

        facts = []
        for eid in entity_ids:
            for fact in self.get_active_facts_for_entity(eid, at_time):
                # Include fact if both endpoints are in our entity set
                if (
                    fact.object_id is not None
                    and fact.subject_id in entity_set
                    and fact.object_id in entity_set
                ) and fact not in facts:
                    facts.append(fact)

        return entities, facts

    def supersede_fact(
        self,
        fact_id: str,
        superseded_by: str,
        valid_to: datetime | None = None,
    ) -> bool:
        """Mark a fact as superseded.

        Returns True if the fact existed and was updated, else False.
        """
        fact = self._facts.get(fact_id)
        if not fact:
            return False

        # TODO: Ensure indices remain consistent when superseding facts.
        # - If the store maintains secondary indices (by relation/subject/object),
        #   they may need updates when facts change state. Consider tracking
        #   an explicit 'is_active' flag or maintaining a tombstone index.
        # - Also consider emitting an event or audit log when supersession
        #   occurs to aid debugging and traceability.
        updated = fact.model_copy(
            update={
                "superseded_by": superseded_by,
                "valid_to": valid_to or datetime.now(timezone.utc),
            }
        )
        self._facts[fact_id] = updated
        return True

    # =========================================================================
    # Utilities
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Get store statistics."""
        active_facts = sum(1 for f in self._facts.values() if f.is_active)

        return {
            "total_entities": len(self._entities),
            "entities_by_type": {t.value: len(ids) for t, ids in self._entities_by_type.items()},
            "total_facts": len(self._facts),
            "active_facts": active_facts,
            "superseded_facts": len(self._facts) - active_facts,
            "total_interactions": len(self._interactions),
            "total_rejections": len(self._rejections),
        }

    def clear(self) -> None:
        """Clear all data from the store."""
        self._entities.clear()
        self._facts.clear()
        self._interactions.clear()
        self._rejections.clear()
        self._entities_by_type.clear()
        self._facts_by_subject.clear()
        self._facts_by_object.clear()
        self._facts_by_relation.clear()
        self._entity_name_index.clear()
