"""
Neo4j Graph Store for Active Knowledge

This module implements the Neo4j graph store for the Grounded Memory System's
hybrid persistence architecture. Neo4j holds ONLY the active knowledge graph
(current state), while PostgreSQL remains the bitemporal source of truth.

Design rationale:
- Neo4j excels at multi-hop graph traversal, path-finding, and neighborhood queries
- PostgreSQL excels at bitemporal versioning (valid time + record time), ACID
    transactions, and audit trails
- By keeping only active facts in Neo4j, we avoid temporal property filtering
  on every traversal, preserving Neo4j's native performance advantage

Data flow:
    Write path:  Grounding Operator → PostgreSQL (source of truth) → sync to Neo4j
    Read path:   Graph Retriever → Neo4j (active graph) → AnswerContext
    Temporal:    Point-in-time queries → PostgreSQL (full history)

Node labels match EntityType enum values (capitalized).

Relationship types match RelationType enum values and are domain-configurable.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

try:
    from neo4j import GraphDatabase

    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False
    GraphDatabase = None

if TYPE_CHECKING:
    from neo4j import Driver, ManagedTransaction

from grounded_memory.core.models import (
    Entity,
    EntityType,
    RelationType,
    ValidatedFact,
)
from grounded_memory.core.tuple_normalization import resolve_attribute_key, sanitize_fact_value

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class Neo4jConfig:
    """Configuration for Neo4j connection."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        max_connection_pool_size: int = 50,
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.max_connection_pool_size = max_connection_pool_size

    @classmethod
    def from_env(cls) -> Neo4jConfig:
        """Create config from environment variables."""
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
            max_connection_pool_size=int(os.getenv("NEO4J_MAX_POOL", "50")),
        )


# =============================================================================
# Schema Setup (Constraints & Indexes)
# =============================================================================

# Cypher statements to initialize the graph schema
SCHEMA_CYPHER = [
    # Unique constraint on entity ID for each label
    "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    # Composite uniqueness for scoped entities (guardrail for multi-tenant projections)
    "CREATE CONSTRAINT entity_scope_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE (e.scope_id, e.id) IS UNIQUE",
    # Index on entity name for fast lookups
    "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
    # Index on entity type
    "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
    # Scope index for fast tenant/app/user filtering
    "CREATE INDEX entity_scope_idx IF NOT EXISTS FOR (e:Entity) ON (e.scope_id)",
    # Relationship scope index for path filtering
    "CREATE INDEX rel_scope_idx IF NOT EXISTS FOR ()-[r]-() ON (r.scope_id)",
    # Full-text index for entity name search
    """
    CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS 
    FOR (e:Entity) ON EACH [e.name]
    """,
]


# =============================================================================
# Neo4j Store Implementation
# =============================================================================


class Neo4jStore:
    """
    Neo4j store for the active knowledge graph.

    This store maintains a projection of the current (active) state of the
    knowledge graph. When facts are superseded in PostgreSQL, the corresponding
    relationships are removed from Neo4j.

    Key design decisions:
    - Entities are nodes with label :Entity AND their type label (e.g., :Entity:Patient)
    - ValidatedFacts become relationships with relationship type matching RelationType
    - Only active facts (valid_to IS NULL) exist as relationships
    - Fact metadata (id, confidence, valid_from, attributes) stored as relationship properties
    - Temporal queries go to PostgreSQL, not Neo4j

    Usage:
        config = Neo4jConfig.from_env()
        store = Neo4jStore(config)
        store.initialize()

        # Sync an entity
        store.upsert_entity(entity)

        # Sync an active fact
        store.add_fact(validated_fact)

        # Remove superseded fact
        store.remove_fact(fact_id)

        # Graph queries
        neighbors = store.get_neighbors(entity_id, max_hops=2)

        store.close()
    """

    def __init__(self, config: Neo4jConfig | None = None):
        if not HAS_NEO4J:
            raise ImportError(
                "neo4j driver is required for Neo4j support. Install it with: pip install neo4j"
            )

        self.config = config or Neo4jConfig.from_env()
        self._driver: Driver | None = None

    def initialize(self, create_schema: bool = True) -> None:
        """
        Initialize the Neo4j driver and optionally create schema constraints.

        Args:
            create_schema: If True, create indexes and constraints
        """
        self._driver = GraphDatabase.driver(
            self.config.uri,
            auth=(self.config.user, self.config.password),
            max_connection_pool_size=self.config.max_connection_pool_size,
        )
        # Verify connectivity
        self._driver.verify_connectivity()
        logger.info("Neo4j connection established: %s", self.config.uri)

        if create_schema:
            self._create_schema()

    def _create_schema(self) -> None:
        """Create Neo4j schema constraints and indexes."""
        with self._driver.session(database=self.config.database) as session:
            for cypher in SCHEMA_CYPHER:
                try:
                    session.run(cypher)
                except Exception as e:
                    # Schema creation is idempotent; log but don't fail
                    logger.debug("Schema statement skipped (may already exist): %s", e)

    def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")

    # =========================================================================
    # Entity Operations
    # =========================================================================

    def upsert_entity(self, entity: Entity) -> None:
        """
        Create or update an entity node in Neo4j.

        The node gets both :Entity and the type-specific label (e.g., :Patient).
        """
        type_label = _entity_type_to_label(entity.entity_type)
        scope = _extract_scope_from_attributes(entity.attributes)

        with self._driver.session(database=self.config.database) as session:
            session.execute_write(
                self._upsert_entity_tx,
                entity_id=entity.id,
                type_label=type_label,
                name=entity.name,
                entity_type=entity.entity_type.value,
                canonical_id=entity.canonical_id,
                attributes=entity.attributes,
                scope_id=scope.get("scope_id"),
                tenant_id=scope.get("tenant_id"),
                app_id=scope.get("app_id"),
                user_id=scope.get("user_id"),
                agent_id=scope.get("agent_id"),
                run_id=scope.get("run_id"),
                space_type=scope.get("space_type"),
                created_at=entity.created_at.isoformat(),
                updated_at=entity.updated_at.isoformat(),
            )

    @staticmethod
    def _upsert_entity_tx(
        tx: ManagedTransaction,
        entity_id: str,
        type_label: str,
        name: str,
        entity_type: str,
        canonical_id: str | None,
        attributes: dict,
        scope_id: str | None,
        tenant_id: str | None,
        app_id: str | None,
        user_id: str | None,
        agent_id: str | None,
        run_id: str | None,
        space_type: str | None,
        created_at: str,
        updated_at: str,
    ) -> None:
        # MERGE on id, then SET all properties and add the type label
        tx.run(
            f"""
            MERGE (e:Entity {{id: $id}})
            SET e.name = $name,
                e.entity_type = $entity_type,
                e.canonical_id = $canonical_id,
                e.attributes = $attributes_json,
                e.scope_id = $scope_id,
                e.tenant_id = $tenant_id,
                e.app_id = $app_id,
                e.user_id = $user_id,
                e.agent_id = $agent_id,
                e.run_id = $run_id,
                e.space_type = $space_type,
                e.created_at = $created_at,
                e.updated_at = $updated_at,
                e:{type_label}
            """,
            id=entity_id,
            name=name,
            entity_type=entity_type,
            canonical_id=canonical_id,
            attributes_json=_serialize_attributes(attributes),
            scope_id=scope_id,
            tenant_id=tenant_id,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            space_type=space_type,
            created_at=created_at,
            updated_at=updated_at,
        )

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity node by ID."""
        with self._driver.session(database=self.config.database) as session:
            result = session.execute_read(self._get_entity_tx, entity_id=entity_id)
            return result

    @staticmethod
    def _get_entity_tx(tx: ManagedTransaction, entity_id: str) -> Entity | None:
        result = tx.run(
            "MATCH (e:Entity {id: $id}) RETURN e",
            id=entity_id,
        )
        record = result.single()
        if record is None:
            return None
        return _node_to_entity(record["e"])

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity node and all its relationships."""
        with self._driver.session(database=self.config.database) as session:
            result = session.execute_write(self._delete_entity_tx, entity_id=entity_id)
            return result

    @staticmethod
    def _delete_entity_tx(tx: ManagedTransaction, entity_id: str) -> bool:
        result = tx.run(
            "MATCH (e:Entity {id: $id}) DETACH DELETE e RETURN count(e) as cnt",
            id=entity_id,
        )
        record = result.single()
        return record["cnt"] > 0 if record else False

    # =========================================================================
    # Fact (Relationship) Operations
    # =========================================================================

    def add_fact(self, fact: ValidatedFact) -> None:
        """
        Add an active fact as a relationship in Neo4j.

        Only call this for facts that are currently active (valid_to IS NULL).
        The relationship type matches the fact's RelationType.
        """
        if fact.object_id is None:
            raise ValueError("Neo4j relationship projection requires fact.object_id")

        rel_type = fact.relation.value  # e.g., "PRESCRIBED", "HAS_ALLERGY"
        scope = _extract_scope_from_attributes(fact.attributes)
        attribute_key = resolve_attribute_key(fact.value, fact.attributes)

        with self._driver.session(database=self.config.database) as session:
            created = session.execute_write(
                self._add_fact_tx,
                subject_id=fact.subject_id,
                object_id=fact.object_id,
                rel_type=rel_type,
                fact_id=fact.id,
                value=sanitize_fact_value(fact.value),
                attribute_key=attribute_key,
                confidence=fact.confidence,
                valid_from=fact.valid_from.isoformat(),
                candidate_fact_id=fact.candidate_fact_id,
                source_interaction_id=fact.source_interaction_id,
                attributes=fact.attributes,
                scope_id=scope.get("scope_id"),
                tenant_id=scope.get("tenant_id"),
                app_id=scope.get("app_id"),
                user_id=scope.get("user_id"),
                agent_id=scope.get("agent_id"),
                run_id=scope.get("run_id"),
                space_type=scope.get("space_type"),
            )
            if created == 0:
                raise RuntimeError(
                    "Neo4j relationship projection matched no endpoint nodes "
                    f"for fact {fact.id} ({fact.subject_id} -> {fact.object_id})"
                )

    @staticmethod
    def _add_fact_tx(
        tx: ManagedTransaction,
        subject_id: str,
        object_id: str,
        rel_type: str,
        fact_id: str,
        value: str | None,
        attribute_key: str | None,
        confidence: float,
        valid_from: str,
        candidate_fact_id: str,
        source_interaction_id: str,
        attributes: dict,
        scope_id: str | None,
        tenant_id: str | None,
        app_id: str | None,
        user_id: str | None,
        agent_id: str | None,
        run_id: str | None,
        space_type: str | None,
    ) -> int:
        # Create relationship between existing entity nodes
        # Using APOC or dynamic relationship types via string formatting
        # (safe here because rel_type comes from our RelationType enum)
        result = tx.run(
            f"""
            MATCH (s:Entity {{id: $subject_id}})
            MATCH (o:Entity {{id: $object_id}})
            WHERE $scope_id IS NULL OR coalesce(s.scope_id, $scope_id) = $scope_id
            CREATE (s)-[r:{rel_type} {{
                fact_id: $fact_id,
                subject_id: $subject_id,
                object_id: $object_id,
                value: $value,
                attribute_key: $attribute_key,
                confidence: $confidence,
                valid_from: $valid_from,
                candidate_fact_id: $candidate_fact_id,
                source_interaction_id: $source_interaction_id,
                scope_id: $scope_id,
                tenant_id: $tenant_id,
                app_id: $app_id,
                user_id: $user_id,
                agent_id: $agent_id,
                run_id: $run_id,
                space_type: $space_type,
                attributes: $attributes_json
            }}]->(o)
            RETURN count(r) AS relationships_created
            """,
            subject_id=subject_id,
            object_id=object_id,
            fact_id=fact_id,
            value=value,
            attribute_key=attribute_key,
            confidence=confidence,
            valid_from=valid_from,
            candidate_fact_id=candidate_fact_id,
            source_interaction_id=source_interaction_id,
            scope_id=scope_id,
            tenant_id=tenant_id,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            space_type=space_type,
            attributes_json=_serialize_attributes(attributes),
        )
        record = result.single()
        return int(record["relationships_created"]) if record else 0

    def remove_fact(self, fact_id: str) -> bool:
        """
        Remove a fact (relationship) from Neo4j by its fact_id.

        Called when a fact is superseded in PostgreSQL — the relationship
        should no longer exist in the active graph.
        """
        with self._driver.session(database=self.config.database) as session:
            return session.execute_write(self._remove_fact_tx, fact_id=fact_id)

    @staticmethod
    def _remove_fact_tx(tx: ManagedTransaction, fact_id: str) -> bool:
        result = tx.run(
            """
            MATCH ()-[r]->()
            WHERE r.fact_id = $fact_id
            DELETE r
            RETURN count(r) as cnt
            """,
            fact_id=fact_id,
        )
        record = result.single()
        return record["cnt"] > 0 if record else False

    def remove_facts_for_entity(self, entity_id: str) -> int:
        """Remove all relationships involving an entity."""
        with self._driver.session(database=self.config.database) as session:
            return session.execute_write(self._remove_facts_for_entity_tx, entity_id=entity_id)

    @staticmethod
    def _remove_facts_for_entity_tx(tx: ManagedTransaction, entity_id: str) -> int:
        result = tx.run(
            """
            MATCH (e:Entity {id: $id})-[r]-()
            DELETE r
            RETURN count(r) as cnt
            """,
            id=entity_id,
        )
        record = result.single()
        return record["cnt"] if record else 0

    # =========================================================================
    # Graph Query Operations (used by GraphRetriever)
    # =========================================================================

    def get_neighbors(
        self,
        entity_id: str,
        max_hops: int = 2,
        relation_types: list[RelationType] | None = None,
        scope_id: str | None = None,
    ) -> tuple[dict[str, Entity], list[ValidatedFact]]:
        """
        Get the neighborhood of an entity up to max_hops.

        This is the primary query for graph-based retrieval, replacing
        the NetworkX-based approach.

        Args:
            entity_id: Starting entity ID
            max_hops: Maximum traversal depth (default: 2)
            relation_types: If set, only traverse these relationship types

        Returns:
            (entities dict, list of facts)
        """
        with self._driver.session(database=self.config.database) as session:
            return session.execute_read(
                self._get_neighbors_tx,
                entity_id=entity_id,
                max_hops=max_hops,
                relation_types=relation_types,
                scope_id=scope_id,
            )

    @staticmethod
    def _get_neighbors_tx(
        tx: ManagedTransaction,
        entity_id: str,
        max_hops: int,
        relation_types: list[RelationType] | None,
        scope_id: str | None,
    ) -> tuple[dict[str, Entity], list[ValidatedFact]]:
        # Build relationship type filter
        if relation_types:
            rel_filter = "|".join(rt.value for rt in relation_types)
            pattern = f"[r:{rel_filter}*1..{max_hops}]"
        else:
            pattern = f"[*1..{max_hops}]"

        # Get all nodes and relationships in neighborhood
        result = tx.run(
            f"""
            MATCH path = (start:Entity {{id: $entity_id}})-{pattern}-(neighbor:Entity)
              WHERE $scope_id IS NULL OR all(rel IN relationships(path) WHERE rel.scope_id = $scope_id)
            WITH collect(DISTINCT neighbor) AS neighbors,
                 collect(DISTINCT relationships(path)) AS all_rels
            UNWIND neighbors AS n
            WITH collect(n) AS neighbor_nodes, all_rels
            MATCH (start:Entity {{id: $entity_id}})
            WITH neighbor_nodes + [start] AS all_nodes, all_rels
            UNWIND all_nodes AS node
            WITH collect(DISTINCT node) AS nodes, all_rels
            UNWIND all_rels AS rel_list
            UNWIND rel_list AS rel
            WITH nodes, collect(DISTINCT rel) AS rels
            RETURN nodes, rels
            """,
            entity_id=entity_id,
            scope_id=scope_id,
        )
        record = result.single()

        entities: dict[str, Entity] = {}
        facts: list[ValidatedFact] = []

        if record is None:
            return entities, facts

        # Parse nodes
        for node in record["nodes"]:
            entity = _node_to_entity(node)
            entities[entity.id] = entity

        # Parse relationships
        seen_fact_ids: set[str] = set()
        for rel in record["rels"]:
            fact = _rel_to_fact(rel)
            if fact and fact.id not in seen_fact_ids:
                seen_fact_ids.add(fact.id)
                facts.append(fact)

        return entities, facts

    def get_safety_critical_facts(
        self,
        entity_id: str,
        max_hops: int = 2,
        relation_types: list[RelationType | str] | None = None,
        scope_id: str | None = None,
    ) -> list[ValidatedFact]:
        """
        Get prioritized critical facts for an entity.

        The relation types are supplied by the active retrieval weight profile.
        This keeps the graph query domain-agnostic and avoids hardcoded
        relationship names.
        """
        if relation_types is None:
            # Backward-compatible fallback if caller does not provide explicit critical relations.
            relation_types = [
                RelationType.HAS_ALLERGY,
                RelationType.CONTRAINDICATED_WITH,
                RelationType.PRESCRIBED,
            ]

        rel_types: list[str] = []
        for relation in relation_types:
            if isinstance(relation, RelationType):
                rel_types.append(relation.value)
            else:
                value = str(relation).strip()
                if value:
                    rel_types.append(value)

        rel_types = sorted(set(rel_types))
        if not rel_types:
            return []

        with self._driver.session(database=self.config.database) as session:
            result = session.execute_read(
                self._get_safety_facts_tx,
                entity_id=entity_id,
                rel_types=rel_types,
                max_hops=max_hops,
                scope_id=scope_id,
            )
            return result

    @staticmethod
    def _get_safety_facts_tx(
        tx: ManagedTransaction,
        entity_id: str,
        rel_types: list[str],
        max_hops: int,
        scope_id: str | None,
    ) -> list[ValidatedFact]:
        result = tx.run(
            f"""
            MATCH (start:Entity {{id: $entity_id}})-[r*1..{max_hops}]-(neighbor:Entity)
            UNWIND r AS rel
            WITH DISTINCT rel
            WHERE type(rel) IN $rel_types AND ($scope_id IS NULL OR rel.scope_id = $scope_id)
            WITH DISTINCT rel
            WITH DISTINCT rel, startNode(rel) AS s, endNode(rel) AS o
            RETURN rel, s.id AS subject_id, o.id AS object_id
            """,
            entity_id=entity_id,
            rel_types=rel_types,
            scope_id=scope_id,
        )

        facts: list[ValidatedFact] = []
        seen: set[str] = set()
        for record in result:
            fact = _rel_to_fact(
                record["rel"],
                subject_id=record["subject_id"],
                object_id=record["object_id"],
            )
            if fact and fact.id not in seen:
                seen.add(fact.id)
                facts.append(fact)

        return facts

    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 3,
        scope_id: str | None = None,
    ) -> list[list[ValidatedFact]]:
        """
        Find all paths between two entities.

        Replaces the NetworkX-based path finding in the old GraphRetriever.
        """
        with self._driver.session(database=self.config.database) as session:
            return session.execute_read(
                self._find_paths_tx,
                source_id=source_id,
                target_id=target_id,
                max_length=max_length,
                scope_id=scope_id,
            )

    @staticmethod
    def _find_paths_tx(
        tx: ManagedTransaction,
        source_id: str,
        target_id: str,
        max_length: int,
        scope_id: str | None,
    ) -> list[list[ValidatedFact]]:
        result = tx.run(
            f"""
            MATCH path = allShortestPaths(
                (s:Entity {{id: $source_id}})-[*1..{max_length}]-(t:Entity {{id: $target_id}})
            )
            WHERE $scope_id IS NULL OR all(rel IN relationships(path) WHERE rel.scope_id = $scope_id)
            RETURN relationships(path) AS rels
            """,
            source_id=source_id,
            target_id=target_id,
            scope_id=scope_id,
        )

        paths: list[list[ValidatedFact]] = []
        for record in result:
            fact_path = []
            for rel in record["rels"]:
                fact = _rel_to_fact(rel)
                if fact:
                    fact_path.append(fact)
            if fact_path:
                paths.append(fact_path)

        return paths

    def get_all_facts_by_relation(
        self,
        relation: RelationType,
        scope_id: str | None = None,
    ) -> list[ValidatedFact]:
        """Get all active facts of a specific relation type."""
        with self._driver.session(database=self.config.database) as session:
            return session.execute_read(
                self._get_facts_by_relation_tx,
                relation=relation.value,
                scope_id=scope_id,
            )

    @staticmethod
    def _get_facts_by_relation_tx(
        tx: ManagedTransaction,
        relation: str,
        scope_id: str | None,
    ) -> list[ValidatedFact]:
        result = tx.run(
            f"""
            MATCH (s:Entity)-[r:{relation}]->(o:Entity)
            WHERE $scope_id IS NULL OR r.scope_id = $scope_id
            RETURN r, s.id AS subject_id, o.id AS object_id
            """,
            scope_id=scope_id,
        )

        facts = []
        for record in result:
            fact = _rel_to_fact(
                record["r"],
                subject_id=record["subject_id"],
                object_id=record["object_id"],
            )
            if fact:
                facts.append(fact)
        return facts

    def get_facts_for_entity(
        self,
        entity_id: str,
        relation: RelationType | None = None,
        as_subject: bool = True,
        scope_id: str | None = None,
    ) -> list[ValidatedFact]:
        """Get all active facts for an entity, optionally filtered by relation."""
        with self._driver.session(database=self.config.database) as session:
            return session.execute_read(
                self._get_facts_for_entity_tx,
                entity_id=entity_id,
                relation=relation.value if relation else None,
                as_subject=as_subject,
                scope_id=scope_id,
            )

    @staticmethod
    def _get_facts_for_entity_tx(
        tx: ManagedTransaction,
        entity_id: str,
        relation: str | None,
        as_subject: bool,
        scope_id: str | None,
    ) -> list[ValidatedFact]:
        if as_subject:
            if relation:
                query = f"""
                    MATCH (s:Entity {{id: $entity_id}})-[r:{relation}]->(o:Entity)
                    WHERE $scope_id IS NULL OR r.scope_id = $scope_id
                    RETURN r, s.id AS subject_id, o.id AS object_id
                """
            else:
                query = """
                    MATCH (s:Entity {id: $entity_id})-[r]->(o:Entity)
                    WHERE $scope_id IS NULL OR r.scope_id = $scope_id
                    RETURN r, s.id AS subject_id, o.id AS object_id
                """
        else:
            if relation:
                query = f"""
                    MATCH (s:Entity)-[r:{relation}]->(o:Entity {{id: $entity_id}})
                    WHERE $scope_id IS NULL OR r.scope_id = $scope_id
                    RETURN r, s.id AS subject_id, o.id AS object_id
                """
            else:
                query = """
                    MATCH (s:Entity)-[r]->(o:Entity {id: $entity_id})
                    WHERE $scope_id IS NULL OR r.scope_id = $scope_id
                    RETURN r, s.id AS subject_id, o.id AS object_id
                """

        result = tx.run(query, entity_id=entity_id, scope_id=scope_id)

        facts = []
        for record in result:
            fact = _rel_to_fact(
                record["r"],
                subject_id=record["subject_id"],
                object_id=record["object_id"],
            )
            if fact:
                facts.append(fact)
        return facts

    # =========================================================================
    # Bulk Sync Operations
    # =========================================================================

    def sync_all_entities(self, entities: list[Entity]) -> int:
        """Bulk sync entities to Neo4j. Returns count of synced entities."""
        with self._driver.session(database=self.config.database) as session:
            return session.execute_write(self._sync_all_entities_tx, entities=entities)

    @staticmethod
    def _sync_all_entities_tx(tx: ManagedTransaction, entities: list[Entity]) -> int:
        for entity in entities:
            type_label = _entity_type_to_label(entity.entity_type)
            scope = _extract_scope_from_attributes(entity.attributes)
            tx.run(
                f"""
                MERGE (e:Entity {{id: $id}})
                SET e.name = $name,
                    e.entity_type = $entity_type,
                    e.canonical_id = $canonical_id,
                    e.attributes = $attributes_json,
                    e.scope_id = $scope_id,
                    e.tenant_id = $tenant_id,
                    e.app_id = $app_id,
                    e.user_id = $user_id,
                    e.agent_id = $agent_id,
                    e.run_id = $run_id,
                    e.space_type = $space_type,
                    e.created_at = $created_at,
                    e.updated_at = $updated_at,
                    e:{type_label}
                """,
                id=entity.id,
                name=entity.name,
                entity_type=entity.entity_type.value,
                canonical_id=entity.canonical_id,
                attributes_json=_serialize_attributes(entity.attributes),
                scope_id=scope.get("scope_id"),
                tenant_id=scope.get("tenant_id"),
                app_id=scope.get("app_id"),
                user_id=scope.get("user_id"),
                agent_id=scope.get("agent_id"),
                run_id=scope.get("run_id"),
                space_type=scope.get("space_type"),
                created_at=entity.created_at.isoformat(),
                updated_at=entity.updated_at.isoformat(),
            )
        return len(entities)

    def sync_all_active_facts(self, facts: list[ValidatedFact]) -> int:
        """Bulk sync active facts to Neo4j. Returns count of synced facts."""
        with self._driver.session(database=self.config.database) as session:
            return session.execute_write(self._sync_all_facts_tx, facts=facts)

    @staticmethod
    def _sync_all_facts_tx(tx: ManagedTransaction, facts: list[ValidatedFact]) -> int:
        count = 0
        for fact in facts:
            if not fact.is_active:
                continue
            if fact.object_id is None:
                continue
            scope = _extract_scope_from_attributes(fact.attributes)
            rel_type = fact.relation.value
            attribute_key = resolve_attribute_key(fact.value, fact.attributes)
            tx.run(
                f"""
                MATCH (s:Entity {{id: $subject_id}})
                MATCH (o:Entity {{id: $object_id}})
                WHERE $scope_id IS NULL OR (
                    coalesce(s.scope_id, $scope_id) = $scope_id
                    AND coalesce(o.scope_id, $scope_id) = $scope_id
                )
                CREATE (s)-[r:{rel_type} {{
                    fact_id: $fact_id,
                    subject_id: $subject_id,
                    object_id: $object_id,
                    value: $value,
                    attribute_key: $attribute_key,
                    confidence: $confidence,
                    valid_from: $valid_from,
                    candidate_fact_id: $candidate_fact_id,
                    source_interaction_id: $source_interaction_id,
                    scope_id: $scope_id,
                    tenant_id: $tenant_id,
                    app_id: $app_id,
                    user_id: $user_id,
                    agent_id: $agent_id,
                    run_id: $run_id,
                    space_type: $space_type,
                    attributes: $attributes_json
                }}]->(o)
                """,
                subject_id=fact.subject_id,
                object_id=fact.object_id,
                fact_id=fact.id,
                value=sanitize_fact_value(fact.value),
                attribute_key=attribute_key,
                confidence=fact.confidence,
                valid_from=fact.valid_from.isoformat(),
                candidate_fact_id=fact.candidate_fact_id,
                source_interaction_id=fact.source_interaction_id,
                scope_id=scope.get("scope_id"),
                tenant_id=scope.get("tenant_id"),
                app_id=scope.get("app_id"),
                user_id=scope.get("user_id"),
                agent_id=scope.get("agent_id"),
                run_id=scope.get("run_id"),
                space_type=scope.get("space_type"),
                attributes_json=_serialize_attributes(fact.attributes),
            )
            count += 1
        return count

    def clear(self) -> None:
        """Clear all data from Neo4j (use with caution!)."""
        with self._driver.session(database=self.config.database) as session:
            session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))
            logger.warning("All Neo4j data cleared")

    def get_statistics(self) -> dict[str, Any]:
        """Get Neo4j graph statistics."""
        with self._driver.session(database=self.config.database) as session:
            result = session.run(
                """
                MATCH (n:Entity)
                WITH count(n) AS node_count
                MATCH ()-[r]->()
                WITH node_count, count(r) AS rel_count
                RETURN node_count, rel_count
                """
            )
            record = result.single()
            if record:
                return {
                    "node_count": record["node_count"],
                    "relationship_count": record["rel_count"],
                }
            return {"node_count": 0, "relationship_count": 0}


# =============================================================================
# Helper Functions
# =============================================================================


def _entity_type_to_label(entity_type: EntityType) -> str:
    """Convert EntityType enum to a Neo4j node label (PascalCase)."""
    # "patient" -> "Patient", "therapeutic_class" -> "TherapeuticClass"
    return entity_type.value.replace("_", " ").title().replace(" ", "")


def _extract_scope_from_attributes(attributes: dict[str, Any] | None) -> dict[str, str | None]:
    attrs = attributes or {}
    tenant_id = attrs.get("tenant_id")
    app_id = attrs.get("app_id")
    user_id = attrs.get("user_id")
    scope_id = attrs.get("scope_id")
    if scope_id is None and tenant_id and app_id and user_id:
        scope_id = f"{tenant_id}:{app_id}:{user_id}"

    return {
        "scope_id": scope_id,
        "tenant_id": tenant_id,
        "app_id": app_id,
        "user_id": user_id,
        "agent_id": attrs.get("agent_id"),
        "run_id": attrs.get("run_id"),
        "space_type": attrs.get("space_type"),
    }


def _serialize_attributes(attrs: dict[str, Any]) -> str:
    """Serialize attributes dict to JSON string for Neo4j storage."""
    import json

    return json.dumps(attrs) if attrs else "{}"


def _node_to_entity(node) -> Entity:
    """Convert a Neo4j node to an Entity model."""
    import json

    props = dict(node)
    attributes = props.get("attributes", "{}")
    if isinstance(attributes, str):
        attributes = json.loads(attributes)

    for scope_key in (
        "scope_id",
        "tenant_id",
        "app_id",
        "user_id",
        "agent_id",
        "run_id",
        "space_type",
    ):
        if props.get(scope_key) is not None and attributes.get(scope_key) is None:
            attributes[scope_key] = props.get(scope_key)

    if (
        attributes.get("scope_id") is None
        and attributes.get("tenant_id")
        and attributes.get("app_id")
        and attributes.get("user_id")
    ):
        attributes["scope_id"] = (
            f"{attributes['tenant_id']}:{attributes['app_id']}:{attributes['user_id']}"
        )

    return Entity(
        id=props["id"],
        entity_type=EntityType(props["entity_type"]),
        name=props["name"],
        canonical_id=props.get("canonical_id"),
        attributes=attributes,
        created_at=datetime.fromisoformat(props["created_at"])
        if props.get("created_at")
        else datetime.now(timezone.utc),
        updated_at=datetime.fromisoformat(props["updated_at"])
        if props.get("updated_at")
        else datetime.now(timezone.utc),
    )


def _rel_to_fact(
    rel,
    subject_id: str | None = None,
    object_id: str | None = None,
) -> ValidatedFact | None:
    """Convert a Neo4j relationship to a ValidatedFact model.

    Args:
        rel: Neo4j relationship record.
        subject_id: Override start-node entity ID (from Cypher RETURN).
        object_id: Override end-node entity ID (from Cypher RETURN).
    """
    import json

    props = dict(rel)
    fact_id = props.get("fact_id")
    if not fact_id:
        return None

    attributes = props.get("attributes", "{}")
    if isinstance(attributes, str):
        attributes = json.loads(attributes)

    for scope_key in (
        "scope_id",
        "tenant_id",
        "app_id",
        "user_id",
        "agent_id",
        "run_id",
        "space_type",
    ):
        if props.get(scope_key) is not None and attributes.get(scope_key) is None:
            attributes[scope_key] = props.get(scope_key)

    if (
        attributes.get("scope_id") is None
        and attributes.get("tenant_id")
        and attributes.get("app_id")
        and attributes.get("user_id")
    ):
        attributes["scope_id"] = (
            f"{attributes['tenant_id']}:{attributes['app_id']}:{attributes['user_id']}"
        )

    # The relationship type IS the relation
    rel_type = rel.type  # e.g., "PRESCRIBED"

    # Determine subject/object IDs: prefer explicit args > rel properties > node IDs
    sid = subject_id or props.get("subject_id")
    oid = object_id or props.get("object_id")

    # Fallback: try to read from relationship nodes (works for full-path results)
    if not sid and hasattr(rel, "nodes") and rel.nodes:
        sid = rel.nodes[0].get("id")
    if not oid and hasattr(rel, "nodes") and rel.nodes:
        oid = rel.nodes[1].get("id") if len(rel.nodes) > 1 else None

    if not sid or not oid:
        return None

    return ValidatedFact(
        id=fact_id,
        candidate_fact_id=props.get("candidate_fact_id", ""),
        source_interaction_id=props.get("source_interaction_id", ""),
        subject_id=sid,
        relation=RelationType(rel_type),
        object_id=oid,
        value=sanitize_fact_value(props.get("value")),
        valid_from=datetime.fromisoformat(props["valid_from"])
        if props.get("valid_from")
        else datetime.now(timezone.utc),
        valid_to=None,  # Only active facts exist in Neo4j
        confidence=props.get("confidence", 1.0),
        attributes=attributes,
    )


# =============================================================================
# Factory Function
# =============================================================================


def create_neo4j_store(
    config: Neo4jConfig | None = None,
    create_schema: bool = True,
) -> Neo4jStore:
    """
    Factory function to create and initialize a Neo4jStore.

    Usage:
        store = create_neo4j_store()
        # Use the store...
        store.close()
    """
    store = Neo4jStore(config)
    store.initialize(create_schema=create_schema)
    return store
