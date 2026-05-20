#!/usr/bin/env python3
"""
Hybrid Storage Verification Tests

Tests that data is correctly saved with all required fields
(source_text, timestamp, entities, metadata, confidence) across
the in-memory store, and verifies data completeness and richness.

Run:
    PYTHONPATH=src python -m pytest tests/test_hybrid_storage.py -v
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grounded_memory.core.constraints import ConstraintValidator
from grounded_memory.core.grounding import GroundingDecision, GroundingOperator
from grounded_memory.core.models import (
    ActorType,
    CandidateFact,
    Entity,
    EntityType,
    Interaction,
    RelationType,
    ValidatedFact,
)
from grounded_memory.core.neo4j_store import Neo4jStore
from grounded_memory.core.store import MemoryStore

# =============================================================================
# Helpers
# =============================================================================


def _setup() -> tuple[GroundingOperator, MemoryStore]:
    """Create a grounding operator with clean in-memory store."""
    store = MemoryStore()
    validator = ConstraintValidator()
    operator = GroundingOperator(validator=validator, memory_store=store)
    return operator, store


def _create_rich_scenario(store: MemoryStore) -> dict:
    """Create a realistic scenario with multiple entity types and relations."""
    # Create diverse entities
    alice = Entity(id="person-alice", entity_type=EntityType.PERSON, name="Alice Johnson")
    bob = Entity(id="person-bob", entity_type=EntityType.PERSON, name="Bob Smith")
    paris = Entity(id="place-paris", entity_type=EntityType.PLACE, name="Paris")
    acme = Entity(id="org-acme", entity_type=EntityType.ORGANIZATION, name="Acme Corp")
    ml_project = Entity(id="proj-ml", entity_type=EntityType.PROJECT, name="ML Pipeline")
    python = Entity(id="concept-python", entity_type=EntityType.CONCEPT, name="Python")
    k8s = Entity(id="tool-k8s", entity_type=EntityType.TOOL, name="Kubernetes")

    for entity in [alice, bob, paris, acme, ml_project, python, k8s]:
        store.add_entity(entity)

    return {
        "alice": alice,
        "bob": bob,
        "paris": paris,
        "acme": acme,
        "ml_project": ml_project,
        "python": python,
        "k8s": k8s,
    }


# =============================================================================
# Test: Entity Storage Completeness
# =============================================================================


class TestEntityStorage:
    """Verify entities are stored correctly with all fields."""

    def test_entity_types_generic(self):
        """Generic entity types (PERSON, PLACE, etc.) should work."""
        store = MemoryStore()
        person = Entity(id="p1", entity_type=EntityType.PERSON, name="John")
        place = Entity(id="p2", entity_type=EntityType.PLACE, name="London")
        org = Entity(id="p3", entity_type=EntityType.ORGANIZATION, name="Google")
        concept = Entity(id="p4", entity_type=EntityType.CONCEPT, name="AI")

        for e in [person, place, org, concept]:
            store.add_entity(e)

        assert store.get_entity("p1").entity_type == EntityType.PERSON
        assert store.get_entity("p2").entity_type == EntityType.PLACE
        assert store.get_entity("p3").entity_type == EntityType.ORGANIZATION
        assert store.get_entity("p4").entity_type == EntityType.CONCEPT

    def test_entity_has_timestamps(self):
        """Entities should have created_at and updated_at timestamps."""
        store = MemoryStore()
        entity = Entity(id="e1", entity_type=EntityType.PERSON, name="Test")
        store.add_entity(entity)

        retrieved = store.get_entity("e1")
        assert retrieved.created_at is not None
        assert retrieved.updated_at is not None
        assert isinstance(retrieved.created_at, datetime)

    def test_entity_attributes_preserved(self):
        """Entity attributes should be fully preserved."""
        store = MemoryStore()
        entity = Entity(
            id="e1",
            entity_type=EntityType.PERSON,
            name="Alice",
            attributes={
                "email": "alice@example.com",
                "department": "Engineering",
                "skills": ["Python", "ML"],
            },
        )
        store.add_entity(entity)

        retrieved = store.get_entity("e1")
        assert retrieved.attributes["email"] == "alice@example.com"
        assert retrieved.attributes["department"] == "Engineering"
        assert "Python" in retrieved.attributes["skills"]


class TestTemporalComparisons:
    """Regression tests for mixed timezone-aware and naive fact timestamps."""

    def test_validated_fact_active_checks_accept_aware_postgres_timestamps(self):
        fact = ValidatedFact(
            candidate_fact_id="candidate-aware-time",
            source_interaction_id="interaction-aware-time",
            subject_id="subject-aware-time",
            relation=RelationType.RELATED_TO,
            object_id="object-aware-time",
            confidence=0.9,
            valid_from=datetime.now(timezone.utc) - timedelta(minutes=5),
        )

        assert fact.is_active
        assert fact.is_active_at(datetime.now(timezone.utc).replace(tzinfo=None))

    def test_validated_fact_active_checks_accept_naive_runtime_timestamps(self):
        fact = ValidatedFact(
            candidate_fact_id="candidate-naive-time",
            source_interaction_id="interaction-naive-time",
            subject_id="subject-naive-time",
            relation=RelationType.RELATED_TO,
            object_id="object-naive-time",
            confidence=0.9,
            valid_from=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=5),
        )

        assert fact.is_active
        assert fact.is_active_at(datetime.now(timezone.utc))


class TestNeo4jProjection:
    """Verify Neo4j projection supports shared concept nodes across scopes."""

    def test_add_fact_tx_does_not_require_object_scope_match(self):
        class FakeResult:
            def single(self):
                return {"relationships_created": 1}

        class FakeTx:
            def __init__(self):
                self.query = ""
                self.params = {}

            def run(self, query, **params):
                self.query = query
                self.params = params
                return FakeResult()

        tx = FakeTx()
        created = Neo4jStore._add_fact_tx(
            tx,
            subject_id="patient-haroun",
            object_id="allergy-penicillin",
            rel_type="HAS_ALLERGY",
            fact_id="fact-haroun-penicillin",
            value=None,
            attribute_key=None,
            confidence=0.95,
            valid_from=datetime.now(timezone.utc).isoformat(),
            candidate_fact_id="candidate-haroun-penicillin",
            source_interaction_id="interaction-haroun-penicillin",
            attributes={"scope_id": "demo:gmem:live-user"},
            scope_id="demo:gmem:live-user",
            tenant_id="demo",
            app_id="gmem",
            user_id="live-user",
            agent_id="agent",
            run_id="run",
            space_type="user",
        )

        assert created == 1
        assert "coalesce(s.scope_id, $scope_id) = $scope_id" in tx.query
        assert "coalesce(o.scope_id, $scope_id)" not in tx.query
        assert tx.params["object_id"] == "allergy-penicillin"


# =============================================================================
# Test: Fact Data Completeness
# =============================================================================


class TestFactDataCompleteness:
    """Verify that stored facts contain all required rich data fields."""

    def test_fact_has_source_text(self):
        """ValidatedFact should carry the original sentence text."""
        operator, store = _setup()
        entities = _create_rich_scenario(store)

        raw_text = "Alice works at Acme Corp as a senior engineer"
        interaction = Interaction(actor=ActorType.USER, raw_text=raw_text)
        store.add_interaction(interaction)

        candidate = CandidateFact(
            source_interaction_id=interaction.id,
            subject_entity_id=entities["alice"].id,
            relation=RelationType.WORKS_AT,
            object_entity_id=entities["acme"].id,
            confidence=0.92,
        )
        result = operator.ground(candidate)

        assert result.is_success
        fact = result.validated_fact
        assert fact.source_text == raw_text, "source_text must match interaction raw_text"

    def test_fact_has_timestamps(self):
        """ValidatedFact should have proper temporal markers."""
        operator, store = _setup()
        entities = _create_rich_scenario(store)
        interaction = Interaction(actor=ActorType.USER, raw_text="Test timestamps")
        store.add_interaction(interaction)

        candidate = CandidateFact(
            source_interaction_id=interaction.id,
            subject_entity_id=entities["alice"].id,
            relation=RelationType.LOCATED_IN,
            object_entity_id=entities["paris"].id,
            confidence=0.88,
        )
        result = operator.ground(candidate)

        fact = result.validated_fact
        assert fact.valid_from is not None
        assert fact.valid_to is None, "Active fact should have NULL valid_to"
        assert fact.validated_at is not None
        assert isinstance(fact.valid_from, datetime)

    def test_fact_has_confidence(self):
        """ValidatedFact confidence should match the candidate."""
        operator, store = _setup()
        entities = _create_rich_scenario(store)
        interaction = Interaction(actor=ActorType.USER, raw_text="Confidence check")
        store.add_interaction(interaction)

        candidate = CandidateFact(
            source_interaction_id=interaction.id,
            subject_entity_id=entities["alice"].id,
            relation=RelationType.MANAGES,
            object_entity_id=entities["ml_project"].id,
            confidence=0.87,
        )
        result = operator.ground(candidate)

        assert result.validated_fact.confidence == 0.87

    def test_fact_has_source_metadata(self):
        """ValidatedFact should have source_metadata with actor info."""
        operator, store = _setup()
        entities = _create_rich_scenario(store)
        interaction = Interaction(actor=ActorType.SYSTEM, raw_text="System fact")
        store.add_interaction(interaction)

        candidate = CandidateFact(
            source_interaction_id=interaction.id,
            subject_entity_id=entities["ml_project"].id,
            relation=RelationType.DEPENDS_ON,
            object_entity_id=entities["k8s"].id,
            confidence=0.95,
        )
        result = operator.ground(candidate)

        fact = result.validated_fact
        assert fact.source_metadata is not None
        assert fact.source_metadata.get("actor") == "system"
        assert "interaction_timestamp" in fact.source_metadata

    def test_fact_attributes_preserved(self):
        """Custom attributes on facts should be preserved through grounding."""
        operator, store = _setup()
        entities = _create_rich_scenario(store)
        interaction = Interaction(actor=ActorType.USER, raw_text="Attribute test")
        store.add_interaction(interaction)

        candidate = CandidateFact(
            source_interaction_id=interaction.id,
            subject_entity_id=entities["alice"].id,
            relation=RelationType.HAS_ATTRIBUTE,
            object_entity_id=entities["python"].id,
            value="python",
            confidence=0.9,
            attributes={
                "key": "primary_language",
                "source": "user_profile",
                "custom_field": "custom_value",
            },
        )
        result = operator.ground(candidate)

        fact = result.validated_fact
        assert fact.attributes["key"] == "primary_language"
        assert fact.attributes["source"] == "user_profile"
        assert fact.attributes["custom_field"] == "custom_value"


# =============================================================================
# Test: Embedding Field
# =============================================================================


class TestEmbeddingField:
    """Verify embedding vector field on ValidatedFact."""

    def test_embedding_is_optional(self):
        """Embedding should be None by default (not required)."""
        fact = ValidatedFact(
            candidate_fact_id="cf-1",
            source_interaction_id="int-1",
            subject_id="user-1",
            relation=RelationType.HAS_ATTRIBUTE,
            value="test",
            valid_from=datetime.now(timezone.utc),
            confidence=0.9,
        )
        assert fact.embedding is None

    def test_embedding_can_be_set(self):
        """Embedding vector should be storable on facts."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5] * 76  # 380-dim test vector
        fact = ValidatedFact(
            candidate_fact_id="cf-1",
            source_interaction_id="int-1",
            subject_id="user-1",
            relation=RelationType.HAS_ATTRIBUTE,
            value="test",
            valid_from=datetime.now(timezone.utc),
            confidence=0.9,
            embedding=embedding,
        )
        assert fact.embedding is not None
        assert len(fact.embedding) == 380


# =============================================================================
# Test: Multi-Relation Scenarios
# =============================================================================


class TestMultiRelationScenarios:
    """Verify that diverse relation types work correctly."""

    def test_generic_relations_grounded(self):
        """All new generic relation types should ground correctly."""
        operator, store = _setup()
        entities = _create_rich_scenario(store)
        interaction = Interaction(actor=ActorType.USER, raw_text="Multi-relation test")
        store.add_interaction(interaction)

        relations_to_test = [
            (entities["alice"], RelationType.WORKS_AT, entities["acme"]),
            (entities["alice"], RelationType.LOCATED_IN, entities["paris"]),
            (entities["alice"], RelationType.MANAGES, entities["ml_project"]),
            (entities["ml_project"], RelationType.DEPENDS_ON, entities["k8s"]),
            (entities["acme"], RelationType.OWNS, entities["ml_project"]),
            (entities["k8s"], RelationType.USED_BY, entities["ml_project"]),
        ]

        for subject, relation, obj in relations_to_test:
            candidate = CandidateFact(
                source_interaction_id=interaction.id,
                subject_entity_id=subject.id,
                relation=relation,
                object_entity_id=obj.id,
                confidence=0.9,
            )
            result = operator.ground(candidate)
            assert result.is_success, (
                f"Failed to ground {relation.value}: {result.get_explanation()}"
            )

        # Verify all facts are in store
        all_facts = store.get_all_validated_facts()
        assert len(all_facts) == 6

    def test_relation_type_on_stored_fact(self):
        """Stored fact should have the correct relation type."""
        operator, store = _setup()
        entities = _create_rich_scenario(store)
        interaction = Interaction(actor=ActorType.USER, raw_text="Relation type test")
        store.add_interaction(interaction)

        candidate = CandidateFact(
            source_interaction_id=interaction.id,
            subject_entity_id=entities["bob"].id,
            relation=RelationType.MEMBER_OF,
            object_entity_id=entities["acme"].id,
            confidence=0.85,
        )
        result = operator.ground(candidate)

        assert result.validated_fact.relation == RelationType.MEMBER_OF


# =============================================================================
# Test: Bitemporal Query
# =============================================================================


class TestBitemporalQuery:
    """Verify point-in-time queries return correct state."""

    def test_active_at_current_time(self):
        """Current facts should be active at present time."""
        operator, store = _setup()
        entities = _create_rich_scenario(store)
        interaction = Interaction(actor=ActorType.USER, raw_text="Active test")
        store.add_interaction(interaction)

        candidate = CandidateFact(
            source_interaction_id=interaction.id,
            subject_entity_id=entities["alice"].id,
            relation=RelationType.WORKS_AT,
            object_entity_id=entities["acme"].id,
            confidence=0.9,
        )
        result = operator.ground(candidate)
        fact = result.validated_fact

        assert fact.is_active
        assert fact.is_active_at(datetime.now(timezone.utc))

    def test_superseded_fact_not_active(self):
        """Superseded fact should not be active at current time."""
        operator, store = _setup()
        entities = _create_rich_scenario(store)

        # Fact 1
        i1 = Interaction(actor=ActorType.USER, raw_text="First job")
        store.add_interaction(i1)
        c1 = CandidateFact(
            source_interaction_id=i1.id,
            subject_entity_id=entities["alice"].id,
            relation=RelationType.HAS_ATTRIBUTE,
            object_entity_id=entities["acme"].id,
            value="acme",
            confidence=0.9,
            attributes={"key": "employer"},
        )
        r1 = operator.ground(c1)

        # Fact 2: supersedes fact 1
        new_org = Entity(id="org-new", entity_type=EntityType.ORGANIZATION, name="NewCo")
        store.add_entity(new_org)
        i2 = Interaction(actor=ActorType.USER, raw_text="Changed jobs")
        store.add_interaction(i2)
        c2 = CandidateFact(
            source_interaction_id=i2.id,
            subject_entity_id=entities["alice"].id,
            relation=RelationType.HAS_ATTRIBUTE,
            object_entity_id=new_org.id,
            value="newco",
            confidence=0.95,
            attributes={"key": "employer"},
        )
        r2 = operator.ground(c2)

        assert r2.decision == GroundingDecision.SUPERSEDED

        # Old fact should not be active
        old_fact = store.get_validated_fact(r1.validated_fact.id)
        assert not old_fact.is_active

        # New fact should be active
        assert r2.validated_fact.is_active


# =============================================================================
# Test: Data Inspection (for demo)
# =============================================================================


class TestDataInspection:
    """Print stored data for visual demo verification."""

    def test_print_stored_data_summary(self, capsys):
        """Print a formatted summary of all stored data."""
        operator, store = _setup()
        entities = _create_rich_scenario(store)

        scenarios = [
            (
                "Alice works at Acme Corp",
                entities["alice"],
                RelationType.WORKS_AT,
                entities["acme"],
                0.92,
            ),
            (
                "Alice lives in Paris",
                entities["alice"],
                RelationType.LOCATED_IN,
                entities["paris"],
                0.88,
            ),
            (
                "Alice manages ML Pipeline",
                entities["alice"],
                RelationType.MANAGES,
                entities["ml_project"],
                0.91,
            ),
            (
                "ML Pipeline depends on Kubernetes",
                entities["ml_project"],
                RelationType.DEPENDS_ON,
                entities["k8s"],
                0.95,
            ),
            (
                "Acme Corp owns ML Pipeline",
                entities["acme"],
                RelationType.OWNS,
                entities["ml_project"],
                0.90,
            ),
            (
                "Bob is member of Acme Corp",
                entities["bob"],
                RelationType.MEMBER_OF,
                entities["acme"],
                0.85,
            ),
        ]

        for text, subject, relation, obj, conf in scenarios:
            interaction = Interaction(actor=ActorType.USER, raw_text=text)
            store.add_interaction(interaction)
            candidate = CandidateFact(
                source_interaction_id=interaction.id,
                subject_entity_id=subject.id,
                relation=relation,
                object_entity_id=obj.id,
                confidence=conf,
            )
            operator.ground(candidate)

        # Print summary
        all_facts = store.get_all_validated_facts()
        all_entities = store.get_all_entities()

        print("\n" + "=" * 70)
        print("📊 STORED DATA SUMMARY (In-Memory)")
        print("=" * 70)

        print(f"\n🔹 Entities: {len(all_entities)}")
        for e in all_entities:
            print(f"   [{e.entity_type.value:15s}] {e.name} (id={e.id[:12]}...)")

        print(f"\n🔹 Validated Facts: {len(all_facts)}")
        for f in all_facts:
            subject = store.get_entity(f.subject_id)
            obj = store.get_entity(f.object_id) if f.object_id else None
            s_name = subject.name if subject else f.subject_id
            o_name = obj.name if obj else (f.value or "—")
            status = "✅ active" if f.is_active else "⏹ superseded"

            print(f"   {s_name} —[{f.relation.value}]→ {o_name}")
            print(f"      confidence={f.confidence:.2f} | {status}")
            print(f'      source_text="{f.source_text or "(none)"}"')
            print(f"      valid_from={f.valid_from.isoformat()}")
            print(f"      source_metadata={f.source_metadata}")

        print("=" * 70)
        assert len(all_facts) == 6
        assert len(all_entities) == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
