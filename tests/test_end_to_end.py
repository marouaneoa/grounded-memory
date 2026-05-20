#!/usr/bin/env python3
"""
End-to-End Verification Tests

Full pipeline tests: Add → Ground → Store → Retrieve → Supersede.
Demonstrates realistic scenarios for the demo.

Run:
    PYTHONPATH=src python -m pytest tests/test_end_to_end.py -v -s
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grounded_memory.core.conflict_resolution import (
    ConflictResolutionStrategy,
)
from grounded_memory.core.constraints import (
    BaseConstraintEvaluator,
    ConstraintValidator,
    ConstraintViolation,
)
from grounded_memory.core.grounding import (
    GroundingDecision,
    GroundingOperator,
)
from grounded_memory.core.models import (
    ActorType,
    CandidateFact,
    Entity,
    EntityType,
    Interaction,
    RelationType,
)
from grounded_memory.core.store import MemoryStore

# =============================================================================
# Helpers
# =============================================================================


def _fresh_system(
    strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.COMPOSITE,
) -> tuple[GroundingOperator, MemoryStore]:
    store = MemoryStore()
    validator = ConstraintValidator()
    operator = GroundingOperator(
        validator=validator,
        memory_store=store,
        conflict_strategy=strategy,
    )
    return operator, store


# =============================================================================
# Scenario 1: Engineering Knowledge Graph
# =============================================================================


class TestEngineeringScenario:
    """End-to-end scenario: building an engineering knowledge graph."""

    def test_build_service_dependency_graph(self, capsys):
        """Build a service dependency graph and verify data quality."""
        operator, store = _fresh_system()

        # Create entities
        api_gw = Entity(id="svc-api-gw", entity_type=EntityType.SERVICE, name="API Gateway")
        auth_svc = Entity(id="svc-auth", entity_type=EntityType.SERVICE, name="Auth Service")
        user_svc = Entity(id="svc-users", entity_type=EntityType.SERVICE, name="User Service")
        db_pg = Entity(id="tool-pg", entity_type=EntityType.TOOL, name="PostgreSQL")
        redis = Entity(id="tool-redis", entity_type=EntityType.TOOL, name="Redis")
        team = Entity(id="org-platform", entity_type=EntityType.ORGANIZATION, name="Platform Team")

        for e in [api_gw, auth_svc, user_svc, db_pg, redis, team]:
            store.add_entity(e)

        # Ground facts
        facts_to_add = [
            (
                "API Gateway depends on Auth Service",
                api_gw,
                RelationType.DEPENDS_ON,
                auth_svc,
                0.95,
            ),
            (
                "API Gateway depends on User Service",
                api_gw,
                RelationType.DEPENDS_ON,
                user_svc,
                0.95,
            ),
            ("Auth Service depends on Redis", auth_svc, RelationType.DEPENDS_ON, redis, 0.90),
            ("User Service depends on PostgreSQL", user_svc, RelationType.DEPENDS_ON, db_pg, 0.90),
            ("Platform Team manages API Gateway", team, RelationType.MANAGES, api_gw, 0.88),
            ("Platform Team manages Auth Service", team, RelationType.MANAGES, auth_svc, 0.88),
        ]

        results = []
        for text, subject, relation, obj, conf in facts_to_add:
            interaction = Interaction(actor=ActorType.SYSTEM, raw_text=text)
            store.add_interaction(interaction)
            candidate = CandidateFact(
                source_interaction_id=interaction.id,
                subject_entity_id=subject.id,
                relation=relation,
                object_entity_id=obj.id,
                confidence=conf,
            )
            result = operator.ground(candidate)
            results.append(result)

        # Verify all facts grounded
        assert all(r.is_success for r in results)
        all_facts = store.get_all_validated_facts()
        assert len(all_facts) == 6

        # Verify dependency chain
        api_gw_deps = store.get_facts_by_relation(
            entity_id=api_gw.id,
            relation=RelationType.DEPENDS_ON,
            as_subject=True,
        )
        assert len(api_gw_deps) == 2

        # Print summary
        print("\n" + "=" * 70)
        print("🏗️  ENGINEERING KNOWLEDGE GRAPH")
        print("=" * 70)
        for f in all_facts:
            s = store.get_entity(f.subject_id)
            o = store.get_entity(f.object_id)
            print(f"   {s.name} —[{f.relation.value}]→ {o.name}  (conf={f.confidence:.2f})")
            print(f'      source: "{f.source_text}"')
        print("=" * 70)


# =============================================================================
# Scenario 2: Supersession Chain (Job History)
# =============================================================================


class TestSupersessionChain:
    """End-to-end scenario: temporal chain of superseding facts."""

    def test_job_history_chain(self, capsys):
        """Track a person's job history through supersession."""
        operator, store = _fresh_system()

        alice = Entity(id="person-alice", entity_type=EntityType.PERSON, name="Alice")
        google = Entity(id="org-google", entity_type=EntityType.ORGANIZATION, name="Google")
        meta = Entity(id="org-meta", entity_type=EntityType.ORGANIZATION, name="Meta")
        openai = Entity(id="org-openai", entity_type=EntityType.ORGANIZATION, name="OpenAI")

        for e in [alice, google, meta, openai]:
            store.add_entity(e)

        # Job 1: Google (2020)
        i1 = Interaction(actor=ActorType.USER, raw_text="Alice works at Google since 2020")
        store.add_interaction(i1)
        c1 = CandidateFact(
            source_interaction_id=i1.id,
            subject_entity_id=alice.id,
            relation=RelationType.WORKS_AT,
            object_entity_id=google.id,
            confidence=0.90,
            attributes={"key": "employer"},
        )
        r1 = operator.ground(c1)
        assert r1.decision == GroundingDecision.APPROVED

        # Job 2: Meta (2022) — supersedes Google
        i2 = Interaction(actor=ActorType.USER, raw_text="Alice moved to Meta in 2022")
        store.add_interaction(i2)
        c2 = CandidateFact(
            source_interaction_id=i2.id,
            subject_entity_id=alice.id,
            relation=RelationType.WORKS_AT,
            object_entity_id=meta.id,
            confidence=0.92,
            attributes={"key": "employer"},
        )
        r2 = operator.ground(c2)
        assert r2.decision == GroundingDecision.SUPERSEDED
        assert len(r2.superseded_facts) == 1

        # Job 3: OpenAI (2024) — supersedes Meta
        i3 = Interaction(actor=ActorType.USER, raw_text="Alice joined OpenAI in 2024")
        store.add_interaction(i3)
        c3 = CandidateFact(
            source_interaction_id=i3.id,
            subject_entity_id=alice.id,
            relation=RelationType.WORKS_AT,
            object_entity_id=openai.id,
            confidence=0.95,
            attributes={"key": "employer"},
        )
        r3 = operator.ground(c3)
        assert r3.decision == GroundingDecision.SUPERSEDED

        # Verify chain
        all_facts = store.get_all_validated_facts()
        active_facts = [f for f in all_facts if f.is_active]
        superseded_facts = [f for f in all_facts if not f.is_active]

        assert len(active_facts) == 1, "Only the latest job should be active"
        assert active_facts[0].object_id == openai.id
        assert len(superseded_facts) == 2, "Two previous jobs should be superseded"

        # Print chain
        print("\n" + "=" * 70)
        print("📜 SUPERSESSION CHAIN: Alice's Job History")
        print("=" * 70)
        for f in all_facts:
            org = store.get_entity(f.object_id)
            status = "✅ ACTIVE" if f.is_active else "⏹ SUPERSEDED"
            valid_to_str = f.valid_to.isoformat() if f.valid_to else "—"
            print(f"   {org.name:10s} | conf={f.confidence:.2f} | {status}")
            print(f"       valid: {f.valid_from.isoformat()} → {valid_to_str}")
            print(f'       source: "{f.source_text}"')
        print("=" * 70)


# =============================================================================
# Scenario 3: Constraint Rejection with Audit
# =============================================================================


class _NoSelfReferenceConstraint(BaseConstraintEvaluator):
    """Reject facts where subject == object."""

    @property
    def constraint_id(self) -> str:
        return "no_self_reference"

    @property
    def constraint_name(self) -> str:
        return "No Self-Reference"

    @property
    def description(self) -> str:
        return "A fact cannot relate an entity to itself"

    def evaluate(self, candidate, knowledge_state) -> ConstraintViolation | None:
        if candidate.object_entity_id and candidate.subject_entity_id == candidate.object_entity_id:
            return ConstraintViolation(
                constraint_id=self.constraint_id,
                constraint_name=self.constraint_name,
                description="Subject and object entity are the same",
                severity="error",
            )
        return None


class TestConstraintRejection:
    """End-to-end: constraint rejects invalid fact with full audit trail."""

    def test_self_reference_rejected(self, capsys):
        """Self-referential fact should be rejected with explanation."""
        store = MemoryStore()
        validator = ConstraintValidator()
        validator.register(_NoSelfReferenceConstraint())
        operator = GroundingOperator(validator=validator, memory_store=store)

        alice = Entity(id="person-alice", entity_type=EntityType.PERSON, name="Alice")
        store.add_entity(alice)

        interaction = Interaction(
            actor=ActorType.USER,
            raw_text="Alice manages Alice",
        )
        store.add_interaction(interaction)

        candidate = CandidateFact(
            source_interaction_id=interaction.id,
            subject_entity_id=alice.id,
            relation=RelationType.MANAGES,
            object_entity_id=alice.id,  # Self-reference!
            confidence=0.8,
        )
        result = operator.ground(candidate)

        assert result.decision == GroundingDecision.REJECTED
        assert result.rejection_record is not None

        print("\n" + "=" * 70)
        print("🚫 CONSTRAINT REJECTION DEMO")
        print("=" * 70)
        print("   Fact: Alice —[MANAGES]→ Alice")
        print(f"   Decision: {result.decision.value}")
        print(f"   Reason: {result.rejection_record.reason}")
        print(f"   Constraint: {result.rejection_record.constraint_name}")
        print(f"   Severity: {result.rejection_record.severity}")
        print("=" * 70)


# =============================================================================
# Scenario 4: Multi-Domain Facts
# =============================================================================


class TestMultiDomainFacts:
    """Demonstrate facts across different domain entity types."""

    def test_mixed_domain_facts(self, capsys):
        """Facts spanning engineering + organizational domains."""
        operator, store = _fresh_system()

        # People
        alice = Entity(
            id="p-alice",
            entity_type=EntityType.PERSON,
            name="Alice",
            attributes={"role": "SRE Lead"},
        )
        bob = Entity(
            id="p-bob",
            entity_type=EntityType.PERSON,
            name="Bob",
            attributes={"role": "Backend Engineer"},
        )

        # Organizations
        infra_team = Entity(
            id="o-infra", entity_type=EntityType.ORGANIZATION, name="Infrastructure"
        )

        # Services & tools
        payment_svc = Entity(id="s-pay", entity_type=EntityType.SERVICE, name="Payment Service")
        grafana = Entity(id="t-graf", entity_type=EntityType.TOOL, name="Grafana")

        # Locations & concepts
        sf_office = Entity(id="l-sf", entity_type=EntityType.PLACE, name="SF Office")
        sla_policy = Entity(id="c-sla", entity_type=EntityType.POLICY, name="99.9% SLA")

        for e in [alice, bob, infra_team, payment_svc, grafana, sf_office, sla_policy]:
            store.add_entity(e)

        facts = [
            ("Alice leads the infrastructure team", alice, RelationType.MANAGES, infra_team, 0.95),
            ("Bob works at SF Office", bob, RelationType.LOCATED_IN, sf_office, 0.88),
            (
                "Infrastructure team owns Payment Service",
                infra_team,
                RelationType.OWNS,
                payment_svc,
                0.92,
            ),
            (
                "Payment Service depends on Grafana for monitoring",
                payment_svc,
                RelationType.DEPENDS_ON,
                grafana,
                0.90,
            ),
            (
                "Payment Service is governed by 99.9% SLA policy",
                payment_svc,
                RelationType.AFFILIATED_WITH,
                sla_policy,
                0.87,
            ),
            ("Bob is a member of Infrastructure", bob, RelationType.MEMBER_OF, infra_team, 0.91),
        ]

        for text, subject, relation, obj, conf in facts:
            interaction = Interaction(actor=ActorType.AGENT, raw_text=text)
            store.add_interaction(interaction)
            candidate = CandidateFact(
                source_interaction_id=interaction.id,
                subject_entity_id=subject.id,
                relation=relation,
                object_entity_id=obj.id,
                confidence=conf,
            )
            result = operator.ground(candidate)
            assert result.is_success, f"Failed: {text} — {result.get_explanation()}"

        all_facts = store.get_all_validated_facts()
        assert len(all_facts) == 6

        # Print
        print("\n" + "=" * 70)
        print("🌐 MULTI-DOMAIN KNOWLEDGE GRAPH")
        print("=" * 70)

        # Entity summary
        entity_types = {}
        for e in store.get_all_entities():
            entity_types.setdefault(e.entity_type.value, []).append(e.name)

        for etype, names in sorted(entity_types.items()):
            print(f"   [{etype:15s}] {', '.join(names)}")

        print(f"\n   Facts ({len(all_facts)}):")
        for f in all_facts:
            s = store.get_entity(f.subject_id)
            o = store.get_entity(f.object_id)
            print(f"     {s.name} —[{f.relation.value}]→ {o.name}")
        print("=" * 70)


# =============================================================================
# Scenario 5: Conflict Resolution Audit
# =============================================================================


class TestConflictResolutionAudit:
    """Verify conflict resolutions produce audit metadata."""

    def test_supersession_includes_resolution_metadata(self, capsys):
        """Supersession should carry conflict resolution reasoning."""
        operator, store = _fresh_system(strategy=ConflictResolutionStrategy.COMPOSITE)

        alice = Entity(id="p-alice", entity_type=EntityType.PERSON, name="Alice")
        ny = Entity(id="l-ny", entity_type=EntityType.PLACE, name="New York")
        london = Entity(id="l-london", entity_type=EntityType.PLACE, name="London")
        store.add_entity(alice)
        store.add_entity(ny)
        store.add_entity(london)

        # Fact 1: Alice lives in NY
        i1 = Interaction(actor=ActorType.USER, raw_text="Alice lives in New York")
        store.add_interaction(i1)
        c1 = CandidateFact(
            source_interaction_id=i1.id,
            subject_entity_id=alice.id,
            relation=RelationType.LOCATED_IN,
            object_entity_id=ny.id,
            confidence=0.80,
            attributes={"key": "location"},
        )
        r1 = operator.ground(c1)
        assert r1.is_success

        # Fact 2: Alice moved to London (higher confidence)
        i2 = Interaction(actor=ActorType.SYSTEM, raw_text="Alice relocated to London")
        store.add_interaction(i2)
        c2 = CandidateFact(
            source_interaction_id=i2.id,
            subject_entity_id=alice.id,
            relation=RelationType.LOCATED_IN,
            object_entity_id=london.id,
            confidence=0.95,
            attributes={"source": "system", "key": "location"},
        )
        r2 = operator.ground(c2)

        assert r2.decision == GroundingDecision.SUPERSEDED
        assert len(r2.conflict_resolutions) > 0

        resolution = r2.conflict_resolutions[0]
        print("\n" + "=" * 70)
        print("⚖️  CONFLICT RESOLUTION AUDIT")
        print("=" * 70)
        print(f"   Strategy: {resolution['strategy']}")
        print(f"   Should supersede: {resolution['should_supersede']}")
        print(f"   Reasoning: {resolution['reasoning']}")
        print(f"   Scores: {resolution['scores']}")
        print("=" * 70)

        assert "strategy" in resolution
        assert "reasoning" in resolution
        assert len(resolution["reasoning"]) > 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
