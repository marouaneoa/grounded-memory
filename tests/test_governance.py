#!/usr/bin/env python3
"""
Governance Verification Tests

Tests for conflict resolution, supersession, constraint enforcement,
and data quality governance — the core of Grounded Memory's value prop.

Run:
    PYTHONPATH=src python -m pytest tests/test_governance.py -v
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grounded_memory.core.conflict_resolution import (
    ConflictResolutionStrategy,
    ConflictResolver,
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
    ValidatedFact,
)
from grounded_memory.core.store import MemoryStore

# =============================================================================
# Helpers
# =============================================================================


def _make_store() -> MemoryStore:
    """Create a clean in-memory store."""
    return MemoryStore()


def _make_operator(
    store: MemoryStore | None = None,
    strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.COMPOSITE,
) -> tuple[GroundingOperator, MemoryStore]:
    """Create a grounding operator with a clean store."""
    store = store or _make_store()
    validator = ConstraintValidator()
    operator = GroundingOperator(
        validator=validator,
        memory_store=store,
        conflict_strategy=strategy,
    )
    return operator, store


def _seed_entities(store: MemoryStore) -> tuple[Entity, Entity, Entity]:
    """Seed user, Python, and Rust entities."""
    user = Entity(id="user-1", entity_type=EntityType.PERSON, name="Alice")
    python = Entity(id="lang-python", entity_type=EntityType.CONCEPT, name="Python")
    rust = Entity(id="lang-rust", entity_type=EntityType.CONCEPT, name="Rust")
    store.add_entity(user)
    store.add_entity(python)
    store.add_entity(rust)
    return user, python, rust


def _add_interaction(
    store: MemoryStore, text: str, actor: ActorType = ActorType.USER
) -> Interaction:
    """Add a test interaction."""
    interaction = Interaction(actor=actor, raw_text=text)
    store.add_interaction(interaction)
    return interaction


# =============================================================================
# Test: Supersession Correctness
# =============================================================================


class TestSupersession:
    """Verify that conflicting facts are correctly superseded."""

    def test_supersede_replaces_old_fact(self):
        """Add fact A, then contradicting fact B → A should be superseded."""
        operator, store = _make_operator()
        user, python, rust = _seed_entities(store)
        interaction = _add_interaction(store, "Alice prefers Python")

        # Fact A: Alice prefers Python
        candidate_a = CandidateFact(
            source_interaction_id=interaction.id,
            subject_entity_id=user.id,
            relation=RelationType.HAS_ATTRIBUTE,
            object_entity_id=python.id,
            value="python",
            confidence=0.85,
            attributes={"key": "prefers"},
        )
        result_a = operator.ground(candidate_a)
        assert result_a.decision == GroundingDecision.APPROVED
        assert result_a.validated_fact is not None
        fact_a_id = result_a.validated_fact.id

        # Fact B: Alice now prefers Rust (same key → supersedes A)
        interaction_b = _add_interaction(store, "Alice now prefers Rust")
        candidate_b = CandidateFact(
            source_interaction_id=interaction_b.id,
            subject_entity_id=user.id,
            relation=RelationType.HAS_ATTRIBUTE,
            object_entity_id=rust.id,
            value="rust",
            confidence=0.90,
            attributes={"key": "prefers"},
        )
        result_b = operator.ground(candidate_b)

        assert result_b.decision == GroundingDecision.SUPERSEDED
        assert len(result_b.superseded_facts) == 1
        assert result_b.superseded_facts[0].id == fact_a_id

        # Verify old fact is no longer active
        old_fact = store.get_validated_fact(fact_a_id)
        assert old_fact is not None
        assert old_fact.valid_to is not None, "Superseded fact must have valid_to set"
        assert old_fact.superseded_by is not None, "Superseded fact must have superseded_by set"

        # New fact should be active
        assert result_b.validated_fact.is_active

    def test_superseded_fact_has_temporal_boundary(self):
        """Superseded facts must have valid_to timestamp set."""
        operator, store = _make_operator()
        user, python, rust = _seed_entities(store)
        interaction = _add_interaction(store, "Temporal test")

        # First fact
        c1 = CandidateFact(
            source_interaction_id=interaction.id,
            subject_entity_id=user.id,
            relation=RelationType.HAS_ATTRIBUTE,
            object_entity_id=python.id,
            value="python",
            confidence=0.9,
            attributes={"key": "language"},
        )
        r1 = operator.ground(c1)
        assert r1.is_success

        # Superseding fact
        i2 = _add_interaction(store, "Change language")
        c2 = CandidateFact(
            source_interaction_id=i2.id,
            subject_entity_id=user.id,
            relation=RelationType.HAS_ATTRIBUTE,
            object_entity_id=rust.id,
            value="rust",
            confidence=0.95,
            attributes={"key": "language"},
        )
        r2 = operator.ground(c2)

        assert r2.decision == GroundingDecision.SUPERSEDED
        old = store.get_validated_fact(r1.validated_fact.id)
        assert old.valid_to is not None
        # valid_to should be recent (within last 5 seconds)
        assert (datetime.now(timezone.utc) - old.valid_to).total_seconds() < 5


# =============================================================================
# Test: Duplicate Detection
# =============================================================================


class TestDuplicateDetection:
    """Verify that identical facts are detected as duplicates."""

    def test_exact_duplicate_detected(self):
        """Same fact twice → second is marked DUPLICATE."""
        operator, store = _make_operator()
        user, python, _ = _seed_entities(store)
        interaction = _add_interaction(store, "Duplicate test")

        candidate = CandidateFact(
            source_interaction_id=interaction.id,
            subject_entity_id=user.id,
            relation=RelationType.HAS_ATTRIBUTE,
            object_entity_id=python.id,
            value="python",
            confidence=0.9,
            attributes={"key": "language"},
        )

        # First insertion
        r1 = operator.ground(candidate)
        assert r1.decision == GroundingDecision.APPROVED

        # Second identical insertion
        r2 = operator.ground(candidate)
        assert r2.decision == GroundingDecision.DUPLICATE


# =============================================================================
# Test: Conflict Resolution Strategies
# =============================================================================


class TestConflictResolution:
    """Verify multi-signal conflict resolution."""

    def test_confidence_wins_strategy(self):
        """Higher confidence fact should win."""
        resolver = ConflictResolver(
            strategy=ConflictResolutionStrategy.CONFIDENCE_WINS,
        )
        existing = ValidatedFact(
            candidate_fact_id="cf-old",
            source_interaction_id="int-old",
            subject_id="user-1",
            relation=RelationType.HAS_ATTRIBUTE,
            object_id="lang-python",
            value="python",
            valid_from=datetime.now(timezone.utc) - timedelta(hours=1),
            confidence=0.7,
            attributes={"key": "prefers"},
        )
        candidate = CandidateFact(
            source_interaction_id="int-new",
            subject_entity_id="user-1",
            relation=RelationType.HAS_ATTRIBUTE,
            object_entity_id="lang-rust",
            value="rust",
            confidence=0.95,
            attributes={"key": "prefers"},
        )

        resolution = resolver.resolve(existing, candidate)
        assert resolution.should_supersede is True
        assert resolution.strategy_used == ConflictResolutionStrategy.CONFIDENCE_WINS
        assert "exceeds" in resolution.reasoning

    def test_confidence_wins_rejects_lower(self):
        """Lower confidence candidate should NOT supersede."""
        resolver = ConflictResolver(
            strategy=ConflictResolutionStrategy.CONFIDENCE_WINS,
        )
        existing = ValidatedFact(
            candidate_fact_id="cf-old",
            source_interaction_id="int-old",
            subject_id="user-1",
            relation=RelationType.HAS_ATTRIBUTE,
            value="expert-answer",
            valid_from=datetime.now(timezone.utc),
            confidence=0.95,
        )
        candidate = CandidateFact(
            source_interaction_id="int-new",
            subject_entity_id="user-1",
            relation=RelationType.HAS_ATTRIBUTE,
            value="weak-answer",
            confidence=0.5,
        )

        resolution = resolver.resolve(existing, candidate)
        assert resolution.should_supersede is False

    def test_recency_wins_strategy(self):
        """More recent fact should win."""
        resolver = ConflictResolver(
            strategy=ConflictResolutionStrategy.RECENCY_WINS,
        )
        existing = ValidatedFact(
            candidate_fact_id="cf-old",
            source_interaction_id="int-old",
            subject_id="user-1",
            relation=RelationType.HAS_ATTRIBUTE,
            value="old-value",
            valid_from=datetime.now(timezone.utc) - timedelta(days=30),
            confidence=0.9,
        )
        candidate = CandidateFact(
            source_interaction_id="int-new",
            subject_entity_id="user-1",
            relation=RelationType.HAS_ATTRIBUTE,
            value="new-value",
            confidence=0.9,
        )

        resolution = resolver.resolve(existing, candidate)
        assert resolution.should_supersede is True
        assert resolution.strategy_used == ConflictResolutionStrategy.RECENCY_WINS

    def test_source_priority_strategy(self):
        """System source should beat user source."""
        resolver = ConflictResolver(
            strategy=ConflictResolutionStrategy.SOURCE_PRIORITY,
        )
        existing = ValidatedFact(
            candidate_fact_id="cf-old",
            source_interaction_id="int-old",
            subject_id="user-1",
            relation=RelationType.HAS_ATTRIBUTE,
            value="user-said",
            valid_from=datetime.now(timezone.utc) - timedelta(hours=1),
            confidence=0.9,
            attributes={"source": "user"},
        )
        candidate = CandidateFact(
            source_interaction_id="int-new",
            subject_entity_id="user-1",
            relation=RelationType.HAS_ATTRIBUTE,
            value="system-said",
            confidence=0.9,
            attributes={"source": "system"},
        )

        resolution = resolver.resolve(existing, candidate)
        assert resolution.should_supersede is True
        assert resolution.strategy_used == ConflictResolutionStrategy.SOURCE_PRIORITY

    def test_composite_strategy(self):
        """Composite strategy uses all signals."""
        resolver = ConflictResolver(
            strategy=ConflictResolutionStrategy.COMPOSITE,
        )
        existing = ValidatedFact(
            candidate_fact_id="cf-old",
            source_interaction_id="int-old",
            subject_id="user-1",
            relation=RelationType.HAS_ATTRIBUTE,
            value="old",
            valid_from=datetime.now(timezone.utc) - timedelta(days=7),
            confidence=0.6,
            attributes={"source": "user"},
        )
        candidate = CandidateFact(
            source_interaction_id="int-new",
            subject_entity_id="user-1",
            relation=RelationType.HAS_ATTRIBUTE,
            value="new",
            confidence=0.95,
            attributes={"source": "system"},
        )

        resolution = resolver.resolve(existing, candidate)
        assert resolution.should_supersede is True
        assert resolution.strategy_used == ConflictResolutionStrategy.COMPOSITE
        assert "composite" in resolution.scores

    def test_conflict_resolution_audit_trail(self):
        """Resolution should include audit metadata."""
        resolver = ConflictResolver(strategy=ConflictResolutionStrategy.COMPOSITE)
        existing = ValidatedFact(
            candidate_fact_id="cf-old",
            source_interaction_id="int-old",
            subject_id="user-1",
            relation=RelationType.HAS_ATTRIBUTE,
            value="old",
            valid_from=datetime.now(timezone.utc),
            confidence=0.8,
        )
        candidate = CandidateFact(
            source_interaction_id="int-new",
            subject_entity_id="user-1",
            relation=RelationType.HAS_ATTRIBUTE,
            value="new",
            confidence=0.9,
        )

        resolution = resolver.resolve(existing, candidate)
        audit = resolution.as_dict()

        assert "should_supersede" in audit
        assert "strategy" in audit
        assert "reasoning" in audit
        assert "scores" in audit
        assert len(audit["reasoning"]) > 0


# =============================================================================
# Test: Constraint Enforcement
# =============================================================================


class _ConfidenceFloorConstraint(BaseConstraintEvaluator):
    """Test constraint: reject facts below confidence 0.5."""

    @property
    def constraint_id(self) -> str:
        return "test_confidence_floor"

    @property
    def constraint_name(self) -> str:
        return "Confidence Floor"

    @property
    def description(self) -> str:
        return "Reject facts with confidence below 0.5"

    def evaluate(self, candidate, knowledge_state) -> ConstraintViolation | None:
        if candidate.confidence < 0.5:
            return ConstraintViolation(
                constraint_id=self.constraint_id,
                constraint_name=self.constraint_name,
                description=f"Confidence {candidate.confidence} below minimum 0.5",
                severity="error",
            )
        return None


class TestConstraintEnforcement:
    """Verify that constraints properly reject invalid facts."""

    def test_low_confidence_rejected(self):
        """Facts with low confidence should be rejected."""
        store = _make_store()
        validator = ConstraintValidator()
        validator.register(_ConfidenceFloorConstraint())
        operator = GroundingOperator(validator=validator, memory_store=store)

        user = Entity(id="user-1", entity_type=EntityType.PERSON, name="Alice")
        store.add_entity(user)
        interaction = _add_interaction(store, "Low confidence test")

        candidate = CandidateFact(
            source_interaction_id=interaction.id,
            subject_entity_id=user.id,
            relation=RelationType.HAS_ATTRIBUTE,
            value="uncertain-info",
            confidence=0.3,
        )

        result = operator.ground(candidate)
        assert result.decision == GroundingDecision.REJECTED
        assert result.rejection_record is not None
        assert "below minimum" in result.rejection_record.reason

    def test_high_confidence_accepted(self):
        """Facts with sufficient confidence should pass."""
        store = _make_store()
        validator = ConstraintValidator()
        validator.register(_ConfidenceFloorConstraint())
        operator = GroundingOperator(validator=validator, memory_store=store)

        user = Entity(id="user-1", entity_type=EntityType.PERSON, name="Alice")
        store.add_entity(user)
        interaction = _add_interaction(store, "High confidence test")

        candidate = CandidateFact(
            source_interaction_id=interaction.id,
            subject_entity_id=user.id,
            relation=RelationType.HAS_ATTRIBUTE,
            value="confident-info",
            confidence=0.85,
        )

        result = operator.ground(candidate)
        assert result.decision == GroundingDecision.APPROVED


# =============================================================================
# Test: Source Text Propagation
# =============================================================================


class TestSourceTextPropagation:
    """Verify that source text is carried from Interaction to ValidatedFact."""

    def test_source_text_on_validated_fact(self):
        """ValidatedFact should carry the original interaction text."""
        operator, store = _make_operator()
        user, python, _ = _seed_entities(store)

        raw_text = "Alice prefers Python for machine learning projects"
        interaction = _add_interaction(store, raw_text)

        candidate = CandidateFact(
            source_interaction_id=interaction.id,
            subject_entity_id=user.id,
            relation=RelationType.HAS_ATTRIBUTE,
            object_entity_id=python.id,
            value="python",
            confidence=0.9,
            attributes={"key": "prefers"},
        )
        result = operator.ground(candidate)

        assert result.is_success
        fact = result.validated_fact
        assert fact.source_text == raw_text
        assert fact.source_metadata.get("actor") == "user"

    def test_source_metadata_has_actor(self):
        """Source metadata should include the actor type."""
        operator, store = _make_operator()
        user, python, _ = _seed_entities(store)
        interaction = _add_interaction(store, "System check", actor=ActorType.SYSTEM)

        candidate = CandidateFact(
            source_interaction_id=interaction.id,
            subject_entity_id=user.id,
            relation=RelationType.HAS_ATTRIBUTE,
            object_entity_id=python.id,
            value="python",
            confidence=0.9,
            attributes={"key": "uses"},
        )
        result = operator.ground(candidate)

        assert result.is_success
        assert result.validated_fact.source_metadata["actor"] == "system"


# =============================================================================
# Test: Grounding Result Explanation
# =============================================================================


class TestGroundingExplanation:
    """Verify that grounding results provide clear explanations."""

    def test_approved_explanation(self):
        operator, store = _make_operator()
        user, python, _ = _seed_entities(store)
        interaction = _add_interaction(store, "Test")
        candidate = CandidateFact(
            source_interaction_id=interaction.id,
            subject_entity_id=user.id,
            relation=RelationType.HAS_ATTRIBUTE,
            object_entity_id=python.id,
            value="python",
            confidence=0.9,
        )
        result = operator.ground(candidate)
        explanation = result.get_explanation()
        assert "validated" in explanation.lower() or "stored" in explanation.lower()

    def test_rejected_explanation_includes_reason(self):
        store = _make_store()
        validator = ConstraintValidator()
        validator.register(_ConfidenceFloorConstraint())
        operator = GroundingOperator(validator=validator, memory_store=store)

        user = Entity(id="user-1", entity_type=EntityType.PERSON, name="Alice")
        store.add_entity(user)
        interaction = _add_interaction(store, "Reject test")

        candidate = CandidateFact(
            source_interaction_id=interaction.id,
            subject_entity_id=user.id,
            relation=RelationType.HAS_ATTRIBUTE,
            value="low-conf",
            confidence=0.2,
        )
        result = operator.ground(candidate)
        explanation = result.get_explanation()
        assert "REJECTED" in explanation


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
