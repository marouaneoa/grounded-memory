"""
Grounding Operator (Γ)

The execution engine of the Grounded Memory System. Unlike traditional
"Read/Write" operations, memory formation is a conditional state transition.

For a proposed fact f̂, the operator computes:
    Γ(f̂, K) = 1(Valid) if ∀c ∈ C : c(f̂, K) = True
              0(Rejected) otherwise

where K is the existing knowledge state.

Key responsibilities:
1. Orchestrate constraint validation
2. Handle temporal supersession
3. Create ValidatedFacts from approved CandidateFacts
4. Generate rejection records with explanations

Supports both sync (in-memory) and async (PostgreSQL) stores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

from grounded_memory.core.conflict_resolution import (
    ConflictResolutionStrategy,
    ConflictResolver,
)
from grounded_memory.core.constraints import (
    ConstraintValidator,
    ValidationResult,
)
from grounded_memory.core.models import (
    CandidateFact,
    CandidateFactStatus,
    RejectionRecord,
    ValidatedFact,
)
from grounded_memory.core.tuple_normalization import (
    build_fact_semantic_key,
    fact_values_equal,
)

if TYPE_CHECKING:
    from grounded_memory.core.postgres_store import PostgresKnowledgeStore
    from grounded_memory.core.store import MemoryStore


# =============================================================================
# Grounding Result
# =============================================================================


class GroundingDecision(str, Enum):
    """Outcome of the grounding operation."""

    APPROVED = "approved"  # Fact is valid, stored in memory
    REJECTED = "rejected"  # Fact violates constraints
    SUPERSEDED = "superseded"  # Fact approved but supersedes existing fact
    DUPLICATE = "duplicate"  # Fact already exists in memory


@dataclass
class GroundingResult:
    """
    Result of applying the Grounding Operator to a CandidateFact.

    Attributes:
        decision: The outcome (approved, rejected, etc.)
        candidate_fact: The original CandidateFact
        validated_fact: The created ValidatedFact (if approved)
        rejection_record: Details of rejection (if rejected)
        superseded_facts: List of facts that were superseded
        validation_result: Full validation details
    """

    decision: GroundingDecision
    candidate_fact: CandidateFact
    validated_fact: ValidatedFact | None = None
    rejection_record: RejectionRecord | None = None
    superseded_facts: list[ValidatedFact] = field(default_factory=list)
    validation_result: ValidationResult | None = None
    conflict_resolutions: list[dict] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_success(self) -> bool:
        """Check if the fact was successfully grounded."""
        return self.decision in (GroundingDecision.APPROVED, GroundingDecision.SUPERSEDED)

    def get_explanation(self) -> str:
        """Get a human-readable explanation of the result."""
        if self.decision == GroundingDecision.APPROVED:
            return "Fact validated and stored successfully."

        elif self.decision == GroundingDecision.SUPERSEDED:
            superseded_ids = [f.id[:8] for f in self.superseded_facts]
            return f"Fact validated. Superseded {len(self.superseded_facts)} previous fact(s): {superseded_ids}"

        elif self.decision == GroundingDecision.REJECTED:
            if self.rejection_record:
                msg = f"REJECTED: {self.rejection_record.reason}"
                if self.rejection_record.domain_reasoning:
                    msg += f"\nDomain reasoning: {self.rejection_record.domain_reasoning}"
                if self.rejection_record.alternatives:
                    msg += f"\nAlternatives: {', '.join(self.rejection_record.alternatives)}"
                return msg
            return "Fact rejected due to constraint violations."

        elif self.decision == GroundingDecision.DUPLICATE:
            return "Fact already exists in memory."

        return "Unknown grounding result."


# =============================================================================
# Grounding Operator
# =============================================================================


class GroundingOperator:
    """
    The Grounding Operator (Γ) - core execution engine for memory formation.

    This operator ensures that only validated facts enter long-term memory.
    It orchestrates:
    1. Constraint validation against all registered rules
    2. Temporal supersession when new facts update existing knowledge
    3. Creation of ValidatedFacts with proper provenance
    4. Generation of rejection records with explanations

    Usage:
        operator = GroundingOperator(validator, memory_store)
        result = operator.ground(candidate_fact)

        if result.is_success:
            print(f"Stored: {result.validated_fact.id}")
        else:
            print(f"Rejected: {result.rejection_record.reason}")
    """

    def __init__(
        self,
        validator: ConstraintValidator,
        memory_store: MemoryStore,
        auto_supersede: bool = True,
        conflict_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.COMPOSITE,
        conflict_resolver: ConflictResolver | None = None,
    ):
        """
        Initialize the Grounding Operator.

        Args:
            validator: ConstraintValidator with registered constraints
            memory_store: The memory store for persistence
            auto_supersede: Whether to automatically supersede conflicting facts
            conflict_strategy: Strategy for resolving fact conflicts
            conflict_resolver: Optional pre-configured ConflictResolver
        """
        self.validator = validator
        self.memory_store = memory_store
        self.auto_supersede = auto_supersede
        self.conflict_resolver = conflict_resolver or ConflictResolver(
            strategy=conflict_strategy,
        )

    def ground(self, candidate: CandidateFact) -> GroundingResult:
        """
        Apply the grounding operator to a CandidateFact.

        This is the main entry point. It:
        1. Checks for duplicates
        2. Validates against all constraints
        3. Handles supersession if needed
        4. Persists the ValidatedFact or creates a RejectionRecord

        Args:
            candidate: The CandidateFact to ground

        Returns:
            GroundingResult with the outcome
        """
        # Step 1: Check for duplicates
        if self._is_duplicate(candidate):
            return GroundingResult(
                decision=GroundingDecision.DUPLICATE,
                candidate_fact=candidate,
            )

        # Step 2: Validate against constraints
        validation_result = self.validator.validate(
            candidate=candidate,
            knowledge_state=self.memory_store,
        )

        # Step 3: Handle rejection
        if not validation_result.is_valid:
            rejection_record = validation_result.to_rejection_record()

            # Store the rejection for audit
            if rejection_record:
                rejection_record.subject_entity_id = candidate.subject_entity_id
                self.memory_store.add_rejection(rejection_record)

            # Update candidate status
            candidate.status = CandidateFactStatus.REJECTED

            return GroundingResult(
                decision=GroundingDecision.REJECTED,
                candidate_fact=candidate,
                rejection_record=rejection_record,
                validation_result=validation_result,
            )

        # Step 4: Check for supersession
        superseded_facts = []
        conflict_resolutions = []
        if self.auto_supersede:
            superseded_facts, conflict_resolutions = self._find_and_supersede(candidate)

        # Step 5: Create ValidatedFact
        validated_fact = self._create_validated_fact(candidate)

        if conflict_resolutions:
            validated_fact.source_metadata["conflict_resolutions"] = conflict_resolutions

        # Step 6: Persist
        self.memory_store.add_validated_fact(validated_fact)

        # Determine decision type
        decision = GroundingDecision.SUPERSEDED if superseded_facts else GroundingDecision.APPROVED

        return GroundingResult(
            decision=decision,
            candidate_fact=candidate,
            validated_fact=validated_fact,
            superseded_facts=superseded_facts,
            validation_result=validation_result,
            conflict_resolutions=conflict_resolutions,
        )

    def ground_batch(
        self,
        candidates: list[CandidateFact],
    ) -> list[GroundingResult]:
        """
        Ground multiple CandidateFacts.

        Facts are processed sequentially to ensure consistency.
        Each fact is validated against the state that includes
        previously approved facts from this batch.
        """
        results = []
        for candidate in candidates:
            result = self.ground(candidate)
            results.append(result)
        return results

    def _is_duplicate(self, candidate: CandidateFact) -> bool:
        """Check if this fact already exists in memory."""
        existing = self.memory_store.get_facts_by_relation(
            entity_id=candidate.subject_entity_id,
            relation=candidate.relation,
            as_subject=True,
        )

        for fact in existing:
            if not fact.is_active:
                continue

            # Duplicate means same tuple payload after canonical value normalization AND same attributes.
            if (
                fact.object_id == candidate.object_entity_id
                and fact_values_equal(fact.value, candidate.value)
                and fact.attributes == candidate.attributes
            ):
                return True

        return False

    def _find_and_supersede(
        self, candidate: CandidateFact
    ) -> tuple[list[ValidatedFact], list[dict]]:
        """
        Find and supersede facts that the new fact replaces.

        Supersession happens when:
        1. Same subject-relation-object triple (update)
        2. Same subject-relation-value tuple (attribute update)
        3. Conflict resolver determines the candidate should win
        """
        superseded = []
        resolutions = []

        # Find active facts with same subject and relation
        existing = self.memory_store.get_facts_by_relation(
            entity_id=candidate.subject_entity_id,
            relation=candidate.relation,
            as_subject=True,
        )

        for fact in existing:
            if not fact.is_active:
                continue

            # Check if this is the same relationship that should be superseded
            if self._should_supersede(fact, candidate):
                # Use conflict resolver to decide the winner
                resolution = self.conflict_resolver.resolve(fact, candidate)
                resolutions.append(resolution.as_dict())

                if resolution.should_supersede:
                    self.memory_store.supersede_fact(
                        fact_id=fact.id,
                        superseded_by=candidate.id,
                        valid_to=candidate.extracted_at,
                    )
                    superseded.append(fact)

        return superseded, resolutions

    def _should_supersede(
        self,
        existing: ValidatedFact,
        candidate: CandidateFact,
    ) -> bool:
        """
        Determine if a candidate should supersede an existing fact.

        This implements the temporal supersession logic based on
        the relation type and fact attributes.

        This is intentionally relation-agnostic in core memory behavior.
        """
        # Same canonical tuple payload = existing fact should be superseded.
        if existing.object_id == candidate.object_entity_id and fact_values_equal(
            existing.value, candidate.value
        ):
            return True

        # Slot-level semantic key captures keyed-attribute updates robustly.
        existing_key = self._fact_semantic_key(
            existing.relation,
            existing.object_id,
            existing.value,
            existing.attributes,
        )
        candidate_key = self._fact_semantic_key(
            candidate.relation,
            candidate.object_entity_id,
            candidate.value,
            candidate.attributes,
        )
        return bool(existing_key and candidate_key and existing_key == candidate_key)

    @staticmethod
    def _fact_semantic_key(
        relation,
        object_id: str | None,
        value: str | None,
        attributes: dict[str, object] | None,
    ) -> str | None:
        return build_fact_semantic_key(
            subject_id="grounding",
            relation=relation,
            object_id=object_id,
            value=value,
            attributes=attributes,
            include_subject=False,
        )

    def _create_validated_fact(self, candidate: CandidateFact) -> ValidatedFact:
        """Create a ValidatedFact from an approved CandidateFact."""
        # Retrieve the source interaction to carry forward the raw text
        source_text = None
        source_metadata: dict = {}
        interaction = self.memory_store.get_interaction(candidate.source_interaction_id)
        if interaction is not None:
            source_text = interaction.raw_text
            source_metadata = {
                "actor": interaction.actor.value,
                "interaction_timestamp": interaction.timestamp.isoformat(),
                "session_id": getattr(interaction, "session_id", None),
            }

        return ValidatedFact(
            candidate_fact_id=candidate.id,
            source_interaction_id=candidate.source_interaction_id,
            subject_id=candidate.subject_entity_id,
            relation=candidate.relation,
            object_id=candidate.object_entity_id,
            value=candidate.value,
            valid_from=candidate.extracted_at,  # Use extracted_at as the start of validity
            valid_to=None,  # Active until superseded
            confidence=candidate.confidence,
            attributes=candidate.attributes.copy() if candidate.attributes else {},
            source_text=source_text,
            source_metadata=source_metadata,
        )


# =============================================================================
# Batch Processing Utilities
# =============================================================================


def process_interaction_facts(
    operator: GroundingOperator,
    candidates: list[CandidateFact],
    interaction_id: str,
) -> tuple[list[ValidatedFact], list[RejectionRecord]]:
    """
    Process all candidate facts from a single interaction.

    Returns:
        (approved_facts, rejections)
    """
    approved = []
    rejections = []

    for candidate in candidates:
        # Ensure interaction ID is set
        if candidate.source_interaction_id != interaction_id:
            candidate.source_interaction_id = interaction_id

        result = operator.ground(candidate)

        if result.is_success and result.validated_fact:
            approved.append(result.validated_fact)
        elif result.rejection_record:
            rejections.append(result.rejection_record)

    return approved, rejections


# =============================================================================
# Async Grounding Operator (for PostgreSQL store)
# =============================================================================


class AsyncGroundingOperator:
    """
    Async version of the Grounding Operator for database-backed stores.

    Works with PostgresKnowledgeStore to perform constraint validation
    and store validated facts or rejection records.

    Usage:
        operator = AsyncGroundingOperator(validator, postgres_store)
        result = await operator.ground(candidate_fact)

        if result.is_success:
            print(f"Stored: {result.validated_fact.id}")
        else:
            print(f"Rejected: {result.rejection_record.reason}")
    """

    def __init__(
        self,
        validator: ConstraintValidator,
        store: PostgresKnowledgeStore,
        auto_supersede: bool = True,
    ):
        """
        Initialize the Async Grounding Operator.

        Args:
            validator: ConstraintValidator with registered constraints
            store: PostgresKnowledgeStore for persistence
            auto_supersede: Whether to automatically supersede conflicting facts
        """
        self.validator = validator
        self.store = store
        self.auto_supersede = auto_supersede
        self.conflict_resolver = ConflictResolver(
            strategy=ConflictResolutionStrategy.COMPOSITE,
        )

    async def ground(self, candidate: CandidateFact) -> GroundingResult:
        """
        Apply the grounding operator to a CandidateFact (async).

        This is the main entry point. It:
        1. Checks for duplicates
        2. Validates against all constraints
        3. Handles supersession if needed
        4. Persists the ValidatedFact or creates a RejectionRecord

        Args:
            candidate: The CandidateFact to ground

        Returns:
            GroundingResult with the outcome
        """
        # Step 1: Check for duplicates
        if await self._is_duplicate(candidate):
            return GroundingResult(
                decision=GroundingDecision.DUPLICATE,
                candidate_fact=candidate,
            )

        # Step 2: Validate against constraints
        # Note: The validator itself is sync, but uses in-memory data
        validation_result = self.validator.validate(
            candidate=candidate,
            knowledge_state=self.store,
        )

        # Step 3: Handle rejection
        if not validation_result.is_valid:
            rejection_record = validation_result.to_rejection_record()

            # Store the rejection for audit
            if rejection_record:
                await self.store.reject_candidate_with_record(
                    fact_id=candidate.id,
                    reason=rejection_record.reason,
                    violated_constraints=[v.constraint_name for v in validation_result.violations],
                    domain_reasoning=rejection_record.domain_reasoning,
                    alternatives=rejection_record.alternatives,
                )

            # Update candidate status
            candidate.status = CandidateFactStatus.REJECTED

            return GroundingResult(
                decision=GroundingDecision.REJECTED,
                candidate_fact=candidate,
                rejection_record=rejection_record,
                validation_result=validation_result,
            )

        # Step 4: Check for supersession
        superseded_facts = []
        conflict_resolutions = []
        if self.auto_supersede:
            superseded_facts, conflict_resolutions = await self._find_and_supersede(candidate)

        # Step 5: Promote to ValidatedFact
        validated_fact = await self.store.promote_candidate_to_validated(candidate.id)

        if validated_fact is None:
            # Promotion failed - shouldn't happen but handle gracefully
            return GroundingResult(
                decision=GroundingDecision.REJECTED,
                candidate_fact=candidate,
            )

        if conflict_resolutions and validated_fact.source_metadata is not None:
            validated_fact.source_metadata["conflict_resolutions"] = conflict_resolutions
            # TODO: Add a store method to update source_metadata in postgres if necessary

        # Update candidate status
        candidate.status = CandidateFactStatus.ACCEPTED

        # Determine decision type
        decision = GroundingDecision.SUPERSEDED if superseded_facts else GroundingDecision.APPROVED

        return GroundingResult(
            decision=decision,
            candidate_fact=candidate,
            validated_fact=validated_fact,
            superseded_facts=superseded_facts,
            validation_result=validation_result,
            conflict_resolutions=conflict_resolutions,
        )

    async def ground_batch(
        self,
        candidates: list[CandidateFact],
    ) -> list[GroundingResult]:
        """
        Ground multiple CandidateFacts (async).

        Facts are processed sequentially to ensure consistency.
        Each fact is validated against the state that includes
        previously approved facts from this batch.
        """
        results = []
        for candidate in candidates:
            result = await self.ground(candidate)
            results.append(result)
        return results

    async def _is_duplicate(self, candidate: CandidateFact) -> bool:
        """Check if this fact already exists in memory."""
        existing = await self.store.get_facts_by_relation(
            entity_id=candidate.subject_entity_id,
            relation=candidate.relation,
            as_subject=True,
        )

        for fact in existing:
            if not fact.is_active:
                continue

            if fact.object_id == candidate.object_entity_id and fact_values_equal(
                fact.value, candidate.value
            ):
                return True

        return False

    async def _find_and_supersede(
        self, candidate: CandidateFact
    ) -> tuple[list[ValidatedFact], list[dict]]:
        """
        Find and supersede facts that the new fact replaces.

        Supersession happens when a new fact updates knowledge about
        the same subject-relation-object triple (or subject-relation
        if the relation is functional).
        """
        superseded = []
        resolutions = []

        # Find active facts with same subject and relation
        existing = await self.store.get_facts_by_relation(
            entity_id=candidate.subject_entity_id,
            relation=candidate.relation,
            as_subject=True,
        )

        for fact in existing:
            if not fact.is_active:
                continue

            # Check if this is the same relationship that should be superseded
            if self._should_supersede(fact, candidate):
                resolution = self.conflict_resolver.resolve(fact, candidate)
                resolutions.append(resolution.as_dict())

                if resolution.should_supersede:
                    # Supersede by setting valid_to
                    await self.store.supersede_fact(
                        fact_id=fact.id,
                        superseded_by=candidate.id,
                        valid_to=candidate.extracted_at,
                    )
                    superseded.append(fact)

        return superseded, resolutions

    def _should_supersede(
        self,
        existing: ValidatedFact,
        candidate: CandidateFact,
    ) -> bool:
        """
        Determine if a candidate should supersede an existing fact.
        """
        if existing.object_id == candidate.object_entity_id and fact_values_equal(
            existing.value, candidate.value
        ):
            return True

        existing_key = self._fact_semantic_key(
            existing.relation,
            existing.object_id,
            existing.value,
            existing.attributes,
        )
        candidate_key = self._fact_semantic_key(
            candidate.relation,
            candidate.object_entity_id,
            candidate.value,
            candidate.attributes,
        )
        return bool(existing_key and candidate_key and existing_key == candidate_key)

    @staticmethod
    def _fact_semantic_key(
        relation,
        object_id: str | None,
        value: str | None,
        attributes: dict[str, object] | None,
    ) -> str | None:
        return build_fact_semantic_key(
            subject_id="grounding",
            relation=relation,
            object_id=object_id,
            value=value,
            attributes=attributes,
            include_subject=False,
        )


async def process_interaction_facts_async(
    operator: AsyncGroundingOperator,
    candidates: list[CandidateFact],
    interaction_id: str,
) -> tuple[list[ValidatedFact], list[RejectionRecord]]:
    """
    Process all candidate facts from a single interaction (async).

    Returns:
        (approved_facts, rejections)
    """
    approved = []
    rejections = []

    for candidate in candidates:
        # Ensure interaction ID is set
        if candidate.source_interaction_id != interaction_id:
            candidate.source_interaction_id = interaction_id

        result = await operator.ground(candidate)

        if result.is_success and result.validated_fact:
            approved.append(result.validated_fact)
        elif result.rejection_record:
            rejections.append(result.rejection_record)

    return approved, rejections
