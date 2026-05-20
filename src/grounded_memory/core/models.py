"""
Core Memory Taxonomy: Six-Object Model

This module implements the fundamental memory objects for the Grounded Memory System:

1. Interaction (I): Immutable event log of raw user inputs
2. Entity (E): Symbolic anchors with defined schemas and attributes
3. CandidateFact (f̂): LLM-proposed facts awaiting validation (untrusted)
4. ValidatedFact (F*): System-approved knowledge with temporal boundaries (trusted)
5. Constraint (C): Declarative governance rules
6. AnswerContext (X): Ephemeral query view containing only active, constraint-compliant facts

The memory is treated as a temporal property graph where ValidatedFacts are edges
between Entity nodes.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def as_utc_datetime(value: datetime) -> datetime:
    """Return a timezone-aware UTC datetime for safe comparisons."""
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def datetime_before(left: datetime, right: datetime) -> bool:
    return as_utc_datetime(left) < as_utc_datetime(right)


def datetime_after(left: datetime, right: datetime) -> bool:
    return as_utc_datetime(left) > as_utc_datetime(right)


def datetime_on_or_before(left: datetime, right: datetime) -> bool:
    return as_utc_datetime(left) <= as_utc_datetime(right)


def datetime_on_or_after(left: datetime, right: datetime) -> bool:
    return as_utc_datetime(left) >= as_utc_datetime(right)


# =============================================================================
# Enumerations
# =============================================================================


class CandidateFactStatus(str, Enum):
    """Status of a candidate fact in the validation pipeline."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ActorType(str, Enum):
    """Types of actors in interactions."""

    USER = "user"
    AGENT = "agent"
    TOOL = "tool"
    SYSTEM = "system"


class RelationType(str, Enum):
    """Types of relationships between entities in the knowledge graph."""

    # Legacy domain-oriented relation examples (still supported)
    HAS_ALLERGY = "HAS_ALLERGY"
    HAS_CONDITION = "HAS_CONDITION"
    PRESCRIBED = "PRESCRIBED"
    DISCONTINUED = "DISCONTINUED"
    TREATS = "TREATS"
    CONTAINS_INGREDIENT = "CONTAINS_INGREDIENT"
    SAME_THERAPEUTIC_CLASS = "SAME_THERAPEUTIC_CLASS"
    CONTRAINDICATED_WITH = "CONTRAINDICATED_WITH"

    # General relations
    HAS_ATTRIBUTE = "HAS_ATTRIBUTE"
    RELATED_TO = "RELATED_TO"
    PART_OF = "PART_OF"
    INSTANCE_OF = "INSTANCE_OF"

    # Domain-agnostic relations (generic knowledge graph)
    OWNS = "OWNS"
    WORKS_AT = "WORKS_AT"
    LOCATED_IN = "LOCATED_IN"
    MEMBER_OF = "MEMBER_OF"
    CREATED = "CREATED"
    DEPENDS_ON = "DEPENDS_ON"
    MANAGES = "MANAGES"
    USED_BY = "USED_BY"
    PRODUCED_BY = "PRODUCED_BY"
    AFFILIATED_WITH = "AFFILIATED_WITH"
    REPORTED_BY = "REPORTED_BY"
    APPROVED_BY = "APPROVED_BY"


class EntityType(str, Enum):
    """Types of entities in the knowledge graph."""

    # Legacy domain-oriented entity labels (still supported)
    PATIENT = "patient"
    MEDICATION = "medication"
    CONDITION = "condition"
    ALLERGY = "allergy"
    INGREDIENT = "ingredient"
    THERAPEUTIC_CLASS = "therapeutic_class"
    CLINICIAN = "clinician"
    FACILITY = "facility"

    # Domain-agnostic entity types (generic knowledge graph)
    PERSON = "person"
    PLACE = "place"
    ORGANIZATION = "organization"
    CONCEPT = "concept"
    ASSET = "asset"
    SERVICE = "service"
    EVENT = "event"
    DOCUMENT = "document"
    PROJECT = "project"
    TOOL = "tool"
    METRIC = "metric"
    POLICY = "policy"


class ConstraintType(str, Enum):
    """Types of constraints that can be applied."""

    PROHIBITION = "prohibition"  # Must NOT happen
    REQUIREMENT = "requirement"  # Must happen
    CARDINALITY = "cardinality"  # Limits on count
    TEMPORAL = "temporal"  # Time-based rules
    CONSISTENCY = "consistency"  # Logical consistency


class MemoryDisposition(str, Enum):
    """High-level consolidation intent/outcome for memory writes.

    These labels intentionally avoid CRUD-style naming while preserving the
    same operational semantics used by modern memory systems.
    """

    CAPTURE = "capture"  # introduce a new durable memory
    REFINE = "refine"  # replace or sharpen an existing memory
    RETIRE = "retire"  # deactivate an existing memory
    PASS = "pass"  # intentionally make no memory change


# =============================================================================
# Core Memory Objects
# =============================================================================


class Interaction(BaseModel):
    """
    Immutable event log of raw user/agent interactions.

    This represents the foundational audit trail - what was actually said or done.
    Interactions are never modified after creation.

    Attributes:
        id: Unique identifier for this interaction
        tenant_id: Tenant identifier for multi-tenant isolation
        app_id: Application/workspace identifier
        user_id: ID of the user involved in the interaction
        agent_id: Agent identifier for multi-agent separation
        run_id: Short-lived run/session identifier
        session_id: Session identifier for grouping related interactions
        space_type: Logical memory space (global, user, agent, run)
        actor: Who created this interaction (user, agent, tool, system)
        raw_text: Raw input/output text
        timestamp: When the interaction occurred
        metadata: Additional context (facility, device, etc.)
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str | None = Field(None, description="Tenant identifier")
    app_id: str | None = Field(None, description="Application identifier")
    user_id: str | None = Field(None, description="User identifier")
    agent_id: str | None = Field(None, description="Agent identifier")
    run_id: str | None = Field(None, description="Run identifier")
    session_id: str | None = Field(None, description="Session identifier for grouping")
    space_type: str | None = Field(None, description="Logical memory space")
    actor: ActorType = Field(default=ActorType.USER, description="Who created this interaction")
    raw_text: str = Field(..., description="Raw input/output text")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Legacy compatibility
    @property
    def content(self) -> str:
        """Legacy property - use raw_text instead."""
        return self.raw_text

    @property
    def source(self) -> str:
        """Legacy property - use actor instead."""
        return self.actor.value

    model_config = ConfigDict(frozen=True)  # Immutable after creation


class Entity(BaseModel):
    """
    Symbolic anchor representing a real-world object.

    Entities are nodes in the knowledge graph. They have defined schemas
    and attributes based on their type. Entities provide stable references
    that facts (edges) connect.

    Attributes:
        id: Unique identifier
        entity_type: Category of entity (person, asset, service, etc.)
        name: Human-readable name
        canonical_id: External system ID from upstream systems
        attributes: Type-specific attributes
        created_at: When this entity was first created
        updated_at: Last update timestamp
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entity_type: EntityType
    name: str = Field(..., description="Human-readable name")
    canonical_id: str | None = Field(None, description="External system identifier")
    attributes: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Entity):
            return self.id == other.id
        return False


class CandidateFact(BaseModel):
    """
    Untrusted fact proposed by the LLM awaiting validation.

    CandidateFacts represent the LLM's interpretation of information from
    interactions. They are "proposals" that must pass through the Grounding
    Operator before becoming ValidatedFacts.

    This follows a triplet structure: (subject, relation, object) OR (subject, relation, value)
    for attribute-like facts.

    Attributes:
        id: Unique identifier
        source_interaction_id: Link to the originating Interaction
        subject_entity_id: Entity ID of the subject
        relation: Type of relationship
        object_entity_id: Entity ID of the object (for entity-to-entity relations)
        value: String value (for entity-to-value relations like attributes)
        confidence: LLM's confidence score (0.0-1.0)
        extracted_at: When the LLM extracted this fact
        status: Current status (pending, accepted, rejected)
        rejection_reason: If rejected, the reason why
        attributes: Additional fact attributes
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_interaction_id: str = Field(..., description="ID of source Interaction")
    subject_entity_id: str = Field(..., description="Entity ID of the subject")
    relation: RelationType
    object_entity_id: str | None = Field(None, description="Entity ID of the object")
    value: str | None = Field(None, description="Value for attribute-like relations")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0, description="LLM confidence score")
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: CandidateFactStatus = Field(default=CandidateFactStatus.PENDING)
    rejection_reason: str | None = Field(None, description="Reason if rejected")
    attributes: dict[str, Any] = Field(default_factory=dict)

    # Legacy compatibility
    @property
    def subject_id(self) -> str:
        """Legacy property - use subject_entity_id instead."""
        return self.subject_entity_id

    @property
    def object_id(self) -> str | None:
        """Legacy property - use object_entity_id instead."""
        return self.object_entity_id

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v

    @model_validator(mode="after")
    def validate_object_or_value(self) -> CandidateFact:
        """Require either object_entity_id or value for tuple completeness."""
        if self.object_entity_id is None and (self.value is None or not str(self.value).strip()):
            raise ValueError("CandidateFact requires either object_entity_id or value")
        return self


class ValidatedFact(BaseModel):
    """
    System-approved knowledge with valid-time boundaries.

    ValidatedFacts are the "Ground Truth" of the memory system. They have
    passed all constraint validations and are stored with explicit temporal
    boundaries. ValidatedFacts are edges in the knowledge graph connecting
    Entity nodes.

    Key principle: Facts are never deleted - they are superseded. When knowledge
    changes, the old fact gets a valid_to timestamp and a new fact is created.

    Together with persistence metadata in the storage layer (for example
    ``created_at`` on persisted rows and immutable interaction timestamps),
    this forms a bitemporal model:
    - valid time: when the fact is true in the modeled domain
    - record time: when the system observed and accepted the fact

    Attributes:
        id: Unique identifier
        candidate_fact_id: Link to the original CandidateFact
        source_interaction_id: Link to the originating Interaction (provenance)
        subject_id: Entity ID of the subject
        relation: Type of relationship
        object_id: Entity ID of the object (optional for value facts)
        value: Value payload for attribute-like facts
        valid_from: Start of valid-time period
        valid_to: End of valid-time period (None = still valid/active)
        validated_at: Application-level validation timestamp (record-time companion)
        validated_by: What validated this (constraint set version)
        superseded_by: ID of the fact that superseded this one
        confidence: Original LLM confidence (preserved for analysis)
        attributes: Additional fact attributes
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    candidate_fact_id: str = Field(..., description="ID of the source CandidateFact")
    source_interaction_id: str = Field(..., description="ID of source Interaction")
    subject_id: str = Field(..., description="Entity ID of the subject")
    relation: RelationType
    object_id: str | None = Field(None, description="Entity ID of the object")
    value: str | None = Field(None, description="Value for attribute-like relations")
    valid_from: datetime = Field(..., description="Start of validity period")
    valid_to: datetime | None = Field(None, description="End of validity (None = active)")
    validated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    validated_by: str = Field(default="constraint_validator_v1")
    superseded_by: str | None = Field(None, description="ID of superseding fact")
    confidence: float = Field(..., ge=0.0, le=1.0)
    attributes: dict[str, Any] = Field(default_factory=dict)

    # Rich provenance fields for hybrid storage
    source_text: str | None = Field(
        None,
        description="Original sentence/text from which this fact was extracted",
    )
    embedding: list[float] | None = Field(
        None,
        description="Vector embedding of the source text for semantic retrieval",
    )
    source_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra provenance: source system, extraction confidence signals, actor info",
    )

    @model_validator(mode="after")
    def validate_object_or_value(self) -> ValidatedFact:
        """Require either object_id or value for tuple completeness."""
        if self.object_id is None and (self.value is None or not str(self.value).strip()):
            raise ValueError("ValidatedFact requires either object_id or value")
        return self

    @property
    def is_active(self) -> bool:
        """Check if this fact is currently active (not superseded)."""
        now = datetime.now(timezone.utc)
        if self.valid_to is not None and datetime_before(self.valid_to, now):
            return False
        if self.superseded_by is not None:
            return False
        return datetime_on_or_before(self.valid_from, now)

    def is_active_at(self, timestamp: datetime) -> bool:
        """Check if this fact was active at a specific timestamp."""
        if datetime_after(self.valid_from, timestamp):
            return False
        return not (self.valid_to is not None and datetime_before(self.valid_to, timestamp))


class Constraint(BaseModel):
    """
    Declarative governance rule that controls what can enter memory.

    Constraints are the core of write-time validation. They define domain-specific
    rules that CandidateFacts must satisfy before becoming ValidatedFacts.

    Attributes:
        id: Unique identifier
        name: Human-readable name
        description: Detailed description of what this constraint enforces
        constraint_type: Category of constraint
        applies_to_relations: Which relation types this constraint applies to
        condition: The validation logic (evaluated by ConstraintValidator)
        severity: How critical is this constraint (error, warning, info)
        enabled: Whether this constraint is active
        metadata: Additional configuration
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Constraint name (e.g., 'domain_invariant')")
    description: str = Field(..., description="Human-readable description")
    constraint_type: ConstraintType
    applies_to_relations: list[RelationType] = Field(default_factory=list)
    condition: dict[str, Any] = Field(..., description="Validation logic specification")
    severity: str = Field(default="error", description="error, warning, or info")
    enabled: bool = Field(default=True)
    metadata: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Rejection Record
# =============================================================================


class RejectionRecord(BaseModel):
    """
    Record of why a CandidateFact was rejected.

    This is crucial for explainability - every rejection must have a
    clear reason and, where applicable, suggested alternatives.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    candidate_fact_id: str
    subject_entity_id: str | None = Field(
        default=None,
        description="Entity ID of the subject this rejection pertains to (e.g., patient)",
    )
    rejected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    constraint_id: str = Field(..., description="Which constraint caused rejection")
    constraint_name: str
    reason: str = Field(..., description="Human-readable explanation")
    domain_reasoning: str | None = Field(
        default=None,
        description="Domain-specific reasoning",
        serialization_alias="domain_reasoning",
    )
    alternatives: list[str] = Field(default_factory=list, description="Suggested alternatives")
    severity: str = Field(default="error")


# =============================================================================
# Answer Context (Ephemeral Query View)
# =============================================================================


class AnswerContext(BaseModel):
    """
    Ephemeral query view containing only active, constraint-compliant facts.

    This is the output of graph-based retrieval — a snapshot of the knowledge
    graph relevant to a specific query at a specific point in time. It is
    never persisted; it exists only for the duration of answering a query.

    Attributes:
        query: The original query or context
        timestamp: Point-in-time for fact validity
        seed_entities: Entity IDs used as retrieval starting points
        facts: Retrieved ValidatedFacts (active at timestamp)
        entities: Entities referenced by the retrieved facts
        retrieval_metadata: Stats about the retrieval process
    """

    query: str = Field(..., description="Original query or context")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    seed_entities: list[str] = Field(default_factory=list)
    facts: list[ValidatedFact] = Field(default_factory=list)
    entities: dict[str, Entity] = Field(default_factory=dict)
    retrieval_metadata: dict[str, Any] = Field(default_factory=dict)
