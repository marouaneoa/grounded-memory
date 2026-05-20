"""
Constraint Validation Framework

This module implements the constraint validation logic for the Grounded Memory System.
Constraints are evaluated at write-time (before facts enter memory) to ensure
that only valid, consistent information is stored.

Key concepts:
- Constraints are declarative rules that facts must satisfy
- Validation happens BEFORE persistence, not during retrieval
- Every rejection includes a clear explanation and alternatives where possible
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol

from grounded_memory.core.models import (
    CandidateFact,
    Entity,
    RejectionRecord,
    RelationType,
)

# =============================================================================
# Validation Result Types
# =============================================================================


@dataclass
class ConstraintViolation:
    """Represents a single constraint violation."""

    constraint_id: str
    constraint_name: str
    description: str
    severity: str  # "error", "warning", "info"
    domain_reasoning: str | None = None
    alternatives: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of validating a CandidateFact against all constraints."""

    is_valid: bool
    candidate_fact_id: str
    violations: list[ConstraintViolation] = field(default_factory=list)
    warnings: list[ConstraintViolation] = field(default_factory=list)
    checked_constraints: list[str] = field(default_factory=list)
    validation_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def has_errors(self) -> bool:
        return len(self.violations) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def get_primary_rejection_reason(self) -> str | None:
        """Get the primary reason for rejection."""
        if self.violations:
            v = self.violations[0]
            return f"{v.constraint_name}: {v.description}"
        return None

    def to_rejection_record(self) -> RejectionRecord | None:
        """Convert to a RejectionRecord if validation failed."""
        if self.is_valid or not self.violations:
            return None

        primary = self.violations[0]
        return RejectionRecord(
            candidate_fact_id=self.candidate_fact_id,
            constraint_id=primary.constraint_id,
            constraint_name=primary.constraint_name,
            reason=primary.description,
            domain_reasoning=primary.domain_reasoning,
            alternatives=primary.alternatives,
            severity=primary.severity,
        )


# =============================================================================
# Knowledge State Protocol
# =============================================================================


class KnowledgeState(Protocol):
    """
    Protocol for accessing the current knowledge state.

    The ConstraintValidator needs access to existing facts and entities
    to evaluate constraints. This protocol defines the required interface.
    """

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        ...


# =============================================================================
# Dynamic Constraint Governance
# =============================================================================


class ConstraintLifecycleStatus(str, Enum):
    """Lifecycle states for managed constraints."""

    PROPOSED = "proposed"
    SHADOW = "shadow"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class ConstraintSource(str, Enum):
    """Who authored the constraint."""

    HUMAN = "human"
    AGENT = "agent"


@dataclass(frozen=True)
class ConstraintFormTemplate:
    """
    Canonical form for a constraint family.

    A form template defines required metadata fields and optional scope limits,
    so rule authoring follows a consistent contract instead of ad-hoc prose.
    """

    form_id: str
    description: str
    required_metadata_fields: list[str] = field(default_factory=list)
    optional_metadata_fields: list[str] = field(default_factory=list)
    allowed_relations: list[RelationType] = field(default_factory=list)

    def validate_metadata(self, metadata: dict[str, Any]) -> list[str]:
        """Return missing required fields for the provided metadata."""
        missing: list[str] = []
        for field_name in self.required_metadata_fields:
            if field_name not in metadata or metadata[field_name] in (None, ""):
                missing.append(field_name)
        return missing


@dataclass
class DynamicConstraintScope:
    """
    Scope selector for adaptive constraints.

    Constraints can be activated only when their scope matches the
    current candidate and runtime context.
    """

    relation_types: list[RelationType] = field(default_factory=list)
    required_context: dict[str, Any] = field(default_factory=dict)
    min_candidate_confidence: float = 0.0

    def matches(
        self,
        candidate: CandidateFact,
        runtime_context: dict[str, Any] | None,
    ) -> bool:
        if self.relation_types and candidate.relation not in self.relation_types:
            return False

        if candidate.confidence < self.min_candidate_confidence:
            return False

        if not self.required_context:
            return True

        runtime_context = runtime_context or {}
        for key, expected_value in self.required_context.items():
            if runtime_context.get(key) != expected_value:
                return False

        return True


@dataclass
class ManagedConstraint:
    """Managed wrapper with lifecycle and runtime governance metadata."""

    evaluator: BaseConstraintEvaluator
    source: ConstraintSource = ConstraintSource.HUMAN
    lifecycle: ConstraintLifecycleStatus = ConstraintLifecycleStatus.ACTIVE
    priority: int = 100
    scope: DynamicConstraintScope = field(default_factory=DynamicConstraintScope)
    form_id: str | None = None
    form_metadata: dict[str, Any] = field(default_factory=dict)
    shadow_hits: int = 0
    shadow_violations: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def constraint_id(self) -> str:
        return self.evaluator.constraint_id

    def mark_shadow_observation(self, violated: bool) -> None:
        self.shadow_hits += 1
        if violated:
            self.shadow_violations += 1
        self.last_updated = datetime.now(timezone.utc)


@dataclass
class ConstraintReplayMetrics:
    """Offline replay metrics for a managed dynamic constraint."""

    constraint_id: str
    lifecycle: ConstraintLifecycleStatus
    evaluated_candidates: int = 0
    violations: int = 0
    incremental_blocks: int = 0
    covered_existing_blocks: int = 0

    @property
    def trigger_rate(self) -> float:
        if self.evaluated_candidates == 0:
            return 0.0
        return self.violations / self.evaluated_candidates

    @property
    def projected_false_block_rate(self) -> float:
        if self.evaluated_candidates == 0:
            return 0.0
        return self.incremental_blocks / self.evaluated_candidates

    @property
    def projected_miss_coverage(self) -> float:
        if self.evaluated_candidates == 0:
            return 0.0
        return self.covered_existing_blocks / self.evaluated_candidates


# =============================================================================
# Base Constraint Validator
# =============================================================================


class BaseConstraintEvaluator(ABC):
    """
    Abstract base class for constraint evaluators.

    Each constraint type has its own evaluator that knows how to
    check if a CandidateFact satisfies the constraint.
    """

    @property
    @abstractmethod
    def constraint_id(self) -> str:
        """Unique identifier for this constraint."""
        pass

    @property
    @abstractmethod
    def constraint_name(self) -> str:
        """Human-readable name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this constraint enforces."""
        pass

    @property
    def applies_to_relations(self) -> list[RelationType]:
        """Which relation types this constraint applies to."""
        return []  # Empty means applies to all

    @property
    def severity(self) -> str:
        """Severity level: error, warning, or info."""
        return "error"

    @abstractmethod
    def evaluate(
        self,
        candidate: CandidateFact,
        knowledge_state: KnowledgeState,
    ) -> ConstraintViolation | None:
        """
        Evaluate the constraint against a candidate fact.

        Returns None if constraint is satisfied, or a ConstraintViolation
        if the constraint is violated.
        """
        pass

    def applies_to(self, candidate: CandidateFact) -> bool:
        """Check if this constraint applies to the given candidate."""
        if not self.applies_to_relations:
            return True
        return candidate.relation in self.applies_to_relations


# =============================================================================
# Constraint Validator (Main Class)
# =============================================================================


class ConstraintValidator:
    """
    Main constraint validation engine.

    This class orchestrates the evaluation of all registered constraints
    against incoming CandidateFacts. It is the core of write-time validation.

    Usage:
        validator = ConstraintValidator()
        validator.register(MyConstraintA())
        validator.register(MyConstraintB())

        result = validator.validate(candidate_fact, knowledge_state)
        if result.is_valid:
            # Safe to persist as ValidatedFact
        else:
            # Reject with explanation
    """

    def __init__(self):
        self._evaluators: list[BaseConstraintEvaluator] = []
        self._evaluator_map: dict[str, BaseConstraintEvaluator] = {}
        self._managed_constraints: dict[str, ManagedConstraint] = {}
        self._form_templates: dict[str, ConstraintFormTemplate] = {}
        self._validation_signals: list[dict[str, Any]] = []
        self._register_default_form_templates()

    def _register_default_form_templates(self) -> None:
        """Register built-in form templates for consistent rule authoring."""
        self.register_form_template(
            ConstraintFormTemplate(
                form_id="safety_control",
                description="Safety rules for high-risk relation and contradiction checks.",
                required_metadata_fields=[
                    "owner",
                    "rationale",
                    "subject_type",
                    "object_type",
                    "violation_logic",
                    "fallback_action",
                ],
                optional_metadata_fields=["evidence_refs", "jurisdiction", "version_note"],
                allowed_relations=[],
            )
        )
        self.register_form_template(
            ConstraintFormTemplate(
                form_id="temporal_consistency",
                description="Temporal ordering and validity-window constraints.",
                required_metadata_fields=[
                    "owner",
                    "rationale",
                    "time_reference",
                    "temporal_assumption",
                    "fallback_action",
                ],
                optional_metadata_fields=["max_lookback_days", "evidence_refs"],
            )
        )
        self.register_form_template(
            ConstraintFormTemplate(
                form_id="cardinality_control",
                description="Limits for duplicate, overlap, and saturation conditions.",
                required_metadata_fields=[
                    "owner",
                    "rationale",
                    "aggregation_key",
                    "threshold",
                    "fallback_action",
                ],
                optional_metadata_fields=["window", "evidence_refs"],
            )
        )

    def register_form_template(self, template: ConstraintFormTemplate) -> None:
        """Register or replace a constraint form template."""
        self._form_templates[template.form_id] = template

    def get_form_template(self, form_id: str) -> ConstraintFormTemplate | None:
        """Get a form template by ID."""
        return self._form_templates.get(form_id)

    def list_form_templates(self) -> list[ConstraintFormTemplate]:
        """List all registered form templates."""
        return list(self._form_templates.values())

    def register(
        self,
        evaluator: BaseConstraintEvaluator,
        *,
        source: ConstraintSource = ConstraintSource.HUMAN,
        lifecycle: ConstraintLifecycleStatus = ConstraintLifecycleStatus.ACTIVE,
        priority: int = 100,
        scope: DynamicConstraintScope | None = None,
        form_id: str | None = None,
        form_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a constraint evaluator with governance metadata."""
        self._evaluators.append(evaluator)
        self._evaluator_map[evaluator.constraint_id] = evaluator
        self._managed_constraints[evaluator.constraint_id] = ManagedConstraint(
            evaluator=evaluator,
            source=source,
            lifecycle=lifecycle,
            priority=priority,
            scope=scope or DynamicConstraintScope(),
            form_id=form_id,
            form_metadata=form_metadata or {},
        )

    def register_with_form(
        self,
        evaluator: BaseConstraintEvaluator,
        *,
        form_id: str,
        form_metadata: dict[str, Any],
        source: ConstraintSource = ConstraintSource.HUMAN,
        lifecycle: ConstraintLifecycleStatus = ConstraintLifecycleStatus.ACTIVE,
        priority: int = 100,
        scope: DynamicConstraintScope | None = None,
    ) -> None:
        """
        Register a constraint with required form-template validation.

        Raises:
            ValueError: when the form ID is unknown, metadata is incomplete,
            or relation applicability conflicts with the form.
        """
        template = self.get_form_template(form_id)
        if template is None:
            raise ValueError(f"Unknown constraint form template: {form_id}")

        missing_fields = template.validate_metadata(form_metadata)
        if missing_fields:
            missing_text = ", ".join(missing_fields)
            raise ValueError(
                f"Constraint {evaluator.constraint_id} missing required form metadata fields: {missing_text}"
            )

        if template.allowed_relations and evaluator.applies_to_relations:
            invalid_relations = [
                relation
                for relation in evaluator.applies_to_relations
                if relation not in template.allowed_relations
            ]
            if invalid_relations:
                invalid_text = ", ".join(relation.value for relation in invalid_relations)
                raise ValueError(
                    f"Constraint {evaluator.constraint_id} uses relations not allowed by form {form_id}: {invalid_text}"
                )

        self.register(
            evaluator,
            source=source,
            lifecycle=lifecycle,
            priority=priority,
            scope=scope,
            form_id=form_id,
            form_metadata=form_metadata,
        )

    def register_dynamic(
        self,
        evaluator: BaseConstraintEvaluator,
        *,
        lifecycle: ConstraintLifecycleStatus = ConstraintLifecycleStatus.PROPOSED,
        priority: int = 50,
        scope: DynamicConstraintScope | None = None,
        form_id: str | None = None,
        form_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register an agent-proposed adaptive constraint."""
        if form_id is not None:
            self.register_with_form(
                evaluator,
                form_id=form_id,
                form_metadata=form_metadata or {},
                source=ConstraintSource.AGENT,
                lifecycle=lifecycle,
                priority=priority,
                scope=scope,
            )
            return

        self.register(
            evaluator,
            source=ConstraintSource.AGENT,
            lifecycle=lifecycle,
            priority=priority,
            scope=scope,
            form_id=form_id,
            form_metadata=form_metadata,
        )

    def unregister(self, constraint_id: str) -> bool:
        """Unregister a constraint evaluator by ID."""
        if constraint_id in self._evaluator_map:
            evaluator = self._evaluator_map.pop(constraint_id)
            self._evaluators.remove(evaluator)
            self._managed_constraints.pop(constraint_id, None)
            return True
        return False

    def get_evaluator(self, constraint_id: str) -> BaseConstraintEvaluator | None:
        """Get an evaluator by ID."""
        return self._evaluator_map.get(constraint_id)

    @property
    def registered_constraints(self) -> list[str]:
        """List of registered constraint IDs."""
        return list(self._evaluator_map.keys())

    def set_lifecycle(
        self,
        constraint_id: str,
        lifecycle: ConstraintLifecycleStatus,
    ) -> bool:
        """Update lifecycle stage for a managed constraint."""
        managed = self._managed_constraints.get(constraint_id)
        if managed is None:
            return False
        managed.lifecycle = lifecycle
        managed.last_updated = datetime.now(timezone.utc)
        return True

    def set_priority(self, constraint_id: str, priority: int) -> bool:
        """Update execution priority for a managed constraint."""
        managed = self._managed_constraints.get(constraint_id)
        if managed is None:
            return False
        managed.priority = priority
        managed.last_updated = datetime.now(timezone.utc)
        return True

    def get_managed_constraint(self, constraint_id: str) -> ManagedConstraint | None:
        """Get managed constraint metadata by ID."""
        return self._managed_constraints.get(constraint_id)

    def list_managed_constraints(self) -> list[ManagedConstraint]:
        """List all managed constraints with lifecycle metadata."""
        return list(self._managed_constraints.values())

    def record_validation_signal(
        self,
        candidate: CandidateFact,
        result: ValidationResult,
        runtime_context: dict[str, Any] | None = None,
    ) -> None:
        """Record write-time governance signal for discovery/replay workflows."""
        self._validation_signals.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "candidate_id": candidate.id,
                "relation": candidate.relation.value,
                "candidate_confidence": candidate.confidence,
                "candidate_attributes": dict(candidate.attributes or {}),
                "has_object": candidate.object_entity_id is not None,
                "has_value": bool(candidate.value and str(candidate.value).strip()),
                "is_valid": result.is_valid,
                "violations": [v.constraint_id for v in result.violations],
                "warnings": [w.constraint_id for w in result.warnings],
                "runtime_context": runtime_context or {},
            }
        )

    def list_validation_signals(self, limit: int = 500) -> list[dict[str, Any]]:
        """Get recent governance signals for mining and diagnostics."""
        if limit <= 0:
            return []
        return self._validation_signals[-limit:]

    def replay_dynamic_constraints(
        self,
        candidates: list[CandidateFact],
        knowledge_state: KnowledgeState,
    ) -> dict[str, ConstraintReplayMetrics]:
        """
        Offline replay for dynamic constraints.

        Metrics meaning:
        - incremental_blocks: candidate would pass current human-active rules but fail this dynamic rule.
        - covered_existing_blocks: candidate already blocked by current human-active rules and also hit by this dynamic rule.
        """
        dynamic_constraints = [
            managed
            for managed in self._managed_constraints.values()
            if managed.source == ConstraintSource.AGENT
            and managed.lifecycle
            in {
                ConstraintLifecycleStatus.PROPOSED,
                ConstraintLifecycleStatus.SHADOW,
                ConstraintLifecycleStatus.ACTIVE,
            }
        ]

        human_active = [
            managed.evaluator
            for managed in self._managed_constraints.values()
            if managed.source == ConstraintSource.HUMAN
            and managed.lifecycle == ConstraintLifecycleStatus.ACTIVE
        ]

        metrics: dict[str, ConstraintReplayMetrics] = {
            managed.constraint_id: ConstraintReplayMetrics(
                constraint_id=managed.constraint_id,
                lifecycle=managed.lifecycle,
            )
            for managed in dynamic_constraints
        }

        for candidate in candidates:
            human_blocked = False
            for evaluator in human_active:
                if evaluator.applies_to(candidate) and evaluator.evaluate(
                    candidate, knowledge_state
                ):
                    human_blocked = True
                    break

            for managed in dynamic_constraints:
                evaluator = managed.evaluator
                if not evaluator.applies_to(candidate):
                    continue

                replay = metrics[managed.constraint_id]
                replay.evaluated_candidates += 1
                violation = evaluator.evaluate(candidate, knowledge_state)
                if violation is None:
                    continue

                replay.violations += 1
                if human_blocked:
                    replay.covered_existing_blocks += 1
                else:
                    replay.incremental_blocks += 1

        return metrics

    def promote_dynamic_constraints(
        self,
        replay_metrics: dict[str, ConstraintReplayMetrics],
        *,
        min_trigger_rate: float = 0.01,
        max_projected_false_block_rate: float = 0.02,
        min_candidates: int = 100,
    ) -> list[str]:
        """Promote dynamic constraints from proposed/shadow to active based on replay evidence."""
        promoted: list[str] = []

        for constraint_id, metric in replay_metrics.items():
            managed = self._managed_constraints.get(constraint_id)
            if managed is None:
                continue

            if managed.lifecycle not in {
                ConstraintLifecycleStatus.PROPOSED,
                ConstraintLifecycleStatus.SHADOW,
            }:
                continue

            if metric.evaluated_candidates < min_candidates:
                continue

            if metric.trigger_rate < min_trigger_rate:
                continue

            if metric.projected_false_block_rate > max_projected_false_block_rate:
                continue

            managed.lifecycle = ConstraintLifecycleStatus.ACTIVE
            managed.last_updated = datetime.now(timezone.utc)
            promoted.append(constraint_id)

        return promoted

    def _select_managed_constraints(
        self,
        candidate: CandidateFact,
        runtime_context: dict[str, Any] | None,
        max_dynamic_constraints: int,
        max_shadow_constraints: int,
    ) -> list[ManagedConstraint]:
        """
        Select a bounded set of constraints for this candidate.

        Rules:
        - Human-authored active constraints are always included.
        - Agent-authored active constraints are context-filtered and capped.
        - Proposed/shadow constraints are observed in shadow mode and capped.
        """
        human_active: list[ManagedConstraint] = []
        agent_active: list[ManagedConstraint] = []
        shadow_constraints: list[ManagedConstraint] = []

        for managed in self._managed_constraints.values():
            evaluator = managed.evaluator

            if not evaluator.applies_to(candidate):
                continue

            if not managed.scope.matches(candidate, runtime_context):
                continue

            if managed.lifecycle == ConstraintLifecycleStatus.DEPRECATED:
                continue

            if managed.lifecycle == ConstraintLifecycleStatus.ACTIVE:
                if managed.source == ConstraintSource.HUMAN:
                    human_active.append(managed)
                else:
                    agent_active.append(managed)
            elif managed.lifecycle in (
                ConstraintLifecycleStatus.PROPOSED,
                ConstraintLifecycleStatus.SHADOW,
            ):
                shadow_constraints.append(managed)

        agent_active.sort(key=lambda c: c.priority, reverse=True)
        shadow_constraints.sort(key=lambda c: c.priority, reverse=True)

        bounded_agent = agent_active[:max_dynamic_constraints]
        bounded_shadow = shadow_constraints[:max_shadow_constraints]

        return human_active + bounded_agent + bounded_shadow

    def validate(
        self,
        candidate: CandidateFact,
        knowledge_state: KnowledgeState,
        stop_on_first_error: bool = False,
        runtime_context: dict[str, Any] | None = None,
        max_dynamic_constraints: int = 20,
        max_shadow_constraints: int = 10,
    ) -> ValidationResult:
        """
        Validate a CandidateFact against all applicable constraints.

        Args:
            candidate: The fact to validate
            knowledge_state: Access to existing knowledge
            stop_on_first_error: If True, stop after first error

        Returns:
            ValidationResult with all violations and warnings
        """
        violations: list[ConstraintViolation] = []
        warnings: list[ConstraintViolation] = []
        checked: list[str] = []

        selected_constraints = self._select_managed_constraints(
            candidate=candidate,
            runtime_context=runtime_context,
            max_dynamic_constraints=max_dynamic_constraints,
            max_shadow_constraints=max_shadow_constraints,
        )

        for managed in selected_constraints:
            evaluator = managed.evaluator

            checked.append(evaluator.constraint_id)

            # Evaluate the constraint
            violation = evaluator.evaluate(candidate, knowledge_state)

            # Proposed/shadow constraints are observed but never block writes.
            if managed.lifecycle in (
                ConstraintLifecycleStatus.PROPOSED,
                ConstraintLifecycleStatus.SHADOW,
            ):
                managed.mark_shadow_observation(violated=violation is not None)

                if violation is not None:
                    warnings.append(
                        ConstraintViolation(
                            constraint_id=violation.constraint_id,
                            constraint_name=violation.constraint_name,
                            description=f"[shadow] {violation.description}",
                            severity="warning",
                            domain_reasoning=violation.domain_reasoning,
                            alternatives=violation.alternatives,
                            metadata={
                                **(violation.metadata or {}),
                                "mode": "shadow",
                                "source": managed.source.value,
                            },
                        )
                    )
                continue

            if violation is not None:
                if violation.severity == "error":
                    violations.append(violation)
                    if stop_on_first_error:
                        break
                elif violation.severity == "warning":
                    warnings.append(violation)
                # "info" level violations are logged but don't affect validity

        result = ValidationResult(
            is_valid=len(violations) == 0,
            candidate_fact_id=candidate.id,
            violations=violations,
            warnings=warnings,
            checked_constraints=checked,
        )
        self.record_validation_signal(
            candidate=candidate,
            result=result,
            runtime_context=runtime_context,
        )
        return result

    def validate_batch(
        self,
        candidates: list[CandidateFact],
        knowledge_state: KnowledgeState,
        runtime_context: dict[str, Any] | None = None,
        max_dynamic_constraints: int = 20,
        max_shadow_constraints: int = 10,
    ) -> dict[str, ValidationResult]:
        """
        Validate multiple CandidateFacts.

        Returns a dict mapping candidate IDs to their ValidationResults.
        """
        return {
            c.id: self.validate(
                c,
                knowledge_state,
                runtime_context=runtime_context,
                max_dynamic_constraints=max_dynamic_constraints,
                max_shadow_constraints=max_shadow_constraints,
            )
            for c in candidates
        }


# =============================================================================
# Constraint Builder (for YAML/declarative constraints)
# =============================================================================


class DeclarativeConstraint(BaseConstraintEvaluator):
    """
    Constraint built from declarative specification (e.g., YAML).

    This allows constraints to be defined in configuration rather than code.
    """

    def __init__(
        self,
        constraint_id: str,
        name: str,
        description: str,
        condition_fn: Callable[[CandidateFact, KnowledgeState], ConstraintViolation | None],
        applies_to: list[RelationType] = None,
        severity: str = "error",
    ):
        self._constraint_id = constraint_id
        self._name = name
        self._description = description
        self._condition_fn = condition_fn
        self._applies_to = applies_to or []
        self._severity = severity

    @property
    def constraint_id(self) -> str:
        return self._constraint_id

    @property
    def constraint_name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def applies_to_relations(self) -> list[RelationType]:
        return self._applies_to

    @property
    def severity(self) -> str:
        return self._severity

    def evaluate(
        self,
        candidate: CandidateFact,
        knowledge_state: KnowledgeState,
    ) -> ConstraintViolation | None:
        return self._condition_fn(candidate, knowledge_state)


# =============================================================================
# Common Constraint Patterns
# =============================================================================


class ProhibitionConstraint(BaseConstraintEvaluator):
    """
    Base class for constraints that prohibit certain fact patterns.

    Example: "Never create relation R when a prohibited condition is present"
    """

    @abstractmethod
    def is_prohibited(
        self,
        candidate: CandidateFact,
        knowledge_state: KnowledgeState,
    ) -> tuple[bool, str | None, list[str]]:
        """
        Check if the candidate fact is prohibited.

        Returns:
            (is_prohibited, reason, alternatives)
        """
        pass

    def evaluate(
        self,
        candidate: CandidateFact,
        knowledge_state: KnowledgeState,
    ) -> ConstraintViolation | None:
        prohibited, reason, alternatives = self.is_prohibited(candidate, knowledge_state)

        if prohibited:
            return ConstraintViolation(
                constraint_id=self.constraint_id,
                constraint_name=self.constraint_name,
                description=reason or self.description,
                severity=self.severity,
                alternatives=alternatives,
            )
        return None


class CardinalityConstraint(BaseConstraintEvaluator):
    """
    Base class for constraints that limit the count of certain facts.

    Example: "An entity can have at most N active facts in a relation scope"
    """

    @property
    @abstractmethod
    def max_count(self) -> int:
        """Maximum allowed count."""
        pass

    @abstractmethod
    def count_existing(
        self,
        candidate: CandidateFact,
        knowledge_state: KnowledgeState,
    ) -> int:
        """Count existing facts that would conflict with the candidate."""
        pass

    def evaluate(
        self,
        candidate: CandidateFact,
        knowledge_state: KnowledgeState,
    ) -> ConstraintViolation | None:
        current_count = self.count_existing(candidate, knowledge_state)

        if current_count >= self.max_count:
            return ConstraintViolation(
                constraint_id=self.constraint_id,
                constraint_name=self.constraint_name,
                description=f"{self.description} (current: {current_count}, max: {self.max_count})",
                severity=self.severity,
            )
        return None


class TemporalConstraint(BaseConstraintEvaluator):
    """
    Base class for constraints involving temporal relationships.

    Example: "Event timestamp must occur after prerequisite timestamp"
    """

    @abstractmethod
    def check_temporal_validity(
        self,
        candidate: CandidateFact,
        knowledge_state: KnowledgeState,
    ) -> tuple[bool, str | None]:
        """
        Check if temporal relationships are valid.

        Returns:
            (is_valid, error_message)
        """
        pass

    def evaluate(
        self,
        candidate: CandidateFact,
        knowledge_state: KnowledgeState,
    ) -> ConstraintViolation | None:
        is_valid, error_message = self.check_temporal_validity(candidate, knowledge_state)

        if not is_valid:
            return ConstraintViolation(
                constraint_id=self.constraint_id,
                constraint_name=self.constraint_name,
                description=error_message or self.description,
                severity=self.severity,
            )
        return None
