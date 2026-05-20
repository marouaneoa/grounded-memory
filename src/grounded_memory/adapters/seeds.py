"""Reusable seed-based constraints for user-defined governance rules."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from grounded_memory.core.constraints import BaseConstraintEvaluator, ConstraintViolation
from grounded_memory.core.models import CandidateFact, RelationType, datetime_on_or_after


class SeedConstraintEvaluator(BaseConstraintEvaluator):
    """Declarative constraint evaluator generated from a user seed payload."""

    def __init__(
        self,
        *,
        constraint_id: str,
        constraint_name: str,
        description: str,
        applies_to_relations: list[RelationType] | None = None,
        severity: str = "error",
        required_attributes: dict[str, Any] | None = None,
        required_attribute_keys: list[str] | None = None,
        forbidden_attributes: list[str] | None = None,
        require_object: bool = False,
        require_value: bool = False,
        value_regex: str | None = None,
    ) -> None:
        self._constraint_id = constraint_id
        self._constraint_name = constraint_name
        self._description = description
        self._applies_to_relations = applies_to_relations or []
        self._severity = severity
        self.required_attributes = dict(required_attributes or {})
        self.required_attribute_keys = list(required_attribute_keys or [])
        self.forbidden_attributes = list(forbidden_attributes or [])
        self.require_object = require_object
        self.require_value = require_value
        self.value_regex = value_regex

    @property
    def constraint_id(self) -> str:
        return self._constraint_id

    @property
    def constraint_name(self) -> str:
        return self._constraint_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def applies_to_relations(self) -> list[RelationType]:
        return self._applies_to_relations

    @property
    def severity(self) -> str:
        return self._severity

    def evaluate(
        self,
        candidate: CandidateFact,
        knowledge_state: Any,
    ) -> ConstraintViolation | None:
        if self.require_object and not candidate.object_entity_id:
            return ConstraintViolation(
                constraint_id=self.constraint_id,
                constraint_name=self.constraint_name,
                description="object_entity_id is required by this seed",
                severity=self.severity,
            )

        if self.require_value and not (candidate.value and str(candidate.value).strip()):
            return ConstraintViolation(
                constraint_id=self.constraint_id,
                constraint_name=self.constraint_name,
                description="value is required by this seed",
                severity=self.severity,
            )

        if (
            self.value_regex
            and candidate.value is not None
            and re.search(self.value_regex, str(candidate.value)) is None
        ):
            return ConstraintViolation(
                constraint_id=self.constraint_id,
                constraint_name=self.constraint_name,
                description="value does not match required pattern",
                severity=self.severity,
                metadata={"value_regex": self.value_regex},
            )

        for key, expected in self.required_attributes.items():
            current = candidate.attributes.get(key)
            if current != expected:
                return ConstraintViolation(
                    constraint_id=self.constraint_id,
                    constraint_name=self.constraint_name,
                    description=(f"attribute '{key}' must equal {expected!r}; got {current!r}"),
                    severity=self.severity,
                    metadata={"required_attribute": key},
                )

        for key in self.required_attribute_keys:
            current = candidate.attributes.get(key)
            if current in (None, ""):
                return ConstraintViolation(
                    constraint_id=self.constraint_id,
                    constraint_name=self.constraint_name,
                    description=f"attribute '{key}' is required by this seed",
                    severity=self.severity,
                    metadata={"required_attribute_key": key},
                )

        for key in self.forbidden_attributes:
            if key in candidate.attributes and candidate.attributes.get(key) not in (None, ""):
                return ConstraintViolation(
                    constraint_id=self.constraint_id,
                    constraint_name=self.constraint_name,
                    description=f"attribute '{key}' is forbidden by this seed",
                    severity=self.severity,
                    metadata={"forbidden_attribute": key},
                )

        return None


class CardinalitySeedConstraintEvaluator(BaseConstraintEvaluator):
    """Seed evaluator enforcing max active fact count for a relation scope."""

    def __init__(
        self,
        *,
        constraint_id: str,
        constraint_name: str,
        description: str,
        relation: RelationType,
        max_count: int,
        severity: str = "error",
        require_same_subject: bool = True,
    ) -> None:
        self._constraint_id = constraint_id
        self._constraint_name = constraint_name
        self._description = description
        self.relation = relation
        self.max_count = max_count
        self._severity = severity
        self.require_same_subject = require_same_subject

    @property
    def constraint_id(self) -> str:
        return self._constraint_id

    @property
    def constraint_name(self) -> str:
        return self._constraint_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def applies_to_relations(self) -> list[RelationType]:
        return [self.relation]

    @property
    def severity(self) -> str:
        return self._severity

    def evaluate(
        self,
        candidate: CandidateFact,
        knowledge_state: Any,
    ) -> ConstraintViolation | None:
        if self.max_count < 0:
            return ConstraintViolation(
                constraint_id=self.constraint_id,
                constraint_name=self.constraint_name,
                description="max_count must be >= 0",
                severity="error",
            )

        if self.require_same_subject:
            existing = knowledge_state.get_facts_by_relation(
                entity_id=candidate.subject_entity_id,
                relation=self.relation,
                as_subject=True,
            )
        else:
            existing = knowledge_state.get_all_facts_by_relation(self.relation)

        active_count = sum(1 for fact in existing if fact.is_active)

        if active_count >= self.max_count:
            return ConstraintViolation(
                constraint_id=self.constraint_id,
                constraint_name=self.constraint_name,
                description=(
                    f"{self.description} (active_count={active_count}, max_count={self.max_count})"
                ),
                severity=self.severity,
                metadata={
                    "active_count": active_count,
                    "max_count": self.max_count,
                    "relation": self.relation.value,
                },
            )

        return None


class TemporalCardinalitySeedConstraintEvaluator(BaseConstraintEvaluator):
    """Seed evaluator enforcing max writes per relation in a rolling time window."""

    def __init__(
        self,
        *,
        constraint_id: str,
        constraint_name: str,
        description: str,
        relation: RelationType,
        max_count: int,
        window_seconds: int,
        severity: str = "error",
        require_same_subject: bool = True,
    ) -> None:
        self._constraint_id = constraint_id
        self._constraint_name = constraint_name
        self._description = description
        self.relation = relation
        self.max_count = max_count
        self.window_seconds = window_seconds
        self._severity = severity
        self.require_same_subject = require_same_subject

    @property
    def constraint_id(self) -> str:
        return self._constraint_id

    @property
    def constraint_name(self) -> str:
        return self._constraint_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def applies_to_relations(self) -> list[RelationType]:
        return [self.relation]

    @property
    def severity(self) -> str:
        return self._severity

    def evaluate(
        self,
        candidate: CandidateFact,
        knowledge_state: Any,
    ) -> ConstraintViolation | None:
        if self.max_count < 0:
            return ConstraintViolation(
                constraint_id=self.constraint_id,
                constraint_name=self.constraint_name,
                description="max_count must be >= 0",
                severity="error",
            )

        if self.window_seconds <= 0:
            return ConstraintViolation(
                constraint_id=self.constraint_id,
                constraint_name=self.constraint_name,
                description="window_seconds must be > 0",
                severity="error",
            )

        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=self.window_seconds)

        existing_facts: list[Any]
        if hasattr(knowledge_state, "get_all_validated_facts"):
            existing_facts = knowledge_state.get_all_validated_facts()
        elif self.require_same_subject:
            existing_facts = knowledge_state.get_facts_by_relation(
                entity_id=candidate.subject_entity_id,
                relation=self.relation,
                as_subject=True,
            )
        else:
            existing_facts = knowledge_state.get_all_facts_by_relation(self.relation)

        window_count = 0
        for fact in existing_facts:
            if fact.relation != self.relation:
                continue
            if self.require_same_subject and fact.subject_id != candidate.subject_entity_id:
                continue
            if datetime_on_or_after(fact.valid_from, window_start):
                window_count += 1

        if window_count >= self.max_count:
            return ConstraintViolation(
                constraint_id=self.constraint_id,
                constraint_name=self.constraint_name,
                description=(
                    f"{self.description} (window_count={window_count}, "
                    f"max_count={self.max_count}, window_seconds={self.window_seconds})"
                ),
                severity=self.severity,
                metadata={
                    "window_count": window_count,
                    "max_count": self.max_count,
                    "window_seconds": self.window_seconds,
                    "relation": self.relation.value,
                },
            )

        return None
