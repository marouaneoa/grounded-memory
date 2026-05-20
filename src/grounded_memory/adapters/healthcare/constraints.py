"""
YAML Constraint Loader for Healthcare

Parses healthcare_constraints.yaml and generates BaseConstraintEvaluator
instances for the Grounded Memory System.
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any

import yaml

from grounded_memory.adapters.healthcare.knowledge import (
    check_major_interaction,
    check_moderate_interaction,
    get_cross_reactive_ingredients,
    get_therapeutic_classes,
    normalize_drug_name,
)
from grounded_memory.core.constraints import (
    BaseConstraintEvaluator,
    ConstraintViolation,
    KnowledgeState,
)
from grounded_memory.core.models import (
    CandidateFact,
    RelationType,
    datetime_after,
    datetime_on_or_after,
)


class YamlConstraintEvaluator(BaseConstraintEvaluator):
    """
    Dynamically evaluates a constraint loaded from a YAML dictionary.
    """

    def __init__(self, config_dict: dict[str, Any]):
        self._config = config_dict
        self._id = config_dict.get("id", "unknown_yaml_constraint")
        self._name = config_dict.get("name", "Unnamed Constraint")
        self._description = config_dict.get("description", "")
        self._severity = config_dict.get("severity", "error")

        # Parse applies_to
        applies_to_str = config_dict.get("applies_to", [])
        self._applies_to = []
        for relation_str in applies_to_str:
            with suppress(ValueError):
                self._applies_to.append(RelationType(relation_str))

    @property
    def constraint_id(self) -> str:
        return self._id

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
        self, candidate: CandidateFact, knowledge_state: KnowledgeState
    ) -> ConstraintViolation | None:

        condition = self._config.get("condition", {})
        check = condition.get("check")

        if check == "intersection_empty":
            return self._evaluate_intersection_empty(candidate, knowledge_state, condition)
        elif check == "no_major_interactions":
            return self._evaluate_no_major_interactions(candidate, knowledge_state, condition)
        elif check == "no_moderate_interactions":
            return self._evaluate_no_moderate_interactions(candidate, knowledge_state, condition)
        elif check == "cardinality_limit":
            return self._evaluate_cardinality_limit(candidate, knowledge_state, condition)

        # Other temporal or informational checks can be implemented here
        return None

    @staticmethod
    def _safe_get_entity_name(knowledge_state: KnowledgeState, entity_id: str | None) -> str | None:
        if not entity_id:
            return None
        entity = knowledge_state.get_entity(entity_id)
        if entity is None:
            return None
        name = str(getattr(entity, "name", "")).strip()
        return name or None

    @staticmethod
    def _safe_get_facts_by_relation(
        knowledge_state: KnowledgeState,
        *,
        entity_id: str,
        relation: RelationType,
        as_subject: bool = True,
    ) -> list[Any]:
        getter = getattr(knowledge_state, "get_facts_by_relation", None)
        if not callable(getter):
            return []
        return list(getter(entity_id=entity_id, relation=relation, as_subject=as_subject) or [])

    def _fact_medication_names(self, fact: Any, knowledge_state: KnowledgeState) -> set[str]:
        names: set[str] = set()
        object_name = self._safe_get_entity_name(knowledge_state, getattr(fact, "object_id", None))
        if object_name:
            names.add(object_name)
        value = getattr(fact, "value", None)
        if isinstance(value, str) and value.strip():
            names.add(value.strip())
        return names

    def _candidate_medication_names(
        self, candidate: CandidateFact, knowledge_state: KnowledgeState
    ) -> set[str]:
        names: set[str] = set()

        object_name = self._safe_get_entity_name(knowledge_state, candidate.object_entity_id)
        if object_name:
            names.add(object_name)

        medication_name = candidate.attributes.get("medication_name")
        if isinstance(medication_name, str) and medication_name.strip():
            names.add(medication_name.strip())

        if candidate.value and candidate.value.strip():
            names.add(candidate.value.strip())

        return names

    def _active_patient_prescribed_facts(
        self,
        patient_id: str,
        knowledge_state: KnowledgeState,
    ) -> list[Any]:
        prescribed = self._safe_get_facts_by_relation(
            knowledge_state,
            entity_id=patient_id,
            relation=RelationType.PRESCRIBED,
            as_subject=True,
        )
        discontinued = self._safe_get_facts_by_relation(
            knowledge_state,
            entity_id=patient_id,
            relation=RelationType.DISCONTINUED,
            as_subject=True,
        )

        latest_discontinued_by_object: dict[str, Any] = {}
        latest_discontinued_by_name: dict[str, Any] = {}

        for fact in discontinued:
            timestamp = getattr(fact, "valid_from", None)
            object_id = getattr(fact, "object_id", None)
            if object_id:
                previous = latest_discontinued_by_object.get(object_id)
                if previous is None or (
                    timestamp is not None and datetime_after(timestamp, previous)
                ):
                    latest_discontinued_by_object[object_id] = timestamp

            for med_name in self._fact_medication_names(fact, knowledge_state):
                normalized = normalize_drug_name(med_name)
                previous = latest_discontinued_by_name.get(normalized)
                if previous is None or (
                    timestamp is not None and datetime_after(timestamp, previous)
                ):
                    latest_discontinued_by_name[normalized] = timestamp

        active: list[Any] = []
        for fact in prescribed:
            prescribed_at = getattr(fact, "valid_from", None)
            object_id = getattr(fact, "object_id", None)

            if object_id and object_id in latest_discontinued_by_object:
                discontinued_at = latest_discontinued_by_object[object_id]
                if discontinued_at is None or (
                    prescribed_at is not None
                    and datetime_on_or_after(discontinued_at, prescribed_at)
                ):
                    continue

            med_names = self._fact_medication_names(fact, knowledge_state)
            skip = False
            for med_name in med_names:
                normalized = normalize_drug_name(med_name)
                if normalized not in latest_discontinued_by_name:
                    continue
                discontinued_at = latest_discontinued_by_name[normalized]
                if discontinued_at is None or (
                    prescribed_at is not None
                    and datetime_on_or_after(discontinued_at, prescribed_at)
                ):
                    skip = True
                    break
            if skip:
                continue

            active.append(fact)

        return active

    def _active_patient_medication_names(
        self,
        patient_id: str,
        knowledge_state: KnowledgeState,
    ) -> set[str]:
        names: set[str] = set()
        for fact in self._active_patient_prescribed_facts(patient_id, knowledge_state):
            names.update(self._fact_medication_names(fact, knowledge_state))
        return names

    def _active_patient_allergy_names(
        self,
        patient_id: str,
        knowledge_state: KnowledgeState,
    ) -> set[str]:
        allergies: set[str] = set()
        allergy_facts = self._safe_get_facts_by_relation(
            knowledge_state,
            entity_id=patient_id,
            relation=RelationType.HAS_ALLERGY,
            as_subject=True,
        )

        for fact in allergy_facts:
            object_name = self._safe_get_entity_name(
                knowledge_state, getattr(fact, "object_id", None)
            )
            if object_name:
                allergies.add(object_name)
            value = getattr(fact, "value", None)
            if isinstance(value, str) and value.strip():
                allergies.add(value.strip())

        return allergies

    @staticmethod
    def _normalized_drug_names(names: set[str]) -> set[str]:
        return {normalize_drug_name(name) for name in names if name and str(name).strip()}

    def _evaluate_intersection_empty(
        self, candidate: CandidateFact, knowledge_state: KnowledgeState, condition: dict[str, Any]
    ) -> ConstraintViolation | None:
        """
        Evaluates conditions like:
        check: intersection_empty
        left: medication.ingredients
        right: patient.allergies
        """
        # If it's not a medication prescription, skip
        if candidate.relation != RelationType.PRESCRIBED:
            return None

        patient_id = candidate.subject_entity_id
        candidate_medications = self._candidate_medication_names(candidate, knowledge_state)
        if not candidate_medications:
            return None

        allergy_names = self._active_patient_allergy_names(patient_id, knowledge_state)
        if not allergy_names:
            return None

        candidate_terms: set[str] = set()
        for med_name in candidate_medications:
            candidate_terms.update(
                {normalize_drug_name(med_name), *get_cross_reactive_ingredients(med_name)}
            )

        for allergy in allergy_names:
            cross_reactive = get_cross_reactive_ingredients(allergy)
            overlap = candidate_terms & cross_reactive
            if overlap:
                matched = sorted(overlap)[0]
                return ConstraintViolation(
                    constraint_id=self.constraint_id,
                    constraint_name=self.constraint_name,
                    description=(
                        f"Patient has a documented allergy to {allergy} and "
                        f"the proposed medication cross-reacts ({matched})."
                    ),
                    severity=self.severity,
                    alternatives=[
                        "Consider prescribing Macrolides or Cephalosporins instead if appropriate."
                    ],
                )

        return None

    def _evaluate_no_major_interactions(
        self, candidate: CandidateFact, knowledge_state: KnowledgeState, condition: dict[str, Any]
    ) -> ConstraintViolation | None:
        """
        Evaluates conditions like:
        check: no_major_interactions
        lookup: drug_interaction_database
        """
        if candidate.relation != RelationType.PRESCRIBED:
            return None

        medication_names = self._candidate_medication_names(candidate, knowledge_state)
        if not medication_names:
            return None

        active_prescriptions = self._active_patient_medication_names(
            candidate.subject_entity_id, knowledge_state
        )
        if not active_prescriptions:
            return None

        for medication_name in medication_names:
            for active_med in active_prescriptions:
                if check_major_interaction(medication_name, active_med):
                    return ConstraintViolation(
                        constraint_id=self.constraint_id,
                        constraint_name=self.constraint_name,
                        description=(
                            f"Major drug-drug interaction between {medication_name} "
                            f"and active medication {active_med}."
                        ),
                        severity=self.severity,
                        alternatives=["Review interactions or consider alternative therapy."],
                    )

        return None

    def _evaluate_no_moderate_interactions(
        self,
        candidate: CandidateFact,
        knowledge_state: KnowledgeState,
        condition: dict[str, Any],
    ) -> ConstraintViolation | None:
        if candidate.relation != RelationType.PRESCRIBED:
            return None

        medication_names = self._candidate_medication_names(candidate, knowledge_state)
        if not medication_names:
            return None

        active_prescriptions = self._active_patient_medication_names(
            candidate.subject_entity_id, knowledge_state
        )
        if not active_prescriptions:
            return None

        for medication_name in medication_names:
            for active_med in active_prescriptions:
                if check_moderate_interaction(medication_name, active_med):
                    return ConstraintViolation(
                        constraint_id=self.constraint_id,
                        constraint_name=self.constraint_name,
                        description=(
                            f"Moderate interaction risk between {medication_name} "
                            f"and active medication {active_med}."
                        ),
                        severity=self.severity,
                        alternatives=["Review dosing and monitoring requirements."],
                    )

        return None

    def _evaluate_cardinality_limit(
        self,
        candidate: CandidateFact,
        knowledge_state: KnowledgeState,
        condition: dict[str, Any],
    ) -> ConstraintViolation | None:
        if candidate.relation != RelationType.PRESCRIBED:
            return None

        where = str(condition.get("where", "")).strip().lower()
        try:
            max_allowed = int(condition.get("max", 1))
        except (TypeError, ValueError):
            max_allowed = 1

        if max_allowed < 1:
            max_allowed = 1

        active_facts = self._active_patient_prescribed_facts(
            candidate.subject_entity_id, knowledge_state
        )
        active_names = self._active_patient_medication_names(
            candidate.subject_entity_id, knowledge_state
        )
        candidate_names = self._candidate_medication_names(candidate, knowledge_state)
        if not candidate_names:
            return None

        if where == "duplicate_active_medication":
            candidate_normalized = self._normalized_drug_names(candidate_names)
            active_normalized = self._normalized_drug_names(active_names)

            # If this is an in-place update to the same object entity, let supersession handle it.
            if candidate.object_entity_id and any(
                getattr(fact, "object_id", None) == candidate.object_entity_id
                for fact in active_facts
            ):
                return None

            duplicates = sorted(candidate_normalized & active_normalized)
            if duplicates:
                return ConstraintViolation(
                    constraint_id=self.constraint_id,
                    constraint_name=self.constraint_name,
                    description=f"Duplicate active medication detected: {duplicates[0]}",
                    severity=self.severity,
                    alternatives=["Use existing active order or retire it before re-prescribing."],
                )

            return None

        if where == "same_therapeutic_class":
            candidate_classes: set[str] = set()
            for medication_name in candidate_names:
                candidate_classes.update(get_therapeutic_classes(medication_name))

            if not candidate_classes:
                return None

            matching_existing = []
            for active_fact in active_facts:
                if (
                    candidate.object_entity_id
                    and getattr(active_fact, "object_id", None) == candidate.object_entity_id
                ):
                    # Allow in-place updates for the same medication; supersession will reconcile versions.
                    continue

                for active_name in self._fact_medication_names(active_fact, knowledge_state):
                    if get_therapeutic_classes(active_name) & candidate_classes:
                        matching_existing.append(active_name)
                        break

            if len(matching_existing) >= max_allowed:
                classes_list = ", ".join(sorted(candidate_classes))
                return ConstraintViolation(
                    constraint_id=self.constraint_id,
                    constraint_name=self.constraint_name,
                    description=(
                        "Therapeutic duplication risk: active medications in class "
                        f"{classes_list} already meet limit ({max_allowed})."
                    ),
                    severity=self.severity,
                    alternatives=["Discontinue existing therapy or choose another drug class."],
                )

            return None

        return None


def load_healthcare_constraints() -> list[BaseConstraintEvaluator]:
    """
    Load constraints from healthcare_constraints.yaml.
    """
    config_path = Path(__file__).resolve().parents[2] / "configs" / "healthcare_constraints.yaml"

    if not config_path.exists():
        print(f"Warning: Configuration file not found at {config_path}")
        return []

    try:
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        constraints_config = data.get("constraints", [])
        evaluators = []

        for c_dict in constraints_config:
            # We only implement evaluators for certain types of checks in this demo
            condition = c_dict.get("condition", {})
            if "check" in condition:
                evaluators.append(YamlConstraintEvaluator(c_dict))

        return evaluators

    except Exception as e:
        print(f"Error loading healthcare constraints: {e}")
        return []
