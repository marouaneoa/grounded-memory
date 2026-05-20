#!/usr/bin/env python3
"""Healthcare adapter reconciliation regression tests.

Run:
    PYTHONPATH=src python -m pytest tests/test_healthcare_reconciliation.py -v
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grounded_memory import Memory
from grounded_memory.adapters.healthcare.constraints import load_healthcare_constraints
from grounded_memory.adapters.healthcare.extractor import HealthcareDatabaseExtractor
from grounded_memory.adapters.healthcare.models import (
    ClinicalExtractionResult,
    ExtractedAllergyLLM,
    ExtractedMedicationLLM,
    ExtractedPatientLLM,
)
from grounded_memory.core.constraints import ConstraintValidator
from grounded_memory.core.grounding import GroundingDecision, GroundingOperator
from grounded_memory.core.models import (
    CandidateFact,
    Entity,
    EntityType,
    Interaction,
    RelationType,
)
from grounded_memory.core.store import MemoryStore
from grounded_memory.retrieval import RelationshipPreset


class _DummyAgent:
    def process(self, *_args, **_kwargs):
        return None


class _SequenceClinicalExtractor:
    def __init__(self, results: list[ClinicalExtractionResult]):
        self._results = results
        self._index = 0

    def extract(self, _text: str) -> ClinicalExtractionResult:
        if self._index >= len(self._results):
            return self._results[-1]
        result = self._results[self._index]
        self._index += 1
        return result


def _make_store_with_operator() -> tuple[MemoryStore, GroundingOperator]:
    store = MemoryStore()
    validator = ConstraintValidator()
    for evaluator in load_healthcare_constraints():
        validator.register(evaluator)
    return store, GroundingOperator(validator=validator, memory_store=store)


def _add_interaction(store: MemoryStore, text: str) -> Interaction:
    interaction = Interaction(raw_text=text)
    store.add_interaction(interaction)
    return interaction


def test_patient_identity_disambiguates_same_name_with_identifier():
    store = MemoryStore()
    extractor = HealthcareDatabaseExtractor(
        store=store,
        llm_extractor=_SequenceClinicalExtractor(
            [
                ClinicalExtractionResult(
                    patient=ExtractedPatientLLM(name="Faycel", identifier="MRN-1001"),
                    medications=[],
                    allergies=[],
                    conditions=[],
                ),
                ClinicalExtractionResult(
                    patient=ExtractedPatientLLM(name="Faycel", identifier="MRN-2002"),
                    medications=[],
                    allergies=[],
                    conditions=[],
                ),
            ]
        ),
    )

    asyncio.run(extractor.process_interaction("Faycel is admitted", metadata={}))
    asyncio.run(extractor.process_interaction("Faycel is in room 5", metadata={}))

    patients = store.get_entities_by_type(EntityType.PATIENT)
    assert len(patients) == 2
    canonical_ids = {entity.canonical_id for entity in patients}
    assert canonical_ids == {"MRN-1001", "MRN-2002"}


def test_medication_dosage_is_stored_on_prescription_relation_not_medication_node():
    store = MemoryStore()
    store.add_candidate_facts = lambda facts: None
    extractor = HealthcareDatabaseExtractor(
        store=store,
        llm_extractor=_SequenceClinicalExtractor(
            [
                ClinicalExtractionResult(
                    patient=ExtractedPatientLLM(name="John Doe", identifier="MRN-1001"),
                    medications=[
                        ExtractedMedicationLLM(
                            name="Lisinopril",
                            dosage="10mg",
                            frequency="daily",
                            action="prescribe",
                        )
                    ],
                    allergies=[],
                    conditions=[],
                ),
                ClinicalExtractionResult(
                    patient=ExtractedPatientLLM(name="Jane Doe", identifier="MRN-2002"),
                    medications=[
                        ExtractedMedicationLLM(
                            name="Lisinopril",
                            dosage="20mg",
                            frequency="daily",
                            action="prescribe",
                        )
                    ],
                    allergies=[],
                    conditions=[],
                ),
            ]
        ),
    )

    first = asyncio.run(
        extractor.process_interaction(
            "Prescribe Lisinopril 10mg daily for John Doe MRN-1001",
            metadata={},
        )
    )
    second = asyncio.run(
        extractor.process_interaction(
            "Prescribe Lisinopril 20mg daily for Jane Doe MRN-2002",
            metadata={},
        )
    )

    medication_entities = store.get_entities_by_type(EntityType.MEDICATION)
    assert len(medication_entities) == 1

    medication_entity = medication_entities[0]
    assert medication_entity.name == "Lisinopril"
    assert medication_entity.attributes.get("normalized_name") == "lisinopril"
    assert "dosage" not in medication_entity.attributes
    assert "frequency" not in medication_entity.attributes
    assert "route" not in medication_entity.attributes
    assert "duration" not in medication_entity.attributes

    first_fact = first.candidate_facts[0]
    second_fact = second.candidate_facts[0]
    assert first_fact.object_entity_id == medication_entity.id
    assert second_fact.object_entity_id == medication_entity.id
    assert first_fact.attributes["dosage"] == "10mg"
    assert second_fact.attributes["dosage"] == "20mg"
    assert first_fact.attributes["frequency"] == "daily"
    assert second_fact.attributes["frequency"] == "daily"


def test_allergy_node_is_shared_but_patient_allergy_relations_are_distinct():
    store = MemoryStore()
    store.add_candidate_facts = lambda facts: None
    extractor = HealthcareDatabaseExtractor(
        store=store,
        llm_extractor=_SequenceClinicalExtractor(
            [
                ClinicalExtractionResult(
                    patient=ExtractedPatientLLM(name="John Doe", identifier="JD-001"),
                    medications=[],
                    allergies=[
                        ExtractedAllergyLLM(
                            allergen="Penicillin",
                            reaction="anaphylaxis",
                            severity="severe",
                        )
                    ],
                    conditions=[],
                ),
                ClinicalExtractionResult(
                    patient=ExtractedPatientLLM(name="Haroun", identifier="JD-045"),
                    medications=[],
                    allergies=[
                        ExtractedAllergyLLM(
                            allergen="Penicillin",
                            reaction="anaphylaxis",
                            severity="severe",
                        )
                    ],
                    conditions=[],
                ),
            ]
        ),
    )

    first = asyncio.run(
        extractor.process_interaction(
            "Patient John Doe, MRN JD-001, has a severe Penicillin allergy.",
            metadata={"tenant_id": "demo", "app_id": "gmem", "user_id": "demo-user"},
        )
    )
    second = asyncio.run(
        extractor.process_interaction(
            "Patient Haroun, MRN JD-045, has a severe Penicillin allergy.",
            metadata={"tenant_id": "demo", "app_id": "gmem", "user_id": "demo-user"},
        )
    )

    allergy_entities = store.get_entities_by_type(EntityType.ALLERGY)
    patient_entities = store.get_entities_by_type(EntityType.PATIENT)

    assert len(allergy_entities) == 1
    assert len(patient_entities) == 2
    assert "scope_id" not in allergy_entities[0].attributes

    first_fact = first.candidate_facts[0]
    second_fact = second.candidate_facts[0]
    assert first_fact.relation == RelationType.HAS_ALLERGY
    assert second_fact.relation == RelationType.HAS_ALLERGY
    assert first_fact.subject_entity_id != second_fact.subject_entity_id
    assert first_fact.object_entity_id == allergy_entities[0].id
    assert second_fact.object_entity_id == allergy_entities[0].id


def test_allergy_conflict_rejects_cross_reactive_prescription():
    store, operator = _make_store_with_operator()

    patient = Entity(entity_type=EntityType.PATIENT, name="Faycel")
    allergy = Entity(entity_type=EntityType.ALLERGY, name="Penicillin")
    medication = Entity(entity_type=EntityType.MEDICATION, name="Amoxicillin")

    store.add_entity(patient)
    store.add_entity(allergy)
    store.add_entity(medication)

    interaction_1 = _add_interaction(store, "Faycel is allergic to Penicillin")
    allergy_fact = CandidateFact(
        source_interaction_id=interaction_1.id,
        subject_entity_id=patient.id,
        relation=RelationType.HAS_ALLERGY,
        object_entity_id=allergy.id,
        confidence=0.99,
    )
    allergy_result = operator.ground(allergy_fact)
    assert allergy_result.decision in {GroundingDecision.APPROVED, GroundingDecision.SUPERSEDED}

    interaction_2 = _add_interaction(store, "Prescribe Amoxicillin to Faycel")
    med_fact = CandidateFact(
        source_interaction_id=interaction_2.id,
        subject_entity_id=patient.id,
        relation=RelationType.PRESCRIBED,
        object_entity_id=medication.id,
        confidence=0.95,
        attributes={"medication_name": "Amoxicillin"},
    )
    med_result = operator.ground(med_fact)

    assert med_result.decision == GroundingDecision.REJECTED
    assert med_result.rejection_record is not None
    assert med_result.rejection_record.constraint_id == "allergy_conflict"


def test_major_interaction_rejects_conflicting_prescription():
    store, operator = _make_store_with_operator()

    patient = Entity(entity_type=EntityType.PATIENT, name="Faycel")
    warfarin = Entity(entity_type=EntityType.MEDICATION, name="Warfarin")
    amiodarone = Entity(entity_type=EntityType.MEDICATION, name="Amiodarone")

    store.add_entity(patient)
    store.add_entity(warfarin)
    store.add_entity(amiodarone)

    interaction_1 = _add_interaction(store, "Patient takes warfarin")
    baseline = CandidateFact(
        source_interaction_id=interaction_1.id,
        subject_entity_id=patient.id,
        relation=RelationType.PRESCRIBED,
        object_entity_id=warfarin.id,
        confidence=0.95,
        attributes={"medication_name": "Warfarin"},
    )
    baseline_result = operator.ground(baseline)
    assert baseline_result.is_success

    interaction_2 = _add_interaction(store, "Start amiodarone")
    conflicting = CandidateFact(
        source_interaction_id=interaction_2.id,
        subject_entity_id=patient.id,
        relation=RelationType.PRESCRIBED,
        object_entity_id=amiodarone.id,
        confidence=0.95,
        attributes={"medication_name": "Amiodarone"},
    )
    conflict_result = operator.ground(conflicting)

    assert conflict_result.decision == GroundingDecision.REJECTED
    assert conflict_result.rejection_record is not None
    assert conflict_result.rejection_record.constraint_id == "drug_interaction_major"


def test_prescription_update_triggers_supersession():
    store, operator = _make_store_with_operator()

    patient = Entity(entity_type=EntityType.PATIENT, name="Faycel")
    amoxicillin = Entity(entity_type=EntityType.MEDICATION, name="Amoxicillin")

    store.add_entity(patient)
    store.add_entity(amoxicillin)

    interaction_1 = _add_interaction(store, "Amoxicillin 500mg bid")
    old_fact = CandidateFact(
        source_interaction_id=interaction_1.id,
        subject_entity_id=patient.id,
        relation=RelationType.PRESCRIBED,
        object_entity_id=amoxicillin.id,
        value="500mg twice daily",
        confidence=0.91,
        attributes={"medication_name": "Amoxicillin", "dosage": "500mg", "frequency": "bid"},
    )
    old_result = operator.ground(old_fact)
    assert old_result.decision == GroundingDecision.APPROVED
    assert old_result.validated_fact is not None

    interaction_2 = _add_interaction(store, "Amoxicillin adjusted to 250mg daily")
    new_fact = CandidateFact(
        source_interaction_id=interaction_2.id,
        subject_entity_id=patient.id,
        relation=RelationType.PRESCRIBED,
        object_entity_id=amoxicillin.id,
        value="250mg daily",
        confidence=0.94,
        attributes={"medication_name": "Amoxicillin", "dosage": "250mg", "frequency": "daily"},
    )
    new_result = operator.ground(new_fact)

    assert new_result.decision == GroundingDecision.SUPERSEDED
    assert old_result.validated_fact.id in {fact.id for fact in new_result.superseded_facts}

    old_record = store.get_validated_fact(old_result.validated_fact.id)
    assert old_record is not None
    assert old_record.valid_to is not None


def test_healthcare_memory_uses_safety_retrieval_profile_by_default():
    memory = Memory(
        adapter="healthcare",
        storage_backend="memory",
        use_llm=False,
        agent=_DummyAgent(),
    )

    try:
        assert memory.relationship_preset == RelationshipPreset.SAFETY
        has_allergy_weight = memory.retriever.get_weight(RelationType.HAS_ALLERGY)
        assert has_allergy_weight.weight == pytest.approx(10.0)
        assert has_allergy_weight.decay_per_hop == pytest.approx(0.1)
    finally:
        memory.close()
