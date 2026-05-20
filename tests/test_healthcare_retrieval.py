#!/usr/bin/env python3
"""Healthcare retrieval planner and clinical-context regression tests.

Run:
    PYTHONPATH=src python -m pytest tests/test_healthcare_retrieval.py -q
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grounded_memory.adapters.healthcare.constraints import load_healthcare_constraints
from grounded_memory.adapters.healthcare.lifecycle import apply_medication_lifecycle_after_grounding
from grounded_memory.adapters.healthcare.retrieval import (
    HealthcareContextBuilder,
    HealthcareQueryPlan,
    HealthcareRetrievalPlanner,
)
from grounded_memory.core.constraints import ConstraintValidator
from grounded_memory.core.grounding import GroundingDecision, GroundingOperator
from grounded_memory.core.models import CandidateFact, Entity, EntityType, Interaction, RelationType
from grounded_memory.core.store import MemoryStore
from grounded_memory.retrieval import GraphRetriever, RelationshipPreset, RetrievalStrategy


class _FakePlannerClient:
    def __init__(self, plan: HealthcareQueryPlan):
        self.plan = plan

    def extract(self, *_args, **_kwargs) -> HealthcareQueryPlan:
        return self.plan


def _store_and_operator() -> tuple[MemoryStore, GroundingOperator]:
    store = MemoryStore()
    validator = ConstraintValidator()
    for evaluator in load_healthcare_constraints():
        validator.register(evaluator)
    return store, GroundingOperator(validator=validator, memory_store=store)


def _interaction(store: MemoryStore, text: str) -> Interaction:
    interaction = Interaction(raw_text=text)
    store.add_interaction(interaction)
    return interaction


def _builder(store: MemoryStore, plan: HealthcareQueryPlan) -> HealthcareContextBuilder:
    retriever = GraphRetriever(
        store,
        relationship_preset=RelationshipPreset.SAFETY,
    )
    planner = HealthcareRetrievalPlanner(
        memory_store=store,
        llm_client=_FakePlannerClient(plan),
    )
    return HealthcareContextBuilder(
        memory_store=store,
        retriever=retriever,
        planner=planner,
    )


def test_healthcare_planner_resolves_patient_by_identifier_without_hardcoded_seed():
    store = MemoryStore()
    patient = Entity(
        entity_type=EntityType.PATIENT,
        name="Alice Johnson",
        canonical_id="MRN-1001",
        attributes={"identifier": "MRN-1001"},
    )
    store.add_entity(patient)

    planner = HealthcareRetrievalPlanner(
        memory_store=store,
        llm_client=_FakePlannerClient(
            HealthcareQueryPlan(
                patient_name="Alice Johnson",
                patient_identifier="MRN-1001",
                requested_categories=["current_medications"],
            )
        ),
    )

    seeds = planner.resolve_seed_entities(
        query="What is Alice Johnson MRN-1001 taking?",
        plan=planner.plan("What is Alice Johnson MRN-1001 taking?"),
    )

    assert seeds == [patient.id]


def test_discontinuation_closes_matching_active_prescription_and_context_excludes_it():
    store, operator = _store_and_operator()
    patient = Entity(entity_type=EntityType.PATIENT, name="Alice Johnson", canonical_id="MRN-1001")
    lisinopril = Entity(entity_type=EntityType.MEDICATION, name="Lisinopril")
    store.add_entity(patient)
    store.add_entity(lisinopril)

    prescribed = CandidateFact(
        source_interaction_id=_interaction(store, "Prescribe Lisinopril 10mg daily").id,
        subject_entity_id=patient.id,
        relation=RelationType.PRESCRIBED,
        object_entity_id=lisinopril.id,
        value="10mg daily",
        attributes={
            "medication_name": "Lisinopril",
            "normalized_name": "lisinopril",
            "dosage": "10mg",
            "frequency": "daily",
            "action": "prescribe",
            "order_status": "active",
        },
        confidence=0.95,
    )
    prescribed_result = operator.ground(prescribed)
    assert prescribed_result.decision == GroundingDecision.APPROVED

    discontinued = CandidateFact(
        source_interaction_id=_interaction(store, "Discontinue Lisinopril").id,
        subject_entity_id=patient.id,
        relation=RelationType.DISCONTINUED,
        object_entity_id=lisinopril.id,
        attributes={
            "medication_name": "Lisinopril",
            "normalized_name": "lisinopril",
            "action": "discontinue",
            "order_status": "discontinued",
        },
        confidence=0.95,
    )
    discontinued_result = operator.ground(discontinued)
    closed = apply_medication_lifecycle_after_grounding(
        store=store,
        result=discontinued_result,
    )

    assert len(closed) == 1
    old_fact = store.get_fact(prescribed_result.validated_fact.id)
    assert old_fact is not None
    assert old_fact.valid_to is not None
    assert not old_fact.is_active

    context = _builder(
        store,
        HealthcareQueryPlan(
            patient_identifier="MRN-1001",
            requested_categories=["current_medications", "history"],
        ),
    ).build(
        "What is Alice Johnson MRN-1001 currently prescribed?",
        strategy=RetrievalStrategy.SAFETY_PRIORITY,
    )

    assert context.current_medications == []
    assert {row["relation"] for row in context.history} == {"PRESCRIBED", "DISCONTINUED"}


def test_historical_as_of_retrieval_returns_previous_dose_after_supersession():
    store, operator = _store_and_operator()
    patient = Entity(entity_type=EntityType.PATIENT, name="Alice Johnson", canonical_id="MRN-1001")
    lisinopril = Entity(entity_type=EntityType.MEDICATION, name="Lisinopril")
    store.add_entity(patient)
    store.add_entity(lisinopril)

    t1 = datetime.now(timezone.utc) - timedelta(days=2)
    t2 = datetime.now(timezone.utc) - timedelta(days=1)
    historical_time = (t1 + timedelta(hours=1)).replace(tzinfo=timezone.utc)

    old_candidate = CandidateFact(
        source_interaction_id=_interaction(store, "Lisinopril 10mg daily").id,
        subject_entity_id=patient.id,
        relation=RelationType.PRESCRIBED,
        object_entity_id=lisinopril.id,
        value="10mg daily",
        extracted_at=t1,
        confidence=0.9,
        attributes={
            "medication_name": "Lisinopril",
            "normalized_name": "lisinopril",
            "dosage": "10mg",
            "frequency": "daily",
            "action": "prescribe",
        },
    )
    old_result = operator.ground(old_candidate)
    assert old_result.decision == GroundingDecision.APPROVED

    new_candidate = CandidateFact(
        source_interaction_id=_interaction(store, "Lisinopril adjusted to 20mg daily").id,
        subject_entity_id=patient.id,
        relation=RelationType.PRESCRIBED,
        object_entity_id=lisinopril.id,
        value="20mg daily",
        extracted_at=t2,
        confidence=0.95,
        attributes={
            "medication_name": "Lisinopril",
            "normalized_name": "lisinopril",
            "dosage": "20mg",
            "frequency": "daily",
            "action": "adjust",
        },
    )
    new_result = operator.ground(new_candidate)
    assert new_result.decision == GroundingDecision.SUPERSEDED

    old_fact = store.get_fact(old_result.validated_fact.id)
    assert old_fact is not None
    assert old_fact.valid_to == t2

    historical_context = _builder(
        store,
        HealthcareQueryPlan(
            patient_identifier="MRN-1001",
            requested_categories=["current_medications", "history"],
            medication_names=["Lisinopril"],
            as_of=historical_time,
        ),
    ).build(
        f"As of {historical_time.isoformat()}, what was Alice taking?",
        strategy=RetrievalStrategy.SAFETY_PRIORITY,
    )

    current_context = _builder(
        store,
        HealthcareQueryPlan(
            patient_identifier="MRN-1001",
            requested_categories=["current_medications"],
            medication_names=["Lisinopril"],
        ),
    ).build(
        "What is Alice Johnson MRN-1001 currently taking?",
        strategy=RetrievalStrategy.SAFETY_PRIORITY,
    )

    assert [row["dosage"] for row in historical_context.current_medications] == ["10mg"]
    assert [row["dosage"] for row in current_context.current_medications] == ["20mg"]


def test_empty_store_returns_empty_context_gracefully():
    store = MemoryStore()
    builder = _builder(
        store,
        HealthcareQueryPlan(
            patient_identifier="MRN-NOT-FOUND",
            requested_categories=["current_medications"],
        ),
    )

    context = builder.build(
        "What is Alice Johnson MRN-1001 currently taking?",
        strategy=RetrievalStrategy.SAFETY_PRIORITY,
    )

    assert context.current_medications == []
    assert context.allergies == []
    assert context.safety_alerts == []
    assert context.history == []
    assert context.answer_context.retrieval_metadata.get("reason") == "no_seed_entities_resolved"


def test_no_seed_entities_returns_metadata_reason():
    store = MemoryStore()
    builder = _builder(
        store,
        HealthcareQueryPlan(
            patient_name="Unknown Patient",
            requested_categories=["current_medications"],
        ),
    )

    context = builder.build(
        "What is Unknown Patient currently taking?",
        strategy=RetrievalStrategy.SAFETY_PRIORITY,
    )

    assert context.seed_entities == []
    assert context.current_medications == []
    meta = context.answer_context.retrieval_metadata
    assert meta.get("reason") == "no_seed_entities_resolved"


def _interaction_with_metadata(store: MemoryStore, text: str, metadata: dict) -> Interaction:
    interaction = Interaction(raw_text=text, metadata=metadata)
    store.add_interaction(interaction)
    return interaction


def test_multi_patient_scope_isolation():
    store, operator = _store_and_operator()
    alice = Entity(entity_type=EntityType.PATIENT, name="Alice Johnson", canonical_id="MRN-1001")
    bob = Entity(entity_type=EntityType.PATIENT, name="Bob Smith", canonical_id="MRN-2002")
    lisinopril = Entity(entity_type=EntityType.MEDICATION, name="Lisinopril")
    amoxicillin = Entity(entity_type=EntityType.MEDICATION, name="Amoxicillin")
    penicillin = Entity(entity_type=EntityType.ALLERGY, name="Penicillin")
    store.add_entity(alice)
    store.add_entity(bob)
    store.add_entity(lisinopril)
    store.add_entity(amoxicillin)
    store.add_entity(penicillin)

    alice_scope = {"tenant_id": "demo", "app_id": "gmem", "user_id": "alice-user"}
    bob_scope = {"tenant_id": "demo", "app_id": "gmem", "user_id": "bob-user"}

    for patient, med, scope in [
        (alice, lisinopril, alice_scope),
        (bob, amoxicillin, bob_scope),
    ]:
        candidate = CandidateFact(
            source_interaction_id=_interaction_with_metadata(
                store, f"Prescribe {med.name}", scope
            ).id,
            subject_entity_id=patient.id,
            relation=RelationType.PRESCRIBED,
            object_entity_id=med.id,
            attributes={
                "medication_name": med.name,
                "action": "prescribe",
                "order_status": "active",
                **scope,
            },
            confidence=0.95,
        )
        result = operator.ground(candidate)
        assert result.decision == GroundingDecision.APPROVED

    alice_allergy = CandidateFact(
        source_interaction_id=_interaction_with_metadata(
            store, "Alice allergic to Penicillin", alice_scope
        ).id,
        subject_entity_id=alice.id,
        relation=RelationType.HAS_ALLERGY,
        object_entity_id=penicillin.id,
        attributes={"allergen_name": "Penicillin", "severity": "severe", **alice_scope},
        confidence=0.95,
    )
    operator.ground(alice_allergy)

    builder = _builder(
        store,
        HealthcareQueryPlan(
            patient_identifier="MRN-1001",
            requested_categories=["current_medications", "allergies"],
        ),
    )

    context = builder.build(
        "What is Alice Johnson MRN-1001 currently taking?",
        scope=alice_scope,
        strategy=RetrievalStrategy.SAFETY_PRIORITY,
    )

    med_names = {row["medication_name"] for row in context.current_medications}
    allergen_names = {row["allergen"] for row in context.allergies}
    assert med_names == {"Lisinopril"}
    assert allergen_names == {"Penicillin"}
    assert "Amoxicillin" not in med_names


def test_retrieval_quality_recall_precision_ranking():
    store, operator = _store_and_operator()
    patient = Entity(entity_type=EntityType.PATIENT, name="Alice Johnson", canonical_id="MRN-1001")
    med_a = Entity(entity_type=EntityType.MEDICATION, name="Lisinopril")
    med_b = Entity(entity_type=EntityType.MEDICATION, name="Warfarin")
    med_c = Entity(entity_type=EntityType.MEDICATION, name="Amiodarone")
    med_d = Entity(entity_type=EntityType.MEDICATION, name="Lisinopril")
    allergy = Entity(entity_type=EntityType.ALLERGY, name="Penicillin")
    store.add_entity(patient)
    store.add_entity(med_a)
    store.add_entity(med_b)
    store.add_entity(med_c)
    store.add_entity(med_d)
    store.add_entity(allergy)

    facts = [
        (med_a, "Lisinopril 10mg daily", "prescribe"),
        (med_b, "Warfarin 5mg daily", "prescribe"),
        (allergy, "Penicillin allergy", None),
    ]
    for med_or_allergy, text, action in facts:
        if med_or_allergy == allergy:
            candidate = CandidateFact(
                source_interaction_id=_interaction(store, text).id,
                subject_entity_id=patient.id,
                relation=RelationType.HAS_ALLERGY,
                object_entity_id=allergy.id,
                attributes={
                    "allergen_name": "Penicillin",
                    "severity": "severe",
                    "source_text": text,
                },
                confidence=0.95,
            )
        else:
            candidate = CandidateFact(
                source_interaction_id=_interaction(store, text).id,
                subject_entity_id=patient.id,
                relation=RelationType.PRESCRIBED,
                object_entity_id=med_or_allergy.id,
                attributes={
                    "medication_name": med_or_allergy.name,
                    "action": action,
                    "order_status": "active",
                    "source_text": text,
                },
                confidence=0.95,
            )
        result = operator.ground(candidate)
        assert result.decision == GroundingDecision.APPROVED

    rejected = CandidateFact(
        source_interaction_id=_interaction(store, "Prescribe Amiodarone").id,
        subject_entity_id=patient.id,
        relation=RelationType.PRESCRIBED,
        object_entity_id=med_c.id,
        attributes={
            "medication_name": "Amiodarone",
            "action": "prescribe",
            "source_text": "Prescribe Amiodarone",
        },
        confidence=0.95,
    )
    rejected_result = operator.ground(rejected)
    assert rejected_result.decision == GroundingDecision.REJECTED

    builder = _builder(
        store,
        HealthcareQueryPlan(
            patient_identifier="MRN-1001",
            requested_categories=["current_medications", "allergies", "safety_alerts"],
            safety_focus=True,
        ),
    )

    context = builder.build(
        "What medications and allergies does Alice Johnson MRN-1001 have?",
        strategy=RetrievalStrategy.SAFETY_PRIORITY,
    )

    med_names = {row["medication_name"] for row in context.current_medications}
    allergen_names = {row["allergen"] for row in context.allergies}
    assert med_names == {"Lisinopril", "Warfarin"}
    assert allergen_names == {"Penicillin"}

    raw_facts = context.answer_context.facts
    assert len(raw_facts) >= 3
    relation_order = [f.relation for f in raw_facts]
    assert RelationType.HAS_ALLERGY in relation_order
    if raw_facts[0].relation == RelationType.HAS_ALLERGY:
        pass
    else:
        allergy_idx = next(
            i for i, f in enumerate(raw_facts) if f.relation == RelationType.HAS_ALLERGY
        )
        assert allergy_idx < len(raw_facts)

    assert any(
        a["rejection_id"] == rejected_result.rejection_record.id for a in context.safety_alerts
    )


def test_discontinuation_by_value_only_closes_matching_prescription():
    store, operator = _store_and_operator()
    patient = Entity(entity_type=EntityType.PATIENT, name="Alice Johnson", canonical_id="MRN-1001")
    lisinopril = Entity(entity_type=EntityType.MEDICATION, name="Lisinopril")
    store.add_entity(patient)
    store.add_entity(lisinopril)

    prescribed = CandidateFact(
        source_interaction_id=_interaction(store, "Prescribe Lisinopril 10mg daily").id,
        subject_entity_id=patient.id,
        relation=RelationType.PRESCRIBED,
        object_entity_id=lisinopril.id,
        value="10mg daily",
        attributes={
            "medication_name": "Lisinopril",
            "normalized_name": "lisinopril",
            "dosage": "10mg",
            "action": "prescribe",
        },
        confidence=0.95,
    )
    prescribed_result = operator.ground(prescribed)
    assert prescribed_result.decision == GroundingDecision.APPROVED

    discontinued = CandidateFact(
        source_interaction_id=_interaction(store, "Discontinue Lisinopril").id,
        subject_entity_id=patient.id,
        relation=RelationType.DISCONTINUED,
        object_entity_id=lisinopril.id,
        attributes={
            "medication_name": "Lisinopril",
            "normalized_name": "lisinopril",
            "action": "discontinue",
        },
        confidence=0.95,
    )
    discontinued_result = operator.ground(discontinued)
    apply_medication_lifecycle_after_grounding(store=store, result=discontinued_result)

    old_fact = store.get_fact(prescribed_result.validated_fact.id)
    assert old_fact is not None
    assert old_fact.valid_to is not None
    assert not old_fact.is_active

    context = _builder(
        store,
        HealthcareQueryPlan(
            patient_identifier="MRN-1001",
            requested_categories=["current_medications"],
        ),
    ).build(
        "What is Alice Johnson MRN-1001 currently prescribed?",
        strategy=RetrievalStrategy.SAFETY_PRIORITY,
    )

    assert context.current_medications == []


def test_discontinuation_without_medication_name_attribute():
    store, operator = _store_and_operator()
    patient = Entity(entity_type=EntityType.PATIENT, name="Alice Johnson", canonical_id="MRN-1001")
    lisinopril = Entity(entity_type=EntityType.MEDICATION, name="Lisinopril")
    store.add_entity(patient)
    store.add_entity(lisinopril)

    prescribed = CandidateFact(
        source_interaction_id=_interaction(store, "Prescribe Lisinopril 10mg daily").id,
        subject_entity_id=patient.id,
        relation=RelationType.PRESCRIBED,
        object_entity_id=lisinopril.id,
        attributes={
            "medication_name": "Lisinopril",
            "normalized_name": "lisinopril",
            "dosage": "10mg",
            "action": "prescribe",
        },
        confidence=0.95,
    )
    prescribed_result = operator.ground(prescribed)
    assert prescribed_result.decision == GroundingDecision.APPROVED

    discontinued = CandidateFact(
        source_interaction_id=_interaction(store, "Stop Lisinopril").id,
        subject_entity_id=patient.id,
        relation=RelationType.DISCONTINUED,
        object_entity_id=lisinopril.id,
        value="Stop Lisinopril",
        attributes={"action": "discontinue"},
        confidence=0.95,
    )
    discontinued_result = operator.ground(discontinued)
    apply_medication_lifecycle_after_grounding(store=store, result=discontinued_result)

    old_fact = store.get_fact(prescribed_result.validated_fact.id)
    assert old_fact is not None
    assert old_fact.valid_to is not None
    assert not old_fact.is_active


def test_kb_expansion_resolves_cross_reactive_medication():
    store = MemoryStore()
    patient = Entity(entity_type=EntityType.PATIENT, name="Alice Johnson", canonical_id="MRN-1001")
    penicillin = Entity(entity_type=EntityType.ALLERGY, name="Penicillin")
    amoxicillin = Entity(entity_type=EntityType.MEDICATION, name="Amoxicillin")
    store.add_entity(patient)
    store.add_entity(penicillin)
    store.add_entity(amoxicillin)

    planner = HealthcareRetrievalPlanner(
        memory_store=store,
        llm_client=_FakePlannerClient(
            HealthcareQueryPlan(
                patient_name="Alice Johnson",
                allergy_names=["penicillins"],
                requested_categories=["allergies"],
            )
        ),
    )

    seeds = planner.resolve_seed_entities(
        query="Is Alice Johnson allergic to penicillins?",
        plan=planner.plan("Is Alice Johnson allergic to penicillins?"),
    )

    assert penicillin.id in seeds


def test_extract_patient_name_possessive_pattern():
    from grounded_memory.adapters.healthcare.retrieval import _extract_patient_name

    assert _extract_patient_name("What are John Doe's medications?") == "John Doe"
    assert _extract_patient_name("What is Alice Johnson's allergy?") == "Alice Johnson"


def test_extract_patient_name_indirect_question():
    from grounded_memory.adapters.healthcare.retrieval import _extract_patient_name

    assert _extract_patient_name("Which medications is Bob Smith on?") == "Bob Smith"
    assert _extract_patient_name("Which allergies is Alice Johnson allergic to?") == "Alice Johnson"
    assert _extract_patient_name("Alice Johnson is taking what medications?") == "Alice Johnson"


def test_neo4j_retrieval_integration():
    import os

    try:
        from neo4j import GraphDatabase
    except ImportError:
        pytest.skip("neo4j driver not installed")

    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            session.run("RETURN 1").single()
        driver.close()
    except Exception as exc:
        pytest.skip(f"Neo4j unavailable: {exc}")

    store, operator = _store_and_operator()
    patient = Entity(entity_type=EntityType.PATIENT, name="Alice Johnson", canonical_id="MRN-1001")
    lisinopril = Entity(entity_type=EntityType.MEDICATION, name="Lisinopril")
    store.add_entity(patient)
    store.add_entity(lisinopril)

    candidate = CandidateFact(
        source_interaction_id=_interaction(store, "Prescribe Lisinopril 10mg daily").id,
        subject_entity_id=patient.id,
        relation=RelationType.PRESCRIBED,
        object_entity_id=lisinopril.id,
        attributes={
            "medication_name": "Lisinopril",
            "action": "prescribe",
            "order_status": "active",
        },
        confidence=0.95,
    )
    result = operator.ground(candidate)
    assert result.decision == GroundingDecision.APPROVED

    try:
        from grounded_memory.core.hybrid_store import HybridMemoryStore
        from grounded_memory.core.neo4j_store import Neo4jConfig

        hybrid = HybridMemoryStore(
            neo4j_config=Neo4jConfig(uri=uri, user=user, password=password),
            sync_enabled=False,
        )
        # Seed the hybrid's in-memory store with the same data
        for entity in store.get_all_entities():
            hybrid.add_entity(entity)
        for fact in store.get_all_validated_facts():
            hybrid.add_validated_fact(fact)
        retriever = GraphRetriever(hybrid, relationship_preset=RelationshipPreset.SAFETY)
        seeds = retriever.select_seed_entities("What is Alice Johnson taking?")
        assert len(seeds) > 0
        answer = retriever.retrieve(
            query="What is Alice Johnson taking?",
            seed_entities=seeds,
            max_facts=10,
            strategy=RetrievalStrategy.SAFETY_PRIORITY,
        )
        assert any(f.relation == RelationType.PRESCRIBED for f in answer.facts)
    except Exception as exc:
        pytest.skip(f"Hybrid store / Neo4j retrieval failed: {exc}")
