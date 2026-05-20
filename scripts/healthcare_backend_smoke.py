#!/usr/bin/env python3
"""Postgres+Neo4j healthcare smoke for the thesis demo stack.

Run:
    make services-up
    PYTHONPATH=src python scripts/healthcare_backend_smoke.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
load_dotenv(REPO_ROOT / ".env", override=False)

from grounded_memory.adapters.healthcare.lifecycle import (  # noqa: E402
    apply_medication_lifecycle_after_grounding,
)
from grounded_memory.core.models import (  # noqa: E402
    CandidateFact,
    Entity,
    EntityType,
    Interaction,
    RelationType,
)
from grounded_memory.system import GroundedMemorySystem  # noqa: E402


def main() -> int:
    system = GroundedMemorySystem(
        adapter="healthcare",
        storage_backend="postgres_hybrid",
    )

    try:
        store = system.memory_store
        patient = Entity(
            entity_type=EntityType.PATIENT,
            name="Smoke Patient",
            canonical_id="SMOKE-MRN-001",
            attributes={
                "tenant_id": "smoke-tenant",
                "app_id": "ground-memory-core",
                "user_id": "healthcare-smoke",
                "scope_id": "smoke-tenant:ground-memory-core:healthcare-smoke",
            },
        )
        med = Entity(
            entity_type=EntityType.MEDICATION,
            name="Lisinopril",
            attributes=patient.attributes,
        )
        store.add_entity(patient)
        store.add_entity(med)

        interaction = Interaction(
            raw_text="Healthcare backend smoke: prescribe then discontinue lisinopril.",
            metadata=patient.attributes,
        )
        store.add_interaction(interaction)

        prescribed = CandidateFact(
            source_interaction_id=interaction.id,
            subject_entity_id=patient.id,
            relation=RelationType.PRESCRIBED,
            object_entity_id=med.id,
            value="10mg daily",
            attributes={
                **patient.attributes,
                "medication_name": "Lisinopril",
                "normalized_name": "lisinopril",
                "dosage": "10mg",
                "frequency": "daily",
                "action": "prescribe",
            },
        )
        prescribed_result = system.grounding_operator.ground(prescribed)
        if not prescribed_result.is_success:
            raise RuntimeError(prescribed_result.get_explanation())

        discontinued = CandidateFact(
            source_interaction_id=interaction.id,
            subject_entity_id=patient.id,
            relation=RelationType.DISCONTINUED,
            object_entity_id=med.id,
            attributes={
                **patient.attributes,
                "medication_name": "Lisinopril",
                "normalized_name": "lisinopril",
                "action": "discontinue",
            },
        )
        discontinued_result = system.grounding_operator.ground(discontinued)
        closed = apply_medication_lifecycle_after_grounding(
            store=store,
            result=discontinued_result,
        )
        if not discontinued_result.is_success or not closed:
            raise RuntimeError("Expected discontinuation to close active prescription")

        stats = system.get_statistics()
        print(json.dumps(stats, indent=2, default=str))
        if not stats.get("neo4j_available"):
            raise RuntimeError("Neo4j was not available")
        if not stats.get("postgres_available"):
            raise RuntimeError("PostgreSQL was not available")
        return 0
    finally:
        system.close()


if __name__ == "__main__":
    raise SystemExit(main())
