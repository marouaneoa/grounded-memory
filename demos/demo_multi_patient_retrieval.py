#!/usr/bin/env python3
"""Multi-patient healthcare retrieval scale demo for GMem.

This demo tests the retrieval phase of the healthcare application.
It requires that `demo_healthcare_multi_patient.py` has been run first to populate the database.

Run:
    make services-up
    PYTHONPATH=src python demos/demo_multi_patient_retrieval.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "demos"))
load_dotenv(REPO_ROOT / ".env", override=False)

from demo_healthcare_data import BASE_SCOPE, PATIENTS  # noqa: E402
from gmem import Memory  # noqa: E402
from grounded_memory.adapters.healthcare.retrieval import (  # noqa: E402
    HealthcareRetrievalService,
)
from grounded_memory.core.models import RelationType  # noqa: E402
from grounded_memory.llm.client import LLMConfig, SyncLLMClient  # noqa: E402


def _print_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _direct_backend_counts() -> dict[str, Any]:
    payload: dict[str, Any] = {"postgres": {}, "neo4j": {}}

    try:
        import psycopg2

        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "gmem"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            connect_timeout=3,
        )
        with conn, conn.cursor() as cur:
            for table in [
                "entities",
                "candidate_facts",
                "validated_facts",
                "interactions",
                "rejection_records",
            ]:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                payload["postgres"][table] = int(cur.fetchone()[0])
        conn.close()
    except Exception as exc:
        payload["postgres"]["error"] = f"{type(exc).__name__}: {exc}"

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
            auth=(
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "password"),
            ),
        )
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            payload["neo4j"]["nodes"] = int(
                session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            )
            payload["neo4j"]["relationships"] = int(
                session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            )
        driver.close()
    except Exception as exc:
        payload["neo4j"]["error"] = f"{type(exc).__name__}: {exc}"

    return payload


def run_demo() -> None:
    config = LLMConfig.from_env()
    llm_client = SyncLLMClient(config)

    _print_header("GMem Multi-Patient Healthcare Retrieval Demo")
    print(f"LLM provider: {config.provider}")
    print(f"LLM model: {config.model}")
    print("Storage backend: postgres_hybrid + Neo4j active graph")
    print(f"Patients: {len(PATIENTS)}")

    with Memory(
        adapter="healthcare",
        storage_backend="postgres_hybrid",
        llm_config=config,
        optimization_profile="balanced",
        require_scope=True,
    ) as memory:
        runtime = memory.runtime_status()
        if not runtime.get("storage", {}).get("neo4j_enabled"):
            raise RuntimeError("Neo4j is required for this demo. Run `make services-up` first.")

        service = HealthcareRetrievalService(
            memory_store=memory.system.memory_store,
            retriever=memory.retriever,
            llm_client=llm_client,
        )
        historical_as_of = datetime.now(timezone.utc)

        # ------------------------------------------------------------------
        # Retrieval Phase — Per-Patient Current State
        # ------------------------------------------------------------------
        _print_header("Current Retrieval (Per-Patient)")
        retrieval_times: list[float] = []
        for patient in PATIENTS:
            patient_scope = {**BASE_SCOPE, "user_id": f"user-{patient['mrn'].lower()}"}
            query = (
                f"For patient {patient['name']} MRN {patient['mrn']}, "
                "what is currently prescribed and what allergies or safety alerts exist?"
            )
            t0 = time.perf_counter()
            ctx = service.retrieve_current_state(query, scope=patient_scope, max_facts=20)
            elapsed = time.perf_counter() - t0
            retrieval_times.append(elapsed)
            print(f"\n{patient['name']} ({patient['mrn']}):")
            print(f"  current_medications: {len(ctx.current_medications)}")
            print(f"  allergies: {len(ctx.allergies)}")
            print(f"  safety_alerts: {len(ctx.safety_alerts)}")
            print(f"  history: {len(ctx.history)}")
            print(f"  retrieval_latency_ms: {elapsed * 1000:.2f}")
            if ctx.current_medications:
                for med in ctx.current_medications:
                    print(
                        f"    - {med['medication_name']} {med.get('dosage', '')} ({med['order_status']})"
                    )
            if ctx.allergies:
                for allergy in ctx.allergies:
                    print(f"    - ALLERGY: {allergy['allergen']} ({allergy.get('severity', '')})")
            if ctx.safety_alerts:
                for alert in ctx.safety_alerts[:3]:
                    print(f"    - ALERT: {alert['constraint_name']}: {alert['reason']}")

        # ------------------------------------------------------------------
        # Retrieval Phase — Historical as_of
        # ------------------------------------------------------------------
        _print_header("Historical Retrieval (as_of)")
        historical_query = (
            f"As of {historical_as_of.isoformat()}, what Lisinopril dose was prescribed "
            "for patient John Doe MRN JD-001?"
        )
        t0 = time.perf_counter()
        historical_ctx = service.retrieve_historical_state(
            historical_query,
            as_of=historical_as_of,
            scope={**BASE_SCOPE, "user_id": "user-jd-001"},
            max_facts=20,
        )
        hist_elapsed = time.perf_counter() - t0
        print(f"Historical query latency: {hist_elapsed * 1000:.2f} ms")
        print(f"Historical medications found: {len(historical_ctx.current_medications)}")
        for med in historical_ctx.current_medications:
            print(f"  - {med['medication_name']} {med.get('dosage', '')}")

        # ------------------------------------------------------------------
        # Cross-Patient Isolation Check
        # ------------------------------------------------------------------
        _print_header("Cross-Patient Isolation Check")
        alice_scope = {**BASE_SCOPE, "user_id": "user-aj-002"}
        cross_query = "What medications are prescribed for Alice Johnson MRN AJ-002?"
        t0 = time.perf_counter()
        is_isolated, alice_meds = service.check_cross_patient_isolation(
            cross_query,
            scope=alice_scope,
            forbidden_medication_names={"Lisinopril", "Warfarin"},
            max_facts=20,
        )
        cross_elapsed = time.perf_counter() - t0
        print(f"Alice's meds: {alice_meds}")
        print("John's meds should NOT appear: {'Lisinopril', 'Warfarin'}")
        print(f"Isolation check: {is_isolated}")
        print(f"Cross-patient query latency: {cross_elapsed * 1000:.2f} ms")

        # ------------------------------------------------------------------
        # Shared Entity Verification
        # ------------------------------------------------------------------
        _print_header("Shared Entity Verification")

        penicillin_patients = service.find_patients_by_shared_entity(
            "Penicillin", RelationType.HAS_ALLERGY
        )
        print(f"Patients with Penicillin allergy: {penicillin_patients}")
        print(f"  Shared allergy entity count: {len(penicillin_patients)}")

        metformin_patients = service.find_patients_by_shared_entity(
            "Metformin", RelationType.PRESCRIBED
        )
        print(f"Patients prescribed Metformin: {metformin_patients}")
        print(f"  Shared medication entity count: {len(metformin_patients)}")

        amlodipine_patients = service.find_patients_by_shared_entity(
            "Amlodipine", RelationType.PRESCRIBED
        )
        print(f"Patients prescribed Amlodipine: {amlodipine_patients}")
        print(f"  Shared medication entity count: {len(amlodipine_patients)}")

        simvastatin_patients = service.find_patients_by_shared_entity(
            "Simvastatin", RelationType.PRESCRIBED
        )
        print(f"Patients prescribed Simvastatin: {simvastatin_patients}")
        print(f"  Shared medication entity count: {len(simvastatin_patients)}")

        shared_entity_ok = (
            len(penicillin_patients) >= 2
            and len(metformin_patients) >= 2
            and len(amlodipine_patients) >= 2
        )
        print(f"\nShared entity link check: {shared_entity_ok}")

        # ------------------------------------------------------------------
        # Performance Summary
        # ------------------------------------------------------------------
        _print_header("Performance Summary")
        _print_json(
            {
                "retrieval_phase": {
                    "per_patient_avg_latency_ms": round(
                        sum(retrieval_times) / len(retrieval_times) * 1000, 2
                    )
                    if retrieval_times
                    else 0,
                    "per_patient_min_latency_ms": round(min(retrieval_times) * 1000, 2)
                    if retrieval_times
                    else 0,
                    "per_patient_max_latency_ms": round(max(retrieval_times) * 1000, 2)
                    if retrieval_times
                    else 0,
                    "historical_query_latency_ms": round(hist_elapsed * 1000, 2),
                    "cross_patient_query_latency_ms": round(cross_elapsed * 1000, 2),
                },
                "backend_counts": _direct_backend_counts(),
            }
        )

        # ------------------------------------------------------------------
        # Strict Grounded Answer (Emma Davis)
        # ------------------------------------------------------------------
        _print_header("Strict Grounded Answer (Emma Davis)")
        emma_scope = {**BASE_SCOPE, "user_id": "user-ed-006"}
        t0 = time.perf_counter()
        response = service.generate_grounded_answer(
            "What medications and safety alerts does Emma Davis MRN ED-006 have?",
            scope=emma_scope,
            llm_client=llm_client,
            max_facts=20,
        )
        answer_latency = time.perf_counter() - t0
        print(response)
        print(f"\nLLM answer generation latency: {answer_latency * 1000:.2f} ms")


if __name__ == "__main__":
    run_demo()
