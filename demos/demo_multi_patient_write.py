#!/usr/bin/env python3
"""Multi-patient healthcare medication reconciliation scale demo for GMem.

This demo exercises the full stack at scale with multiple patients:
- healthcare adapter extraction via HealthcareMemoryAgent
- write-time healthcare constraints (allergy, interaction, duplication)
- postgres_hybrid storage with Neo4j active projection
- performance timing for each phase

Run:
    make services-up
    PYTHONPATH=src python demos/demo_multi_patient_write.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "demos"))
load_dotenv(REPO_ROOT / ".env", override=False)

from demo_healthcare_data import BASE_SCOPE, PATIENTS  # noqa: E402
from gmem import Memory  # noqa: E402
from grounded_memory.llm.client import LLMConfig  # noqa: E402


def _print_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _summarize_grounding(result: Any) -> dict[str, int]:
    groundings = getattr(result, "grounding_results", []) or []
    return {
        "approved": sum(1 for item in groundings if getattr(item, "is_success", False)),
        "rejected": sum(
            1
            for item in groundings
            if getattr(getattr(item, "decision", None), "value", None) == "rejected"
        ),
        "superseded": sum(
            1
            for item in groundings
            if getattr(getattr(item, "decision", None), "value", None) == "superseded"
        ),
        "grounded": len(groundings),
    }


def _print_grounding_details(result: Any) -> None:
    for item in getattr(result, "grounding_results", []) or []:
        candidate = getattr(item, "candidate_fact", None)
        if candidate is None:
            continue
        decision = getattr(getattr(item, "decision", None), "value", "unknown")
        relation = getattr(getattr(candidate, "relation", None), "value", "?")
        med_name = candidate.attributes.get("medication_name") or candidate.attributes.get(
            "allergen_name"
        )
        obj = med_name or candidate.object_entity_id or candidate.value
        line = f"  - {decision:10s} {relation:18s} {obj}"
        rejection = getattr(item, "rejection_record", None)
        if rejection is not None:
            line += f" | {rejection.constraint_id}: {rejection.reason}"
        superseded = getattr(item, "superseded_facts", None) or []
        if superseded:
            line += f" | superseded={len(superseded)}"
        print(line)


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

    _print_header("GMem Multi-Patient Healthcare Scale Demo")
    print(f"LLM provider: {config.provider}")
    print(f"LLM model: {config.model}")
    print("Storage backend: postgres_hybrid + Neo4j active graph")
    print(f"Patients: {len(PATIENTS)}")
    print(f"Total interactions: {sum(len(p['interactions']) for p in PATIENTS)}")

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

        # ------------------------------------------------------------------
        # Write Phase
        # ------------------------------------------------------------------
        _print_header("Write-Time Grounding (Multi-Patient)")
        write_times: list[float] = []
        total_approved = 0
        total_rejected = 0
        total_superseded = 0

        for patient in PATIENTS:
            patient_scope = {**BASE_SCOPE, "user_id": f"user-{patient['mrn'].lower()}"}
            print(f"\n--- Patient {patient['name']} ({patient['mrn']}) ---")
            for index, text in enumerate(patient["interactions"], start=1):
                print(f"\n  Turn {index}/{len(patient['interactions'])}: {text}")
                t0 = time.perf_counter()
                result = memory.add(text, source="user", **patient_scope)
                elapsed = time.perf_counter() - t0
                write_times.append(elapsed)

                summary = _summarize_grounding(result)
                total_approved += summary["approved"]
                total_rejected += summary["rejected"]
                total_superseded += summary["superseded"]
                print(
                    f"  Result: approved={summary['approved']}, rejected={summary['rejected']}, superseded={summary['superseded']} | latency={elapsed:.3f}s"
                )
                _print_grounding_details(result)

        # ------------------------------------------------------------------
        # Performance Summary
        # ------------------------------------------------------------------
        _print_header("Performance Summary")
        _print_json(
            {
                "write_phase": {
                    "total_interactions": len(write_times),
                    "total_approved": total_approved,
                    "total_rejected": total_rejected,
                    "total_superseded": total_superseded,
                    "avg_latency_ms": round(sum(write_times) / len(write_times) * 1000, 2)
                    if write_times
                    else 0,
                    "min_latency_ms": round(min(write_times) * 1000, 2) if write_times else 0,
                    "max_latency_ms": round(max(write_times) * 1000, 2) if write_times else 0,
                    "p50_latency_ms": round(sorted(write_times)[len(write_times) // 2] * 1000, 2)
                    if write_times
                    else 0,
                },
                "backend_counts": _direct_backend_counts(),
            }
        )

        print(
            "\nDemo write phase complete! Now run `python demos/demo_multi_patient_retrieval.py` to test retrieval."
        )


if __name__ == "__main__":
    run_demo()
