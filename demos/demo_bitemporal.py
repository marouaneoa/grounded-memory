#!/usr/bin/env python3
"""Healthcare medication reconciliation demo for GMem.

This demo uses the implementation path intended for thesis day:
- healthcare adapter extraction through HealthcareMemoryAgent
- write-time healthcare constraints
- postgres_hybrid storage with Neo4j active projection
- healthcare retrieval planning with no hardcoded seed entity IDs

Run:
    make services-up
    PYTHONPATH=src python demos/demo_bitemporal.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
load_dotenv(REPO_ROOT / ".env", override=False)

from gmem import Memory  # noqa: E402
from grounded_memory.adapters.healthcare.retrieval import (  # noqa: E402
    HealthcareRetrievalService,
)
from grounded_memory.llm.client import LLMConfig, SyncLLMClient  # noqa: E402


def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _wait_for_enter(message: str) -> None:
    if not sys.stdin.isatty():
        return
    try:
        input(message)
    except EOFError:
        return


def _summarize_grounding(result: Any) -> dict[str, int]:
    groundings = getattr(result, "grounding_results", []) or []
    return {
        "approved": sum(
            1
            for item in groundings
            if getattr(getattr(item, "decision", None), "value", None) == "approved"
        ),
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


def _phase_1_write_time_grounding(
    memory: Memory,
    scope: dict[str, Any],
) -> datetime:
    """Phase 1: Ingest clinical observations with write-time governance."""
    _print_header("Phase 1: Write-Time Grounding")
    interactions = [
        "Patient John Doe, MRN JD-001, has a severe Penicillin allergy with anaphylaxis.",
        "Prescribe Lisinopril 10mg daily for patient John Doe, MRN JD-001.",
        "Adjust Lisinopril to 20mg daily for patient John Doe, MRN JD-001.",
        "Continue Warfarin 5mg daily for patient John Doe, MRN JD-001.",
        "Prescribe Amiodarone 200mg daily for patient John Doe, MRN JD-001.",
        "Prescribe Penicillin 500mg daily for patient John Doe, MRN JD-001.",
        "Discontinue Lisinopril for patient John Doe, MRN JD-001.",
    ]

    historical_as_of: datetime | None = None
    total_approved = total_rejected = total_superseded = 0

    for index, text in enumerate(interactions, start=1):
        print(f"\n┌─ Turn {index}/{len(interactions)} ─{'─' * 60}")
        print(f"│ Prompt: {text}")
        print(f"└{'─' * 72}")
        # _wait_for_enter("Press Enter to run this prompt...")

        result = memory.add(text, source="user", **scope)

        if "Prescribe Lisinopril 10mg" in text:
            historical_as_of = datetime.now(timezone.utc)

        summary = _summarize_grounding(result)
        total_approved += summary["approved"]
        total_rejected += summary["rejected"]
        total_superseded += summary["superseded"]

        _print_grounding_details(result)
        print(
            f"\n  Summary: approved={summary['approved']} "
            f"rejected={summary['rejected']} "
            f"superseded={summary['superseded']}"
        )

    print(f"\n{'=' * 72}")
    print(f"Phase 1 Complete: {len(interactions)} turns")
    print(f"  Total approved:   {total_approved}")
    print(f"  Total rejected:   {total_rejected}")
    print(f"  Total superseded: {total_superseded}")
    print(f"{'=' * 72}")

    if historical_as_of is None:
        historical_as_of = datetime.now(timezone.utc)
    return historical_as_of


def _phase_2_current_retrieval(
    service: HealthcareRetrievalService,
    scope: dict[str, Any],
) -> None:
    """Phase 2: Retrieve current state without hardcoded seed entities."""
    _print_header("Phase 2: Current Retrieval (No Hardcoded Seeds)")
    current_query = (
        "For patient John Doe MRN JD-001, what is currently prescribed "
        "and what allergies or safety alerts exist?"
    )
    print(f"Query: {current_query}")

    current_context = service.retrieve_current_state(
        current_query,
        scope=scope,
        max_facts=20,
    )
    _print_json(current_context.to_dict())


def _phase_3_historical_retrieval(
    service: HealthcareRetrievalService,
    scope: dict[str, Any],
    historical_as_of: datetime,
) -> None:
    """Phase 3: Point-in-time historical retrieval."""
    _print_header("Phase 3: Historical Retrieval (Point-in-Time)")
    historical_query = (
        f"As of {historical_as_of.isoformat()}, "
        "what Lisinopril dose was prescribed for patient John Doe MRN JD-001?"
    )
    print(f"Query: {historical_query}")

    historical_context = service.retrieve_historical_state(
        historical_query,
        as_of=historical_as_of,
        scope=scope,
        max_facts=20,
    )
    _print_json(historical_context.to_dict())


def _phase_4_grounded_answer(
    service: HealthcareRetrievalService,
    scope: dict[str, Any],
    llm_client: SyncLLMClient,
) -> None:
    """Phase 4: Generate strictly-grounded LLM answer."""
    _print_header("Phase 4: Strictly-Grounded LLM Answer")
    current_query = (
        "For patient John Doe MRN JD-001, what is currently prescribed "
        "and what allergies or safety alerts exist?"
    )
    print(f"Query: {current_query}")

    response = service.generate_grounded_answer(
        current_query,
        scope=scope,
        llm_client=llm_client,
        max_facts=20,
    )
    print(response)


def _phase_5_bitemporal_summary(
    historical_as_of: datetime,
) -> None:
    """Phase 5: Display bitemporal timeline summary."""
    _print_header("Phase 5: Bitemporal Timeline Summary")
    print(f"Historical snapshot (as_of): {historical_as_of.isoformat()}")
    print("")
    print("  Timeline:")
    print("    T0  ── Patient identity + Penicillin allergy recorded")
    print("    T1  ── Lisinopril 10mg prescribed  (ACTIVE at snapshot)")
    print("    T2  ── Lisinopril 10mg → 20mg  (superseded)")
    print("    T3  ── Warfarin 5mg prescribed  (ACTIVE)")
    print("    T4  ── Amiodarone 200mg REJECTED  (major interaction with Warfarin)")
    print("    T5  ── Penicillin 500mg REJECTED  (allergy conflict)")
    print("    T6  ── Lisinopril DISCONTINUED  (closed, no longer active)")
    print("")
    print("  At snapshot T1-T2: Lisinopril 10mg was the active dose.")
    print("  At current time T6+: Only Warfarin 5mg remains active.")


def run_demo() -> None:
    config = LLMConfig.from_env()
    llm_client = SyncLLMClient(config)

    run_id = (
        os.getenv("GM_HEALTHCARE_DEMO_RUN_ID")
        or os.getenv("GM_SCOPE_RUN_ID")
        or f"healthcare-demo-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    )
    scope = {
        "tenant_id": os.getenv("GM_SCOPE_TENANT_ID", "demo-tenant"),
        "app_id": os.getenv("GM_SCOPE_APP_ID", "ground-memory-core"),
        "user_id": os.getenv("GM_SCOPE_USER_ID", "healthcare-demo-user"),
        "agent_id": os.getenv("GM_SCOPE_AGENT_ID", "healthcare-demo-agent"),
        "run_id": run_id,
        "space_type": os.getenv("GM_SCOPE_SPACE_TYPE", "user"),
    }

    _print_header("GMem Healthcare Medication Reconciliation Demo")
    print(f"  LLM provider:  {config.provider}")
    print(f"  LLM model:     {config.model}")
    print("  Storage:       postgres_hybrid + Neo4j active graph")
    print(f"  Scope run_id:  {run_id}")
    print(f"{'=' * 72}")

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

        historical_as_of = _phase_1_write_time_grounding(memory, scope)
        _phase_2_current_retrieval(service, scope)
        _phase_3_historical_retrieval(service, scope, historical_as_of)
        _phase_4_grounded_answer(service, scope, llm_client)
        _phase_5_bitemporal_summary(historical_as_of)

        _print_header("Demo Complete")
        print("All phases executed successfully.")
        print(f"Run ID: {run_id}")

    return None


if __name__ == "__main__":
    run_demo()
