#!/usr/bin/env python3
"""Live interactive demo for real-time memory extraction, validation, and persistence.

This script opens a REPL where each prompt is processed through:
1) LLM extraction
2) grounding/validation
3) persistence to configured backends

After each turn it prints:
- end-to-end latency
- extraction/grounding summary
- runtime storage statistics
- optional direct Postgres/Neo4j counts

Healthcare adapter adds retrieval commands:
- /patient <query>       Retrieve current state for a patient
- /history <query>       Retrieve historical state (as_of)
- /medication <name>     Find patients prescribed a medication
- /allergy <name>        Find patients with an allergy
- /shared <entity> <rel> Find patients linked to a shared entity
- /answer <question>     Generate a strictly-grounded LLM answer

Run:
    python demos/demo_interactive.py --adapter healthcare
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure src-layout imports work when executed from repository root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

# Load .env from the repository root so that GM_STORAGE_BACKEND and other
# variables are always available, regardless of how the script is launched.
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env", override=False)
except ImportError:
    _env_file = _REPO_ROOT / ".env"
    if _env_file.exists():
        with _env_file.open() as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip())

from gmem import Memory  # noqa: E402
from grounded_memory.llm.client import LLMConfig, SyncLLMClient  # noqa: E402


def _print_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def _safe_len(obj: Any, field: str) -> int:
    value = getattr(obj, field, None)
    return len(value) if isinstance(value, list) else 0


def _summarize_agent_result(result: Any) -> dict[str, int]:
    return {
        "approved": _safe_len(result, "approved_facts"),
        "rejected": _safe_len(result, "rejected_facts"),
        "grounded": _safe_len(result, "grounding_results"),
        "warnings": _safe_len(result, "warnings"),
        "dispositions": _safe_len(result, "dispositions"),
    }


def _print_process_result(result: dict[str, Any]) -> None:
    """Pretty-print the result from Memory.process()."""
    intent_info = result.get("intent", {})
    action = intent_info.get("action", "UNKNOWN")
    results = result.get("results", {})

    if action == "REMEMBER":
        summary = _summarize_agent_result(results)
        print(
            f"Write result: approved={summary['approved']} "
            f"rejected={summary['rejected']} grounded={summary['grounded']} "
            f"warnings={summary['warnings']}"
        )
        _print_extraction_diagnostics(results)
        _print_grounding_diagnostics(results)
    elif action in ("RECALL", "FIND_RELATED", "EXPLAIN"):
        if isinstance(results, list):
            print(f"Read result: {len(results)} fact(s) retrieved")
            for row in results[:5]:
                print(
                    "- "
                    + f"{row.get('subject_name')} [{row.get('relation')}] "
                    + f"{row.get('object_name') or row.get('value')} "
                    + f"score={row.get('score')}"
                )
        else:
            print(f"Read result: {results}")
    else:
        print(f"Unknown action result: {results}")


def _coerce_source(value: str) -> str:
    normalized = (value or "user").strip().lower()
    allowed = {"user", "assistant", "agent", "tool", "system"}
    if normalized not in allowed:
        raise ValueError(f"Unsupported source '{value}'. Allowed: {sorted(allowed)}")
    return normalized


def _format_delta(current: int | None, previous: int | None) -> str:
    if current is None:
        return "n/a"
    if previous is None:
        return str(current)
    delta = current - previous
    sign = "+" if delta >= 0 else ""
    return f"{current} ({sign}{delta})"


def _extract_int(mapping: dict[str, Any], key: str) -> int | None:
    value = mapping.get(key)
    return value if isinstance(value, int) else None


def _print_runtime_stats(current: dict[str, Any], previous: dict[str, Any] | None) -> None:
    prev_top = previous or {}

    mem_entities = _extract_int(current, "total_entities")
    mem_facts = _extract_int(current, "total_facts")
    mem_interactions = _extract_int(current, "total_interactions")

    prev_mem_entities = _extract_int(prev_top, "total_entities")
    prev_mem_facts = _extract_int(prev_top, "total_facts")
    prev_mem_interactions = _extract_int(prev_top, "total_interactions")

    print(
        "Runtime store: "
        f"entities={_format_delta(mem_entities, prev_mem_entities)} "
        f"facts={_format_delta(mem_facts, prev_mem_facts)} "
        f"interactions={_format_delta(mem_interactions, prev_mem_interactions)}"
    )

    pg = current.get("postgres") if isinstance(current.get("postgres"), dict) else None
    prev_pg = prev_top.get("postgres") if isinstance(prev_top.get("postgres"), dict) else None
    if pg:
        pg_entities = _extract_int(pg, "total_entities")
        pg_facts = _extract_int(pg, "total_facts")
        pg_interactions = _extract_int(pg, "total_interactions")

        prev_pg_entities = _extract_int(prev_pg or {}, "total_entities")
        prev_pg_facts = _extract_int(prev_pg or {}, "total_facts")
        prev_pg_interactions = _extract_int(prev_pg or {}, "total_interactions")

        print(
            "Postgres: "
            f"entities={_format_delta(pg_entities, prev_pg_entities)} "
            f"facts={_format_delta(pg_facts, prev_pg_facts)} "
            f"interactions={_format_delta(pg_interactions, prev_pg_interactions)}"
        )

    neo = current.get("neo4j") if isinstance(current.get("neo4j"), dict) else None
    prev_neo = prev_top.get("neo4j") if isinstance(prev_top.get("neo4j"), dict) else None
    if neo:
        nodes = _extract_int(neo, "node_count")
        rels = _extract_int(neo, "relationship_count")
        prev_nodes = _extract_int(prev_neo or {}, "node_count")
        prev_rels = _extract_int(prev_neo or {}, "relationship_count")
        print(
            f"Neo4j: nodes={_format_delta(nodes, prev_nodes)} rels={_format_delta(rels, prev_rels)}"
        )


def _direct_postgres_counts() -> dict[str, Any]:
    try:
        import psycopg2
    except Exception as exc:
        return {"ok": False, "error": f"psycopg2 unavailable: {exc}"}

    host = os.getenv("POSTGRES_HOST", "127.0.0.1")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    database = os.getenv("POSTGRES_DB", "grounded_memory")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            connect_timeout=2,
        )
        with conn, conn.cursor() as cur:
            counts: dict[str, int] = {}
            for table in [
                "entities",
                "candidate_facts",
                "validated_facts",
                "interactions",
                "rejection_records",
            ]:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = int(cur.fetchone()[0])
        conn.close()
        return {"ok": True, "counts": counts}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _direct_neo4j_counts() -> dict[str, Any]:
    try:
        from neo4j import GraphDatabase
    except Exception as exc:
        return {"ok": False, "error": f"neo4j driver unavailable: {exc}"}

    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session(database=database) as session:
            nodes = int(session.run("MATCH (n) RETURN count(n) AS c").single()["c"])
            rels = int(session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"])
        driver.close()
        return {"ok": True, "counts": {"nodes": nodes, "relationships": rels}}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _print_direct_db_counts() -> None:
    pg = _direct_postgres_counts()
    neo = _direct_neo4j_counts()

    if pg.get("ok"):
        counts = pg.get("counts", {})
        print(
            "Direct Postgres: "
            f"entities={counts.get('entities')} "
            f"candidate_facts={counts.get('candidate_facts')} "
            f"validated_facts={counts.get('validated_facts')} "
            f"interactions={counts.get('interactions')}"
        )
    else:
        print(f"Direct Postgres: unavailable ({pg.get('error')})")

    if neo.get("ok"):
        counts = neo.get("counts", {})
        print(
            f"Direct Neo4j: nodes={counts.get('nodes')} relationships={counts.get('relationships')}"
        )
    else:
        print(f"Direct Neo4j: unavailable ({neo.get('error')})")


def _grounding_preview(result: Any, limit: int = 5) -> list[str]:
    rows: list[str] = []
    items = getattr(result, "grounding_results", None)
    if not isinstance(items, list):
        return rows

    for item in items[:limit]:
        decision = getattr(getattr(item, "decision", None), "value", "unknown")
        candidate = getattr(item, "candidate_fact", None)
        relation = "?"
        subject = "?"
        object_value = "?"

        if candidate is not None:
            relation_obj = getattr(candidate, "relation", None)
            relation = getattr(relation_obj, "value", str(relation_obj))
            subject = str(getattr(candidate, "subject_entity_id", "?"))

            object_id = getattr(candidate, "object_entity_id", None)
            value = getattr(candidate, "value", None)
            object_value = str(object_id) if object_id is not None else str(value)

        rows.append(f"- {decision}: {subject} [{relation}] {object_value}")

    return rows


def _print_extraction_diagnostics(result: Any) -> None:
    extracted = getattr(result, "extracted_items", None)
    if extracted is None:
        return

    entities = getattr(extracted, "entities", None) or []
    candidate_facts = getattr(extracted, "candidate_facts", None) or []

    if entities:
        print("Extracted entities:")
        for entity in entities:
            name = getattr(entity, "name", "?")
            entity_type = getattr(
                getattr(entity, "entity_type", None), "value", getattr(entity, "entity_type", "?")
            )
            canonical_id = getattr(entity, "canonical_id", None)
            suffix = f" canonical_id={canonical_id}" if canonical_id else ""
            print(f"  - {name} ({entity_type}){suffix}")

    if candidate_facts:
        print("Candidate facts (pre-grounding):")
        for candidate in candidate_facts:
            relation = getattr(
                getattr(candidate, "relation", None), "value", getattr(candidate, "relation", "?")
            )
            subject = str(getattr(candidate, "subject_entity_id", "?"))[:8]
            object_id = getattr(candidate, "object_entity_id", None)
            value = getattr(candidate, "value", None)
            obj_text = f"obj={str(object_id)[:8]}" if object_id is not None else f"val={value}"
            print(f"  - [{subject}] [{relation}] ({obj_text})")


def _print_grounding_diagnostics(result: Any, limit: int = 10) -> None:
    items = getattr(result, "grounding_results", None)
    if not isinstance(items, list) or not items:
        return

    print("Grounding decisions:")
    for item in items[:limit]:
        decision = getattr(getattr(item, "decision", None), "value", "unknown")
        candidate = getattr(item, "candidate_fact", None)
        if candidate is None:
            continue

        relation = getattr(
            getattr(candidate, "relation", None), "value", getattr(candidate, "relation", "?")
        )
        subject = str(getattr(candidate, "subject_entity_id", "?"))[:8]
        object_id = getattr(candidate, "object_entity_id", None)
        value = getattr(candidate, "value", None)
        obj_text = f"obj={str(object_id)[:8]}" if object_id is not None else f"val={value}"

        line = f"  [{decision.upper():10}] [{subject}] [{relation:16}] ({obj_text})"

        if decision == "approved":
            line += " ✓ VALIDATED"
        elif decision == "superseded":
            superseded = getattr(item, "superseded_facts", None) or []
            line += f" → SUPERSEDED {len(superseded)} fact(s)"
        elif decision == "rejected":
            rejection = getattr(item, "rejection_record", None)
            constraint_id = getattr(rejection, "constraint_id", None) if rejection else None
            reason = getattr(rejection, "reason", None) if rejection else None
            if constraint_id:
                line += f" ✗ REJECTED ({constraint_id})"
            elif reason:
                line += f" ✗ REJECTED ({reason})"
            else:
                line += " ✗ REJECTED"
        elif decision == "duplicate":
            line += " ⚠ DUPLICATE"

        print(line)

        if decision == "superseded":
            validated = getattr(item, "validated_fact", None)
            if validated is not None:
                old_fact = getattr(item, "superseded_facts", None) or []
                print("  Supersession chain:")
                for fact in old_fact[:limit]:
                    fact_id = getattr(fact, "fact_id", getattr(fact, "id", "?"))
                    valid_to = getattr(fact, "valid_to", None)
                    print(f"    - old_fact_id={fact_id} valid_to={valid_to}")
                new_fact_id = getattr(validated, "fact_id", getattr(validated, "id", "?"))
                print(f"    - new_fact_id={new_fact_id} active=true")


def _print_healthcare_ctx(ctx: Any) -> None:
    """Pretty-print a HealthcareClinicalContext."""
    print(f"  Query: {ctx.query}")
    print(f"  Seed entities: {ctx.seed_entities}")
    print(f"  Current medications ({len(ctx.current_medications)}):")
    for med in ctx.current_medications:
        print(
            f"    - {med['medication_name']} {med.get('dosage', '')} "
            f"({med.get('order_status', 'active')})"
        )
    print(f"  Allergies ({len(ctx.allergies)}):")
    for allergy in ctx.allergies:
        print(f"    - {allergy['allergen']} ({allergy.get('severity', 'unknown')})")
    print(f"  Safety alerts ({len(ctx.safety_alerts)}):")
    for alert in ctx.safety_alerts[:5]:
        print(f"    - {alert['constraint_name']}: {alert['reason']}")
    print(f"  History ({len(ctx.history)}):")
    for hist in ctx.history[:5]:
        print(f"    - {hist['medication_name']} {hist.get('dosage', '')}")


def _print_help(is_healthcare: bool) -> None:
    print("Commands:")
    print("  /help                 Show this help")
    print("  /status               Show runtime status summary")
    print("  /stats                Show store statistics")
    print("  /db                   Run direct Postgres/Neo4j counts")
    print("  /search <query>       Run a scoped memory search")
    print("  /source <actor>       Set source actor (user/assistant/agent/tool/system)")
    print("  /scope                Show active scope")
    print("  /quit                 Exit")
    if is_healthcare:
        print("Healthcare retrieval commands:")
        print("  /patient <query>      Retrieve current state for a patient")
        print("  /history <query>      Retrieve historical state (as_of)")
        print("  /medication <name>    Find patients prescribed a medication")
        print("  /allergy <name>       Find patients with an allergy")
        print("  /shared <entity> <rel>  Find patients linked to a shared entity")
        print("  /answer <question>    Generate a strictly-grounded LLM answer")
        print("You may also type natural-language queries directly; the demo will")
        print("auto-route them (remember / recall / find_related / explain).")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live interactive Grounded Memory demo")
    parser.add_argument("--adapter", default=os.getenv("GM_ADAPTER", "generic"))
    parser.add_argument(
        "--storage-backend", default=os.getenv("GM_STORAGE_BACKEND", "postgres_hybrid")
    )
    parser.add_argument("--optimization-profile", default="balanced")
    parser.add_argument("--source", default="user")
    parser.add_argument(
        "--no-direct-db-check", action="store_true", help="Skip direct DB counts each turn"
    )
    parser.add_argument("--tenant-id", default=os.getenv("GM_SCOPE_TENANT_ID", "demo-tenant"))
    parser.add_argument("--app-id", default=os.getenv("GM_SCOPE_APP_ID", "ground-memory-core"))
    parser.add_argument("--user-id", default=os.getenv("GM_SCOPE_USER_ID", "healthcare-demo-user"))
    parser.add_argument(
        "--agent-id", default=os.getenv("GM_SCOPE_AGENT_ID", "healthcare-demo-agent")
    )
    parser.add_argument(
        "--run-id",
        default=os.getenv("GM_HEALTHCARE_DEMO_RUN_ID")
        or os.getenv("GM_SCOPE_RUN_ID", "healthcare-demo"),
    )
    parser.add_argument("--space-type", default=os.getenv("GM_SCOPE_SPACE_TYPE", "user"))
    parser.add_argument("--max-grounding-preview", type=int, default=5)
    return parser.parse_args()


@dataclass
class SessionMetrics:
    turns: int = 0
    total_latency_ms: float = 0.0

    def add(self, latency_ms: float) -> None:
        self.turns += 1
        self.total_latency_ms += latency_ms

    @property
    def avg_latency_ms(self) -> float:
        if self.turns == 0:
            return 0.0
        return self.total_latency_ms / self.turns


def main() -> int:
    args = _parse_args()
    source = _coerce_source(args.source)
    direct_db_check = not args.no_direct_db_check
    is_healthcare = args.adapter.strip().lower() == "healthcare"

    llm_config = LLMConfig.from_env()

    scope = {
        "tenant_id": args.tenant_id,
        "app_id": args.app_id,
        "user_id": args.user_id,
        "agent_id": args.agent_id,
        "run_id": args.run_id,
        "space_type": args.space_type,
    }

    _print_header("Live Interactive Grounded Memory Demo")
    print(f"Provider: {llm_config.provider}")
    print(f"Model: {llm_config.model}")
    print(f"Storage backend: {args.storage_backend}")
    print(f"Adapter: {args.adapter}")
    print(
        f"Scope: tenant={scope['tenant_id']} app={scope['app_id']} "
        f"user={scope['user_id']} agent={scope['agent_id']} "
        f"run={scope['run_id']} space={scope['space_type']}"
    )
    print("Type /help for commands.")

    metrics = SessionMetrics()
    previous_stats: dict[str, Any] | None = None

    with Memory(
        adapter=args.adapter,
        domain_profile=args.adapter,
        storage_backend=args.storage_backend,
        llm_config=llm_config,
        optimization_profile=args.optimization_profile,
        require_scope=True,
    ) as memory:
        status = memory.runtime_status()
        print("Startup status:")
        print(
            json.dumps(
                {
                    "storage": status.get("storage", {}),
                    "statistics": status.get("statistics", {}),
                    "scope": status.get("scope", {}),
                },
                indent=2,
            )
        )

        # Initialise healthcare retrieval service when using the healthcare adapter
        hc_service: Any | None = None
        if is_healthcare:
            from grounded_memory.adapters.healthcare.retrieval import HealthcareRetrievalService
            from grounded_memory.core.models import RelationType

            hc_service = HealthcareRetrievalService(
                memory_store=memory.system.memory_store,
                retriever=memory.retriever,
                llm_client=SyncLLMClient(llm_config),
            )

        while True:
            try:
                prompt = input(f"\n[{source}]> ").strip()
            except EOFError:
                print("\nEnd of input. Exiting.")
                break
            except KeyboardInterrupt:
                print("\nInterrupted. Type /quit to exit.")
                continue

            if not prompt:
                continue

            if prompt.startswith("/"):
                command, _, payload = prompt.partition(" ")
                command = command.lower()
                payload = payload.strip()

                if command in {"/quit", "/exit"}:
                    break
                if command == "/help":
                    _print_help(is_healthcare)
                    continue
                if command == "/scope":
                    print(json.dumps(scope, indent=2))
                    continue
                if command == "/source":
                    if not payload:
                        print("Usage: /source <user|assistant|agent|tool|system>")
                        continue
                    try:
                        source = _coerce_source(payload)
                    except ValueError as exc:
                        print(exc)
                        continue
                    print(f"Source set to: {source}")
                    continue
                if command == "/status":
                    runtime = memory.runtime_status()
                    print(
                        json.dumps(
                            {
                                "storage": runtime.get("storage", {}),
                                "llm": runtime.get("llm", {}),
                                "scope": runtime.get("scope", {}),
                            },
                            indent=2,
                        )
                    )
                    continue
                if command == "/stats":
                    runtime = memory.runtime_status()
                    stats = runtime.get("statistics", {})
                    _print_runtime_stats(stats, previous_stats)
                    print(
                        f"Session metrics: turns={metrics.turns} "
                        f"avg_latency_ms={metrics.avg_latency_ms:.1f}"
                    )
                    continue
                if command == "/db":
                    _print_direct_db_counts()
                    continue
                if command == "/search":
                    if not payload:
                        print("Usage: /search <query>")
                        continue
                    results = memory.search(payload, limit=5, **scope)
                    print(f"Search hits: {len(results)}")
                    for row in results[:5]:
                        print(
                            "- "
                            f"{row.get('subject_name')} [{row.get('relation')}] "
                            f"{row.get('object_name') or row.get('value')} "
                            f"score={row.get('score')}"
                        )
                    continue

                # ------------------------------------------------------------------
                # Healthcare-specific retrieval commands
                # ------------------------------------------------------------------
                if is_healthcare and hc_service is not None:
                    if command == "/patient":
                        if not payload:
                            print("Usage: /patient <query>")
                            print('  e.g. /patient "What is John Doe MRN JD-001 taking?"')
                            continue
                        ctx = hc_service.retrieve_current_state(payload, scope=scope)
                        _print_healthcare_ctx(ctx)
                        continue

                    if command == "/history":
                        if not payload:
                            print("Usage: /history <query>")
                            print('  e.g. /history "What was John Doe taking as of 2024-01-01?"')
                            continue
                        # Default to now for as_of; user can include a date in the query
                        as_of = datetime.now(timezone.utc)
                        ctx = hc_service.retrieve_historical_state(
                            payload, as_of=as_of, scope=scope
                        )
                        _print_healthcare_ctx(ctx)
                        continue

                    if command == "/medication":
                        if not payload:
                            print("Usage: /medication <name>")
                            print("  e.g. / medication Metformin")
                            continue
                        patients = hc_service.find_patients_by_medication(payload)
                        print(f"Patients prescribed {payload}: {patients}")
                        continue

                    if command == "/allergy":
                        if not payload:
                            print("Usage: /allergy <name>")
                            print("  e.g. /allergy Penicillin")
                            continue
                        patients = hc_service.find_patients_by_allergy(payload)
                        print(f"Patients allergic to {payload}: {patients}")
                        continue

                    if command == "/shared":
                        parts = payload.split(None, 1)
                        if len(parts) < 2:
                            print("Usage: /shared <entity_name> <relation>")
                            print("  e.g. /shared Penicillin HAS_ALLERGY")
                            print("  e.g. /shared Metformin PRESCRIBED")
                            continue
                        entity_name, rel_str = parts
                        try:
                            relation = RelationType(rel_str.strip().upper())
                        except ValueError:
                            print(f"Unknown relation: {rel_str}")
                            print(f"Allowed: {', '.join(r.value for r in RelationType)}")
                            continue
                        patients = hc_service.find_patients_by_shared_entity(entity_name, relation)
                        print(f"Patients linked to {entity_name} via {relation.value}: {patients}")
                        continue

                    if command == "/answer":
                        if not payload:
                            print("Usage: /answer <question>")
                            print('  e.g. /answer "What medications does John Doe have?"')
                            continue
                        answer = hc_service.generate_grounded_answer(
                            payload,
                            scope=scope,
                            llm_client=SyncLLMClient(llm_config),
                        )
                        print(answer)
                        continue

                print("Unknown command. Type /help")
                continue

            # ------------------------------------------------------------------
            # Auto-routing for natural-language input
            # ------------------------------------------------------------------
            intent = memory.route(prompt)
            print(
                f"[intent]> {intent.action.value.upper()} "
                f"(confidence: {intent.confidence:.2f}) — {intent.explanation}"
            )

            if is_healthcare and hc_service is not None:
                if intent.is_write():
                    start = time.perf_counter()
                    try:
                        result = memory.add(prompt, source=source, **scope)
                    except Exception as exc:
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        print(f"Turn failed in {elapsed_ms:.1f} ms: {type(exc).__name__}: {exc}")
                        continue

                    elapsed_ms = (time.perf_counter() - start) * 1000
                    metrics.add(elapsed_ms)

                    summary = _summarize_agent_result(result)
                    print(
                        f"Latency: {elapsed_ms:.1f} ms | "
                        f"approved={summary['approved']} rejected={summary['rejected']} "
                        f"grounded={summary['grounded']} warnings={summary['warnings']}"
                    )

                    _print_extraction_diagnostics(result)
                    _print_grounding_diagnostics(result, limit=args.max_grounding_preview)

                    preview = _grounding_preview(result, limit=args.max_grounding_preview)
                    if preview:
                        print("Grounding preview:")
                        for line in preview:
                            print(line)

                    runtime = memory.runtime_status()
                    current_stats = runtime.get("statistics", {})
                    _print_runtime_stats(current_stats, previous_stats)
                    previous_stats = current_stats

                    if direct_db_check:
                        _print_direct_db_counts()

                    print(
                        f"Session metrics: turns={metrics.turns} "
                        f"avg_latency_ms={metrics.avg_latency_ms:.1f}"
                    )
                    continue

                if intent.action.value == "recall":
                    ctx = hc_service.retrieve_current_state(prompt, scope=scope)
                    _print_healthcare_ctx(ctx)
                    continue

                if intent.action.value == "find_related":
                    # Naïve entity extraction: prefer quoted strings, then last capitalised word.
                    import re as _re

                    quoted = _re.findall(r'"([^"]+)"', prompt)
                    entity_name = quoted[-1] if quoted else None
                    if entity_name is None:
                        caps = _re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", prompt)
                        if len(caps) > 1:
                            # Skip the first capitalised word (often the interrogative "Who")
                            entity_name = caps[-1]
                        elif caps:
                            entity_name = caps[0]
                        else:
                            entity_name = prompt.split()[-1]
                    if "allerg" in prompt.lower():
                        patients = hc_service.find_patients_by_allergy(entity_name)
                        print(f"Patients allergic to {entity_name}: {patients}")
                    else:
                        patients = hc_service.find_patients_by_medication(entity_name)
                        print(f"Patients prescribed {entity_name}: {patients}")
                    continue

                if intent.action.value == "explain":
                    answer = hc_service.generate_grounded_answer(
                        prompt,
                        scope=scope,
                        llm_client=SyncLLMClient(llm_config),
                    )
                    print(answer)
                    continue

                # UNKNOWN
                print("Ambiguous input. Use /help for commands, or try:")
                print('  /patient "What is John Doe taking?"')
                print('  /answer "Summarise Jane Doe\'s clinical picture"')
                continue

            # Generic adapter: use process() for auto-routing.
            start = time.perf_counter()
            try:
                result = memory.process(prompt, source=source, **scope)
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                print(f"Turn failed in {elapsed_ms:.1f} ms: {type(exc).__name__}: {exc}")
                continue

            elapsed_ms = (time.perf_counter() - start) * 1000
            metrics.add(elapsed_ms)

            # result from process() is a dict with "intent" and "results" keys
            _print_process_result(result)

            runtime = memory.runtime_status()
            current_stats = runtime.get("statistics", {})
            _print_runtime_stats(current_stats, previous_stats)
            previous_stats = current_stats

            if direct_db_check:
                _print_direct_db_counts()

            print(
                f"Session metrics: turns={metrics.turns} "
                f"avg_latency_ms={metrics.avg_latency_ms:.1f}"
            )

    _print_header("Session Ended")
    print(f"Total turns: {metrics.turns}")
    print(f"Average latency: {metrics.avg_latency_ms:.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
