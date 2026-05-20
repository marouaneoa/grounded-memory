#!/usr/bin/env python3
"""Run a deterministic hybrid smoke write, then inspect stored facts and graph state.

This script avoids LLM calls by writing structured CandidateFacts directly through the
GroundingOperator, then inspects:
- primary in-process store (facts + interactions)
- Neo4j active graph projection
- PostgreSQL table counts (optional; best-effort)

Usage:
    PYTHONPATH=src /Users/faycalamrouche/Desktop/ground-memory-core/.venv/bin/python scripts/smoke_and_inspect_backends.py

Optional flags:
    --no-write            Only inspect existing data
    --tenant-id <value>
    --app-id <value>
    --user-id <value>
    --agent-id <value>
    --run-id <value>
    --space-type <value>
    --limit <n>
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grounded_memory.core.models import (
    ActorType,
    CandidateFact,
    Entity,
    EntityType,
    Interaction,
    RelationType,
    ValidatedFact,
)
from grounded_memory.system import GroundedMemorySystem


@dataclass(frozen=True)
class Scope:
    tenant_id: str
    app_id: str
    user_id: str
    agent_id: str
    run_id: str
    space_type: str

    @property
    def scope_id(self) -> str:
        return f"{self.tenant_id}:{self.app_id}:{self.user_id}"

    def as_dict(self) -> dict[str, str]:
        return {
            "tenant_id": self.tenant_id,
            "app_id": self.app_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "space_type": self.space_type,
            "scope_id": self.scope_id,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid smoke write + backend inspection")
    parser.add_argument("--no-write", action="store_true", help="Skip writing new facts")
    parser.add_argument("--tenant-id", default=os.getenv("GM_SCOPE_TENANT_ID", "demo-tenant"))
    parser.add_argument("--app-id", default=os.getenv("GM_SCOPE_APP_ID", "ground-memory-core"))
    parser.add_argument("--user-id", default=os.getenv("GM_SCOPE_USER_ID", "inspection-user"))
    parser.add_argument("--agent-id", default=os.getenv("GM_SCOPE_AGENT_ID", "inspection-agent"))
    parser.add_argument("--run-id", default=os.getenv("GM_SCOPE_RUN_ID", "inspection-run"))
    parser.add_argument("--space-type", default=os.getenv("GM_SCOPE_SPACE_TYPE", "user"))
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def _find_or_create_entity(
    system: GroundedMemorySystem, name: str, scope: Scope, kind: str
) -> Entity:
    entity, _created = system.memory_store.find_or_create_entity(
        name=name,
        entity_type=EntityType.FACILITY,
        uniqueness_key=f"{scope.scope_id}:{kind}:{name}",
        create_fn=lambda: Entity(
            entity_type=EntityType.FACILITY,
            name=name,
            attributes={**scope.as_dict(), "kind": kind},
        ),
    )
    return entity


def _fact_matches_scope(fact: ValidatedFact, store: Any, scope: Scope) -> bool:
    interaction = store.get_interaction(fact.source_interaction_id)
    if interaction is None:
        attrs = fact.attributes or {}
        return attrs.get("scope_id") == scope.scope_id

    metadata = interaction.metadata or {}
    interaction_scope_id = metadata.get("scope_id")
    if interaction_scope_id is None:
        tenant = getattr(interaction, "tenant_id", None) or metadata.get("tenant_id")
        app = getattr(interaction, "app_id", None) or metadata.get("app_id")
        user = getattr(interaction, "user_id", None) or metadata.get("user_id")
        if tenant and app and user:
            interaction_scope_id = f"{tenant}:{app}:{user}"
    return interaction_scope_id == scope.scope_id


def _write_smoke_facts(system: GroundedMemorySystem, scope: Scope) -> None:
    interaction = Interaction(
        actor=ActorType.USER,
        raw_text="Smoke write for backend inspection.",
        tenant_id=scope.tenant_id,
        app_id=scope.app_id,
        user_id=scope.user_id,
        agent_id=scope.agent_id,
        run_id=scope.run_id,
        session_id=scope.run_id,
        space_type=scope.space_type,
        metadata=scope.as_dict(),
    )
    system.memory_store.add_interaction(interaction)

    user_entity = _find_or_create_entity(
        system,
        name=f"user:{scope.user_id}",
        scope=scope,
        kind="user",
    )
    project_entity = _find_or_create_entity(
        system,
        name="ground-memory-core",
        scope=scope,
        kind="project",
    )
    stack_entity = _find_or_create_entity(
        system,
        name="python-stack",
        scope=scope,
        kind="stack",
    )
    ontology_entity = _find_or_create_entity(
        system,
        name="ontology:project",
        scope=scope,
        kind="ontology",
    )

    candidates = [
        CandidateFact(
            source_interaction_id=interaction.id,
            subject_entity_id=user_entity.id,
            relation=RelationType.RELATED_TO,
            object_entity_id=project_entity.id,
            confidence=0.95,
            attributes={**scope.as_dict(), "fact_kind": "ownership"},
        ),
        CandidateFact(
            source_interaction_id=interaction.id,
            subject_entity_id=project_entity.id,
            relation=RelationType.RELATED_TO,
            object_entity_id=stack_entity.id,
            confidence=0.93,
            attributes={**scope.as_dict(), "fact_kind": "tech_stack"},
        ),
        CandidateFact(
            source_interaction_id=interaction.id,
            subject_entity_id=project_entity.id,
            relation=RelationType.RELATED_TO,
            object_entity_id=ontology_entity.id,
            confidence=0.92,
            attributes={**scope.as_dict(), "fact_kind": "ontology"},
        ),
    ]

    print("\n[write] grounding candidates")
    for candidate in candidates:
        result = system.grounding_operator.ground(candidate)
        fact_id = result.validated_fact.id if result.validated_fact is not None else None
        print(f"- decision={result.decision.value} fact_id={fact_id}")


def _inspect_primary_store(system: GroundedMemorySystem, scope: Scope, limit: int) -> None:
    all_facts = system.memory_store.get_all_validated_facts()
    scoped_facts = [f for f in all_facts if _fact_matches_scope(f, system.memory_store, scope)]

    print("\n[primary-store] scoped facts")
    print(f"- total_scoped_facts={len(scoped_facts)}")

    for fact in scoped_facts[:limit]:
        print(
            "- "
            f"fact_id={fact.id} "
            f"relation={fact.relation.value} "
            f"subject={fact.subject_id} "
            f"object={fact.object_id} "
            f"active={fact.is_active}"
        )


def _inspect_neo4j(system: GroundedMemorySystem, scope: Scope, limit: int) -> None:
    print("\n[neo4j] graph projection")
    if not system.has_neo4j:
        print("- neo4j_unavailable=true")
        return

    neo4j_store = system.memory_store.neo4j  # type: ignore[attr-defined]
    stats = neo4j_store.get_statistics()
    print(
        f"- node_count={stats.get('node_count', 0)} relationship_count={stats.get('relationship_count', 0)}"
    )

    with neo4j_store._driver.session(database=neo4j_store.config.database) as session:  # type: ignore[attr-defined]
        rows = session.run(
            """
            MATCH (s:Entity)-[r]->(o:Entity)
            WHERE r.scope_id = $scope_id
            RETURN s.name AS subject, type(r) AS relation, o.name AS object, r.fact_id AS fact_id
            ORDER BY subject, relation, object
            LIMIT $limit
            """,
            scope_id=scope.scope_id,
            limit=limit,
        )
        records = list(rows)

    print(f"- scoped_relationships={len(records)}")
    for record in records:
        print(
            "- "
            f"fact_id={record['fact_id']} "
            f"{record['subject']} -[{record['relation']}]-> {record['object']}"
        )


def _inspect_postgres(limit: int) -> None:
    print("\n[postgres] table inspection (best-effort)")

    try:
        import psycopg2
    except Exception:
        print("- psycopg2_not_installed=true")
        return

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    database = os.getenv("POSTGRES_DB", "gmem")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            connect_timeout=3,
        )
    except Exception as exc:
        print(f"- connect_error={type(exc).__name__}: {exc}")
        return

    table_names = [
        "entities",
        "validated_facts",
        "interactions",
        "rejection_records",
    ]

    with conn, conn.cursor() as cur:
        cur.execute(
            """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = ANY(%s)
                ORDER BY table_name
                """,
            (table_names,),
        )
        existing = [row[0] for row in cur.fetchall()]

        if not existing:
            print("- schema_not_initialized=true")
            return

        for table in existing:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            print(f"- {table}_rows={count}")

        if "validated_facts" in existing:
            cur.execute(
                """
                    SELECT id, relation, subject_id, object_id, valid_from, valid_to
                    FROM validated_facts
                    ORDER BY created_at DESC NULLS LAST
                    LIMIT %s
                    """,
                (limit,),
            )
            rows = cur.fetchall()
            print(f"- validated_facts_sample={len(rows)}")
            for row in rows:
                print(f"  - {row}")

    conn.close()


def main() -> int:
    args = _parse_args()
    scope = Scope(
        tenant_id=args.tenant_id,
        app_id=args.app_id,
        user_id=args.user_id,
        agent_id=args.agent_id,
        run_id=args.run_id,
        space_type=args.space_type,
    )

    print("[scope]")
    print(f"- scope_id={scope.scope_id}")
    print(f"- tenant_id={scope.tenant_id} app_id={scope.app_id} user_id={scope.user_id}")
    print(f"- agent_id={scope.agent_id} run_id={scope.run_id} space_type={scope.space_type}")

    system = GroundedMemorySystem(
        storage_backend="postgres_hybrid",
        adapter="generic",
    )

    try:
        if not args.no_write:
            _write_smoke_facts(system, scope)

        _inspect_primary_store(system, scope, args.limit)
        _inspect_neo4j(system, scope, args.limit)
        _inspect_postgres(args.limit)
    finally:
        system.close()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
