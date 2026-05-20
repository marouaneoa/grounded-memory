#!/usr/bin/env python3
"""End-to-end OpenRouter demo for the gmem runtime.

This script exercises the major framework surfaces in one run:
1) LLM-backed ingestion via Memory.add and Memory.add_many
2) Grounded structured writes via add_entity/add_fact
3) Supersession and lifecycle history via update_fact/delete_fact/history
4) Retrieval via search/build_context/build_memory_prompt
5) Optional FastAPI integration via create_app and /v1/memories/search

Neo4j is required for this demo. The run fails fast if graph sync is unavailable.

Run:
    /Users/faycalamrouche/Desktop/ground-memory-core/.venv/bin/python demos/demo_openrouter.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure src-layout imports work when executed from repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gmem import Memory, create_app
from grounded_memory.core.models import EntityType, RelationType
from grounded_memory.llm.client import LLMConfig, LLMProvider


def _print_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def _print_json(payload: Any) -> None:
    """Print payload as JSON with a safe fallback for uncommon objects."""
    print(json.dumps(payload, indent=2, default=str))


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


def _require_openrouter_config() -> LLMConfig:
    config = LLMConfig.from_env()
    if config.provider != LLMProvider.OPENROUTER:
        raise RuntimeError(
            "This demo is OpenRouter-only. Set LLM_PROVIDER=openrouter and OPENROUTER_API_KEY."
        )
    return config


def _run_optional_service_check(memory: Memory, scope: dict[str, str]) -> None:
    _print_header("Step 8: Optional Service Layer Check")
    try:
        from fastapi.testclient import TestClient
    except Exception as exc:
        print("Skipping service check: fastapi[testclient] not available")
        print(f"Reason: {type(exc).__name__}: {exc}")
        return

    app = create_app(memory=memory)
    client = TestClient(app)

    payload = {
        "query": "What is my project codename and stack preference?",
        "tenant_id": scope["tenant_id"],
        "app_id": scope["app_id"],
        "user_id": scope["user_id"],
        "agent_id": scope["agent_id"],
        "run_id": scope["run_id"],
        "space_type": scope["space_type"],
        "limit": 5,
    }
    response = client.post("/v1/memories/search", json=payload)
    print(f"HTTP {response.status_code}")

    body = response.json()
    if response.status_code != 200:
        print(json.dumps(body, indent=2))
        raise RuntimeError("Service call failed")

    hits = body.get("data", {}).get("results", [])
    print(f"Service results: {len(hits)}")
    if hits:
        print("Top service result:")
        print(json.dumps(hits[0], indent=2))


def run_demo() -> None:
    config = _require_openrouter_config()

    _print_header("Step 0: Runtime Configuration")
    print(f"Provider: {config.provider}")
    print(f"Model: {config.model}")
    print(f"Base URL: {config.base_url}")

    user_id = "openrouter-demo-user"
    scope = {
        "tenant_id": "demo-tenant",
        "app_id": "ground-memory-core",
        "user_id": user_id,
        "agent_id": "openrouter-demo-agent",
        "run_id": "openrouter-demo-run",
        "space_type": "user",
    }

    with Memory(
        adapter="generic",
        domain_profile="generic",
        storage_backend="postgres_hybrid",
        llm_config=config,
        optimization_profile="balanced",
        require_scope=True,
    ) as memory:
        _print_header("Step 1: Health and Runtime Status")
        _print_json(memory.healthcheck())
        runtime_status = memory.runtime_status()
        _print_json(runtime_status)

        if not runtime_status.get("storage", {}).get("neo4j_enabled", False):
            raise RuntimeError(
                "Neo4j must be enabled for this demo. Start services with `make services-up` "
                "and verify NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD in .env."
            )

        _print_header("Step 2: LLM-backed Ingestion (Memory.add)")
        interactions = [
            "My name is Faycal and I work on grounded-memory-core.",
            "Our internal project codename is Atlas.",
            "I prefer Python and typed APIs for backend work.",
            "I no longer use Java for this project.",
        ]

        for idx, text in enumerate(interactions, start=1):
            result = memory.add(text, source="user", **scope)
            print(f"Add {idx}: {text}")
            _print_json(_summarize_agent_result(result))

        _print_header("Step 3: Batch Ingestion (Memory.add_many)")
        batch_result = memory.add_many(
            [
                {
                    "text": "I deploy on macOS for local dev and Linux in CI.",
                    "source": "user",
                    **scope,
                },
                {
                    "content": "My retrieval preference is balanced mode unless debugging.",
                    "role": "user",
                    **scope,
                },
            ],
            continue_on_error=False,
        )
        _print_json(batch_result)

        _print_header("Step 4: Structured Writes (add_entity/add_fact)")
        project_entity = memory.add_entity(
            name="ground-memory-core",
            entity_type=EntityType.FACILITY,
            attributes={"kind": "repository", "language": "python"},
            uniqueness_key="repo:ground-memory-core",
        )
        _print_json(project_entity)

        ontology_entity = memory.add_entity(
            name="ontology:project",
            entity_type=EntityType.FACILITY,
            attributes={"kind": "ontology_anchor"},
            uniqueness_key="ontology:project",
        )
        _print_json(ontology_entity)

        project_entity_id = project_entity["entity"]["id"]
        ontology_entity_id = ontology_entity["entity"]["id"]

        project_relation_fact = memory.add_fact(
            subject_id=project_entity_id,
            relation=RelationType.RELATED_TO,
            object_id=ontology_entity_id,
            source="system",
            **scope,
        )
        _print_json(project_relation_fact)

        project_priority_fact = memory.add_fact(
            subject_id=project_entity_id,
            relation=RelationType.HAS_ATTRIBUTE,
            value="priority=high",
            source="system",
            **scope,
        )
        _print_json(project_priority_fact)

        fact_payload = project_priority_fact.get("fact") or {}
        original_fact_id = fact_payload.get("id")
        if not original_fact_id:
            raise RuntimeError("Expected add_fact to return an approved fact id")

        _print_header("Step 5: Supersession and Retirement")
        updated = memory.update_fact(
            original_fact_id,
            value="priority=critical",
            source="system",
            **scope,
        )
        _print_json(updated)

        updated_fact_payload = updated.get("fact") or {}
        updated_fact_id = updated_fact_payload.get("id")
        if updated_fact_id:
            deleted = memory.delete_fact(updated_fact_id, reason="demo_cleanup")
            _print_json(deleted)

        history = memory.history(fact_id=original_fact_id, include_inactive=True, limit=20)
        print(f"History count for original fact lineage: {len(history)}")
        if history:
            _print_json(history[:2])

        _print_header("Step 6: Retrieval and Context")
        search_results = memory.search(
            "What is my project codename and stack preference?",
            **scope,
            limit=8,
            rerank_debug=True,
        )
        print(f"Search hits: {len(search_results)}")
        if search_results:
            _print_json(search_results[:3])

        context = memory.build_context(
            "Summarize user preferences and project metadata.",
            **scope,
            max_facts=10,
        )
        _print_json(
            {
                "seed_entities": len(context.seed_entities),
                "facts": len(context.facts),
                "entities": len(context.entities),
                "retrieval_metadata": context.retrieval_metadata,
            }
        )

        prompt_block = memory.build_memory_prompt(
            "What should I remember about this user and project?",
            **scope,
            limit=8,
        )
        print("Prompt block:")
        print(prompt_block)

        _print_header("Step 7: Snapshot Views")
        all_data = memory.get_all(**scope)
        _print_json(
            {
                "entities": len(all_data.get("entities", [])),
                "facts": len(all_data.get("facts", [])),
                "interactions": len(all_data.get("interactions", [])),
                "statistics": all_data.get("statistics", {}),
                "optimization": all_data.get("optimization", {}),
            }
        )

        _run_optional_service_check(memory, scope=scope)

        _print_header("Neo4j Visualization")
        print("Open Neo4j Browser at http://localhost:7474 and run:")
        print("MATCH (n:Entity)-[r]->(m:Entity) RETURN n,r,m LIMIT 100;")
        print("")
        print("To focus this demo graph:")
        print("MATCH (n:Entity)-[r]->(m:Entity)")
        print(
            "WHERE n.name CONTAINS 'openrouter-demo-user' OR n.name CONTAINS 'ground-memory-core'"
        )
        print("RETURN n,r,m;")

        _print_header("Done")
        print("End-to-end OpenRouter demo completed successfully.")


if __name__ == "__main__":
    try:
        run_demo()
    except Exception as exc:
        print(f"Demo failed: {type(exc).__name__}: {exc}")
        raise
