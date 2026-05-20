#!/usr/bin/env python3
"""
Test PostgreSQL storage backend with new rich provenance fields.

Run:
    PYTHONPATH=src python tests/test_postgres_storage.py
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from grounded_memory.core.models import (
    CandidateFact,
    Entity,
    EntityType,
    RelationType,
    ValidatedFact,
)
from grounded_memory.core.postgres_store import PostgresConfig, PostgresStore


async def main():
    import uuid

    from grounded_memory.core.models import ActorType, Interaction

    config = PostgresConfig(
        host="localhost",
        port=5432,
        database="grounded_memory",
        user="postgres",
        password="postgres",
    )
    store = PostgresStore(config=config)
    await store.initialize()

    run_id = str(uuid.uuid4())[:8]

    # 1. Create and add entities
    user = Entity(id=f"u-postgres-{run_id}", entity_type=EntityType.PERSON, name="PG User")
    tool = Entity(id=f"t-postgres-{run_id}", entity_type=EntityType.TOOL, name="PG Tool")
    await store.add_entity(user)
    await store.add_entity(tool)

    # 2. Add interaction
    raw_text = "The PG User uses the PG Tool daily."
    interaction = Interaction(id=f"int-pg-{run_id}", actor=ActorType.USER, raw_text=raw_text)
    await store.add_interaction(interaction)

    # 3. Add Candidate Fact
    candidate = CandidateFact(
        id=f"cf-pg-{run_id}",
        source_interaction_id=interaction.id,
        subject_entity_id=user.id,
        relation=RelationType.USED_BY,
        object_entity_id=tool.id,
        confidence=0.95,
        attributes={"key": "primary_tool"},
    )
    # PostgresStore doesn't have an add_candidate_fact method natively exposed in the interface?
    # Let's check... if not, we do it via raw SQL.
    async with store.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO candidate_facts (
                id, source_interaction_id, subject_entity_id, relation, object_entity_id,
                value, confidence, attributes
            ) VALUES ($1, $2, $3, $4::relation_type, $5, $6, $7, $8::jsonb)
            """,
            candidate.id,
            candidate.source_interaction_id,
            candidate.subject_entity_id,
            candidate.relation.value.lower(),
            candidate.object_entity_id,
            candidate.value,
            candidate.confidence,
            json.dumps(candidate.attributes),
        )

    # 4. Ground fact with rich provenance
    embedding_data = [0.1, 0.2, 0.3, 0.4]
    fact = ValidatedFact(
        id=f"vf-pg-{run_id}",
        candidate_fact_id=candidate.id,
        source_interaction_id=interaction.id,
        subject_id=user.id,
        relation=RelationType.USED_BY,
        object_id=tool.id,
        confidence=0.95,
        valid_from=datetime.now(timezone.utc),
        source_text=raw_text,
        embedding=embedding_data,
        source_metadata={"actor": "user", "context": "demo"},
        attributes={"key": "primary_tool"},
    )

    fact_id = await store.add_validated_fact(fact)
    print(f"✅ Fact inserted with ID: {fact_id}")

    # 5. Retrieve fact and verify fields
    retrieved = await store.get_fact(fact_id)
    assert retrieved is not None, "Fact not found!"
    assert retrieved.source_text == raw_text, "source_text mismatch"
    assert retrieved.embedding == embedding_data, "embedding mismatch"
    assert retrieved.source_metadata == {"actor": "user", "context": "demo"}, (
        "source_metadata mismatch"
    )
    assert retrieved.attributes == {"key": "primary_tool"}, "attributes mismatch"

    print("✅ Fact retrieved successfully with all provenance fields intact!")

    # 6. Clean up
    async with store.acquire() as conn:
        await conn.execute("DELETE FROM validated_facts WHERE id = $1", fact.id)
        await conn.execute("DELETE FROM candidate_facts WHERE id = $1", candidate.id)
        await conn.execute("DELETE FROM interactions WHERE id = $1", interaction.id)
        await conn.execute("DELETE FROM entities WHERE id IN ($1, $2)", user.id, tool.id)
    print("✅ Cleanup complete.")


if __name__ == "__main__":
    asyncio.run(main())
