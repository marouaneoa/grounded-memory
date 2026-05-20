"""PostgreSQL-backed hybrid memory store for the synchronous SDK/runtime.

This store keeps the existing in-process MemoryStore behavior for reads while
persisting all writes to PostgreSQL for durability. Neo4j graph projection
remains optional and is handled via HybridMemoryStore.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeVar

from grounded_memory.core.hybrid_store import HybridMemoryStore

if TYPE_CHECKING:
    from grounded_memory.core.neo4j_store import Neo4jStore
    from grounded_memory.core.system import Neo4jConfig
from grounded_memory.core.models import (
    CandidateFact,
    CandidateFactStatus,
    Entity,
    Interaction,
    RejectionRecord,
    ValidatedFact,
)
from grounded_memory.core.postgres_store import PostgresConfig, PostgresStore
from grounded_memory.core.store import MemoryStore

T = TypeVar("T")


class _AsyncLoopRunner:
    """Run async PostgreSQL operations from synchronous code safely."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run,
            name="gmem-postgres-loop",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coroutine: Coroutine[Any, Any, T]) -> T:
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result()

    def close(self) -> None:
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)


class PostgresHybridMemoryStore(HybridMemoryStore):
    """HybridMemoryStore with durable PostgreSQL persistence."""

    def __init__(
        self,
        neo4j_config: Neo4jConfig | None = None,
        neo4j_store: Neo4jStore | None = None,
        sync_enabled: bool = True,
        postgres_config: PostgresConfig | None = None,
        create_postgres_schema: bool = True,
    ):
        super().__init__(
            neo4j_config=neo4j_config,
            neo4j_store=neo4j_store,
            sync_enabled=sync_enabled,
        )

        self._postgres_runner: _AsyncLoopRunner | None = _AsyncLoopRunner()
        self._postgres_store: PostgresStore | None = None

        try:
            config = postgres_config or PostgresConfig.from_env()
            self._postgres_store = PostgresStore(config)
            self._sync_to_postgres(
                "initialize",
                self._postgres_store.initialize(create_schema=create_postgres_schema),
            )
            self._rehydrate_from_postgres()
        except Exception as exc:
            if self._postgres_runner is not None:
                self._postgres_runner.close()
            self._postgres_runner = None
            self._postgres_store = None
            raise RuntimeError(f"Failed to initialize PostgreSQL: {exc}") from exc

    @property
    def has_postgres(self) -> bool:
        return self._postgres_store is not None and self._postgres_runner is not None

    def _sync_to_postgres(self, operation: str, coroutine: Coroutine[Any, Any, T]) -> T:
        if self._postgres_store is None or self._postgres_runner is None:
            raise RuntimeError("PostgreSQL store is not initialized")
        try:
            return self._postgres_runner.run(coroutine)
        except Exception as exc:
            raise RuntimeError(f"PostgreSQL sync failed for {operation}: {exc}") from exc

    def _rehydrate_from_postgres(self) -> None:
        """Load persisted Postgres state into the in-process source-of-truth cache."""
        if self._postgres_store is None:
            return

        entities = self._sync_to_postgres(
            "get_all_entities",
            self._postgres_store.get_all_entities(),
        )
        interactions = self._sync_to_postgres(
            "get_interactions",
            self._postgres_store.get_interactions(limit=100000),
        )
        facts = self._sync_to_postgres(
            "get_all_validated_facts",
            self._postgres_store.get_all_validated_facts(),
        )
        rejections = self._sync_to_postgres(
            "get_all_rejections",
            self._postgres_store.get_all_rejections(),
        )

        for entity in entities:
            MemoryStore.add_entity(self, entity)
        for interaction in reversed(interactions):
            MemoryStore.add_interaction(self, interaction)
        for fact in facts:
            MemoryStore.add_validated_fact(self, fact)
        for rejection in rejections:
            MemoryStore.add_rejection(self, rejection)

    def _ensure_candidate_fact(self, fact: ValidatedFact) -> None:
        if self._postgres_store is None:
            return

        existing = self._sync_to_postgres(
            "get_candidate_fact",
            self._postgres_store.get_candidate_fact(fact.candidate_fact_id),
        )
        if existing is not None:
            self._sync_to_postgres(
                "accept_candidate_fact",
                self._postgres_store.accept_candidate_fact(fact.candidate_fact_id),
            )
            return

        candidate = CandidateFact(
            id=fact.candidate_fact_id,
            source_interaction_id=fact.source_interaction_id,
            subject_entity_id=fact.subject_id,
            relation=fact.relation,
            object_entity_id=fact.object_id,
            value=fact.value,
            confidence=fact.confidence,
            extracted_at=fact.valid_from,
            status=CandidateFactStatus.ACCEPTED,
            attributes=dict(fact.attributes),
        )
        self._sync_to_postgres(
            "add_candidate_fact",
            self._postgres_store.add_candidate_fact(candidate),
        )

    def add_entity(self, entity: Entity) -> None:
        super().add_entity(entity)
        if self._postgres_store is not None:
            self._sync_to_postgres("add_entity", self._postgres_store.add_entity(entity))

    def add_interaction(self, interaction: Interaction) -> None:
        super().add_interaction(interaction)
        if self._postgres_store is not None:
            self._sync_to_postgres(
                "add_interaction",
                self._postgres_store.add_interaction(interaction),
            )

    def add_candidate_facts(self, facts: list[CandidateFact]) -> list[str]:
        if self._postgres_store is None:
            raise RuntimeError("PostgreSQL store is not initialized")
        return self._sync_to_postgres(
            "add_candidate_facts",
            self._postgres_store.add_candidate_facts(facts),
        )

    def add_validated_fact(self, fact: ValidatedFact) -> None:
        super().add_validated_fact(fact)
        if self._postgres_store is not None:
            self._ensure_candidate_fact(fact)
            self._sync_to_postgres(
                "add_validated_fact",
                self._postgres_store.add_validated_fact(fact),
            )

    def supersede_fact(
        self,
        fact_id: str,
        superseded_by: str,
        valid_to: datetime | None = None,
    ) -> None:
        existing = self._facts.get(fact_id)
        super().supersede_fact(fact_id, superseded_by, valid_to)

        if existing is None or self._postgres_store is None:
            return

        updated = self._facts.get(fact_id)
        resolved_valid_to = valid_to
        if resolved_valid_to is None and updated is not None:
            resolved_valid_to = updated.valid_to

        resolved_superseded_by: str | None = superseded_by
        if superseded_by not in self._facts:
            resolved_superseded_by = None

        self._sync_to_postgres(
            "supersede_fact",
            self._postgres_store.supersede_fact(
                fact_id=fact_id,
                superseded_by=resolved_superseded_by,
                valid_to=resolved_valid_to,
            ),
        )

    def add_rejection(self, rejection: RejectionRecord) -> None:
        super().add_rejection(rejection)
        if self._postgres_store is not None:
            self._sync_to_postgres("add_rejection", self._postgres_store.add_rejection(rejection))

    def initialize_neo4j(self, create_schema: bool = True) -> bool:
        initialized = super().initialize_neo4j(create_schema=create_schema)
        if initialized and self.has_neo4j:
            self.rebuild_neo4j()
        return initialized

    def get_statistics(self) -> dict[str, Any]:
        stats = super().get_statistics()
        if self._postgres_store is not None:
            stats["postgres"] = self._sync_to_postgres(
                "get_statistics",
                self._postgres_store.get_statistics(),
            )
            stats["postgres_available"] = True
        else:
            stats["postgres_available"] = False
        return stats

    def clear(self) -> None:
        super().clear()
        if self._postgres_store is not None:
            self._sync_to_postgres("clear", self._postgres_store.clear())

    def close(self) -> None:
        try:
            super().close()
        finally:
            if self._postgres_store is not None:
                self._sync_to_postgres("close", self._postgres_store.close())
                self._postgres_store = None

            if self._postgres_runner is not None:
                self._postgres_runner.close()
                self._postgres_runner = None
