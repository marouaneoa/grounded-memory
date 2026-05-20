"""
Domain-agnostic Grounded Memory System orchestration.

This module defines the core runtime flow independent from any specific domain.
Domain packages can plug in constraint registration and expert knowledge while
reusing the same grounding lifecycle.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from grounded_memory.core.constraints import ConstraintValidator
from grounded_memory.core.grounding import GroundingOperator
from grounded_memory.core.store import MemoryStore

if TYPE_CHECKING:
    from grounded_memory.core.neo4j_store import Neo4jConfig as Neo4jConfigType

StorageBackend = Literal["memory", "hybrid", "postgres", "postgres_hybrid"]

# Hybrid store dependency
try:
    from grounded_memory.core.hybrid_store import HybridMemoryStore
    from grounded_memory.core.neo4j_store import Neo4jConfig

    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False
    HybridMemoryStore = None
    Neo4jConfig = None

try:
    from grounded_memory.core.postgres_hybrid_store import PostgresHybridMemoryStore

    HAS_POSTGRES_RUNTIME = True
except ImportError:
    HAS_POSTGRES_RUNTIME = False
    PostgresHybridMemoryStore = None


ConstraintConfigurator = Callable[[ConstraintValidator], None]


class GroundedMemorySystem:
    """
    Domain-agnostic high-level interface to the Grounded Memory System.

    Supports storage modes:
    - In-memory only (default): MemoryStore
    - In-memory + Neo4j graph projection: HybridMemoryStore
    - PostgreSQL durable store (+ optional Neo4j projection): PostgresHybridMemoryStore

    Domain-specific constraints are injected through `configure_validator`.
    """

    def __init__(
        self,
        neo4j_config: Neo4jConfigType | None = None,
        storage_backend: StorageBackend | str | None = None,
        configure_validator: ConstraintConfigurator | None = None,
    ):
        resolved_backend = (storage_backend or "").strip().lower()
        if not resolved_backend:
            resolved_backend = "postgres_hybrid"

        valid_backends = {"memory", "hybrid", "postgres", "postgres_hybrid"}
        if resolved_backend not in valid_backends:
            allowed = ", ".join(sorted(valid_backends))
            raise ValueError(
                f"Unsupported storage backend '{resolved_backend}'. Expected one of: {allowed}."
            )

        self.storage_backend: StorageBackend = resolved_backend  # type: ignore[assignment]

        if self.storage_backend in {"postgres", "postgres_hybrid"}:
            if not HAS_POSTGRES_RUNTIME or PostgresHybridMemoryStore is None:
                raise RuntimeError(
                    "PostgreSQL backend requested, but postgres runtime dependencies are unavailable. "
                    "Install postgres extras and ensure asyncpg is available."
                )

            self.memory_store = PostgresHybridMemoryStore(
                neo4j_config=neo4j_config,
                sync_enabled=(self.storage_backend == "postgres_hybrid"),
            )
            if self.storage_backend == "postgres_hybrid":
                self.memory_store.initialize_neo4j()
        elif self.storage_backend == "hybrid":
            if not HAS_NEO4J or HybridMemoryStore is None:
                raise RuntimeError(
                    "Neo4j-enabled system requested, but Neo4j dependencies are unavailable. "
                    "Install required Neo4j packages before continuing."
                )
            self.memory_store = HybridMemoryStore(
                neo4j_config=neo4j_config,
            )
            self.memory_store.initialize_neo4j()
        else:
            self.memory_store = MemoryStore()

        self.validator = ConstraintValidator()
        if configure_validator is not None:
            configure_validator(self.validator)

        self.grounding_operator = GroundingOperator(
            validator=self.validator,
            memory_store=self.memory_store,
        )

    @property
    def has_neo4j(self) -> bool:
        """Check if Neo4j is active in this system."""
        if hasattr(self.memory_store, "has_neo4j"):
            return bool(self.memory_store.has_neo4j)
        return False

    def rebuild_neo4j(self) -> dict[str, int] | None:
        """Rebuild Neo4j graph from primary store. Returns sync counts or None."""
        if hasattr(self.memory_store, "rebuild_neo4j") and self.has_neo4j:
            return self.memory_store.rebuild_neo4j()
        return None

    def get_statistics(self) -> dict[str, int]:
        """Get system statistics (includes Neo4j stats if available)."""
        return self.memory_store.get_statistics()

    def close(self) -> None:
        """Clean up resources (close store connections, background threads, etc.)."""
        if hasattr(self.memory_store, "close") and callable(self.memory_store.close):
            self.memory_store.close()
