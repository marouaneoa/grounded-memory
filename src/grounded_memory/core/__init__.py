"""Core module for Grounded Memory System."""

from grounded_memory.core.grounding import (
    AsyncGroundingOperator,
    GroundingDecision,
    GroundingOperator,
    GroundingResult,
    process_interaction_facts,
    process_interaction_facts_async,
)
from grounded_memory.core.intent import (
    BaseIntentRouter,
    HybridIntentRouter,
    IntentAction,
    KeywordIntentRouter,
    LLMIntentRouter,
    UserIntent,
)
from grounded_memory.core.models import (
    ActorType,
    CandidateFact,
    CandidateFactStatus,
    Constraint,
    ConstraintType,
    Entity,
    EntityType,
    Interaction,
    RejectionRecord,
    RelationType,
    ValidatedFact,
)
from grounded_memory.core.store import MemoryStore
from grounded_memory.core.system import GroundedMemorySystem

# PostgreSQL store is optional (requires asyncpg)
try:
    from grounded_memory.core.postgres_store import (
        PostgresConfig,
        PostgresStore,
        create_postgres_store,
    )

    # Alias for future naming convention
    PostgresKnowledgeStore = PostgresStore
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False
    PostgresKnowledgeStore = None
    PostgresStore = None
    PostgresConfig = None
    create_postgres_store = None

# Neo4j store is optional (requires neo4j driver)
try:
    from grounded_memory.core.neo4j_store import (
        Neo4jConfig,
        Neo4jStore,
        create_neo4j_store,
    )

    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False
    Neo4jStore = None
    Neo4jConfig = None
    create_neo4j_store = None

# Hybrid store (requires neo4j)
try:
    from grounded_memory.core.hybrid_store import HybridMemoryStore
except ImportError:
    HybridMemoryStore = None

try:
    from grounded_memory.core.postgres_hybrid_store import PostgresHybridMemoryStore
except ImportError:
    PostgresHybridMemoryStore = None

__all__ = [
    # Models
    "Interaction",
    "Entity",
    "CandidateFact",
    "ValidatedFact",
    "Constraint",
    "RelationType",
    "EntityType",
    "ConstraintType",
    "RejectionRecord",
    "ActorType",
    "CandidateFactStatus",
    # Intent routing
    "IntentAction",
    "UserIntent",
    "BaseIntentRouter",
    "KeywordIntentRouter",
    "LLMIntentRouter",
    "HybridIntentRouter",
    # Grounding
    "GroundingOperator",
    "AsyncGroundingOperator",
    "GroundingDecision",
    "GroundingResult",
    "process_interaction_facts",
    "process_interaction_facts_async",
    # Stores
    "MemoryStore",
    "GroundedMemorySystem",
    "HybridMemoryStore",
    "PostgresHybridMemoryStore",
    "PostgresKnowledgeStore",
    "PostgresStore",
    "PostgresConfig",
    "create_postgres_store",
    "HAS_POSTGRES",
    # Neo4j
    "Neo4jStore",
    "Neo4jConfig",
    "create_neo4j_store",
    "HAS_NEO4J",
]
