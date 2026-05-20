"""
PostgreSQL Memory Store with Bitemporal Management

This module implements persistent PostgreSQL storage for the Grounded Memory System.
It provides:
- Bitemporal semantics via valid_from/valid_to (valid time) and
    created_at/interaction timestamps (record time)
- Efficient graph traversal queries
- Full-text search on entities
- Audit trail for rejections

Uses asyncpg for async PostgreSQL operations.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

try:
    import asyncpg
    from asyncpg import Connection, Pool

    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False
    asyncpg = None
    Pool = None
    Connection = None

if TYPE_CHECKING:
    pass

from grounded_memory.core.models import (
    ActorType,
    CandidateFact,
    CandidateFactStatus,
    Entity,
    EntityType,
    Interaction,
    RejectionRecord,
    RelationType,
    ValidatedFact,
)

# =============================================================================
# Database Configuration
# =============================================================================


class PostgresConfig:
    """Configuration for PostgreSQL connection."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5432,
        database: str = "grounded_memory",
        user: str = "postgres",
        password: str = "postgres",
        min_connections: int = 2,
        max_connections: int = 10,
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.min_connections = min_connections
        self.max_connections = max_connections

    @classmethod
    def from_env(cls) -> PostgresConfig:
        """Create config from environment variables."""
        return cls(
            host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "grounded_memory"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            min_connections=int(os.getenv("POSTGRES_MIN_CONN", "2")),
            max_connections=int(os.getenv("POSTGRES_MAX_CONN", "10")),
        )

    @property
    def dsn(self) -> str:
        """Get PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


# =============================================================================
# SQL Schema
# =============================================================================

SCHEMA_SQL = """
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For fuzzy text search

-- Actor Types Enum
DO $$ BEGIN
    CREATE TYPE actor_type AS ENUM ('user', 'agent', 'tool', 'system');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Candidate Fact Status Enum
DO $$ BEGIN
    CREATE TYPE candidate_fact_status AS ENUM ('pending', 'accepted', 'rejected');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Entities Table
CREATE TABLE IF NOT EXISTS entities (
    id VARCHAR(255) PRIMARY KEY,
    entity_type VARCHAR(100) NOT NULL,
    name VARCHAR(500) NOT NULL,
    canonical_id VARCHAR(255),
    attributes JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for entity search
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_canonical_id ON entities(canonical_id);
CREATE INDEX IF NOT EXISTS idx_entities_name_trgm ON entities USING gin(name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_entities_name_lower ON entities(LOWER(name));

-- Interactions Table (immutable event log) - UPDATED SCHEMA
CREATE TABLE IF NOT EXISTS interactions (
    id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255),
    app_id VARCHAR(255),
    user_id VARCHAR(255),
    agent_id VARCHAR(255),
    run_id VARCHAR(255),
    session_id VARCHAR(255),
    space_type VARCHAR(100),
    actor actor_type NOT NULL DEFAULT 'user',
    raw_text TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_interactions_tenant_app_user ON interactions(tenant_id, app_id, user_id);
CREATE INDEX IF NOT EXISTS idx_interactions_user_id ON interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_interactions_run_id ON interactions(run_id);
CREATE INDEX IF NOT EXISTS idx_interactions_session_id ON interactions(session_id);
CREATE INDEX IF NOT EXISTS idx_interactions_actor ON interactions(actor);

-- Candidate Facts Table (LLM extracted, pending validation)
CREATE TABLE IF NOT EXISTS candidate_facts (
    id VARCHAR(255) PRIMARY KEY,
    source_interaction_id VARCHAR(255) NOT NULL REFERENCES interactions(id),
    subject_entity_id VARCHAR(255) NOT NULL REFERENCES entities(id),
    relation VARCHAR(100) NOT NULL,
    object_entity_id VARCHAR(255) REFERENCES entities(id),
    value TEXT,
    confidence FLOAT DEFAULT 0.9,
    extracted_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    status candidate_fact_status NOT NULL DEFAULT 'pending',
    rejection_reason TEXT,
    attributes JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_candidate_facts_interaction ON candidate_facts(source_interaction_id);
CREATE INDEX IF NOT EXISTS idx_candidate_facts_subject ON candidate_facts(subject_entity_id);
CREATE INDEX IF NOT EXISTS idx_candidate_facts_status ON candidate_facts(status);
CREATE INDEX IF NOT EXISTS idx_candidate_facts_extracted ON candidate_facts(extracted_at DESC);

-- Validated Facts Table (bitemporal foundation: valid time + record time)
CREATE TABLE IF NOT EXISTS validated_facts (
    id VARCHAR(255) PRIMARY KEY,
    candidate_fact_id VARCHAR(255) REFERENCES candidate_facts(id),
    subject_id VARCHAR(255) NOT NULL REFERENCES entities(id),
    relation VARCHAR(100) NOT NULL,
    object_id VARCHAR(255) REFERENCES entities(id),
    value TEXT,
    confidence FLOAT DEFAULT 1.0,
    source_interaction_id VARCHAR(255) REFERENCES interactions(id),
    valid_from TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    valid_to TIMESTAMP WITH TIME ZONE,
    superseded_by VARCHAR(255) REFERENCES validated_facts(id),
    metadata JSONB DEFAULT '{}',
    source_text TEXT,
    source_metadata JSONB DEFAULT '{}',
    embedding JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()  -- Record-time insert timestamp
);

-- Indices for fact queries
CREATE INDEX IF NOT EXISTS idx_facts_subject ON validated_facts(subject_id);
CREATE INDEX IF NOT EXISTS idx_facts_object ON validated_facts(object_id);
CREATE INDEX IF NOT EXISTS idx_facts_relation ON validated_facts(relation);
CREATE INDEX IF NOT EXISTS idx_facts_valid_from ON validated_facts(valid_from);
CREATE INDEX IF NOT EXISTS idx_facts_valid_to ON validated_facts(valid_to);
CREATE INDEX IF NOT EXISTS idx_facts_active ON validated_facts(subject_id, relation) 
    WHERE valid_to IS NULL;
CREATE INDEX IF NOT EXISTS idx_facts_candidate ON validated_facts(candidate_fact_id);

-- Rejection Records Table (audit trail)
CREATE TABLE IF NOT EXISTS rejection_records (
    id VARCHAR(255) PRIMARY KEY,
    candidate_fact_id VARCHAR(255) NOT NULL REFERENCES candidate_facts(id),
    constraint_id VARCHAR(255) NOT NULL,
    constraint_name VARCHAR(255),
    reason TEXT NOT NULL,
    domain_reasoning TEXT,
    alternatives TEXT[],
    severity VARCHAR(50) DEFAULT 'error',
    rejected_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rejections_candidate ON rejection_records(candidate_fact_id);
CREATE INDEX IF NOT EXISTS idx_rejections_constraint ON rejection_records(constraint_id);
CREATE INDEX IF NOT EXISTS idx_rejections_time ON rejection_records(rejected_at DESC);

-- View for active facts (helper view)
CREATE OR REPLACE VIEW active_facts AS
SELECT * FROM validated_facts 
WHERE valid_to IS NULL;

-- View for pending candidate facts
CREATE OR REPLACE VIEW pending_candidates AS
SELECT * FROM candidate_facts
WHERE status = 'pending';

-- Function to supersede a fact
CREATE OR REPLACE FUNCTION supersede_fact(
    old_fact_id VARCHAR(255),
    new_fact_id VARCHAR(255),
    supersede_time TIMESTAMP WITH TIME ZONE DEFAULT NOW()
) RETURNS VOID AS $$
BEGIN
    UPDATE validated_facts 
    SET valid_to = supersede_time, superseded_by = new_fact_id
    WHERE id = old_fact_id AND valid_to IS NULL;
END;
$$ LANGUAGE plpgsql;

-- Function to accept a candidate fact
CREATE OR REPLACE FUNCTION accept_candidate_fact(
    fact_id VARCHAR(255)
) RETURNS VOID AS $$
BEGIN
    UPDATE candidate_facts
    SET status = 'accepted'
    WHERE id = fact_id AND status = 'pending';
END;
$$ LANGUAGE plpgsql;

-- Function to reject a candidate fact
CREATE OR REPLACE FUNCTION reject_candidate_fact(
    fact_id VARCHAR(255),
    reason TEXT
) RETURNS VOID AS $$
BEGIN
    UPDATE candidate_facts
    SET status = 'rejected', rejection_reason = reason
    WHERE id = fact_id AND status = 'pending';
END;
$$ LANGUAGE plpgsql;
"""

MIGRATION_SQL = [
    # Drop helper views first so type migration can alter dependent columns.
    "DROP VIEW IF EXISTS active_facts;",
    "DROP VIEW IF EXISTS pending_candidates;",
    """
    DO $$ BEGIN
        IF EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'entities'
              AND column_name = 'entity_type'
              AND udt_name = 'entity_type'
        ) THEN
            ALTER TABLE entities
            ALTER COLUMN entity_type TYPE VARCHAR(100)
            USING entity_type::text;
        END IF;
    EXCEPTION
        WHEN undefined_table THEN null;
    END $$;
    """,
    """
    DO $$ BEGIN
        ALTER TABLE entities ADD COLUMN IF NOT EXISTS canonical_id VARCHAR(255);
    EXCEPTION
        WHEN undefined_table THEN null;
    END $$;
    """,
    """
    DO $$ BEGIN
        CREATE INDEX IF NOT EXISTS idx_entities_canonical_id ON entities(canonical_id);
    EXCEPTION
        WHEN undefined_table THEN null;
    END $$;
    """,
    """
    DO $$ BEGIN
        ALTER TABLE interactions ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(255);
        ALTER TABLE interactions ADD COLUMN IF NOT EXISTS app_id VARCHAR(255);
        ALTER TABLE interactions ADD COLUMN IF NOT EXISTS agent_id VARCHAR(255);
        ALTER TABLE interactions ADD COLUMN IF NOT EXISTS run_id VARCHAR(255);
        ALTER TABLE interactions ADD COLUMN IF NOT EXISTS space_type VARCHAR(100);
    EXCEPTION
        WHEN undefined_table THEN null;
    END $$;
    """,
    """
    DO $$ BEGIN
        CREATE INDEX IF NOT EXISTS idx_interactions_tenant_app_user ON interactions(tenant_id, app_id, user_id);
        CREATE INDEX IF NOT EXISTS idx_interactions_run_id ON interactions(run_id);
    EXCEPTION
        WHEN undefined_table THEN null;
    END $$;
    """,
    """
    DO $$ BEGIN
        ALTER TABLE validated_facts ADD COLUMN IF NOT EXISTS source_text TEXT;
        ALTER TABLE validated_facts ADD COLUMN IF NOT EXISTS source_metadata JSONB DEFAULT '{}';
        ALTER TABLE validated_facts ADD COLUMN IF NOT EXISTS embedding JSONB;
    EXCEPTION
        WHEN undefined_table THEN null;
    END $$;
    """,
    """
    DO $$ BEGIN
        IF EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'candidate_facts'
              AND column_name = 'relation'
              AND udt_name = 'relation_type'
        ) THEN
            ALTER TABLE candidate_facts
            ALTER COLUMN relation TYPE VARCHAR(100)
            USING relation::text;
        END IF;
    EXCEPTION
        WHEN undefined_table THEN null;
    END $$;
    """,
    """
    DO $$ BEGIN
        IF EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'validated_facts'
              AND column_name = 'relation'
              AND udt_name = 'relation_type'
        ) THEN
            ALTER TABLE validated_facts
            ALTER COLUMN relation TYPE VARCHAR(100)
            USING relation::text;
        END IF;
    EXCEPTION
        WHEN undefined_table THEN null;
    END $$;
    """,
]


def _coerce_entity_type(value: Any) -> EntityType:
    """Coerce DB taxonomy values into the current EntityType enum."""
    if value is None:
        return EntityType.FACILITY

    normalized = str(value).strip()
    for candidate in (normalized, normalized.lower()):
        try:
            return EntityType(candidate)
        except ValueError:
            continue

    legacy_aliases: dict[str, EntityType] = {
        "provider": EntityType.CLINICIAN,
    }
    return legacy_aliases.get(normalized.lower(), EntityType.FACILITY)


def _coerce_relation_type(value: Any) -> RelationType:
    """Coerce DB relation strings across legacy/new formats."""
    if value is None:
        return RelationType.RELATED_TO

    normalized = str(value).strip()
    for candidate in (normalized, normalized.upper()):
        try:
            return RelationType(candidate)
        except ValueError:
            continue

    legacy_aliases: dict[str, RelationType] = {
        "takes_medication": RelationType.PRESCRIBED,
        "prescribes": RelationType.PRESCRIBED,
        "treats_condition": RelationType.TREATS,
        "interacts_with": RelationType.RELATED_TO,
        "administered_at": RelationType.RELATED_TO,
        "documented_in": RelationType.RELATED_TO,
    }
    return legacy_aliases.get(normalized.lower(), RelationType.RELATED_TO)


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    """Read an asyncpg Record key with compatibility for older schemas."""
    try:
        return row[key]
    except (KeyError, IndexError):
        return default


# =============================================================================
# PostgreSQL Store Implementation
# =============================================================================


class PostgresStore:
    """
    PostgreSQL implementation of the knowledge store.

    This store provides persistent, production-ready storage with:
    - Connection pooling for performance
    - Temporal queries (point-in-time state)
    - Full-text search on entities
    - Graph traversal queries

    Usage:
        config = PostgresConfig.from_env()
        store = PostgresStore(config)
        await store.initialize()

        # Use the store
        await store.add_entity(entity)

        # Clean up
        await store.close()
    """

    def __init__(self, config: PostgresConfig | None = None):
        if not HAS_ASYNCPG:
            raise ImportError(
                "asyncpg is required for PostgreSQL support. Install it with: pip install asyncpg"
            )

        self.config = config or PostgresConfig.from_env()
        self._pool: Any = None

    async def initialize(self, create_schema: bool = True) -> None:
        """
        Initialize the database connection pool and optionally create schema.

        Args:
            create_schema: If True, create tables if they don't exist
        """
        self._pool = await asyncpg.create_pool(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.user,
            password=self.config.password,
            min_size=self.config.min_connections,
            max_size=self.config.max_connections,
        )

        if create_schema:
            await self._create_schema()

    async def _create_schema(self) -> None:
        """Create the database schema if it doesn't exist."""
        async with self._pool.acquire() as conn:
            for statement in MIGRATION_SQL:
                await conn.execute(statement)
            await conn.execute(SCHEMA_SQL)

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        async with self._pool.acquire() as conn:
            yield conn

    # =========================================================================
    # Entity Operations
    # =========================================================================

    async def add_entity(self, entity: Entity) -> str:
        """Add or update an entity. Returns the entity ID."""
        async with self.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO entities (id, entity_type, name, canonical_id, attributes, updated_at)
                VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    entity_type = EXCLUDED.entity_type,
                    canonical_id = EXCLUDED.canonical_id,
                    attributes = EXCLUDED.attributes,
                    updated_at = NOW()
                """,
                entity.id,
                entity.entity_type.value,
                entity.name,
                entity.canonical_id,
                json.dumps(entity.attributes),
            )
        return entity.id

    async def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM entities WHERE id = $1",
                entity_id,
            )
            return self._row_to_entity(row) if row else None

    async def get_entities_by_type(self, entity_type: EntityType) -> list[Entity]:
        """Get all entities of a specific type."""
        async with self.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM entities WHERE LOWER(entity_type) = LOWER($1)",
                entity_type.value,
            )
            return [self._row_to_entity(row) for row in rows]

    async def find_entity_by_name(
        self,
        name: str,
        entity_type: EntityType | None = None,
    ) -> Entity | None:
        """Find an entity by exact name (case-insensitive)."""
        async with self.acquire() as conn:
            if entity_type:
                row = await conn.fetchrow(
                    """
                    SELECT * FROM entities 
                    WHERE LOWER(name) = LOWER($1) AND LOWER(entity_type) = LOWER($2)
                    LIMIT 1
                    """,
                    name,
                    entity_type.value,
                )
            else:
                row = await conn.fetchrow(
                    "SELECT * FROM entities WHERE LOWER(name) = LOWER($1) LIMIT 1",
                    name,
                )
            return self._row_to_entity(row) if row else None

    async def search_entities(
        self,
        query: str,
        entity_type: EntityType | None = None,
        limit: int = 10,
    ) -> list[Entity]:
        """Search entities by name (fuzzy match)."""
        async with self.acquire() as conn:
            if entity_type:
                rows = await conn.fetch(
                    """
                    SELECT *, similarity(name, $1) as sim 
                    FROM entities 
                    WHERE name % $1 AND LOWER(entity_type) = LOWER($2)
                    ORDER BY sim DESC
                    LIMIT $3
                    """,
                    query,
                    entity_type.value,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT *, similarity(name, $1) as sim 
                    FROM entities 
                    WHERE name % $1
                    ORDER BY sim DESC
                    LIMIT $2
                    """,
                    query,
                    limit,
                )
            return [self._row_to_entity(row) for row in rows]

    async def get_all_entities(self) -> list[Entity]:
        """Get all entities from PostgreSQL."""
        async with self.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM entities ORDER BY created_at ASC")
            return [self._row_to_entity(row) for row in rows]

    def _row_to_entity(self, row) -> Entity:
        """Convert a database row to an Entity."""
        return Entity(
            id=row["id"],
            entity_type=_coerce_entity_type(row["entity_type"]),
            name=row["name"],
            canonical_id=_row_get(row, "canonical_id"),
            attributes=json.loads(row["attributes"])
            if isinstance(row["attributes"], str)
            else row["attributes"],
        )

    # =========================================================================
    # Fact Operations
    # =========================================================================

    async def add_validated_fact(self, fact: ValidatedFact) -> str:
        """Add a validated fact to the store. Returns the fact ID."""
        async with self.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO validated_facts 
                (id, candidate_fact_id, subject_id, relation, object_id, value, confidence,
                 source_interaction_id, valid_from, valid_to, superseded_by, metadata,
                 source_text, source_metadata, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                """,
                fact.id,
                fact.candidate_fact_id,
                fact.subject_id,
                fact.relation.value,
                fact.object_id,
                fact.value,
                fact.confidence,
                fact.source_interaction_id,
                fact.valid_from,
                fact.valid_to,
                fact.superseded_by,
                json.dumps(fact.attributes),
                fact.source_text,
                json.dumps(fact.source_metadata),
                json.dumps(fact.embedding) if fact.embedding is not None else None,
            )
        return fact.id

    async def get_fact(self, fact_id: str) -> ValidatedFact | None:
        """Get a fact by ID."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM validated_facts WHERE id = $1",
                fact_id,
            )
            return self._row_to_fact(row) if row else None

    async def get_active_facts_for_entity(
        self,
        entity_id: str,
        at_time: datetime | None = None,
    ) -> list[ValidatedFact]:
        """Get all active facts where entity is subject or object."""
        at_time = at_time or datetime.now(timezone.utc)

        async with self.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM validated_facts
                WHERE (subject_id = $1 OR object_id = $1)
                AND valid_from <= $2
                AND (valid_to IS NULL OR valid_to > $2)
                """,
                entity_id,
                at_time,
            )
            return [self._row_to_fact(row) for row in rows]

    async def get_facts_by_relation(
        self,
        entity_id: str,
        relation: RelationType,
        as_subject: bool = True,
        at_time: datetime | None = None,
    ) -> list[ValidatedFact]:
        """Get facts with a specific relation for an entity."""
        at_time = at_time or datetime.now(timezone.utc)

        async with self.acquire() as conn:
            if as_subject:
                rows = await conn.fetch(
                    """
                    SELECT * FROM validated_facts
                    WHERE subject_id = $1 AND UPPER(relation) = $2
                    AND valid_from <= $3
                    AND (valid_to IS NULL OR valid_to > $3)
                    """,
                    entity_id,
                    relation.value,
                    at_time,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM validated_facts
                    WHERE object_id = $1 AND UPPER(relation) = $2
                    AND valid_from <= $3
                    AND (valid_to IS NULL OR valid_to > $3)
                    """,
                    entity_id,
                    relation.value,
                    at_time,
                )
            return [self._row_to_fact(row) for row in rows]

    async def get_all_facts_by_relation(
        self,
        relation: RelationType,
        at_time: datetime | None = None,
    ) -> list[ValidatedFact]:
        """Get all facts of a specific relation type."""
        at_time = at_time or datetime.now(timezone.utc)

        async with self.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM validated_facts
                WHERE UPPER(relation) = $1
                AND valid_from <= $2
                AND (valid_to IS NULL OR valid_to > $2)
                """,
                relation.value,
                at_time,
            )
            return [self._row_to_fact(row) for row in rows]

    async def get_all_validated_facts(self) -> list[ValidatedFact]:
        """Get all validated facts, including inactive/superseded facts."""
        async with self.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM validated_facts ORDER BY created_at ASC")
            return [self._row_to_fact(row) for row in rows]

    async def supersede_fact(
        self,
        fact_id: str,
        superseded_by: str | None,
        valid_to: datetime | None = None,
    ) -> bool:
        """Supersede a fact by setting its valid_to and superseded_by fields."""
        valid_to = valid_to or datetime.now(timezone.utc)

        async with self.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE validated_facts 
                SET valid_to = $1, superseded_by = $2
                WHERE id = $3 AND valid_to IS NULL
                """,
                valid_to,
                superseded_by,
                fact_id,
            )
            return result != "UPDATE 0"

    def _row_to_fact(self, row) -> ValidatedFact:
        """Convert a database row to a ValidatedFact."""
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        source_metadata = _row_get(row, "source_metadata", {})
        if isinstance(source_metadata, str):
            source_metadata = json.loads(source_metadata)

        embedding = _row_get(row, "embedding")
        if isinstance(embedding, str):
            embedding = json.loads(embedding)

        validated_at = datetime.now(timezone.utc)
        if "created_at" in row and row["created_at"] is not None:
            validated_at = row["created_at"]

        return ValidatedFact(
            id=row["id"],
            candidate_fact_id=row["candidate_fact_id"] or row["id"],
            source_interaction_id=row["source_interaction_id"],
            subject_id=row["subject_id"],
            relation=_coerce_relation_type(row["relation"]),
            object_id=row["object_id"],
            value=row["value"],
            confidence=row["confidence"],
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            validated_at=validated_at,
            superseded_by=row["superseded_by"],
            attributes=metadata or {},
            source_text=_row_get(row, "source_text"),
            source_metadata=source_metadata or {},
            embedding=embedding,
        )

    # =========================================================================
    # Interaction Operations
    # =========================================================================

    async def add_interaction(self, interaction: Interaction) -> str:
        """
        Add an interaction to the log.

        Returns the interaction ID.
        """
        async with self.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO interactions 
                (id, tenant_id, app_id, user_id, agent_id, run_id, session_id,
                 space_type, actor, raw_text, timestamp, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """,
                interaction.id,
                interaction.tenant_id,
                interaction.app_id,
                interaction.user_id,
                interaction.agent_id,
                interaction.run_id,
                interaction.session_id,
                interaction.space_type,
                interaction.actor.value.lower(),  # DB enum uses lowercase
                interaction.raw_text,
                interaction.timestamp,
                json.dumps(interaction.metadata),
            )
        return interaction.id

    async def get_interaction(self, interaction_id: str) -> Interaction | None:
        """Get an interaction by ID."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM interactions WHERE id = $1",
                interaction_id,
            )
            return self._row_to_interaction(row) if row else None

    async def get_interactions(
        self,
        limit: int = 100,
        before: datetime | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> list[Interaction]:
        """Get recent interactions with optional filtering."""
        async with self.acquire() as conn:
            query = "SELECT * FROM interactions WHERE 1=1"
            params = []
            param_count = 0

            if before:
                param_count += 1
                query += f" AND timestamp < ${param_count}"
                params.append(before)

            if user_id:
                param_count += 1
                query += f" AND user_id = ${param_count}"
                params.append(user_id)

            if session_id:
                param_count += 1
                query += f" AND session_id = ${param_count}"
                params.append(session_id)

            param_count += 1
            query += f" ORDER BY timestamp DESC LIMIT ${param_count}"
            params.append(limit)

            rows = await conn.fetch(query, *params)
            return [self._row_to_interaction(row) for row in rows]

    async def get_interactions_by_session(self, session_id: str) -> list[Interaction]:
        """Get all interactions for a session."""
        async with self.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM interactions WHERE session_id = $1 ORDER BY timestamp ASC",
                session_id,
            )
            return [self._row_to_interaction(row) for row in rows]

    def _row_to_interaction(self, row) -> Interaction:
        """Convert a database row to an Interaction."""
        # ActorType uses lowercase values (user, agent, tool, system)
        actor_value = row["actor"] if row["actor"] else "user"

        return Interaction(
            id=row["id"],
            tenant_id=_row_get(row, "tenant_id"),
            app_id=_row_get(row, "app_id"),
            user_id=row["user_id"],
            agent_id=_row_get(row, "agent_id"),
            run_id=_row_get(row, "run_id"),
            session_id=row["session_id"],
            space_type=_row_get(row, "space_type"),
            actor=ActorType(actor_value),
            raw_text=row["raw_text"],
            timestamp=row["timestamp"],
            metadata=json.loads(row["metadata"])
            if isinstance(row["metadata"], str)
            else row["metadata"],
        )

    # =========================================================================
    # Candidate Fact Operations
    # =========================================================================

    async def add_candidate_fact(self, fact: CandidateFact) -> str:
        """
        Add a candidate fact to the store.

        Returns the fact ID.
        """
        async with self.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO candidate_facts 
                (id, source_interaction_id, subject_entity_id, relation, 
                 object_entity_id, value, confidence, extracted_at, status, 
                 rejection_reason, attributes)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                fact.id,
                fact.source_interaction_id,
                fact.subject_entity_id,
                fact.relation.value,
                fact.object_entity_id,
                fact.value,
                fact.confidence,
                fact.extracted_at,
                fact.status.value.lower(),  # DB enum uses lowercase
                fact.rejection_reason,
                json.dumps(fact.attributes),
            )
        return fact.id

    async def add_candidate_facts(self, facts: list[CandidateFact]) -> list[str]:
        """Add multiple candidate facts in a batch."""
        ids = []
        async with self.acquire() as conn:
            for fact in facts:
                await conn.execute(
                    """
                    INSERT INTO candidate_facts 
                    (id, source_interaction_id, subject_entity_id, relation, 
                     object_entity_id, value, confidence, extracted_at, status, 
                     rejection_reason, attributes)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """,
                    fact.id,
                    fact.source_interaction_id,
                    fact.subject_entity_id,
                    fact.relation.value,
                    fact.object_entity_id,
                    fact.value,
                    fact.confidence,
                    fact.extracted_at,
                    fact.status.value.lower(),  # DB enum uses lowercase
                    fact.rejection_reason,
                    json.dumps(fact.attributes),
                )
                ids.append(fact.id)
        return ids

    async def get_candidate_fact(self, fact_id: str) -> CandidateFact | None:
        """Get a candidate fact by ID."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM candidate_facts WHERE id = $1",
                fact_id,
            )
            return self._row_to_candidate_fact(row) if row else None

    async def get_pending_candidate_facts(
        self,
        limit: int = 100,
        interaction_id: str | None = None,
    ) -> list[CandidateFact]:
        """Get pending candidate facts awaiting validation."""
        async with self.acquire() as conn:
            if interaction_id:
                rows = await conn.fetch(
                    """
                    SELECT * FROM candidate_facts 
                    WHERE status = 'pending' AND source_interaction_id = $1
                    ORDER BY extracted_at ASC
                    LIMIT $2
                    """,
                    interaction_id,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM candidate_facts 
                    WHERE status = 'pending'
                    ORDER BY extracted_at ASC
                    LIMIT $1
                    """,
                    limit,
                )
            return [self._row_to_candidate_fact(row) for row in rows]

    async def get_candidate_facts_for_interaction(
        self,
        interaction_id: str,
    ) -> list[CandidateFact]:
        """Get all candidate facts from a specific interaction."""
        async with self.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM candidate_facts 
                WHERE source_interaction_id = $1
                ORDER BY extracted_at ASC
                """,
                interaction_id,
            )
            return [self._row_to_candidate_fact(row) for row in rows]

    async def update_candidate_fact_status(
        self,
        fact_id: str,
        status: CandidateFactStatus,
        rejection_reason: str | None = None,
    ) -> bool:
        """Update the status of a candidate fact."""
        async with self.acquire() as conn:
            if status == CandidateFactStatus.REJECTED and rejection_reason:
                result = await conn.execute(
                    """
                    UPDATE candidate_facts 
                    SET status = $1, rejection_reason = $2
                    WHERE id = $3
                    """,
                    status.value.lower(),  # DB enum uses lowercase
                    rejection_reason,
                    fact_id,
                )
            else:
                result = await conn.execute(
                    """
                    UPDATE candidate_facts 
                    SET status = $1
                    WHERE id = $2
                    """,
                    status.value.lower(),  # DB enum uses lowercase
                    fact_id,
                )
            return result != "UPDATE 0"

    async def accept_candidate_fact(self, fact_id: str) -> bool:
        """Accept a candidate fact."""
        return await self.update_candidate_fact_status(fact_id, CandidateFactStatus.ACCEPTED)

    async def reject_candidate_fact(
        self,
        fact_id: str,
        reason: str,
    ) -> bool:
        """Reject a candidate fact with a reason."""
        return await self.update_candidate_fact_status(
            fact_id, CandidateFactStatus.REJECTED, reason
        )

    def _row_to_candidate_fact(self, row) -> CandidateFact:
        """Convert a database row to a CandidateFact."""
        attributes = row["attributes"]
        if isinstance(attributes, str):
            attributes = json.loads(attributes)

        # CandidateFactStatus uses lowercase values
        status_value = row["status"] if row["status"] else "pending"

        return CandidateFact(
            id=row["id"],
            source_interaction_id=row["source_interaction_id"],
            subject_entity_id=row["subject_entity_id"],
            relation=_coerce_relation_type(row["relation"]),
            object_entity_id=row["object_entity_id"],
            value=row["value"],
            confidence=row["confidence"],
            extracted_at=row["extracted_at"],
            status=CandidateFactStatus(status_value),
            rejection_reason=row["rejection_reason"],
            attributes=attributes,
        )

    # =========================================================================
    # Rejection Operations
    # =========================================================================

    async def add_rejection(self, rejection: RejectionRecord) -> str:
        """Add a rejection record."""
        async with self.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO rejection_records 
                (id, candidate_fact_id, constraint_id, constraint_name, reason, 
                 domain_reasoning, alternatives, severity, rejected_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                rejection.id,
                rejection.candidate_fact_id,
                rejection.constraint_id,
                rejection.constraint_name,
                rejection.reason,
                rejection.domain_reasoning,
                rejection.alternatives,
                rejection.severity,
                rejection.rejected_at,
            )
        return rejection.id

    async def get_rejection(self, rejection_id: str) -> RejectionRecord | None:
        """Get a rejection record by ID."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM rejection_records WHERE id = $1",
                rejection_id,
            )
            return self._row_to_rejection(row) if row else None

    async def get_rejections_for_candidate(
        self,
        candidate_fact_id: str,
    ) -> list[RejectionRecord]:
        """Get all rejections for a candidate fact."""
        async with self.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM rejection_records WHERE candidate_fact_id = $1",
                candidate_fact_id,
            )
            return [self._row_to_rejection(row) for row in rows]

    async def get_all_rejections(self) -> list[RejectionRecord]:
        """Get all rejection records sorted newest first."""
        async with self.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM rejection_records ORDER BY rejected_at DESC")
            return [self._row_to_rejection(row) for row in rows]

    def _row_to_rejection(self, row) -> RejectionRecord:
        """Convert a database row to a RejectionRecord."""
        return RejectionRecord(
            id=row["id"],
            candidate_fact_id=row["candidate_fact_id"],
            constraint_id=row["constraint_id"],
            constraint_name=row["constraint_name"] or "",
            reason=row["reason"],
            domain_reasoning=row["domain_reasoning"],
            alternatives=list(row["alternatives"]) if row["alternatives"] else [],
            severity=row["severity"],
            rejected_at=row["rejected_at"],
        )

    async def get_rejections_for_constraint(
        self,
        constraint_id: str,
        limit: int = 100,
    ) -> list[RejectionRecord]:
        """Get rejections caused by a specific constraint."""
        async with self.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM rejection_records 
                WHERE constraint_id = $1 
                ORDER BY rejected_at DESC 
                LIMIT $2
                """,
                constraint_id,
                limit,
            )
            return [self._row_to_rejection(row) for row in rows]

    # =========================================================================
    # Validation Pipeline Operations
    # =========================================================================

    async def promote_candidate_to_validated(
        self,
        fact_id_or_fact: str | CandidateFact,
    ) -> ValidatedFact | None:
        """
        Promote a candidate fact to a validated fact.

        Args:
            fact_id_or_fact: Either a candidate fact ID (string) or CandidateFact object

        This:
        1. Creates a new ValidatedFact from the CandidateFact
        2. Updates the CandidateFact status to ACCEPTED
        3. Stores the ValidatedFact in the database

        Returns the new ValidatedFact, or None if candidate fact not found.
        """
        # Resolve to CandidateFact object
        if isinstance(fact_id_or_fact, str):
            candidate_fact = await self.get_candidate_fact(fact_id_or_fact)
            if candidate_fact is None:
                return None
        else:
            candidate_fact = fact_id_or_fact

        interaction = await self.get_interaction(candidate_fact.source_interaction_id)
        source_text = interaction.raw_text if interaction is not None else None
        source_metadata = {}
        if interaction is not None:
            source_metadata = {
                "actor": interaction.actor.value,
                "interaction_timestamp": interaction.timestamp.isoformat(),
                "session_id": interaction.session_id,
            }

        # Create the validated fact
        validated_fact = ValidatedFact(
            candidate_fact_id=candidate_fact.id,
            source_interaction_id=candidate_fact.source_interaction_id,
            subject_id=candidate_fact.subject_entity_id,
            relation=candidate_fact.relation,
            object_id=candidate_fact.object_entity_id,
            value=candidate_fact.value,
            valid_from=candidate_fact.extracted_at,
            confidence=candidate_fact.confidence,
            attributes=candidate_fact.attributes,
            source_text=source_text,
            source_metadata=source_metadata,
        )

        async with self.acquire() as conn, conn.transaction():
            # Add the validated fact
            await conn.execute(
                """
                    INSERT INTO validated_facts 
                    (id, candidate_fact_id, subject_id, relation, object_id, value,
                     confidence, source_interaction_id, valid_from, valid_to, 
                     superseded_by, metadata, source_text, source_metadata, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    """,
                validated_fact.id,
                validated_fact.candidate_fact_id,
                validated_fact.subject_id,
                validated_fact.relation.value,
                validated_fact.object_id,
                candidate_fact.value,
                validated_fact.confidence,
                validated_fact.source_interaction_id,
                validated_fact.valid_from,
                validated_fact.valid_to,
                validated_fact.superseded_by,
                json.dumps(validated_fact.attributes),
                validated_fact.source_text,
                json.dumps(validated_fact.source_metadata),
                json.dumps(validated_fact.embedding)
                if validated_fact.embedding is not None
                else None,
            )

            # Update candidate fact status
            await conn.execute(
                """
                    UPDATE candidate_facts 
                    SET status = 'accepted'
                    WHERE id = $1
                    """,
                candidate_fact.id,
            )

        return validated_fact

    async def reject_candidate_with_record(
        self,
        fact_id: str | None = None,
        reason: str | None = None,
        violated_constraints: list[str] | None = None,
        domain_reasoning: str | None = None,
        alternatives: list[str] | None = None,
        *,
        candidate_fact: CandidateFact | None = None,
        rejection_record: RejectionRecord | None = None,
    ) -> RejectionRecord | None:
        """
        Reject a candidate fact and create a rejection record.

        Can be called either with:
        1. fact_id, reason, and optional details
        2. candidate_fact and rejection_record objects

        Returns the created RejectionRecord, or None if fact not found.
        """
        # Handle object-based call
        if candidate_fact is not None and rejection_record is not None:
            cf = candidate_fact
            rr = rejection_record
        # Handle ID-based call
        elif fact_id is not None and reason is not None:
            cf = await self.get_candidate_fact(fact_id)
            if cf is None:
                return None

            constraint_name = violated_constraints[0] if violated_constraints else "unknown"
            constraint_id = constraint_name  # Use name as ID if not specified

            rr = RejectionRecord(
                candidate_fact_id=fact_id,
                constraint_id=constraint_id,
                constraint_name=constraint_name,
                reason=reason,
                domain_reasoning=domain_reasoning,
                alternatives=alternatives or [],
            )
        else:
            raise ValueError(
                "Must provide either (fact_id, reason) or (candidate_fact, rejection_record)"
            )

        async with self.acquire() as conn, conn.transaction():
            # Update candidate fact status
            await conn.execute(
                """
                    UPDATE candidate_facts 
                    SET status = 'rejected', rejection_reason = $1
                    WHERE id = $2
                    """,
                rr.reason,
                cf.id,
            )

            # Add rejection record
            await conn.execute(
                """
                    INSERT INTO rejection_records 
                    (id, candidate_fact_id, constraint_id, constraint_name, reason, 
                     domain_reasoning, alternatives, severity, rejected_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                rr.id,
                rr.candidate_fact_id,
                rr.constraint_id,
                rr.constraint_name,
                rr.reason,
                rr.domain_reasoning,
                rr.alternatives,
                rr.severity,
                rr.rejected_at,
            )

        return rr

    # =========================================================================
    # Graph Operations
    # =========================================================================

    async def get_connected_entities(
        self,
        entity_id: str,
        max_hops: int = 2,
        at_time: datetime | None = None,
    ) -> dict[str, Entity]:
        """Get all entities connected to a seed entity within N hops."""
        at_time = at_time or datetime.now(timezone.utc)

        async with self.acquire() as conn:
            # Use recursive CTE for graph traversal
            rows = await conn.fetch(
                """
                WITH RECURSIVE connected AS (
                    -- Base case: the seed entity
                    SELECT id, 0 as depth FROM entities WHERE id = $1
                    
                    UNION
                    
                    -- Recursive case: entities connected through facts
                    SELECT DISTINCT 
                        CASE 
                            WHEN f.subject_id = c.id THEN f.object_id 
                            ELSE f.subject_id 
                        END as id,
                        c.depth + 1 as depth
                    FROM connected c
                    JOIN validated_facts f ON (f.subject_id = c.id OR f.object_id = c.id)
                    WHERE c.depth < $2
                    AND f.valid_from <= $3
                    AND (f.valid_to IS NULL OR f.valid_to > $3)
                )
                SELECT e.* FROM entities e
                JOIN connected c ON e.id = c.id
                """,
                entity_id,
                max_hops,
                at_time,
            )
            return {row["id"]: self._row_to_entity(row) for row in rows}

    async def get_subgraph(
        self,
        entity_ids: list[str],
        at_time: datetime | None = None,
    ) -> tuple[dict[str, Entity], list[ValidatedFact]]:
        """Get a subgraph containing specified entities and facts between them."""
        at_time = at_time or datetime.now(timezone.utc)

        async with self.acquire() as conn:
            # Get entities
            entity_rows = await conn.fetch(
                "SELECT * FROM entities WHERE id = ANY($1)",
                entity_ids,
            )
            entities = {row["id"]: self._row_to_entity(row) for row in entity_rows}

            # Get facts between these entities
            fact_rows = await conn.fetch(
                """
                SELECT * FROM validated_facts
                WHERE subject_id = ANY($1) AND object_id = ANY($1)
                AND valid_from <= $2
                AND (valid_to IS NULL OR valid_to > $2)
                """,
                entity_ids,
                at_time,
            )
            facts = [self._row_to_fact(row) for row in fact_rows]

            return entities, facts

    # =========================================================================
    # Utilities
    # =========================================================================

    async def get_statistics(self) -> dict[str, Any]:
        """Get store statistics."""
        async with self.acquire() as conn:
            stats = {}

            # Entity counts
            entity_counts = await conn.fetch(
                """
                SELECT entity_type, COUNT(*) as count 
                FROM entities 
                GROUP BY entity_type
                """
            )
            stats["entities_by_type"] = {row["entity_type"]: row["count"] for row in entity_counts}
            stats["total_entities"] = sum(stats["entities_by_type"].values())

            # Fact counts
            fact_row = await conn.fetchrow(
                """
                SELECT 
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE valid_to IS NULL) as active
                FROM validated_facts
                """
            )
            stats["total_facts"] = fact_row["total"]
            stats["active_facts"] = fact_row["active"]
            stats["superseded_facts"] = fact_row["total"] - fact_row["active"]

            # Interaction and rejection counts
            stats["total_interactions"] = await conn.fetchval("SELECT COUNT(*) FROM interactions")
            stats["total_rejections"] = await conn.fetchval(
                "SELECT COUNT(*) FROM rejection_records"
            )

            return stats

    async def clear(self) -> None:
        """Clear all data from the store (use with caution!)."""
        async with self.acquire() as conn:
            await conn.execute("TRUNCATE rejection_records CASCADE")
            await conn.execute("TRUNCATE interactions CASCADE")
            await conn.execute("TRUNCATE validated_facts CASCADE")
            await conn.execute("TRUNCATE entities CASCADE")


# =============================================================================
# Factory Function
# =============================================================================


async def create_postgres_store(
    config: PostgresConfig | None = None,
    create_schema: bool = True,
) -> PostgresStore:
    """
    Factory function to create and initialize a PostgresStore.

    Usage:
        store = await create_postgres_store()
        # Use the store...
        await store.close()
    """
    store = PostgresStore(config)
    await store.initialize(create_schema=create_schema)
    return store
