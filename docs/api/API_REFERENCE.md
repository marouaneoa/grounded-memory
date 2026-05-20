# Grounded Memory — Technical API Reference

> **Version:** 0.1.0  
> **Package layout:** `gmem` (public facade) → `grounded_memory` (implementation runtime)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Core Data Model](#2-core-data-model)
3. [Storage Backends](#3-storage-backends)
4. [Grounding Operator](#4-grounding-operator)
5. [Constraint Validation](#5-constraint-validation)
6. [Conflict Resolution](#6-conflict-resolution)
7. [Tuple Normalization](#7-tuple-normalization)
8. [Entity Identity](#8-entity-identity)
9. [Intent Routing](#9-intent-routing)
10. [LLM Integration](#10-llm-integration)
11. [Graph Retrieval](#11-graph-retrieval)
12. [Adapters](#12-adapters)
13. [Public SDK (`Memory` Class)](#13-public-sdk-memory-class)
14. [FastAPI Service](#14-fastapi-service)
15. [Configuration](#15-configuration)
16. [Package Imports](#16-package-imports)

---

## 1. System Overview

The Grounded Memory System implements a **temporal property graph** where entities are nodes and validated facts are edges. Memory formation is treated as a **conditional state transition**: a candidate fact enters long-term memory only if it passes all registered governance constraints.

### Write Pipeline

```
text → Memory.add()
     → LLM extraction (extracts CandidateFact[])
     → GroundingOperator.ground()
         → ConstraintValidator.validate()  (all registered constraints)
         → ConflictResolver               (supersession decisions)
         → Duplicate detection
     → ValidatedFact (stored) | RejectionRecord (audit)
```

### Read Pipeline

```
query → Memory.search()
      → GraphRetriever
          → select_seed_entities()
          → multi-hop expansion (up to max_hops)
          → scoring + reranking
      → AnswerContext → list[SearchResult dicts]
```

### Package Architecture

```
from gmem import Memory              # Public facade — stable surface
from grounded_memory import ...      # Implementation — internal modules
```

The `gmem` package re-exports the primary symbols (`Memory`, `GroundedMemorySystem`, `LLMConfig`, `ConstraintValidator`, etc.) for end users. The `grounded_memory` package contains the full implementation across sub-modules (`core/`, `llm/`, `retrieval/`, `adapters/`, `service/`).

### Storage Backends

| Backend Key | Class | Description |
|---|---|---|
| `memory` | `MemoryStore` | Pure in-process dict; no persistence |
| `hybrid` | `HybridMemoryStore` | In-memory + Neo4j graph projection |
| `postgres` | `PostgresHybridMemoryStore` | PostgreSQL primary, no Neo4j sync |
| `postgres_hybrid` | `PostgresHybridMemoryStore` | PostgreSQL + Neo4j (default) |

---

## 2. Core Data Model

**Module:** `grounded_memory.core.models`

Six core objects form the memory taxonomy:

| # | Class | Role | Persisted? |
|---|---|---|---|
| 1 | `Interaction` | Immutable event log | Yes |
| 2 | `Entity` | Symbolic anchor (graph node) | Yes |
| 3 | `CandidateFact` | Untrusted LLM proposal | Yes |
| 4 | `ValidatedFact` | System-approved knowledge (graph edge) | Yes |
| 5 | `Constraint` | Declarative governance rule | Yes |
| 6 | `AnswerContext` | Ephemeral query view | No |

### 2.1 `Interaction`

Immutable audit trail of raw user/agent interactions. **Frozen** after creation.

```python
class Interaction(BaseModel, frozen=True):
    id: str              # UUID4
    tenant_id: str | None
    app_id: str | None
    user_id: str | None
    agent_id: str | None
    run_id: str | None
    session_id: str | None
    space_type: str | None
    actor: ActorType     # default: USER
    raw_text: str
    timestamp: datetime  # default: now(UTC)
    metadata: dict[str, Any]
```

Properties: `content` (alias for `raw_text`), `source` (alias for `actor.value`).

### 2.2 `Entity`

Symbolic anchor representing a real-world object. Nodes in the knowledge graph.

```python
class Entity(BaseModel):
    id: str              # UUID4
    entity_type: EntityType
    name: str
    canonical_id: str | None
    attributes: dict[str, Any]
    created_at: datetime
    updated_at: datetime
```

Implements `__hash__` and `__eq__` by `id`.

### 2.3 `CandidateFact`

Untrusted fact proposed by the LLM. A triplet `(subject, relation, object)` or `(subject, relation, value)`.

```python
class CandidateFact(BaseModel):
    id: str                       # UUID4
    source_interaction_id: str
    subject_entity_id: str
    relation: RelationType
    object_entity_id: str | None
    value: str | None
    confidence: float             # 0.0-1.0, default 0.9
    extracted_at: datetime
    status: CandidateFactStatus   # default: PENDING
    rejection_reason: str | None
    attributes: dict[str, Any]
```

Validators: confidence in [0.0, 1.0]; at least one of `object_entity_id` or `value` must be present.  
Legacy properties: `subject_id` → `subject_entity_id`, `object_id` → `object_entity_id`.

### 2.4 `ValidatedFact`

System-approved knowledge with **bitemporal boundaries**. Edges in the knowledge graph. Facts are **never deleted** — only superseded.

```python
class ValidatedFact(BaseModel):
    id: str
    candidate_fact_id: str
    source_interaction_id: str
    subject_id: str
    relation: RelationType
    object_id: str | None
    value: str | None
    valid_from: datetime          # Start of valid-time period
    valid_to: datetime | None     # None = still active
    validated_at: datetime
    validated_by: str             # default: "constraint_validator_v1"
    superseded_by: str | None     # ID of superseding fact
    confidence: float
    attributes: dict[str, Any]
    source_text: str | None       # Original sentence (provenance)
    embedding: list[float] | None # Vector embedding
    source_metadata: dict[str, Any]
```

Properties:
- `is_active` — `True` if `valid_to` is not in the past, `superseded_by` is None, and `valid_from` ≤ now
- `is_active_at(timestamp: datetime) -> bool` — checks if fact was active at a specific timestamp

### 2.5 `Constraint`

Declarative governance rule for write-time validation.

```python
class Constraint(BaseModel):
    id: str
    name: str
    description: str
    constraint_type: ConstraintType
    applies_to_relations: list[RelationType]
    condition: dict[str, Any]
    severity: str               # "error" | "warning" | "info"
    enabled: bool
    metadata: dict[str, Any]
```

### 2.6 `AnswerContext`

Ephemeral query view — never persisted. Output of graph-based retrieval.

```python
class AnswerContext(BaseModel):
    query: str
    timestamp: datetime
    seed_entities: list[str]
    facts: list[ValidatedFact]
    entities: dict[str, Entity]
    retrieval_metadata: dict[str, Any]
```

### 2.7 `RejectionRecord`

Audit trail for rejected candidate facts. Enables explainability.

```python
class RejectionRecord(BaseModel):
    id: str
    candidate_fact_id: str
    subject_entity_id: str | None
    rejected_at: datetime
    constraint_id: str
    constraint_name: str
    reason: str
    domain_reasoning: str | None    # serialized as "domain_reasoning"
    alternatives: list[str]
    severity: str
```

### 2.8 Enumerations

#### `ActorType` (str, Enum)
`USER` | `AGENT` | `TOOL` | `SYSTEM`

#### `CandidateFactStatus` (str, Enum)
`PENDING` | `ACCEPTED` | `REJECTED`

#### `RelationType` (str, Enum) — 27 values

**Legacy domain-oriented:**
`HAS_ALLERGY` | `HAS_CONDITION` | `PRESCRIBED` | `DISCONTINUED` | `TREATS` | `CONTAINS_INGREDIENT` | `SAME_THERAPEUTIC_CLASS` | `CONTRAINDICATED_WITH`

**General:** `HAS_ATTRIBUTE` | `RELATED_TO` | `PART_OF` | `INSTANCE_OF`

**Domain-agnostic (generic knowledge graph):**
`OWNS` | `WORKS_AT` | `LOCATED_IN` | `MEMBER_OF` | `CREATED` | `DEPENDS_ON` | `MANAGES` | `USED_BY` | `PRODUCED_BY` | `AFFILIATED_WITH` | `REPORTED_BY` | `APPROVED_BY`

#### `EntityType` (str, Enum) — 22 values

**Legacy domain-oriented:** `PATIENT` | `MEDICATION` | `CONDITION` | `ALLERGY` | `INGREDIENT` | `THERAPEUTIC_CLASS` | `CLINICIAN` | `FACILITY`

**Domain-agnostic:** `PERSON` | `PLACE` | `ORGANIZATION` | `CONCEPT` | `ASSET` | `SERVICE` | `EVENT` | `DOCUMENT` | `PROJECT` | `TOOL` | `METRIC` | `POLICY`

#### `ConstraintType` (str, Enum)
`PROHIBITION` — must NOT happen  
`REQUIREMENT` — must happen  
`CARDINALITY` — limits on count  
`TEMPORAL` — time-based rules  
`CONSISTENCY` — logical consistency

#### `MemoryDisposition` (str, Enum)
`CAPTURE` — introduce a new durable memory  
`REFINE` — replace or sharpen an existing memory  
`RETIRE` — deactivate an existing memory  
`PASS` — intentionally make no memory change

### 2.9 Utility Functions

| Function | Signature | Description |
|---|---|---|
| `as_utc_datetime` | `(value: datetime) -> datetime` | Returns timezone-aware UTC datetime |
| `datetime_before` | `(left, right) -> bool` | `left < right` under UTC normalization |
| `datetime_after` | `(left, right) -> bool` | `left > right` under UTC normalization |
| `datetime_on_or_before` | `(left, right) -> bool` | `left <= right` under UTC normalization |
| `datetime_on_or_after` | `(left, right) -> bool` | `left >= right` under UTC normalization |

---

## 3. Storage Backends

### 3.1 `MemoryStore`

**Module:** `grounded_memory.core.store`

In-memory implementation with bitemporal semantics. Implements the `KnowledgeState` protocol used by the `ConstraintValidator`.

```python
class MemoryStore:
    def __init__(self)
```

#### Entity Operations

| Method | Signature |
|---|---|
| `add_entity` | `(entity: Entity) -> None` |
| `get_entity` | `(entity_id: str) -> Entity \| None` |
| `get_entities_by_type` | `(entity_type: EntityType) -> list[Entity]` |
| `find_entity_by_name` | `(name: str, entity_type: EntityType \| None = None) -> Entity \| None` |
| `find_or_create_entity` | `(name, entity_type, create_fn, uniqueness_key=None) -> tuple[Entity, bool]` |
| `search_entities` | `(query: str, entity_type=None, limit=10) -> list[Entity]` |
| `iter_entities` | `() -> list[Entity]` |
| `find_entity_ids_by_name_fragment` | `(text: str) -> list[str]` |
| `get_all_entities` | `() -> list[Entity]` |

`find_or_create_entity` returns `(entity, created)` where `created` is `True` if a new entity was made. Supports scope-aware uniqueness via `uniqueness_key`.

#### Fact Operations

| Method | Signature |
|---|---|
| `add_validated_fact` | `(fact: ValidatedFact) -> None` |
| `get_fact` | `(fact_id: str) -> ValidatedFact \| None` |
| `get_validated_fact` | `(fact_id: str) -> ValidatedFact \| None` |
| `get_active_facts_for_entity` | `(entity_id: str, at_time: datetime \| None = None) -> list[ValidatedFact]` |
| `get_facts_by_relation` | `(entity_id, relation, as_subject=True, at_time=None) -> list[ValidatedFact]` |
| `get_all_facts_by_relation` | `(relation, at_time=None) -> list[ValidatedFact]` |
| `get_all_validated_facts` | `() -> list[ValidatedFact]` |
| `iter_active_facts` | `(at_time=None) -> list[ValidatedFact]` |
| `get_facts_for_entity` | `(entity_id, include_superseded=False) -> list[ValidatedFact]` |
| `supersede_fact` | `(fact_id, superseded_by, valid_to=None) -> bool` |

`supersede_fact` marks a fact as superseded (sets `valid_to` and `superseded_by`). Returns `True` if the fact existed. The fact is **never deleted**.

#### Interaction Operations

| Method | Signature |
|---|---|
| `add_interaction` | `(interaction: Interaction) -> None` |
| `get_interaction` | `(interaction_id: str) -> Interaction \| None` |
| `get_interactions` | `(limit=100, before: datetime \| None = None) -> list[Interaction]` |

#### Rejection Operations

| Method | Signature |
|---|---|
| `add_rejection` | `(rejection: RejectionRecord) -> None` |
| `get_rejection` | `(rejection_id: str) -> RejectionRecord \| None` |
| `get_all_rejections` | `() -> list[RejectionRecord]` |
| `get_rejections_for_constraint` | `(constraint_id: str, limit=100) -> list[RejectionRecord]` |

#### Graph Operations

| Method | Signature |
|---|---|
| `get_connected_entities` | `(entity_id, max_hops=2, at_time=None) -> dict[str, Entity]` |
| `get_subgraph` | `(entity_ids: list[str], at_time=None) -> tuple[dict[str, Entity], list[ValidatedFact]]` |

#### Utilities

| Method | Signature |
|---|---|
| `get_statistics` | `() -> dict[str, Any]` |
| `clear` | `() -> None` |

`get_statistics()` returns: `total_entities`, `entities_by_type`, `total_facts`, `active_facts`, `superseded_facts`, `total_interactions`, `total_rejections`.

### 3.2 `HybridMemoryStore`

**Module:** `grounded_memory.core.hybrid_store`

Bridges `MemoryStore` (source of truth) with Neo4j (active graph projection). All mutations go through `MemoryStore` first, then synchronously replicate to Neo4j. Supports `rebuild_neo4j()` for recovery from the primary store.

### 3.3 `PostgresStore`

**Module:** `grounded_memory.core.postgres_store`

Full PostgreSQL persistence with asyncpg-based pool connections. Implements full SQL schema with bitemporal tables (`entities`, `facts`, `interactions`, `rejections`, `candidates`), JSONB attributes, and text search indexes.

### 3.4 `Neo4jStore`

**Module:** `grounded_memory.core.neo4j_store`

Cypher-based graph store for active knowledge. Provides `upsert_entity`, `add_fact`, `remove_fact`, `get_neighbors` (multi-hop graph traversal), `find_paths`, `get_safety_critical_facts`, `sync_all_entities`, `sync_all_active_facts`.

### 3.5 `PostgresHybridMemoryStore`

**Module:** `grounded_memory.core.postgres_hybrid_store`

Inherits `HybridMemoryStore` and adds PostgreSQL persistence. Uses a background thread with asyncio event loop to bridge synchronous SDK calls with async PostgreSQL operations. Rehydrates in-memory state from PostgreSQL on startup.

---

## 4. Grounding Operator

**Module:** `grounded_memory.core.grounding`

The Grounding Operator (Γ) is the execution engine for memory formation:

```
Γ(f̂, K) = 1(Valid) if ∀c ∈ C : c(f̂, K) = True
          0(Rejected) otherwise
```

### 4.1 `GroundingDecision` (str, Enum)

`APPROVED` — fact is valid, stored in memory  
`REJECTED` — fact violates constraints  
`SUPERSEDED` — fact approved but supersedes existing fact(s)  
`DUPLICATE` — fact already exists in memory

### 4.2 `GroundingResult` (dataclass)

```python
@dataclass
class GroundingResult:
    decision: GroundingDecision
    candidate_fact: CandidateFact
    validated_fact: ValidatedFact | None = None
    rejection_record: RejectionRecord | None = None
    superseded_facts: list[ValidatedFact] = field(default_factory=list)
    validation_result: ValidationResult | None = None
    conflict_resolutions: list[dict] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
```

Properties:
- `is_success` — `True` for `APPROVED` or `SUPERSEDED`

Methods:
- `get_explanation() -> str` — human-readable summary including superseded IDs, rejection reasons, domain reasoning, and alternatives.

### 4.3 `GroundingOperator` (sync)

```python
class GroundingOperator:
    def __init__(
        self,
        validator: ConstraintValidator,
        memory_store: MemoryStore,
        auto_supersede: bool = True,
        conflict_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.COMPOSITE,
        conflict_resolver: ConflictResolver | None = None,
    )
```

#### Methods

| Method | Signature | Description |
|---|---|---|
| `ground` | `(candidate: CandidateFact) -> GroundingResult` | Main entry point. 6-step pipeline: (1) duplicate check, (2) constraint validation, (3) rejection handling with RejectionRecord storage, (4) supersession via `_find_and_supersede`, (5) `_create_validated_fact` creation, (6) persistence. |
| `ground_batch` | `(candidates: list[CandidateFact]) -> list[GroundingResult]` | Sequential batch processing; each fact is validated against state that includes previously approved facts. |

#### 6-Step Grounding Pipeline

1. **Duplicate check** — same subject + relation + object_id/value/attributes → `DUPLICATE`
2. **Constraint validation** — `validator.validate(candidate, knowledge_state)`
3. **Rejection handling** — stores `RejectionRecord`, sets candidate status to `REJECTED`
4. **Supersession** — finds active facts with same subject/relation; uses `ConflictResolver` to decide winner; calls `memory_store.supersede_fact` on losing facts
5. **Fact creation** — builds `ValidatedFact` with `valid_from=candidate.extracted_at`, carries forward source text and metadata from the originating `Interaction`
6. **Persistence** — `memory_store.add_validated_fact(validated_fact)`

#### Private Methods

| Method | Description |
|---|---|
| `_is_duplicate(candidate) -> bool` | Compares active facts by `object_id`, normalized value, and attributes |
| `_find_and_supersede(candidate) -> tuple[list[ValidatedFact], list[dict]]` | Finds conflicting active facts, resolves via `ConflictResolver`, supersedes losers |
| `_should_supersede(existing, candidate) -> bool` | Two-part check: (1) exact tuple match (object_id + value), (2) semantic-key match via `build_fact_semantic_key` |
| `_fact_semantic_key(relation, object_id, value, attributes) -> str \| None` | Delegates to `build_fact_semantic_key` with `include_subject=False` |
| `_create_validated_fact(candidate) -> ValidatedFact` | Builds `ValidatedFact` with provenance from the source `Interaction` |

### 4.4 `AsyncGroundingOperator`

Async version for database-backed stores (`PostgresKnowledgeStore`). Mirror of the sync operator:

```python
class AsyncGroundingOperator:
    def __init__(
        self,
        validator: ConstraintValidator,
        store: PostgresKnowledgeStore,
        auto_supersede: bool = True,
    )

    async def ground(self, candidate: CandidateFact) -> GroundingResult
    async def ground_batch(self, candidates: list[CandidateFact]) -> list[GroundingResult]
```

Uses `await self.store.reject_candidate_with_record(...)`, `await self.store.promote_candidate_to_validated(...)`, and `await self.store.supersede_fact(...)`.

### 4.5 Batch Processing Helpers

```python
def process_interaction_facts(
    operator: GroundingOperator,
    candidates: list[CandidateFact],
    interaction_id: str,
) -> tuple[list[ValidatedFact], list[RejectionRecord]]

async def process_interaction_facts_async(
    operator: AsyncGroundingOperator,
    candidates: list[CandidateFact],
    interaction_id: str,
) -> tuple[list[ValidatedFact], list[RejectionRecord]]
```

---

## 5. Constraint Validation

**Module:** `grounded_memory.core.constraints`

### 5.1 Core Types

#### `ConstraintViolation` (dataclass)

```python
@dataclass
class ConstraintViolation:
    constraint_id: str
    constraint_name: str
    description: str
    severity: str                  # "error" | "warning" | "info"
    domain_reasoning: str | None = None
    alternatives: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
```

#### `ValidationResult` (dataclass)

```python
@dataclass
class ValidationResult:
    is_valid: bool
    candidate_fact_id: str
    violations: list[ConstraintViolation]
    warnings: list[ConstraintViolation]
    checked_constraints: list[str]
    validation_timestamp: datetime
```

Properties: `has_errors`, `has_warnings`

Methods:
- `get_primary_rejection_reason() -> str | None`
- `to_rejection_record() -> RejectionRecord | None`

#### `KnowledgeState` (Protocol)

```python
class KnowledgeState(Protocol):
    def get_entity(self, entity_id: str) -> Entity | None: ...
```

### 5.2 Constraint Lifecycle

#### `ConstraintLifecycleStatus` (str, Enum)
`PROPOSED` — submitted for consideration  
`SHADOW` — observing traffic without blocking  
`ACTIVE` — enforcing on writes  
`DEPRECATED` — no longer evaluated

#### `ConstraintSource` (str, Enum)
`HUMAN` | `AGENT`

#### `ManagedConstraint` (dataclass)

Wraps an evaluator with lifecycle metadata, priority, scope, shadow observation counters (`shadow_hits`, `shadow_violations`), and form metadata.

```python
@dataclass
class ManagedConstraint:
    evaluator: BaseConstraintEvaluator
    source: ConstraintSource
    lifecycle: ConstraintLifecycleStatus
    priority: int
    scope: DynamicConstraintScope
    form_id: str | None
    form_metadata: dict[str, Any]
    shadow_hits: int
    shadow_violations: int
    last_updated: datetime

    # Properties: constraint_id
    # Methods: mark_shadow_observation(violated: bool)
```

#### `ConstraintFormTemplate` (dataclass, frozen)

Canonical form for a constraint family. Built-in templates:
- `safety_control` — safety rules for high-risk relation checks
- `temporal_consistency` — temporal ordering and validity-window constraints
- `cardinality_control` — limits for duplicate, overlap, and saturation conditions

#### `ConstraintReplayMetrics` (dataclass)

Offline replay metrics for dynamic constraints. Properties: `trigger_rate`, `projected_false_block_rate`, `projected_miss_coverage`.

### 5.3 `BaseConstraintEvaluator` (ABC)

Abstract base class for all constraint evaluators.

```python
class BaseConstraintEvaluator(ABC):
    @property
    @abstractmethod
    def constraint_id(self) -> str: ...
    @property
    @abstractmethod
    def constraint_name(self) -> str: ...
    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    def applies_to_relations(self) -> list[RelationType]: return []
    @property
    def severity(self) -> str: return "error"

    @abstractmethod
    def evaluate(self, candidate: CandidateFact, knowledge_state: KnowledgeState) -> ConstraintViolation | None: ...

    def applies_to(self, candidate: CandidateFact) -> bool
```

#### Concrete Base Classes

| Class | Abstract Method | Description |
|---|---|---|
| `ProhibitionConstraint` | `is_prohibited(candidate, state) -> tuple[bool, str\|None, list[str]]` | Prohibit certain fact patterns |
| `CardinalityConstraint` | `max_count -> int` / `count_existing(candidate, state) -> int` | Limit count of facts |
| `TemporalConstraint` | `check_temporal_validity(candidate, state) -> tuple[bool, str\|None]` | Temporal relationship rules |
| `DeclarativeConstraint` | `condition_fn(candidate, state)` | Built from YAML specification |

### 5.4 `ConstraintValidator`

Main validation engine. Orchestrates evaluation of all registered constraints.

```python
class ConstraintValidator:
    def __init__(self)
```

#### Registration Methods

| Method | Description |
|---|---|
| `register(evaluator, *, source, lifecycle, priority, scope, form_id, form_metadata)` | Register a constraint with full governance metadata |
| `register_with_form(evaluator, *, form_id, form_metadata, ...)` | Register with required form-template validation |
| `register_dynamic(evaluator, *, lifecycle, priority, scope, form_id, form_metadata)` | Register an agent-proposed adaptive constraint (default: `PROPOSED`) |
| `unregister(constraint_id) -> bool` | Remove a constraint by ID |
| `register_form_template(template)` | Register a constraint form template |

#### Validation Methods

| Method | Signature | Description |
|---|---|---|
| `validate` | `(candidate, knowledge_state, stop_on_first_error=False, runtime_context=None, max_dynamic_constraints=20, max_shadow_constraints=10) -> ValidationResult` | Validate a single candidate against all applicable constraints |
| `validate_batch` | `(candidates, knowledge_state, ...) -> dict[str, ValidationResult]` | Validate multiple candidates |

**Constraint selection rules:**
- Human-authored `ACTIVE` constraints are always included
- Agent-authored `ACTIVE` constraints are context-filtered and capped at `max_dynamic_constraints`
- `PROPOSED`/`SHADOW` constraints are observed but never block writes (they produce warnings with `[shadow]` prefix)

#### Lifecycle Management Methods

| Method | Description |
|---|---|
| `set_lifecycle(constraint_id, lifecycle) -> bool` | Update lifecycle stage |
| `set_priority(constraint_id, priority) -> bool` | Update execution priority |
| `get_managed_constraint(constraint_id) -> ManagedConstraint \| None` | Get managed constraint metadata |
| `list_managed_constraints() -> list[ManagedConstraint]` | List all managed constraints |
| `get_evaluator(constraint_id) -> BaseConstraintEvaluator \| None` | Get evaluator by ID |
| `registered_constraints` (property) `-> list[str]` | List of registered constraint IDs |
| `get_form_template(form_id) -> ConstraintFormTemplate \| None` | Get form template by ID |
| `list_form_templates() -> list[ConstraintFormTemplate]` | List all form templates |

#### Signal Recording and Replay

| Method | Description |
|---|---|
| `record_validation_signal(candidate, result, runtime_context)` | Record write-time governance signal |
| `list_validation_signals(limit=500) -> list[dict]` | Get recent governance signals |
| `replay_dynamic_constraints(candidates, knowledge_state) -> dict[str, ConstraintReplayMetrics]` | Offline replay for dynamic constraints |
| `promote_dynamic_constraints(replay_metrics, *, min_trigger_rate=0.01, max_projected_false_block_rate=0.02, min_candidates=100) -> list[str]` | Promote seeds to ACTIVE based on replay evidence |

### 5.5 Constraint Lifecycle in Detail

```
proposed → shadow → active → deprecated
```

- **proposed** — agent-synthesized seed, not yet observed
- **shadow** — observing traffic; violations recorded but never block writes (produce warnings)
- **active** — full enforcement; violations cause rejection
- **deprecated** — no longer considered

Shadow mode provides a safe rollout mechanism: constraints can observe real traffic patterns before blocking any writes.

---

## 6. Conflict Resolution

**Module:** `grounded_memory.core.conflict_resolution`

When new information contradicts existing knowledge, the `ConflictResolver` determines which fact wins.

### 6.1 `ConflictResolutionStrategy` (str, Enum)

| Strategy | Description |
|---|---|
| `CONFIDENCE_WINS` | Higher confidence score wins (with configurable minimum delta) |
| `RECENCY_WINS` | More recent fact supersedes older |
| `SOURCE_PRIORITY` | Ranked: `system(100) > tool(75) > agent(50) > user(25)` |
| `COMPOSITE` | Weighted combination of all signals (default) |

### 6.2 `ConflictSignal` (dataclass, frozen)

Normalized quality signals for a fact:

```python
@dataclass(frozen=True)
class ConflictSignal:
    confidence: float
    timestamp: datetime
    source_rank: int
    embedding_similarity: float = 0.0

    @classmethod
    def from_validated_fact(cls, fact: ValidatedFact) -> "ConflictSignal"
    @classmethod
    def from_candidate_fact(cls, candidate: CandidateFact) -> "ConflictSignal"
```

### 6.3 `ConflictResolution` (dataclass)

```python
@dataclass
class ConflictResolution:
    should_supersede: bool
    strategy_used: ConflictResolutionStrategy
    winning_signal: ConflictSignal
    losing_signal: ConflictSignal
    reasoning: str
    scores: dict[str, float]
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]
```

### 6.4 `ConflictResolver`

```python
class ConflictResolver:
    DEFAULT_WEIGHTS = {"confidence": 0.40, "recency": 0.30, "source": 0.30}

    def __init__(
        self,
        strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.COMPOSITE,
        *,
        weights: dict[str, float] | None = None,
        confidence_threshold: float = 0.05,
    )

    def resolve(
        self,
        existing: ValidatedFact,
        candidate: CandidateFact,
    ) -> ConflictResolution
```

**COMPOSITE strategy** computes a normalized score from:
- **Confidence signal** (0.40 weight) — ratio `candidate.confidence / max(existing.confidence, 0.01)`, capped at 2.0, normalized to [0, 1]
- **Recency signal** (0.30 weight) — time decay: `1.0 / (1.0 + max(-delta_seconds, 0) / 3600.0)`
- **Source signal** (0.30 weight) — normalized rank ratio: `candidate.source_rank / max(candidate.source_rank, existing.source_rank, 1)`

If `composite > 0.5`, the candidate supersedes the existing fact.

---

## 7. Tuple Normalization

**Module:** `grounded_memory.core.tuple_normalization`

Utilities for canonical tuple normalization and semantic-key derivation. Used by the grounding operator for duplicate detection, supersession, and retire semantics.

### Constants

`ATTRIBUTE_KEY_ALIASES` — canonicalization map for attribute keys:
```python
{
    "preference": "prefers", "preferences": "prefers", "preferred": "prefers",
    "prefer": "prefers", "like": "likes",
    "loc": "location", "lives_in": "location", "located_in": "location",
    "work_on": "works_on", "working_on": "works_on",
}
```

### Functions

| Function | Signature | Description |
|---|---|---|
| `sanitize_fact_value` | `(value: str \| None) -> str \| None` | Trims whitespace, collapses multi-spaces, strips trailing punctuation. Returns None for empty results. Preserves casing. |
| `normalize_attribute_key` | `(key: str \| None) -> str \| None` | Canonicalizes attribute keys: lowercases, replaces non-alphanumeric with underscores, collapses multi-underscores, strips, maps through `ATTRIBUTE_KEY_ALIASES`. |
| `parse_keyed_value` | `(value: str \| None) -> tuple[str \| None, str \| None]` | Parses `key=value` or `key: value` strings. Returns `(normalized_key, parsed_value)` or `(None, sanitized_value)`. |
| `resolve_attribute_key` | `(value: str \| None, attributes: dict \| None = None) -> str \| None` | Resolves a tuple slot key: first checks `attributes["key"]`, then falls back to `parse_keyed_value(value)`. |
| `normalize_fact_attributes` | `(value: str \| None, attributes: dict \| None) -> dict[str, Any]` | Returns normalized attribute dict with canonical `"key"` entry if one can be resolved. |
| `normalize_fact_value_for_match` | `(value: str \| None) -> str \| None` | Canonical value for dedup/retire matching. For keyed values: `"key=parsed_value_lower"`; otherwise: `sanitized.lower()`. |
| `fact_values_equal` | `(left: str \| None, right: str \| None) -> bool` | Compares two fact values under canonical normalization. |
| `should_materialize_attribute_object` | `(value: str \| None) -> bool` | Heuristic: determines if a value is specific enough to become an entity node. Rejects length > 80, > 8 words, generic prefixes, control/JSON chars. |
| `build_fact_semantic_key` | `(*, subject_id, relation, object_id, value, attributes, include_subject=True) -> str \| None` | Builds semantic slot key as `"subject\|relation\|..."`. Third segment priority: (1) `k:<slot_key>`, (2) `o:<object_id>`, (3) `v:<normalized_value>`. |

---

## 8. Entity Identity

**Module:** `grounded_memory.core.entity_identity`

Deterministic entity identity helpers for cross-run idempotency.

### Functions

| Function | Signature | Description |
|---|---|---|
| `build_entity_uniqueness_key` | `(*, name, entity_type, attributes=None, canonical_id=None, uniqueness_key=None) -> str` | Builds a stable semantic key: `"scope:ID\|type:EntityType\|name:normalizedName"`. Priority: `uniqueness_key` > `canonical_id` > `name`. |
| `stable_entity_id` | `(uniqueness_key: str) -> str` | Derives a deterministic UUID5 from a semantic uniqueness key: `uuid5(NAMESPACE_URL, f"gmem:entity:{key}")`. |

---

## 9. Intent Routing

**Module:** `grounded_memory.core.intent`

Domain-agnostic intent routing layer that bridges natural language to memory operations.

### 9.1 `IntentAction` (str, Enum)

`REMEMBER` — write/store new information  
`RECALL` — read/lookup existing information  
`FIND_RELATED` — cross-entity lookup  
`EXPLAIN` — synthesize grounded answer  
`UNKNOWN` — unable to classify

### 9.2 `UserIntent` (BaseModel)

```python
class UserIntent(BaseModel):
    action: IntentAction = IntentAction.UNKNOWN
    confidence: float = 1.0            # 0.0-1.0
    mentions: list[str] = []           # Entity names in query
    temporal_anchor: str | None = None # ISO timestamp or relative expression
    explanation: str = ""              # Rationale for classification
```

Methods: `is_write() -> bool`, `is_read() -> bool`.

### 9.3 `KeywordIntentRouter`

Fast deterministic router using regex patterns. Adapters can register domain-specific patterns via `register_patterns()`.

```python
class KeywordIntentRouter(BaseIntentRouter):
    def __init__(
        self,
        *,
        remember_patterns: list[str] | None = None,
        explain_patterns: list[str] | None = None,
        find_related_patterns: list[str] | None = None,
        recall_patterns: list[str] | None = None,
    )

    def register_patterns(self, category: str, patterns: list[str], *, prepend: bool = False)
    def route(self, query: str) -> UserIntent
```

**Priority:** `EXPLAIN` > `FIND_RELATED` > `REMEMBER/RECALL` (heuristic based on statement vs. question patterns).  
If the text looks like a statement (no `?`, no interrogative words at start) and contains assertive verbs → `REMEMBER`; otherwise → `RECALL`.

### 9.4 `LLMIntentRouter`

Uses the LLM with `INTENT_ROUTING_SYSTEM_PROMPT` (domain-agnostic cognitive language only).

```python
class LLMIntentRouter(BaseIntentRouter):
    def __init__(self, llm_client=None)
    def route(self, query: str) -> UserIntent
```

### 9.5 `HybridIntentRouter`

Fast keyword path + LLM fallback when confidence below threshold. **Recommended for production.**

```python
class HybridIntentRouter(BaseIntentRouter):
    def __init__(
        self,
        keyword_router: KeywordIntentRouter | None = None,
        llm_router: LLMIntentRouter | None = None,
        confidence_threshold: float = 0.75,
    )

    def route(self, query: str) -> UserIntent
```

---

## 10. LLM Integration

**Module:** `grounded_memory.llm`

### 10.1 Configuration

#### `LLMProvider`
```python
class LLMProvider:
    LOCAL = "local"
    OPENROUTER = "openrouter"
```

#### `LLMConfig` (dataclass)

```python
@dataclass
class LLMConfig:
    provider: str = "openrouter"
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "z-ai/glm-4.5-air:free"
    api_key: str = ""
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: float = 120.0
    max_retries: int = 3
    retry_delay: float = 1.0
    site_url: str = ""
    site_name: str = "GroundedMemory"

    @classmethod
    def from_env(cls) -> "LLMConfig"
    def validate(self) -> None
    @classmethod
    def openrouter(cls, api_key: str, model: str = "...") -> "LLMConfig"
    @classmethod
    def local(cls, base_url: str, model: str, api_key: str = "") -> "LLMConfig"
```

`from_env()` reads: `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_MODEL`, `OPENROUTER_API_KEY`, `LLM_API_KEY`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_TIMEOUT`, `LLM_MAX_RETRIES`, `LLM_RETRY_DELAY`, `OPENROUTER_SITE_URL`, `OPENROUTER_SITE_NAME`.

### 10.2 `LLMClient` (async)

```python
class LLMClient:
    def __init__(self, config: LLMConfig | None = None)

    async def close(self) -> None
    async def complete(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str
    async def extract(
        self, text: str, output_model: Type[T],
        system_prompt: str | None = None,
        extraction_prompt: str | None = None,
    ) -> T
```

Uses `httpx.AsyncClient`. Retry logic with exponential backoff for 429/5xx errors. OpenRouter-specific headers (`HTTP-Referer`, `X-Title`).

### 10.3 `SyncLLMClient`

Synchronous wrapper for non-async code. Uses `httpx.Client` in a context manager.

```python
class SyncLLMClient:
    def __init__(self, config: LLMConfig | None = None)

    def complete(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str
    def extract(
        self, text: str, output_model: Type[T],
        system_prompt: str | None = None,
    ) -> T
```

### 10.4 `LLMFactExtractor` (dataclass)

Generic LLM-powered fact extractor for domain text.

```python
@dataclass
class LLMFactExtractor:
    config: LLMConfig
    client: SyncLLMClient

    def extract(
        self, text: str, output_model: Type[T],
        system_prompt: str,
        include_context: str | None = None,
    ) -> T

    def test_connection(self) -> bool
```

### 10.5 Prompt Definitions

**Module:** `grounded_memory.llm.prompts`

#### System Prompt Constants

| Constant | Purpose |
|---|---|
| `STRUCTURED_EXTRACTION_SYSTEM_PROMPT` | Deterministic JSON extraction; no hallucination, schema-first |
| `CLINICAL_EXTRACTION_SYSTEM_PROMPT` | Clinical text extraction; preserve verbatim drug names, no inference |
| `GENERIC_TUPLE_EXTRACTION_SYSTEM_PROMPT` | Tuple proposal engine; `HAS_ATTRIBUTE` vs `RELATED_TO`, disposition policy |
| `ENTITY_EXTRACTION_SYSTEM_PROMPT` | Entity extraction; only explicit entities, no pronouns |
| `EDGE_EXTRACTION_SYSTEM_PROMPT` | Relationship extraction; known entity endpoints only, temporal rules |
| `TEMPORAL_GROUNDING_SYSTEM_PROMPT` | Temporal normalization; convert relative to absolute ranges |
| `INTENT_ROUTING_SYSTEM_PROMPT` | Domain-agnostic intent classification (REMEMBER/RECALL/FIND/EXPLAIN) |
| `CONNECTIVITY_TEST_SYSTEM_PROMPT` | Minimal test prompt |

#### Prompt Builder Functions

| Function | Signature |
|---|---|
| `build_structured_extraction_user_prompt` | `(*, input_text, output_schema) -> str` |
| `build_clinical_extraction_user_prompt` | `(*, input_text, context_text=None) -> str` |
| `build_generic_tuple_extraction_user_prompt` | `(*, input_text, source_actor, user_identifier) -> str` |
| `build_entity_extraction_user_prompt` | `(*, input_text, output_schema, entity_types=None, previous_context_text=None, custom_instructions=None) -> str` |
| `build_edge_extraction_user_prompt` | `(*, input_text, output_schema, known_entities, reference_time_iso, allowed_relation_types=None, previous_context_text=None, current_date_iso=None, custom_instructions=None) -> str` |
| `build_temporal_grounding_user_prompt` | `(*, input_text, output_schema, reference_time_iso, current_date_iso=None, custom_instructions=None) -> str` |
| `build_chat_with_memory_system_prompt` | `(*, memory_block) -> str` |

---

## 11. Graph Retrieval

**Module:** `grounded_memory.retrieval.graph`

### 11.1 `RetrievalStrategy` (str, Enum)

`BREADTH_FIRST` — standard BFS expansion from seed entities  
`WEIGHTED` — weight-based expansion with relationship weights  
`SAFETY_PRIORITY` — prioritize safety-critical facts first

### 11.2 `RelationshipPreset` (str, Enum)

`GENERIC` — domain-agnostic default weights  
`SAFETY` — pre-configured safety-critical weights (HAS_ALLERGY=10.0, CONTRAINDICATED_WITH=10.0, PRESCRIBED=5.0, etc.)

### 11.3 `RelationshipWeight` (dataclass)

```python
@dataclass
class RelationshipWeight:
    relation: RelationType
    weight: float = 1.0
    is_safety_critical: bool = False
    decay_per_hop: float = 0.2  # Decay in [0, 1] for next hop carry
```

### 11.4 `QueryProfile` (dataclass, frozen)

Simple intent profile to rebalance retrieval signals:
```python
@dataclass(frozen=True)
class QueryProfile:
    prefers_recency: bool = False
    prefers_profile_facts: bool = False
    prefers_safety: bool = False
    prefers_relational_context: bool = False
```

### 11.5 `GraphRetriever`

```python
class GraphRetriever:
    def __init__(
        self,
        memory_store,
        *,
        max_hops: int = 2,
        max_facts: int = 30,
        strategy: RetrievalStrategy = RetrievalStrategy.WEIGHTED,
        relationship_preset: RelationshipPreset = RelationshipPreset.GENERIC,
        relationship_weights: dict[RelationType, RelationshipWeight] | None = None,
        lookback_days: int | None = None,
        score_threshold: float = 0.0,
        neo4j_store=None,
        diversity_penalty: float = 0.0,
        rank_by_entity: bool = False,
        bidirectional: bool = False,
    )
```

#### Methods

| Method | Signature | Description |
|---|---|---|
| `retrieve` | `(query, *, seed_entities=None, at_time=None, max_seeds=None, max_hops=None, max_facts=None, strategy=None, scope=None, rerank_debug=False) -> AnswerContext` | Main retrieval method: seed identification → multi-hop expansion → scoring → reranking |
| `set_relationship_weights` | `(weights: dict[RelationType, RelationshipWeight])` | Update per-relation weights at runtime |
| `add_relationship_weight` | `(weight: RelationshipWeight)` | Add/override a single relationship weight |
| `apply_preset` | `(preset: RelationshipPreset)` | Apply a named weight preset |
| `find_paths` | `(source_id, target_id, *, max_hops=3, at_time=None, max_paths=5) -> list[dict]` | Find paths between two entities |
| `get_entity_neighborhood` | `(entity_id, *, max_hops=1, at_time=None, strategy=None) -> dict` | Get neighborhood of an entity |

#### Key Features

- **Neo4j-backed** for current-time queries (native graph traversal)
- **NetworkX/InMemory fallback** for temporal/point-in-time queries
- **Multi-hop expansion** with configurable weight decay per hop
- **Reranking** with relevance-gated scoring and content-aware diversity (via semantic key grouping)
- **Temporal context filtering** via `lookback_days`
- **Scope-based filtering** (tenant/app/user) using `source_metadata` from facts

### 11.6 Seed Entity Selection

```python
def select_seed_entities(
    query: str,
    store,
    *,
    max_seeds: int = 6,
    scope=None,
    entity_type: EntityType | None = None,
) -> list[str]
```

Selects seed entities by:
1. Matching entity name fragments contained in the query text
2. Scored fuzzy name matching for remaining capacity
3. Filtering by `entity_type` if specified
4. Scope filtering (tenant/app/user) if scope is provided

---

## 12. Adapters

**Module:** `grounded_memory.adapters`

Adapters provide domain-specific constraint configuration and agent implementation. They are registered in a central registry and activated via `GM_ADAPTER` env var or the `Memory(adapter=...)` constructor parameter.

### 12.1 Registry

**Module:** `grounded_memory.adapters.registry`

```python
@dataclass(frozen=True)
class AdapterSpec:
    key: str
    configure_validator: ValidatorConfigurator
    create_agent: AgentCreator
    # Property: profile (alias for key)
```

#### Built-in Adapters

| Key | Validator | Agent |
|---|---|---|
| `generic` | generic (YAML constraints) | `GenericMemoryAgent` |
| `core` | generic | `GenericMemoryAgent` |
| `none` | generic | `GenericMemoryAgent` |
| `engineering` | generic | `GenericMemoryAgent` |
| `finance` | generic | `GenericMemoryAgent` |
| `legal` | generic | `GenericMemoryAgent` |
| `healthcare` | healthcare-specific | `HealthcareMemoryAgent` |

#### Registry Functions

```python
def register_adapter(*, key, configure_validator, create_agent, overwrite=False) -> None
def unregister_adapter(key: str) -> bool
def list_registered_adapters() -> list[str]
def get_adapter_spec_by_key(key: str) -> AdapterSpec
# Backward-compatible aliases:
def register_adapter_spec(*, profile, configure_validator, create_agent, overwrite=False) -> None
def unregister_adapter_spec(profile: str) -> bool
def list_supported_profiles() -> list[str]
def get_adapter_spec(profile: str) -> AdapterSpec
```

### 12.2 Generic Agent

**Module:** `grounded_memory.adapters.generic_agent`

LLM-backed agent for open-domain memory writes.

#### `GenericMemoryAgent`

```python
class GenericMemoryAgent:
    def __init__(
        self,
        *,
        memory_store,
        grounding_operator: GroundingOperator,
        llm_config: LLMConfig | None = None,
        adapter_key: str = "generic",
        domain_profile: str | None = None,
    )

    def process(
        self,
        input_text: str,
        source: str = "user",
        *,
        tenant_id: str | None = None,
        app_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        space_type: str | None = None,
        metadata: dict | None = None,
        fact: dict | None = None,
        **kwargs,
    ) -> GenericProcessingResult
```

#### Write Pipeline (inside `process`)

1. Creates `Interaction` from input text
2. Optionally grounds a programmatically-supplied `fact` dict
3. Skips assistant-source text (prevents self-referential memory noise)
4. Runs LLM extraction → `GenericExtractionResult`
5. For each extracted fact:
   - Handles dispositions: `CAPTURE` → ground, `REFINE` → ground, `RETIRE` → find and supersede matching facts, `PASS` → skip
   - Resolves entities (subject/object) using `find_or_create_entity`
   - Coerces relations and entity types
   - Normalizes values and attribute keys
   - Optionally materializes attribute-value entities
6. Grounds each candidate via `grounding_operator.ground()`
7. Returns `GenericProcessingResult` with aggregated stats

#### Extraction Models

```python
class GenericExtractedFact(BaseModel):
    subject_name: str
    subject_type: EntityType
    relation: RelationType
    object_name: str | None
    object_type: EntityType | None
    value: str | None
    disposition: MemoryDisposition
    confidence: float            # 0.0-1.0, default 0.9
    attributes: dict[str, Any]

class GenericExtractionResult(BaseModel):
    facts: list[GenericExtractedFact]
```

#### `GenericProcessingResult` (dataclass)

```python
@dataclass
class GenericProcessingResult:
    interaction_id: str
    grounding_results: list[GroundingResult]
    approved_facts: list[ValidatedFact]
    rejected_facts: list[GroundingResult]
    warnings: list[str]
    dispositions: list[dict[str, Any]]
```

#### Fallback Extraction

When LLM extraction fails, `_heuristic_extract()` uses regex patterns for common statements:
- `"I use X"`, `"I prefer X"`, `"I like X"` → preference facts
- `"I no longer use X"`, `"I stopped X"` → retire dispositions
- Key-value lines with `=` or `:` separators → attribute facts

#### Static Helpers

| Method | Description |
|---|---|
| `_coerce_actor(source) -> ActorType` | Maps "assistant"/"agent" → AGENT, "tool" → TOOL, "system" → SYSTEM, else USER |
| `_coerce_relation(relation) -> RelationType` | Strict coercion (raises ValueError for unknown) |
| `_coerce_relation_dynamic(relation, has_object) -> RelationType` | Falls back to `RELATED_TO` or `HAS_ATTRIBUTE` |
| `_coerce_entity_type(value, *, default) -> EntityType` | Coerces with fallback to default |
| `_coerce_disposition(value) -> MemoryDisposition` | Coerces string; defaults to CAPTURE |
| `_looks_like_summary(value) -> bool` | Detects LLM summary artifacts (backticks, JSON, long text) |
| `_should_skip_unstructured_text(text, *, source) -> bool` | Skips empty text and "noted"/"okay" phrases |

### 12.3 Constraint Seed Mining

**Module:** `grounded_memory.adapters.discovery`

#### `DiscoveredConstraintSeed` (dataclass)

```python
@dataclass
class DiscoveredConstraintSeed:
    constraint_id: str
    name: str
    description: str
    relation_types: list[str]
    required_attribute_keys: list[str]
    require_value: bool
    confidence: float
    evidence_count: int
    mining_rule: str

    def as_dict(self) -> dict[str, Any]
```

#### `ConstraintSeedDiscoverer`

Mines validation/rejection signals and synthesizes candidate constraint seeds.

```python
class ConstraintSeedDiscoverer:
    def __init__(
        self,
        *,
        min_samples_per_relation: int = 20,
        min_rejections_per_relation: int = 6,
        min_gap: float = 0.35,
        min_gap_mode: str = "fixed",
        min_gap_floor: float = 0.15,
        min_gap_ceiling: float = 0.60,
        target_false_block_rate: float = 0.10,
        max_suggestions: int = 20,
    )

    def discover(
        self,
        *,
        validation_signals: list[dict],
        existing_constraint_ids: set[str] | None = None,
    ) -> list[DiscoveredConstraintSeed]
```

Discovery logic:
- Groups signals by relation type
- Computes rejection rate gap between facts missing vs. having a required attribute/value
- Synthesizes `require_value` seeds (missing value correlates with rejection)
- Synthesizes `required_attribute_key` seeds (missing attribute key correlates with rejection)
- Sorts by confidence × evidence_count; returns top `max_suggestions`

### 12.4 Seed Constraint Evaluators

**Module:** `grounded_memory.adapters.seeds`

Reusable evaluators generated from user seed payloads.

#### `SeedConstraintEvaluator`

```python
class SeedConstraintEvaluator(BaseConstraintEvaluator):
    def __init__(
        self,
        *,
        constraint_id: str,
        constraint_name: str,
        description: str,
        applies_to_relations=None,
        severity="error",
        required_attributes=None,
        required_attribute_keys=None,
        forbidden_attributes=None,
        require_object=False,
        require_value=False,
        value_regex=None,
    )
```

Checks: `require_object`, `require_value`, `value_regex`, exact `required_attributes` values, `required_attribute_keys` presence, `forbidden_attributes` absence.

#### `CardinalitySeedConstraintEvaluator`

```python
class CardinalitySeedConstraintEvaluator(BaseConstraintEvaluator):
    def __init__(
        self,
        *,
        constraint_id, constraint_name, description,
        relation: RelationType,
        max_count: int,
        severity="error",
        require_same_subject=True,
    )
```

Enforces max active fact count for a relation scope. Counts active facts by relation (optionally scoped to same subject).

#### `TemporalCardinalitySeedConstraintEvaluator`

```python
class TemporalCardinalitySeedConstraintEvaluator(BaseConstraintEvaluator):
    def __init__(
        self,
        *,
        constraint_id, constraint_name, description,
        relation: RelationType,
        max_count: int,
        window_seconds: int,
        severity="error",
        require_same_subject=True,
    )
```

Enforces max writes per relation in a rolling time window.

### 12.5 Healthcare Adapter

**Package:** `grounded_memory.adapters.healthcare` (10 files)

#### Healthcare Memory Agent

```python
class HealthcareMemoryAgent(BaseAsyncAdapterAgent):
    def __init__(self, memory_store, grounding_operator, llm_config, domain_profile="healthcare")

    async def process_interaction(
        self, raw_text, user_id=None, session_id=None,
        actor="user", metadata=None, **kwargs,
    ) -> Any  # HealthcareAgentResult
```

Pipeline: (1) LLM extraction via `HealthcareDatabaseExtractor`, (2) sync grounding for each candidate, (3) medication lifecycle side effects via `apply_medication_lifecycle_after_grounding`.

#### Healthcare Extraction Models

```python
class ExtractedMedicationLLM(BaseModel):
    name: str
    dosage: str | None
    frequency: str | None
    route: str | None
    duration: str | None
    action: str = "prescribe"
    confidence: float = 0.9

class ExtractedPatientLLM(BaseModel):
    name: str
    identifier: str | None
    age: str | None
    gender: str | None

class ExtractedAllergyLLM(BaseModel):
    allergen: str
    reaction: str | None
    severity: str | None

class ExtractedConditionLLM(BaseModel):
    name: str
    status: str | None
    diagnosed_date: str | None

class ClinicalExtractionResult(BaseModel):
    patient: ExtractedPatientLLM | None
    medications: list[ExtractedMedicationLLM]
    allergies: list[ExtractedAllergyLLM]
    conditions: list[ExtractedConditionLLM]
    clinical_intent: str | None
    extraction_notes: str | None
```

#### Healthcare Constraints

`YamlConstraintEvaluator` evaluates constraints from `configs/healthcare_constraints.yaml`:

- `intersection_empty` — allergy cross-reactivity check (medication ingredients ∩ patient allergy cross-reactive set)
- `no_major_interactions` / `no_moderate_interactions` — drug-drug interaction checks
- `cardinality_limit` — duplicate active medication or same therapeutic class limits

#### Medication Lifecycle

```python
def apply_medication_lifecycle_after_grounding(
    *, store, result: GroundingResult
) -> list[ValidatedFact]
```

Post-grounding hook: when a `DISCONTINUED` fact is approved, closes all active `PRESCRIBED` facts for matching medications.

#### Healthcare Retrieval

```python
class HealthcareQueryPlan(BaseModel):
    patient_name: str | None
    patient_identifier: str | None
    requested_categories: list[str]
    medication_names: list[str]
    allergy_names: list[str]
    safety_focus: bool
    as_of: datetime | None
    raw_time_expression: str | None
    ambiguous: bool
    ambiguity_reason: str | None

class HealthcareClinicalContext:
    query: str
    plan: HealthcareQueryPlan
    answer_context: AnswerContext
    seed_entities: list[str]
    current_medications: list[dict]
    allergies: list[dict]
    safety_alerts: list[dict]
    history: list[dict]
    def to_dict(self) -> dict: ...

class HealthcareRetrievalService:
    def __init__(self, memory_store, retriever: GraphRetriever, llm_client=None)
    def retrieve_current_state(self, query, scope=None, **kwargs) -> HealthcareClinicalContext
    def retrieve_historical_state(self, query, as_of: datetime, scope=None, **kwargs) -> HealthcareClinicalContext
    def check_cross_patient_isolation(self, query, scope=None, forbidden_medication_names=None, **kwargs) -> tuple[bool, set[str]]
    def find_patients_by_medication(self, medication_name) -> list[str]
    def find_patients_by_allergy(self, allergen_name) -> list[str]
    def find_patients_by_shared_entity(self, entity_name, relation) -> list[str]
    def generate_grounded_answer(self, query, scope, llm_client, **kwargs) -> str
```

#### Healthcare Knowledge Base

**Module:** `grounded_memory.adapters.healthcare.knowledge`

In-memory KB with default data for allergy cross-reactivity (penicillin ↔ amoxicillin/ampicillin), drug aliases (Advil → ibuprofen), drug ingredients, therapeutic classes, and major/moderate drug-drug interactions.

Key functions:
```python
def normalize_drug_name(name: str) -> str
def get_drug_ingredients(drug_name: str) -> set[str]
def get_therapeutic_classes(drug_name: str) -> set[str]
def get_cross_reactive_ingredients(allergen: str) -> set[str]
def expand_drug_terms(drug_name: str) -> set[str]
def check_major_interaction(drug1: str, drug2: str) -> bool
def check_moderate_interaction(drug1: str, drug2: str) -> bool
def register_source(kb: InMemoryKnowledgeBase) -> None
def load_json_file(path) -> InMemoryKnowledgeBase
def load_yaml_file(path) -> InMemoryKnowledgeBase
def load_csv_interactions(path, target="major") -> InMemoryKnowledgeBase
```

#### KB Manager

```python
class KBManager:
    def __init__(self, config_path: str | None = None)
    def initialize_knowledge_base(self) -> bool

def initialize_from_config(config_path=None) -> bool
```

Loads external data sources (RxNorm, openFDA, JSON/YAML files) based on `configs/healthcare_kb.yaml`.

#### External Loaders

**RxNorm** (`loaders/rxnorm.py`):
```python
class RxNormLoader:
    def lookup_rxcui(self, drug_name, search_type="all") -> str | None
    def fetch_interactions_for_rxcui(self, rxcui) -> dict[str, list[str]]
    def get_drug_properties(self, rxcui) -> dict
    def build_kb_from_rxnorm(self, drug_names, include_minor=False, ...) -> InMemoryKnowledgeBase
```

**openFDA** (`loaders/openfda.py`):
```python
class OpenFDALoader:
    def search_drug_labels(self, query, limit=100) -> dict
    def fetch_drug_label(self, drug_name) -> dict | None
    def extract_ingredients_from_labels(self, labels) -> InMemoryKnowledgeBase
    def fetch_batch_labels(self, drug_names, batch_size=10, ...) -> InMemoryKnowledgeBase
```

**Cache** (`loaders/cache.py`):
```python
class FileCache:
    def get(self, key) -> Any | None
    def put(self, key, value, ttl_hours=None) -> None
    def evict_expired(self) -> int
    def clear(self) -> None
    def stats(self) -> dict
```

---

## 13. Public SDK (`Memory` Class)

**Module:** `grounded_memory.memory`  
**Import:** `from gmem import Memory`

The `Memory` class is the primary user-facing API for the Grounded Memory System.

### 13.1 Constructor

```python
class Memory:
    def __init__(
        self,
        adapter: str = "generic",
        domain_profile: str | None = None,
        *,
        storage_backend: str | None = None,         # "memory"|"hybrid"|"postgres"|"postgres_hybrid"
        neo4j_config: dict | None = None,
        use_llm: bool = True,
        llm_config: LLMConfig | None = None,
        configure_validator: callable | None = None,
        agent_factory: callable | None = None,
        agent: Any | None = None,
        intent_router: Any | None = None,
        optimization_profile: OptimizationProfile | str = OptimizationProfile.BALANCED,
        default_max_seeds: int | None = None,
        default_max_hops: int | None = None,
        default_max_facts: int | None = None,
        default_strategy: str | None = None,
        relationship_preset: str | None = None,
        require_scope: bool = False,
        default_tenant_id: str | None = None,
        default_app_id: str | None = None,
        default_user_id: str | None = None,
        default_agent_id: str | None = None,
        default_space_type: str | None = None,
    )
```

### 13.2 Write Methods

| Method | Signature | Description |
|---|---|---|
| `add` | `(text \| list[dict], *, source="user", tenant_id=None, ..., metadata=None, **kwargs) -> Any` | Ingest interaction text via LLM-backed agent |
| `remember` | `(...) -> Any` | Alias for `add()` |
| `add_many` | `(messages, *, source="user", ..., continue_on_error=True) -> dict` | Batch ingestion with per-item results |
| `add_entity` | `(name, *, entity_type="FACILITY", attributes=None, canonical_id=None, uniqueness_key=None, entity_id=None) -> dict` | Create/reuse an entity |
| `add_fact` | `(*, subject_id, relation, object_id=None, value=None, confidence=0.9, attributes=None, source="system", tenant_id=None, ..., source_interaction_id=None) -> dict` | Write structured fact through grounding pipeline |
| `update_fact` | `(fact_id, *, relation=None, object_id=None, value=None, confidence=None, attributes=None, ...) -> dict` | Update a fact via supersession |
| `delete_fact` | `(fact_id, *, reason="deleted via api") -> dict` | Soft-delete by closing temporal validity window |
| `process` | `(text, *, source="user", tenant_id=None, ..., **kwargs) -> dict` | Auto-route: `REMEMBER` → `add`, `RECALL`/`FIND`/`EXPLAIN` → `search` |

### 13.3 Read Methods

| Method | Signature | Description |
|---|---|---|
| `search` | `(query, *, tenant_id=None, ..., at_time=None, lookback_days=None, limit=10, threshold=0.0, rerank_debug=False, max_hops=None, max_seeds=None, strategy=None) -> list[dict]` | Search memory; returns serialized `SearchResult` dicts |
| `query` | `(query, **kwargs) -> list[dict]` | Alias for `search()` |
| `retrieve` | `(query, **kwargs) -> list[dict]` | Alias for `search()` |
| `build_context` | `(query, *, tenant_id=None, ..., at_time=None, lookback_days=None, max_seeds=None, max_hops=None, max_facts=None, strategy=None) -> AnswerContext` | Retrieve structured `AnswerContext` |
| `build_memory_prompt` | `(query, *, tenant_id=None, ..., limit=10, threshold=0.0, ...) -> str` | Retrieve and render as prompt-ready bullet lines |
| `render_memories` | `(items: list[dict], *, empty_text="No relevant memory.") -> str` | Render retrieved items into compact prompt block |
| `route` | `(query: str) -> UserIntent` | Classify user intent |
| `history` | `(*, fact_id=None, entity_id=None, memory_id=None, tenant_id=None, ..., include_inactive=False, limit=50) -> list[dict]` | Temporal fact lineage queries |

### 13.4 Introspection Methods

| Method | Signature | Description |
|---|---|---|
| `get` | `(memory_id: str) -> dict \| None` | Get object by ID (checks entities, facts, interactions, rejections) |
| `get_all` | `(*, tenant_id=None, ...) -> dict` | Return all entities and active facts |
| `list_entities` | `(*, entity_type=None, limit=50) -> list[dict]` | List entities, optionally filtered |
| `list_facts` | `(*, tenant_id=None, ..., active_only=True, limit=50) -> list[dict]` | List facts |
| `list_interactions` | `(*, tenant_id=None, ..., limit=50) -> list[dict]` | List recent interactions |
| `runtime_status` | `() -> dict` | Operational metadata for health checks |
| `healthcheck` | `() -> dict` | Lightweight readiness payload |

### 13.5 Constraint Governance Methods

| Method | Signature | Description |
|---|---|---|
| `add_constraint_seed` | `(*, constraint_id, name, description, relation_types, lifecycle, priority, severity, ..., form_id, form_metadata) -> dict` | Register a user-defined dynamic constraint seed |
| `add_cardinality_constraint_seed` | `(*, constraint_id, name, description, relation, max_count, ...) -> dict` | Register a count/threshold-based dynamic seed |
| `add_temporal_cardinality_constraint_seed` | `(*, constraint_id, name, description, relation, max_count, window_seconds, ...) -> dict` | Register a rolling-window count seed |
| `discover_constraint_seeds` | `(*, signal_limit=500, min_samples_per_relation=20, ..., max_suggestions=20) -> list[dict]` | Mine validation signals and synthesize seeds |
| `register_discovered_constraint_seeds` | `(seeds, *, lifecycle="shadow", priority=50, continue_on_error=True) -> dict` | Register discovered seeds |
| `discover_and_register_constraint_seeds` | `(*, signal_limit=500, ..., lifecycle="shadow", priority=50) -> dict` | Mine and immediately register |
| `list_constraint_seeds` | `() -> list[dict]` | List managed constraints with lifecycle metadata |
| `set_constraint_seed_lifecycle` | `(constraint_id, lifecycle) -> bool` | Change lifecycle stage |
| `replay_constraint_seeds` | `(candidates) -> dict[str, dict]` | Replay seeds over candidates; return metrics |
| `promote_constraint_seeds` | `(replay_metrics, *, min_trigger_rate=0.01, max_projected_false_block_rate=0.02, min_candidates=100) -> list[str]` | Promote seeds to active |

### 13.6 Configuration and Lifecycle

| Method | Signature | Description |
|---|---|---|
| `configure_optimization` | `(*, profile=None, max_seeds=None, max_hops=None, max_facts=None, strategy=None) -> dict` | Update runtime retrieval defaults |
| `close` | `() -> None` | Close resources (connections, background threads) |
| `__enter__` / `__exit__` | | Context manager support |

### 13.7 Supporting Types

#### `SearchResult` (dataclass)

```python
@dataclass
class SearchResult:
    fact_id: str
    relation: str
    subject_id: str
    subject_name: str
    object_id: str | None
    object_name: str | None
    value: str | None
    confidence: float
    valid_from: str
    valid_to: str | None
    score: float | None = None
    signals: dict[str, Any] = field(default_factory=dict)
```

#### `OptimizationProfile` (str, Enum)

| Profile | max_seeds | max_hops | max_facts | strategy |
|---|---|---|---|---|
| `LATENCY` | 3 | 1 | 12 | `WEIGHTED` |
| `BALANCED` | 6 | 2 | 30 | `SAFETY_PRIORITY` |
| `RECALL` | 10 | 3 | 60 | `BREADTH_FIRST` |

#### `ScopeContext` (dataclass, frozen)

```python
@dataclass(frozen=True)
class ScopeContext:
    tenant_id: str | None
    app_id: str | None
    user_id: str | None
    agent_id: str | None = None
    run_id: str | None = None
    space_type: str | None = None

    # Properties: scope_id -> "tenant_id:app_id:user_id"
    # Methods: as_dict() -> dict[str, str]
```

### 13.8 Scope System

Every write/read method accepts scope fields: `tenant_id`, `app_id`, `user_id`, `agent_id`, `run_id`, `space_type`. When `require_scope=True`, the `tenant_id:app_id:user_id` triplet is mandatory. Facts inherit scope from the `Interaction` they originate from.

---

## 14. FastAPI Service

**Module:** `grounded_memory.service`

### 14.1 `create_app()`

```python
def create_app(memory: Memory | None = None) -> FastAPI
```

Creates a FastAPI application configured with all routes, middleware, and exception handlers. If no `Memory` instance is provided, uses the `_lifespan` async context manager to create one from environment variables.

**Environment-based initialization:**
```python
def _memory_from_env() -> Memory:
    backend = os.getenv("GM_STORAGE_BACKEND")
    adapter = os.getenv("GM_ADAPTER") or os.getenv("GM_DOMAIN_PROFILE", "generic")
    return Memory(adapter=adapter, domain_profile=adapter, storage_backend=backend)
```

### 14.2 API Endpoints

#### Health

| Method | Path | Handler | Response |
|---|---|---|---|
| `GET` | `/health/live` | `live()` | `HealthResponse(data={"service": "grounded-memory-api"})` |
| `GET` | `/health/ready` | `ready()` | `HealthResponse(data=memory.healthcheck())` |

#### Status

| Method | Path | Handler | Response |
|---|---|---|---|
| `GET` | `/v1/status` | `status()` | `ApiEnvelope(data=memory.runtime_status())` |

#### Memories

| Method | Path | Request Body / Query | Calls |
|---|---|---|---|
| `POST` | `/v1/memories/add` | `AddMemoryRequest` | `memory.add(text)` or `memory.add(messages)` |
| `POST` | `/v1/memories/search` | `SearchMemoryRequest` | `memory.search(query, ...)` |
| `GET` | `/v1/memories/prompt` | `query`, `tenant_id`, ..., `limit`, `at_time`, `lookback_days` | `memory.build_memory_prompt(query, ...)` |
| `GET` | `/v1/memories/facts` | `tenant_id`, ..., `active_only`, `limit` | `memory.list_facts(...)` |
| `GET` | `/v1/memories/interactions` | `tenant_id`, ..., `limit` | `memory.list_interactions(...)` |
| `GET` | `/v1/memories/all` | `tenant_id`, ..., `space_type` | `memory.get_all(...)` |

#### Entities

| Method | Path | Request Body | Calls |
|---|---|---|---|
| `POST` | `/v1/entities` | `AddEntityRequest` | `memory.add_entity(...)` |

#### Facts

| Method | Path | Request Body | Calls |
|---|---|---|---|
| `POST` | `/v1/facts` | `AddFactRequest` | `memory.add_fact(...)` |
| `PATCH` | `/v1/facts/{fact_id}` | `UpdateFactRequest` | `memory.update_fact(fact_id, ...)` |
| `DELETE` | `/v1/facts/{fact_id}` | query param `reason` | `memory.delete_fact(fact_id, reason=...)` |

### 14.3 Request/Response Models

```python
class MessagePayload(BaseModel):
    role: str = "user"
    content: str                          # min_length=1
    tenant_id, app_id, user_id, agent_id, run_id, session_id, space_type: str | None
    metadata: dict[str, Any] | None

class AddMemoryRequest(BaseModel):
    text: str | None                      # Exactly one of text or messages required
    messages: list[MessagePayload] | None
    source: str = "user"
    # ... all scope fields ...

class SearchMemoryRequest(BaseModel):
    query: str                            # min_length=1
    # ... scope fields ...
    at_time: datetime | None
    lookback_days: int | None             # ge=1, le=3650
    limit: int = 10                       # ge=1, le=100
    max_hops: int | None                  # ge=1, le=5
    max_seeds: int | None                 # ge=1, le=20
    strategy: str | None

class AddEntityRequest(BaseModel):
    name: str
    entity_type: str = "FACILITY"
    attributes: dict | None
    canonical_id: str | None
    uniqueness_key: str | None
    entity_id: str | None

class AddFactRequest(BaseModel):
    subject_id: str
    relation: str
    object_id: str | None
    value: str | None                    # At least one of object_id or value required
    confidence: float = 0.9
    attributes: dict | None
    source: str = "system"
    # ... scope fields ...
    source_interaction_id: str | None

class UpdateFactRequest(BaseModel):
    relation, object_id, value, confidence, attributes: str | None
    source: str = "system"
    # ... scope fields ...

class DeleteFactRequest(BaseModel):
    reason: str = "deleted via api"

class ApiEnvelope(BaseModel):
    ok: bool = True
    data: Any

class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    data: dict[str, Any]
```

### 14.4 Middleware

**Request logging middleware** — intercepts every HTTP request:
- Extracts or generates `x-request-id` header
- Times the request
- Logs success/failure with structured JSON
- On unhandled exceptions, returns `500` with `{"ok": false, "error": {...}}`

### 14.5 Exception Handlers

| Exception | HTTP Status | Response |
|---|---|---|
| `ValueError` | 400 | `{"ok": false, "error": {"type": "ValueError", "message": "..."}}` |
| `RuntimeError` | 503 | `{"ok": false, "error": {"type": "RuntimeError", "message": "..."}}` |
| `HTTPException` | per exception | `{"ok": false, "error": {"type": "HTTPException", "message": exc.detail}}` |

### 14.6 Module-Level App Instance

```python
from grounded_memory.service import app  # Pre-created FastAPI instance (or None if FastAPI unavailable)
```

---

## 15. Configuration

### 15.1 Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `GM_ADAPTER` | Adapter key (`generic`, `healthcare`, ...) | — |
| `GM_DOMAIN_PROFILE` | Fallback adapter key | `"generic"` |
| `GM_STORAGE_BACKEND` | Storage backend (`memory`, `hybrid`, `postgres`, `postgres_hybrid`) | — |
| `GM_REQUIRE_SCOPE` | Set `1` to enforce `tenant_id:app_id:user_id` on every call | — |
| `GM_SCOPE_TENANT_ID` | Default scope tenant ID | — |
| `GM_SCOPE_APP_ID` | Default scope app ID | — |
| `GM_SCOPE_USER_ID` | Default scope user ID | — |
| `GM_SCOPE_AGENT_ID` | Default scope agent ID | — |
| `GM_LOG_LEVEL` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `"INFO"` |
| `GM_LOG_JSON` | Set `1` for JSON-formatted logs | — |
| `LLM_PROVIDER` | LLM provider (`openrouter`, `local`) | — |
| `LLM_BASE_URL` | LLM API base URL | `https://openrouter.ai/api/v1` |
| `LLM_MODEL` | Model identifier | `z-ai/glm-4.5-air:free` |
| `OPENROUTER_API_KEY` | API key for OpenRouter | — |
| `LLM_API_KEY` | Generic API key (local endpoints) | — |
| `LLM_TEMPERATURE` | Generation temperature | `0.1` |
| `LLM_MAX_TOKENS` | Max generation tokens | `2048` |
| `LLM_TIMEOUT` | Request timeout seconds | `120.0` |
| `LLM_MAX_RETRIES` | Max retry attempts | `3` |
| `LLM_RETRY_DELAY` | Base retry delay seconds | `1.0` |
| `OPENROUTER_SITE_URL` | OpenRouter rankings site URL | — |
| `OPENROUTER_SITE_NAME` | OpenRouter app name | `"GroundedMemory"` |
| `POSTGRES_HOST` | PostgreSQL host | — |
| `POSTGRES_PORT` | PostgreSQL port | `35432` |
| `POSTGRES_DB` | PostgreSQL database | — |
| `POSTGRES_USER` | PostgreSQL user | — |
| `POSTGRES_PASSWORD` | PostgreSQL password | — |
| `NEO4J_URI` | Neo4j connection URI | — |
| `NEO4J_USER` | Neo4j username | — |
| `NEO4J_PASSWORD` | Neo4j password | — |

### 15.2 Configuration Files

| File | Purpose |
|---|---|
| `configs/healthcare_constraints.yaml` | Healthcare domain-specific constraint definitions |
| `configs/healthcare_kb.yaml` | Healthcare knowledge base source configuration (RxNorm, openFDA) |
| `configs/generic_constraints.yaml` | Generic domain constraint definitions |
| `configs/engineering_constraints.yaml` | Engineering domain constraints |
| `configs/finance_constraints.yaml` | Finance domain constraints |
| `configs/legal_constraints.yaml` | Legal domain constraints |
| `configs/llm_config.yaml` | LLM configuration overrides |
| `configs/neo4j_config.yaml` | Neo4j connection configuration |

### 15.3 Logging

```python
from grounded_memory.logging_utils import configure_logging

configure_logging(level="DEBUG", json_output=True)
```

`JsonLogFormatter` produces structured JSON log lines for service deployments. Reads `GM_LOG_LEVEL` and `GM_LOG_JSON` from environment by default.

---

## 16. Package Imports

### From `gmem` (Public Facade)

```python
from gmem import (
    Memory,                    # Main SDK entry point
    ConstraintValidator,       # Constraint validation engine
    GroundedMemorySystem,      # System orchestrator
    GroundingOperator,         # Memory formation engine
    LLMConfig,                 # LLM configuration
    MemoryStore,               # In-memory store
    configure_logging,         # Logging setup
    create_app,                # FastAPI app factory
    list_registered_adapters,  # Adapter registry query
    register_adapter,          # Dynamic adapter registration
    unregister_adapter,        # Adapter removal
    __version__,               # "0.1.0"
)
```

### From `grounded_memory` (Implementation)

```python
# Core models
from grounded_memory.core.models import (
    Interaction, Entity, CandidateFact, ValidatedFact,
    Constraint, AnswerContext, RejectionRecord,
    ActorType, RelationType, EntityType, ConstraintType,
    CandidateFactStatus, MemoryDisposition,
)

# Grounding
from grounded_memory.core.grounding import (
    GroundingOperator, AsyncGroundingOperator,
    GroundingDecision, GroundingResult,
    process_interaction_facts, process_interaction_facts_async,
)

# Constraints
from grounded_memory.core.constraints import (
    ConstraintValidator, BaseConstraintEvaluator,
    ConstraintViolation, ValidationResult, KnowledgeState,
    ConstraintLifecycleStatus, ConstraintSource,
    ManagedConstraint, ConstraintReplayMetrics,
    ProhibitionConstraint, CardinalityConstraint, TemporalConstraint,
)

# Conflict resolution
from grounded_memory.core.conflict_resolution import (
    ConflictResolver, ConflictResolutionStrategy,
    ConflictSignal, ConflictResolution,
)

# Tuple normalization
from grounded_memory.core.tuple_normalization import (
    build_fact_semantic_key, fact_values_equal,
    normalize_fact_value_for_match, sanitize_fact_value,
    normalize_attribute_key, parse_keyed_value,
)

# Entity identity
from grounded_memory.core.entity_identity import (
    build_entity_uniqueness_key, stable_entity_id,
)

# Intent routing
from grounded_memory.core.intent import (
    IntentAction, UserIntent,
    KeywordIntentRouter, LLMIntentRouter, HybridIntentRouter,
)

# LLM
from grounded_memory.llm import (
    LLMClient, SyncLLMClient, LLMConfig, LLMProvider, LLMFactExtractor,
)

# Retrieval
from grounded_memory.retrieval import (
    GraphRetriever, RetrievalStrategy, RelationshipPreset,
    RelationshipWeight, select_seed_entities,
)

# Adapters
from grounded_memory.adapters import (
    GenericMemoryAgent, GenericProcessingResult,
    register_adapter, list_registered_adapters,
    SeedConstraintEvaluator, CardinalitySeedConstraintEvaluator,
    TemporalCardinalitySeedConstraintEvaluator,
    ConstraintSeedDiscoverer, DiscoveredConstraintSeed,
)

# Service (FastAPI)
from grounded_memory.service import app, create_app

# Logging
from grounded_memory.logging_utils import configure_logging, JsonLogFormatter
```
