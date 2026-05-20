# Grounded Memory SDK Reference

This document is the complete developer-facing reference for the high-level SDK.

## Import Surface

```python
from grounded_memory import (
    Memory,
    OptimizationProfile,
    SearchResult,
)
```

## Memory Constructor

```python
Memory(
  adapter: str | None = None,
    domain_profile: str = "generic",
    *,
  storage_backend: "memory" | "hybrid" | "postgres" | "postgres_hybrid" | None = None,
    neo4j_config: Neo4jConfig | None = None,
    use_llm: bool = False,
    llm_config: LLMConfig | None = None,
    configure_validator: Callable | None = None,
    agent_factory: Callable | None = None,
    agent: Any | None = None,
    optimization_profile: OptimizationProfile | str = "balanced",
    default_max_seeds: int | None = None,
    default_max_hops: int | None = None,
    default_max_facts: int | None = None,
    default_strategy: RetrievalStrategy | str | None = None,
    relationship_preset: RelationshipPreset | str | None = None,
)
```

### Constructor Behavior

- `adapter`
  - runtime adapter key (`generic` by default)
  - selects validator/agent wiring through the adapter registry
- `domain_profile`
  - compatibility alias for legacy profile-based callers
  - when both are provided, `adapter` is preferred
- `storage_backend`
  - `memory`: in-process storage
  - `hybrid`: in-process source + Neo4j projection
  - `postgres`: in-process source + PostgreSQL durability
  - `postgres_hybrid`: in-process source + PostgreSQL durability + Neo4j projection
- `use_llm=True` creates an adapter agent with extraction/generation support.
- `optimization_profile` seeds default retrieval depth/breadth/strategy.
- Explicit `default_*` parameters override profile defaults.

## Optimization Profiles

- `latency`
  - `max_seeds=3`, `max_hops=1`, `max_facts=12`, `strategy=weighted`
- `balanced`
  - `max_seeds=6`, `max_hops=2`, `max_facts=30`, `strategy=safety_priority`
- `recall`
  - `max_seeds=10`, `max_hops=3`, `max_facts=60`, `strategy=breadth_first`

## Core DX Methods

### `add(text, source="user", **kwargs)`
Ingest one interaction. If agent exists, routes through agent processing; otherwise logs event.

### `remember(...)`
Alias of `add(...)` for compatibility semantics.

### `add_many(messages, source="user", continue_on_error=False)`
Batch ingest helper.

Accepted payloads:
- string: treated as text
- dict: expected keys include `text` or `content`, optional `source`, plus passthrough metadata

Returns summary:
- `ingested`, `failed`, `total`, `results[]`

### `add_entity(name, ...)`
Creates or reuses an entity.

### `add_fact(subject_id, relation, object_id|value, ...)`
Writes a candidate fact through the grounding + validation pipeline.

## Intent Routing Methods

### `route(query) -> UserIntent`
Classifies the cognitive intent of a natural-language query.

Returns a `UserIntent` with fields:
- `action`: `REMEMBER`, `RECALL`, `FIND_RELATED`, `EXPLAIN`, or `UNKNOWN`
- `confidence`: float in [0.0, 1.0]
- `mentions`: list of entity names detected in the query
- `temporal_anchor`: time expression if present
- `explanation`: human-readable classification rationale

```python
intent = memory.route("What allergies does Jane Doe have?")
print(intent.action)      # "recall"
print(intent.confidence)  # 0.85
```

### `process(text, source="user", **kwargs) -> dict`
Auto-routes input based on inferred intent:
- `REMEMBER` → calls `add(text)` and returns write results
- `RECALL` / `FIND_RELATED` / `EXPLAIN` → calls `search(text)` and returns read results
- `UNKNOWN` → tries search first, then add if empty

Returns a dict with keys:
- `intent`: serialized `UserIntent`
- `results`: the operation result (write summary or search results)

```python
result = memory.process("Jane Doe is allergic to penicillin", user_id="demo")
print(result["intent"]["action"])   # "remember"
print(result["results"])            # write result summary
```

## Retrieval Methods

### `build_context(query, at_time=None, lookback_days=None, max_seeds=None, max_hops=None, max_facts=None, strategy=None)`
Returns `AnswerContext`. Any `None` values use optimization defaults.

Temporal context options:
- `at_time`: point-in-time retrieval timestamp
- `lookback_days`: include only facts introduced within a recent temporal window

Bitemporal note:
- Retrieval-time filtering (`at_time`) applies to valid time (`valid_from`/`valid_to`).
- Record-time provenance is maintained by the storage layer (for example interaction timestamps and persistence `created_at`) for audit/history workflows.

### `search(query, limit=10, at_time=None, lookback_days=None, max_hops=None, max_seeds=None, strategy=None)`
Primary retrieval method returning serialized fact rows.

### `query(...)`
Alias of `search(...)`.

### `retrieve(...)`
Alias of `search(...)` for compatibility naming.

### `configure_optimization(...)`
Mutates runtime retrieval defaults.

```python
memory.configure_optimization(profile="latency")
memory.configure_optimization(max_hops=2, max_facts=20)
```

## Lifecycle + CRUD

- `get(memory_id)`
- `update_fact(fact_id, ...)`
- `delete_fact(fact_id, reason="...")`
- `history(entity_id=None, relation=None, include_inactive=True, limit=100)`
- `list_entities(...)`
- `list_facts(...)`
- `list_interactions(...)`
- `get_all()`

## Dynamic Seed APIs

- `add_constraint_seed(...)`
- `add_cardinality_constraint_seed(...)`
- `add_temporal_cardinality_constraint_seed(...)`
- `list_constraint_seeds()`
- `set_constraint_seed_lifecycle(...)`
- `replay_constraint_seeds(candidates)`
- `promote_constraint_seeds(replay_metrics, ...)`

### Autonomous Discovery APIs

- `discover_constraint_seeds(...)`
  - mines validation signals and synthesizes seed proposals
  - supports `min_gap_mode="fixed"|"adaptive"`
  - adaptive controls: `min_gap_floor`, `min_gap_ceiling`, `target_false_block_rate`
- `register_discovered_constraint_seeds(seeds, lifecycle="shadow", ...)`
  - registers synthesized seeds as managed dynamic constraints
- `discover_and_register_constraint_seeds(...)`
  - one-call mining + synthesis + registration

Example:

```python
proposals = memory.discover_constraint_seeds(signal_limit=2000)
result = memory.register_discovered_constraint_seeds(proposals, lifecycle="shadow")

# or one call
autodiscovery = memory.discover_and_register_constraint_seeds(
    signal_limit=2000,
    min_samples_per_relation=20,
    min_rejections_per_relation=6,
  min_gap_mode="adaptive",
  target_false_block_rate=0.10,
    lifecycle="shadow",
)
```

## Context Management

- `close()`
- context manager support:

```python
with Memory(domain_profile="generic") as memory:
    memory.add("hello")
```

## Minimal End-to-End Example

```python
from grounded_memory import Memory, OptimizationProfile

memory = Memory(domain_profile="generic", optimization_profile=OptimizationProfile.BALANCED)

memory.remember("Project Atlas uses PostgreSQL", source="user")

batch = memory.add_many([
    "Atlas runs in eu-west-1",
    {"content": "Atlas owner is platform-team", "source": "assistant"},
])

results = memory.retrieve("what database does atlas use", limit=5)

memory.configure_optimization(profile="latency")
fast_results = memory.search("atlas owner", limit=3)
```
