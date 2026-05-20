# Hybrid Storage Architecture: PostgreSQL + Neo4j

The thesis demo uses a hybrid backend:

```python
Memory(adapter="healthcare", storage_backend="postgres_hybrid")
```

PostgreSQL is the durable bitemporal source of truth. Neo4j is a rebuildable
active graph projection for current-time traversal.

## Why Two Stores

| Store | Role | Strength |
| --- | --- | --- |
| PostgreSQL | Durable truth and audit store | transactions, temporal queries, rejection history, scope filtering |
| Neo4j | Active graph projection | fast relationship traversal for current context |

The key invariant is simple:

```text
PostgreSQL can rebuild Neo4j.
Neo4j must never be treated as the only copy of truth.
```

## Storage Modes

| Backend | Class | Notes |
| --- | --- | --- |
| `memory` | `MemoryStore` | in-process source of truth; used by most deterministic tests |
| `hybrid` | `HybridMemoryStore` | in-process store plus Neo4j projection |
| `postgres` | `PostgresHybridMemoryStore` with graph sync off | durable primary store without Neo4j |
| `postgres_hybrid` | `PostgresHybridMemoryStore` with graph sync on | thesis demo target |

`PostgresHybridMemoryStore` wraps an async PostgreSQL store and keeps an
in-process cache compatible with the existing synchronous runtime contracts. On
startup it rehydrates entities, interactions, validated facts, and rejections
from PostgreSQL, then rebuilds Neo4j when graph sync is enabled.

## Write Path

```text
Interaction / Entity / ValidatedFact / RejectionRecord
  |
  v
PostgreSQL write
  |
  v
in-process cache update
  |
  v
Neo4j active projection update, if enabled
```

For accepted active facts, Neo4j receives a relationship. For superseded or
closed facts, Neo4j removes the active relationship while PostgreSQL retains the
historical row.

## Supersession and Discontinuation

Dose changes are handled by grounding supersession:

```text
old PRESCRIBED fact.valid_to = new_candidate.extracted_at
old PRESCRIBED fact.superseded_by = new_fact.id
new PRESCRIBED fact remains active
```

Medication lifecycle events are handled by the healthcare adapter:

```text
DISCONTINUED / hold fact approved
  -> find matching active PRESCRIBED facts
  -> set their valid_to to the discontinuation valid_from
  -> set superseded_by to the discontinuation fact id
  -> remove closed prescription edges from Neo4j
```

This keeps “current medications” clean while preserving medication history.

## Read Path

Current-time retrieval:

```text
query plan -> seed entities -> GraphRetriever -> Neo4j active projection
```

Historical retrieval:

```text
query plan with as_of -> seed entities -> GraphRetriever(at_time=...)
  -> bitemporal filtering from primary store
```

Generic graph retrieval lives in `src/grounded_memory/retrieval/graph.py`.
Healthcare-specific query planning and clinical post-filtering live in
`src/grounded_memory/adapters/healthcare/retrieval.py`.

## PostgreSQL Data Responsibilities

PostgreSQL stores:

- `entities`, including `canonical_id` and JSON attributes
- `interactions`, including raw text and scope/provenance fields
- `validated_facts`, including relation tuple, valid-time boundaries,
  supersession, source text, source metadata, and attributes
- `rejection_records`, including constraint id/name, reason, alternatives,
  severity, and domain reasoning

Scope fields are persisted where available:

```text
tenant_id, app_id, user_id, agent_id, run_id, space_type
```

## Neo4j Projection Responsibilities

Neo4j stores active entities and currently-active validated facts as graph
relationships. The projection is optimized for traversal, not audit.

Typical healthcare relationship types:

```text
PRESCRIBED
DISCONTINUED
HAS_ALLERGY
HAS_CONDITION
TREATS
INTERACTS_WITH
CONTRAINDICATED_WITH
```

Projection drift can be corrected by rebuilding from the primary store.

## Demo Operations

Start services:

```bash
make services-up
```

Run the healthcare backend smoke:

```bash
make smoke-healthcare-backends
```

Run the full healthcare demo:

```bash
PYTHONPATH=src python demos/demo_bitemporal.py
```

The smoke and demo report runtime statistics and backend counts so demo-day
claims can be checked against both stores.

## Failure Modes

| Scenario | Expected behavior |
| --- | --- |
| PostgreSQL unavailable | `postgres` / `postgres_hybrid` initialization fails; use `make services-up` |
| Neo4j unavailable | graph projection cannot initialize; current-time graph traversal falls back where possible |
| Neo4j drift | call rebuild logic through the hybrid store or restart with rehydration |
| LLM unavailable | deterministic tests still validate retrieval/lifecycle behavior; live demo extraction needs configured LLM |
