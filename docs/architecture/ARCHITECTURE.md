# Grounded Memory Architecture

Grounded Memory, or GMem, is a correctness-first memory runtime for LLM agents.
The current thesis demo uses medical medication reconciliation to show the core
claim: candidate knowledge proposed by an LLM should be governed before it
becomes durable memory, and retrieval should answer from grounded state rather
than from unvalidated text.

For formal definitions of actors, components, workflows, and axioms, see
[System Definition](./SYSTEM_DEFINITION.md).

This document reflects the current implementation. Older usecase-coupled and
finance-labelled descriptions have been removed from the architecture story.

## Core Idea

Traditional RAG usually follows this path:

```text
store text -> retrieve likely snippets -> ask the LLM to reason safely
```

GMem follows this path:

```text
observe interaction -> extract candidate facts -> validate constraints
  -> persist accepted facts / rejected audit records -> retrieve structured context
```

The distinction matters in high-stakes domains. Invalid prescriptions, allergy
conflicts, and unsafe drug combinations should be rejected at write time, not
left for a later prompt to notice.

## Runtime Layers

```text
Application / demo
  |
  |  Memory(adapter="healthcare", storage_backend="postgres_hybrid",
  |         require_scope=True)
  v
SDK facade
  src/grounded_memory/memory.py
  src/gmem/
  |
  v
System orchestration
  src/grounded_memory/system.py
  src/grounded_memory/core/system.py
  |
  +-- Adapter layer
  |     src/grounded_memory/adapters/generic_agent.py
  |     src/grounded_memory/adapters/healthcare/
  |
  +-- Governance layer
  |     src/grounded_memory/core/constraints.py
  |     src/grounded_memory/core/grounding.py
  |
  +-- Storage layer
  |     src/grounded_memory/core/store.py
  |     src/grounded_memory/core/postgres_store.py
  |     src/grounded_memory/core/postgres_hybrid_store.py
  |     src/grounded_memory/core/neo4j_store.py
  |
  +-- Retrieval layer
        src/grounded_memory/retrieval/graph.py
        src/grounded_memory/adapters/healthcare/retrieval.py
```

The core runtime is domain-neutral. Healthcare behavior lives in the healthcare
adapter: clinical extraction, clinical constraints, medication lifecycle rules,
and medication-reconciliation retrieval views.

## Six-Object Memory Taxonomy

GMem keeps the memory model explicit instead of collapsing everything into
documents or summaries.

| Object | Implementation | Role |
| --- | --- | --- |
| `Interaction` | `core/models.py` | Immutable event log of raw observed text plus scope/provenance metadata |
| `Entity` | `core/models.py` | Symbolic node such as patient, medication, allergy, service, project |
| `CandidateFact` | `core/models.py` | Untrusted tuple proposed by extraction or SDK writes |
| `ValidatedFact` | `core/models.py` | Accepted tuple with valid-time boundaries and provenance |
| `Constraint` | `core/constraints.py` | Write-time governance rule that approves, warns, or rejects |
| `AnswerContext` | `core/models.py` | Ephemeral retrieval payload passed to answer generation |

Validated facts form a temporal property graph:

```text
(Entity)-[ValidatedFact: RELATION {valid_from, valid_to, attributes}]->(Entity)
```

Facts are superseded or closed with `valid_to`; they are not deleted from the
truth store. Rejections are also stored so safety decisions remain auditable.

## Write Path

```text
raw text + scope
  |
  v
Interaction
  |
  v
LLM extractor / structured SDK call
  |
  v
CandidateFact[]
  |
  v
GroundingOperator
  |
  +-- ConstraintValidator checks candidate against current KnowledgeState
  |
  +-- APPROVED / SUPERSEDED
  |     -> ValidatedFact persisted
  |     -> active Neo4j projection updated
  |
  +-- REJECTED
        -> RejectionRecord persisted with constraint/reason
```

The grounding operator is implemented in `src/grounded_memory/core/grounding.py`.
It performs duplicate checks, constraint validation, supersession, and promotion
from candidate fact to validated fact.

The healthcare agent adds one extra domain step after grounding:

```text
DISCONTINUED or hold event
  -> close matching active PRESCRIBED facts
  -> remove closed prescriptions from the active Neo4j projection
```

That lifecycle behavior is implemented in
`src/grounded_memory/adapters/healthcare/lifecycle.py`.

## Storage Architecture

The demo target is the hybrid backend:

```python
Memory(
    adapter="healthcare",
    storage_backend="postgres_hybrid",
    require_scope=True,
)
```

Supported storage modes:

| Backend | Source of Truth | Graph Projection | Use |
| --- | --- | --- | --- |
| `memory` | In-process `MemoryStore` | None | Fast local/unit tests |
| `hybrid` | In-process `MemoryStore` | Neo4j | Graph demos without Postgres durability |
| `postgres` | PostgreSQL | None | Durable bitemporal/audit store |
| `postgres_hybrid` | PostgreSQL plus rehydrated in-process cache | Neo4j active projection | Thesis demo target |

PostgreSQL stores interactions, entities, validated facts, and rejection records
with scope fields and temporal metadata. `PostgresHybridMemoryStore` rehydrates
from PostgreSQL on initialization and can rebuild the Neo4j active projection
from the durable facts.

Neo4j is not the durable source of truth. It is an active graph projection used
for current-time traversal. Superseded or closed facts are removed from Neo4j,
while their historical records remain queryable through the primary store.

## Scope Model

The SDK can enforce scoped reads and writes with `require_scope=True`.
The normalized scope envelope includes:

```text
tenant_id, app_id, user_id, agent_id, run_id, space_type, scope_id
```

Healthcare extraction copies these fields into interactions, entities, and fact
attributes so retrieval and backend inspection can isolate one demo run from
another.

## Healthcare Adapter

The healthcare adapter is the thesis demonstration domain.

Main modules:

| Module | Responsibility |
| --- | --- |
| `adapters/healthcare/extractor.py` | LLM-backed clinical extraction into patient, medication, allergy, condition entities and candidate facts |
| `adapters/healthcare/constraints.py` | Allergy conflict, drug interaction, therapeutic duplication, and temporal checks |
| `adapters/healthcare/knowledge.py` | Mock drug knowledge base and normalization helpers |
| `adapters/healthcare/agent.py` | Interaction processing pipeline for healthcare |
| `adapters/healthcare/lifecycle.py` | Medication lifecycle closure for discontinuation/hold events |
| `adapters/healthcare/retrieval.py` | Healthcare query planning, entity resolution, and clinical context views |

Medication prescription attributes are standardized as:

```text
medication_name, normalized_name, dosage, frequency, route,
action, order_status, source_text
```

The demo knowledge base is intentionally lightweight and should be described as
a mock/synthetic drug knowledge base in thesis material.

## Healthcare Retrieval

Generic graph retrieval still exists in `src/grounded_memory/retrieval/graph.py`.
Healthcare retrieval adds a domain-specific layer on top.

```text
natural-language query
  |
  v
HealthcareRetrievalPlanner
  -> HealthcareQueryPlan
       patient_name
       patient_identifier
       requested_categories
       medication_names
       allergy_names
       safety_focus
       as_of
       ambiguity flags
  |
  v
resolver
  1. canonical patient identifier / MRN
  2. exact scoped patient name
  3. exact scoped clinical entity names
  4. fuzzy patient-name fallback
  5. generic seed selection fallback
  |
  v
GraphRetriever / bitemporal store
  |
  v
HealthcareClinicalContext
```

Current-time queries use the active graph projection when Neo4j is available.
Historical `as_of` queries pass the timestamp to the retriever and use temporal
fact filtering from the primary store.

The healthcare context builder post-filters graph facts into medication
reconciliation views:

| View | Contents |
| --- | --- |
| `current_medications` | Active `PRESCRIBED` facts, excluding superseded and discontinued orders |
| `allergies` | Active `HAS_ALLERGY` facts |
| `safety_alerts` | Rejected candidates and warning/error constraint evidence |
| `history` | Prescription/discontinuation timeline including superseded facts |

Final answer generation for the healthcare demo should use only the serialized
`HealthcareClinicalContext.to_dict()` / `AnswerContext` payload.

## Thesis Demo Flow

Canonical healthcare demo:

```bash
make services-up
PYTHONPATH=src python demos/demo_bitemporal.py
```

The script ingests:

1. Patient identity with MRN and allergy.
2. Baseline medication.
3. Dose adjustment that supersedes the old dose.
4. Warfarin baseline medication.
5. Unsafe Amiodarone attempt rejected by major interaction.
6. Unsafe Penicillin attempt rejected by allergy conflict.
7. Lisinopril discontinuation that closes the active prescription.
8. Current retrieval with active meds and allergies.
9. Historical retrieval with the previous dose visible at the as-of time.
10. Final answer generated strictly from structured context JSON.

Backend smoke for the demo stack:

```bash
make services-up
make smoke-healthcare-backends
```

## Testing Strategy

Run tests/ with the project environment:

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

Important healthcare tests:

| Test file | Evidence |
| --- | --- |
| `tests/test_healthcare_reconciliation.py` | Constraint behavior: identity, allergy conflict, interactions, supersession |
| `tests/test_healthcare_retrieval.py` | Retrieval planning/resolution, no hardcoded seed IDs, discontinuation closure, historical as-of retrieval |
| `scripts/healthcare_backend_smoke.py` | Optional live Postgres+Neo4j smoke behind `make smoke-healthcare-backends` |

The live demo depends on LLM and database availability; deterministic fake-LLM
tests protect the critical retrieval and lifecycle behavior from provider
variability.

## Intent Routing SDK Integration

The `Memory` facade now exposes intent classification directly:

```python
intent = memory.route("What are Jane Doe's current medications?")
# -> UserIntent(action=RECALL, confidence=0.85, ...)

result = memory.process("Jane Doe is allergic to penicillin")
# -> auto-routes to memory.add() because action=REMEMBER
```

- `route(query)` returns a `UserIntent` via the configured `BaseIntentRouter`.
- `process(text)` auto-routes: `REMEMBER` → `add()`, read intents → `search()`,
  `UNKNOWN` → tries search first, then add if empty.

The default router is `HybridIntentRouter` (keyword fast-path + LLM fallback).
Domain adapters can register extra keyword patterns via
`KeywordIntentRouter.register_patterns(category, patterns)` without subclassing.

## Retrieval Improvements

- **Pluggable query hints**: `GraphRetriever` accepts an optional
  `QueryHintRegistry` so domain adapters can register safety-critical lexical
  cues (e.g., healthcare "allergy") without modifying core code.
- **Graceful Neo4j fallback**: if the Neo4j projection fails during retrieval,
  the system logs a warning and automatically falls back to the in-memory
  NetworkX path instead of raising `RuntimeError`.
- **Weighted traversal deduplication**: `_retrieve_weighted` now tracks
  `visited_facts` to prevent the same fact from being scored multiple times via
  different graph paths, eliminating exponential expansion on cyclic graphs.
- **Smart seed fallback**: when lexical seed selection finds no match, entities
  are ranked by active-fact density rather than returned in arbitrary order.

## Current Boundaries

- The healthcare drug knowledge base is a demo/mock KB, not a clinical-grade
  drug database.
- Neo4j is an active projection, not the audit source.
- Retrieval planning has deterministic fallbacks, but the intended demo path is
  LLM-backed query understanding and LLM-backed clinical extraction.
- The architecture currently prioritizes medication reconciliation; broader
  clinical workflows need additional constraints, schemas, and evaluation.
