# Grounded Memory Middleware Architecture

## Purpose

Grounded Memory is designed as a **memory middleware service**. It is not tied to:

- one specific LLM provider,
- one specific graph database,
- one specific SQL/NoSQL backend,
- one specific front-end or agent framework.

The system acts as a governed memory layer that any application can call.

---

## Architectural Position

Grounded Memory sits between application events and model responses:

1. receives interaction text and metadata,
2. extracts candidate knowledge,
3. validates facts against constraints,
4. stores only accepted facts,
5. retrieves structured context for downstream reasoning.

This separation allows product teams to swap model, UI, or database technology without rewriting memory logic.

---

## Decoupling Guarantees

### 1) LLM-Agnostic Runtime

The middleware depends on an OpenAI-compatible request contract, not on one model vendor.

- Extraction/generation is configured through `LLMConfig`.
- Provider switching is done via environment configuration.
- Domain logic and storage semantics remain unchanged when model changes.

### 2) Storage-Agnostic Truth Layer

The logical truth model (Interaction, Entity, CandidateFact, ValidatedFact, Constraint, AnswerContext) is independent from physical persistence.

- `MemoryStore` provides in-memory canonical behavior.
- `PostgresStore` can persist the same model with temporal semantics.
- Additional SQL/NoSQL adapters can implement the same logical operations.

### 3) Graph Engine-Agnostic Retrieval Layer

Graph retrieval is built around behavior contracts, not hard-coding one graph engine.

- `GraphRetriever` consumes a graph-capable memory interface.
- In-memory traversal uses NetworkX-style graph construction.
- Neo4j-backed traversal is optional.
- Additional adapters (for example Kùzu) can be integrated without changing retrieval strategy semantics.

### 4) Front-End / Application-Agnostic API

The middleware treats all clients as event producers and query consumers.

- UI apps, agents, workflows, and tools all call the same memory operations.
- No UI state or product-specific components are embedded in core memory modules.

---

## Functional Responsibilities

### A) Ingestion Pipeline

Input: raw interaction text + actor metadata.

Output:

- interaction log record,
- approved validated facts,
- rejection audit entries for blocked facts.

Key properties:

- write-time governance,
- deterministic persistence semantics,
- provenance links from facts to source interactions.

### B) Governance Pipeline

Candidate facts are evaluated by a constraint validator before persistence.

- hard constraints can block writes,
- shadow/proposed constraints can observe without blocking,
- replay and promotion enable dynamic governance lifecycle.

### C) Retrieval Pipeline

Input: query + seed entities.

Output: `AnswerContext` with facts, entities, and retrieval metadata.

Strategies supported:

- breadth-first,
- weighted,
- safety-priority.

Domain behavior can be tuned with relationship presets (generic vs healthcare)
without changing retrieval APIs. Healthcare adds an adapter-level retrieval
planner and clinical context builder on top of the generic graph retriever.

---

## Portability Model

### What can change without memory redesign

- LLM model/provider endpoint,
- graph engine (NetworkX, Neo4j, future Kùzu adapter),
- relational backend (in-memory, PostgreSQL, SQLite adapter),
- client application (CLI, web, agent runtime, microservice caller).

### What must stay stable

- six-object memory taxonomy,
- grounding contract (propose -> validate -> persist),
- temporal/supersession semantics,
- explainability and rejection audit trail.

---

## Suggested Deployment Modes

### Embedded Library Mode

Application imports the package and calls memory objects in-process.

### Sidecar / Service Mode

Memory layer runs as a separate process with service endpoints (REST/gRPC) and independent scaling.

### Shared Platform Mode

Multiple applications call one memory service with tenant/session isolation.

---

## Backend Adapter Roadmap

To match mature memory systems that swap engines, keep adapters explicit:

- Truth-store adapters: `MemoryStore`, `PostgresStore`, `SQLiteStore` (planned).
- Graph adapters: `NetworkX` fallback, `Neo4jStore`, `KuzuStore` (planned).
- Embedding/vector adapters can be added independently from governance and temporal facts.

The core rule is: adapters conform to middleware contracts; core logic does not branch on product-specific behavior.

---

## Documentation Map

- `README.md`: project overview and quick start.
- `docs/architecture/ARCHITECTURE.md`: full conceptual and implementation detail.
- `docs/architecture/HYBRID_ARCHITECTURE.md`: storage projection and graph retrieval internals.
- `docs/architecture/MIDDLEWARE_ARCHITECTURE.md`: decoupling contract and portability model (this document).
