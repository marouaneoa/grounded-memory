# GMem — System Definition

> Formal definitions, actors, components, and workflows for the Grounded Memory
> runtime, including concrete mappings for the healthcare medication reconciliation
> use case.

---

## 1. System Identity and Purpose

**Definition 1 (GMem).** GMem is a **correctness-first memory runtime for LLM
agents**. Its core thesis is that candidate knowledge proposed by a language model
must be governed _before_ it becomes durable memory, and retrieval must answer from
validated state rather than unstructured text.

GMem contrasts with traditional RAG systems on five structural axes:

| Property | Traditional RAG | GMem |
|---|---|---|
| Write path | Store raw text chunks | Extract structured tuples → validate constraints → persist accepted facts |
| Read path | Retrieve likely text snippets | Retrieve structured, governed context from validated knowledge graph |
| Governance | None at write time; filtering only at read time | Constraint validation _before_ any fact enters durable memory |
| Temporal model | Snapshot or none | Bitemporal: valid-time (`valid_from`, `valid_to`) and record-time (interaction timestamps) |
| Mutability | Overwrite or append-only | Supersession: old facts are closed with `valid_to`, never deleted |
| Explainability | Opaque similarity scores | Full rejection audit trail + fact provenance with source attribution |

---

## 2. System Boundary

### Definition 2 (Narrow Boundary).

The **narrow system boundary** defines GMem as a standalone memory middleware
service. It receives observations, extracts and validates facts, stores them, and
retrieves structured context on demand. The host LLM agent is an _external actor_
that calls GMem through its SDK or REST API.

```
                  Narrow Boundary
          ┌──────────────────────────────────────────┐
          │  GMem                                    │
 Host ────┤                                          │
 Agent    │  Write: Ingestion → Governance → Storage  │
  (𝒜_H)   │                                          │
          │  Read:  Query ← Retrieval ← Storage      │
          └──────────────────────────┬───────────────┘
                                     │
                               Primary Store
                              Graph Projection
```

Under the narrow boundary, GMem is **passive**: it does not initiate
communication outward and only responds to calls from the Host Agent.

### Definition 3 (Wide Boundary).

The **wide system boundary** defines GMem and its Host Agent as a single
**grounded LLM agent system**. The Host Agent uses GMem's memory context to
ground its own reasoning, forming a feedback loop between memory and action.

```
          Wide Boundary
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  End     Host Agent (𝒜_H) ◄── context ── GMem          │
│  User ──►         │                       │             │
│   (𝕌)             │── observation ───────►│             │
│                   │                       │             │
│                   ◄── memory prompt ──────│             │
│                   │                       │             │
│             Response/Utterance            │             │
│                   │                    Storage           │
└───────────────────┼─────────────────────────────────────┘
                    │
              Response to 𝕌
```

Both views are valid. The narrow view specifies GMem's contracts and
responsibilities in isolation. The wide view specifies the behavior that the
End User experiences: a grounded agent whose responses are traceable to validated
memory rather than hallucination.

---

## 3. Actors and Roles

### Definition 4 (End User, 𝕌).

The **End User** is the human who produces natural-language observations and
poses natural-language queries. The End User never interacts with GMem directly;
all inputs and queries are mediated by the Host Agent.

An End User produces two categories of input:

1. **Observations**: statements about the world (e.g., "Patient Jane Doe is
   prescribed Lisinopril 10mg daily").
2. **Queries**: questions about remembered state (e.g., "What are patient Jane
   Doe's current medications?").

### Definition 5 (Host Agent, 𝒜_H).

The **Host Agent** is the external LLM-powered agent that calls GMem's SDK or
REST API. It serves two roles:

1. **Producer**: forwards textual observations (from the End User, external tools,
   or its own reasoning) to GMem's write path via `add()` or `add_fact()`.
2. **Consumer**: queries GMem's read path via `search()` or `build_context()` and
   uses the returned structured context to ground its responses and reasoning.

The Host Agent is the _sole_ caller of GMem. GMem does not initiate
communication outward.

### Definition 6 (Application Developer, 𝒟).

The **Application Developer** configures GMem at deployment time: selects the
adapter domain (generic, healthcare, etc.), chooses the storage backend
(memory, postgres, hybrid), sets retrieval parameters, and integrates GMem into
the Host Agent's runtime. 𝒟 determines which LLM provider GMem uses for
extraction and query planning.

### Definition 7 (Governance Administrator, 𝒢).

The **Governance Administrator** defines, promotes, and retires constraint
policies. 𝒢 may coincide with 𝒟. 𝒢 registers constraint evaluators, sets
their lifecycle state (`proposed`, `shadow`, `active`, `deprecated`), and
reviews constraint discovery proposals generated by the
Constraint Discovery Engine (§5.6).

---

## 4. Healthcare Use Case: Concrete Actor Mapping

This section maps the abstract actors from §3 to specific roles in the
healthcare medication reconciliation domain.

### 4.1 Actor Mapping Table

| Abstract Actor | Healthcare Role | Actions via GMem |
|---|---|---|
| End User (𝕌) | **Prescribing Physician** | Issues medication orders, dosage changes, and discontinuation orders (via CPOE or natural language) |
| End User (𝕌) | **Pharmacist** | Reviews medication orders; queries for drug interactions, allergy conflicts, therapeutic duplication |
| End User (𝕌) | **Nurse** | Transcribes administration records; queries current medication list at shift change |
| End User (𝕌) | **Clinical Decision Support System** | Automated admission reconciliation; queries comprehensive patient context for safety alerts |
| Host Agent (𝒜_H) | **Clinical Memory Agent** | The LLM co-pilot that receives clinician inputs, calls GMem to ingest and retrieve, and produces grounded clinical summaries |
| Application Developer (𝒟) | **Health IT Engineer** | Configures the healthcare adapter, constraint policies, LLM provider, and storage backends |
| Governance Administrator (𝒢) | **Clinical Informaticist / Pharmacy Director** | Defines which constraints are active vs. shadow; reviews drug interaction rules; promotes discovered constraints |

### 4.2 Who Does What: Write Side

The following table specifies which actor provides which input at each stage of
the write path in the healthcare use case:

| Step | Actor | Action | GMem Component |
|---|---|---|---|
| 1 | Prescribing Physician (via 𝕌) | Writes a medication order in CPOE or describes it in natural language | — |
| 2 | Clinical Memory Agent (𝒜_H) | Receives the observation text; calls `memory.add(text, source="clinician", ...)` | Memory Facade |
| 3 | GMem (Extraction Engine) | Extracts structured clinical data using the LLM with `ClinicalExtractionResult` schema | HealthcareDatabaseExtractor |
| 4 | GMem (Entity Resolution) | Resolves patient identity by MRN/canonical_id; resolves medication/allergy entities by name | HealthcareDatabaseExtractor |
| 5 | GMem (Grounding Operator) | Validates each candidate fact against healthcare constraints (allergy conflict, drug interaction, therapeutic duplication, etc.) | GroundingOperator + ConstraintValidator |
| 6 | GMem (Lifecycle Manager) | For DISCONTINUED actions: closes matching active PRESCRIBED facts, updates temporal boundaries | Healthcare Lifecycle |
| 7 | GMem (Storage) | Persists ValidatedFacts and RejectionRecords with full scope and provenance | Primary Store + Graph Projection |

**Key insight**: The End User (clinician) never directly calls GMem. The Clinical
Memory Agent mediates all interactions, transforming unstructured clinical
observations into GMem API calls and returning grounded context to the clinician.

### 4.3 Who Does What: Read Side

| Step | Actor | Action | GMem Component |
|---|---|---|---|
| 1 | Pharmacist or Physician (via 𝕌) | Asks a clinical question: "What are Jane Doe's current medications?" | — |
| 2 | Clinical Memory Agent (𝒜_H) | Calls `memory.search(query, ...)` or `context_builder.build(query, ...)` | Memory Facade / HealthcareContextBuilder |
| 3 | GMem (Query Planner) | Parses query into `HealthcareQueryPlan`: patient identity, requested categories, temporal scope, safety focus | HealthcareRetrievalPlanner |
| 4 | GMem (Entity Resolver) | Resolves patient by canonical_id → exact name → fuzzy match fallback; resolves medication/allergy entities | HealthcareRetrievalPlanner |
| 5 | GMem (Graph Retriever) | Traverses knowledge graph from seed entities with strategy (breadth-first, weighted, safety-priority) and reranking | GraphRetriever |
| 6 | GMem (Context Builder) | Post-processes raw facts into clinical views: current_medications, allergies, safety_alerts, history | HealthcareContextBuilder |
| 7 | Clinical Memory Agent (𝒜_H) | Receives structured `HealthcareClinicalContext`; generates grounded response to clinician | — |

**Key insight**: The retrieved context is structured (not raw text), so the Host
Agent can produce answers traceable to validated facts rather than hallucinated
content.

### 4.4 What is Agentic vs. Deterministic in Healthcare

| Component | Agentic (LLM) | Deterministic (Algorithmic) |
|---|---|---|
| Clinical fact extraction | ✓ (`HealthcareLLMExtractor` with `ClinicalExtractionResult` schema) | |
| Query plan generation | ✓ (`HealthcareRetrievalPlanner` with LLM-backed `HealthcareQueryPlan`) | Falls back to regex-based pattern extraction |
| Constraint validation | | ✓ (Allergy conflict, drug interaction, therapeutic duplication — rule-based checks) |
| Conflict resolution / supersession | | ✓ (strategy-based: confidence, recency, source priority, composite) |
| Medication lifecycle closure | | ✓ (DISCONTINUED → close matching PRESCRIBED facts) |
| Graph retrieval + reranking | | ✓ (query-aware BFS/weighted/safety-priority traversal with calibrated relation scoring, relevance-gated combination, and two-layer diversity penalty) |
| Entity identity resolution | | ✓ (UUID5 from deterministic uniqueness key) |
| Tuple normalization | | ✓ (canonical key/value normalization) |

The **only agentic components** are the extraction engine (turning raw text into
structured candidate facts) and the query planner (understanding natural-language
queries). All governance, lifecycle, retrieval, and storage operations are
fully deterministic.

### 4.5 Intent Routing Layer

GMem introduces a **domain-agnostic intent routing layer** that sits between the
Host Agent's natural-language utterances and the concrete memory operations.
Rather than hardcoding domain keywords into the core system, the router
classifies every user input into one of five generic cognitive actions:

| Action | Cognitive Meaning | Typical Memory Operation |
|---|---|---|
| `REMEMBER` | The user is stating a new fact or updating existing information. | `memory.add()` |
| `RECALL` | The user is asking about a specific entity or fact. | `retrieve_current_state()` |
| `FIND_RELATED` | The user is asking which entities share a property. | `find_patients_by_shared_entity()` |
| `EXPLAIN` | The user wants a summary or explanation. | `generate_grounded_answer()` |
| `UNKNOWN` | The intent cannot be determined; caller should prompt for clarification. | — |

The router is implemented as a protocol (`BaseIntentRouter`) with three
concrete strategies:

1. **KeywordIntentRouter** — deterministic, zero-latency keyword matching using
   *cognitive* patterns ("what", "who", "summarize") rather than domain terms.
2. **LLMIntentRouter** — generic LLM-backed classifier with a domain-agnostic
   system prompt; falls back to `UNKNOWN` on failure.
3. **HybridIntentRouter** — fast-path keyword routing for clear cases;
   falls back to the LLM router only when keyword confidence is below a threshold.

**SDK integration.** The `Memory` facade exposes two new methods:

- `Memory.route(query) -> UserIntent` — explicit intent classification.
- `Memory.process(text) -> dict` — auto-routing convenience method that
  dispatches to `add()` for `REMEMBER`, `search()` for read intents, and
  attempts search-then-add for `UNKNOWN`.

**Pluggable patterns.** `KeywordIntentRouter` keeps domain-agnostic cognitive
patterns by default. Adapters register extra patterns at runtime via
`register_patterns(category, patterns, prepend=False)` without subclassing:

```python
router = KeywordIntentRouter()
router.register_patterns("remember", [r"\bdiagnosed\b", r"\bprescribed\b"])
router.register_patterns("recall", [r"\ballergic\b", r"\btaking\b"])
```

The healthcare adapter registers its clinical patterns automatically when
`adapter="healthcare"` is selected.

**Key insight**: Because the intent layer is generic and pattern registration
is runtime-pluggable, the same router works across healthcare, legal, finance,
or any other adapter. The adapter provides the concrete mapping from generic
actions to domain-specific service calls, keeping the core architecture free
of domain coupling.

---

## 5. Components (Formal Definitions)

### 5.1 Interaction Logger

**Definition 8 (Interaction Logger).** The **Interaction Logger** records every
raw observation as an immutable `Interaction` object containing the original
text, actor type (`user`, `agent`, `tool`, `system`), scope metadata
(`tenant_id`, `app_id`, `user_id`, `agent_id`, `run_id`, `space_type`), and a
timestamp. Interactions form an append-only audit trail: they are never modified
or deleted.

Implementation: `core/models.py` (`Interaction`), `core/store.py` (persistence).

### 5.2 Extraction Engine

**Definition 9 (Extraction Engine).** The **Extraction Engine** transforms raw
text into one or more `CandidateFact` proposals. It operates in two modes:

1. **Agentic mode**: uses an LLM (via `SyncLLMClient`) with a domain-specific
   extraction prompt and structured output schema (e.g.,
   `ClinicalExtractionResult` for healthcare, `GenericExtractionResult` for the
   generic adapter).
2. **Structured mode**: the Host Agent calls `add_fact()` directly, bypassing
   LLM extraction and providing subject, relation, and object/value explicitly.

Each extracted fact carries a **disposition**:
- `CAPTURE`: accept as new knowledge.
- `RETIRE`: close matching active facts (e.g., "no longer prefers X").
- `REFINE`: update existing knowledge (handled as supersession).
- `PASS`: skip (not relevant to memory).

The Extraction Engine is the only component that invokes the LLM on the write
path. It operates under the adapter layer's control.

Implementation: `adapters/generic_agent.py` (`GenericMemoryAgent`),
`adapters/healthcare/extractor.py` (`HealthcareDatabaseExtractor`),
`llm/extractor.py` (`LLMFactExtractor`).

### 5.3 Grounding Operator (Γ)

**Definition 10 (Grounding Operator, Γ).** The **Grounding Operator** is the
core write-path gatekeeper. It receives a `CandidateFact` 𝑓̂ and a knowledge
state 𝒦, and produces a `GroundingResult`:

```
Γ(𝑓̂, 𝒦) → GroundingResult ∈ {APPROVED, REJECTED, SUPERSEDED, DUPLICATE}
```

The grounding pipeline executes these steps in order:

1. **Duplicate check**: if 𝑓̂ is semantically identical to an existing
   `ValidatedFact` in 𝒦 (same subject, relation, object/value, and normalized
   attributes), return `DUPLICATE`.
2. **Constraint validation**: evaluate 𝑓̂ against all applicable
   `Constraint` objects registered in the `ConstraintValidator`.
3. **Rejection**: if any `ACTIVE` constraint produces an `error`-severity
   violation, persist a `RejectionRecord` and return `REJECTED`.
4. **Conflict resolution**: if 𝑓̂ conflicts with existing facts, the
   `ConflictResolver` determines whether 𝑓̂ supersedes the existing fact or
   is superseded by it, using a pluggable strategy (confidence, recency, source
   priority, or composite).
5. **Supersession**: for superseded existing facts, set their `valid_to` to
   the current timestamp and `superseded_by` to the new fact's ID.
6. **Acceptance**: persist 𝑓̂ as a `ValidatedFact` with `valid_from` set to
   the current timestamp and `valid_to = None`. Return `APPROVED`.

**Axiom G1 (Governance First).** No candidate fact enters durable memory without
passing through Γ. This is the system's central invariant.

**Axiom G2 (Non-Destruction).** No `ValidatedFact` is ever physically deleted.
Facts are closed by setting `valid_to` and `superseded_by`. The complete
bitemporal history of every fact is always reconstructable.

**Axiom G3 (Rejection Audit).** Every rejected candidate produces a
`RejectionRecord` containing the constraint ID, violation reason, severity,
and suggested alternatives. Rejection records are immutable and queryable.

Implementation: `core/grounding.py` (`GroundingOperator`),
`core/conflict_resolution.py` (`ConflictResolver`),
`core/tuple_normalization.py` (canonical matching).

### 5.4 Constraint Validator

**Definition 11 (Constraint Validator).** The **Constraint Validator** evaluates
each `CandidateFact` against registered `Constraint` objects before it can become
a `ValidatedFact`. Constraints follow a governance lifecycle:

| Lifecycle State | Effect |
|---|---|
| `PROPOSED` | Observed but never blocks writes |
| `SHADOW` | Observed and recorded; warns but never blocks |
| `ACTIVE` | Fully enforced; blocks writes on `error`-severity violations |
| `DEPRECATED` | Ignored entirely |

Constraint sources: `HUMAN` (manually configured) or `AGENT` (autonomously
discovered).

**Axiom CV1 (Shadow Safety).** Constraints in `PROPOSED` or `SHADOW` state
never prevent a candidate fact from being accepted. They produce observations
that feed the Constraint Discovery Engine but do not alter the knowledge state.

Implementation: `core/constraints.py` (`ConstraintValidator`),
`adapters/seeds.py` (`SeedConstraintEvaluator`),
`adapters/healthcare/constraints.py` (`YamlConstraintEvaluator`).

### 5.5 Conflict Resolver

**Definition 12 (Conflict Resolver).** The **Conflict Resolver** determines
whether a new `CandidateFact` supersedes an existing `ValidatedFact` when they
occupy the same semantic slot. Strategies:

| Strategy | Decision Rule |
|---|---|
| `CONFIDENCE_WINS` | Higher confidence score wins |
| `RECENCY_WINS` | More recent timestamp wins |
| `SOURCE_PRIORITY` | Source priority: `system > tool > agent > user` |
| `COMPOSITE` (default) | Weighted blend: confidence 0.4, recency 0.3, source 0.3 |

The losing fact is closed (superseded), not deleted.

Implementation: `core/conflict_resolution.py`.

### 5.6 Constraint Discovery Engine

**Definition 13 (Constraint Discovery Engine).** The **Constraint Discovery
Engine** autonomously mines accumulated validation signals (rejection records,
shadow observations) to synthesize new constraint proposals. It groups signals
by relation type, computes rejection rates for missing-value and
missing-attribute patterns, and generates `SeedConstraintEvaluator` candidates
with confidence scores. Discovered constraints start in `SHADOW` lifecycle
state and must be promoted to `ACTIVE` by the Governance Administrator (𝒢)
after review.

Implementation: `adapters/discovery.py` (`ConstraintSeedDiscoverer`).

### 5.7 Primary Store

**Definition 14 (Primary Store).** The **Primary Store** is the durable source
of truth for all `Interaction`, `Entity`, `ValidatedFact`, and
`RejectionRecord` objects. It supports four backend implementations:

| Backend | Storage | Graph Projection | Use Case |
|---|---|---|---|
| `memory` | In-process `MemoryStore` | None | Fast local testing, prototyping |
| `hybrid` | In-process `MemoryStore` | Neo4j | Graph demos without PostgreSQL durability |
| `postgres` | PostgreSQL | None | Durable bitemporal audit store |
| `postgres_hybrid` | PostgreSQL + rehydrated in-process cache | Neo4j | Production: durable + fast graph traversal |

**Axiom PS1 (Source of Truth).** PostgreSQL is the source of truth. Neo4j is a
rebuildable active projection. Neo4j must never be treated as the sole copy of
any fact.

**Axiom PS2 (Rebuildability).** The Neo4j active projection can be fully rebuilt
from the PostgreSQL primary store at any time.

Implementation: `core/store.py` (`MemoryStore`), `core/postgres_store.py`
(`PostgresKnowledgeStore`), `core/postgres_hybrid_store.py`
(`PostgresHybridMemoryStore`), `core/hybrid_store.py`
(`HybridMemoryStore`).

### 5.8 Active Graph Projection

**Definition 15 (Active Graph Projection).** The **Active Graph Projection** is
a read-optimized view of currently-active `ValidatedFact` objects stored in
Neo4j. It contains only facts where `valid_to = None` (currently active) and
is optimized for graph traversal queries. Superseded or closed facts are
removed from the projection, while their historical records remain queryable
through the primary store.

The projection is updated synchronously on every write for hybrid backends.
Historical `as_of` queries bypass the projection and use bitemporal filtering
from the primary store.

Implementation: `core/neo4j_store.py`.

### 5.9 Query Planner

**Definition 16 (Query Planner).** The **Query Planner** transforms a
natural-language query into a structured query plan. In the generic adapter,
this is a deterministic seed-selection process. In the healthcare adapter, the
`HealthcareRetrievalPlanner` uses LLM-backed extraction to produce a
`HealthcareQueryPlan` containing:

- `patient_name`: extracted or inferred patient identifier
- `patient_identifier`: MRN or canonical ID
- `requested_categories`: types of clinical data requested (medications,
  allergies, conditions, safety alerts)
- `medication_names`: specific drug names mentioned
- `safety_focus`: whether the query emphasizes safety-critical information
- `as_of`: temporal reference for historical queries
- Ambiguity flags

When the LLM is unavailable, the planner falls back to deterministic regex-based
pattern extraction for patient names, identifiers, and temporal expressions.

Implementation: `adapters/healthcare/retrieval.py`
(`HealthcareRetrievalPlanner`), `retrieval/graph.py`
(`GraphRetriever.select_seed_entities`).

### 5.10 Graph Retriever

**Definition 17 (Graph Retriever).** The **Graph Retriever** traverses the
knowledge graph from seed entities to collect relevant facts. It supports three
strategies:

| Strategy | Behavior |
|---|---|---|
| `BREADTH_FIRST` | BFS from seed entities with query-aware fact sorting and edge pruning (irrelevant non-safety edges beyond the first hop are not expanded) |
| `WEIGHTED` | Builds a weighted `MultiDiGraph`; explores with cumulative weight × carry multiplier; query relevance breaks ties among equal-weight paths |
| `SAFETY_PRIORITY` | Two-phase: first all safety-critical facts, then remaining facts; query-aware pruning skips irrelevant non-safety facts at deeper hops |

After traversal, the retriever applies:
- **Temporal filtering**: `at_time` parameter restricts to facts active at a
  point in time; `lookback_days` restricts to facts with `valid_from` within
  a window.
- **Reranking**: five weighted signals — relation weight, relevance, recency,
  safety, and profile match — with calibrated relation scaling and
  relevance-gated combination. Relation weights are normalized against the
  maximum configured weight so high-importance relationships separate clearly
  from low-importance ones. Relevance modulates the structural score;
  safety-critical facts receive a direct bypass so they surface even with
  weak lexical overlap.
- **Diversity penalty**: two-layer deduplication. A semantic-key penalty
  (0.08 × count) deduplicates exact semantic duplicates, and a token-overlap
  redundancy penalty (0.10 × max overlap ratio) suppresses near-duplicate
  facts that share most of their content but differ in minor attributes.

**Query-hint registry.** The retriever accepts an optional `QueryHintRegistry`
that maps lexical cues to retrieval-signal boosts. The default registry is
domain-agnostic; adapters register domain-specific hints at runtime (e.g.,
healthcare registers "allergy" and "contraindicated" as safety cues).

**Graceful Neo4j fallback.** When Neo4j is configured, current-time queries are
delegated to the native graph projection for performance. If Neo4j fails
(transient connection error, query timeout), the retriever logs a warning and
automatically falls back to the in-memory NetworkX path instead of surfacing a
hard error to the caller.

**Weighted traversal deduplication.** The `WEIGHTED` strategy now tracks a
`visited_facts` set during recursive exploration. This prevents the same fact
from being scored multiple times via different graph paths, eliminating
exponential expansion on cyclic knowledge graphs.

**Smart seed fallback.** When lexical seed selection finds no entity name
matches, the fallback ranks all entities by active-fact density (number of
active facts connected to the entity) and selects the top-N, rather than
returning entities in arbitrary store order.

**Axiom GR1 (Deterministic Retrieval).** Given the same knowledge state and
query, the Graph Retriever produces the same result set (modulo LLM-assisted
seed selection when used).

Implementation: `retrieval/graph.py` (`GraphRetriever`).

### 5.11 Context Builder

**Definition 18 (Context Builder).** The **Context Builder** assembles the
final retrieval output into a structured `AnswerContext` (generic) or
domain-specific view (e.g., `HealthcareClinicalContext`). It is never
persisted; it is an ephemeral payload delivered to the Host Agent.

In the generic adapter, `Memory.build_context()` returns an `AnswerContext`
containing the query, seed entities, facts, entities, and retrieval metadata.

In the healthcare adapter, `HealthcareContextBuilder` produces a
`HealthcareClinicalContext` with four domain-specific views:

| View | Contents |
|---|---|
| `current_medications` | Active `PRESCRIBED` facts excluding superseded and discontinued orders |
| `allergies` | Active `HAS_ALLERGY` facts |
| `safety_alerts` | All `RejectionRecord` entries for the patient's scope |
| `history` | Full bitemporal timeline of `PRESCRIBED` and `DISCONTINUED` facts |

Implementation: `memory.py` (`Memory.build_context`),
`adapters/healthcare/retrieval.py` (`HealthcareContextBuilder`,
`HealthcareClinicalContext`).

### 5.12 Adapter Layer

**Definition 19 (Adapter Layer).** The **Adapter Layer** provides
domain-specific wiring for extraction, constraint governance, and retrieval. An
adapter is specified by an `AdapterSpec` that provides two extension points:

1. **Constraint configuration**: a callable that registers domain-specific
   constraint evaluators on the `ConstraintValidator`.
2. **Agent creation**: a callable that creates a domain-specific processing
   agent.

Built-in adapters: `generic` (default), `healthcare`, `engineering`, `finance`,
`legal`, `core`, `none`. Custom adapters can be registered at runtime.

The healthcare adapter is the richest: it provides clinical extraction
(`HealthcareDatabaseExtractor`), clinical constraints (`YamlConstraintEvaluator`
loaded from `configs/healthcare_constraints.yaml`), medication lifecycle
management (`apply_medication_lifecycle_after_grounding`), and healthcare-
specific retrieval (`HealthcareRetrievalPlanner`, `HealthcareContextBuilder`).

Implementation: `adapters/registry.py` (`AdapterSpec`, adapter registry),
`adapters/healthcare/__init__.py` (healthcare adapter wiring).

---

## 6. The Six-Object Model

### Definition 20 (Interaction).

An **Interaction** is an immutable event log entry recording a raw observation.
Fields: `id`, `raw_text`, `actor` (enum: `user`, `agent`, `tool`, `system`),
scope envelope (`tenant_id`, `app_id`, `user_id`, `agent_id`, `run_id`,
`space_type`), `timestamp`, and `metadata`.

**Axiom I1 (Immutability).** Interactions are frozen: once created, they cannot
be modified.

### Definition 21 (Entity).

An **Entity** is a symbolic node in the knowledge graph. Fields: `id` (UUID5,
deterministic from uniqueness key), `entity_type` (enum: `PATIENT`, `MEDICATION`,
`ALLERGY`, `CONDITION`, `PERSON`, etc.), `name`, `canonical_id`, `attributes`,
and `updated_at`.

**Axiom E1 (Deterministic Identity).** The same entity (same scope, type, and
name/canonical_id) always receives the same ID across runs, via UUID5 from a
semantic uniqueness key.

### Definition 22 (CandidateFact, 𝑓̂).

A **CandidateFact** 𝑓̂ is an untrusted tuple proposed by the Extraction Engine or
a structured SDK write. Fields: `source_interaction_id`, `subject_entity_id`,
`relation` (enum: 21 `RelationType` values), `object_entity_id` or `value`,
`confidence`, `attributes`, `status` (`pending`, `accepted`, `rejected`).

**Axiom CF1 (Transience).** A CandidateFact is a proposal, not knowledge. It
must pass through Γ before it can influence any retrieval result.

### Definition 23 (ValidatedFact, F★).

A **ValidatedFact** F★ is a system-approved knowledge tuple with bitemporal
boundaries. Fields: `subject_id`, `relation`, `object_id` or `value`,
`valid_from`, `valid_to`, `superseded_by`, `confidence`, `source_text`,
`embedding`, `source_metadata`.

**Axiom VF1 (Non-Deletion).** A ValidatedFact is never physically deleted. When
superseded, its `valid_to` is set to the superseding timestamp and
`superseded_by` is set to the new fact's ID.

**Axiom VF2 (Bitemporal Completeness).** At any point in time 𝑡, the set of
active facts is exactly {F★ | F★.valid_from ≤ 𝑡 ∧ (F★.valid_to = None ∨
F★.valid_to > 𝑡)}. This enables point-in-time historical queries via the `as_of`
parameter.

### Definition 24 (Constraint).

A **Constraint** is a declarative governance rule evaluated on write. Fields:
`constraint_type`, `applies_to_relations`, `condition`, `severity` (`error`,
`warning`, `info`).

**Axiom C1 (Governance Before Persistence).** Every CandidateFact must be
evaluated against all applicable active constraints before it can become a
ValidatedFact.

### Definition 25 (AnswerContext, X).

An **AnswerContext** X is an ephemeral retrieval payload. Fields: `query`,
`seed_entities`, `facts`, `entities`, `retrieval_metadata`. It is never
persisted.

**Axiom AC1 (Ephemerality).** An AnswerContext exists only for the duration of
a single read request. It is not stored and does not influence future writes.

The six objects form a temporal property graph:

```
(Entity)-[ValidatedFact: RELATION {valid_from, valid_to, attributes}]->(Entity)
```

Where `Entity` nodes are connected by `ValidatedFact` edges carrying temporal
and provenance metadata.

---

## 7. The Grounding Contract

**Definition 26 (Grounding Contract).** The **Grounding Contract** is the
system's central invariant:

> For every CandidateFact 𝑓̂ that enters durable memory as a ValidatedFact F★,
> there must exist a `GroundingResult` with decision in {APPROVED, SUPERSEDED}
> and a complete audit trail including: (1) all constraints evaluated, (2) all
> conflict resolution decisions, (3) all facts superseded by F★.

This contract has three corollaries:

1. **No ungoverned writes.** Every fact in memory has passed through the
   Grounding Operator and has a recorded decision.
2. **No silent failures.** Every rejected candidate produces an auditable
   `RejectionRecord` with the constraint ID, reason, and severity.
3. **Complete lineage.** Every ValidatedFact carries `source_interaction_id`
   linking it to the original observation, and `superseded_by` linking it to
   any fact that replaces it.

---

## 8. Agentic vs. Deterministic Boundary

### Definition 27 (Agentic Component).

A component is **agentic** if its behavior depends on an LLM call whose output
is non-deterministic. GMem has exactly two agentic components:

1. **Extraction Engine** (write path): transforms raw text into `CandidateFact`
   proposals using LLM-backed structured extraction.
2. **Query Planner** (read path, healthcare adapter only): transforms
   natural-language queries into `HealthcareQueryPlan` using LLM-backed
   understanding, with deterministic fallback.

All other components — Grounding Operator, Constraint Validator, Conflict
Resolver, Graph Retriever, Context Builder, Medication Lifecycle, Tuple
Normalization, Entity Identity, Storage — are **deterministic**: given the
same inputs and knowledge state, they always produce the same outputs.

### Definition 28 (Deterministic Component).

A component is **deterministic** if its behavior is fully algorithmic with no
LLM dependency. For deterministic components, the `MemoryStore` state plus the
input fully determines the output.

| Component | Type | Depends on LLM? |
|---|---|---|
| Extraction Engine (generic) | Agentic | Yes |
| Extraction Engine (healthcare) | Agentic | Yes |
| Query Planner (healthcare) | Agentic (with fallback) | Yes, with regex fallback |
| Grounding Operator (Γ) | Deterministic | No |
| Constraint Validator | Deterministic | No |
| Conflict Resolver | Deterministic | No |
| Constraint Discovery Engine | Deterministic (statistical) | No |
| Graph Retriever | Deterministic | No |
| Context Builder | Deterministic | No |
| Medication Lifecycle | Deterministic | No |
| Tuple Normalization | Deterministic | No |
| Entity Identity | Deterministic | No |
| Storage Operations | Deterministic | No |

---

## 9. Workflows

### 9.1 General Write Path

```
Input: raw text + scope envelope + source actor
  │
  ▼
Step 1. Memory Facade receives add(text, source, **scope)
  │
  ▼
Step 2. Interaction Logger creates an immutable Interaction record
  │
  ▼
Step 3. Extraction Engine (GenericMemoryAgent or HealthcareMemoryAgent)
        processes text:
  │        ├── LLM extraction: raw text → structured CandidateFact[]
  │        └── Heuristic fallback (generic only): regex-based extraction
  │
  ▼
Step 4. For each CandidateFact 𝑓̂:
  │   ├── If disposition = CAPTURE or REFINE:
  │   │     Entity Resolution: find_or_create_entity for subject and object
  │   │     Grounding Operator Γ(𝑓̂, 𝒦):
  │   │       ├── Duplicate check
  │   │       ├── Constraint validation
  │   │       ├── If rejected → RejectionRecord persisted → skip
  │   │       ├── Conflict resolution (supersession)
  │   │       └── ValidatedFact persisted
  │   ├── If disposition = RETIRE:
  │   │     Find active facts matching the retire pattern
  │   │     Close matching facts (set valid_to)
  │   └── If disposition = PASS: skip
  │
  ▼
Output: GroundingResult for each CandidateFact
```

### 9.2 General Read Path

```
Input: natural-language query + scope envelope + optional temporal parameters
  │
  ▼
Step 1. Memory Facade receives search(query, **scope)
  │
  ▼
Step 2. Scope Resolution: resolve tenant/app/user/agent/run/space
  │
  ▼
Step 3. Seed Entity Selection:
  │   ├── (Healthcare) HealthcareRetrievalPlanner → HealthcareQueryPlan
  │   │     → resolve patient/entity by MRN → name → fuzzy fallback
  │   └── (Generic) GraphRetriever.select_seed_entities(query)
  │         → lexical token matching against entity names
  │
  ▼
Step 4. Graph Retrieval:
  │   ├── Strategy selection (BREADTH_FIRST, WEIGHTED, SAFETY_PRIORITY)
  │   ├── Temporal filtering (at_time, lookback_days)
  │   ├── Neo4j active projection (for current-time queries)
  │   │     OR bitemporal filtering from primary store (for as_of queries)
  │   └── Reranking (calibrated relation weight, relevance-gated structural
  │       score, recency, safety bypass, profile match) + two-layer diversity
  │       penalty (semantic-key deduplication + token-overlap redundancy)
  │
  ▼
Step 5. Context Assembly:
  │   ├── (Generic) AnswerContext with seed entities, facts, entities
  │   └── (Healthcare) HealthcareClinicalContext with:
  │         ├── current_medications (active PRESCRIBED minus DISCONTINUED)
  │         ├── allergies (HAS_ALLERGY facts)
  │         ├── safety_alerts (RejectionRecords)
  │         └── history (bitemporal PRESCRIBED + DISCONTINUED timeline)
  │
  ▼
Step 6. Scope filter + threshold filter applied
  │
  ▼
Step 7. Lexical fallback if graph retrieval returns no results
  │
  ▼
Output: structured context to Host Agent
```

### 9.3 Healthcare Write Workflow (Step-by-Step)

This section concretely traces a clinical observation through the healthcare
write path.

**Scenario**: A physician orders "Start Amiodarone 200mg daily for patient
Jane Doe (MRN: PAT-001), who is currently on Warfarin."

```
Actor: Physician (End User)
Input: "Start Amiodarone 200mg daily for patient Jane Doe (MRN: PAT-001)"
```

**Step 1 — Observation Capture:**

The Clinical Memory Agent (𝒜_H) receives the physician's order text and calls:

```python
memory.add(
    "Start Amiodarone 200mg daily for patient Jane Doe (MRN: PAT-001)",
    source="clinician",
    tenant_id="hospital",
    app_id="med_recon",
    user_id="physician_42",
    agent_id="clinical_agent",
    run_id="shift_2024_01_15"
)
```

**Step 2 — Interaction Logging:**

GMem creates an immutable `Interaction` record with the raw text, actor type,
scope metadata, and timestamp.

**Step 3 — Clinical Fact Extraction:**

`HealthcareDatabaseExtractor` calls `HealthcareLLMExtractor.extract(raw_text)`
using the `CLINICAL_EXTRACTION_SYSTEM_PROMPT` and `ClinicalExtractionResult`
schema. The LLM returns:

```json
{
  "patient": {"name": "Jane Doe", "identifier": "PAT-001"},
  "medications": [
    {"name": "Amiodarone", "dosage": "200mg", "frequency": "daily",
     "action": "prescribe", "confidence": 0.95}
  ],
  "clinical_intent": "new_prescription"
}
```

**Step 4 — Entity Resolution:**

- Patient Jane Doe: resolved via MRN "PAT-001" → `canonical_id` match →
  reuses existing Entity with deterministic UUID5.
- Amiodarone: resolved via `find_entity_by_name("Amiodarone")` → creates
  or reuses MEDICATION entity.
- Normalization: `normalize_drug_name("Amiodarone")` → "amiodarone".

**Step 5 — Candidate Fact Construction:**

A `CandidateFact` is created:

```
subject: Jane Doe (PATIENT)
relation: PRESCRIBED
object: Amiodarone (MEDICATION)
value: 200mg daily
attributes: {medication_name: "Amiodarone", normalized_name: "amiodarone",
             dosage: "200mg", frequency: "daily", action: "prescribe",
             order_status: "active"}
disposition: CAPTURE
```

**Step 6 — Grounding (Constraint Validation):**

`GroundingOperator.ground(candidate)` runs the `ConstraintValidator` against
all registered healthcare constraints:

| Constraint | Result | Detail |
|---|---|---|
| `allergy_conflict` | PASS | Jane Doe has no known cross-reactive allergy to Amiodarone |
| `drug_interaction_major` | **REJECT** | Amiodarone + Warfarin is a major interaction (risk of bleeding) |
| `duplicate_active_medication` | PASS | No existing active Amiodarone prescription |
| `same_therapeutic_class` | PASS | No therapeutic duplication |

**Step 7 — Grounding Result:**

The `GroundingResult` is `REJECTED` with a `RejectionRecord` containing:

```
constraint_id: "drug_interaction_major"
reason: "Major drug interaction between amiodarone and warfarin (risk of bleeding)"
severity: "error"
alternatives: []
```

**Step 8 — Persistence:**

The `Interaction` (raw observation) is persisted in the Primary Store.
The `RejectionRecord` is persisted in the Primary Store.
No `ValidatedFact` is created. The drug interaction alert is later available in
`HealthcareClinicalContext.safety_alerts`.

**Outcome**: The unsafe prescription is blocked at write time. The clinician
sees an immediate safety alert with a clear explanation, not a latent warning
during a later query.

### 9.4 Healthcare Read Workflow (Step-by-Step)

**Scenario**: A pharmacist asks "What are Jane Doe's current medications and
are there any safety concerns?"

```
Actor: Pharmacist (End User)
Input: "What are Jane Doe's current medications and safety concerns?"
```

**Step 1 — Query Capture:**

The Clinical Memory Agent (𝒜_H) calls:

```python
context = context_builder.build(
    query="What are Jane Doe's current medications and safety concerns?",
    scope={"tenant_id": "hospital", "app_id": "med_recon",
            "user_id": "pharmacist_7", "patient_id": "PAT-001"},
    strategy=RetrievalStrategy.SAFETY_PRIORITY
)
```

**Step 2 — Query Planning:**

`HealthcareRetrievalPlanner.plan(query)` (LLM-assisted with regex fallback)
produces a `HealthcareQueryPlan`:

```json
{
  "patient_name": "Jane Doe",
  "patient_identifier": null,
  "requested_categories": ["medications", "safety"],
  "safety_focus": true,
  "as_of": null
}
```

**Step 3 — Entity Resolution:**

`resolve_seed_entities(query, plan, scope)`:

1. No MRN match →
2. Exact name match: PATIENT entity "Jane Doe" found → seed entity.
3. Medication/allergy name tokens matched against entity store.

**Step 4 — Graph Retrieval:**

`GraphRetriever.retrieve(query, seed_entities, strategy=SAFETY_PRIORITY)`:

- **Phase 1**: Collect all safety-critical facts (`PRESCRIBED`, `DISCONTINUED`,
  `HAS_ALLERGY`, `HAS_CONDITION`) connected to the seed entities, sorted by
  local query relevance.
- **Phase 2**: Collect remaining facts; irrelevant non-safety facts beyond the
  first hop are pruned to avoid graph drift.
- Apply scope filter (tenant, app, patient).
- Apply temporal filter (current-time query uses Neo4j active projection).

**Step 5 — Reranking:**

Five-signal reranking with `SAFETY_PRIORITY` weight profile:
- Calibrated relation weight: normalized against the max configured weight.
- Relevance-gated combination: relevance modulates structural signals;
  safety-critical facts receive a direct bypass.
- Two-layer diversity penalty: semantic-key deduplication + token-overlap
  redundancy suppression.
- Result: ranked list of `ValidatedFact` objects.

**Step 6 — Context Assembly:**

`HealthcareContextBuilder` post-processes raw facts into clinical views:

```json
{
  "current_medications": [
    {"name": "Warfarin", "dosage": "5mg", "frequency": "daily",
     "route": "oral", "action": "prescribe", "order_status": "active"},
    {"name": "Lisinopril", "dosage": "20mg", "frequency": "daily",
     "route": "oral", "action": "prescribe", "order_status": "active",
     "note": "superseded from 10mg on 2024-01-20"}
  ],
  "allergies": [
    {"allergen": "Penicillin", "reaction": "anaphylaxis", "severity": "severe"}
  ],
  "safety_alerts": [
    {"constraint": "drug_interaction_major",
     "detail": "Major interaction: Amiodarone + Warfarin (risk of bleeding)",
     "severity": "error",
     "timestamp": "2024-01-15T10:30:00Z"}
  ],
  "history": [
    {"medication": "Lisinopril", "dosage": "10mg", "action": "prescribed",
     "valid_from": "2024-01-10", "valid_to": "2024-01-20", "superseded_by": "..."},
    {"medication": "Lisinopril", "dosage": "20mg", "action": "prescribed",
     "valid_from": "2024-01-20", "valid_to": null}
  ]
}
```

**Step 7 — Response Generation:**

The Clinical Memory Agent receives `HealthcareClinicalContext.to_dict()` and
generates a grounded response to the pharmacist:

> "Jane Doe (PAT-001) is currently on Warfarin 5mg daily and Lisinopril 20mg
> daily (dose increased from 10mg on Jan 20). She has a severe penicillin
> allergy (anaphylaxis). **Safety alert**: Amiodarone was attempted but
> rejected due to a major interaction with Warfarin (bleeding risk)."

**Key insight**: The response is grounded entirely in validated, governed facts
— not in raw text or hallucinated content. Every statement is traceable to a
`ValidatedFact` with provenance, and every rejection is traceable to a
`RejectionRecord` with a constraint ID and reason.

### 9.5 Governance Lifecycle

```
Constraint Lifecycle States:

  PROPOSED ──► SHADOW ──► ACTIVE ──► DEPRECATED
      │            │          │
      │            │          └── blocks writes on error violations
      │            └── observes & records, never blocks
      └── observes, never records or blocks

Transition Triggers:
  - PROPOSED → SHADOW: Governance Administrator (𝒢) promotes after initial observation
  - SHADOW → ACTIVE: 𝒢 promotes after replay evidence meets confidence threshold
  - ACTIVE → DEPRECATED: 𝒢 retires when constraint is no longer relevant
  - AGENT → PROPOSED: Constraint Discovery Engine synthesizes from validation signals
```

---

## 10. Healthcare Domain-Specific Components

### 10.1 Drug Knowledge Base

**Definition 29 (Drug Knowledge Base).** The **Drug Knowledge Base** is a
mock/demonstration knowledge base providing clinical drug data for the
healthcare adapter. It contains:

| Data | Purpose |
|---|---|
| `ALLERGY_CROSS_REACTIVITY` | Maps allergens to cross-reactive substance sets (e.g., penicillin ↔ amoxicillin) |
| `DRUG_ALIASES` | Brand-to-generic name mapping (e.g., Advil → ibuprofen, Zocor → simvastatin) |
| `DRUG_INGREDIENTS` | Drug-to-active-ingredient mapping (e.g., Amoxicillin → {amoxicillin, penicillin}) |
| `DRUG_THERAPEUTIC_CLASSES` | Drug-to-class mapping (e.g., Lisinopril → ACE inhibitor) |
| `MAJOR_DRUG_INTERACTIONS` | Drug pairs withmajor interactions (e.g., Warfarin + Amiodarone) |
| `MODERATE_DRUG_INTERACTIONS` | Drug pairs with moderate interactions (e.g., Ibuprofen + Lisinopril) |

Helper functions: `normalize_drug_name()`, `get_cross_reactive_ingredients()`,
`check_major_interaction()`, `expand_drug_terms()`.

**Important**: This knowledge base is explicitly a mock/demo and must be
described as such in thesis materials. It is not clinical-grade.

Implementation: `adapters/healthcare/knowledge.py`.

### 10.2 Clinical Constraints

**Definition 30 (Clinical Constraints).** The healthcare adapter registers
constraint evaluators loaded from `configs/healthcare_constraints.yaml` via
`YamlConstraintEvaluator`. The constraint checks are:

| Constraint | Check Type | Severity | Behavior |
|---|---|---|---|
| `allergy_conflict` | `intersection_empty` | Error | Rejects if proposed medication ingredients overlap with patient's cross-reactive allergy set |
| `drug_interaction_major` | `no_major_interactions` | Error | Rejects if proposed medication has a major interaction with any active medication |
| `drug_interaction_moderate` | `no_moderate_interactions` | Warning | Warns if proposed medication has a moderate interaction with any active medication |
| `duplicate_active_medication` | `cardinality_limit` | Error | Rejects duplicate active prescriptions (allows in-place dose updates via same object_entity_id) |
| `same_therapeutic_class` | `cardinality_limit` | Error | Rejects if max_count medications already exist in the same therapeutic class |
| `timeline_consistency` | Temporal | Error | Enforces temporal ordering for clinical events |
| `high_risk_medication` | Informational | Info | Flags high-risk medications for pharmacist review |
| `renal_dose_adjustment` | Requirement | Warning | Suggests dose adjustment for renal impairment |
| `generic_substitution` | Informational | Info | Notes availability of generic alternatives |

All checks respect discontinuation: `_active_patient_prescribed_facts()` filters
out prescriptions closed by DISCONTINUED facts.

Implementation: `adapters/healthcare/constraints.py`.

### 10.3 Medication Lifecycle Management

**Definition 31 (Medication Lifecycle).** The **Medication Lifecycle** manages
the state transitions of medication orders:

```
PRESCRIBED ──dose_change──► PRESCRIBED (new fact, old superseded)
PRESCRIBED ──discontinue──► DISCONTINUED (old PRESCRIBED closed)
PRESCRIBED ──hold─────────► DISCONTINUED (old PRESCRIBED closed)
PRESCRIBED ──continue─────► PRESCRIBED (same fact reaffirmed)
```

The lifecycle closure function (`apply_medication_lifecycle_after_grounding`)
runs after grounding for each DISCONTINUED-type candidate fact that was
APPROVED:

1. Find all active `PRESCRIBED` facts for the same patient and medication
   (matched by `object_entity_id` or normalized medication name).
2. Set their `valid_to` to the discontinuation timestamp.
3. Set their `superseded_by` to the DISCONTINUED fact's ID.

This ensures that `current_medications` retrieval never shows discontinued
orders.

**Axiom ML1 (Lifecycle Closure).** Every APPROVED DISCONTINUED fact triggers
immediate closure of all matching active PRESCRIBED facts. No PRESCRIBED fact
remains active after its discontinuation has been grounded.

Implementation: `adapters/healthcare/lifecycle.py`.

### 10.4 Clinical Context Views

**Definition 32 (Clinical Context Views).** The healthcare context builder
produces four domain-specific views from the raw `AnswerContext`:

1. **`current_medications`**: Active `PRESCRIBED` facts, excluding those closed
   by `DISCONTINUED` facts. Each row includes: medication_name, normalized_name,
   dosage, frequency, route, action, order_status.

2. **`allergies`**: Active `HAS_ALLERGY` facts for the patient's scope. Each row
   includes: allergen, reaction, severity.

3. **`safety_alerts`**: All `RejectionRecord` entries for the patient's scope.
   Each row includes: constraint_id, constraint_name, reason, severity,
   alternatives, timestamp.

4. **`history`**: Full bitemporal timeline of `PRESCRIBED` and `DISCONTINUED`
   facts, including superseded ones. Each row includes: medication name, dosage,
   action, valid_from, valid_to, superseded_by.

These views are the _only_ data sources used for generating final clinical
answers. No hidden state or hardcoded demo facts are used.

Implementation: `adapters/healthcare/retrieval.py` (`HealthcareContextBuilder`,
`HealthcareClinicalContext`).

### 10.5 Healthcare Query Planning and Entity Resolution

**Definition 33 (Healthcare Query Planner).** The `HealthcareRetrievalPlanner`
produces a `HealthcareQueryPlan` from a natural-language query using LLM-backed
extraction with deterministic fallback:

| Method | Priority | Behavior |
|---|---|---|
| MRN extraction | 1 | Regex match for MRN-like patterns → `canonical_id` lookup |
| Exact name match | 2 | Regex for possessive/indirect patterns → exact PATIENT entity match |
| Medication/allergy name | 3 | Match tokens against medication/allergy entity names |
| KB-aware expansion | 4 | Drug term expansion (aliases, ingredients, therapeutic classes) |
| Fuzzy name tokens | 5 | Token overlap scoring against PATIENT entity names |
| Generic fallback | 6 | `GraphRetriever.select_seed_entities()` |

**Axiom QP1 (Graceful Degradation).** If the LLM is unavailable for query
planning, the deterministic fallback produces a usable (though less nuanced)
query plan. Query planning failure does not block retrieval.

Implementation: `adapters/healthcare/retrieval.py`
(`HealthcareRetrievalPlanner`).

---

## 11. Deployment Modes

GMem can be deployed in three modes:

| Mode | Description | Use Case |
|---|---|---|
| **Embedded Library** | Application imports `gmem` package and calls `Memory` objects in-process | Single-process agents, notebooks, testing |
| **Sidecar Service** | GMem runs as a separate process with REST API (FastAPI); Host Agent calls via HTTP | Microservices, multi-agent platforms |
| **Shared Platform** | Multiple Host Agents call one GMem instance with tenant/session isolation | Multi-tenant SaaS, hospital information systems |

In all modes, the SDK API surface is identical. The FastAPI service layer
exposes the same write and read operations as the in-process `Memory` class.

---

## 12. Summary of Formal Axioms

| Axiom | Statement |
|---|---|
| **G1** (Governance First) | No candidate fact enters durable memory without passing through the Grounding Operator Γ. |
| **G2** (Non-Destruction) | No ValidatedFact is ever physically deleted. Facts are closed by setting `valid_to`. |
| **G3** (Rejection Audit) | Every rejected candidate produces an auditable RejectionRecord. |
| **I1** (Immutability) | Interactions are frozen; once created, they cannot be modified. |
| **E1** (Deterministic Identity) | The same entity always receives the same ID across runs (UUID5 from semantic key). |
| **CF1** (Transience) | A CandidateFact is a proposal, not knowledge; it must pass through Γ before influencing any retrieval result. |
| **VF1** (Non-Deletion, restate) | ValidatedFacts are superseded, not deleted. |
| **VF2** (Bitemporal Completeness) | Point-in-time state is exactly reconstructable from valid_from/valid_to. |
| **C1** (Governance Before Persistence) | Every CandidateFact is evaluated against all applicable active constraints before becoming a ValidatedFact. |
| **AC1** (Ephemerality) | An AnswerContext exists only for the duration of a single read request. |
| **PS1** (Source of Truth) | PostgreSQL is the durable source of truth; Neo4j is a rebuildable projection. |
| **PS2** (Rebuildability) | Neo4j can be fully rebuilt from PostgreSQL at any time. |
| **GR1** (Deterministic Retrieval) | Given the same state and query (excluding LLM seed selection), the retriever produces the same results. |
| **ML1** (Lifecycle Closure) | Every approved DISCONTINUED fact triggers immediate closure of matching active PRESCRIBED facts. |
| **QP1** (Graceful Degradation) | LLM failure does not block retrieval; deterministic fallbacks are always available. |

---

## 13. Summary of System Components

| Component | Type | Role |
|---|---|---|
| Interaction Logger | Deterministic | Records raw observations as immutable `Interaction` objects |
| Extraction Engine | **Agentic** | Transforms raw text into `CandidateFact` proposals via LLM |
| Grounding Operator (Γ) | Deterministic | Validates, deduplicates, resolves conflicts, persists accepted facts |
| Constraint Validator | Deterministic | Evaluates candidates against governance rules |
| Conflict Resolver | Deterministic | Decides supersession order for conflicting facts |
| Constraint Discovery Engine | Deterministic | Mines validation signals to synthesize new constraint proposals |
| Primary Store | Deterministic | Durable bitemporal source of truth (PostgreSQL or in-memory) |
| Active Graph Projection | Deterministic | Read-optimized Neo4j projection of currently-active facts |
| Query Planner | **Agentic** (with fallback) | Transforms natural-language queries into structured plans |
| Graph Retriever | Deterministic | Traverses knowledge graph from seed entities with reranking |
| Context Builder | Deterministic | Assembles structured AnswerContext or domain-specific views |
| Adapter Layer | Configuration | Domain-specific wiring for extraction, constraints, and retrieval |
| Drug Knowledge Base | Deterministic | Mock clinical drug data for the healthcare adapter |
| Medication Lifecycle | Deterministic | Manages PRESCRIBED/DISCONTINUED state transitions |
| Clinical Constraints | Deterministic | Allergy conflict, drug interaction, therapeutic duplication checks |
| Clinical Context Views | Deterministic | current_medications, allergies, safety_alerts, history |

---

## 14. Actor ↔ Component Interaction Matrix

| Component | End User (𝕌) | Host Agent (𝒜_H) | Governance Admin (𝒢) | App Developer (𝒟) |
|---|---|---|---|---|
| Interaction Logger | — | Provides raw text via `add()` | — | — |
| Extraction Engine | — | Triggered by `add()` | — | Configures adapter + LLM |
| Grounding Operator | — | Triggered by `add()` | Reviews rejection audit | — |
| Constraint Validator | — | Triggered by `add()` | Registers/promotes constraints | Configures constraint YAML |
| Conflict Resolver | — | Triggered by `add()` | — | Configures strategy |
| Constraint Discovery | — | Triggered by `discover_constraint_seeds()` | Reviews and promotes proposals | — |
| Primary Store | — | Read/write via SDK | — | Selects backend |
| Active Graph Projection | — | Transparent read optimization | — | Enables/disables Neo4j |
| Query Planner | Asks question (via 𝒜_H) | Calls `search()` or `build_context()` | — | — |
| Graph Retriever | — | Triggered by read path | — | Configures strategy + weights |
| Context Builder | — | Receives context output | — | Configures adapter |
| Drug Knowledge Base | — | Transparent | — | May extend mock KB |
| Medication Lifecycle | — | Triggered by `add()` | — | — |
| Clinical Constraints | — | Transparent (blocks/rejects) | Configures active/shadow states | Configures YAML |

**Key principle**: The End User (𝕌) never directly interacts with any GMem
component. All interactions are mediated by the Host Agent (𝒜_H), which is
the sole caller of GMem.