<p align="center">
  <img src="assets/gmem-logo.svg" alt="gmem" width="860" />
</p>

<p align="center">
  <strong>gmem</strong> is a correctness-first memory runtime for LLM agents.<br/>
  It stores only grounded facts, keeps temporal history, and stays adapter-driven instead of usecase-coupled.
</p>

---

## Why gmem

Most memory layers optimize retrieval speed first.
gmem optimizes correctness first:

- candidate tuples are proposed by the LLM
- constraints validate before persistence
- accepted facts become durable memory with supersession
- retrieval builds compact context from governed state

## Core Model

gmem keeps six objects explicit:

1. `Interaction` (immutable observed event)
2. `Entity` (symbolic anchor)
3. `CandidateFact` (untrusted proposal)
4. `ValidatedFact` (accepted fact with valid-time boundaries)
5. `Constraint` (write-time governance)
6. `AnswerContext` (ephemeral retrieval view)

## Installation

From PyPI (stable):

```bash
# Core runtime
pip install grounded-memory

# With LLM extraction support
pip install grounded-memory[llm]

# With PostgreSQL + Neo4j backends
pip install grounded-memory[postgres,neo4j]

# With FastAPI service layer
pip install grounded-memory[api]

# Everything (dev, llm, api, benchmark, postgres, neo4j)
pip install grounded-memory[all]
```

For local development, install in editable mode:

```bash
pip install -e ".[dev,llm,api,postgres,neo4j]"
```

## Quick Start

Minimal usage:

```python
from gmem import Memory

memory = Memory(adapter="generic", storage_backend="memory")

memory.add("My project codename is Atlas.", user_id="demo")
results = memory.search("What is my project codename?", user_id="demo", limit=3)

print(results)
memory.close()
```

Intent routing (auto-classify and route natural-language input):

```python
from gmem import Memory

memory = Memory(adapter="generic", storage_backend="memory")

# Explicit intent classification
intent = memory.route("What is my project codename?")
print(intent.action)  # "recall"

# Auto-routing: REMEMBER -> add(), RECALL -> search(), etc.
result = memory.process("My project codename is Atlas.", user_id="demo")
print(result["intent"]["action"])  # "remember"

memory.close()
```

## OpenRouter Setup

In `.env`:

```bash
LLM_PROVIDER=openrouter
LLM_MODEL=z-ai/glm-4.5-air:free
OPENROUTER_API_KEY=...
```

Quick API smoke check:

```bash
make smoke-openrouter
```

## Local Dev UX

Start databases:

```bash
make services-up
```

Stop databases:

```bash
make services-down
```

Run memory smoke behavior test:

```bash
make smoke-memory
```

Run deterministic hybrid write + backend inspection (in-memory facts, Neo4j graph, PostgreSQL counts):

```bash
make inspect-backends
```

Run the healthcare medication-reconciliation demo stack:

```bash
make services-up
PYTHONPATH=src python demos/demo_bitemporal.py
```

Then query the same data interactively (scope values match automatically):

```bash
PYTHONPATH=src python demos/demo_interactive.py --adapter healthcare
```

> **Scope alignment note:** Both demos use `require_scope=True` by default. Facts are tagged with scope fields (`tenant_id`, `app_id`, `user_id`, `agent_id`, `run_id`). To query data written by one demo from another, the scope values must match. The demos share default scope values so they align automatically; override with `--user-id`, `--agent-id`, `--run-id`, or env vars (`GM_SCOPE_*`) if needed.

Run the Postgres + Neo4j healthcare smoke check:

```bash
make smoke-healthcare-backends
```

## Adapter-Decoupled Runtime

The runtime is no longer tied to repository `usecases/` trees.

- `GM_ADAPTER` selects behavior profile (default: `generic`)
- adapters are registered via runtime adapter registry APIs
- `domain_profile` remains as compatibility alias in the API
- experiment-stage labels for research ablations stay in the sibling [ClinicaLongMem-Interact](../ClinicaLongMem-Interact/) benchmark repository and docs, not as hardcoded runtime stage classes

## Engineering Workflow

- contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- development conventions: [DEVELOPING.md](DEVELOPING.md)

## API Stability

Stable entry points:

- `gmem` facade imports
- `grounded_memory.memory.Memory` SDK facade
- FastAPI service endpoints under `grounded_memory.service`

Internal modules may evolve faster as research and storage internals iterate.

## Repository Layout

```text
src/
  gmem/                  # facade package
  grounded_memory/       # implementation runtime
    adapters/            # adapter registry + generic agent
    core/                # models, constraints, stores
    llm/                 # LLM client + extraction
    retrieval/           # graph retrieval
    service/             # FastAPI service layer

assets/                  # brand and diagram assets
demos/                   # runnable demos and showcase scripts
tests/                   # regression tests
configs/                 # runtime yaml configs
scripts/                 # operational and migration scripts
docs/
  architecture/          # architecture and roadmap docs
  sdk/                   # SDK reference docs
```

Benchmark runners, scenarios, and evaluation scripts live in the sibling repository [ClinicaLongMem-Interact](../ClinicaLongMem-Interact/).

## Compose Stack

The compose stack is intentionally simple:

- PostgreSQL: source of truth store
- Neo4j: active graph projection

Configured via [docker-compose.yml](docker-compose.yml) and [.env.example](.env.example).

## Demos

| Demo | Command | What it shows |
|---|---|---|
| **Bitemporal medication reconciliation** | `python demos/demo_bitemporal.py` | Write-time constraints, supersession, discontinuation, current + historical retrieval |
| **Multi-patient write** | `python demos/demo_multi_patient_write.py` | Scale write-phase with 10 patients, allergy/interaction rejection |
| **Multi-patient retrieval** | `python demos/demo_multi_patient_retrieval.py` | Cross-patient isolation, shared entity queries, historical as-of |
| **Interactive REPL** | `python demos/demo_interactive.py --adapter healthcare` | Intent routing, auto-routing of natural language, real-time grounding diagnostics |
| **OpenRouter system** | `python demos/demo_openrouter.py` | Full OpenRouter pipeline with extraction, grounding, and retrieval |

## Tests

Run all regression tests:

```bash
pytest tests/ -q
```

Key test suites:

| Test file | Coverage |
|---|---|
| `tests/test_governance.py` | Supersession, duplicate detection, conflict resolution strategies, constraint enforcement |
| `tests/test_healthcare_reconciliation.py` | Allergy conflict, drug interaction, therapeutic duplication, dose supersession |
| `tests/test_healthcare_retrieval.py` | Retrieval planning, seed resolution, discontinuation closure, historical as-of, scope isolation |
| `tests/test_end_to_end.py` | Engineering knowledge graph, multi-domain facts, constraint rejection with audit trail |
| `tests/test_adapters.py` | Adapter registry, YAML constraint configs, entity/relation type coverage |

## Healthcare Demo

The demo uses the healthcare adapter with PostgreSQL as the durable
bitemporal store and Neo4j as the active graph projection:

```python
Memory(
    adapter="healthcare",
    storage_backend="postgres_hybrid",
    require_scope=True,
)
```

It demonstrates LLM-backed clinical extraction, write-time safety constraints,
allergy/interaction rejections, dose supersession, discontinuation lifecycle
closure, current retrieval, and historical as-of retrieval.

## Academic Direction

The current roadmap emphasizes novelty around:

- bitemporal memory semantics for grounded tuples
- constraint lifecycle governance (`proposed -> shadow -> active -> deprecated`)
- adapter-level safety policies without domain hard-coding into core memory runtime

See [docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md) for
the implementation-aligned architecture and
[docs/architecture/GMEM_V2_BLUEPRINT.md](docs/architecture/GMEM_V2_BLUEPRINT.md)
for the broader roadmap.

## Status

This project is in **alpha**. The core APIs (`gmem`, `grounded_memory.memory`, and service endpoints) are stabilizing; internal modules may evolve as storage and research internals iterate.

## License

MIT
