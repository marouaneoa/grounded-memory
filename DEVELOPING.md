# Developing gmem

This document defines repository structure and engineering conventions for day-to-day development.

## Repository Layout

- src/gmem: public facade package
- src/grounded_memory: implementation runtime
  - adapters: adapter registry and agent implementations
  - core: models, stores, grounding, constraint governance
  - llm: provider clients and extraction helpers
  - retrieval: retrieval and context-building logic
  - service: FastAPI layer and request/response models
- configs: runtime configuration files
- demos: runnable demonstrations and scenario checks
- benchmarks: live in the sibling [ClinicaLongMem-Interact](../ClinicaLongMem-Interact/) repository (runners, cases, and evaluation scripts)
- docs: architecture and SDK documentation
- scripts: operational utilities

## Public API and Stability

Stable entry points:

- Python facade imports from gmem
- Memory SDK facade in grounded_memory.memory
- FastAPI surface in grounded_memory.service.app

Internal modules may evolve faster, especially under:

- grounded_memory.core
- grounded_memory.retrieval internals
- adapter internals

When changing stable surfaces, update README and docs/sdk/SDK_REFERENCE.md in the same PR.

## Runtime Design Principles

- Runtime behavior should remain adapter-driven and composable.
- Research pipeline labels used in papers and ablations should stay in experiment configuration and benchmark orchestration.
- Avoid embedding stage-specific experiment logic as hardcoded runtime orchestration APIs.

## Code Conventions

- Keep functions focused and side effects explicit.
- Use type hints and concise docstrings for public methods.
- Prefer small helper functions over deeply nested flow logic.
- Reuse tuple normalization helpers from grounded_memory.core.tuple_normalization for duplicate, supersession, and retire semantics.

## Test Strategy

Minimum expectation for non-trivial changes:

1. Unit or integration coverage for changed behavior.
2. make lint and make test pass locally.
3. Targeted smoke check for SDK/service flows when relevant.

## Configuration and Experiments

- Keep runtime defaults in code minimal and predictable.
- Keep benchmark and ablation controls in benchmarks and config files.
- Document new config keys in README and docs/sdk when they affect user workflows.
