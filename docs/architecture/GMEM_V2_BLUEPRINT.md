# gmem v2 Blueprint

## Objective

Build a professional OSS-grade memory runtime that is:

- adapter-driven (not repository-usecase coupled)
- correctness-first at write time
- bitemporal and audit-friendly
- easy to run locally with predictable developer workflows

## Design Principles

1. Keep core memory runtime domain-neutral.
2. Push domain behavior to adapter boundaries.
3. Preserve backward compatibility during migration.
4. Maintain deterministic storage semantics.
5. Prefer explicit governance over prompt-only behavior.

## Architecture Layers

### 1. Facade Layer (`gmem`)

Purpose:

- stable product-facing import path
- compact public API for OSS users

Current direction:

- `from gmem import Memory`
- `from gmem.service import create_app`

### 2. Runtime Layer (`grounded_memory`)

Purpose:

- implementation details and internal modules
- orchestration, grounding, retrieval, service, and stores

Rules:

- no hard dependency on repository `usecases/` trees
- adapter selection via runtime keys

### 3. Adapter Layer

Purpose:

- configurable wiring for validator setup and agent creation

Rules:

- adapters register through registry APIs
- built-in `generic` remains default
- legacy `domain_profile` remains an alias while migration is active

### 4. Storage Layer

Purpose:

- durable source of truth + graph projection

Semantics:

- valid time: `valid_from`, `valid_to`
- record time: interaction timestamps and persistence `created_at`

## Migration Plan

### Phase A (Current)

- remove deprecated/usecase-coupled repo modules
- introduce adapter-first naming in APIs
- keep compatibility aliases for profile-based callers

### Phase B

- move docs/examples to `gmem` import path
- publish adapter authoring guide
- add compatibility tests for alias behavior

### Phase C

- deprecate profile wording in user-facing docs
- eventually remove alias-only paths after one release cycle

## Academic Novelty Track

Primary novelty claim:

- **Constraint-governed bitemporal memory for agent systems**, where acceptance is determined by explicit symbolic governance before persistence.

Proposed empirical package:

1. **Governance Yield**
   - fraction of candidate facts rejected/accepted by lifecycle state
2. **Constraint Drift Sensitivity**
   - behavior under evolving shadow/proposed constraints
3. **Temporal Reconstruction Fidelity**
   - ability to answer historical as-of queries accurately
4. **Safety-Retention Tradeoff**
   - compare strict write-time governance vs retrieval-time filtering baselines

Candidate baseline families:

- retrieval-only memory stores
- summary-first memory systems
- post-hoc validation pipelines

## OSS Quality Bar

For each release candidate:

1. smoke tests with live LLM provider
2. health endpoint pass (`/health/live`, `/health/ready`)
3. adapter registry backward-compat checks
4. clean local startup (`docker compose`, `make` commands)
5. top-level README stays aligned with implementation
