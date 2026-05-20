# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-05-20

### Added
- Initial public release of the correctness-first memory runtime for LLM agents.
- Core memory model: `Interaction`, `Entity`, `CandidateFact`, `ValidatedFact`, `Constraint`, `AnswerContext`.
- Write pipeline with LLM extraction, grounding, and constraint validation.
- Bitemporal semantics (`valid_from` / `valid_to`) with supersession and retire logic.
- Adapter-driven runtime (`generic`, `healthcare`, `engineering`, `finance`, `legal`).
- Storage backends: `memory`, `hybrid`, `postgres`, `postgres_hybrid`.
- Graph retrieval with optimization profiles (`latency`, `balanced`, `recall`).
- Intent routing (`REMEMBER`, `RECALL`, `FIND_RELATED`, `EXPLAIN`).
- Scope system (`tenant_id`, `app_id`, `user_id`, `agent_id`, `run_id`, `space_type`).
- FastAPI service layer under `grounded_memory.service`.
- Public facade package `gmem` with stable import surface.

[Unreleased]: https://github.com/marouaneoa/GroundedMemory/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/marouaneoa/GroundedMemory/releases/tag/v0.1.0
