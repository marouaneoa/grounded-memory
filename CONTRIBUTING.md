# Contributing to gmem

Thanks for helping improve gmem.

## Scope

This repository prioritizes a correctness-first memory runtime for LLM agents.
Contributions should preserve these core goals:

- write-time governance over unvalidated persistence
- temporal supersession over destructive overwrite
- adapter-driven behavior over usecase-coupled core logic

## Development Setup

1. Clone the repository.
2. Create and activate a Python 3.10+ virtual environment.
3. Install development dependencies:

   make install-dev

## Local Quality Gates

Run these before opening a pull request:

- lint: make lint
- format: make format
- tests: make test
- smoke checks (recommended for runtime-facing changes):
  - make smoke-memory
  - make smoke-openrouter (when OpenRouter credentials are configured)

## Pull Request Checklist

Please include the following in each PR description:

1. What changed and why.
2. Behavioral impact and migration notes (if any).
3. Test evidence (commands and relevant outputs).
4. Any documentation updates.

Before requesting review, confirm:

- code follows existing module boundaries
- changed behavior is covered by tests or smoke checks
- docs are updated when APIs or workflows change
- no secrets or environment files are committed

## Commit Style

Conventional Commits are preferred:

- feat: new capability
- fix: bug fix
- refactor: internal restructuring without behavior change
- docs: documentation-only changes
- test: test-only changes
- chore: maintenance tasks

## Design Guardrails

- Keep the public SDK/API simple and stable (`gmem`, `grounded_memory.memory`, service endpoints).
- Keep research/ablation orchestration in benchmarks and configs, not as hardcoded runtime stage classes.
- Avoid introducing new dependencies unless they are justified and documented.
- Prefer small, composable changes over broad rewrites.

## Questions

If requirements are ambiguous, open an issue or draft PR early with assumptions and open questions.
