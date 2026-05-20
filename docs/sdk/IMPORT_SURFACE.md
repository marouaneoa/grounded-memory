# Import Surface

This page lists all symbols exported at package top level.

```python
from grounded_memory import ...
```

## Core Models

- `Interaction`
- `Entity`
- `CandidateFact`
- `ValidatedFact`
- `Constraint`
- `FactStatus`
- `RelationType`

## Grounding

- `GroundingOperator`
- `GroundingResult`

## Store / System

- `MemoryStore`
- `GroundedMemorySystem`
- `CoreGroundedMemorySystem`

## Constraints

- `ConstraintValidator`
- `ValidationResult`
- `ConstraintViolation`

## LLM Layer

- `LLM_AVAILABLE`
- `LLMConfig`
- `SyncLLMClient`
- `LLMClient`
- `LLMFactExtractor`

## SDK Facade

- `Memory`
- `SearchResult`
- `OptimizationProfile`
- `OptimizationSettings`

## Dynamic Seed / Registry APIs

- `SeedConstraintEvaluator`
- `CardinalitySeedConstraintEvaluator`
- `TemporalCardinalitySeedConstraintEvaluator`
- `DiscoveredConstraintSeed`
- `ConstraintSeedDiscoverer`
- `list_registered_adapters`
- `get_adapter_spec_by_key`
- `register_adapter`
- `unregister_adapter`
- `list_supported_profiles`
- `get_adapter_spec`
- `register_adapter_spec`
- `unregister_adapter_spec`
