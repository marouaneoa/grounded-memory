# Optimization Guide

This guide describes practical tuning for developer experience, latency, and retrieval quality.

## Optimization Layers

1. SDK defaults (`Memory` optimization profile)
2. Retrieval topology (`max_seeds`, `max_hops`, `max_facts`)
3. Retrieval strategy (`weighted`, `safety_priority`, `breadth_first`)
4. Relationship weighting (`GraphRetriever` relation weights)
5. Storage backend (MemoryStore-only vs Neo4j-enabled hybrid)

## Quick Start Profiles

### Latency-First

```python
from grounded_memory import Memory

memory = Memory(optimization_profile="latency")
```

Use when:
- high QPS
- short context windows
- interactive chat with strict p95 requirements

### Balanced

```python
memory = Memory(optimization_profile="balanced")
```

Use when:
- default production behavior
- mixed precision/recall requirements

### Recall-First

```python
memory = Memory(optimization_profile="recall")
```

Use when:
- investigation workflows
- analytics and offline diagnostics

## Runtime Tuning

```python
memory.configure_optimization(
    profile="balanced",
    max_seeds=8,
    max_hops=2,
    max_facts=40,
    strategy="safety_priority",
)
```

### Parameter Impact

- `max_seeds`
  - Higher: better recall, more expansion cost
  - Lower: lower latency, risk of misses
- `max_hops`
  - Higher: captures indirect links, can add noise
  - Lower: faster and stricter locality
- `max_facts`
  - Higher: richer context, larger prompt footprint
  - Lower: faster downstream LLM calls
- `strategy`
  - `weighted`: predictable latency with relation weighting
  - `safety_priority`: prioritizes safety-critical relation profiles
  - `breadth_first`: broader exploration for coverage

## Retrieval Weight Tuning

For advanced control, tune relation weights through `GraphRetriever`.

```python
retriever = memory.retriever
retriever.set_weight("HAS_ATTRIBUTE", weight=2.5, decay_per_hop=0.2)
```

Recommended process:
1. Baseline with profile defaults
2. Change one relation weight at a time
3. Evaluate with benchmark casebook
4. Keep one profile per workload class

## Write Path Throughput

Use `add_many(...)` for batch ingestion to simplify caller code and collect per-item outcomes.

```python
payloads = [
    "Event A",
    {"content": "Event B", "source": "assistant"},
]
summary = memory.add_many(payloads, continue_on_error=True)
```

## Governance Cost Controls

Dynamic constraints can become expensive if unconstrained. Keep these bounded:

- Use relation scope in seed registration
- Keep `required_context` selective
- Maintain a low `max_dynamic_constraints` runtime budget
- Promote only seeds with acceptable replay metrics

## Suggested Profiles by Use Case

- Online assistant: `latency`
- Clinical safety workflows: `balanced` + `safety_priority`
- Forensic/review workflows: `recall`
- Seed discovery and replay: `recall` for analysis, `balanced` for production
