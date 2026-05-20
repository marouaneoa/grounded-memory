# Quick Reference: Constraint Learning Loop with Scalable KB

## Complete Data Flow: Knowledge → Validation → Learning → Promotion

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. KNOWLEDGE BASE (External Sources)                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   RxNorm API              openFDA API           Local JSON/YAML            │
│   ├─ Drug interactions    ├─ Labels            └─ Custom data             │
│   └─ Therapeutic class    └─ Ingredients                                   │
│           │                    │                    │                      │
│           └────────────────────┴────────────────────┘                      │
│                         ▼                                                  │
│        InMemoryKnowledgeBase (knowledge.py)                                │
│        ├─ aliases: {"advil": "ibuprofen", ...}                            │
│        ├─ ingredients: {"ibuprofen": {"nsaid", ...}}                      │
│        ├─ therapeutic_classes: {...}                                      │
│        ├─ allergy_cross_reactivity: {...}                                 │
│        ├─ major_interactions: {frozenset({"drug_a", "drug_b"}), ...}      │
│        └─ moderate_interactions: {...}                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. WRITE-TIME VALIDATION (Feedback)                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   memory.add_fact(                                                          │
│       subject_id="patient_1",                                              │
│       relation="PRESCRIBED",                                               │
│       object_id="drug_1",  # or value="Ibuprofen 400mg"                   │
│       confidence=0.9,                                                       │
│   )                                                                          │
│           │                                                                │
│           ▼                                                                │
│   CandidateFact created → grounding_operator.ground()                      │
│           │                                                                │
│           ▼                                                                │
│   ConstraintValidator.validate()                                           │
│   ├─ Human-active PRESCRIBED constraints (always run)                      │
│   │  ├─ No major drug interactions (uses KB.check_major_interaction)       │
│   │  ├─ No allergy cross-reactivity (uses KB.get_cross_reactive_ingredients)
│   │  └─ Duplicate medication check                                         │
│   ├─ Agent-active constraints (capped to 20)                               │
│   │  └─ Discovered rules, priority-sorted                                  │
│   ├─ Proposed/Shadow constraints (capped to 10, observe only)              │
│   │  └─ New auto-discovered rules (never block)                            │
│   │                                                                        │
│   └─ Decision: ACCEPT or REJECT (recorded in ValidationResult)             │
│           │                                                                │
│           ▼                                                                │
│   record_validation_signal()                                               │
│   {                                                                         │
│       "timestamp": "2026-05-09T...",                                       │
│       "candidate_id": "...",                                               │
│       "relation": "PRESCRIBED",                                            │
│       "is_valid": true/false,                                              │
│       "violations": ["constraint_id_1", ...],  # Why rejected             │
│       "candidate_attributes": {...},                                       │
│       "has_value": true,                       # Did it have a value?     │
│       "has_object": true,                      # Did it have an object?   │
│   }                                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                  ▼
                    [Signals accumulate (~500 stored)]
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. PATTERN MINING (Discovery)                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   memory.discover_constraint_seeds(signal_limit=500)                        │
│           │                                                                │
│           ▼                                                                │
│   ConstraintSeedDiscoverer.discover()                                       │
│   ├─ Group signals by relation: {PRESCRIBED: [...], HAS_ALLERGY: [...]}   │
│   │                                                                        │
│   ├─ For each relation:                                                   │
│   │  ├─ Count rejections (is_valid=false)                                │
│   │  ├─ For each field (has_value, has_object, attributes...):           │
│   │  │  ├─ % rejected WITH field missing                                 │
│   │  │  ├─ % rejected WITH field present                                 │
│   │  │  ├─ Gap = rejected_without - rejected_with                        │
│   │  │  └─ If gap > 0.35: PATTERN FOUND                                 │
│   │  │                                                                    │
│   │  └─ Synthesize seed:                                                 │
│   │     "PRESCRIBED facts missing value field are rejected 40% of time,  │
│   │      but only 5% when value present → require value"                │
│   │                                                                        │
│   └─ Return top-20 seeds by confidence & evidence                         │
│                                                                             │
│   Output: [                                                                 │
│       DiscoveredConstraintSeed(                                            │
│           constraint_id="seed_auto_require_value_PRESCRIBED",              │
│           name="Prescription must include medication name",                │
│           description="...",                                               │
│           relation_types=["PRESCRIBED"],                                   │
│           require_value=True,                                              │
│           confidence=0.92,                                                 │
│           evidence_count=200,  # signals analyzed                          │
│           mining_rule="require_value",                                     │
│       ),                                                                    │
│       ...                                                                   │
│   ]                                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. RULE CREATION (Register Seeds)                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   memory.register_discovered_constraint_seeds(                              │
│       seeds[:5],  # Top 5 by confidence                                    │
│       lifecycle="shadow",  # Start in shadow mode (observe, don't block)   │
│   )                                                                          │
│           │                                                                │
│           ▼                                                                │
│   For each seed:                                                            │
│   ├─ Create SeedConstraintEvaluator                                        │
│   ├─ Register with ConstraintValidator as:                                │
│   │  ├─ lifecycle = SHADOW (observe warnings)                             │
│   │  ├─ source = "agent" (auto-discovered)                                │
│   │  ├─ priority = 50 (lower than human ACTIVE rules)                     │
│   │  └─ shadow_hits = 0, shadow_violations = 0                            │
│   │                                                                        │
│   └─ Constraint now participates in validation on next writes              │
│           │                                                                │
│           ▼                                                                │
│   On next add_fact():                                                       │
│   ├─ SHADOW constraint evaluated                                           │
│   ├─ If violated: record warning (never block)                            │
│   ├─ Update shadow_violations counter                                     │
│   └─ Continue (write succeeds)                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                  ▼
                    [More signals, more shadow observations]
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. REPLAY (Collect Evidence)                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   # Typically done as batch job (e.g., daily):                             │
│   all_candidates = memory.system.memory_store.get_all_validated_facts()   │
│                                                                             │
│   metrics = memory.get_constraint_replay_metrics(                          │
│       candidates=all_candidates,                                           │
│       # Evaluates each dynamic constraint against all candidates           │
│   )                                                                          │
│           │                                                                │
│           ▼                                                                │
│   ConstraintValidator.replay_dynamic_constraints()                         │
│   ├─ Separate human-ACTIVE rules from dynamic (agent) rules               │
│   │                                                                        │
│   ├─ For each candidate:                                                  │
│   │  ├─ Check: is this blocked by human-authored rules?                  │
│   │  └─ For each dynamic rule:                                           │
│   │     ├─ Evaluate (would it block this candidate?)                      │
│   │     ├─ If yes, increment violation counter                           │
│   │     ├─ If also human-blocked: redundant (increment covered)           │
│   │     └─ Else: incremental block (new insight!)                        │
│   │                                                                        │
│   └─ Compute metrics for each dynamic constraint:                         │
│      ├─ trigger_rate = violations / total_candidates                     │
│      ├─ false_block_rate = incremental_blocks / total_candidates         │
│      └─ coverage = covered_existing_blocks / total_candidates             │
│                                                                             │
│   Output: {                                                                 │
│       "seed_auto_require_value_PRESCRIBED": ConstraintReplayMetrics(       │
│           constraint_id="...",                                             │
│           lifecycle="shadow",                                              │
│           evaluated_candidates=2000,                                       │
│           violations=68,         # 3.4% trigger rate                      │
│           incremental_blocks=2,  # 0.1% false positives                  │
│           covered_existing_blocks=66,  # 99% redundant (already blocked)  │
│       ),                                                                    │
│       ...                                                                   │
│   }                                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. PROMOTE (Governance & Activation)                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   memory.promote_constraints(                                               │
│       replay_metrics=metrics,                                              │
│       min_trigger_rate=0.01,              # Must fire on ≥1% of facts    │
│       max_projected_false_block_rate=0.02,  # But ≤2% false positives    │
│       min_candidates=100,                 # Min evidence base             │
│   )                                                                          │
│           │                                                                │
│           ▼                                                                │
│   ConstraintValidator.promote_dynamic_constraints()                        │
│   ├─ For each dynamic constraint with replay evidence:                     │
│   │  ├─ Check: evaluated_candidates ≥ 100? ✓ (2000)                     │
│   │  ├─ Check: trigger_rate ≥ 0.01? ✓ (0.034)                          │
│   │  ├─ Check: false_block_rate ≤ 0.02? ✓ (0.001)                       │
│   │  └─ DECISION: PROMOTE → ACTIVE                                       │
│   │                                                                        │
│   └─ Update lifecycle: SHADOW → ACTIVE                                   │
│           │                                                                │
│           ▼                                                                │
│   On next add_fact():                                                       │
│   ├─ Now-ACTIVE constraint evaluated                                       │
│   ├─ If violated: BLOCK write (hard error)                               │
│   ├─ CandidateFact rejected, reason recorded                              │
│   └─ Result: Only valid facts persist                                      │
│                                                                             │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │ Lifecycle Transitions Possible:                                    │   │
│   ├────────────────────────────────────────────────────────────────────┤   │
│   │ PROPOSED   → SHADOW   (governance approves initial observation)    │   │
│   │ SHADOW     → ACTIVE   (replay evidence meets thresholds)           │   │
│   │ ACTIVE     → DEPRECATED (rule deemed harmful or obsolete)          │   │
│   │ Any state  → Any state (manual override by governance admin)       │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                  ▼
                         [Loop restarts]
                    (More signals, more learning)
```

---

## Code Examples by Stage

### Stage 1: Add Fact (Feedback)
```python
from gmem import Memory

memory = Memory(adapter="healthcare")
patient = memory.add_entity("John Doe", entity_type="person")
drug = memory.add_entity("Warfarin", entity_type="medication")

# Write a prescription
result = memory.add_fact(
    subject_id=patient["entity"]["id"],
    relation="PRESCRIBED",
    object_id=drug["entity"]["id"],
    value="Warfarin 5mg daily",
    confidence=0.95,
)
# ✓ Validates against human-active + shadow constraints
# ✓ Signals recorded in _validation_signals list
```

### Stage 2: Discover Seeds (Pattern Mining)
```python
# After ~500 writes/validations...
seeds = memory.discover_constraint_seeds(
    signal_limit=500,
    min_gap=0.35,
    min_rejections_per_relation=6,
)

for seed in seeds:
    print(f"{seed.name}: confidence={seed.confidence:.2f}, evidence={seed.evidence_count}")
    # seed_auto_require_value_PRESCRIBED: confidence=0.92, evidence=200
```

### Stage 3: Register Seeds (Rule Creation)
```python
memory.register_discovered_constraint_seeds(
    seeds[:5],  # Top 5 by confidence
    lifecycle="shadow",
)

# Check constraint status
managed = memory.system.validator.list_managed_constraints()
for m in managed:
    if m.source.value == "agent":
        print(f"{m.evaluator.constraint_id}: {m.lifecycle.value}")
        # seed_auto_require_value_PRESCRIBED: shadow
```

### Stage 4: Replay (Evidence Collection)
```python
all_facts = memory.system.memory_store.get_all_validated_facts()
metrics = memory.get_constraint_replay_metrics(all_facts)

for constraint_id, metric in metrics.items():
    print(f"{constraint_id}:")
    print(f"  Trigger rate: {metric.trigger_rate:.2%}")
    print(f"  False block rate: {metric.projected_false_block_rate:.2%}")
    # seed_auto_require_value_PRESCRIBED:
    #   Trigger rate: 3.40%
    #   False block rate: 0.10%
```

### Stage 5: Promote (Governance)
```python
promoted = memory.promote_constraints(
    replay_metrics=metrics,
    min_trigger_rate=0.01,
    max_projected_false_block_rate=0.02,
)

print(f"Promoted {len(promoted)} constraints to ACTIVE:")
for cid in promoted:
    print(f"  ✓ {cid}")
    # ✓ seed_auto_require_value_PRESCRIBED
```

---

## Knowledge Base Usage in Constraints

### Healthcare Constraints Use KB Functions
```python
# During validation, constraints call KB functions:

# 1. Check drug interactions
from grounded_memory.adapters.healthcare.knowledge import check_major_interaction
major = check_major_interaction("Warfarin", "Ibuprofen")  # True

# 2. Get cross-reactive allergens
from grounded_memory.adapters.healthcare.knowledge import get_cross_reactive_ingredients
cross_react = get_cross_reactive_ingredients("penicillin")
# {"penicillin", "amoxicillin", "ampicillin", ...}

# 3. Get therapeutic classes
from grounded_memory.adapters.healthcare.knowledge import get_therapeutic_classes
classes = get_therapeutic_classes("ibuprofen")  # {"nsaid"}
```

### Extend KB with External Data
```python
from grounded_memory.adapters.healthcare import knowledge, loaders

# Load RxNorm interactions
rxnorm_kb = loaders.rxnorm.build_kb_from_rxnorm(
    drug_names=["Warfarin", "Ibuprofen", "Aspirin"],
)
knowledge.register_source(rxnorm_kb)

# Load openFDA labels
fda_kb = loaders.openfda.fetch_batch_labels(
    drug_names=["Warfarin", "Ibuprofen", "Aspirin"],
)
knowledge.register_source(fda_kb)

# Now KB has thousands of verified interactions from NLM + FDA
```

---

## Key Thresholds & Parameters

| Parameter | Default | Meaning | Where |
|-----------|---------|---------|-------|
| `min_gap` | 0.35 | Min rejection gap to propose constraint | `discover_constraint_seeds()` |
| `min_samples_per_relation` | 20 | Min signals per relation to analyze | `discover_constraint_seeds()` |
| `min_rejections_per_relation` | 6 | Min rejections to consider pattern | `discover_constraint_seeds()` |
| `min_trigger_rate` | 0.01 | Min % facts to trigger constraint | `promote_constraints()` |
| `max_projected_false_block_rate` | 0.02 | Max % false positives tolerated | `promote_constraints()` |
| `min_candidates` | 100 | Min facts to replay before promotion | `promote_constraints()` |
| `max_dynamic_constraints` | 20 | Max agent rules active per write | `validate()` |
| `max_shadow_constraints` | 10 | Max proposed/shadow rules per write | `validate()` |

---

## Debugging Tips

### See What Signals Are Recorded
```python
signals = memory.system.validator.list_validation_signals(limit=100)
for sig in signals:
    print(f"Relation={sig['relation']}, Valid={sig['is_valid']}")
    print(f"  Violations: {sig['violations']}")
    print(f"  Attributes: {sig['candidate_attributes']}")
```

### Check Constraint Lifecycle
```python
for managed in memory.system.validator.list_managed_constraints():
    print(f"{managed.constraint_id}:")
    print(f"  Lifecycle: {managed.lifecycle.value}")
    print(f"  Source: {managed.source.value}")
    print(f"  Shadow hits: {managed.shadow_hits}")
    print(f"  Shadow violations: {managed.shadow_violations}")
```

### Inspect Replay Metrics
```python
metrics = memory.get_constraint_replay_metrics(all_facts)
for cid, metric in metrics.items():
    print(f"{cid}:")
    print(f"  Evaluated: {metric.evaluated_candidates}")
    print(f"  Trigger rate: {metric.trigger_rate:.4f}")
    print(f"  False block rate: {metric.projected_false_block_rate:.4f}")
```

### Force Promote (Governance Override)
```python
memory.system.validator.set_lifecycle(
    "seed_auto_require_value_PRESCRIBED",
    "active",  # Promote immediately
)
```

---

## Integration with External APIs

Once loaders are implemented, KB automatically enriched at startup:

```yaml
# configs/healthcare_kb.yaml
knowledge_base:
  sources:
    - name: "rxnorm_interactions"
      type: "rxnorm"
      enabled: true
      params:
        drugs: ["Warfarin", "Ibuprofen", "Aspirin", ...]
        max_workers: 4
      cache_ttl_hours: 168

# Loads ~300k+ verified interactions from NLM
# Constraints use this data during validation
# Constraint learning uses this for semantic features
```

