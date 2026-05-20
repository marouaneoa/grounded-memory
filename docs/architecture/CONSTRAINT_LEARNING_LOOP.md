# Constraint Learning Loop: Feedback → Pattern Mining → Rule Creation → Replay → Promote

This document maps the **Constraint Learning** subsystem (L4 from the main architecture) to concrete implementations in the codebase. It shows how facts flow through write-time validation, how signals feed discovery, and how new rules graduate from proposed→shadow→active.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Constraint Learning Loop (L4)                         │
└─────────────────────────────────────────────────────────────────────────┘

  1. FEEDBACK
     ├─ Write-time validation signals (every add/add_fact call)
     ├─ Location: src/grounded_memory/core/constraints.py :: ConstraintValidator.validate()
     └─ Storage: in-memory _validation_signals list

  2. PATTERN MINING (Discovery)
     ├─ Autonomous seed synthesis from signals
     ├─ Location: src/grounded_memory/adapters/discovery.py :: ConstraintSeedDiscoverer.discover()
     ├─ Mining rules: require_value, require_attribute_keys, forbidden patterns
     └─ Output: DiscoveredConstraintSeed objects (confidence-scored proposals)

  3. RULE CREATION (Manual + Auto)
     ├─ Human: YAML/code-based constraint definition → add to validator
     ├─ Auto: discovered seeds → add_constraint_seed() or register_discovered_constraint_seeds()
     ├─ Lifecycle entry: PROPOSED (agents) or ACTIVE (humans)
     ├─ Location: src/grounded_memory/memory.py :: Memory.add_constraint_seed()
     ├─          src/grounded_memory/memory.py :: Memory.register_discovered_constraint_seeds()
     └─ Storage: ConstraintValidator._managed_constraints registry

  4. REPLAY (Evidence Collection)
     ├─ Offline pass of dynamic constraints over historical candidates
     ├─ Metrics: trigger_rate, projected_false_block_rate, coverage
     ├─ Location: src/grounded_memory/core/constraints.py :: ConstraintValidator.replay_dynamic_constraints()
     └─ Output: ConstraintReplayMetrics for each dynamic constraint

  5. PROMOTE (Governance Decision)
     ├─ Automatic promotion rules: min_trigger_rate, max_false_block_rate, min_candidates
     ├─ Manual promotion by governance admin (via set_lifecycle())
     ├─ Lifecycle transitions:
     │  ├─ PROPOSED → SHADOW (after initial observation or manual action)
     │  ├─ SHADOW   → ACTIVE  (after replay threshold met)
     │  └─ ACTIVE   → DEPRECATED (retire old rules)
     ├─ Location: src/grounded_memory/core/constraints.py :: ConstraintValidator.promote_dynamic_constraints()
     ├─          src/grounded_memory/memory.py :: Memory.promote_constraints()
     └─ Storage: ManagedConstraint.lifecycle field
```

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│  User writes facts: add_fact(), remember(), or LLM agent.process()       │
└───────────────┬────────────────────────────────────────────────────────┬──┘
                │                                                        │
        ┌───────▼────────────────────────────────────────────────────────▼──┐
        │  CandidateFact passes through grounding_operator.ground()         │
        │  ├─ Write-time validation (ConstraintValidator.validate)         │
        │  ├─ All constraints checked: ACTIVE, SHADOW, PROPOSED            │
        │  ├─ Shadow constraints observe but never block                   │
        │  └─ Decision: ACCEPT or REJECT                                  │
        └───────┬────────────────────────────────────────────────────────┬──┘
                │                                                        │
        ┌───────▼──────────────────────────────────────────────────────▼───┐
        │ VALIDATION SIGNAL recorded in ConstraintValidator                │
        │ ├─ candidate_id, relation, is_valid                             │
        │ ├─ violations (which constraints rejected)                       │
        │ ├─ attributes that were missing/present                         │
        │ └─ metadata: timestamp, confidence, attributes                  │
        │                                                                  │
        │ Signals accumulated in _validation_signals (limit: ~500 most-recent) │
        └───────┬──────────────────────────────────────────────────────────┘
                │
                │ Memory.discover_constraint_seeds()
                │  (Typically called periodically, e.g., daily batch)
                │
        ┌───────▼──────────────────────────────────────────────────────────┐
        │ PATTERN MINING: ConstraintSeedDiscoverer.discover()              │
        │ ├─ Group signals by relation type                               │
        │ ├─ For each relation:                                           │
        │ │  ├─ Count rejections vs. acceptances                          │
        │ │  ├─ Calculate rejection gap per missing field/attribute       │
        │ │  └─ If gap > min_gap, synthesize a seed constraint           │
        │ └─ Return top-N seeds (confidence-scored)                       │
        │                                                                  │
        │ Output: list[DiscoveredConstraintSeed]                          │
        │ ├─ constraint_id, name, description                            │
        │ ├─ relation_types, required_attribute_keys                     │
        │ ├─ confidence, evidence_count                                   │
        │ └─ mining_rule (e.g., "require_value")                         │
        └───────┬──────────────────────────────────────────────────────────┘
                │
                │ Optional: Manual review by governance admin
                │  Accept/filter seeds before auto-registration
                │
                │ Memory.register_discovered_constraint_seeds(seeds)
                │  (Default: lifecycle="shadow", source="agent")
                │
        ┌───────▼──────────────────────────────────────────────────────────┐
        │ RULE CREATION: Add to ConstraintValidator                        │
        │ ├─ SeedConstraintEvaluator wraps the seed logic                 │
        │ ├─ Register as ManagedConstraint:                               │
        │ │  ├─ lifecycle = SHADOW (observes, never blocks)               │
        │ │  ├─ source = "agent" (auto-discovered)                       │
        │ │  ├─ priority = 50 (lower than human ACTIVE)                  │
        │ │  └─ shadow_hits, shadow_violations counters start at 0       │
        │ └─ Constraint now participates in validation on next writes     │
        │                                                                  │
        │ Shadow mode: Mark observations, emit warnings, but allow writes │
        └───────┬──────────────────────────────────────────────────────────┘
                │
                │ Signal accumulation continues (more writes, more signals)
                │
                │ Memory.replay_dynamic_constraints(candidate_facts)
                │  (Batch operation: replay all dynamic constraints over
                │   all historical or selected candidates)
                │
        ┌───────▼──────────────────────────────────────────────────────────┐
        │ REPLAY & EVIDENCE COLLECTION                                      │
        │                                                                  │
        │ For each dynamic (PROPOSED/SHADOW/ACTIVE-agent) constraint:    │
        │  ├─ Evaluate all candidates against the constraint evaluator    │
        │  ├─ Tally:                                                      │
        │  │  ├─ violations (would block if active)                      │
        │  │  ├─ human_active_blocked (also blocked by existing rules)   │
        │  │  ├─ incremental_blocks (only this rule blocks it)          │
        │  │  └─ covered_existing_blocks (redundant with human rules)    │
        │  └─ Compute metrics:                                           │
        │     ├─ trigger_rate = violations / total                       │
        │     ├─ false_block_rate = incremental_blocks / total           │
        │     └─ miss_coverage = covered_existing_blocks / total         │
        │                                                                  │
        │ Output: dict[constraint_id, ConstraintReplayMetrics]           │
        └───────┬──────────────────────────────────────────────────────────┘
                │
                │ Memory.promote_constraints(replay_metrics)
                │  (Decision: which constraints graduate to next stage)
                │
        ┌───────▼──────────────────────────────────────────────────────────┐
        │ PROMOTE: Governance Decision & Lifecycle Transitions             │
        │                                                                  │
        │ For each constraint with replay evidence:                        │
        │  ├─ Check promotion criteria:                                   │
        │  │  ├─ min_candidates (default: 100)                           │
        │  │  ├─ min_trigger_rate (default: 0.01)                       │
        │  │  ├─ max_false_block_rate (default: 0.02)                   │
        │  │  └─ Other custom rules (extensible)                         │
        │  │                                                               │
        │  └─ Action:                                                     │
        │     ├─ PROPOSED → SHADOW if governance admin approves          │
        │     ├─ SHADOW → ACTIVE if metrics pass all thresholds         │
        │     ├─ ACTIVE → DEPRECATED if deemed harmful                  │
        │     └─ Manual override via set_lifecycle()                     │
        │                                                                  │
        │ Updated constraints now apply at different enforcement levels   │
        │ ├─ ACTIVE rules: block writes (immediate effect)               │
        │ ├─ SHADOW rules: observe warnings (educational)                │
        │ └─ PROPOSED rules: quiet observation (under review)            │
        └────────────────────────────────────────────────────────────────┘
                                    │
                                    │ (Loop restarts with new writes)
                                    │
                                    ▼
                          [Continuous improvement cycle]
```

---

## Implementation Details by Component

### 1. Feedback: Validation Signals

**Files:**
- `src/grounded_memory/core/constraints.py` :: `ConstraintValidator.validate()`
- `src/grounded_memory/core/constraints.py` :: `ConstraintValidator.record_validation_signal()`

**Key Methods:**
```python
def validate(
    candidate: CandidateFact,
    knowledge_state: KnowledgeState,
    stop_on_first_error: bool = False,
    runtime_context: dict[str, Any] | None = None,
    max_dynamic_constraints: int = 20,
    max_shadow_constraints: int = 10,
) -> ValidationResult:
    """
    Evaluate a fact against all registered constraints.
    
    Returns ValidationResult with:
    - is_valid: True if no violations
    - violations: List of ConstraintViolation (hard blocks)
    - warnings: List of ConstraintViolation (info/shadow)
    - checked_constraints: Which constraints ran
    """
    # ... evaluation logic ...
    self.record_validation_signal(candidate, result, runtime_context)
    return result

def record_validation_signal(
    candidate: CandidateFact,
    result: ValidationResult,
    runtime_context: dict[str, Any] | None = None,
) -> None:
    """Store a write-time governance signal for discovery."""
    self._validation_signals.append({
        "timestamp": datetime.utcnow().isoformat(),
        "candidate_id": candidate.id,
        "relation": candidate.relation.value,
        "candidate_confidence": candidate.confidence,
        "candidate_attributes": dict(candidate.attributes or {}),
        "has_object": candidate.object_entity_id is not None,
        "has_value": bool(candidate.value and str(candidate.value).strip()),
        "is_valid": result.is_valid,
        "violations": [v.constraint_id for v in result.violations],
        "warnings": [w.constraint_id for w in result.warnings],
        "runtime_context": runtime_context or {},
    })
```

**Signal Storage:**
- Bounded in-memory list (default limit: 500 most-recent)
- Retrieved via `list_validation_signals(limit=500)`

---

### 2. Pattern Mining: Discovery

**Files:**
- `src/grounded_memory/adapters/discovery.py` :: `ConstraintSeedDiscoverer`

**Key Methods:**
```python
class ConstraintSeedDiscoverer:
    """Mine validation signals and synthesize candidate constraint seeds."""
    
    def discover(
        self,
        *,
        validation_signals: list[dict[str, Any]],
        existing_constraint_ids: set[str] | None = None,
    ) -> list[DiscoveredConstraintSeed]:
        """
        Synthesize candidate constraints from signals.
        
        Mining strategy:
        1. Group signals by relation type
        2. For each relation:
           - Count rejections vs. acceptances
           - Identify systematic patterns (missing fields, attributes, values)
           - Calculate "rejection gap" (% rejected with field X missing - % accepted)
           - If gap > min_gap threshold, propose constraint
        3. Score by confidence and evidence count
        4. Return top-N suggestions
        """
        # Group signals by relation
        # Compute rejection rates for each pattern
        # Synthesize seeds for high-gap patterns
        # Return sorted by confidence descending
```

**Mining Rules Implemented:**
1. `_synthesize_require_value_seed()`: "Facts of relation R with missing value field are rejected at 40% rate, accepted at 5% rate → require value"
2. `_synthesize_required_attribute_key_seeds()`: "Facts of relation R missing attribute key K are rejected more often → make K required"
3. Extensible: Add more patterns (forbidden attributes, cross-field dependencies, etc.)

---

### 3. Rule Creation: Add Seeds to Validator

**Files:**
- `src/grounded_memory/memory.py` :: `Memory.add_constraint_seed()`
- `src/grounded_memory/memory.py` :: `Memory.register_discovered_constraint_seeds()`
- `src/grounded_memory/core/constraints.py` :: `ConstraintValidator.register_dynamic()`

**Key Flow:**
```python
# Option A: Manual seed definition
memory.add_constraint_seed(
    constraint_id="require_medication_value",
    name="Medication prescription must include value",
    description="Prescriptions without a drug name are unreliable",
    relation_types=["PRESCRIBED"],
    require_value=True,
    lifecycle="shadow",  # Start observing, don't block
)

# Option B: Auto-register discovered seeds
discovered = memory.discover_constraint_seeds(signal_limit=1000)
memory.register_discovered_constraint_seeds(
    discovered[:5],  # Top 5 by confidence
    lifecycle="shadow",
)

# Internally: wraps seed in SeedConstraintEvaluator
evaluator = SeedConstraintEvaluator(
    constraint_id=seed.constraint_id,
    # ... seed parameters ...
)

# Register as dynamic constraint
validator.register_dynamic(
    evaluator,
    lifecycle=ConstraintLifecycleStatus.SHADOW,
    source=ConstraintSource.AGENT,
    priority=50,
)
```

**Lifecycle States:**
- `PROPOSED`: Quiet observation only (agent-discovered, under review)
- `SHADOW`: Active observation + warnings, but never blocks writes
- `ACTIVE`: Full enforcement (blocks writes on violation)
- `DEPRECATED`: No longer applied

---

### 4. Replay: Evidence Gathering

**Files:**
- `src/grounded_memory/core/constraints.py` :: `ConstraintValidator.replay_dynamic_constraints()`
- `src/grounded_memory/memory.py` :: `Memory.get_constraint_replay_metrics()`

**Key Logic:**
```python
def replay_dynamic_constraints(
    self,
    candidates: list[CandidateFact],
    knowledge_state: KnowledgeState,
) -> dict[str, ConstraintReplayMetrics]:
    """
    Offline pass: evaluate all dynamic constraints over historical candidates.
    
    For each candidate:
      - Check if human-authored ACTIVE constraints block it
      - Check if each dynamic constraint would block it
    
    For each dynamic constraint, tally:
      - violations: total blocks by this rule
      - incremental_blocks: blocks only this rule catches (not caught by human rules)
      - covered_existing_blocks: redundant blocks (also blocked by human rules)
    
    Compute rates:
      - trigger_rate = violations / total_candidates
      - false_block_rate = incremental_blocks / total_candidates
    """
    metrics: dict[str, ConstraintReplayMetrics] = {}
    
    for candidate in candidates:
        human_blocked = False
        for human_constraint in human_active_constraints:
            if human_constraint.evaluate(candidate, knowledge_state):
                human_blocked = True
                break
        
        for dynamic_constraint in dynamic_constraints:
            violation = dynamic_constraint.evaluate(candidate, knowledge_state)
            if violation is None:
                continue
            
            metrics[id].violations += 1
            if human_blocked:
                metrics[id].covered_existing_blocks += 1
            else:
                metrics[id].incremental_blocks += 1
    
    return metrics
```

---

### 5. Promote: Governance & Transitions

**Files:**
- `src/grounded_memory/core/constraints.py` :: `ConstraintValidator.promote_dynamic_constraints()`
- `src/grounded_memory/memory.py` :: `Memory.promote_constraints()`
- `src/grounded_memory/core/constraints.py` :: `ConstraintValidator.set_lifecycle()`

**Promotion Rules (Default):**
```python
def promote_dynamic_constraints(
    self,
    replay_metrics: dict[str, ConstraintReplayMetrics],
    *,
    min_trigger_rate: float = 0.01,           # Constraint must fire on ≥1% of facts
    max_projected_false_block_rate: float = 0.02,  # But not block >2% falsely
    min_candidates: int = 100,                # Minimum evidence base
) -> list[str]:
    """
    Promote constraints from PROPOSED/SHADOW → ACTIVE if they meet criteria.
    
    Decision logic:
    ├─ Must have evaluated ≥ min_candidates facts
    ├─ Must trigger on ≥ min_trigger_rate of those facts
    ├─ But false positive rate (incremental_blocks) must be ≤ max_projected_false_block_rate
    └─ If all pass, update lifecycle to ACTIVE and return constraint_id
    """
    promoted = []
    for constraint_id, metric in replay_metrics.items():
        if metric.evaluated_candidates < min_candidates:
            continue
        if metric.trigger_rate < min_trigger_rate:
            continue
        if metric.projected_false_block_rate > max_projected_false_block_rate:
            continue
        
        # Safe to promote
        self.set_lifecycle(constraint_id, ConstraintLifecycleStatus.ACTIVE)
        promoted.append(constraint_id)
    
    return promoted
```

---

## Example Workflow: Healthcare Medication Constraint Discovery

### Step 1: Write Facts Over Time
```python
memory = Memory(adapter="healthcare")

# Day 1-7: Multiple writes for PRESCRIBED relation
for day in range(1, 8):
    patient = memory.add_entity("Patient_1", entity_type="person")
    drug = memory.add_entity("Ibuprofen", entity_type="medication")
    
    # Some writes include medication name (has_value=True)
    # Some don't (has_value=False, only object_entity_id)
    memory.add_fact(
        subject_id=patient["entity"]["id"],
        relation="PRESCRIBED",
        object_id=drug["entity"]["id"],
        value="Ibuprofen 400mg" if day % 3 == 0 else None,
        confidence=0.9,
    )
```

### Step 2: Signals Accumulate
Validator records ~50 signals for PRESCRIBED relation:
- 15 with `has_value=True` → all accepted
- 35 with `has_value=False` → 20 rejected, 15 accepted
- Rejection gap: (20/35) - (0/15) = 0.57 > 0.35 threshold

### Step 3: Discovery Synthesizes Seed
```python
seeds = memory.discover_constraint_seeds(signal_limit=500)
# Returns: [DiscoveredConstraintSeed(
#   constraint_id="seed_auto_require_value_PRESCRIBED",
#   name="Prescription must include medication value",
#   description="...rejected at 57% when value missing...",
#   confidence=0.92,
#   evidence_count=50,
#   mining_rule="require_value"
# )]
```

### Step 4: Register in Shadow Mode
```python
memory.register_discovered_constraint_seeds(
    seeds,
    lifecycle="shadow",
)
# Constraint now runs on all future writes, observes patterns, never blocks
```

### Step 5: Replay & Gather Evidence
```python
# After 2 weeks of operation, replay on historical facts
all_facts = memory.system.memory_store.get_all_validated_facts()
metrics = memory.get_constraint_replay_metrics()

# Output:
# {
#   "seed_auto_require_value_PRESCRIBED": ConstraintReplayMetrics(
#     evaluated_candidates=200,
#     violations=68,  # 34% trigger rate
#     incremental_blocks=5,  # Only 2.5% false blocks
#     covered_existing_blocks=63,
#   )
# }
```

### Step 6: Promote to Active
```python
promoted = memory.promote_constraints(
    replay_metrics,
    min_trigger_rate=0.01,
    max_projected_false_block_rate=0.02,
    min_candidates=100,
)
# "seed_auto_require_value_PRESCRIBED" passes all criteria:
# ├─ 200 ≥ 100 candidates ✓
# ├─ 0.34 ≥ 0.01 trigger_rate ✓
# └─ 0.025 ≤ 0.02 false_block_rate... ✗ (marginal, might tune threshold)
```

---

## Integration with External Knowledge Sources

The constraint learning loop **can** be seeded/enriched by external data sources:

1. **Knowledge Base Integration** (via `src/grounded_memory/adapters/healthcare/knowledge.py`):
   - Load drug interactions, therapeutic classes, allergy data from openFDA, RxNorm, etc.
   - Use as semantic features for constraint mining (e.g., "drug in NSAID class has higher rejection rate")

2. **Feedback Loop**:
   - Rejected facts can be tagged with domain reason ("allergy cross-reactivity")
   - Discovery can weight signals by domain category
   - Mined constraints reflect real clinical patterns

3. **Human-in-the-Loop**:
   - Governance admin reviews discovered seeds before auto-registration
   - Seeds can be edited/refined before promotion
   - Promotion thresholds can be tuned per domain/team

---

## Configuration & Tuning

### Constraint Discovery Parameters
Location: `Memory.discover_constraint_seeds()`
```python
min_samples_per_relation = 20        # Min signals to analyze
min_rejections_per_relation = 6      # Min rejections needed
min_gap = 0.35                       # Min rejection gap to trigger
target_false_block_rate = 0.10       # Acceptable false positive rate
max_suggestions = 20                 # Top-N seeds to return
```

### Promotion Criteria
Location: `Memory.promote_constraints()`
```python
min_trigger_rate = 0.01              # Must fire on ≥1% of facts
max_projected_false_block_rate = 0.02    # False block rate ≤2%
min_candidates = 100                 # Minimum evidence base
```

### Validation Execution
Location: `Memory.add()` / `Memory.add_fact()`
```python
max_dynamic_constraints = 20         # Cap on agent-discovered rules active per write
max_shadow_constraints = 10          # Cap on proposed/shadow rules active per write
```

---

## Key Axioms

**Axiom CV1 (Shadow Safety):**
Constraints in `PROPOSED` or `SHADOW` state never block writes; they only observe and record violations. This ensures safety during the learning phase.

**Axiom CV2 (Evidence-Driven Promotion):**
A constraint may be promoted to `ACTIVE` only after offline replay evidence demonstrates sufficient trigger rate and sufficiently low false block rate.

**Axiom CV3 (Governance Transparency):**
All lifecycle transitions (especially SHADOW→ACTIVE) are observable and auditable. Governance administrators have explicit control.

---

## Next Steps: External Data Integration

The knowledge base (Step 1 in feedback loop) can be dramatically enriched:

1. **openFDA Loader** (`src/grounded_memory/adapters/healthcare/loaders/openfda.py`):
   - Fetch drug labels to extract ingredients, contraindications, cross-reactivities
   - Merge into `InMemoryKnowledgeBase`

2. **RxNorm Loader** (`src/grounded_memory/adapters/healthcare/loaders/rxnorm.py`):
   - Map drug names → RxCUI identifiers
   - Fetch canonical interaction pairs
   - Merge interaction database

3. **Config-Driven Initialization** (`configs/healthcare_kb.yaml`):
   - Specify which external sources to load at startup
   - Control caching, rate limits, refresh intervals
   - Enable/disable per-environment

See `docs/architecture/CONSTRAINT_LEARNING_LOOP_IMPLEMENTATION.md` (next file) for detailed API examples and runnable code.

