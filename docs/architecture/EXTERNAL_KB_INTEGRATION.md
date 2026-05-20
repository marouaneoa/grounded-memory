# External Knowledge Integration: openFDA + RxNorm for Constraint Learning

This guide shows how to integrate real drug interaction and allergy data into the constraint learning loop via openFDA and RxNorm APIs.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│            Constraint Learning Loop                             │
│                                                                 │
│  Feedback → Pattern Mining → Rule Creation → Replay → Promote  │
│     ▲                                                           │
│     │                                                           │
│     └──────────────────────────────────────────────────────────│
│                                                                 │
│  Enhanced by External Knowledge Sources:                        │
│  ┌──────────────────┐  ┌──────────────┐  ┌────────────────┐   │
│  │   openFDA API    │  │  RxNorm API  │  │  Local Cache   │   │
│  ├──────────────────┤  ├──────────────┤  ├────────────────┤   │
│  │ Drug labels      │  │ RXCUI lookup │  │ drug_kb.json   │   │
│  │ Ingredients      │  │ Interactions │  │ interactions   │   │
│  │ Contraind.       │  │ Cross-react  │  │ cache.pkl      │   │
│  │ Adverse events   │  │ Therapeutic  │  │                │   │
│  └──────────────────┘  └──────────────┘  └────────────────┘   │
│           ▼                   ▼                    ▼            │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  InMemoryKnowledgeBase (knowledge.py)                 │    │
│  ├────────────────────────────────────────────────────────┤    │
│  │ • aliases (brand→generic mapping)                      │    │
│  │ • ingredients (drug→active compounds)                  │    │
│  │ • therapeutic_classes (drug→class mapping)            │    │
│  │ • allergy_cross_reactivity (allergen→cross-reactants) │    │
│  │ • major_interactions (frozenset pairs)                │    │
│  │ • moderate_interactions (frozenset pairs)             │    │
│  └────────────────────────────────────────────────────────┘    │
│           ▼                                                     │
│  Used by healthcare constraints during validation              │
│  (check_major_interaction, get_cross_reactive_ingredients)     │
└─────────────────────────────────────────────────────────────────┘
```

## File Structure

```
src/grounded_memory/adapters/healthcare/
├── knowledge.py                      # Core KB (already refactored)
├── constraints.py                    # Healthcare constraint evaluators
├── loaders/                          # NEW: External data loaders
│   ├── __init__.py
│   ├── openfda.py                   # openFDA label → ingredients/allergies
│   ├── rxnorm.py                    # RxNorm API → RXCUI + interactions
│   ├── cache.py                     # Caching layer (Redis/disk)
│   └── merger.py                    # Merge multiple KB sources
└── kb_manager.py                    # NEW: Startup & lifecycle mgmt

configs/
└── healthcare_kb.yaml               # NEW: Data source configuration

benchmarks/
└── test_kb_integration.py            # NEW: Integration tests
```

## Implementation Plan

### 1. openFDA Loader (`src/grounded_memory/adapters/healthcare/loaders/openfda.py`)

**Purpose:** Fetch drug labels from openFDA API, extract ingredients and allergy cross-reactivity info.

**Key Functions:**
```python
def fetch_drug_label(
    drug_name: str,
    api_key: str | None = None,
    cache_ttl_hours: int = 168,  # 1 week
) -> dict[str, Any]:
    """
    Fetch FDA drug label for a medication.
    
    Returns:
    {
        "drug_name": "Ibuprofen",
        "brand_names": ["Advil", "Motrin", ...],
        "active_ingredients": ["ibuprofen"],
        "therapeutic_classes": ["nsaid"],
        "contraindications": [...],
        "allergies": [...cross-reactivity info...],
        "interactions_summary": [...],
    }
    """
    # Call FDA API /drug/label.json endpoint
    # Parse response and extract fields
    # Cache result

def extract_ingredients_from_labels(
    labels: list[dict[str, Any]],
) -> InMemoryKnowledgeBase:
    """
    Convert openFDA labels into KB entries.
    
    Returns InMemoryKnowledgeBase with:
    - aliases: brand_name → canonical drug_name
    - ingredients: drug_name → set of active compounds
    - therapeutic_classes: drug_name → classes from FDA label
    """

def fetch_batch_labels(
    drug_names: list[str],
    batch_size: int = 10,
    api_key: str | None = None,
    max_workers: int = 4,
    rate_limit_delay: float = 0.5,
) -> InMemoryKnowledgeBase:
    """
    Fetch labels for a batch of drugs with rate-limiting and retry logic.
    """
```

**Caveats:**
- openFDA has rate limits (~500 req/sec for authenticated requests)
- Needs `FD_API_KEY` environment variable (optional but recommended for higher rates)
- Drug name matching can be fuzzy; may need manual curation

---

### 2. RxNorm Loader (`src/grounded_memory/adapters/healthcare/loaders/rxnorm.py`)

**Purpose:** Map drug names → RxCUI (canonical identifiers) and fetch official interaction pairs.

**Key Functions:**
```python
def lookup_rxcui(
    drug_name: str,
    search_type: str = "all",  # "all" or "exact"
    cache_ttl_hours: int = 168,
) -> str | None:
    """
    Lookup RxCUI (canonical identifier) for a drug name.
    
    RxNorm API: /rxcui?name={drug_name}&search={search_type}
    
    Returns RXCUI string or None if not found.
    """

def fetch_interactions_for_rxcui(
    rxcui: str,
    cache_ttl_hours: int = 168,
) -> dict[str, list[str]]:
    """
    Fetch known interactions for a drug (by RXCUI).
    
    RxNorm Interaction API: /interaction/interaction.json?rxcui={rxcui}
    
    Returns:
    {
        "major": ["drug_A", "drug_B", ...],
        "moderate": ["drug_C", "drug_D", ...],
        "minor": [...],
    }
    """

def build_kb_from_rxnorm(
    drug_names: list[str],
    include_minor: bool = False,
    max_workers: int = 4,
    rate_limit_delay: float = 0.2,
) -> InMemoryKnowledgeBase:
    """
    Lookup RxCUI for each drug, fetch interactions, build KB.
    
    Returns InMemoryKnowledgeBase with:
    - major_interactions: frozenset pairs from RxNorm major
    - moderate_interactions: frozenset pairs from RxNorm moderate
    """
```

**Advantages:**
- Free, no API key required
- High-quality official interaction database (curated by NLM)
- Covers ~300k drug combinations with high precision

---

### 3. Caching Layer (`src/grounded_memory/adapters/healthcare/loaders/cache.py`)

**Purpose:** Reduce API calls via disk cache or Redis.

```python
class KnowledgeBaseCache:
    def __init__(self, backend: str = "disk", ttl_hours: int = 168):
        # backend: "disk" (SQLite) or "redis"
        self.backend = backend
        self.ttl_hours = ttl_hours
    
    def get(self, key: str) -> Any | None:
        # Retrieve from cache, check TTL
    
    def put(self, key: str, value: Any) -> None:
        # Store in cache with TTL
    
    def evict_expired(self) -> int:
        # Clean up stale entries, return count
```

---

### 4. Config-Driven Initialization (`configs/healthcare_kb.yaml`)

```yaml
# Healthcare Knowledge Base Configuration

knowledge_base:
  # External data sources to load at startup
  sources:
    - name: "rxnorm_interactions"
      type: "rxnorm"
      enabled: true
      params:
        # List of drugs to fetch (or "*" for all known)
        drugs:
          - "Ibuprofen"
          - "Warfarin"
          - "Amoxicillin"
          - "Aspirin"
        include_minor: false
        max_workers: 4
        rate_limit_delay: 0.2
      cache_ttl_hours: 168  # 1 week
    
    - name: "openfda_labels"
      type: "openfda"
      enabled: true
      params:
        drugs:
          - "Ibuprofen"
          - "Warfarin"
          - "Amoxicillin"
        max_workers: 2
        rate_limit_delay: 0.5
      cache_ttl_hours: 168
    
    - name: "local_json"
      type: "json_file"
      enabled: true
      params:
        path: "configs/healthcare_drugs.json"
  
  # Merge strategy: later sources override earlier ones
  merge_strategy: "override"  # or "union"
  
  # Auto-refresh interval
  auto_refresh:
    enabled: true
    interval_hours: 24
    on_startup: true

# Constraint learning parameters
constraint_discovery:
  enabled: true
  signal_batch_size: 500
  min_gap: 0.35
  min_samples_per_relation: 20
  
promotion:
  min_trigger_rate: 0.01
  max_false_block_rate: 0.02
  min_candidates: 100
```

**Load at Startup:**
```python
# src/grounded_memory/adapters/healthcare/kb_manager.py

def initialize_knowledge_base(config_path: str = "configs/healthcare_kb.yaml"):
    """
    Load config and initialize KB with external sources.
    """
    config = yaml.safe_load(Path(config_path).read_text())
    sources_config = config.get("knowledge_base", {}).get("sources", [])
    
    for source in sources_config:
        if not source.get("enabled"):
            continue
        
        source_type = source.get("type")
        params = source.get("params", {})
        
        if source_type == "rxnorm":
            kb = rxnorm.build_kb_from_rxnorm(**params)
        elif source_type == "openfda":
            kb = openfda.fetch_batch_labels(**params)
        elif source_type == "json_file":
            kb = knowledge.load_json_file(params["path"])
        
        knowledge.register_source(kb)
```

---

## Integration Points with Constraint Learning

### How External Data Feeds the Loop

```
1. FEEDBACK (Enhanced)
   ├─ Facts written with medication names
   ├─ KB provides semantic features:
   │  ├─ Is drug an NSAID? (therapeutic class)
   │  ├─ Known allergy cross-reactants?
   │  ├─ Major interaction with current meds?
   │  └─ Ingredient composition
   └─ Constraints use KB to make decisions

2. PATTERN MINING (Enhanced)
   ├─ Signals grouped by drug class, not just name
   ├─ Discovery can detect:
   │  ├─ "All NSAIDs in this patient rejected" → class-level constraint
   │  ├─ "Warfarin + NSAID pairs always rejected" → interaction-driven
   │  └─ "Missing allergy field causes high rejection" → data quality
   └─ Higher-level patterns possible

3. REPLAY (Enhanced)
   ├─ Metrics include interaction coverage
   ├─ Can measure: "Does this constraint catch real interactions?"
   └─ Evidence more domain-grounded

4. PROMOTE (Enhanced)
   ├─ Criteria can include domain knowledge
   ├─ E.g., "promote if interaction is in RxNorm major list"
   └─ Safety thresholds can be tuned per class
```

---

## Runnable Example: Load KB and Use in Constraints

```python
# scripts/initialize_healthcare_kb.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))

from grounded_memory.adapters.healthcare import knowledge, loaders

# Load from config
def setup_healthcare_kb():
    """Initialize KB with external sources."""
    
    # 1. Fetch RxNorm interactions
    print("Loading RxNorm interactions...")
    rxnorm_kb = loaders.rxnorm.build_kb_from_rxnorm(
        drug_names=[
            "Warfarin", "Ibuprofen", "Aspirin", 
            "Amoxicillin", "Lisinopril"
        ],
        include_minor=False,
        max_workers=2,
    )
    knowledge.register_source(rxnorm_kb)
    print(f"  Loaded {len(rxnorm_kb.major_interactions)} major interactions")
    
    # 2. Fetch openFDA labels
    print("Loading openFDA labels...")
    fda_kb = loaders.openfda.fetch_batch_labels(
        drug_names=[
            "Warfarin", "Ibuprofen", "Aspirin", 
            "Amoxicillin", "Lisinopril"
        ],
        max_workers=2,
    )
    knowledge.register_source(fda_kb)
    print(f"  Loaded {len(fda_kb.aliases)} drug aliases")
    
    # 3. Verify loading worked
    print("\nKnowledge Base Summary:")
    print(f"  Major interactions: {len(knowledge._KB.major_interactions)}")
    print(f"  Moderate interactions: {len(knowledge._KB.moderate_interactions)}")
    print(f"  Drug aliases: {len(knowledge._KB.aliases)}")
    print(f"  Ingredients known for: {len(knowledge._KB.ingredients)} drugs")
    print(f"  Therapeutic classes: {len(knowledge._KB.therapeutic_classes)} drugs")
    
    # 4. Test constraint logic
    print("\nConstraint Logic Verification:")
    major = knowledge.check_major_interaction("Warfarin", "Aspirin")
    print(f"  Warfarin + Aspirin (major): {major}")
    
    moderate = knowledge.check_moderate_interaction("Ibuprofen", "Lisinopril")
    print(f"  Ibuprofen + Lisinopril (moderate): {moderate}")
    
    ingredients = knowledge.get_drug_ingredients("Amoxicillin")
    print(f"  Amoxicillin ingredients: {ingredients}")

if __name__ == "__main__":
    setup_healthcare_kb()
```

**Run:**
```bash
cd /Users/faycalamrouche/Desktop/GroundedMemory
python scripts/initialize_healthcare_kb.py
```

---

## Rate Limiting & Error Handling

**Rate Limit Strategy:**
- openFDA: default ~4 requests/second
- RxNorm: less restrictive, but be respectful (~2 req/sec)
- Use `rate_limit_delay` parameter in loaders

**Error Handling:**
```python
# Graceful fallback: if API is down, use cached data
try:
    kb = rxnorm.build_kb_from_rxnorm(drugs, max_workers=4)
except requests.ConnectionError:
    log.warning("RxNorm API unavailable; using cached KB")
    kb = load_cache("rxnorm_cache.json")
```

---

## Testing

**Test File:** `benchmarks/test_kb_integration.py`

```python
def test_rxnorm_interactions_loaded():
    """Verify RxNorm data loaded into KB."""
    kb = loaders.rxnorm.build_kb_from_rxnorm(
        ["Warfarin", "Aspirin"],
        include_minor=False,
    )
    assert len(kb.major_interactions) > 0, "No major interactions loaded"
    # Check specific known pair
    assert frozenset({"warfarin", "aspirin"}) in kb.major_interactions

def test_constraint_uses_external_kb():
    """Verify constraint logic respects loaded KB."""
    knowledge.register_source(create_test_kb())
    assert knowledge.check_major_interaction("Warfarin", "Aspirin")
    assert not knowledge.check_major_interaction("Aspirin", "Acetaminophen")

def test_config_driven_initialization():
    """Verify KB initializes from config file."""
    kb = initialize_healthcare_kb("configs/healthcare_kb.yaml")
    assert kb is not None
```

---

## Next Steps (Incremental)

1. **Phase 1 (This PR):**
   - Refactor `knowledge.py` ✓ (already done)
   - Create `loaders/rxnorm.py` (RxNorm API + caching)
   - Create `loaders/openfda.py` (openFDA API + caching)
   - Add `configs/healthcare_kb.yaml` config template

2. **Phase 2:**
   - Add `kb_manager.py` for startup initialization
   - Wire into `Memory.__init__()` to auto-load on startup
   - Add tests

3. **Phase 3 (Future):**
   - Periodic refresh daemon (background task)
   - Monitoring/alerting for API health
   - Telemetry: track constraint decisions vs. KB coverage
   - A/B testing: compare old KB vs. RxNorm-enriched KB

---

## References

- **RxNorm API Docs:** https://lhncbc.nlm.nih.gov/RxNav/APIs/api-RxNorm.html
- **openFDA API Docs:** https://open.fda.gov/apis/
- **FDA Adverse Event Reporting System (FAERS):** https://open.fda.gov/apis/drug/event/

