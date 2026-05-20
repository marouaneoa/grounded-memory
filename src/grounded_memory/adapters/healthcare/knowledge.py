"""Scalable Healthcare Knowledge Provider

This module provides a lightweight, extensible in-memory knowledge base for
drug names, ingredient mappings, therapeutic classes, allergy cross-reactivity
and drug-drug interaction pairs. It keeps the same function-level public API
used by the healthcare constraint adapters but allows adding/loading external
sources (JSON/CSV/YAML) and merging multiple sources to scale beyond the
hardcoded demo data.

Design goals:
- Backwards-compatible functions: `normalize_drug_name`, `get_drug_ingredients`,
  `get_therapeutic_classes`, `get_cross_reactive_ingredients`,
  `check_major_interaction`, `check_moderate_interaction`.
- File-backed loaders and merging utilities for maintainable source-of-truth
- Simple provider registration so tests can inject richer datasets
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Default demo data (kept small) to preserve existing behavior when nothing
# else is registered. Larger datasets should be loaded via `register_source`.
DEFAULT_ALERGY_CROSS_REACTIVITY: dict[str, set[str]] = {
    "penicillin": {"amoxicillin", "ampicillin", "piperacillin", "ticarcillin"},
    "amoxicillin": {"penicillin", "ampicillin", "piperacillin", "ticarcillin"},
    "cephalosporins": {"cephalexin", "cefuroxime", "ceftriaxone"},
    "sulfa": {"sulfamethoxazole", "trimethoprim-sulfamethoxazole", "bactrim"},
    "nsaid": {"ibuprofen", "naproxen", "diclofenac", "celecoxib", "aspirin"},
    "aspirin": {"ibuprofen", "naproxen", "diclofenac"},
    "ace_inhibitors": {"lisinopril", "enalapril", "ramipril", "benazepril"},
    "statins": {"simvastatin", "atorvastatin", "rosuvastatin", "pravastatin"},
}

DEFAULT_DRUG_ALIASES: dict[str, str] = {
    # Normalize plural/abbreviated allergen forms the LLM commonly produces
    "nsaid": "nsaid",
    "nsaids": "nsaid",
    "sulfa drugs": "sulfa",
    "ace inhibitor": "ace inhibitor",
    "ace inhibitors": "ace inhibitor",
    # Brand → generic
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "aleve": "naproxen",
    "tylenol": "acetaminophen",
    "panadol": "acetaminophen",
    "augmentin": "amoxicillin-clavulanate",
    "zithromax": "azithromycin",
    "lasix": "furosemide",
    "coumadin": "warfarin",
    "plavix": "clopidogrel",
    "lipitor": "atorvastatin",
    "zocor": "simvastatin",
    "crestor": "rosuvastatin",
    "glucophage": "metformin",
    "prilosec": "omeprazole",
    "nexium": "esomeprazole",
}

DEFAULT_DRUG_INGREDIENTS: dict[str, set[str]] = {
    "amoxicillin": {"amoxicillin", "penicillin"},
    "ampicillin": {"ampicillin", "penicillin"},
    "amoxicillin-clavulanate": {"amoxicillin", "clavulanate", "penicillin"},
    "cephalexin": {"cephalexin", "cephalosporin"},
    "ceftriaxone": {"ceftriaxone", "cephalosporin"},
    "azithromycin": {"azithromycin", "macrolide"},
    "clarithromycin": {"clarithromycin", "macrolide"},
    "erythromycin": {"erythromycin", "macrolide"},
    "trimethoprim-sulfamethoxazole": {"trimethoprim", "sulfamethoxazole", "sulfa"},
    "bactrim": {"trimethoprim", "sulfamethoxazole", "sulfa"},
    "ibuprofen": {"ibuprofen", "nsaid"},
    "naproxen": {"naproxen", "nsaid"},
    "diclofenac": {"diclofenac", "nsaid"},
    "celecoxib": {"celecoxib", "nsaid"},
    "aspirin": {"aspirin", "nsaid", "salicylate"},
    "warfarin": {"warfarin", "coumarin"},
    "amiodarone": {"amiodarone"},
    "clopidogrel": {"clopidogrel"},
    "omeprazole": {"omeprazole", "ppi"},
    "esomeprazole": {"esomeprazole", "ppi"},
    "simvastatin": {"simvastatin", "statin"},
    "atorvastatin": {"atorvastatin", "statin"},
    "rosuvastatin": {"rosuvastatin", "statin"},
    "pravastatin": {"pravastatin", "statin"},
    "metformin": {"metformin", "biguanide"},
    "insulin glargine": {"insulin glargine", "insulin"},
    "insulin lispro": {"insulin lispro", "insulin"},
    "lisinopril": {"lisinopril", "ace inhibitor"},
    "enalapril": {"enalapril", "ace inhibitor"},
    "ramipril": {"ramipril", "ace inhibitor"},
    "losartan": {"losartan", "arb"},
    "valsartan": {"valsartan", "arb"},
    "furosemide": {"furosemide", "loop diuretic"},
    "spironolactone": {"spironolactone", "potassium-sparing diuretic"},
    "nitroglycerin": {"nitroglycerin", "nitrate"},
    "isosorbide": {"isosorbide", "nitrate"},
    "sildenafil": {"sildenafil", "pde5 inhibitor"},
    "tadalafil": {"tadalafil", "pde5 inhibitor"},
    "sertraline": {"sertraline", "ssri"},
    "fluoxetine": {"fluoxetine", "ssri"},
    "citalopram": {"citalopram", "ssri"},
    "tramadol": {"tramadol", "opioid"},
    "methadone": {"methadone", "opioid"},
    "digoxin": {"digoxin", "cardiac glycoside"},
    "verapamil": {"verapamil", "calcium channel blocker"},
    "diltiazem": {"diltiazem", "calcium channel blocker"},
    "acetaminophen": {"acetaminophen"},
}

DEFAULT_DRUG_THERAPEUTIC_CLASSES: dict[str, set[str]] = {
    "amoxicillin": {"beta_lactam_antibiotic"},
    "ampicillin": {"beta_lactam_antibiotic"},
    "amoxicillin-clavulanate": {"beta_lactam_antibiotic"},
    "cephalexin": {"cephalosporin_antibiotic"},
    "ceftriaxone": {"cephalosporin_antibiotic"},
    "azithromycin": {"macrolide_antibiotic"},
    "clarithromycin": {"macrolide_antibiotic"},
    "erythromycin": {"macrolide_antibiotic"},
    "trimethoprim-sulfamethoxazole": {"sulfonamide_antibiotic"},
    "bactrim": {"sulfonamide_antibiotic"},
    "ibuprofen": {"nsaid"},
    "naproxen": {"nsaid"},
    "diclofenac": {"nsaid"},
    "celecoxib": {"nsaid"},
    "aspirin": {"nsaid", "antiplatelet"},
    "warfarin": {"anticoagulant"},
    "clopidogrel": {"antiplatelet"},
    "simvastatin": {"statin"},
    "atorvastatin": {"statin"},
    "rosuvastatin": {"statin"},
    "pravastatin": {"statin"},
    "metformin": {"antidiabetic"},
    "insulin glargine": {"antidiabetic", "insulin"},
    "insulin lispro": {"antidiabetic", "insulin"},
    "lisinopril": {"ace_inhibitor"},
    "enalapril": {"ace_inhibitor"},
    "ramipril": {"ace_inhibitor"},
    "losartan": {"arb"},
    "valsartan": {"arb"},
    "furosemide": {"loop_diuretic"},
    "spironolactone": {"potassium_sparing_diuretic"},
    "nitroglycerin": {"nitrate"},
    "isosorbide": {"nitrate"},
    "sildenafil": {"pde5_inhibitor"},
    "tadalafil": {"pde5_inhibitor"},
    "sertraline": {"ssri"},
    "fluoxetine": {"ssri"},
    "citalopram": {"ssri"},
    "tramadol": {"opioid"},
    "methadone": {"opioid"},
    "digoxin": {"cardiac_glycoside"},
    "verapamil": {"calcium_channel_blocker"},
    "diltiazem": {"calcium_channel_blocker"},
    "amiodarone": {"antiarrhythmic"},
}

# Clinically established drug-drug interactions used as a default seed so the
# healthcare adapter's safety constraints fire out-of-the-box. Production
# deployments should extend this via `register_source` with RxNorm / openFDA.
DEFAULT_MAJOR_INTERACTIONS: set[tuple[str, str]] = {
    ("warfarin", "amiodarone"),
    ("warfarin", "aspirin"),
    ("warfarin", "ibuprofen"),
    ("warfarin", "naproxen"),
    ("warfarin", "clopidogrel"),
    ("warfarin", "trimethoprim-sulfamethoxazole"),
    ("warfarin", "clarithromycin"),
    ("amiodarone", "digoxin"),
    ("amiodarone", "simvastatin"),
    ("amiodarone", "methadone"),
    ("simvastatin", "clarithromycin"),
    ("digoxin", "verapamil"),
    ("digoxin", "diltiazem"),
    ("clopidogrel", "omeprazole"),
    ("clopidogrel", "esomeprazole"),
    ("sildenafil", "nitroglycerin"),
    ("sildenafil", "isosorbide"),
    ("tadalafil", "nitroglycerin"),
    ("tadalafil", "isosorbide"),
    ("tramadol", "sertraline"),
    ("tramadol", "fluoxetine"),
}

DEFAULT_MODERATE_INTERACTIONS: set[tuple[str, str]] = {
    ("aspirin", "ibuprofen"),
    ("aspirin", "naproxen"),
    ("ibuprofen", "lisinopril"),
    ("ibuprofen", "enalapril"),
    ("ibuprofen", "ramipril"),
    ("ibuprofen", "losartan"),
    ("ibuprofen", "valsartan"),
    ("naproxen", "lisinopril"),
    ("naproxen", "losartan"),
    ("metformin", "furosemide"),
    ("sertraline", "warfarin"),
    ("fluoxetine", "warfarin"),
    ("spironolactone", "lisinopril"),
    ("spironolactone", "enalapril"),
}


@dataclass
class InMemoryKnowledgeBase:
    aliases: dict[str, str] = field(default_factory=dict)
    ingredients: dict[str, set[str]] = field(default_factory=dict)
    therapeutic_classes: dict[str, set[str]] = field(default_factory=dict)
    allergy_cross_reactivity: dict[str, set[str]] = field(default_factory=dict)
    major_interactions: set[frozenset] = field(default_factory=set)
    moderate_interactions: set[frozenset] = field(default_factory=set)

    def merge(self, other: InMemoryKnowledgeBase) -> None:
        for k, v in other.aliases.items():
            self.aliases.setdefault(k, v)
        for k, v in other.ingredients.items():
            self.ingredients.setdefault(k, set()).update(v)
        for k, v in other.therapeutic_classes.items():
            self.therapeutic_classes.setdefault(k, set()).update(v)
        for k, v in other.allergy_cross_reactivity.items():
            self.allergy_cross_reactivity.setdefault(k, set()).update(v)
        self.major_interactions.update(other.major_interactions)
        self.moderate_interactions.update(other.moderate_interactions)


# Single shared knowledge base instance. Tests or startup code can replace
# or extend this via the register_source / load_* helpers.
_KB = InMemoryKnowledgeBase()


def _init_defaults() -> None:
    _KB.aliases.update({k.lower(): v for k, v in DEFAULT_DRUG_ALIASES.items()})
    for k, v in DEFAULT_DRUG_INGREDIENTS.items():
        _KB.ingredients[k.lower()] = {x.lower() for x in v}
    for k, v in DEFAULT_DRUG_THERAPEUTIC_CLASSES.items():
        _KB.therapeutic_classes[k.lower()] = {x.lower() for x in v}
    for k, v in DEFAULT_ALERGY_CROSS_REACTIVITY.items():
        _KB.allergy_cross_reactivity[k.lower()] = {x.lower() for x in v}
    for a, b in DEFAULT_MAJOR_INTERACTIONS:
        _KB.major_interactions.add(frozenset({a.lower(), b.lower()}))
    for a, b in DEFAULT_MODERATE_INTERACTIONS:
        _KB.moderate_interactions.add(frozenset({a.lower(), b.lower()}))


_init_defaults()


def register_source(kb: InMemoryKnowledgeBase) -> None:
    """Merge another knowledge base into the active provider."""
    _KB.merge(kb)


def load_json_file(path: str | Path) -> InMemoryKnowledgeBase:
    p = Path(path)
    kb = InMemoryKnowledgeBase()
    if not p.exists():
        return kb
    data = json.loads(p.read_text(encoding="utf-8"))
    # Expecting top-level keys similar to this module's structures
    for k, v in data.get("aliases", {}).items():
        kb.aliases[k.lower()] = v.lower()
    for k, v in data.get("ingredients", {}).items():
        kb.ingredients[k.lower()] = {x.lower() for x in v}
    for k, v in data.get("therapeutic_classes", {}).items():
        kb.therapeutic_classes[k.lower()] = {x.lower() for x in v}
    for k, v in data.get("allergy_cross_reactivity", {}).items():
        kb.allergy_cross_reactivity[k.lower()] = {x.lower() for x in v}
    for pair in data.get("major_interactions", []):
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            kb.major_interactions.add(frozenset({pair[0].lower(), pair[1].lower()}))
    for pair in data.get("moderate_interactions", []):
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            kb.moderate_interactions.add(frozenset({pair[0].lower(), pair[1].lower()}))
    return kb


def load_yaml_file(path: str | Path) -> InMemoryKnowledgeBase:
    p = Path(path)
    kb = InMemoryKnowledgeBase()
    if not p.exists():
        return kb
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return load_json_file_from_dict(data)


def load_json_file_from_dict(data: dict) -> InMemoryKnowledgeBase:
    kb = InMemoryKnowledgeBase()
    for k, v in data.get("aliases", {}).items():
        kb.aliases[k.lower()] = v.lower()
    for k, v in data.get("ingredients", {}).items():
        kb.ingredients[k.lower()] = {x.lower() for x in v}
    for k, v in data.get("therapeutic_classes", {}).items():
        kb.therapeutic_classes[k.lower()] = {x.lower() for x in v}
    for k, v in data.get("allergy_cross_reactivity", {}).items():
        kb.allergy_cross_reactivity[k.lower()] = {x.lower() for x in v}
    for pair in data.get("major_interactions", []):
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            kb.major_interactions.add(frozenset({pair[0].lower(), pair[1].lower()}))
    for pair in data.get("moderate_interactions", []):
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            kb.moderate_interactions.add(frozenset({pair[0].lower(), pair[1].lower()}))
    return kb


def load_csv_interactions(path: str | Path, target: str = "major") -> InMemoryKnowledgeBase:
    p = Path(path)
    kb = InMemoryKnowledgeBase()
    if not p.exists():
        return kb
    with p.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 2:
                continue
            a, b = row[0].strip().lower(), row[1].strip().lower()
            if not a or not b:
                continue
            if target == "major":
                kb.major_interactions.add(frozenset({a, b}))
            else:
                kb.moderate_interactions.add(frozenset({a, b}))
    return kb


def normalize_drug_name(name: str) -> str:
    if not name:
        return ""
    normalized = name.lower().strip().replace("_", " ")
    normalized = " ".join(normalized.split())
    # map aliases first
    if normalized in _KB.aliases:
        return _KB.aliases[normalized]
    return normalized


def get_drug_ingredients(drug_name: str) -> set[str]:
    canonical = normalize_drug_name(drug_name)
    ingredients = set(_KB.ingredients.get(canonical, set()))
    ingredients.add(canonical)
    return {i.lower().strip() for i in ingredients if i}


def get_therapeutic_classes(drug_name: str) -> set[str]:
    canonical = normalize_drug_name(drug_name)
    classes = set(_KB.therapeutic_classes.get(canonical, set()))
    # fallback heuristic
    if not classes and canonical in {"ibuprofen", "naproxen", "diclofenac", "celecoxib", "aspirin"}:
        classes.add("nsaid")
    return {c.lower().strip() for c in classes if c}


def expand_drug_terms(drug_name: str) -> set[str]:
    canonical = normalize_drug_name(drug_name)
    terms: set[str] = {canonical}
    terms.update(get_drug_ingredients(canonical))
    terms.update(get_therapeutic_classes(canonical))
    return {term.lower().strip() for term in terms if term}


def get_cross_reactive_ingredients(allergen: str) -> set[str]:
    allergen_lower = normalize_drug_name(allergen)
    cross_reactive: set[str] = {allergen_lower}
    cross_reactive.update(get_drug_ingredients(allergen_lower))
    cross_reactive.update(get_therapeutic_classes(allergen_lower))
    if allergen_lower in _KB.allergy_cross_reactivity:
        cross_reactive.update({x.lower() for x in _KB.allergy_cross_reactivity[allergen_lower]})
    return {normalize_drug_name(item) for item in cross_reactive if item}


def _all_interaction_pairs(left_terms: Iterable[str], right_terms: Iterable[str]) -> set[frozenset]:
    pairs: set[frozenset] = set()
    for left in left_terms:
        for right in right_terms:
            if left and right and left != right:
                pairs.add(frozenset({left.lower().strip(), right.lower().strip()}))
    return pairs


def check_major_interaction(drug1: str, drug2: str) -> bool:
    terms1 = expand_drug_terms(drug1)
    terms2 = expand_drug_terms(drug2)
    pairs = _all_interaction_pairs(terms1, terms2)
    return any(pair in _KB.major_interactions for pair in pairs)


def check_moderate_interaction(drug1: str, drug2: str) -> bool:
    terms1 = expand_drug_terms(drug1)
    terms2 = expand_drug_terms(drug2)
    pairs = _all_interaction_pairs(terms1, terms2)
    return any(pair in _KB.moderate_interactions for pair in pairs)


def clear_kb() -> None:
    """Clear the in-memory KB back to empty (useful for tests)."""
    global _KB
    _KB = InMemoryKnowledgeBase()
    _init_defaults()
