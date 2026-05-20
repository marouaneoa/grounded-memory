#!/usr/bin/env python3
"""
Adapter System Verification Tests

Tests that the adapter registry works correctly, all domain profiles
load without errors, and domain-specific constraint configs are valid.

Run:
    PYTHONPATH=src python -m pytest tests/test_adapters.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grounded_memory.adapters.registry import (
    ADAPTER_SPECS,
    get_adapter_spec_by_key,
    list_registered_adapters,
)
from grounded_memory.core.models import EntityType, RelationType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "src" / "grounded_memory" / "configs"


# =============================================================================
# Test: Adapter Registry
# =============================================================================


class TestAdapterRegistry:
    """Verify the adapter registry is correctly configured."""

    def test_all_profiles_registered(self):
        """All expected adapter profiles should be registered."""
        expected = {"generic", "core", "none", "engineering", "finance", "legal", "healthcare"}
        registered = set(list_registered_adapters())
        assert expected.issubset(registered), f"Missing profiles: {expected - registered}"

    def test_adapter_spec_has_key(self):
        """Each adapter spec should have a matching key."""
        for key, spec in ADAPTER_SPECS.items():
            assert spec.key == key

    def test_get_adapter_by_key(self):
        """get_adapter_spec_by_key should return correct spec."""
        for key in ["generic", "engineering", "finance", "legal", "healthcare"]:
            spec = get_adapter_spec_by_key(key)
            assert spec.key == key

    def test_invalid_adapter_raises(self):
        """Requesting non-existent adapter should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported adapter"):
            get_adapter_spec_by_key("nonexistent_adapter_xyz")

    def test_adapter_has_agent_creator(self):
        """Each adapter must have a callable create_agent."""
        for key, spec in ADAPTER_SPECS.items():
            assert callable(spec.create_agent), f"Adapter '{key}' missing create_agent"

    def test_adapter_has_validator_configurator(self):
        """Each adapter must have a callable configure_validator."""
        for key, spec in ADAPTER_SPECS.items():
            assert callable(spec.configure_validator), (
                f"Adapter '{key}' missing configure_validator"
            )


# =============================================================================
# Test: Constraint Config Files
# =============================================================================


class TestConstraintConfigs:
    """Verify that all constraint YAML configs are valid and well-structured."""

    EXPECTED_CONFIGS = [
        "generic_constraints.yaml",
        "engineering_constraints.yaml",
        "finance_constraints.yaml",
        "legal_constraints.yaml",
        "healthcare_constraints.yaml",
    ]

    def test_all_config_files_exist(self):
        """All expected constraint config files should exist."""
        for filename in self.EXPECTED_CONFIGS:
            path = CONFIGS_DIR / filename
            assert path.exists(), f"Missing config: {path}"

    @pytest.mark.parametrize("filename", EXPECTED_CONFIGS)
    def test_config_is_valid_yaml(self, filename):
        """Each config file should be valid YAML."""
        path = CONFIGS_DIR / filename
        if not path.exists():
            pytest.skip(f"{filename} not found")

        with open(path) as f:
            data = yaml.safe_load(f)

        assert data is not None, f"{filename} parsed as empty"
        assert isinstance(data, dict), f"{filename} root should be a dict"

    @pytest.mark.parametrize("filename", EXPECTED_CONFIGS)
    def test_config_has_constraints_section(self, filename):
        """Each config should have a 'constraints' section."""
        path = CONFIGS_DIR / filename
        if not path.exists():
            pytest.skip(f"{filename} not found")

        with open(path) as f:
            data = yaml.safe_load(f)

        assert "constraints" in data, f"{filename} missing 'constraints' section"
        assert isinstance(data["constraints"], list), "constraints should be a list"
        assert len(data["constraints"]) > 0, "constraints list should not be empty"

    @pytest.mark.parametrize("filename", EXPECTED_CONFIGS)
    def test_constraints_have_required_fields(self, filename):
        """Each constraint should have id, name, description, type, severity."""
        path = CONFIGS_DIR / filename
        if not path.exists():
            pytest.skip(f"{filename} not found")

        with open(path) as f:
            data = yaml.safe_load(f)

        required_fields = {"id", "name", "description", "type", "severity"}
        for i, constraint in enumerate(data["constraints"]):
            for field in required_fields:
                assert field in constraint, f"{filename}:constraints[{i}] missing '{field}'"

    @pytest.mark.parametrize("filename", EXPECTED_CONFIGS)
    def test_constraint_ids_unique(self, filename):
        """Constraint IDs within a file should be unique."""
        path = CONFIGS_DIR / filename
        if not path.exists():
            pytest.skip(f"{filename} not found")

        with open(path) as f:
            data = yaml.safe_load(f)

        ids = [c["id"] for c in data["constraints"]]
        assert len(ids) == len(set(ids)), f"Duplicate constraint IDs in {filename}"

    @pytest.mark.parametrize("filename", EXPECTED_CONFIGS)
    def test_config_has_retrieval_weights(self, filename):
        """Each config should have retrieval_weights section."""
        path = CONFIGS_DIR / filename
        if not path.exists():
            pytest.skip(f"{filename} not found")

        with open(path) as f:
            data = yaml.safe_load(f)

        assert "retrieval_weights" in data, f"{filename} missing 'retrieval_weights'"
        weights = data["retrieval_weights"]
        assert isinstance(weights, dict)

        for rel_name, config in weights.items():
            assert "weight" in config, f"{filename}:{rel_name} missing 'weight'"
            assert isinstance(config["weight"], (int, float))

    @pytest.mark.parametrize("filename", EXPECTED_CONFIGS)
    def test_config_has_entity_schemas(self, filename):
        """Each config should have entity_schemas section."""
        path = CONFIGS_DIR / filename
        if not path.exists():
            pytest.skip(f"{filename} not found")

        with open(path) as f:
            data = yaml.safe_load(f)

        assert "entity_schemas" in data, f"{filename} missing 'entity_schemas'"
        schemas = data["entity_schemas"]
        assert isinstance(schemas, dict)
        assert len(schemas) > 0

        for entity_name, schema in schemas.items():
            assert "required" in schema, f"{filename}:{entity_name} missing 'required'"


# =============================================================================
# Test: Entity and Relation Type Coverage
# =============================================================================


class TestTypeCoverage:
    """Verify that models support both legacy and generic types."""

    def test_generic_entity_types_exist(self):
        """All generic entity types should be defined."""
        generic_types = [
            "PERSON",
            "PLACE",
            "ORGANIZATION",
            "CONCEPT",
            "ASSET",
            "SERVICE",
            "EVENT",
            "DOCUMENT",
            "PROJECT",
            "TOOL",
            "METRIC",
            "POLICY",
        ]
        for t in generic_types:
            assert hasattr(EntityType, t), f"EntityType.{t} not found"

    def test_legacy_entity_types_preserved(self):
        """Legacy healthcare entity types should still work."""
        legacy_types = [
            "PATIENT",
            "MEDICATION",
            "CONDITION",
            "ALLERGY",
            "INGREDIENT",
            "THERAPEUTIC_CLASS",
            "CLINICIAN",
            "FACILITY",
        ]
        for t in legacy_types:
            assert hasattr(EntityType, t), f"EntityType.{t} not found"

    def test_generic_relation_types_exist(self):
        """All generic relation types should be defined."""
        generic_relations = [
            "OWNS",
            "WORKS_AT",
            "LOCATED_IN",
            "MEMBER_OF",
            "CREATED",
            "DEPENDS_ON",
            "MANAGES",
            "USED_BY",
            "PRODUCED_BY",
            "AFFILIATED_WITH",
            "REPORTED_BY",
            "APPROVED_BY",
        ]
        for r in generic_relations:
            assert hasattr(RelationType, r), f"RelationType.{r} not found"

    def test_legacy_relation_types_preserved(self):
        """Legacy healthcare relation types should still work."""
        legacy_relations = ["HAS_ALLERGY", "HAS_CONDITION", "PRESCRIBED", "DISCONTINUED", "TREATS"]
        for r in legacy_relations:
            assert hasattr(RelationType, r), f"RelationType.{r} not found"


# =============================================================================
# Test: Domain-Specific Config Quality
# =============================================================================


class TestDomainConfigQuality:
    """Verify domain configs follow SoTA patterns for expert knowledge."""

    def test_engineering_has_dependency_constraints(self):
        """Engineering config should have circular dependency prevention."""
        path = CONFIGS_DIR / "engineering_constraints.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)

        constraint_ids = {c["id"] for c in data["constraints"]}
        assert "no_circular_dependency" in constraint_ids
        assert "ownership_cardinality" in constraint_ids

    def test_finance_has_compliance_constraints(self):
        """Finance config should have regulatory compliance rules."""
        path = CONFIGS_DIR / "finance_constraints.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)

        constraint_ids = {c["id"] for c in data["constraints"]}
        assert "kyc_entity_required" in constraint_ids
        assert "immutable_audit_trail" in constraint_ids

    def test_legal_has_jurisdiction_constraints(self):
        """Legal config should have jurisdiction validation."""
        path = CONFIGS_DIR / "legal_constraints.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)

        constraint_ids = {c["id"] for c in data["constraints"]}
        assert "jurisdiction_required" in constraint_ids
        assert "precedent_chain_integrity" in constraint_ids

    def test_healthcare_has_safety_constraints(self):
        """Healthcare config should have allergy/drug interaction checks."""
        path = CONFIGS_DIR / "healthcare_constraints.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)

        constraint_ids = {c["id"] for c in data["constraints"]}
        assert "allergy_conflict" in constraint_ids
        assert "drug_interaction_major" in constraint_ids

    def test_generic_has_quality_gates(self):
        """Generic config should have basic quality constraints."""
        path = CONFIGS_DIR / "generic_constraints.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)

        constraint_ids = {c["id"] for c in data["constraints"]}
        assert "tuple_completeness" in constraint_ids
        assert "confidence_floor" in constraint_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
