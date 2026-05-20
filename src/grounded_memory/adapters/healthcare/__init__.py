"""
Healthcare Adapter

Provides specific extraction, constraints, and models for clinical
and medical applications of Grounded Memory.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from grounded_memory.adapters.healthcare.agent import HealthcareMemoryAgent
from grounded_memory.adapters.healthcare.constraints import load_healthcare_constraints
from grounded_memory.adapters.healthcare.retrieval import (
    HealthcareClinicalContext,
    HealthcareContextBuilder,
    HealthcareQueryPlan,
    HealthcareRetrievalPlanner,
    HealthcareRetrievalService,
)

_logger = logging.getLogger(__name__)

# Resolved once at import time so it works regardless of CWD.
_KB_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "healthcare_kb.yaml"

__all__ = [
    "HealthcareMemoryAgent",
    "HealthcareClinicalContext",
    "HealthcareContextBuilder",
    "HealthcareQueryPlan",
    "HealthcareRetrievalPlanner",
    "HealthcareRetrievalService",
    "configure_validator",
    "create_agent",
    "load_healthcare_constraints",
]


def _init_kb_from_external_sources() -> None:
    """Merge openFDA drug label data into the active KB singleton.

    Runs once at adapter startup. Falls back to built-in defaults silently
    if the config file is missing or any remote source is unreachable.
    """
    if not _KB_CONFIG_PATH.exists():
        print("[Healthcare KB] healthcare_kb.yaml not found — using built-in defaults")
        return
    print(f"[Healthcare KB] Loading drug label data from openFDA ({_KB_CONFIG_PATH.name}) …")
    try:
        from grounded_memory.adapters.healthcare.kb_manager import initialize_from_config

        loaded = initialize_from_config(str(_KB_CONFIG_PATH))
        if loaded:
            print("[Healthcare KB] ✓ openFDA drug label data merged into knowledge base")
        else:
            print("[Healthcare KB] No external sources loaded — using built-in defaults")
    except Exception as exc:
        print(f"[Healthcare KB] External init failed ({exc}) — using built-in defaults")


def configure_validator(
    system: Any,
    config: dict[str, Any],
) -> Callable[[Any], None] | None:
    """Load healthcare constraints from YAML and seed KB from openFDA."""
    _init_kb_from_external_sources()
    evaluators = load_healthcare_constraints()

    if not evaluators:
        return None

    def _configurator(validator: Any) -> None:
        for evaluator in evaluators:
            validator.register(evaluator)

    return _configurator


def create_agent(
    system: Any,
    use_llm: bool,
    llm_config: Any,
    state: dict[str, Any],
) -> Any:
    """Create the healthcare-specific agent."""
    if not use_llm:
        raise ValueError("HealthcareMemoryAgent requires LLM extraction (use_llm=True).")

    return HealthcareMemoryAgent(
        memory_store=system.memory_store,
        grounding_operator=system.grounding_operator,
        llm_config=llm_config,
        domain_profile=str(state.get("domain_profile", "healthcare")),
    )
