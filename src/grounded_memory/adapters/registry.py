"""Central adapter registry for runtime memory adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

AdapterState = dict[str, Any]
ValidatorConfigurator = Callable[[Any, AdapterState], Callable[[Any], None] | None]
AgentCreator = Callable[[Any, bool, Any, AdapterState], Any]

# Lazy import anchors — resolved at call time to avoid circular imports.
_RelationshipPreset: Any = None
_RetrievalStrategy: Any = None


def _resolve_relationship_preset():
    global _RelationshipPreset
    if _RelationshipPreset is None:
        from grounded_memory.retrieval import RelationshipPreset as _RP

        _RelationshipPreset = _RP
    return _RelationshipPreset


def _resolve_retrieval_strategy():
    global _RetrievalStrategy
    if _RetrievalStrategy is None:
        from grounded_memory.retrieval import RetrievalStrategy as _RS

        _RetrievalStrategy = _RS
    return _RetrievalStrategy


@dataclass(frozen=True)
class AdapterSpec:
    """Defines how an adapter configures validation, agent creation, and retrieval defaults."""

    key: str
    configure_validator: ValidatorConfigurator
    create_agent: AgentCreator

    # Optional retrieval defaults — set by domain adapters to avoid hardcoding
    # domain logic in the generic Memory facade.
    relationship_preset: Any | None = None
    query_hints: dict[str, set[str]] | None = None
    intent_router_patterns: dict[str, list[str]] | None = None

    @property
    def profile(self) -> str:
        """Backward-compatible alias for older profile-based callers."""
        return self.key


def _generic_validator(_: Any, __: AdapterState) -> Callable[[Any], None] | None:
    return None


def _generic_agent(system: Any, _: bool, llm_config: Any, state: AdapterState) -> Any:
    from grounded_memory.adapters.generic_agent import GenericMemoryAgent

    return GenericMemoryAgent(
        memory_store=system.memory_store,
        grounding_operator=system.grounding_operator,
        llm_config=llm_config,
        adapter_key=str(state.get("adapter", "generic")),
    )


def _healthcare_validator(system: Any, config: AdapterState) -> Callable[[Any], None] | None:
    from grounded_memory.adapters.healthcare import configure_validator

    return configure_validator(system, config)


def _healthcare_agent(system: Any, use_llm: bool, llm_config: Any, state: AdapterState) -> Any:
    from grounded_memory.adapters.healthcare import create_agent

    return create_agent(system, use_llm, llm_config, state)


ADAPTER_SPECS: dict[str, AdapterSpec] = {
    "generic": AdapterSpec(
        key="generic",
        configure_validator=_generic_validator,
        create_agent=_generic_agent,
    ),
    "core": AdapterSpec(
        key="core",
        configure_validator=_generic_validator,
        create_agent=_generic_agent,
    ),
    "none": AdapterSpec(
        key="none",
        configure_validator=_generic_validator,
        create_agent=_generic_agent,
    ),
    "engineering": AdapterSpec(
        key="engineering",
        configure_validator=_generic_validator,
        create_agent=_generic_agent,
    ),
    "finance": AdapterSpec(
        key="finance",
        configure_validator=_generic_validator,
        create_agent=_generic_agent,
    ),
    "legal": AdapterSpec(
        key="legal",
        configure_validator=_generic_validator,
        create_agent=_generic_agent,
    ),
    "healthcare": AdapterSpec(
        key="healthcare",
        configure_validator=_healthcare_validator,
        create_agent=_healthcare_agent,
        relationship_preset="safety",
        query_hints={"safety": {"allergy", "allergies", "contraindicated"}},
        intent_router_patterns={
            "remember": [
                r"\bwas\s+(?:diagnosed|given|started|prescribed|adjusted|changed|discontinued)\b",
                r"\bhas\s+(?:diabetes|hypertension|allergies|asthma)\b",
            ],
            "explain": [
                r"\bclinical\s+picture\b",
                r"\bmedical\s+history\b",
            ],
            "find_related": [
                r"\bwhich\s+(?:other|patients?)\b",
                r"\bpatients?\s+(?:on|taking|with|having)\b",
                r"\balso\s+(?:taking|on|prescribed)\b",
            ],
            "recall": [
                r"\bis\s+.*\s+(?:taking|on|prescribed|allergic)\b",
            ],
        },
    ),
}


def register_adapter(
    *,
    key: str,
    configure_validator: ValidatorConfigurator,
    create_agent: AgentCreator,
    overwrite: bool = False,
) -> None:
    """Register a custom adapter at runtime."""
    normalized = key.strip().lower()
    if not normalized:
        raise ValueError("key must be a non-empty string")

    if normalized in ADAPTER_SPECS and not overwrite:
        raise ValueError(
            f"Adapter '{normalized}' already exists. Set overwrite=True to replace it."
        )

    ADAPTER_SPECS[normalized] = AdapterSpec(
        key=normalized,
        configure_validator=configure_validator,
        create_agent=create_agent,
    )


def unregister_adapter(key: str) -> bool:
    """Unregister a custom adapter. Built-in neutral adapters are protected."""
    normalized = key.strip().lower()
    if normalized in {"generic", "core", "none"}:
        return False
    return ADAPTER_SPECS.pop(normalized, None) is not None


def list_registered_adapters() -> list[str]:
    """Return registered adapter keys."""
    return sorted(ADAPTER_SPECS.keys())


def get_adapter_spec_by_key(key: str) -> AdapterSpec:
    """Get adapter spec for key or raise ValueError with registered values."""
    normalized = key.strip().lower()
    spec = ADAPTER_SPECS.get(normalized)
    if spec is not None:
        return spec

    options = ", ".join(list_registered_adapters())
    raise ValueError(f"Unsupported adapter '{key}'. Registered adapters: {options}.")


def register_adapter_spec(
    *,
    profile: str,
    configure_validator: ValidatorConfigurator,
    create_agent: AgentCreator,
    overwrite: bool = False,
) -> None:
    """Backward-compatible wrapper for register_adapter(...)."""
    register_adapter(
        key=profile,
        configure_validator=configure_validator,
        create_agent=create_agent,
        overwrite=overwrite,
    )


def unregister_adapter_spec(profile: str) -> bool:
    """Backward-compatible wrapper for unregister_adapter(...)."""
    return unregister_adapter(profile)


def list_supported_profiles() -> list[str]:
    """Backward-compatible wrapper for list_registered_adapters()."""
    return list_registered_adapters()


def get_adapter_spec(profile: str) -> AdapterSpec:
    """Backward-compatible wrapper for get_adapter_spec_by_key(...)."""
    return get_adapter_spec_by_key(profile)
