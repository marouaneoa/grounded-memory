"""Adapter packages for memory governance and agent wiring."""

from grounded_memory.adapters.discovery import (
    ConstraintSeedDiscoverer,
    DiscoveredConstraintSeed,
)
from grounded_memory.adapters.generic_agent import (
    GenericMemoryAgent,
    GenericProcessingResult,
)
from grounded_memory.adapters.registry import (
    AdapterSpec,
    get_adapter_spec,
    get_adapter_spec_by_key,
    list_registered_adapters,
    # Backward-compatible exports
    list_supported_profiles,
    register_adapter,
    register_adapter_spec,
    unregister_adapter,
    unregister_adapter_spec,
)
from grounded_memory.adapters.seeds import (
    CardinalitySeedConstraintEvaluator,
    SeedConstraintEvaluator,
    TemporalCardinalitySeedConstraintEvaluator,
)

__all__ = [
    "SeedConstraintEvaluator",
    "CardinalitySeedConstraintEvaluator",
    "TemporalCardinalitySeedConstraintEvaluator",
    "DiscoveredConstraintSeed",
    "ConstraintSeedDiscoverer",
    "AdapterSpec",
    "GenericMemoryAgent",
    "GenericProcessingResult",
    "list_registered_adapters",
    "get_adapter_spec_by_key",
    "register_adapter",
    "unregister_adapter",
    "list_supported_profiles",
    "get_adapter_spec",
    "register_adapter_spec",
    "unregister_adapter_spec",
]
