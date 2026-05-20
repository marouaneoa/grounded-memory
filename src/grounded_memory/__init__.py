"""
Grounded Memory System - Core Module

A correctness-first memory architecture for LLM agents.
"""

from grounded_memory.adapters.discovery import (
    ConstraintSeedDiscoverer,
    DiscoveredConstraintSeed,
)
from grounded_memory.adapters.generic_agent import (
    GenericMemoryAgent,
    GenericProcessingResult,
)
from grounded_memory.adapters.registry import (
    get_adapter_spec,
    get_adapter_spec_by_key,
    list_registered_adapters,
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
from grounded_memory.core.constraints import (
    ConstraintValidator,
    ConstraintViolation,
    ValidationResult,
)
from grounded_memory.core.grounding import GroundingOperator, GroundingResult
from grounded_memory.core.models import (
    CandidateFact,
    Constraint,
    Entity,
    Interaction,
    MemoryDisposition,
    RelationType,
    ValidatedFact,
)
from grounded_memory.core.store import MemoryStore
from grounded_memory.core.system import GroundedMemorySystem as CoreGroundedMemorySystem
from grounded_memory.llm.client import LLMClient, LLMConfig, SyncLLMClient
from grounded_memory.llm.extractor import LLMFactExtractor
from grounded_memory.logging_utils import configure_logging
from grounded_memory.memory import (
    Memory,
    OptimizationProfile,
    OptimizationSettings,
    SearchResult,
)
from grounded_memory.service import app as service_app
from grounded_memory.service import create_app
from grounded_memory.system import GroundedMemorySystem

LLM_AVAILABLE = True

__version__ = "0.1.0"

__all__ = [
    # Core Models
    "Interaction",
    "Entity",
    "CandidateFact",
    "ValidatedFact",
    "Constraint",
    "MemoryDisposition",
    "RelationType",
    # Grounding
    "GroundingOperator",
    "GroundingResult",
    # Store
    "MemoryStore",
    "GroundedMemorySystem",
    "CoreGroundedMemorySystem",
    # Constraints
    "ConstraintValidator",
    "ValidationResult",
    "ConstraintViolation",
    # LLM (if available)
    "LLM_AVAILABLE",
    "LLMConfig",
    "SyncLLMClient",
    "LLMClient",
    "LLMFactExtractor",
    # SDK facade
    "Memory",
    "SearchResult",
    "OptimizationProfile",
    "OptimizationSettings",
    # Adapter registry APIs
    "SeedConstraintEvaluator",
    "CardinalitySeedConstraintEvaluator",
    "TemporalCardinalitySeedConstraintEvaluator",
    "DiscoveredConstraintSeed",
    "ConstraintSeedDiscoverer",
    "GenericMemoryAgent",
    "GenericProcessingResult",
    "configure_logging",
    "create_app",
    "service_app",
    "list_registered_adapters",
    "get_adapter_spec_by_key",
    "register_adapter",
    "unregister_adapter",
    "list_supported_profiles",
    "get_adapter_spec",
    "register_adapter_spec",
    "unregister_adapter_spec",
]
