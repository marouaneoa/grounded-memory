"""gmem public facade package.

This package provides a compact import surface while reusing the
`grounded_memory` implementation.
"""

from grounded_memory import (
    ConstraintValidator,
    GroundedMemorySystem,
    GroundingOperator,
    LLMConfig,
    Memory,
    MemoryStore,
    __version__,
    configure_logging,
    create_app,
    list_registered_adapters,
    register_adapter,
    unregister_adapter,
)

__all__ = [
    "__version__",
    "Memory",
    "GroundedMemorySystem",
    "ConstraintValidator",
    "GroundingOperator",
    "MemoryStore",
    "LLMConfig",
    "create_app",
    "configure_logging",
    "list_registered_adapters",
    "register_adapter",
    "unregister_adapter",
]
