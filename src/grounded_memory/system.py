"""Public system entrypoint decoupled from domain package paths."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

from grounded_memory.adapters.registry import get_adapter_spec_by_key
from grounded_memory.core.system import GroundedMemorySystem as CoreGroundedMemorySystem
from grounded_memory.core.system import Neo4jConfig, StorageBackend
from grounded_memory.llm.client import LLMConfig

AgentFactory = Callable[["GroundedMemorySystem", bool, Optional["LLMConfig"]], Any]


class GroundedMemorySystem(CoreGroundedMemorySystem):
    """
    Domain-agnostic public system facade.

    This entrypoint keeps the top-level API neutral while allowing domain
    profiles to plug in their own constraints and specialized agents.
    """

    def __init__(
        self,
        neo4j_config: Neo4jConfig | None = None,
        storage_backend: StorageBackend | str | None = None,
        adapter: str | None = None,
        domain_profile: str = "generic",
        configure_validator: Callable[[Any], None] | None = None,
        agent_factory: AgentFactory | None = None,
    ):
        adapter_key = (adapter or domain_profile).strip().lower()
        self.adapter = adapter_key
        self.domain_profile = adapter_key  # Backward-compatible alias
        self._agent_factory = agent_factory
        self._adapter_state: dict[str, Any] = {
            "adapter": adapter_key,
            "domain_profile": adapter_key,
        }
        self._adapter_spec = None

        if configure_validator is None or agent_factory is None:
            try:
                self._adapter_spec = get_adapter_spec_by_key(adapter_key)
            except ValueError:
                if configure_validator is None and agent_factory is None:
                    raise

        validator_configurator = configure_validator
        if validator_configurator is None:
            validator_configurator = self._resolve_adapter_validator(adapter_key)

        super().__init__(
            neo4j_config=neo4j_config,
            storage_backend=storage_backend,
            configure_validator=validator_configurator,
        )

    @property
    def adapter_spec(self):
        """The resolved AdapterSpec for this system, or None."""
        return self._adapter_spec

    def _resolve_adapter_validator(
        self,
        adapter_key: str,
    ) -> Callable[[Any], None] | None:
        if self._adapter_spec is None:
            return None
        return self._adapter_spec.configure_validator(self, self._adapter_state)

    def _resolve_profile_validator(
        self,
        profile: str,
    ) -> Callable[[Any], None] | None:
        """Backward-compatible alias for older profile-based extension code."""
        return self._resolve_adapter_validator(profile)

    def create_agent(
        self,
        use_llm: bool = True,
        llm_config: LLMConfig | None = None,
    ) -> Any:
        """Create an adapter agent if an agent factory/adapter spec is available."""
        if not use_llm:
            raise RuntimeError(
                "GroundedMemorySystem requires LLM mode for agent creation. "
                "Pass use_llm=True with a valid LLM configuration."
            )

        if self._agent_factory is not None:
            return self._agent_factory(self, True, llm_config)

        if self._adapter_spec is None:
            raise RuntimeError(
                "No built-in adapter is registered for this adapter key. "
                "Provide agent_factory=... or register an adapter via registry."
            )

        return self._adapter_spec.create_agent(
            self,
            True,
            llm_config,
            self._adapter_state,
        )
