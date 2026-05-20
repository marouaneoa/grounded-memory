"""Shared adapter agent primitives."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any


class BaseAsyncAdapterAgent(ABC):
    """Base class that provides a sync wrapper for async adapter agents."""

    @abstractmethod
    async def process_interaction(
        self,
        raw_text: str,
        user_id: str | None = None,
        session_id: str | None = None,
        actor: str = "user",
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Process an interaction asynchronously."""

    def process(self, input_text: str, source: str = "user", **kwargs: Any) -> Any:
        """Backward-compatible synchronous wrapper used by demos and SDK calls."""
        metadata = kwargs.pop("metadata", None)
        user_id = kwargs.pop("user_id", None)
        session_id = kwargs.pop("session_id", None)

        extra_metadata = {k: v for k, v in kwargs.items() if v is not None}
        if metadata is None:
            metadata = extra_metadata
        elif isinstance(metadata, dict):
            metadata = {**metadata, **extra_metadata}

        actor = source.strip().lower() if isinstance(source, str) else "user"

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.process_interaction(
                    raw_text=input_text,
                    user_id=user_id,
                    session_id=session_id,
                    actor=actor,
                    metadata=metadata,
                )
            )

        raise RuntimeError(
            f"{self.__class__.__name__}.process() cannot be used while an event loop is running. "
            "Await process_interaction(...) instead."
        )
