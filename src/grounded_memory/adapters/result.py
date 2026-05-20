"""Shared result contracts for adapter processing pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AdapterProcessingResult:
    """Unified result payload returned by adapter agents."""

    interaction_id: str
    grounding_results: list[Any] = field(default_factory=list)
    approved_facts: list[Any] = field(default_factory=list)
    rejected_facts: list[Any] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dispositions: list[dict[str, Any]] = field(default_factory=list)
    interaction: Any | None = None
    extracted_items: Any | None = None
