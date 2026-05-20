"""Domain-agnostic intent routing layer for GMem.

This module defines the generic interface that bridges natural-language user
inputs to concrete memory operations.  It is intentionally decoupled from any
vertical domain; healthcare (or legal, finance, etc.) adapters provide the
concrete mapping from generic actions to domain-specific service calls.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from grounded_memory.llm.prompts import INTENT_ROUTING_SYSTEM_PROMPT


class IntentAction(str, Enum):
    """Domain-agnostic cognitive actions derived from natural language."""

    REMEMBER = "remember"
    RECALL = "recall"
    FIND_RELATED = "find_related"
    EXPLAIN = "explain"
    UNKNOWN = "unknown"


class UserIntent(BaseModel):
    """Structured intent produced by the routing layer.

    The router never emits domain-specific instructions; it only classifies the
    *cognitive* goal of the utterance.  Downstream adapters map these generic
    actions to concrete memory operations (e.g. ``REMEMBER`` →
    ``memory.add()``, ``RECALL`` → ``retrieve_current_state()``).
    """

    action: IntentAction = Field(
        default=IntentAction.UNKNOWN,
        description="Generic cognitive action inferred from the query.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Router confidence in the classification.",
    )
    mentions: list[str] = Field(
        default_factory=list,
        description="Entity names or identifiers explicitly mentioned in the query.",
    )
    temporal_anchor: str | None = Field(
        default=None,
        description="Temporal expression such as 'now', 'last week', '2024-01-01'.",
    )
    explanation: str = Field(
        default="",
        description="Brief human-readable rationale for the classification.",
    )

    def is_write(self) -> bool:
        """Return True when the intent targets the write path."""
        return self.action == IntentAction.REMEMBER

    def is_read(self) -> bool:
        """Return True when the intent targets the read path."""
        return self.action in {
            IntentAction.RECALL,
            IntentAction.FIND_RELATED,
            IntentAction.EXPLAIN,
        }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseIntentRouter(ABC):
    """Protocol for intent routers.

    Any concrete router—whether keyword-based, LLM-backed, or a hybrid—must
    implement ``route(query) -> UserIntent``.  This abstraction lets the core
    system remain agnostic to the classification mechanism.
    """

    @abstractmethod
    def route(self, query: str) -> UserIntent:
        """Classify *query* and return a ``UserIntent``."""
        ...


# ---------------------------------------------------------------------------
# Keyword fallback router (deterministic, zero-latency)
# ---------------------------------------------------------------------------


class KeywordIntentRouter(BaseIntentRouter):
    """Fast deterministic router using generic cognitive keyword buckets.

    The keyword lists below are *cognitive* and domain-agnostic.  Domain-
    specific adapters can register additional patterns via
    ``register_patterns`` without subclassing.
    """

    # Default patterns that suggest the user is stating / asserting new information.
    _DEFAULT_REMEMBER_PATTERNS: list[str] = [
        r"\bis\b",
        r"\bhas\b",
        r"\bwas\b",
        r"\bstarted\b",
        r"\badjusted\b",
        r"\bchanged\b",
        r"\bdiscontinued\b",
        r"\bgiven\b",
        r"\bprovided\b",
        r"\brecorded\b",
        r"\bnoted\b",
        r"\bremember\b",
        r"\bsave\b",
        r"\bstore\b",
        r"\bupdate\b",
        r"\bset\s+(?:my|the)\b",
        r"\bI\s+(?:like|prefer|use|work|live|am)\b",
    ]

    # Default patterns that suggest the user wants an explanation / summary.
    _DEFAULT_EXPLAIN_PATTERNS: list[str] = [
        r"\bexplain\b",
        r"\bsummar(?:y|ise|ize)\b",
        r"\boverview\b",
        r"\bwhat\s+is\s+(?:the\s+)?(?:status|situation|picture|state)\b",
        r"\btell\s+me\s+about\b",
        r"\bgive\s+me\s+an?\s+(?:overview|summary|explanation)\b",
    ]

    # Default patterns that suggest a cross-entity / set-membership query.
    _DEFAULT_FIND_RELATED_PATTERNS: list[str] = [
        r"\bwho\s+else\b",
        r"\bwhich\s+other\b",
        r"\bshared\b",
        r"\balso\b",
        r"\brelated\b",
        r"\bconnected\b",
    ]

    # Default patterns that suggest a direct lookup about a specific entity.
    _DEFAULT_RECALL_PATTERNS: list[str] = [
        r"\bwhat\b",
        r"\bwho\b",
        r"\bwhich\b",
        r"\blist\b",
        r"\btell\s+me\b",
        r"\bhistory\b",
        r"\bwhen\b",
        r"\bhow\s+(?:long|much|many|old)\b",
        r"\bdo\s+.*\s+have\b",
        r"\bdoes\s+.*\s+have\b",
        r"\bis\s+.*\s+(?:still|currently|now)\b",
    ]

    def __init__(
        self,
        *,
        remember_patterns: list[str] | None = None,
        explain_patterns: list[str] | None = None,
        find_related_patterns: list[str] | None = None,
        recall_patterns: list[str] | None = None,
    ) -> None:
        self.remember_patterns = list(remember_patterns or self._DEFAULT_REMEMBER_PATTERNS)
        self.explain_patterns = list(explain_patterns or self._DEFAULT_EXPLAIN_PATTERNS)
        self.find_related_patterns = list(
            find_related_patterns or self._DEFAULT_FIND_RELATED_PATTERNS
        )
        self.recall_patterns = list(recall_patterns or self._DEFAULT_RECALL_PATTERNS)

    def register_patterns(
        self,
        category: str,
        patterns: list[str],
        *,
        prepend: bool = False,
    ) -> None:
        """Register extra domain-specific regex patterns for a category.

        Args:
            category: One of ``remember``, ``explain``, ``find_related``, ``recall``.
            patterns: Regex patterns to add.
            prepend: If True, insert before defaults (higher priority).
        """
        attr = f"{category}_patterns"
        if not hasattr(self, attr):
            raise ValueError(f"Unknown intent category: {category}")
        existing: list[str] = getattr(self, attr)
        if prepend:
            setattr(self, attr, patterns + existing)
        else:
            setattr(self, attr, existing + patterns)

    def route(self, query: str) -> UserIntent:
        text = query.strip()
        lower = text.lower()

        # 1. Explain (highest priority — explicit summarisation language)
        for pat in self.explain_patterns:
            if re.search(pat, lower):
                return UserIntent(
                    action=IntentAction.EXPLAIN,
                    confidence=0.9,
                    explanation="Query contains explicit summarisation / explanation language.",
                )

        # 2. Find related (set-membership language)
        for pat in self.find_related_patterns:
            if re.search(pat, lower):
                return UserIntent(
                    action=IntentAction.FIND_RELATED,
                    confidence=0.85,
                    explanation="Query asks which entities share a property (cross-entity lookup).",
                )

        # 3. Remember vs Recall — heuristic: if the sentence looks like a statement
        #    (no question mark, no interrogative words at the start) and contains
        #    assertive verbs, treat it as a fact assertion.
        looks_like_statement = "?" not in text and not lower.startswith(
            ("what", "who", "which", "when", "where", "why", "how", "do ", "does ", "is ", "are ")
        )
        remember_hits = sum(1 for pat in self.remember_patterns if re.search(pat, lower))
        recall_hits = sum(1 for pat in self.recall_patterns if re.search(pat, lower))

        if looks_like_statement and remember_hits > recall_hits:
            return UserIntent(
                action=IntentAction.REMEMBER,
                confidence=min(0.7 + 0.1 * remember_hits, 0.95),
                explanation="Query reads as a factual statement with assertive language.",
            )

        # 4. Default to recall if question-like patterns dominate.
        if recall_hits > 0:
            return UserIntent(
                action=IntentAction.RECALL,
                confidence=min(0.7 + 0.1 * recall_hits, 0.95),
                explanation="Query reads as an information request (lookup / question).",
            )

        # 5. Fallback — if it still looks like a statement, store; otherwise unknown.
        if looks_like_statement:
            return UserIntent(
                action=IntentAction.REMEMBER,
                confidence=0.55,
                explanation="No strong question markers detected; treating as a statement.",
            )

        return UserIntent(
            action=IntentAction.UNKNOWN,
            confidence=0.0,
            explanation="Unable to classify intent from keyword patterns.",
        )


# ---------------------------------------------------------------------------
# LLM-backed router (generic, domain-agnostic prompt)
# ---------------------------------------------------------------------------


class LLMIntentRouter(BaseIntentRouter):
    """LLM-backed intent router using a generic, domain-agnostic prompt.

    The system prompt uses *cognitive* language only.  It never mentions
    "medication", "patient", "prescribe", or any other domain term, so the
    same router works for healthcare, legal, finance, or any other adapter.
    """

    _SYSTEM_PROMPT = INTENT_ROUTING_SYSTEM_PROMPT

    def __init__(self, llm_client: Any | None = None) -> None:
        self.llm_client = llm_client

    def route(self, query: str) -> UserIntent:
        if self.llm_client is None:
            return UserIntent(
                action=IntentAction.UNKNOWN,
                confidence=0.0,
                explanation="LLM client not configured; falling back to UNKNOWN.",
            )

        try:
            # SyncLLMClient.extract() is synchronous.
            extracted = self.llm_client.extract(
                text=query,
                output_model=UserIntent,
                system_prompt=self._SYSTEM_PROMPT,
            )
            # Enforce valid action enum
            extracted.action = IntentAction(extracted.action.value.upper())
            return extracted
        except Exception as exc:
            return UserIntent(
                action=IntentAction.UNKNOWN,
                confidence=0.0,
                explanation=f"LLM routing failed: {exc}",
            )


# ---------------------------------------------------------------------------
# Hybrid router: keyword fast-path + LLM fallback
# ---------------------------------------------------------------------------


class HybridIntentRouter(BaseIntentRouter):
    """Fast deterministic keyword router with optional LLM fallback.

    This is the recommended production router: it answers instantly for clear
    cases, and only invokes the LLM when the keyword classifier is uncertain.
    """

    def __init__(
        self,
        keyword_router: KeywordIntentRouter | None = None,
        llm_router: LLMIntentRouter | None = None,
        confidence_threshold: float = 0.75,
    ) -> None:
        self.keyword_router = keyword_router or KeywordIntentRouter()
        self.llm_router = llm_router
        self.confidence_threshold = confidence_threshold

    def route(self, query: str) -> UserIntent:
        intent = self.keyword_router.route(query)
        if intent.confidence >= self.confidence_threshold:
            return intent
        if self.llm_router is not None:
            return self.llm_router.route(query)
        # No LLM available — return the low-confidence keyword result so the
        # caller can decide whether to prompt for clarification.
        return intent


__all__ = [
    "IntentAction",
    "UserIntent",
    "BaseIntentRouter",
    "KeywordIntentRouter",
    "LLMIntentRouter",
    "HybridIntentRouter",
]
