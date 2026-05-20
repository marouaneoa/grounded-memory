"""
Conflict Resolution for Grounded Memory Governance

This module implements multi-signal conflict resolution strategies to determine
which fact should win when new information contradicts existing knowledge.

Key design principles:
- Facts are NEVER deleted — losers are superseded with full audit trail
- Resolution is deterministic and explainable
- Multiple strategies can be composed for domain-specific governance
- Every resolution records its reasoning for compliance and debugging

Strategies:
    CONFIDENCE_WINS  — Higher confidence score wins
    RECENCY_WINS     — More recent fact supersedes older
    SOURCE_PRIORITY  — system > tool > agent > user
    COMPOSITE        — Weighted combination of all signals
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from grounded_memory.core.models import ActorType, CandidateFact, ValidatedFact, as_utc_datetime

# =============================================================================
# Strategy Enumeration
# =============================================================================


class ConflictResolutionStrategy(str, Enum):
    """Strategy for resolving conflicts between existing and incoming facts."""

    CONFIDENCE_WINS = "confidence_wins"
    RECENCY_WINS = "recency_wins"
    SOURCE_PRIORITY = "source_priority"
    COMPOSITE = "composite"


# =============================================================================
# Conflict Signal — normalized inputs to the resolver
# =============================================================================

# Source priority ranking: higher = more authoritative
SOURCE_RANK: dict[str, int] = {
    ActorType.SYSTEM.value: 100,
    ActorType.TOOL.value: 75,
    ActorType.AGENT.value: 50,
    ActorType.USER.value: 25,
    "unknown": 0,
}


@dataclass(frozen=True)
class ConflictSignal:
    """Normalized quality signals for a single fact (existing or incoming)."""

    confidence: float
    timestamp: datetime
    source_rank: int
    embedding_similarity: float = 0.0  # Optional semantic similarity

    @classmethod
    def from_validated_fact(cls, fact: ValidatedFact) -> ConflictSignal:
        """Build signal from an existing validated fact."""
        source = (fact.attributes or {}).get("source") or (
            (fact.source_metadata or {}).get("actor") if hasattr(fact, "source_metadata") else None
        )
        rank = SOURCE_RANK.get(str(source).strip().lower(), SOURCE_RANK["unknown"])
        return cls(
            confidence=fact.confidence,
            timestamp=fact.valid_from,
            source_rank=rank,
        )

    @classmethod
    def from_candidate_fact(cls, candidate: CandidateFact) -> ConflictSignal:
        """Build signal from an incoming candidate fact."""
        source = (candidate.attributes or {}).get("source", "user")
        rank = SOURCE_RANK.get(str(source).strip().lower(), SOURCE_RANK["unknown"])
        return cls(
            confidence=candidate.confidence,
            timestamp=candidate.extracted_at,
            source_rank=rank,
        )


# =============================================================================
# Conflict Resolution Result
# =============================================================================


@dataclass
class ConflictResolution:
    """Outcome of resolving a conflict between two facts."""

    should_supersede: bool
    strategy_used: ConflictResolutionStrategy
    winning_signal: ConflictSignal
    losing_signal: ConflictSignal
    reasoning: str
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize for audit logging."""
        return {
            "should_supersede": self.should_supersede,
            "strategy": self.strategy_used.value,
            "reasoning": self.reasoning,
            "scores": self.scores,
            "metadata": self.metadata,
        }


# =============================================================================
# Conflict Resolver
# =============================================================================


class ConflictResolver:
    """
    Multi-signal conflict resolver for fact governance.

    Determines whether an incoming candidate fact should supersede an
    existing validated fact, using pluggable resolution strategies.

    Usage:
        resolver = ConflictResolver(strategy=ConflictResolutionStrategy.COMPOSITE)
        resolution = resolver.resolve(existing_fact, candidate_fact)
        if resolution.should_supersede:
            # perform supersession
    """

    # Default weights for composite scoring
    DEFAULT_WEIGHTS = {
        "confidence": 0.40,
        "recency": 0.30,
        "source": 0.30,
    }

    def __init__(
        self,
        strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.COMPOSITE,
        *,
        weights: dict[str, float] | None = None,
        confidence_threshold: float = 0.05,
    ) -> None:
        """
        Args:
            strategy: Which resolution strategy to use.
            weights: Custom weights for composite scoring
                     (keys: confidence, recency, source).
            confidence_threshold: Minimum confidence delta to trigger
                                  confidence-based supersession.
        """
        self.strategy = strategy
        self.weights = dict(weights or self.DEFAULT_WEIGHTS)
        self.confidence_threshold = confidence_threshold

    def resolve(
        self,
        existing: ValidatedFact,
        candidate: CandidateFact,
    ) -> ConflictResolution:
        """
        Resolve a conflict between an existing fact and incoming candidate.

        Returns:
            ConflictResolution with the decision and reasoning.
        """
        existing_signal = ConflictSignal.from_validated_fact(existing)
        candidate_signal = ConflictSignal.from_candidate_fact(candidate)

        if self.strategy == ConflictResolutionStrategy.CONFIDENCE_WINS:
            return self._resolve_confidence(existing_signal, candidate_signal)
        elif self.strategy == ConflictResolutionStrategy.RECENCY_WINS:
            return self._resolve_recency(existing_signal, candidate_signal)
        elif self.strategy == ConflictResolutionStrategy.SOURCE_PRIORITY:
            return self._resolve_source_priority(existing_signal, candidate_signal)
        else:
            return self._resolve_composite(existing_signal, candidate_signal)

    # -----------------------------------------------------------------
    # Individual strategies
    # -----------------------------------------------------------------

    def _resolve_confidence(
        self,
        existing: ConflictSignal,
        candidate: ConflictSignal,
    ) -> ConflictResolution:
        delta = candidate.confidence - existing.confidence
        supersede = delta > self.confidence_threshold
        reasoning = (
            f"Candidate confidence ({candidate.confidence:.3f}) "
            f"{'exceeds' if supersede else 'does not exceed'} "
            f"existing ({existing.confidence:.3f}) by threshold {self.confidence_threshold}"
        )
        return ConflictResolution(
            should_supersede=supersede,
            strategy_used=ConflictResolutionStrategy.CONFIDENCE_WINS,
            winning_signal=candidate if supersede else existing,
            losing_signal=existing if supersede else candidate,
            reasoning=reasoning,
            scores={
                "confidence_delta": round(delta, 4),
                "threshold": self.confidence_threshold,
            },
        )

    def _resolve_recency(
        self,
        existing: ConflictSignal,
        candidate: ConflictSignal,
    ) -> ConflictResolution:
        c_ts = as_utc_datetime(candidate.timestamp)
        e_ts = as_utc_datetime(existing.timestamp)
        supersede = c_ts > e_ts
        reasoning = (
            f"Candidate timestamp ({c_ts.isoformat()}) "
            f"{'is newer than' if supersede else 'is not newer than'} "
            f"existing ({e_ts.isoformat()})"
        )
        return ConflictResolution(
            should_supersede=supersede,
            strategy_used=ConflictResolutionStrategy.RECENCY_WINS,
            winning_signal=candidate if supersede else existing,
            losing_signal=existing if supersede else candidate,
            reasoning=reasoning,
            scores={
                "candidate_ts": c_ts.timestamp(),
                "existing_ts": e_ts.timestamp(),
            },
        )

    def _resolve_source_priority(
        self,
        existing: ConflictSignal,
        candidate: ConflictSignal,
    ) -> ConflictResolution:
        supersede = candidate.source_rank > existing.source_rank
        # Tie-break: if same rank, use recency
        if candidate.source_rank == existing.source_rank:
            supersede = candidate.timestamp > existing.timestamp
            tiebreak = " (tie-broken by recency)"
        else:
            tiebreak = ""
        reasoning = (
            f"Candidate source rank ({candidate.source_rank}) "
            f"vs existing ({existing.source_rank})"
            f"{tiebreak}"
        )
        return ConflictResolution(
            should_supersede=supersede,
            strategy_used=ConflictResolutionStrategy.SOURCE_PRIORITY,
            winning_signal=candidate if supersede else existing,
            losing_signal=existing if supersede else candidate,
            reasoning=reasoning,
            scores={
                "candidate_rank": candidate.source_rank,
                "existing_rank": existing.source_rank,
            },
        )

    def _resolve_composite(
        self,
        existing: ConflictSignal,
        candidate: ConflictSignal,
    ) -> ConflictResolution:
        """Weighted combination of all signals."""
        w = self.weights

        # Confidence signal: [0, 1]
        conf_score = candidate.confidence / max(existing.confidence, 0.01)
        conf_signal = min(conf_score, 2.0) / 2.0  # normalize to [0, 1]

        # Recency signal: candidate is newer → 1.0, older → 0.0
        c_ts = candidate.timestamp
        e_ts = existing.timestamp
        if c_ts.tzinfo is None and e_ts.tzinfo is not None:
            c_ts = c_ts.replace(tzinfo=e_ts.tzinfo)
        elif e_ts.tzinfo is None and c_ts.tzinfo is not None:
            e_ts = e_ts.replace(tzinfo=c_ts.tzinfo)
        time_delta = (c_ts - e_ts).total_seconds()
        recency_signal = 1.0 / (1.0 + max(-time_delta, 0.0) / 3600.0)

        # Source signal: normalized rank ratio
        max_rank = max(candidate.source_rank, existing.source_rank, 1)
        source_signal = candidate.source_rank / max_rank

        composite = (
            w.get("confidence", 0.4) * conf_signal
            + w.get("recency", 0.3) * recency_signal
            + w.get("source", 0.3) * source_signal
        )
        supersede = composite > 0.5

        reasoning = (
            f"Composite score: {composite:.4f} "
            f"(confidence={conf_signal:.3f}×{w.get('confidence', 0.4)}, "
            f"recency={recency_signal:.3f}×{w.get('recency', 0.3)}, "
            f"source={source_signal:.3f}×{w.get('source', 0.3)}). "
            f"{'Superseding' if supersede else 'Keeping existing'}."
        )

        return ConflictResolution(
            should_supersede=supersede,
            strategy_used=ConflictResolutionStrategy.COMPOSITE,
            winning_signal=candidate if supersede else existing,
            losing_signal=existing if supersede else candidate,
            reasoning=reasoning,
            scores={
                "composite": round(composite, 6),
                "confidence_signal": round(conf_signal, 6),
                "recency_signal": round(recency_signal, 6),
                "source_signal": round(source_signal, 6),
            },
        )
