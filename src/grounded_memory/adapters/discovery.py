"""Autonomous mining and synthesis of dynamic constraint seeds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DiscoveredConstraintSeed:
    """Synthesized seed proposal mined from governance signals."""

    constraint_id: str
    name: str
    description: str
    relation_types: list[str]
    required_attribute_keys: list[str]
    require_value: bool
    confidence: float
    evidence_count: int
    mining_rule: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "name": self.name,
            "description": self.description,
            "relation_types": list(self.relation_types),
            "required_attribute_keys": list(self.required_attribute_keys),
            "require_value": self.require_value,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "mining_rule": self.mining_rule,
        }


class ConstraintSeedDiscoverer:
    """Mine validation/rejection signals and synthesize candidate seed constraints."""

    def __init__(
        self,
        *,
        min_samples_per_relation: int = 20,
        min_rejections_per_relation: int = 6,
        min_gap: float = 0.35,
        min_gap_mode: str = "fixed",
        min_gap_floor: float = 0.15,
        min_gap_ceiling: float = 0.60,
        target_false_block_rate: float = 0.10,
        max_suggestions: int = 20,
    ) -> None:
        self.min_samples_per_relation = min_samples_per_relation
        self.min_rejections_per_relation = min_rejections_per_relation
        self.min_gap = min_gap
        self.min_gap_mode = min_gap_mode.strip().lower()
        self.min_gap_floor = min_gap_floor
        self.min_gap_ceiling = min_gap_ceiling
        self.target_false_block_rate = target_false_block_rate
        self.max_suggestions = max_suggestions

    def discover(
        self,
        *,
        validation_signals: list[dict[str, Any]],
        existing_constraint_ids: set[str] | None = None,
    ) -> list[DiscoveredConstraintSeed]:
        existing_ids = existing_constraint_ids or set()
        grouped: dict[str, list[dict[str, Any]]] = {}

        for signal in validation_signals:
            relation = str(signal.get("relation") or "").strip()
            if not relation:
                continue
            grouped.setdefault(relation, []).append(signal)

        suggestions: list[DiscoveredConstraintSeed] = []

        for relation, rows in grouped.items():
            if len(rows) < self.min_samples_per_relation:
                continue

            rejected = [row for row in rows if not bool(row.get("is_valid", True))]
            accepted = [row for row in rows if bool(row.get("is_valid", True))]
            if len(rejected) < self.min_rejections_per_relation:
                continue

            effective_min_gap = self._resolve_min_gap(
                total_samples=len(rows),
                rejected_samples=len(rejected),
            )

            suggestion = self._synthesize_require_value_seed(
                relation=relation,
                rejected=rejected,
                accepted=accepted,
                existing_ids=existing_ids,
                min_gap=effective_min_gap,
            )
            if suggestion is not None:
                suggestions.append(suggestion)
                existing_ids.add(suggestion.constraint_id)

            suggestions.extend(
                self._synthesize_required_attribute_key_seeds(
                    relation=relation,
                    rejected=rejected,
                    accepted=accepted,
                    existing_ids=existing_ids,
                    min_gap=effective_min_gap,
                )
            )
            for seed in suggestions:
                existing_ids.add(seed.constraint_id)

        suggestions.sort(
            key=lambda seed: (seed.confidence, seed.evidence_count),
            reverse=True,
        )
        return suggestions[: self.max_suggestions]

    def _synthesize_require_value_seed(
        self,
        *,
        relation: str,
        rejected: list[dict[str, Any]],
        accepted: list[dict[str, Any]],
        existing_ids: set[str],
        min_gap: float,
    ) -> DiscoveredConstraintSeed | None:
        rejected_missing_value = sum(1 for row in rejected if not bool(row.get("has_value", False)))
        accepted_missing_value = sum(1 for row in accepted if not bool(row.get("has_value", False)))

        rejected_rate = rejected_missing_value / max(len(rejected), 1)
        accepted_rate = accepted_missing_value / max(len(accepted), 1)
        gap = rejected_rate - accepted_rate

        if rejected_missing_value == 0 or gap < min_gap:
            return None

        constraint_id = f"seed_auto_require_value_{relation.lower()}"
        if constraint_id in existing_ids:
            return None

        confidence = min(0.99, max(0.5, gap * 0.8 + 0.2))
        return DiscoveredConstraintSeed(
            constraint_id=constraint_id,
            name=f"Auto-discovered value requirement ({relation})",
            description=(
                f"Mined from governance signals: relation {relation} has elevated rejection "
                "rate when candidate value is missing"
            ),
            relation_types=[relation],
            required_attribute_keys=[],
            require_value=True,
            confidence=round(confidence, 4),
            evidence_count=rejected_missing_value,
            mining_rule=f"missing_value_gap(min_gap={min_gap:.3f})",
        )

    def _synthesize_required_attribute_key_seeds(
        self,
        *,
        relation: str,
        rejected: list[dict[str, Any]],
        accepted: list[dict[str, Any]],
        existing_ids: set[str],
        min_gap: float,
    ) -> list[DiscoveredConstraintSeed]:
        attribute_keys: set[str] = set()
        for row in rejected + accepted:
            attrs = row.get("candidate_attributes") or {}
            if isinstance(attrs, dict):
                attribute_keys.update(str(key) for key in attrs)

        seeds: list[DiscoveredConstraintSeed] = []
        for key in sorted(attribute_keys):
            rejected_missing = 0
            for row in rejected:
                attrs = row.get("candidate_attributes") or {}
                value = attrs.get(key) if isinstance(attrs, dict) else None
                if value in (None, ""):
                    rejected_missing += 1

            accepted_missing = 0
            for row in accepted:
                attrs = row.get("candidate_attributes") or {}
                value = attrs.get(key) if isinstance(attrs, dict) else None
                if value in (None, ""):
                    accepted_missing += 1

            rejected_rate = rejected_missing / max(len(rejected), 1)
            accepted_rate = accepted_missing / max(len(accepted), 1)
            gap = rejected_rate - accepted_rate

            if rejected_missing == 0 or gap < min_gap:
                continue

            constraint_id = (
                f"seed_auto_require_attr_{relation.lower()}_{key.lower().replace(' ', '_')}"
            )
            if constraint_id in existing_ids:
                continue

            confidence = min(0.99, max(0.5, gap * 0.8 + 0.2))
            seeds.append(
                DiscoveredConstraintSeed(
                    constraint_id=constraint_id,
                    name=f"Auto-discovered required attribute '{key}' ({relation})",
                    description=(
                        f"Mined from governance signals: relation {relation} has elevated rejection "
                        f"rate when attribute '{key}' is missing"
                    ),
                    relation_types=[relation],
                    required_attribute_keys=[key],
                    require_value=False,
                    confidence=round(confidence, 4),
                    evidence_count=rejected_missing,
                    mining_rule=f"missing_attribute_gap(min_gap={min_gap:.3f})",
                )
            )

        seeds.sort(key=lambda seed: (seed.confidence, seed.evidence_count), reverse=True)
        return seeds

    def _resolve_min_gap(self, *, total_samples: int, rejected_samples: int) -> float:
        if self.min_gap_mode == "fixed":
            return float(self.min_gap)

        if self.min_gap_mode != "adaptive":
            raise ValueError("min_gap_mode must be 'fixed' or 'adaptive'")

        n = max(total_samples, 1)
        rejection_rate = rejected_samples / n
        uncertainty_penalty = 1.0 / (n**0.5)
        strictness_boost = 0.10 if rejection_rate >= 0.50 else 0.0

        adaptive_gap = self.target_false_block_rate + 0.15 + uncertainty_penalty + strictness_boost
        return max(self.min_gap_floor, min(self.min_gap_ceiling, adaptive_gap))
