"""openFDA Loader for healthcare knowledge integration."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from grounded_memory.adapters.healthcare.knowledge import InMemoryKnowledgeBase
from grounded_memory.adapters.healthcare.loaders.cache import cache_get, cache_put

logger = logging.getLogger(__name__)

OPENFDA_BASE_URL = "https://api.fda.gov/drug/label.json"


class OpenFDAError(RuntimeError):
    """Raised when the openFDA API cannot be queried successfully."""


class OpenFDALoader:
    """Fetch FDA drug labels and translate them into KB entries."""

    def __init__(
        self, api_key: str | None = None, rate_limit_delay: float = 0.5, timeout: float = 15.0
    ):
        self.api_key = api_key or os.environ.get("FD_API_KEY")
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout, headers={"User-Agent": "GroundedMemory/1.0"})

    def _search(self, query: str, limit: int = 100) -> dict[str, Any]:
        params: dict[str, Any] = {"search": query, "limit": min(limit, 100)}
        if self.api_key:
            params["api_key"] = self.api_key
        response = self.client.get(OPENFDA_BASE_URL, params=params)
        response.raise_for_status()
        return response.json()

    def search_drug_labels(self, query: str, limit: int = 100) -> dict[str, Any]:
        cache_key = f"openfda:search:{query}:{limit}"
        cached = cache_get(cache_key)
        if isinstance(cached, dict):
            return cached
        try:
            data = self._search(query, limit)
        except Exception as exc:
            raise OpenFDAError(f"Failed to search openFDA for {query}: {exc}") from exc
        cache_put(cache_key, data)
        return data

    def fetch_drug_label(self, drug_name: str) -> dict[str, Any] | None:
        normalized = drug_name.strip()
        if not normalized:
            return None

        cache_key = f"openfda:label:{normalized.lower()}"
        cached = cache_get(cache_key)
        if isinstance(cached, dict):
            return cached

        queries = [
            f'openfda.brand_name:"{normalized}"',
            f'openfda.generic_name:"{normalized}"',
            f'openfda.substance_name:"{normalized}"',
        ]

        for query in queries:
            try:
                data = self.search_drug_labels(query, limit=10)
            except OpenFDAError:
                continue
            results = data.get("results", [])
            if results:
                label = results[0]
                cache_put(cache_key, label)
                return label

        return None

    def extract_ingredients_from_label(self, label: dict[str, Any]) -> dict[str, Any]:
        openfda_fields = label.get("openfda", {})
        generic_names = [
            str(name).lower() for name in openfda_fields.get("generic_name", []) if name
        ]
        brand_names = [str(name).lower() for name in openfda_fields.get("brand_name", []) if name]
        substance_names = [
            str(name).lower() for name in openfda_fields.get("substance_name", []) if name
        ]
        therapeutic_classes = {
            str(name).lower()
            for key in ("pharm_class_epc", "pharm_class_cs", "pharm_class_moa")
            for name in openfda_fields.get(key, [])
            if name
        }

        generic_name = generic_names[0] if generic_names else None
        active_ingredients = set(substance_names)
        if generic_name:
            active_ingredients.add(generic_name)

        warnings: list[str] = []
        for field_name in (
            "warnings",
            "when_using",
            "stop_use",
            "do_not_use",
            "ask_doctor",
            "ask_doctor_or_pharmacist",
        ):
            value = label.get(field_name)
            if isinstance(value, list):
                warnings.extend(str(item) for item in value if item)
            elif isinstance(value, str) and value:
                warnings.append(value)

        return {
            "generic_name": generic_name,
            "brand_names": brand_names,
            "active_ingredients": sorted(active_ingredients),
            "therapeutic_classes": sorted(therapeutic_classes),
            "warnings": warnings,
        }

    def extract_ingredients_from_labels(
        self, labels: list[dict[str, Any]]
    ) -> InMemoryKnowledgeBase:
        kb = InMemoryKnowledgeBase()
        for label in labels:
            extracted = self.extract_ingredients_from_label(label)
            generic_name = extracted["generic_name"]
            if not generic_name:
                continue

            kb.aliases[generic_name] = generic_name
            for brand in extracted["brand_names"]:
                kb.aliases[brand] = generic_name
            kb.ingredients.setdefault(generic_name, set()).update(
                extracted["active_ingredients"] or {generic_name}
            )
            if extracted["therapeutic_classes"]:
                kb.therapeutic_classes.setdefault(generic_name, set()).update(
                    extracted["therapeutic_classes"]
                )

        return kb

    def fetch_batch_labels(
        self,
        drug_names: list[str],
        batch_size: int = 10,
        max_workers: int = 1,
        rate_limit_delay: float | None = None,
    ) -> InMemoryKnowledgeBase:
        delay = self.rate_limit_delay if rate_limit_delay is None else rate_limit_delay
        labels: list[dict[str, Any]] = []
        for drug_name in drug_names:
            label = self.fetch_drug_label(drug_name)
            if label:
                labels.append(label)
            time.sleep(delay)
        return self.extract_ingredients_from_labels(labels)


def fetch_drug_label(drug_name: str, api_key: str | None = None) -> dict[str, Any] | None:
    return OpenFDALoader(api_key=api_key).fetch_drug_label(drug_name)


def fetch_batch_labels(
    drug_names: list[str],
    api_key: str | None = None,
    batch_size: int = 10,
    max_workers: int = 1,
    rate_limit_delay: float = 0.5,
) -> InMemoryKnowledgeBase:
    return OpenFDALoader(api_key=api_key, rate_limit_delay=rate_limit_delay).fetch_batch_labels(
        drug_names,
        batch_size=batch_size,
        max_workers=max_workers,
        rate_limit_delay=rate_limit_delay,
    )
