"""RxNorm Loader for healthcare knowledge integration."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from grounded_memory.adapters.healthcare.knowledge import InMemoryKnowledgeBase
from grounded_memory.adapters.healthcare.loaders.cache import cache_get, cache_put

logger = logging.getLogger(__name__)

RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"
RXCUI_LOOKUP_URL = f"{RXNORM_BASE_URL}/rxcui.json"
PROPERTIES_URL = f"{RXNORM_BASE_URL}/rxcui/{{rxcui}}/properties.json"


class RxNormAPIError(RuntimeError):
    """Raised when the RxNorm API cannot be queried successfully."""


class RxNormLoader:
    """Fetch RxNorm identifiers and best-effort interaction data."""

    def __init__(self, rate_limit_delay: float = 0.5, timeout: float = 15.0):
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout, headers={"User-Agent": "GroundedMemory/1.0"})

    def _get_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def lookup_rxcui(self, drug_name: str, search_type: str = "all") -> str | None:
        normalized = drug_name.strip().lower()
        if not normalized:
            return None

        cache_key = f"rxnorm:rxcui:{normalized}:{search_type}"
        cached = cache_get(cache_key)
        if isinstance(cached, str):
            return cached

        params = {"name": drug_name.strip()}
        if search_type != "all":
            params["search"] = search_type
        try:
            data = self._get_json(RXCUI_LOOKUP_URL, params=params)
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code in {400, 404}:
                return None
            raise RxNormAPIError(f"Failed to lookup RxCUI for {drug_name}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise RxNormAPIError(f"Failed to lookup RxCUI for {drug_name}: {exc}") from exc

        id_group = data.get("idGroup", {})
        rxnorm_ids = id_group.get("rxnormId") or id_group.get("rxuiList") or []
        if rxnorm_ids:
            rxcui = str(rxnorm_ids[0])
            cache_put(cache_key, rxcui)
            return rxcui

        candidates = id_group.get("approximateGroup", {}).get("candidate", [])
        if candidates:
            candidate = candidates[0].get("rxcui") or candidates[0].get("RXCUI")
            if candidate:
                rxcui = str(candidate)
                cache_put(cache_key, rxcui)
                return rxcui

        return None

    def fetch_interactions_for_rxcui(self, rxcui: str) -> dict[str, list[str]]:
        cache_key = f"rxnorm:interactions:{rxcui}"
        cached = cache_get(cache_key)
        if isinstance(cached, dict):
            return cached

        # RxNorm's live REST service does not consistently expose a public
        # interaction endpoint. Probe a few known shapes and degrade gracefully.
        candidates = [
            (f"{RXNORM_BASE_URL}/interaction/interaction.json", {"rxcui": rxcui}),
            (f"{RXNORM_BASE_URL}/interaction/list.json", {"rxcuis": rxcui}),
            (f"{RXNORM_BASE_URL}/interaction/list.json", {"rxcuis": f"{rxcui}+"}),
        ]
        result = {"major": [], "moderate": [], "minor": []}

        for url, params in candidates:
            try:
                response = self.client.get(url, params=params)
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                self._extract_interactions(response.json(), result)
                if any(result.values()):
                    cache_put(cache_key, result)
                    return result
            except httpx.HTTPError:
                continue

        cache_put(cache_key, result)
        return result

    def _extract_interactions(self, data: dict[str, Any], result: dict[str, list[str]]) -> None:
        for group in data.get("fullInteractionTypeGroup", []):
            for interaction_type in group.get("fullInteractionType", []):
                severity_hint = str(interaction_type.get("comment", "")).lower()
                for pair in interaction_type.get("interactionPair", []):
                    severity = str(pair.get("severity", severity_hint)).lower()
                    names: list[str] = []
                    for concept in pair.get("interactionConcept", []):
                        name = concept.get("minConceptItem", {}).get("name")
                        if name:
                            names.append(str(name).lower())
                    for name in names:
                        if any(
                            token in severity for token in ("major", "contraindicated", "severe")
                        ):
                            result["major"].append(name)
                        elif "moderate" in severity:
                            result["moderate"].append(name)
                        else:
                            result["minor"].append(name)

        for key in result:
            result[key] = sorted(set(result[key]))

    def get_drug_properties(self, rxcui: str) -> dict[str, Any]:
        cache_key = f"rxnorm:properties:{rxcui}"
        cached = cache_get(cache_key)
        if isinstance(cached, dict):
            return cached

        try:
            data = self._get_json(PROPERTIES_URL.format(rxcui=rxcui))
        except httpx.HTTPError:
            return {}

        props = data.get("properties", {})
        result = {
            "rxcui": props.get("rxcui"),
            "name": props.get("name"),
            "tty": props.get("tty"),
            "language": props.get("language"),
        }
        cache_put(cache_key, result)
        return result

    def build_kb_from_rxnorm(
        self,
        drug_names: list[str],
        include_minor: bool = False,
        max_workers: int = 1,
        rate_limit_delay: float | None = None,
    ) -> InMemoryKnowledgeBase:
        delay = self.rate_limit_delay if rate_limit_delay is None else rate_limit_delay
        kb = InMemoryKnowledgeBase()

        for drug_name in drug_names:
            rxcui = self.lookup_rxcui(drug_name)
            if not rxcui:
                time.sleep(delay)
                continue

            properties = self.get_drug_properties(rxcui)
            canonical_name = (properties.get("name") or drug_name).strip().lower()
            kb.aliases[canonical_name] = canonical_name
            kb.ingredients.setdefault(canonical_name, set()).add(canonical_name)
            if properties.get("tty"):
                kb.therapeutic_classes.setdefault(canonical_name, set()).add(
                    str(properties["tty"]).lower()
                )

            interactions = self.fetch_interactions_for_rxcui(rxcui)
            for interacting_name in interactions["major"]:
                if interacting_name and interacting_name != canonical_name:
                    kb.major_interactions.add(frozenset({canonical_name, interacting_name.lower()}))
            for interacting_name in interactions["moderate"]:
                if interacting_name and interacting_name != canonical_name:
                    kb.moderate_interactions.add(
                        frozenset({canonical_name, interacting_name.lower()})
                    )

            time.sleep(delay)

        return kb


def lookup_rxcui(drug_name: str, search_type: str = "all") -> str | None:
    return RxNormLoader().lookup_rxcui(drug_name, search_type)


def fetch_interactions_for_rxcui(rxcui: str) -> dict[str, list[str]]:
    return RxNormLoader().fetch_interactions_for_rxcui(rxcui)


def build_kb_from_rxnorm(
    drug_names: list[str],
    include_minor: bool = False,
    max_workers: int = 1,
    rate_limit_delay: float = 0.5,
) -> InMemoryKnowledgeBase:
    return RxNormLoader(rate_limit_delay=rate_limit_delay).build_kb_from_rxnorm(
        drug_names,
        include_minor=include_minor,
        max_workers=max_workers,
        rate_limit_delay=rate_limit_delay,
    )
