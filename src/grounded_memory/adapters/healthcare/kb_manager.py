"""
KB Manager: Configuration-driven initialization of healthcare knowledge base.

Loads external data sources (RxNorm, openFDA) based on YAML config file.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from grounded_memory.adapters.healthcare import knowledge
from grounded_memory.adapters.healthcare.loaders import cache, openfda, rxnorm

logger = logging.getLogger(__name__)


class KBManager:
    """Manager for healthcare knowledge base initialization and lifecycle."""

    def __init__(self, config_path: str | None = None):
        """
        Initialize KB manager.

        Args:
            config_path: Path to healthcare_kb.yaml config file
        """
        self.config_path = config_path or str(
            Path(__file__).resolve().parents[2] / "configs" / "healthcare_kb.yaml"
        )
        self.config: dict[str, Any] = {}
        self.loaded_sources: list[str] = []

        if Path(self.config_path).exists():
            self._load_config()
        else:
            logger.warning(f"Config file not found: {self.config_path}")

    def _load_config(self) -> None:
        """Load and parse YAML config file."""
        try:
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f) or {}
            logger.info(f"Loaded config from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.config = {}

    def initialize_knowledge_base(self) -> bool:
        """
        Initialize KB with external sources from config.

        Returns:
            True if at least one source loaded successfully, False otherwise
        """
        kb_config = self.config.get("knowledge_base", {})
        sources_config = kb_config.get("sources", [])

        if not sources_config:
            logger.warning("No data sources configured in knowledge_base.sources")
            return False

        logger.info(f"Initializing KB with {len(sources_config)} source(s)")

        for source in sources_config:
            if not source.get("enabled", True):
                logger.debug(f"Skipping disabled source: {source.get('name')}")
                continue

            self._load_source(source)

        if self.loaded_sources:
            logger.info(f"Successfully loaded {len(self.loaded_sources)} source(s)")
            logger.info(f"Sources: {', '.join(self.loaded_sources)}")
            return True
        else:
            logger.warning("No sources loaded successfully")
            return False

    def _load_source(self, source_config: dict[str, Any]) -> None:
        """
        Load a single data source.

        Args:
            source_config: Source configuration dict
        """
        source_name = source_config.get("name", "unknown")
        source_type = source_config.get("type", "unknown")
        params = source_config.get("params", {})
        cache_ttl = source_config.get("cache_ttl_hours", 168)

        try:
            logger.debug(f"Loading source: {source_name} (type: {source_type})")

            if source_type == "rxnorm":
                kb = self._load_rxnorm_source(params, cache_ttl)
            elif source_type == "openfda":
                kb = self._load_openfda_source(params, cache_ttl)
            elif source_type == "json_file":
                kb = self._load_json_file_source(params)
            elif source_type == "yaml_file":
                kb = self._load_yaml_file_source(params)
            else:
                logger.error(f"Unknown source type: {source_type}")
                return

            if kb:
                knowledge.register_source(kb)
                self.loaded_sources.append(source_name)
                logger.info(f"✓ Loaded source: {source_name}")

        except Exception as e:
            logger.error(f"Error loading source {source_name}: {e}")

    def _load_rxnorm_source(self, params: dict[str, Any], cache_ttl: int) -> Any | None:
        """Load RxNorm data source."""
        drugs = params.get("drugs", [])
        if not drugs:
            logger.warning("RxNorm source has no drugs configured")
            return None

        include_minor = params.get("include_minor", False)
        rate_limit_delay = params.get("rate_limit_delay", 0.5)
        max_workers = params.get("max_workers", 1)

        try:
            return rxnorm.build_kb_from_rxnorm(
                drugs,
                include_minor=include_minor,
                max_workers=max_workers,
                rate_limit_delay=rate_limit_delay,
            )
        except Exception as e:
            logger.error(f"RxNorm loader error: {e}")
            return None

    def _load_openfda_source(self, params: dict[str, Any], cache_ttl: int) -> Any | None:
        """Load openFDA data source."""
        drugs = params.get("drugs", [])
        if not drugs:
            logger.warning("openFDA source has no drugs configured")
            return None

        api_key = params.get("api_key")
        rate_limit_delay = params.get("rate_limit_delay", 0.5)
        max_workers = params.get("max_workers", 2)

        try:
            return openfda.fetch_batch_labels(
                drugs,
                api_key=api_key,
                rate_limit_delay=rate_limit_delay,
                max_workers=max_workers,
            )
        except Exception as e:
            logger.error(f"openFDA loader error: {e}")
            return None

    def _load_json_file_source(self, params: dict[str, Any]) -> Any | None:
        """Load JSON file data source."""
        file_path = params.get("path")
        if not file_path:
            logger.warning("JSON file source missing path parameter")
            return None

        try:
            kb = knowledge.load_json_file(file_path)
            return kb
        except Exception as e:
            logger.error(f"JSON file loader error: {e}")
            return None

    def _load_yaml_file_source(self, params: dict[str, Any]) -> Any | None:
        """Load YAML file data source."""
        file_path = params.get("path")
        if not file_path:
            logger.warning("YAML file source missing path parameter")
            return None

        try:
            kb = knowledge.load_yaml_file(file_path)
            return kb
        except Exception as e:
            logger.error(f"YAML file loader error: {e}")
            return None

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return cache.cache_stats()

    def clear_cache(self) -> None:
        """Clear all cache entries."""
        cache.cache_clear()
        logger.info("Cleared KB cache")


def initialize_from_config(config_path: str | None = None) -> bool:
    """
    Initialize healthcare KB from config file.

    Args:
        config_path: Path to config file (default: package configs/healthcare_kb.yaml)

    Returns:
        True if initialization succeeded
    """
    manager = KBManager(config_path=config_path)
    return manager.initialize_knowledge_base()
