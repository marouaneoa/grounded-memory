#!/usr/bin/env python3
"""
Initialize healthcare knowledge base with external sources.

This script demonstrates how to:
1. Load RxNorm drug interactions
2. Load openFDA drug labels
3. Register them with the knowledge base
4. Test constraint logic
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from grounded_memory.adapters.healthcare import knowledge, loaders  # noqa: E402
from grounded_memory.adapters.healthcare.kb_manager import KBManager  # noqa: E402


def main():
    """Initialize and test healthcare KB."""

    logger.info("=" * 70)
    logger.info("Healthcare Knowledge Base Initialization")
    logger.info("=" * 70)

    # Approach 1: Config-driven initialization
    logger.info("\n[1/3] Loading from config file...")
    manager = KBManager(config_path="configs/healthcare_kb.yaml")

    if manager.initialize_knowledge_base():
        logger.info("✓ Config-driven initialization succeeded")
    else:
        logger.warning("⚠ Config-driven initialization partially failed")

    # Show what's loaded
    print_kb_stats()

    # Approach 2: Manual RxNorm loading
    logger.info("\n[2/3] Manual RxNorm loading (example)...")
    try:
        rxnorm_loader = loaders.rxnorm.RxNormLoader(rate_limit_delay=0.3)
        test_drugs = ["Warfarin", "Aspirin"]

        logger.info(f"Fetching RxNorm data for: {', '.join(test_drugs)}")
        rxnorm_kb = rxnorm_loader.build_kb_from_rxnorm(
            test_drugs,
            include_minor=False,
        )

        logger.info(f"✓ Loaded {len(rxnorm_kb.major_interactions)} major interaction pairs")

        # Don't re-register (already loaded from config), just show stats

    except Exception as e:
        logger.warning(f"⚠ RxNorm loading failed (may be API issue): {e}")

    # Approach 3: Manual openFDA loading
    logger.info("\n[3/3] Manual openFDA loading (example)...")
    try:
        openfda_loader = loaders.openfda.OpenFDALoader(rate_limit_delay=0.3)
        test_drugs = ["Ibuprofen", "Aspirin"]

        logger.info(f"Fetching openFDA labels for: {', '.join(test_drugs)}")
        fda_kb = openfda_loader.fetch_batch_labels(
            test_drugs,
            rate_limit_delay=0.3,
        )

        logger.info(f"✓ Loaded {len(fda_kb.aliases)} drug aliases")
        logger.info(f"✓ Loaded ingredients for {len(fda_kb.ingredients)} drugs")

        # Don't re-register (already loaded from config), just show stats

    except Exception as e:
        logger.warning(f"⚠ openFDA loading failed (may be API issue): {e}")

    # Test constraint logic with loaded KB
    logger.info("\n" + "=" * 70)
    logger.info("Testing Constraint Logic with Loaded KB")
    logger.info("=" * 70)

    test_drug_pairs = [
        ("Warfarin", "Ibuprofen"),
        ("Warfarin", "Aspirin"),
        ("Ibuprofen", "Aspirin"),
        ("Aspirin", "Acetaminophen"),
    ]

    for drug1, drug2 in test_drug_pairs:
        major = knowledge.check_major_interaction(drug1, drug2)
        moderate = knowledge.check_moderate_interaction(drug1, drug2)
        status = "✓ MAJOR" if major else ("⚠ MODERATE" if moderate else "OK")
        logger.info(f"  {drug1} + {drug2}: {status}")

    # Cache stats
    logger.info("\n" + "=" * 70)
    logger.info("Cache Statistics")
    logger.info("=" * 70)

    cache_stats = manager.get_cache_stats()
    logger.info(f"Cache entries: {cache_stats.get('num_entries', 0)}")
    logger.info(f"Cache size: {cache_stats.get('total_size_kb', 0)} KB")
    logger.info(f"Cache directory: {cache_stats.get('cache_dir', 'N/A')}")

    logger.info("\n" + "=" * 70)
    logger.info("✓ Healthcare KB Initialization Complete")
    logger.info("=" * 70)


def print_kb_stats():
    """Print current KB statistics."""
    kb = knowledge._KB
    logger.info("\nKnowledge Base Summary:")
    logger.info(f"  Major interactions: {len(kb.major_interactions)}")
    logger.info(f"  Moderate interactions: {len(kb.moderate_interactions)}")
    logger.info(f"  Drug aliases: {len(kb.aliases)}")
    logger.info(f"  Drugs with ingredients: {len(kb.ingredients)}")
    logger.info(f"  Drugs with classes: {len(kb.therapeutic_classes)}")
    logger.info(f"  Allergy cross-reactivity: {len(kb.allergy_cross_reactivity)} allergens")


if __name__ == "__main__":
    main()
