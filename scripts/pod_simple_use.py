#!/usr/bin/env python3
"""Simple pod-ready usage example for grounded-memory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grounded_memory import Memory
from grounded_memory.core.models import EntityType


def main() -> int:
    memory = Memory(domain_profile="generic", storage_backend="memory")

    service = memory.add_entity("Inference Service", entity_type=EntityType.FACILITY)["entity"]
    memory.add_fact(
        subject_id=service["id"],
        relation="HAS_ATTRIBUTE",
        value="endpoint=http://model:8000",
    )

    results = memory.retrieve("what is the inference endpoint", limit=3)

    print("retrieve_results", len(results))
    if results:
        print("top_value", results[0].get("value"))
    print("ok")
    return 0


if __name__ == "__main__":
    main()
