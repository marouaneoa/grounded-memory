#!/usr/bin/env python3
"""Smoke test for compatibility-oriented grounded-memory usage in remote environments."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grounded_memory import Memory
from grounded_memory.core.models import EntityType


def main() -> int:
    memory = Memory(domain_profile="generic", storage_backend="memory")

    model_entity = memory.add_entity("Model Serving", entity_type=EntityType.FACILITY)["entity"]
    memory.add_fact(
        subject_id=model_entity["id"],
        relation="HAS_ATTRIBUTE",
        value="model_host=hpc-gpu-node-01",
    )

    facts = memory.list_facts()
    hits = memory.retrieve("model host", limit=3)

    print(f"facts={len(facts)}")
    print(f"hits={len(hits)}")
    print("smoke_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
