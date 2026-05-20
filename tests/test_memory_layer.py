#!/usr/bin/env python3
"""LLM-only smoke test for the grounded-memory layer.

Usage:
    PYTHONPATH=src python scripts/test_memory_layer.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grounded_memory import Memory


def run_llm_smoke() -> None:
    model = os.getenv("LLM_MODEL", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    api_key = os.getenv("LLM_API_KEY", "").strip()

    if not model:
        raise RuntimeError("LLM_MODEL is required")
    if not base_url:
        raise RuntimeError("LLM_BASE_URL is required")
    if not api_key:
        raise RuntimeError("LLM_API_KEY is required")

    os.environ["LLM_PROVIDER"] = "local"

    memory = Memory(domain_profile="generic", storage_backend="memory", use_llm=True)
    user_id = "default_user"
    memory.add(
        [
            {"role": "user", "content": "I use Qwen3.5-122B-A10B-FP8 on vllm-nodeport."},
            {"role": "assistant", "content": "Noted: you use Qwen3.5-122B-A10B-FP8."},
        ],
        user_id=user_id,
    )

    results = memory.retrieve("what model do I use", user_id=user_id, limit=5)
    print(f"llm_results={len(results)}")
    if not results:
        raise RuntimeError("LLM smoke test failed: no retrieval results produced")
    print(f"llm_top_hit={results[0].get('value')}")


def main() -> int:
    run_llm_smoke()

    print("ok")
    return 0


if __name__ == "__main__":
    main()
