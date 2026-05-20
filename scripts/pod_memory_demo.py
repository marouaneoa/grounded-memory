import asyncio
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grounded_memory import Memory
from grounded_memory.llm.prompts import build_chat_with_memory_system_prompt


class VLLMClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def chat(self, message: str) -> str:
        return await self.chat_messages(
            [
                {"role": "user", "content": message},
            ]
        )

    async def chat_messages(self, messages: list[dict[str, str]]) -> str:
        timeout = httpx.Timeout(120.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]


BASE_URL = os.getenv("LLM_BASE_URL", "http://vllm-nodeport.vllm-ns.svc.cluster.local:8000/v1")
MODEL = os.getenv("LLM_MODEL", "Qwen3.5-122B-A10B-FP8")
API_KEY = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "dummy-key"))
USER_ID = os.getenv("GM_USER_ID", "pod-user")


os.environ.setdefault("LLM_PROVIDER", "local")
os.environ.setdefault("LLM_BASE_URL", BASE_URL)
os.environ.setdefault("LLM_MODEL", MODEL)
os.environ.setdefault("LLM_API_KEY", API_KEY)


vllm = VLLMClient(BASE_URL, MODEL)
memory = Memory(domain_profile="generic")


async def chat_with_memory(message: str, user_id: str = USER_ID) -> dict:
    memory_block = memory.build_memory_prompt(
        message,
        user_id=user_id,
        limit=5,
        threshold=0.10,
    )

    system_prompt = build_chat_with_memory_system_prompt(memory_block=memory_block)

    assistant_text = await vllm.chat_messages(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ]
    )

    write_result = memory.add(
        [
            {"role": "user", "content": message},
            {"role": "assistant", "content": assistant_text},
        ],
        user_id=user_id,
    )

    return {
        "memory_block": memory_block,
        "assistant_text": assistant_text,
        "write_result": write_result,
    }


async def verify_graph(user_id: str = USER_ID) -> dict:
    entities = memory.list_entities()
    facts = memory.list_facts(user_id=user_id)
    context = memory.build_context("what do you know about me", user_id=user_id, max_facts=10)

    neighborhood = {}
    if entities:
        entity_id = entities[0]["id"]
        sub_entities, sub_facts = memory.retriever.get_entity_neighborhood(entity_id, radius=1)
        neighborhood = {
            "seed_entity_id": entity_id,
            "neighbor_entities": [entity.model_dump() for entity in sub_entities.values()],
            "neighbor_facts": [fact.model_dump() for fact in sub_facts],
        }

    return {
        "runtime_status": memory.runtime_status(),
        "entities": entities,
        "facts": facts,
        "context": context.model_dump(),
        "neighborhood": neighborhood,
    }


async def test_main_functionalities(user_id: str = USER_ID) -> dict:
    results = {}

    results["write_preference"] = memory.add(
        [{"role": "user", "content": "I prefer Rust."}],
        user_id=user_id,
    )
    results["write_location"] = memory.add(
        [{"role": "user", "content": "I live in Paris."}],
        user_id=user_id,
    )

    results["search_before_refine"] = memory.search(
        "what do I prefer",
        user_id=user_id,
        limit=5,
        rerank_debug=True,
    )

    results["refine_preference"] = memory.add(
        [{"role": "user", "content": "I prefer Python now."}],
        user_id=user_id,
    )

    results["retire_preference"] = memory.add(
        [{"role": "user", "content": "Remove my preference for Python."}],
        user_id=user_id,
    )

    all_facts = memory.list_facts(user_id=user_id)
    results["list_facts"] = all_facts
    results["list_interactions"] = memory.list_interactions(limit=20, user_id=user_id)
    results["prompt_block"] = memory.build_memory_prompt(
        "summarize what you remember about me",
        user_id=user_id,
        limit=5,
    )

    if all_facts:
        fact_id = all_facts[0]["id"]
        results["history"] = memory.history(fact_id=fact_id, user_id=user_id)

    return results


async def main() -> None:
    print("=== chat_with_memory demo ===")
    chat_result = await chat_with_memory("My favorite database for Apollo is PostgreSQL.")
    print(chat_result["memory_block"])
    print(chat_result["assistant_text"])

    print("\n=== main functionality check ===")
    functional = await test_main_functionalities()
    print(functional)

    print("\n=== graph verification ===")
    graph = await verify_graph()
    print(graph)


def run_main():
    """Run the demo in scripts or schedule it safely inside notebooks."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(main())

    task = loop.create_task(main())
    print(
        "Active event loop detected. Scheduled demo task; use `await main()` in notebooks for direct execution."
    )
    return task


if __name__ == "__main__":
    demo_task = run_main()
