"""
LLM-Based Fact Extractor

Uses Pydantic AI and an LLM to extract structured domain facts from unstructured text.
This module is domain-agnostic. For use-case specific extractors, see the `adapters` module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

from grounded_memory.llm.client import LLMConfig, SyncLLMClient
from grounded_memory.llm.prompts import CONNECTIVITY_TEST_SYSTEM_PROMPT

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMFactExtractor:
    """
    Generic LLM-powered fact extractor for domain text.
    """

    config: LLMConfig = None
    client: SyncLLMClient = None

    def __post_init__(self):
        if self.config is None:
            self.config = LLMConfig.from_env()
        if self.client is None:
            self.client = SyncLLMClient(self.config)

    def extract(
        self,
        text: str,
        output_model: type[T],
        system_prompt: str,
        include_context: str | None = None,
    ) -> T:
        """
        Extract structured information from text using LLM.
        """
        prompt = text
        if include_context:
            prompt = f"Context:\n{include_context}\n\nInput:\n{text}"

        result = self.client.extract(
            text=prompt,
            output_model=output_model,
            system_prompt=system_prompt,
        )
        return result

    def test_connection(self) -> bool:
        """
        Test the LLM connection.
        """
        try:
            response = self.client.complete(
                "Respond with exactly: OK",
                system_prompt=CONNECTIVITY_TEST_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=10,
            )
            return "OK" in response
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
