"""
LLM Integration Module

This module provides LLM integration for the Grounded Memory System,
supporting multiple providers: OpenRouter, OpenAI, and local endpoints.
"""

from grounded_memory.llm.client import (
    LLMClient,
    LLMConfig,
    LLMProvider,
    SyncLLMClient,
)
from grounded_memory.llm.extractor import LLMFactExtractor
from grounded_memory.llm.prompts import (
    CLINICAL_EXTRACTION_SYSTEM_PROMPT,
    CONNECTIVITY_TEST_SYSTEM_PROMPT,
    EDGE_EXTRACTION_SYSTEM_PROMPT,
    ENTITY_EXTRACTION_SYSTEM_PROMPT,
    GENERIC_TUPLE_EXTRACTION_SYSTEM_PROMPT,
    STRUCTURED_EXTRACTION_SYSTEM_PROMPT,
    TEMPORAL_GROUNDING_SYSTEM_PROMPT,
    build_chat_with_memory_system_prompt,
    build_clinical_extraction_user_prompt,
    build_edge_extraction_user_prompt,
    build_entity_extraction_user_prompt,
    build_generic_tuple_extraction_user_prompt,
    build_structured_extraction_user_prompt,
    build_temporal_grounding_user_prompt,
)

__all__ = [
    "LLMClient",
    "LLMConfig",
    "LLMProvider",
    "SyncLLMClient",
    "LLMFactExtractor",
    "STRUCTURED_EXTRACTION_SYSTEM_PROMPT",
    "CLINICAL_EXTRACTION_SYSTEM_PROMPT",
    "GENERIC_TUPLE_EXTRACTION_SYSTEM_PROMPT",
    "ENTITY_EXTRACTION_SYSTEM_PROMPT",
    "EDGE_EXTRACTION_SYSTEM_PROMPT",
    "TEMPORAL_GROUNDING_SYSTEM_PROMPT",
    "CONNECTIVITY_TEST_SYSTEM_PROMPT",
    "build_structured_extraction_user_prompt",
    "build_clinical_extraction_user_prompt",
    "build_generic_tuple_extraction_user_prompt",
    "build_entity_extraction_user_prompt",
    "build_edge_extraction_user_prompt",
    "build_temporal_grounding_user_prompt",
    "build_chat_with_memory_system_prompt",
]
