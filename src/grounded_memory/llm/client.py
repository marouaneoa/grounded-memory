"""
LLM Client

Provides a unified interface to LLM APIs (OpenAI-compatible endpoints).
Supports both direct API calls and Pydantic AI integration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel

from grounded_memory.llm.prompts import (
    STRUCTURED_EXTRACTION_SYSTEM_PROMPT,
    build_structured_extraction_user_prompt,
)

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


class LLMProvider:
    """Enum-like class for LLM providers."""

    LOCAL = "local"
    OPENROUTER = "openrouter"


@dataclass
class LLMConfig:
    """Configuration for LLM connection."""

    # Provider selection
    provider: str = LLMProvider.OPENROUTER

    # OpenRouter configuration (default)
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "z-ai/glm-4.5-air:free"  # Default OpenRouter model
    api_key: str = ""  # Set via OPENROUTER_API_KEY env var

    # Generation parameters
    temperature: float = 0.1  # Low for factual extraction
    max_tokens: int = 2048
    timeout: float = 120.0  # 2 minutes timeout

    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0

    # OpenRouter-specific headers
    site_url: str = ""  # Your site URL for OpenRouter rankings
    site_name: str = "GroundedMemory"  # Your app name for OpenRouter

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Create config from environment variables."""
        provider = os.getenv("LLM_PROVIDER", LLMProvider.OPENROUTER)
        if provider not in {LLMProvider.OPENROUTER, LLMProvider.LOCAL}:
            raise ValueError(
                f"Unsupported LLM_PROVIDER '{provider}'. "
                f"Expected one of: {LLMProvider.OPENROUTER}, {LLMProvider.LOCAL}"
            )

        # Set defaults based on provider
        if provider == LLMProvider.OPENROUTER:
            default_url = "https://openrouter.ai/api/v1"
            default_model = "z-ai/glm-4.5-air:free"
            default_key = os.getenv("OPENROUTER_API_KEY", "")
        else:  # LOCAL
            default_url = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
            default_model = os.getenv("LLM_MODEL", "local-model")
            default_key = os.getenv("LLM_API_KEY", "")

        config = cls(
            provider=provider,
            base_url=os.getenv("LLM_BASE_URL", default_url),
            model=os.getenv("LLM_MODEL", default_model),
            api_key=os.getenv("LLM_API_KEY", default_key)
            if provider == LLMProvider.LOCAL
            else os.getenv("OPENROUTER_API_KEY", default_key),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
            timeout=float(os.getenv("LLM_TIMEOUT", "120")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("LLM_RETRY_DELAY", "1.0")),
            site_url=os.getenv("OPENROUTER_SITE_URL", ""),
            site_name=os.getenv("OPENROUTER_SITE_NAME", "GroundedMemory"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Validate configuration and fail fast on invalid setup."""
        if not self.base_url.strip():
            raise ValueError("LLM_BASE_URL must not be empty")

        if not self.model.strip():
            raise ValueError("LLM_MODEL must not be empty")

        if self.provider == LLMProvider.OPENROUTER and not self.api_key.strip():
            raise ValueError("OPENROUTER_API_KEY must be set when LLM_PROVIDER=openrouter")

        if self.timeout <= 0:
            raise ValueError("LLM_TIMEOUT must be > 0")

        if self.max_tokens <= 0:
            raise ValueError("LLM_MAX_TOKENS must be > 0")

        if self.max_retries < 0:
            raise ValueError("LLM_MAX_RETRIES must be >= 0")

    @classmethod
    def openrouter(cls, api_key: str, model: str = "z-ai/glm-4.5-air:free") -> LLMConfig:
        """Create an OpenRouter configuration."""
        return cls(
            provider=LLMProvider.OPENROUTER,
            base_url="https://openrouter.ai/api/v1",
            model=model,
            api_key=api_key,
        )

    @classmethod
    def local(cls, base_url: str, model: str, api_key: str = "") -> LLMConfig:
        """Create a local LLM configuration."""
        return cls(
            provider=LLMProvider.LOCAL,
            base_url=base_url,
            model=model,
            api_key=api_key,
        )


T = TypeVar("T", bound=BaseModel)


def _normalize_message_content(message: dict[str, Any]) -> str:
    content = message.get("content")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return "\n".join(parts)

    reasoning = message.get("reasoning")
    if isinstance(reasoning, str):
        return reasoning

    return ""


def _compute_retry_delay(
    base_delay: float,
    attempt: int,
    retry_after_header: str | None = None,
) -> float:
    if retry_after_header:
        try:
            return max(float(retry_after_header), 0.0)
        except ValueError:
            pass
    return base_delay * (2**attempt)


def _chat_completions_path() -> str:
    """Return a relative OpenAI-compatible chat completions path.

    Using a leading slash would discard any base path suffix such as `/v1`
    when combined with `httpx` `base_url`.
    """
    return "chat/completions"


class LLMClient:
    """
    Client for interacting with OpenAI-compatible LLM APIs.

    Supports structured output extraction using Pydantic models.

    Usage:
        config = LLMConfig()
        client = LLMClient(config)

        # Simple completion
        response = await client.complete("What is 2+2?")

        # Structured extraction
        result = await client.extract(
            "Patient Alice takes Aspirin 100mg daily",
            ExtractionResult,
        )
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self.config.validate()
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }

            # Add OpenRouter-specific headers
            if self.config.provider == LLMProvider.OPENROUTER:
                headers["HTTP-Referer"] = (
                    self.config.site_url or "https://github.com/grounded-memory"
                )
                headers["X-Title"] = self.config.site_name

            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.config.timeout),
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Get a text completion from the LLM.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            The LLM's response text
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        if kwargs.get("response_format") is not None:
            payload["response_format"] = kwargs["response_format"]

        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self.client.post(_chat_completions_path(), json=payload)
                response.raise_for_status()
                data = response.json()
                message = data["choices"][0]["message"]
                return _normalize_message_content(message)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                retryable = status in {429, 500, 502, 503, 504}
                if not retryable or attempt >= self.config.max_retries:
                    raise
                delay = _compute_retry_delay(
                    self.config.retry_delay,
                    attempt,
                    exc.response.headers.get("Retry-After"),
                )
                logger.warning(
                    "LLM request failed with HTTP %s; retrying in %.1fs (attempt %s/%s)",
                    status,
                    delay,
                    attempt + 1,
                    self.config.max_retries,
                )
                await asyncio.sleep(delay)
            except httpx.RequestError:
                if attempt >= self.config.max_retries:
                    raise
                delay = _compute_retry_delay(self.config.retry_delay, attempt)
                logger.warning(
                    "LLM request transport error; retrying in %.1fs (attempt %s/%s)",
                    delay,
                    attempt + 1,
                    self.config.max_retries,
                )
                await asyncio.sleep(delay)

        raise RuntimeError("LLM completion failed after retries")

    async def extract(
        self,
        text: str,
        output_model: type[T],
        system_prompt: str | None = None,
        extraction_prompt: str | None = None,
    ) -> T:
        """
        Extract structured data from text using the LLM.

        Args:
            text: The text to extract from
            output_model: Pydantic model for the output
            system_prompt: Optional system prompt override
            extraction_prompt: Optional extraction prompt override

        Returns:
            An instance of output_model with extracted data
        """
        # Build JSON schema from Pydantic model
        schema = output_model.model_json_schema()
        schema_json = json.dumps(schema, indent=2)

        # Build the prompt
        if system_prompt is None:
            system_prompt = self._build_extraction_system_prompt(schema)

        if extraction_prompt is None:
            extraction_prompt = self._build_extraction_prompt(text, schema)
        else:
            extraction_prompt = extraction_prompt.format(
                input_text=text,
                output_schema_json=schema_json,
                text=text,
                schema=schema_json,
            )

        # Get completion
        response = await self.complete(
            extraction_prompt,
            system_prompt,
            response_format={"type": "json_object"},
        )

        # Parse JSON from response
        json_data = self._extract_json(response)

        # Validate and return
        return output_model.model_validate(json_data)

    def _build_extraction_system_prompt(self, _schema: dict) -> str:
        """Build system prompt for structured extraction."""
        return STRUCTURED_EXTRACTION_SYSTEM_PROMPT

    def _build_extraction_prompt(self, text: str, schema: dict) -> str:
        """Build extraction prompt."""
        return build_structured_extraction_user_prompt(input_text=text, output_schema=schema)

    def _extract_json(self, response: str) -> dict:
        """Extract JSON from LLM response."""
        if not isinstance(response, str) or not response.strip():
            raise ValueError("Unable to extract valid JSON from LLM response: empty content")

        # Try direct parsing first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in code blocks
        import re

        json_patterns = [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
            r"\{[\s\S]*\}",
        ]

        for pattern in json_patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    json_str = match.group(1) if "```" in pattern else match.group(0)
                    return json.loads(json_str)
                except (json.JSONDecodeError, IndexError):
                    continue

        preview = response[:400].replace("\n", " ")
        raise ValueError(f"Unable to extract valid JSON from LLM response. Preview: {preview}")


class SyncLLMClient:
    """
    Synchronous wrapper for LLMClient.

    Provides a blocking interface for use in non-async code.
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig.from_env()
        self.config.validate()

    def _get_headers(self) -> dict:
        """Get request headers based on provider."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        # Add OpenRouter-specific headers
        if self.config.provider == LLMProvider.OPENROUTER:
            headers["HTTP-Referer"] = self.config.site_url or "https://github.com/grounded-memory"
            headers["X-Title"] = self.config.site_name

        return headers

    def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Synchronous text completion."""
        import httpx

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        if kwargs.get("response_format") is not None:
            payload["response_format"] = kwargs["response_format"]

        with httpx.Client(
            base_url=self.config.base_url,
            headers=self._get_headers(),
            timeout=httpx.Timeout(self.config.timeout),
        ) as client:
            for attempt in range(self.config.max_retries + 1):
                try:
                    response = client.post(_chat_completions_path(), json=payload)
                    response.raise_for_status()
                    data = response.json()

                    if "error" in data:
                        raise RuntimeError(
                            f"LLM API error: {data['error'].get('message', 'Unknown error')}"
                        )

                    message = data["choices"][0]["message"]
                    return _normalize_message_content(message)
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    retryable = status in {429, 500, 502, 503, 504}
                    if not retryable or attempt >= self.config.max_retries:
                        raise
                    delay = _compute_retry_delay(
                        self.config.retry_delay,
                        attempt,
                        exc.response.headers.get("Retry-After"),
                    )
                    logger.warning(
                        "LLM request failed with HTTP %s; retrying in %.1fs (attempt %s/%s)",
                        status,
                        delay,
                        attempt + 1,
                        self.config.max_retries,
                    )
                    time.sleep(delay)
                except httpx.RequestError:
                    if attempt >= self.config.max_retries:
                        raise
                    delay = _compute_retry_delay(self.config.retry_delay, attempt)
                    logger.warning(
                        "LLM request transport error; retrying in %.1fs (attempt %s/%s)",
                        delay,
                        attempt + 1,
                        self.config.max_retries,
                    )
                    time.sleep(delay)

        raise RuntimeError("LLM completion failed after retries")

    def extract(
        self,
        text: str,
        output_model: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Synchronous structured extraction."""
        schema = output_model.model_json_schema()

        if system_prompt is None:
            system_prompt = STRUCTURED_EXTRACTION_SYSTEM_PROMPT

        extraction_prompt = build_structured_extraction_user_prompt(
            input_text=text,
            output_schema=schema,
        )

        response = self.complete(
            extraction_prompt,
            system_prompt,
            response_format={"type": "json_object"},
        )

        # Parse JSON from response
        json_data = self._extract_json(response)

        return output_model.model_validate(json_data)

    def _extract_json(self, response: str) -> dict:
        """Extract JSON from LLM response."""
        if not isinstance(response, str) or not response.strip():
            raise ValueError("Unable to extract valid JSON from LLM response: empty content")

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        import re

        json_patterns = [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
            r"\{[\s\S]*\}",
        ]

        for pattern in json_patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    json_str = match.group(1) if "```" in pattern else match.group(0)
                    return json.loads(json_str)
                except (json.JSONDecodeError, IndexError):
                    continue

        preview = response[:400].replace("\n", " ")
        raise ValueError(f"Unable to extract valid JSON from LLM response. Preview: {preview}")
