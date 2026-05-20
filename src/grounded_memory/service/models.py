"""Pydantic request/response models for the Grounded Memory API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class MessagePayload(BaseModel):
    """Chat-style message used for memory ingestion."""

    role: str = Field(default="user")
    content: str = Field(..., min_length=1)
    tenant_id: str | None = None
    app_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    space_type: str | None = None
    metadata: dict[str, Any] | None = None


class AddMemoryRequest(BaseModel):
    """Request payload for conversational memory ingestion."""

    text: str | None = None
    messages: list[MessagePayload] | None = None
    source: str = "user"
    tenant_id: str | None = None
    app_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    space_type: str | None = None
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_exactly_one_input(self) -> AddMemoryRequest:
        has_text = bool(self.text and self.text.strip())
        has_messages = bool(self.messages)
        if has_text == has_messages:
            raise ValueError("Provide exactly one of 'text' or 'messages'")
        return self


class SearchMemoryRequest(BaseModel):
    """Request payload for search/retrieve calls."""

    query: str = Field(..., min_length=1)
    tenant_id: str | None = None
    app_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    space_type: str | None = None
    at_time: datetime | None = None
    lookback_days: int | None = Field(default=None, ge=1, le=3650)
    limit: int = Field(default=10, ge=1, le=100)
    max_hops: int | None = Field(default=None, ge=1, le=5)
    max_seeds: int | None = Field(default=None, ge=1, le=20)
    strategy: str | None = None


class AddEntityRequest(BaseModel):
    name: str = Field(..., min_length=1)
    entity_type: str = "FACILITY"
    attributes: dict[str, Any] | None = None
    canonical_id: str | None = None
    uniqueness_key: str | None = None
    entity_id: str | None = None


class AddFactRequest(BaseModel):
    subject_id: str
    relation: str
    object_id: str | None = None
    value: str | None = None
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    attributes: dict[str, Any] | None = None
    source: str = "system"
    tenant_id: str | None = None
    app_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    space_type: str | None = None
    source_interaction_id: str | None = None

    @model_validator(mode="after")
    def validate_object_or_value(self) -> AddFactRequest:
        if self.object_id is None and (self.value is None or not self.value.strip()):
            raise ValueError("Provide 'object_id' or 'value'")
        return self


class UpdateFactRequest(BaseModel):
    relation: str | None = None
    object_id: str | None = None
    value: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    attributes: dict[str, Any] | None = None
    source: str = "system"
    tenant_id: str | None = None
    app_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    space_type: str | None = None


class DeleteFactRequest(BaseModel):
    reason: str = "deleted via api"


class ApiEnvelope(BaseModel):
    """Small consistent response envelope."""

    ok: bool = True
    data: Any


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    data: dict[str, Any]
