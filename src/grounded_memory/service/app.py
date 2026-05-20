"""FastAPI service layer for Grounded Memory."""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse

    _FASTAPI_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - optional dependency guard
    FastAPI = Any  # type: ignore[assignment]
    HTTPException = Any  # type: ignore[assignment]
    Request = Any  # type: ignore[assignment]
    JSONResponse = Any  # type: ignore[assignment]
    _FASTAPI_IMPORT_ERROR = exc

from grounded_memory.logging_utils import configure_logging
from grounded_memory.memory import Memory
from grounded_memory.service.models import (
    AddEntityRequest,
    AddFactRequest,
    AddMemoryRequest,
    ApiEnvelope,
    HealthResponse,
    SearchMemoryRequest,
    UpdateFactRequest,
)

logger = logging.getLogger(__name__)


def _memory_from_env() -> Memory:
    backend = os.getenv("GM_STORAGE_BACKEND")
    adapter = os.getenv("GM_ADAPTER") or os.getenv("GM_DOMAIN_PROFILE", "generic")
    return Memory(
        adapter=adapter,
        domain_profile=adapter,
        storage_backend=backend,
    )


@asynccontextmanager
async def _lifespan(app: Any):
    configure_logging()
    memory = _memory_from_env()
    app.state.memory = memory
    logger.info("Grounded Memory API started")
    try:
        yield
    finally:
        memory.close()
        logger.info("Grounded Memory API stopped")


def create_app(memory: Memory | None = None) -> Any:
    if _FASTAPI_IMPORT_ERROR is not None:
        raise RuntimeError(
            "FastAPI is not installed. Install grounded-memory with the api extra: pip install 'grounded-memory[api]'"
        ) from _FASTAPI_IMPORT_ERROR

    app = FastAPI(
        title="Grounded Memory API",
        version="0.1.0",
        description="Service layer for tuple-governed grounded memory.",
        lifespan=None if memory is not None else _lifespan,
    )

    if memory is not None:
        configure_logging()
        app.state.memory = memory

    @app.middleware("http")
    async def request_logging_middleware(request: Any, call_next):
        start = time.perf_counter()
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        try:
            response = await call_next(request)
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.exception(
                "request_failed",
                extra={
                    "request_id": request_id,
                    "path": request.url.path,
                    "method": request.method,
                    "latency_ms": latency_ms,
                },
            )
            return JSONResponse(
                status_code=500,
                content={
                    "ok": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "request_id": request_id,
                    },
                },
            )

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["x-request-id"] = request_id
        logger.info(
            "request_complete",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "latency_ms": latency_ms,
            },
        )
        return response

    def _memory() -> Memory:
        return app.state.memory

    @app.get("/health/live", response_model=HealthResponse)
    async def live() -> HealthResponse:
        return HealthResponse(data={"service": "grounded-memory-api"})

    @app.get("/health/ready", response_model=HealthResponse)
    async def ready() -> HealthResponse:
        return HealthResponse(data=_memory().healthcheck())

    @app.get("/v1/status", response_model=ApiEnvelope)
    async def status() -> ApiEnvelope:
        return ApiEnvelope(data=_memory().runtime_status())

    @app.post("/v1/memories/add", response_model=ApiEnvelope)
    async def add_memory(payload: AddMemoryRequest) -> ApiEnvelope:
        memory = _memory()
        if payload.text is not None:
            result = memory.add(
                payload.text,
                source=payload.source,
                tenant_id=payload.tenant_id,
                app_id=payload.app_id,
                user_id=payload.user_id,
                agent_id=payload.agent_id,
                run_id=payload.run_id,
                session_id=payload.session_id,
                space_type=payload.space_type,
                metadata=payload.metadata,
            )
        else:
            result = memory.add(
                [item.model_dump(exclude_none=True) for item in payload.messages or []],
                source=payload.source,
                tenant_id=payload.tenant_id,
                app_id=payload.app_id,
                user_id=payload.user_id,
                agent_id=payload.agent_id,
                run_id=payload.run_id,
                session_id=payload.session_id,
                space_type=payload.space_type,
                metadata=payload.metadata,
            )
        return ApiEnvelope(data=result)

    @app.post("/v1/memories/search", response_model=ApiEnvelope)
    async def search_memory(payload: SearchMemoryRequest) -> ApiEnvelope:
        results = _memory().search(
            payload.query,
            tenant_id=payload.tenant_id,
            app_id=payload.app_id,
            user_id=payload.user_id,
            agent_id=payload.agent_id,
            run_id=payload.run_id,
            space_type=payload.space_type,
            at_time=payload.at_time,
            lookback_days=payload.lookback_days,
            limit=payload.limit,
            max_hops=payload.max_hops,
            max_seeds=payload.max_seeds,
            strategy=payload.strategy,
        )
        return ApiEnvelope(data={"results": results})

    @app.get("/v1/memories/prompt", response_model=ApiEnvelope)
    async def build_prompt(
        query: str,
        tenant_id: str | None = None,
        app_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        space_type: str | None = None,
        limit: int = 5,
        at_time: str | None = None,
        lookback_days: int | None = None,
    ) -> ApiEnvelope:
        resolved_at_time = None
        if at_time:
            normalized_at_time = at_time.replace("Z", "+00:00")
            resolved_at_time = datetime.fromisoformat(normalized_at_time)
        prompt = _memory().build_memory_prompt(
            query,
            tenant_id=tenant_id,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            space_type=space_type,
            limit=limit,
            at_time=resolved_at_time,
            lookback_days=lookback_days,
        )
        return ApiEnvelope(data={"prompt": prompt})

    @app.get("/v1/memories/facts", response_model=ApiEnvelope)
    async def list_facts(
        tenant_id: str | None = None,
        app_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        space_type: str | None = None,
        active_only: bool = False,
        limit: int = 100,
    ) -> ApiEnvelope:
        return ApiEnvelope(
            data={
                "results": _memory().list_facts(
                    tenant_id=tenant_id,
                    app_id=app_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    run_id=run_id,
                    space_type=space_type,
                    active_only=active_only,
                    limit=limit,
                )
            }
        )

    @app.get("/v1/memories/interactions", response_model=ApiEnvelope)
    async def list_interactions(
        tenant_id: str | None = None,
        app_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        space_type: str | None = None,
        limit: int = 100,
    ) -> ApiEnvelope:
        return ApiEnvelope(
            data={
                "results": _memory().list_interactions(
                    tenant_id=tenant_id,
                    app_id=app_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    run_id=run_id,
                    space_type=space_type,
                    limit=limit,
                )
            }
        )

    @app.get("/v1/memories/all", response_model=ApiEnvelope)
    async def get_all(
        tenant_id: str | None = None,
        app_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        space_type: str | None = None,
    ) -> ApiEnvelope:
        return ApiEnvelope(
            data=_memory().get_all(
                tenant_id=tenant_id,
                app_id=app_id,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                space_type=space_type,
            )
        )

    @app.post("/v1/entities", response_model=ApiEnvelope)
    async def add_entity(payload: AddEntityRequest) -> ApiEnvelope:
        result = _memory().add_entity(**payload.model_dump(exclude_none=True))
        return ApiEnvelope(data=result)

    @app.post("/v1/facts", response_model=ApiEnvelope)
    async def add_fact(payload: AddFactRequest) -> ApiEnvelope:
        result = _memory().add_fact(**payload.model_dump(exclude_none=True))
        return ApiEnvelope(data=result)

    @app.patch("/v1/facts/{fact_id}", response_model=ApiEnvelope)
    async def update_fact(fact_id: str, payload: UpdateFactRequest) -> ApiEnvelope:
        result = _memory().update_fact(fact_id, **payload.model_dump(exclude_none=True))
        return ApiEnvelope(data=result)

    @app.delete("/v1/facts/{fact_id}", response_model=ApiEnvelope)
    async def delete_fact(fact_id: str, reason: str = "deleted via api") -> ApiEnvelope:
        result = _memory().delete_fact(fact_id, reason=reason)
        return ApiEnvelope(data=result)

    @app.exception_handler(ValueError)
    async def handle_value_error(_: Any, exc: ValueError):
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": {"type": "ValueError", "message": str(exc)}},
        )

    @app.exception_handler(RuntimeError)
    async def handle_runtime_error(_: Any, exc: RuntimeError):
        return JSONResponse(
            status_code=503,
            content={"ok": False, "error": {"type": "RuntimeError", "message": str(exc)}},
        )

    @app.exception_handler(HTTPException)
    async def handle_http_error(_: Any, exc: Any):
        return JSONResponse(
            status_code=exc.status_code,
            content={"ok": False, "error": {"type": "HTTPException", "message": exc.detail}},
        )

    return app


app = create_app() if _FASTAPI_IMPORT_ERROR is None else None
