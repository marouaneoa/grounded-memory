"""Logging helpers for Grounded Memory runtime services."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any


class JsonLogFormatter(logging.Formatter):
    """Minimal JSON log formatter for service deployments."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "request_id"):
            payload["request_id"] = record.request_id
        if hasattr(record, "path"):
            payload["path"] = record.path
        if hasattr(record, "method"):
            payload["method"] = record.method
        if hasattr(record, "latency_ms"):
            payload["latency_ms"] = record.latency_ms
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(*, level: str | None = None, json_output: bool | None = None) -> None:
    """Configure root logging once for CLI and service entrypoints."""
    resolved_level = (level or os.getenv("GM_LOG_LEVEL") or "INFO").upper()
    resolved_json = json_output if json_output is not None else os.getenv("GM_LOG_JSON", "0") == "1"

    root = logging.getLogger()
    root.setLevel(resolved_level)

    if root.handlers:
        for handler in root.handlers:
            handler.setLevel(resolved_level)
            if resolved_json:
                handler.setFormatter(JsonLogFormatter())
            else:
                handler.setFormatter(
                    logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
                )
        return

    handler = logging.StreamHandler()
    handler.setLevel(resolved_level)
    if resolved_json:
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    root.addHandler(handler)
