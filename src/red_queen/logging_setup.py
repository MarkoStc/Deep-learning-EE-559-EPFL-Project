"""Logging configuration.

Modules log through ``logging.getLogger(__name__)`` and attach structured fields
with ``extra={"fields": {...}}``. :func:`configure_logging` picks the output
format: readable ``key=value`` lines for a terminal, or one JSON object per line
for log collectors.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, Literal

LogFormat = Literal["text", "json"]

_NOISY_LIBRARIES = ("urllib3", "filelock", "fsspec", "datasets", "huggingface_hub", "matplotlib")


def _fields(record: logging.LogRecord) -> dict[str, Any]:
    fields = getattr(record, "fields", None)
    return dict(fields) if isinstance(fields, dict) else {}


class KeyValueFormatter(logging.Formatter):
    """``2026-10-14 12:00:00 INFO red_queen.x: message key=value``."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        extras = " ".join(f"{k}={_format_value(v)}" for k, v in _fields(record).items())
        return f"{line} {extras}" if extras else line


class JsonFormatter(logging.Formatter):
    """One JSON object per record, with structured fields at the top level."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **_fields(record),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def configure_logging(level: str = "INFO", fmt: LogFormat = "text") -> None:
    """Install one stderr handler on the root logger, replacing earlier ones."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else KeyValueFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for name in _NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)


def fields(**values: Any) -> dict[str, Any]:
    """Build the ``extra`` argument for a structured log call."""
    return {"fields": values}
