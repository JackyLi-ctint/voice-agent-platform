"""Minimal structured logging that does not reconfigure the legacy Loguru app."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from ..session.contracts import SessionIdentity

_CORRELATION_FIELDS = (
    "session_id",
    "transport",
    "attempt_id",
    "legacy_call_id",
    "contact_id",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in _CORRELATION_FIELDS:
            value = getattr(record, field, None)
            if value:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging(*, level: str = "INFO", json_output: bool = False) -> None:
    """Configure stdlib logging for new modules only when explicitly invoked."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter() if json_output else logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    ))
    root = logging.getLogger("voice_agent")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


def session_logger(
    identity: SessionIdentity,
    logger: logging.Logger | None = None,
) -> logging.LoggerAdapter:
    """Bind stable correlation fields without exposing caller data or secrets."""
    base = logger or logging.getLogger("voice_agent.session")
    return logging.LoggerAdapter(base, identity.log_fields())
