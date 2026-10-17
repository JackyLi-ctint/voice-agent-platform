"""Side-effect-free settings and secret handling conventions.

This module deliberately does not load ``.env`` files or contact AWS. Callers may
load a local development file before invoking :meth:`ApplicationSettings.from_env`.
Production values must come from the process environment or an explicit secret
provider before application construction.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", ""})
_ALLOWED_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"})


class SecretValue:
    """Opaque secret whose display and dataclass traversal never expose value."""

    __slots__ = ("__value",)

    def __init__(self, value: str) -> None:
        self.__value = str(value)

    def reveal(self) -> str:
        """Return the underlying value at the integration boundary only."""
        return self.__value

    def __bool__(self) -> bool:
        return bool(self.__value)

    def __str__(self) -> str:
        return "***" if self.__value else ""

    def __repr__(self) -> str:
        return "SecretValue('***')" if self.__value else "SecretValue('')"


def _parse_bool(name: str, value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be one of true/false, 1/0, yes/no, or on/off")


@dataclass(frozen=True, slots=True)
class ApplicationSettings:
    """Stable process settings for the new application foundation.

    Precedence is explicit constructor overrides, then the supplied/process
    environment, then safe defaults. Existing legacy environment names remain
    owned by ``bot.py`` until their focused migration.
    """

    service_name: str = "voice-agent-ac"
    environment: str = "development"
    log_level: str = "INFO"
    log_json: bool = False
    connect_a2a_api_key: SecretValue = SecretValue("")

    @classmethod
    def from_env(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        overrides: Mapping[str, object] | None = None,
    ) -> "ApplicationSettings":
        source = os.environ if environ is None else environ
        defaults = cls()
        values: dict[str, object] = {
            "service_name": source.get("SERVICE_NAME", defaults.service_name),
            "environment": source.get("APP_ENV", defaults.environment),
            "log_level": source.get("LOG_LEVEL", defaults.log_level),
            "log_json": source.get("LOG_JSON", str(defaults.log_json)),
            "connect_a2a_api_key": source.get("CONNECT_A2A_API_KEY", ""),
        }
        values.update(overrides or {})

        service_name = str(values["service_name"]).strip()
        environment = str(values["environment"]).strip().lower()
        log_level = str(values["log_level"]).strip().upper()
        if not service_name:
            raise ValueError("SERVICE_NAME must not be empty")
        if not environment:
            raise ValueError("APP_ENV must not be empty")
        if log_level not in _ALLOWED_LOG_LEVELS:
            raise ValueError(
                "LOG_LEVEL must be one of " + ", ".join(sorted(_ALLOWED_LOG_LEVELS))
            )

        raw_secret = values["connect_a2a_api_key"]
        secret = raw_secret if isinstance(raw_secret, SecretValue) else SecretValue(str(raw_secret))
        return cls(
            service_name=service_name,
            environment=environment,
            log_level=log_level,
            log_json=_parse_bool("LOG_JSON", values["log_json"]),
            connect_a2a_api_key=secret,
        )

    def safe_dict(self) -> dict[str, object]:
        """Return a log-safe settings view."""
        return {
            "service_name": self.service_name,
            "environment": self.environment,
            "log_level": self.log_level,
            "log_json": self.log_json,
            "connect_a2a_api_key": str(self.connect_a2a_api_key),
        }
