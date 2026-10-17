"""Dependency-free media, identity, lifecycle, event, and trace contracts."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


@dataclass(frozen=True, slots=True)
class AudioFormat:
    sample_rate_hz: int
    channels: int = 1
    sample_width_bits: int = 16
    encoding: str = "pcm_s16le"

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.channels != 1:
            raise ValueError("Phase 1 supports mono PCM only")
        if self.sample_width_bits != 16:
            raise ValueError("Phase 1 supports 16-bit PCM only")
        if self.encoding != "pcm_s16le":
            raise ValueError("Phase 1 supports pcm_s16le encoding only")


@dataclass(frozen=True, slots=True)
class AudioChunk:
    data: bytes
    format: AudioFormat
    sequence: int
    timestamp_s: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes):
            raise TypeError("audio data must be bytes")
        if self.sequence < 0:
            raise ValueError("sequence must not be negative")
        bytes_per_frame = (self.format.sample_width_bits // 8) * self.format.channels
        if len(self.data) % bytes_per_frame != 0:
            raise ValueError("audio data must contain complete interleaved PCM frames")


@dataclass(frozen=True, slots=True)
class SessionIdentity:
    logical_session_id: str
    transport: str
    attempt_id: str
    transport_session_id: str | None = None
    caller_id: str | None = None
    connect_context_id: str | None = None
    contact_id: str | None = None
    legacy_call_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("logical_session_id", "transport", "attempt_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")

    @classmethod
    def for_connect(
        cls,
        *,
        attempt_id: str,
        context_id: str | None,
        contact_id: str | None,
        caller_id: str | None = None,
        transport_session_id: str | None = None,
    ) -> "SessionIdentity":
        logical_id = (context_id or contact_id or "").strip()
        if not logical_id:
            raise ValueError("Connect identity requires context_id or contact_id")
        return cls(
            logical_session_id=logical_id,
            transport="amazon-connect-a2a",
            attempt_id=attempt_id,
            transport_session_id=transport_session_id,
            caller_id=caller_id,
            connect_context_id=context_id,
            contact_id=contact_id,
            legacy_call_id=contact_id or logical_id,
        )

    def log_fields(self) -> dict[str, str]:
        fields = {
            "session_id": self.logical_session_id,
            "transport": self.transport,
            "attempt_id": self.attempt_id,
        }
        if self.legacy_call_id:
            fields["legacy_call_id"] = self.legacy_call_id
        if self.contact_id:
            fields["contact_id"] = self.contact_id
        return fields


class LifecyclePhase(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    READY = "ready"
    ACTIVE = "active"
    FINISHING = "finishing"
    CLOSED = "closed"


class SessionOutcome(str, Enum):
    COMPLETE = "complete"
    ESCALATE = "escalate"
    REMOTE_DISCONNECT = "remote_disconnect"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class EngineResult:
    outcome: SessionOutcome
    reason: str = ""


@dataclass(frozen=True, slots=True)
class SessionResult:
    identity: SessionIdentity
    outcome: SessionOutcome
    reason: str
    started_at_s: float
    ended_at_s: float


@dataclass(frozen=True, slots=True)
class SessionEvent:
    name: str
    timestamp_s: float
    payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    phase: LifecyclePhase
    timestamp_s: float
    outcome: SessionOutcome | None = None
    reason: str = ""


class TraceStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class TraceSpan:
    name: str
    trace_id: str
    span_id: str
    started_at_s: float
    ended_at_s: float
    status: TraceStatus
    attributes: Mapping[str, object] = field(default_factory=dict)
    error_type: str | None = None


class InboundAudioSource(Protocol):
    def __aiter__(self) -> AsyncIterator[AudioChunk]: ...

    async def aclose(self) -> None: ...


class OutboundAudioSink(Protocol):
    async def write(self, chunk: AudioChunk) -> None: ...

    async def interrupt(self, reason: str) -> None: ...

    async def aclose(self) -> None: ...


class EventSink(Protocol):
    async def emit(self, event: SessionEvent) -> None: ...


class LifecycleSink(Protocol):
    async def transition(self, event: LifecycleEvent) -> None: ...


class TraceSink(Protocol):
    async def emit(self, span: TraceSpan) -> None: ...


@dataclass(frozen=True, slots=True)
class SessionPorts:
    inbound_audio: InboundAudioSource
    outbound_audio: OutboundAudioSink
    events: EventSink
    lifecycle: LifecycleSink
    traces: TraceSink


@dataclass(frozen=True, slots=True)
class SessionContext:
    identity: SessionIdentity
    ports: SessionPorts


class SessionEngine(Protocol):
    async def run(self, context: SessionContext) -> EngineResult: ...
