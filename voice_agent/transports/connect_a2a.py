"""Amazon Connect external A2A audio-session transport primitives.

The module implements the documented A2A v1.0 JSON-RPC event shapes used by
Amazon Connect Customer's external audio collaborators. It intentionally does
*not* mount a FastAPI route yet: the existing Pipecat core still owns a legacy
FastAPI WebSocket transport, and mounting this adapter before that core is
migrated would create a second, misleading production path.

A future ASGI handler should validate the upgrade headers, call
:meth:`ConnectA2AAdapter.initialize`, dispatch every JSON-RPC request through
:meth:`ConnectA2ASession.handle_request`, and write returned envelopes to the
same socket. The internal session ports are already transport-independent.

Contract reference:
https://docs.aws.amazon.com/connect/latest/devguide/a2a-developer-guide.html
https://docs.aws.amazon.com/connect/latest/devguide/a2a-sample-events.html
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import hmac
import json
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from ..app.readiness import ReadinessState
from ..app.settings import SecretValue
from ..session.contracts import (
    AudioChunk,
    AudioFormat,
    EngineResult,
    EventSink,
    LifecycleSink,
    SessionContext,
    SessionEngine,
    SessionEvent,
    SessionIdentity,
    SessionOutcome,
    SessionPorts,
    TraceSink,
    TraceSpan,
    TraceStatus,
)
from ..session.runner import SessionRunner
from .memory import MemoryEventSink, MemoryLifecycleSink, MemoryTraceSink

EXTENSION_URI = "https://docs.aws.amazon.com/connect/a2a/ext/v1"
EVENT_TYPE_KEY = f"{EXTENSION_URI}/eventType"
EXTENSION_HEADER = "A2A-Extensions"
SUPPORTED_SAMPLE_RATES = frozenset({8000, 16000, 24000})

SendEnvelope = Callable[[dict[str, object]], Awaitable[None]]
IdFactory = Callable[[], str]


class ConnectA2AError(RuntimeError):
    """Base error carrying a safe JSON-RPC/A2A failure classification."""

    error_code = 400
    error_category = "BAD_REQUEST"
    close_code = 1008


class ConnectA2AAuthenticationError(ConnectA2AError):
    error_code = 401
    error_category = "BAD_REQUEST"


class ConnectA2ANotReadyError(ConnectA2AError):
    error_code = 503
    error_category = "DEPENDENCY_FAILURE"
    close_code = 1013


class ConnectA2AConflictError(ConnectA2AError):
    error_code = 409
    error_category = "BAD_REQUEST"


class ConnectA2AOverloadError(ConnectA2AError):
    error_code = 429
    error_category = "DEPENDENCY_FAILURE"
    close_code = 1013


class ConnectA2AProtocolError(ConnectA2AError):
    """Malformed or unsupported A2A wire payload."""


@dataclass(frozen=True, slots=True)
class ConnectAudioConfiguration:
    """Negotiated Connect audio configuration (LINEAR_PCM, PCM16 mono)."""

    sample_rate_hz: int
    encoding: str = "LINEAR_PCM"
    sample_size_bits: int = 16
    channel_count: int = 1

    def __post_init__(self) -> None:
        if self.encoding != "LINEAR_PCM":
            raise ConnectA2AProtocolError("audio encoding must be LINEAR_PCM")
        if self.sample_rate_hz not in SUPPORTED_SAMPLE_RATES:
            raise ConnectA2AProtocolError(
                "audio sampleRateHertz must be one of 8000, 16000, or 24000"
            )
        if self.sample_size_bits != 16:
            raise ConnectA2AProtocolError("audio sampleSizeBits must be 16")
        if self.channel_count != 1:
            raise ConnectA2AProtocolError("audio channelCount must be 1")

    @classmethod
    def from_wire(cls, payload: Mapping[str, object]) -> "ConnectAudioConfiguration":
        try:
            return cls(
                encoding=str(payload["encoding"]),
                sample_rate_hz=int(payload["sampleRateHertz"]),
                sample_size_bits=int(payload["sampleSizeBits"]),
                channel_count=int(payload["channelCount"]),
            )
        except KeyError as exc:
            raise ConnectA2AProtocolError(
                f"audio configuration missing {exc.args[0]}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ConnectA2AProtocolError("invalid audio configuration") from exc

    def to_wire(self) -> dict[str, object]:
        return {
            "encoding": self.encoding,
            "sampleRateHertz": self.sample_rate_hz,
            "sampleSizeBits": self.sample_size_bits,
            "channelCount": self.channel_count,
        }

    def to_audio_format(self) -> AudioFormat:
        return AudioFormat(sample_rate_hz=self.sample_rate_hz)


@dataclass(frozen=True, slots=True)
class ConnectTraceContext:
    trace_id: str
    parent_span_id: str | None = None

    @classmethod
    def from_headers(
        cls, headers: Mapping[str, str], *, id_factory: IdFactory
    ) -> "ConnectTraceContext":
        trace_id = ""
        w3c_parent = ""
        traceparent = _header(headers, "traceparent")
        if traceparent:
            fields = traceparent.split("-")
            if len(fields) == 4 and len(fields[1]) == 32 and len(fields[2]) == 16:
                if all(c in "0123456789abcdefABCDEF" for c in fields[1] + fields[2]):
                    trace_id = fields[1].lower()
                    w3c_parent = fields[2].lower()
        if not trace_id:
            trace_id = id_factory().replace("-", "")[:32]

        # Connect supplies its parent span as a tracestate member. Prefer that
        # relationship over the generic W3C parent when it is a valid OTLP span
        # identifier; retain the W3C value as a safe fallback for older callers.
        connect_parent = _connect_parent_span_id(_header(headers, "tracestate"))
        return cls(trace_id=trace_id, parent_span_id=connect_parent or w3c_parent or None)


@dataclass(frozen=True, slots=True)
class ConnectA2AInit:
    request_id: str | int
    context_id: str
    contact_arn: str | None
    caller_id: str | None
    supported_finish_types: frozenset[str]
    subscribe_to_tracing_events: bool
    input_audio: ConnectAudioConfiguration
    output_audio: ConnectAudioConfiguration


@dataclass(frozen=True, slots=True)
class ConnectA2ARequest:
    request_id: str | int
    context_id: str
    event_type: str
    message: Mapping[str, object]
    part: Mapping[str, object]


class ConnectA2AWireCodec:
    """Codec for documented Connect A2A JSON-RPC frames.

    It maps only the public A2A events required by the first audio milestone.
    Fragment reassembly, DTMF and full OTLP child-span encoding are deliberately
    left for a subsequent hardening milestone.
    """

    def __init__(self, *, id_factory: IdFactory = lambda: str(uuid.uuid4())) -> None:
        self._id_factory = id_factory

    def parse_request(self, raw: str | bytes | Mapping[str, object]) -> ConnectA2ARequest:
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ConnectA2AProtocolError("JSON-RPC frame must be UTF-8") from exc
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ConnectA2AProtocolError("invalid JSON-RPC frame") from exc
        if not isinstance(raw, Mapping):
            raise ConnectA2AProtocolError("JSON-RPC frame must be an object")
        if raw.get("jsonrpc") != "2.0" or raw.get("method") != "SendMessage":
            raise ConnectA2AProtocolError("expected JSON-RPC SendMessage request")
        if "id" not in raw:
            raise ConnectA2AProtocolError("JSON-RPC request id is required")
        params = raw.get("params")
        if not isinstance(params, Mapping):
            raise ConnectA2AProtocolError("SendMessage params are required")
        message = params.get("message")
        if not isinstance(message, Mapping):
            raise ConnectA2AProtocolError("SendMessage message is required")
        context_id = message.get("contextId")
        if not isinstance(context_id, str) or not context_id.strip():
            raise ConnectA2AProtocolError("message contextId is required")
        parts = message.get("parts")
        if not isinstance(parts, list) or len(parts) != 1 or not isinstance(parts[0], Mapping):
            raise ConnectA2AProtocolError("message must contain exactly one part")
        part = parts[0]
        metadata = message.get("metadata")
        event_type = ""
        if isinstance(metadata, Mapping):
            value = metadata.get(EVENT_TYPE_KEY)
            if isinstance(value, str):
                event_type = value
        if not event_type and part.get("mediaType") == "audio/lpcm" and "raw" in part:
            event_type = "AUDIO_INPUT_CHUNK"
        if not event_type:
            raise ConnectA2AProtocolError("A2A extension eventType is required")
        return ConnectA2ARequest(
            request_id=raw["id"],
            context_id=context_id,
            event_type=event_type,
            message=message,
            part=part,
        )

    def parse_init(self, request: ConnectA2ARequest) -> ConnectA2AInit:
        if request.event_type != "INIT_SESSION":
            raise ConnectA2AProtocolError("first event must be INIT_SESSION")
        data = _part_data(request.part)
        init = data.get("initSession")
        if not isinstance(init, Mapping):
            raise ConnectA2AProtocolError("INIT_SESSION initSession payload is required")
        input_cfg = init.get("audioInputConfiguration")
        output_cfg = init.get("audioOutputConfiguration")
        if not isinstance(input_cfg, Mapping) or not isinstance(output_cfg, Mapping):
            raise ConnectA2AProtocolError("voice INIT_SESSION requires both audio configurations")
        finish_types = init.get("supportedFinishTypes") or []
        if not isinstance(finish_types, list) or not all(isinstance(x, str) for x in finish_types):
            raise ConnectA2AProtocolError("supportedFinishTypes must be a string list")
        contact_context = init.get("contactContext")
        caller_id = None
        if isinstance(contact_context, Mapping):
            customer = contact_context.get("customerEndpoint")
            if isinstance(customer, Mapping) and isinstance(customer.get("address"), str):
                caller_id = customer["address"]
        return ConnectA2AInit(
            request_id=request.request_id,
            context_id=request.context_id,
            contact_arn=_optional_str(init.get("contactArn")),
            caller_id=caller_id,
            supported_finish_types=frozenset(finish_types),
            subscribe_to_tracing_events=bool(init.get("subscribeToTracingEvents")),
            input_audio=ConnectAudioConfiguration.from_wire(input_cfg),
            output_audio=ConnectAudioConfiguration.from_wire(output_cfg),
        )

    def decode_audio(
        self,
        request: ConnectA2ARequest,
        fmt: AudioFormat,
        sequence: int,
        *,
        max_bytes: int,
    ) -> AudioChunk:
        if request.event_type != "AUDIO_INPUT_CHUNK" or request.part.get("mediaType") != "audio/lpcm":
            raise ConnectA2AProtocolError("expected an audio/lpcm input chunk")
        raw = request.part.get("raw")
        if not isinstance(raw, str):
            raise ConnectA2AProtocolError("audio input raw base64 payload is required")
        # Base64 expands source bytes by at most 4/3. Reject before allocating
        # an unbounded decoded payload from an authenticated network frame.
        if len(raw) > ((max_bytes + 2) // 3) * 4:
            raise ConnectA2AOverloadError("audio input frame exceeds byte budget")
        try:
            payload = base64.b64decode(raw, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ConnectA2AProtocolError("audio input is not valid base64") from exc
        if len(payload) > max_bytes:
            raise ConnectA2AOverloadError("audio input frame exceeds byte budget")
        return AudioChunk(data=payload, format=fmt, sequence=sequence, timestamp_s=time.monotonic())

    def init_response(
        self,
        init: ConnectA2AInit,
        *,
        input_audio: ConnectAudioConfiguration,
        output_audio: ConnectAudioConfiguration,
    ) -> dict[str, object]:
        response: dict[str, object] = {"statusCode": "200"}
        if input_audio != init.input_audio:
            response["audioInputConfiguration"] = input_audio.to_wire()
        if output_audio != init.output_audio:
            response["audioOutputConfiguration"] = output_audio.to_wire()
        return {
            "jsonrpc": "2.0",
            "id": init.request_id,
            "result": {
                "message": self._message(
                    context_id=init.context_id,
                    event_type="INIT_SESSION_RESPONSE",
                    data={"initSessionResponse": response},
                )
            },
        }

    def rpc_ack(self, request_id: str | int) -> dict[str, object]:
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}

    def artifact_audio(
        self,
        *,
        context_id: str,
        task_id: str,
        artifact_id: str,
        audio: bytes,
        append: bool,
        last_chunk: bool,
    ) -> dict[str, object]:
        return self._result(
            "artifactUpdate",
            {
                "taskId": task_id,
                "contextId": context_id,
                "artifact": {
                    "artifactId": artifact_id,
                    "parts": [
                        {
                            "raw": base64.b64encode(audio).decode("ascii"),
                            "mediaType": "audio/lpcm",
                        }
                    ],
                    "metadata": {EVENT_TYPE_KEY: "AUDIO_RESPONSE_CHUNK"},
                },
                "append": append,
                "lastChunk": last_chunk,
            },
        )

    def interruption(self, *, context_id: str, reason: str) -> dict[str, object]:
        return self._result(
            "message",
            self._message(
                context_id=context_id,
                event_type="INTERRUPTION",
                data={"interruption": {"type": "USER_AUDIO_INPUT", "reason": reason}},
            ),
        )

    def status_update(
        self,
        *,
        context_id: str,
        task_id: str,
        state: str,
        outcome: SessionOutcome | None = None,
        reason: str = "",
    ) -> dict[str, object]:
        status: dict[str, object] = {
            "state": state,
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
        }
        if outcome in (SessionOutcome.COMPLETE, SessionOutcome.ESCALATE):
            status["message"] = self._message(
                context_id=context_id,
                event_type="FINISH",
                data={"finish": {"type": outcome.value.upper(), "reason": reason}},
            )
        elif outcome is SessionOutcome.ERROR:
            status["message"] = self._message(
                context_id=context_id,
                event_type="ERROR",
                data={
                    "error": {
                        "errorCode": 500,
                        "errorReason": reason or "external agent failure",
                        "errorCategory": "INTERNAL_ERROR",
                    }
                },
            )
        return self._result(
            "statusUpdate",
            {"taskId": task_id, "contextId": context_id, "status": status},
        )

    def tracing_span(
        self,
        *,
        context_id: str,
        task_id: str,
        artifact_id: str,
        trace_context: ConnectTraceContext,
        outcome: SessionOutcome,
        spans: list[TraceSpan],
        monotonic_to_epoch_ns: Callable[[float], int],
        input_bytes: int,
        output_bytes: int,
        tool_events: list[SessionEvent],
    ) -> dict[str, object]:
        now_monotonic = time.monotonic()
        source_spans = spans or [
            TraceSpan(
                name="invoke_agent voice-agent-ac",
                trace_id=trace_context.trace_id,
                span_id="",
                started_at_s=now_monotonic,
                ended_at_s=now_monotonic,
                status=TraceStatus.ERROR if outcome is SessionOutcome.ERROR else TraceStatus.OK,
            )
        ]
        otlp_spans = []
        root_span_id = ""
        for index, span in enumerate(source_spans):
            span_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{task_id}:{index}:{span.name}").hex[:16]
            if index == 0:
                root_span_id = span_id
                parent_span_id = trace_context.parent_span_id or ""
                name = "invoke_agent voice-agent-ac"
            else:
                parent_span_id = root_span_id
                name = span.name
            attributes = [
                {"key": "gen_ai.agent.name", "value": {"stringValue": "voice-agent-ac"}},
                {"key": "gen_ai.operation.name", "value": {"stringValue": "invoke_agent" if index == 0 else span.name}},
                {"key": "aws.connect.session.outcome", "value": {"stringValue": outcome.value}},
            ]
            for key, value in span.attributes.items():
                attributes.append({"key": f"voice_agent.{key}", "value": {"stringValue": str(value)}})
            events: list[dict[str, object]] = []
            if index == 0:
                if input_bytes:
                    events.append(
                        _connect_message_event(
                            "input", "AUDIO_MESSAGE", "user", f"{input_bytes} PCM bytes",
                            monotonic_to_epoch_ns(span.started_at_s),
                        )
                    )
                if output_bytes:
                    events.append(
                        _connect_message_event(
                            "output", "AUDIO_MESSAGE", "assistant", f"{output_bytes} PCM bytes",
                            monotonic_to_epoch_ns(span.ended_at_s),
                        )
                    )
                for tool_event in tool_events:
                    tool_name = str(tool_event.payload.get("name") or tool_event.name)
                    events.append(
                        _connect_message_event(
                            "instruction", "TOOL_CALL", "assistant", tool_name,
                            monotonic_to_epoch_ns(tool_event.timestamp_s),
                        )
                    )
            start_ns = monotonic_to_epoch_ns(span.started_at_s)
            end_ns = max(start_ns, monotonic_to_epoch_ns(span.ended_at_s))
            otlp_spans.append(
                {
                    "traceId": trace_context.trace_id,
                    "spanId": span_id,
                    "parentSpanId": parent_span_id,
                    "name": name,
                    "kind": 2 if index == 0 else 3,
                    "startTimeUnixNano": str(start_ns),
                    "endTimeUnixNano": str(end_ns),
                    "attributes": attributes,
                    "events": events,
                    "status": {"code": 2 if span.status is TraceStatus.ERROR else 1},
                }
            )
        return self._result(
            "artifactUpdate",
            {
                "taskId": task_id,
                "contextId": context_id,
                "artifact": {
                    "artifactId": artifact_id,
                    "parts": [
                        {
                            "data": {
                                "tracingSpan": {
                                    "otlp": {
                                        "resourceSpans": [
                                            {
                                                "scopeSpans": [
                                                    {"scope": {"name": "voice_agent.connect_a2a"}, "spans": otlp_spans}
                                                ]
                                            }
                                        ]
                                    }
                                }
                            },
                            "mediaType": "application/json",
                        }
                    ],
                    "metadata": {EVENT_TYPE_KEY: "TRACING_SPAN"},
                },
                "append": False,
                "lastChunk": True,
            },
        )

    def _message(self, *, context_id: str, event_type: str, data: Mapping[str, object]) -> dict[str, object]:
        return {
            "messageId": self._id_factory(),
            "contextId": context_id,
            "role": "ROLE_AGENT",
            "parts": [{"data": data, "mediaType": "application/json"}],
            "extensions": [EXTENSION_URI],
            "metadata": {EVENT_TYPE_KEY: event_type},
        }

    def _result(self, key: str, payload: Mapping[str, object]) -> dict[str, object]:
        return {"jsonrpc": "2.0", "id": self._id_factory(), "result": {key: payload}}


class ConnectA2AQueueAudioSource:
    """Bounded queue-backed continuous audio source for persistent sessions."""

    def __init__(
        self,
        *,
        max_pending_chunks: int = 128,
        max_pending_bytes: int = 512 * 1024,
    ) -> None:
        if max_pending_chunks <= 0 or max_pending_bytes <= 0:
            raise ValueError("audio queue limits must be positive")
        self._queue: asyncio.Queue[AudioChunk | None] = asyncio.Queue(maxsize=max_pending_chunks)
        self._max_pending_bytes = max_pending_bytes
        self._pending_bytes = 0
        self._closed = False

    async def feed(self, chunk: AudioChunk) -> None:
        if self._closed:
            raise ConnectA2AConflictError("audio input is closed")
        if len(chunk.data) > self._max_pending_bytes:
            raise ConnectA2AOverloadError("audio input chunk exceeds byte budget")
        if self._pending_bytes + len(chunk.data) > self._max_pending_bytes:
            raise ConnectA2AOverloadError("audio input byte budget is exhausted")
        try:
            self._queue.put_nowait(chunk)
            self._pending_bytes += len(chunk.data)
        except asyncio.QueueFull as exc:
            raise ConnectA2AOverloadError("audio input queue is full") from exc

    async def next_chunk(self) -> AudioChunk | None:
        item = await self._queue.get()
        if item is not None:
            self._pending_bytes -= len(item.data)
        return item

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        while True:
            item = await self.next_chunk()
            if item is None:
                return
            yield item

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        # A terminal/disconnected record must not retain queued PCM indefinitely.
        while True:
            with contextlib.suppress(asyncio.QueueEmpty):
                item = self._queue.get_nowait()
                if item is not None:
                    self._pending_bytes -= len(item.data)
                continue
            break
        self._queue.put_nowait(None)


class ConnectA2AOutputSink:
    """One-active-artifact output sink with independently playable PCM chunks."""

    def __init__(
        self,
        *,
        codec: ConnectA2AWireCodec,
        context_id: str,
        audio_format: AudioFormat,
        send: SendEnvelope,
        id_factory: IdFactory,
        max_chunk_bytes: int,
    ) -> None:
        self._codec = codec
        self._context_id = context_id
        self._audio_format = audio_format
        self._send = send
        self._id_factory = id_factory
        if max_chunk_bytes <= 0:
            raise ValueError("max_chunk_bytes must be positive")
        self._max_chunk_bytes = max_chunk_bytes
        self._ready = False
        self._closed = False
        self._pending: AudioChunk | None = None
        self._task_id: str | None = None
        self._artifact_id: str | None = None
        self._sent_chunks = 0
        self._bytes_sent = 0
        self._generation = 0
        self._lock = asyncio.Lock()

    @property
    def task_id(self) -> str | None:
        return self._task_id

    @property
    def bytes_sent(self) -> int:
        return self._bytes_sent

    def set_ready(self) -> None:
        self._ready = True

    def rebind(self, send: SendEnvelope) -> None:
        self._send = send

    async def write(self, chunk: AudioChunk) -> None:
        if chunk.format != self._audio_format:
            raise ConnectA2AProtocolError("output audio does not match negotiated format")
        if len(chunk.data) > self._max_chunk_bytes:
            raise ConnectA2AOverloadError("output audio frame exceeds byte budget")
        async with self._lock:
            if self._closed:
                raise ConnectA2AConflictError("output sink is closed")
            if not self._ready:
                raise ConnectA2ANotReadyError("CHANNEL_STATE READY has not been received")
            if self._pending is not None:
                await self._emit(self._pending, last_chunk=False)
            self._pending = chunk

    async def interrupt(self, reason: str) -> str | None:
        """Discard pending output without lastChunk and return canceled task id."""
        async with self._lock:
            canceled = self._task_id
            self._generation += 1
            self._pending = None
            self._task_id = None
            self._artifact_id = None
            self._sent_chunks = 0
            return canceled

    async def abort(self) -> None:
        """Drop queued media after a transport loss without sending more frames."""
        async with self._lock:
            self._pending = None
            self._closed = True
            self._task_id = None
            self._artifact_id = None
            self._sent_chunks = 0

    async def aclose(self) -> None:
        async with self._lock:
            if self._closed:
                return
            if self._pending is not None:
                await self._emit(self._pending, last_chunk=True)
                self._pending = None
            self._closed = True

    def ensure_task(self) -> str:
        if self._task_id is None:
            self._task_id = f"task-{self._id_factory()}"
            self._artifact_id = f"artifact-{self._id_factory()}"
            self._sent_chunks = 0
        return self._task_id

    async def _emit(self, chunk: AudioChunk, *, last_chunk: bool) -> None:
        task_id = self.ensure_task()
        assert self._artifact_id is not None
        await self._send(
            self._codec.artifact_audio(
                context_id=self._context_id,
                task_id=task_id,
                artifact_id=self._artifact_id,
                audio=chunk.data,
                append=self._sent_chunks > 0,
                last_chunk=last_chunk,
            )
        )
        self._sent_chunks += 1
        self._bytes_sent += len(chunk.data)


@dataclass(slots=True)
class ConnectA2ASession:
    """A negotiated persistent voice session, independent of ASGI/FastAPI."""

    identity: SessionIdentity
    init: ConnectA2AInit
    input_format: AudioFormat
    output_format: AudioFormat
    trace_context: ConnectTraceContext
    source: ConnectA2AQueueAudioSource
    output: ConnectA2AOutputSink
    events: EventSink
    lifecycle: LifecycleSink
    traces: MemoryTraceSink
    codec: ConnectA2AWireCodec
    registry: "ConnectA2ASessionRegistry"
    send: SendEnvelope
    id_factory: IdFactory
    monotonic_epoch_offset_ns: int
    max_pending_audio_chunks: int
    max_pending_audio_bytes: int
    max_audio_chunk_bytes: int
    ready: bool = False
    terminal: bool = False
    transport_lost: bool = False
    inbound_sequence: int = 0
    input_bytes: int = 0
    trace_events: list[SessionEvent] = field(default_factory=list)
    generation: int = 0
    _engine_tasks: dict[int, asyncio.Task[EngineResult]] = field(default_factory=dict, init=False, repr=False)
    _state_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    @property
    def ports(self) -> SessionPorts:
        return SessionPorts(
            inbound_audio=self.source,
            outbound_audio=self.output,
            events=self.events,
            lifecycle=self.lifecycle,
            traces=self.traces,
        )

    async def handle_request(self, request: ConnectA2ARequest) -> dict[str, object]:
        async with self._state_lock:
            if self.terminal or self.transport_lost:
                raise ConnectA2AConflictError("session is not accepting input")
            self._assert_context(request.context_id)
            if request.event_type == "CHANNEL_STATE":
                state = _part_data(request.part).get("channelState")
                if not isinstance(state, Mapping) or state.get("state") != "READY":
                    raise ConnectA2AProtocolError("CHANNEL_STATE must be READY")
                self.ready = True
                self.output.set_ready()
                await self._emit_event(SessionEvent("a2a_channel_ready", time.monotonic()))
                return self.codec.rpc_ack(request.request_id)
            if request.event_type == "AUDIO_INPUT_CHUNK":
                if not self.ready:
                    raise ConnectA2ANotReadyError("audio received before CHANNEL_STATE READY")
                chunk = self.codec.decode_audio(
                    request,
                    self.input_format,
                    self.inbound_sequence,
                    max_bytes=self.max_audio_chunk_bytes,
                )
                self.inbound_sequence += 1
                await self.source.feed(chunk)
                self.input_bytes += len(chunk.data)
                await self._emit_event(
                    SessionEvent(
                        "a2a_audio_input",
                        time.monotonic(),
                        {"sequence": chunk.sequence, "bytes": len(chunk.data)},
                    )
                )
                return self.codec.rpc_ack(request.request_id)
            if request.event_type == "INTERRUPTION":
                data = _part_data(request.part).get("interruption")
                reason = "remote_interruption"
                if isinstance(data, Mapping) and isinstance(data.get("reason"), str):
                    reason = data["reason"]
                await self._flush_output_turn_locked(reason, notify_connect=False)
                return self.codec.rpc_ack(request.request_id)
            if request.event_type == "ERROR":
                await self._emit_event(SessionEvent("a2a_remote_error", time.monotonic()))
                return self.codec.rpc_ack(request.request_id)
            raise ConnectA2AProtocolError(f"unsupported A2A event {request.event_type}")

    async def signal_local_interruption(self, reason: str = "user_speech") -> None:
        """Flush Connect playback after local VAD without ending the engine."""
        await self.flush_output_turn(reason)

    async def flush_output_turn(
        self,
        reason: str = "user_speech",
        *,
        notify_connect: bool = True,
    ) -> None:
        """Flush one active response without cancelling the persistent engine."""
        async with self._state_lock:
            if self.terminal or self.transport_lost:
                return
            await self._flush_output_turn_locked(reason, notify_connect=notify_connect)

    async def _interrupt(self, reason: str, *, notify_connect: bool) -> None:
        old_generation = self.generation
        self.generation += 1
        task = self._engine_tasks.get(old_generation)
        if task and task is not asyncio.current_task() and not task.done():
            task.cancel()
        await self._flush_output_turn_locked(reason, notify_connect=notify_connect)

    async def _flush_output_turn_locked(self, reason: str, *, notify_connect: bool) -> None:
        task_id = await self.output.interrupt(reason)
        # TASK_STATE_CANCELED ends collaboration in Connect. A normal completed
        # status closes only the interrupted turn, allowing a fresh artifact on
        # the next user turn while intentionally omitting lastChunk.
        if task_id:
            await self.send(
                self.codec.status_update(
                    context_id=self.identity.logical_session_id,
                    task_id=task_id,
                    state="TASK_STATE_COMPLETED",
                    outcome=None,
                    reason=reason,
                )
            )
        if notify_connect:
            await self.send(self.codec.interruption(context_id=self.identity.logical_session_id, reason=reason))
        await self._emit_event(SessionEvent("a2a_interruption", time.monotonic(), {"reason": reason}))

    async def run_engine(self, engine: SessionEngine, *, runner: SessionRunner | None = None) -> EngineResult:
        if not self.ready:
            raise ConnectA2ANotReadyError("CHANNEL_STATE READY has not been received")
        async with self._state_lock:
            if self.terminal or self.transport_lost:
                raise ConnectA2AConflictError("session is not available for an engine")
            generation = self.generation
            source = self.source
            output = self.output
            task = asyncio.current_task()
            if task is not None:
                self._engine_tasks[generation] = task
            ports = SessionPorts(
                inbound_audio=source,
                outbound_audio=_GenerationBoundOutputSink(self, generation, output),
                events=self.events,
                lifecycle=self.lifecycle,
                traces=self.traces,
            )
        try:
            result = await (runner or SessionRunner()).run(
                identity=self.identity,
                ports=ports,
                engine=engine,
            )
        except asyncio.CancelledError:
            if self._owns_generation(generation, output):
                await self.finish(SessionOutcome.ERROR, "engine task canceled")
            raise
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"[:500]
            if self._owns_generation(generation, output):
                await self.finish(SessionOutcome.ERROR, reason)
            return EngineResult(SessionOutcome.ERROR, reason)
        finally:
            task = asyncio.current_task()
            if task is not None and self._engine_tasks.get(generation) is task:
                self._engine_tasks.pop(generation, None)
        if not self._owns_generation(generation, output):
            return EngineResult(SessionOutcome.ERROR, "transport disconnected or response superseded")
        await self.finish(result.outcome, result.reason)
        return EngineResult(result.outcome, result.reason)

    async def finish(self, outcome: SessionOutcome, reason: str = "") -> None:
        async with self._state_lock:
            if self.terminal or self.transport_lost:
                return
            task_id = self.output.ensure_task()
            # The runner normally closes both ports first. Finish can also be
            # called directly, so drain input and flush a pending final output
            # before trace/status emission to avoid retaining PCM in a terminal record.
            await self.source.aclose()
            await self.output.aclose()
            if outcome in (SessionOutcome.COMPLETE, SessionOutcome.ESCALATE):
                required = outcome.value.upper()
                if required not in self.init.supported_finish_types:
                    outcome = SessionOutcome.ERROR
                    reason = f"unsupported finish type {required}"
                    status = "TASK_STATE_FAILED"
                else:
                    status = "TASK_STATE_COMPLETED"
            else:
                status = "TASK_STATE_FAILED"
            if self.init.subscribe_to_tracing_events:
                await self.send(
                    self.codec.tracing_span(
                        context_id=self.identity.logical_session_id,
                        task_id=task_id,
                        artifact_id=f"trace-{self.id_factory()}",
                        trace_context=self.trace_context,
                        outcome=outcome,
                        spans=list(self.traces.spans),
                        monotonic_to_epoch_ns=self._monotonic_to_epoch_ns,
                        input_bytes=self.input_bytes,
                        output_bytes=self.output.bytes_sent,
                        tool_events=[event for event in self.trace_events if "tool" in event.name],
                    )
                )
            await self.send(
                self.codec.status_update(
                    context_id=self.identity.logical_session_id,
                    task_id=task_id,
                    state=status,
                    outcome=outcome,
                    reason=reason,
                )
            )
            self.terminal = True
            logical_session_id = self.identity.logical_session_id
        # Never acquire registry state while holding the session lock. Registry
        # claim/rebind is ordered registry -> session, preventing lock inversion.
        await self.registry.release(logical_session_id, terminal=True)

    async def disconnect(self, *, unexpected: bool) -> None:
        await self.registry.disconnect(self, unexpected=unexpected)

    async def _cleanup_for_disconnect(self, *, terminal: bool) -> None:
        """Cancel local work and close ports; called by registry under its lock."""
        async with self._state_lock:
            if self.terminal:
                return
            self.transport_lost = True
            self.generation += 1
            for task in list(self._engine_tasks.values()):
                if task is not asyncio.current_task() and not task.done():
                    task.cancel()
            await self.source.aclose()
            await self.output.abort()
            self.terminal = terminal

    async def rebind(self, *, identity: SessionIdentity, send: SendEnvelope) -> None:
        async with self._state_lock:
            if self.terminal:
                raise ConnectA2AConflictError("cannot rebind a terminal session")
            self.identity = identity
            self.send = send
            self.generation += 1
            self.source = ConnectA2AQueueAudioSource(
                max_pending_chunks=self.max_pending_audio_chunks,
                max_pending_bytes=self.max_pending_audio_bytes,
            )
            self.output = ConnectA2AOutputSink(
                codec=self.codec,
                context_id=identity.logical_session_id,
                audio_format=self.output_format,
                send=send,
                id_factory=self.id_factory,
                max_chunk_bytes=self.max_audio_chunk_bytes,
            )
            self.events = MemoryEventSink()
            self.lifecycle = MemoryLifecycleSink()
            self.traces = MemoryTraceSink()
            self.ready = False
            self.transport_lost = False
            self.inbound_sequence = 0
            self.input_bytes = 0
            self.trace_events.clear()

    def _owns_generation(self, generation: int, output: ConnectA2AOutputSink) -> bool:
        return (
            not self.terminal
            and not self.transport_lost
            and generation == self.generation
            and output is self.output
        )

    async def _emit_event(self, event: SessionEvent) -> None:
        self.trace_events.append(event)
        await self.events.emit(event)

    def _monotonic_to_epoch_ns(self, value: float) -> int:
        return int(value * 1_000_000_000) + self.monotonic_epoch_offset_ns

    def _assert_context(self, context_id: str) -> None:
        if context_id != self.identity.logical_session_id:
            raise ConnectA2AConflictError("contextId does not match active session")


@dataclass(slots=True)
class _RegistryRecord:
    session: ConnectA2ASession
    active: bool = True
    terminal: bool = False
    reconnects: int = 0


class ConnectA2ASessionRegistry:
    """Bounded registry enforcing Connect's one-reconnect voice-session rule."""

    def __init__(self, *, max_terminal_records: int = 1024) -> None:
        if max_terminal_records <= 0:
            raise ValueError("max_terminal_records must be positive")
        self._lock = asyncio.Lock()
        self._records: dict[str, _RegistryRecord] = {}
        self._max_terminal_records = max_terminal_records

    async def claim(
        self,
        *,
        identity: SessionIdentity,
        factory: Callable[[], ConnectA2ASession],
        send: SendEnvelope,
    ) -> ConnectA2ASession:
        async with self._lock:
            record = self._records.get(identity.logical_session_id)
            if record is None:
                session = factory()
                self._records[identity.logical_session_id] = _RegistryRecord(session=session)
                return session
            if record.terminal:
                raise ConnectA2AConflictError("cannot reconnect a terminal session")
            if record.active:
                raise ConnectA2AConflictError("session already has an active connection")
            if record.reconnects >= 1:
                raise ConnectA2AConflictError("only one reconnect is permitted")
            old = record.session.identity
            if old.connect_context_id != identity.connect_context_id or old.contact_id != identity.contact_id:
                raise ConnectA2AConflictError("reconnect identity does not match original session")
            record.active = True
            record.reconnects += 1
            await record.session.rebind(identity=identity, send=send)
            return record.session

    async def release(self, logical_session_id: str, *, terminal: bool) -> bool:
        async with self._lock:
            record = self._records.get(logical_session_id)
            if record is None:
                return True
            record.active = False
            record.terminal = record.terminal or terminal
            self._trim_terminal_records_locked()
            return record.terminal

    async def disconnect(self, session: ConnectA2ASession, *, unexpected: bool) -> bool:
        """Atomically record loss and clean session resources (registry -> session)."""
        async with self._lock:
            logical_session_id = session.identity.logical_session_id
            record = self._records.get(logical_session_id)
            if record is None:
                await session._cleanup_for_disconnect(terminal=True)
                return True
            if record.session is not session:
                return record.terminal
            if record.terminal:
                return True
            record.active = False
            # Connect allows one unexpected voice reconnect. The second loss is
            # final even before a third connection attempt reaches this process.
            record.terminal = record.terminal or not unexpected or record.reconnects >= 1
            await session._cleanup_for_disconnect(terminal=record.terminal)
            self._trim_terminal_records_locked()
            return record.terminal

    def _trim_terminal_records_locked(self) -> None:
        terminal_ids = [key for key, record in self._records.items() if record.terminal]
        overflow = len(terminal_ids) - self._max_terminal_records
        for logical_session_id in terminal_ids[:max(0, overflow)]:
            self._records.pop(logical_session_id, None)

    async def purge_terminal(self, logical_session_id: str) -> None:
        """Forget a terminal record after the host's contact-retention window."""
        async with self._lock:
            record = self._records.get(logical_session_id)
            if record and record.terminal:
                self._records.pop(logical_session_id, None)


class ConnectA2AAdapter:
    """Factory/controller for authenticated Connect A2A voice sessions.

    The ASGI WebSocket bridge is intentionally deferred. This object is the
    tested semantic core that bridge will call after upgrade authentication.
    """

    def __init__(
        self,
        *,
        api_key: SecretValue,
        readiness: ReadinessState,
        codec: ConnectA2AWireCodec | None = None,
        registry: ConnectA2ASessionRegistry | None = None,
        max_pending_audio_chunks: int = 128,
        max_pending_audio_bytes: int = 512 * 1024,
        max_audio_chunk_bytes: int = 32 * 1024,
        id_factory: IdFactory = lambda: str(uuid.uuid4()),
    ) -> None:
        self._api_key = api_key
        self._readiness = readiness
        if (
            max_pending_audio_chunks <= 0
            or max_pending_audio_bytes <= 0
            or max_audio_chunk_bytes <= 0
            or max_audio_chunk_bytes > max_pending_audio_bytes
        ):
            raise ValueError("invalid A2A audio queue/frame limits")
        self._max_pending_audio_chunks = max_pending_audio_chunks
        self._max_pending_audio_bytes = max_pending_audio_bytes
        self._max_audio_chunk_bytes = max_audio_chunk_bytes
        self._id_factory = id_factory
        self._codec = codec or ConnectA2AWireCodec(id_factory=id_factory)
        self._registry = registry or ConnectA2ASessionRegistry()

    @property
    def extension_response_headers(self) -> dict[str, str]:
        return {EXTENSION_HEADER: EXTENSION_URI}

    async def initialize(
        self,
        *,
        headers: Mapping[str, str],
        raw_request: str | bytes | Mapping[str, object],
        send: SendEnvelope,
    ) -> tuple[ConnectA2ASession, dict[str, object]]:
        if not validate_bearer_authorization(_header(headers, "authorization"), self._api_key):
            raise ConnectA2AAuthenticationError("invalid bearer token")
        extensions = _header(headers, EXTENSION_HEADER)
        if not extensions or EXTENSION_URI not in {value.strip() for value in extensions.split(",")}:
            raise ConnectA2AProtocolError("required A2A extension was not requested")
        if not self._readiness.snapshot()["ok"]:
            raise ConnectA2ANotReadyError("service is not ready")
        request = self._codec.parse_request(raw_request)
        init = self._codec.parse_init(request)
        input_audio = _negotiate(init.input_audio, preferred_rate=16000)
        output_audio = _negotiate(init.output_audio, preferred_rate=24000)
        attempt_id = f"attempt-{self._id_factory()}"
        identity = SessionIdentity.for_connect(
            attempt_id=attempt_id,
            context_id=init.context_id,
            contact_id=init.contact_arn,
            caller_id=init.caller_id,
            transport_session_id=_optional_str(_header(headers, "sec-websocket-key")),
        )
        trace_context = ConnectTraceContext.from_headers(headers, id_factory=self._id_factory)

        def factory() -> ConnectA2ASession:
            source = ConnectA2AQueueAudioSource(
                max_pending_chunks=self._max_pending_audio_chunks,
                max_pending_bytes=self._max_pending_audio_bytes,
            )
            output = ConnectA2AOutputSink(
                codec=self._codec,
                context_id=identity.logical_session_id,
                audio_format=output_audio.to_audio_format(),
                send=send,
                id_factory=self._id_factory,
                max_chunk_bytes=self._max_audio_chunk_bytes,
            )
            return ConnectA2ASession(
                identity=identity,
                init=init,
                input_format=input_audio.to_audio_format(),
                output_format=output_audio.to_audio_format(),
                trace_context=trace_context,
                source=source,
                output=output,
                events=MemoryEventSink(),
                lifecycle=MemoryLifecycleSink(),
                traces=MemoryTraceSink(),
                codec=self._codec,
                registry=self._registry,
                send=send,
                id_factory=self._id_factory,
                monotonic_epoch_offset_ns=time.time_ns() - int(time.monotonic() * 1_000_000_000),
                max_pending_audio_chunks=self._max_pending_audio_chunks,
                max_pending_audio_bytes=self._max_pending_audio_bytes,
                max_audio_chunk_bytes=self._max_audio_chunk_bytes,
            )

        session = await self._registry.claim(identity=identity, factory=factory, send=send)
        response = self._codec.init_response(
            init, input_audio=input_audio, output_audio=output_audio
        )
        return session, response


def validate_bearer_authorization(header: str | None, expected: SecretValue) -> bool:
    """Strict, constant-time validation of ``Authorization: Bearer <apiKey>``."""
    if not header or not expected:
        return False
    scheme, separator, token = header.partition(" ")
    if scheme != "Bearer" or not separator or not token or token.strip() != token:
        return False
    return hmac.compare_digest(token.encode("utf-8"), expected.reveal().encode("utf-8"))


def _negotiate(proposed: ConnectAudioConfiguration, *, preferred_rate: int) -> ConnectAudioConfiguration:
    """Use the architecture-preferred rate while retaining documented PCM shape."""
    return ConnectAudioConfiguration(sample_rate_hz=preferred_rate)


def _header(headers: Mapping[str, str], name: str) -> str | None:
    lower_name = name.lower()
    for key, value in headers.items():
        if key.lower() == lower_name:
            return value
    return None


def _part_data(part: Mapping[str, object]) -> Mapping[str, object]:
    data = part.get("data")
    if not isinstance(data, Mapping):
        raise ConnectA2AProtocolError("event data part is required")
    return data


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _connect_parent_span_id(tracestate: str | None) -> str | None:
    """Extract Connect's propagated parent span from W3C tracestate."""
    if not tracestate:
        return None
    for member in tracestate.split(","):
        key, separator, value = member.strip().partition("=")
        if key != "connect" or not separator:
            continue
        prefix = "parentSpanId:"
        if not value.startswith(prefix):
            continue
        normalized = value[len(prefix):].replace("-", "")
        if all(char in "0123456789abcdefABCDEF" for char in normalized):
            # OTLP span IDs are 16 hex chars. Connect may propagate a UUID-like
            # identifier, so retain its lower 64 bits as the parent span ID.
            if len(normalized) >= 16:
                return normalized[-16:].lower()
    return None


def _connect_message_event(
    role: str,
    message_type: str,
    participant: str,
    content: str,
    timestamp_ns: int,
) -> dict[str, object]:
    return {
        "name": "aws.connect.span.message",
        "timeUnixNano": str(timestamp_ns),
        "attributes": [
            {"key": "aws.connect.span.message.role", "value": {"stringValue": role}},
            {"key": "aws.connect.span.message.type", "value": {"stringValue": message_type}},
            {"key": "aws.connect.span.message.participant", "value": {"stringValue": participant}},
            {"key": "aws.connect.span.message.content", "value": {"stringValue": content}},
        ],
    }


# The generation-bound sink is defined after the adapter helpers so it can stay
# private to this module. A running engine captures one instance per attempt.
class _GenerationBoundOutputSink:
    """Reject stale producer writes after interruption, disconnect, or rebind."""

    def __init__(
        self,
        session: ConnectA2ASession,
        generation: int,
        output: ConnectA2AOutputSink,
    ) -> None:
        self._session = session
        self._generation = generation
        self._output = output

    def _assert_current(self) -> None:
        if not self._session._owns_generation(self._generation, self._output):
            raise ConnectA2AConflictError("stale response generation")

    async def write(self, chunk: AudioChunk) -> None:
        async with self._session._state_lock:
            self._assert_current()
            await self._output.write(chunk)

    async def interrupt(self, reason: str) -> None:
        async with self._session._state_lock:
            if self._session._owns_generation(self._generation, self._output):
                await self._session._flush_output_turn_locked(reason, notify_connect=True)

    async def aclose(self) -> None:
        async with self._session._state_lock:
            if self._session._owns_generation(self._generation, self._output):
                await self._output.aclose()
            else:
                await self._output.abort()
