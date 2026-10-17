"""Hermetic contract tests for the Amazon Connect external A2A audio adapter."""

from __future__ import annotations

import asyncio
import base64
import uuid
from collections.abc import Mapping

import pytest

from voice_agent.app import ReadinessState, SecretValue
from voice_agent.session.contracts import (
    AudioChunk,
    AudioFormat,
    EngineResult,
    SessionContext,
    SessionOutcome,
    TraceSpan,
    TraceStatus,
)
from voice_agent.transports.connect_a2a import (
    EXTENSION_HEADER,
    EXTENSION_URI,
    EVENT_TYPE_KEY,
    ConnectA2AAdapter,
    ConnectA2AAuthenticationError,
    ConnectA2AConflictError,
    ConnectA2AOverloadError,
    ConnectA2AProtocolError,
    ConnectA2AWireCodec,
    _GenerationBoundOutputSink,
)


class _Ids:
    def __init__(self) -> None:
        self._value = 0

    def __call__(self) -> str:
        self._value += 1
        return str(uuid.UUID(int=self._value))


def _ready_state(*, ok: bool = True) -> ReadinessState:
    state = ReadinessState("test-a2a")
    state.begin_startup()
    state.set_check("core", ok=ok, critical=True)
    state.mark_started()
    return state


def _headers(
    *, token: str = "api-key", traceparent: bool = False, tracestate: str | None = None
) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {token}",
        EXTENSION_HEADER: EXTENSION_URI,
    }
    if traceparent:
        headers["traceparent"] = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
    if tracestate:
        headers["tracestate"] = tracestate
    return headers


def _request(
    *,
    request_id: str,
    context_id: str,
    event_type: str | None,
    part: Mapping[str, object],
) -> dict[str, object]:
    metadata = {EVENT_TYPE_KEY: event_type} if event_type else {}
    return {
        "jsonrpc": "2.0",
        "method": "SendMessage",
        "id": request_id,
        "params": {
            "message": {
                "messageId": f"message-{request_id}",
                "contextId": context_id,
                "role": "ROLE_USER",
                "parts": [dict(part)],
                "metadata": metadata,
            }
        },
    }


def _init_request(
    *,
    context_id: str = "context-1",
    input_rate: int = 8000,
    output_rate: int = 16000,
    finish_types: list[str] | None = None,
) -> dict[str, object]:
    return _request(
        request_id="init-1",
        context_id=context_id,
        event_type="INIT_SESSION",
        part={
            "data": {
                "initSession": {
                    "instanceArn": "arn:aws:connect:us-east-1:111111111111:instance/abc",
                    "contactArn": "arn:aws:connect:us-east-1:111111111111:instance/abc/contact/contact-1",
                    "supportedFinishTypes": finish_types or ["COMPLETE", "ESCALATE"],
                    "subscribeToTracingEvents": True,
                    "audioInputConfiguration": {
                        "encoding": "LINEAR_PCM",
                        "sampleRateHertz": input_rate,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                    },
                    "audioOutputConfiguration": {
                        "encoding": "LINEAR_PCM",
                        "sampleRateHertz": output_rate,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                    },
                }
            },
            "mediaType": "application/json",
        },
    )


class _Sender:
    def __init__(self) -> None:
        self.frames: list[dict[str, object]] = []

    async def send(self, frame: dict[str, object]) -> None:
        self.frames.append(frame)


class _OutputEngine:
    async def run(self, context: SessionContext) -> EngineResult:
        fmt = AudioFormat(sample_rate_hz=24000)
        await context.ports.outbound_audio.write(AudioChunk(b"\x01\x00", fmt, 0))
        await context.ports.outbound_audio.write(AudioChunk(b"\x02\x00", fmt, 1))
        return EngineResult(SessionOutcome.ESCALATE, "caller requested an agent")


@pytest.mark.asyncio
async def test_init_response_negotiates_architecture_rates_and_requires_bearer():
    ids = _Ids()
    adapter = ConnectA2AAdapter(
        api_key=SecretValue("api-key"), readiness=_ready_state(), id_factory=ids
    )
    sender = _Sender()

    with pytest.raises(ConnectA2AAuthenticationError):
        await adapter.initialize(headers=_headers(token="wrong"), raw_request=_init_request(), send=sender.send)

    session, response = await adapter.initialize(
        headers=_headers(traceparent=True), raw_request=_init_request(), send=sender.send
    )

    payload = response["result"]["message"]
    init_response = payload["parts"][0]["data"]["initSessionResponse"]
    assert init_response["audioInputConfiguration"]["sampleRateHertz"] == 16000
    assert init_response["audioOutputConfiguration"]["sampleRateHertz"] == 24000
    assert session.identity.logical_session_id == "context-1"
    assert session.identity.contact_id.endswith("contact-1")
    assert session.trace_context.trace_id == "0123456789abcdef0123456789abcdef"
    assert adapter.extension_response_headers == {EXTENSION_HEADER: EXTENSION_URI}


@pytest.mark.asyncio
async def test_ready_audio_and_artifact_output_follow_a2a_contract():
    ids = _Ids()
    adapter = ConnectA2AAdapter(api_key=SecretValue("api-key"), readiness=_ready_state(), id_factory=ids)
    sender = _Sender()
    session, _ = await adapter.initialize(headers=_headers(), raw_request=_init_request(), send=sender.send)

    ready = _request(
        request_id="ready-1",
        context_id="context-1",
        event_type="CHANNEL_STATE",
        part={"data": {"channelState": {"state": "READY"}}, "mediaType": "application/json"},
    )
    assert await session.handle_request(adapter._codec.parse_request(ready)) == {
        "jsonrpc": "2.0", "id": "ready-1", "result": {}
    }

    pcm = b"\x01\x00\x02\x00"
    audio = _request(
        request_id="audio-1",
        context_id="context-1",
        event_type=None,
        part={"raw": base64.b64encode(pcm).decode("ascii"), "mediaType": "audio/lpcm"},
    )
    await session.handle_request(adapter._codec.parse_request(audio))
    inbound = await session.source.next_chunk()
    assert inbound is not None
    assert inbound.data == pcm
    assert inbound.format == AudioFormat(sample_rate_hz=16000)
    assert inbound.sequence == 0

    output_format = AudioFormat(sample_rate_hz=24000)
    await session.output.write(AudioChunk(b"\x03\x00", output_format, 0))
    await session.output.write(AudioChunk(b"\x04\x00", output_format, 1))
    await session.output.aclose()

    artifacts = [frame["result"]["artifactUpdate"] for frame in sender.frames]
    assert len(artifacts) == 2
    assert artifacts[0]["append"] is False and artifacts[0]["lastChunk"] is False
    assert artifacts[1]["append"] is True and artifacts[1]["lastChunk"] is True
    assert artifacts[0]["artifact"]["artifactId"] == artifacts[1]["artifact"]["artifactId"]
    assert base64.b64decode(artifacts[0]["artifact"]["parts"][0]["raw"]) == b"\x03\x00"


@pytest.mark.asyncio
async def test_engine_completion_emits_trace_then_escalate_finish():
    ids = _Ids()
    adapter = ConnectA2AAdapter(api_key=SecretValue("api-key"), readiness=_ready_state(), id_factory=ids)
    sender = _Sender()
    session, _ = await adapter.initialize(headers=_headers(), raw_request=_init_request(), send=sender.send)
    ready = _request(
        request_id="ready-1",
        context_id="context-1",
        event_type="CHANNEL_STATE",
        part={"data": {"channelState": {"state": "READY"}}, "mediaType": "application/json"},
    )
    await session.handle_request(adapter._codec.parse_request(ready))

    result = await session.run_engine(_OutputEngine())

    assert result.outcome is SessionOutcome.ESCALATE
    result_keys = [next(iter(frame["result"])) for frame in sender.frames]
    assert result_keys[-2:] == ["artifactUpdate", "statusUpdate"]
    trace = sender.frames[-2]["result"]["artifactUpdate"]
    assert trace["artifact"]["metadata"][EVENT_TYPE_KEY] == "TRACING_SPAN"
    finish = sender.frames[-1]["result"]["statusUpdate"]["status"]["message"]
    assert finish["metadata"][EVENT_TYPE_KEY] == "FINISH"
    assert finish["parts"][0]["data"]["finish"]["type"] == "ESCALATE"


class _ErrorEngine:
    async def run(self, context: SessionContext) -> EngineResult:
        return EngineResult(SessionOutcome.ERROR, "model dependency unavailable")


@pytest.mark.asyncio
async def test_failed_engine_emits_terminal_a2a_error_status():
    ids = _Ids()
    adapter = ConnectA2AAdapter(api_key=SecretValue("api-key"), readiness=_ready_state(), id_factory=ids)
    sender = _Sender()
    session, _ = await adapter.initialize(headers=_headers(), raw_request=_init_request(), send=sender.send)
    ready = _request(
        request_id="ready-1",
        context_id="context-1",
        event_type="CHANNEL_STATE",
        part={"data": {"channelState": {"state": "READY"}}, "mediaType": "application/json"},
    )
    await session.handle_request(adapter._codec.parse_request(ready))

    result = await session.run_engine(_ErrorEngine())

    assert result.outcome is SessionOutcome.ERROR
    status = sender.frames[-1]["result"]["statusUpdate"]
    assert status["status"]["state"] == "TASK_STATE_FAILED"
    error_message = status["status"]["message"]
    assert error_message["metadata"][EVENT_TYPE_KEY] == "ERROR"
    assert error_message["parts"][0]["data"]["error"]["errorReason"] == "model dependency unavailable"


@pytest.mark.asyncio
async def test_interruption_closes_turn_without_last_chunk_and_notifies_connect():
    ids = _Ids()
    adapter = ConnectA2AAdapter(api_key=SecretValue("api-key"), readiness=_ready_state(), id_factory=ids)
    sender = _Sender()
    session, _ = await adapter.initialize(headers=_headers(), raw_request=_init_request(), send=sender.send)
    ready = _request(
        request_id="ready-1",
        context_id="context-1",
        event_type="CHANNEL_STATE",
        part={"data": {"channelState": {"state": "READY"}}, "mediaType": "application/json"},
    )
    await session.handle_request(adapter._codec.parse_request(ready))
    fmt = AudioFormat(sample_rate_hz=24000)
    await session.output.write(AudioChunk(b"\x01\x00", fmt, 0))
    await session.output.write(AudioChunk(b"\x02\x00", fmt, 1))

    await session.signal_local_interruption("barge_in")

    assert sender.frames[0]["result"]["artifactUpdate"]["lastChunk"] is False
    assert sender.frames[1]["result"]["statusUpdate"]["status"]["state"] == "TASK_STATE_COMPLETED"
    interruption = sender.frames[2]["result"]["message"]
    assert interruption["metadata"][EVENT_TYPE_KEY] == "INTERRUPTION"
    assert interruption["parts"][0]["data"]["interruption"]["reason"] == "barge_in"

    # The next turn opens a distinct task/artifact after the interrupted turn
    # closes normally; it must not reuse or continue the old artifact.
    await session.output.write(AudioChunk(b"\x03\x00", fmt, 2))
    await session.output.aclose()
    next_artifact = sender.frames[3]["result"]["artifactUpdate"]
    assert next_artifact["append"] is False and next_artifact["lastChunk"] is True
    assert next_artifact["taskId"] != sender.frames[0]["result"]["artifactUpdate"]["taskId"]


@pytest.mark.asyncio
async def test_terminal_session_cannot_reopen_and_unsupported_finish_fails_task():
    ids = _Ids()
    adapter = ConnectA2AAdapter(api_key=SecretValue("api-key"), readiness=_ready_state(), id_factory=ids)
    sender = _Sender()
    session, _ = await adapter.initialize(
        headers=_headers(), raw_request=_init_request(finish_types=["COMPLETE"]), send=sender.send
    )

    await session.finish(SessionOutcome.ESCALATE, "human requested")

    status = sender.frames[-1]["result"]["statusUpdate"]
    assert status["status"]["state"] == "TASK_STATE_FAILED"
    assert status["status"]["message"]["metadata"][EVENT_TYPE_KEY] == "ERROR"
    with pytest.raises(ConnectA2AConflictError):
        await adapter.initialize(
            headers=_headers(), raw_request=_init_request(finish_types=["COMPLETE"]), send=_Sender().send
        )


@pytest.mark.asyncio
async def test_registry_allows_exactly_one_reconnect_for_same_context():
    ids = _Ids()
    adapter = ConnectA2AAdapter(api_key=SecretValue("api-key"), readiness=_ready_state(), id_factory=ids)
    first_sender = _Sender()
    session, _ = await adapter.initialize(headers=_headers(), raw_request=_init_request(), send=first_sender.send)
    first_attempt = session.identity.attempt_id

    await session.disconnect(unexpected=True)
    second_sender = _Sender()
    rebound, _ = await adapter.initialize(headers=_headers(), raw_request=_init_request(), send=second_sender.send)
    assert rebound is session
    assert rebound.identity.logical_session_id == "context-1"
    assert rebound.identity.attempt_id != first_attempt

    await rebound.disconnect(unexpected=True)
    with pytest.raises(ConnectA2AConflictError):
        await adapter.initialize(headers=_headers(), raw_request=_init_request(), send=_Sender().send)


@pytest.mark.asyncio
async def test_rejects_not_ready_and_invalid_audio_configuration():
    ids = _Ids()
    adapter = ConnectA2AAdapter(api_key=SecretValue("api-key"), readiness=_ready_state(ok=False), id_factory=ids)
    with pytest.raises(Exception) as not_ready:
        await adapter.initialize(headers=_headers(), raw_request=_init_request(), send=_Sender().send)
    assert "not ready" in str(not_ready.value)

    adapter = ConnectA2AAdapter(api_key=SecretValue("api-key"), readiness=_ready_state(), id_factory=ids)
    with pytest.raises(ConnectA2AProtocolError):
        await adapter.initialize(
            headers=_headers(), raw_request=_init_request(input_rate=44100), send=_Sender().send
        )


def test_bearer_header_shape_is_strict_and_constant_time_helper_is_safe():
    from voice_agent.transports.connect_a2a import validate_bearer_authorization

    expected = SecretValue("api-key")
    assert validate_bearer_authorization("Bearer api-key", expected)
    assert not validate_bearer_authorization("bearer api-key", expected)
    assert not validate_bearer_authorization("Bearer  api-key", expected)
    assert not validate_bearer_authorization("Basic api-key", expected)
    assert not validate_bearer_authorization(None, expected)


@pytest.mark.asyncio
async def test_trace_uses_connect_tracestate_parent_and_preserves_span_duration():
    ids = _Ids()
    adapter = ConnectA2AAdapter(api_key=SecretValue("api-key"), readiness=_ready_state(), id_factory=ids)
    sender = _Sender()
    session, _ = await adapter.initialize(
        headers=_headers(
            traceparent=True,
            tracestate="connect=parentSpanId:1111111111111111",
        ),
        raw_request=_init_request(),
        send=sender.send,
    )
    await session.traces.emit(
        TraceSpan(
            name="voice_agent.session",
            trace_id="ignored",
            span_id="ignored",
            started_at_s=10.0,
            ended_at_s=11.5,
            status=TraceStatus.OK,
        )
    )
    await session.finish(SessionOutcome.COMPLETE, "done")

    trace = sender.frames[-2]["result"]["artifactUpdate"]
    span = trace["artifact"]["parts"][0]["data"]["tracingSpan"]["otlp"]["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert span["traceId"] == "0123456789abcdef0123456789abcdef"
    assert span["parentSpanId"] == "1111111111111111"
    assert int(span["endTimeUnixNano"]) - int(span["startTimeUnixNano"]) == 1_500_000_000


@pytest.mark.asyncio
async def test_bounded_input_and_disconnect_cleanup_prevent_stale_reuse():
    ids = _Ids()
    adapter = ConnectA2AAdapter(
        api_key=SecretValue("api-key"),
        readiness=_ready_state(),
        max_pending_audio_chunks=1,
        id_factory=ids,
    )
    sender = _Sender()
    session, _ = await adapter.initialize(headers=_headers(), raw_request=_init_request(), send=sender.send)
    ready = _request(
        request_id="ready-1",
        context_id="context-1",
        event_type="CHANNEL_STATE",
        part={"data": {"channelState": {"state": "READY"}}, "mediaType": "application/json"},
    )
    await session.handle_request(adapter._codec.parse_request(ready))
    audio = _request(
        request_id="audio-1",
        context_id="context-1",
        event_type=None,
        part={"raw": base64.b64encode(b"\x01\x00").decode("ascii"), "mediaType": "audio/lpcm"},
    )
    await session.handle_request(adapter._codec.parse_request(audio))
    with pytest.raises(ConnectA2AOverloadError):
        await session.handle_request(adapter._codec.parse_request(audio))

    output_format = AudioFormat(sample_rate_hz=24000)
    await session.output.write(AudioChunk(b"\x02\x00", output_format, 0))
    await session.disconnect(unexpected=True)
    with pytest.raises(ConnectA2AConflictError):
        await session.source.feed(AudioChunk(b"\x03\x00", session.input_format, 1))
    with pytest.raises(ConnectA2AConflictError):
        await session.output.write(AudioChunk(b"\x03\x00", output_format, 1))

    rebound, _ = await adapter.initialize(headers=_headers(), raw_request=_init_request(), send=_Sender().send)
    assert rebound is session
    await rebound.disconnect(unexpected=True)
    assert rebound.terminal is True
    with pytest.raises(ConnectA2AConflictError):
        await adapter.initialize(headers=_headers(), raw_request=_init_request(), send=_Sender().send)


@pytest.mark.asyncio
async def test_disconnect_and_reconnect_race_completes_without_deadlock():
    ids = _Ids()
    adapter = ConnectA2AAdapter(api_key=SecretValue("api-key"), readiness=_ready_state(), id_factory=ids)
    session, _ = await adapter.initialize(headers=_headers(), raw_request=_init_request(), send=_Sender().send)

    disconnect = asyncio.create_task(session.disconnect(unexpected=True))
    reconnect = asyncio.create_task(
        adapter.initialize(headers=_headers(), raw_request=_init_request(), send=_Sender().send)
    )
    disconnected, reconnect_result = await asyncio.wait_for(
        asyncio.gather(disconnect, reconnect, return_exceptions=True), timeout=1.0
    )
    assert disconnected is None
    if isinstance(reconnect_result, ConnectA2AConflictError):
        # The reconnect won the race while the old connection was still active.
        # It is safe to retry after disconnect completes, and must not deadlock.
        rebound, _ = await adapter.initialize(
            headers=_headers(), raw_request=_init_request(), send=_Sender().send
        )
        assert rebound is session
    else:
        rebound, _ = reconnect_result
        assert rebound is session


class _BlockingEngine:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release_after_cancel = asyncio.Event()
        self.stale_write_rejected = False

    async def run(self, context: SessionContext) -> EngineResult:
        fmt = AudioFormat(sample_rate_hz=24000)
        await context.ports.outbound_audio.write(AudioChunk(b"\x01\x00", fmt, 0))
        await context.ports.outbound_audio.write(AudioChunk(b"\x02\x00", fmt, 1))
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await self.release_after_cancel.wait()
            try:
                await context.ports.outbound_audio.write(AudioChunk(b"\x02\x00", fmt, 1))
            except ConnectA2AConflictError:
                self.stale_write_rejected = True
            raise


class _WaitingEngine:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def run(self, context: SessionContext) -> EngineResult:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_rebind_fences_canceled_old_engine_from_new_attempt():
    ids = _Ids()
    adapter = ConnectA2AAdapter(api_key=SecretValue("api-key"), readiness=_ready_state(), id_factory=ids)
    session, _ = await adapter.initialize(headers=_headers(), raw_request=_init_request(), send=_Sender().send)
    ready = _request(
        request_id="ready-1", context_id="context-1", event_type="CHANNEL_STATE",
        part={"data": {"channelState": {"state": "READY"}}, "mediaType": "application/json"},
    )
    await session.handle_request(adapter._codec.parse_request(ready))
    old_engine = _BlockingEngine()
    old_task = asyncio.create_task(session.run_engine(old_engine))
    await old_engine.started.wait()

    await session.disconnect(unexpected=True)
    rebound, _ = await adapter.initialize(headers=_headers(), raw_request=_init_request(), send=_Sender().send)
    await rebound.handle_request(adapter._codec.parse_request(ready))
    new_engine = _WaitingEngine()
    new_task = asyncio.create_task(rebound.run_engine(new_engine))
    await new_engine.started.wait()

    old_engine.release_after_cancel.set()
    with pytest.raises(asyncio.CancelledError):
        await old_task
    assert old_engine.stale_write_rejected is True
    assert rebound.terminal is False
    assert rebound._engine_tasks[rebound.generation] is new_task

    new_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await new_task


@pytest.mark.asyncio
async def test_local_barge_in_fences_stale_engine_output_and_limits_frame_bytes():
    ids = _Ids()
    adapter = ConnectA2AAdapter(
        api_key=SecretValue("api-key"),
        readiness=_ready_state(),
        max_pending_audio_chunks=4,
        max_pending_audio_bytes=4,
        max_audio_chunk_bytes=2,
        id_factory=ids,
    )
    sender = _Sender()
    session, _ = await adapter.initialize(headers=_headers(), raw_request=_init_request(), send=sender.send)
    ready = _request(
        request_id="ready-1", context_id="context-1", event_type="CHANNEL_STATE",
        part={"data": {"channelState": {"state": "READY"}}, "mediaType": "application/json"},
    )
    await session.handle_request(adapter._codec.parse_request(ready))
    engine = _BlockingEngine()
    task = asyncio.create_task(session.run_engine(engine))
    await engine.started.wait()
    await session.signal_local_interruption("barge_in")
    engine.release_after_cancel.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert engine.stale_write_rejected is True
    assert len([f for f in sender.frames if "artifactUpdate" in f["result"]]) == 1

    oversized = _request(
        request_id="audio-big", context_id="context-1", event_type=None,
        part={"raw": base64.b64encode(b"\x01\x00\x02\x00").decode("ascii"), "mediaType": "audio/lpcm"},
    )
    with pytest.raises(ConnectA2AOverloadError):
        await session.handle_request(adapter._codec.parse_request(oversized))


class _YieldingSender(_Sender):
    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def send(self, frame: dict[str, object]) -> None:
        self.entered.set()
        await self.release.wait()
        await super().send(frame)


@pytest.mark.asyncio
async def test_interruption_serializes_with_inflight_sender_and_fences_followup_output():
    ids = _Ids()
    adapter = ConnectA2AAdapter(api_key=SecretValue("api-key"), readiness=_ready_state(), id_factory=ids)
    sender = _YieldingSender()
    session, _ = await adapter.initialize(headers=_headers(), raw_request=_init_request(), send=sender.send)
    ready = _request(
        request_id="ready-1", context_id="context-1", event_type="CHANNEL_STATE",
        part={"data": {"channelState": {"state": "READY"}}, "mediaType": "application/json"},
    )
    await session.handle_request(adapter._codec.parse_request(ready))
    fmt = AudioFormat(sample_rate_hz=24000)
    stale = _GenerationBoundOutputSink(session, session.generation, session.output)

    first = asyncio.create_task(stale.write(AudioChunk(b"\x01\x00", fmt, 0)))
    await first
    second = asyncio.create_task(stale.write(AudioChunk(b"\x02\x00", fmt, 1)))
    await sender.entered.wait()
    interruption = asyncio.create_task(session.signal_local_interruption("barge_in"))
    await asyncio.sleep(0)
    assert not interruption.done()
    sender.release.set()
    await second
    await interruption

    with pytest.raises(ConnectA2AConflictError):
        # The output sink remains available for a deliberate new turn, but the
        # stale generation-bound port from the producer cannot write after VAD.
        await stale.write(AudioChunk(b"\x03\x00", fmt, 2))


@pytest.mark.asyncio
async def test_audio_byte_budgets_cover_pending_and_output_frames():
    ids = _Ids()
    adapter = ConnectA2AAdapter(
        api_key=SecretValue("api-key"),
        readiness=_ready_state(),
        max_pending_audio_chunks=4,
        max_pending_audio_bytes=2,
        max_audio_chunk_bytes=2,
        id_factory=ids,
    )
    sender = _Sender()
    session, _ = await adapter.initialize(headers=_headers(), raw_request=_init_request(), send=sender.send)
    ready = _request(
        request_id="ready-1", context_id="context-1", event_type="CHANNEL_STATE",
        part={"data": {"channelState": {"state": "READY"}}, "mediaType": "application/json"},
    )
    await session.handle_request(adapter._codec.parse_request(ready))
    audio = _request(
        request_id="audio-1", context_id="context-1", event_type=None,
        part={"raw": base64.b64encode(b"\x01\x00").decode("ascii"), "mediaType": "audio/lpcm"},
    )
    await session.handle_request(adapter._codec.parse_request(audio))
    with pytest.raises(ConnectA2AOverloadError):
        await session.handle_request(adapter._codec.parse_request(audio))
    with pytest.raises(ConnectA2AOverloadError):
        await session.output.write(AudioChunk(b"\x01\x00\x02\x00", AudioFormat(sample_rate_hz=24000), 0))
