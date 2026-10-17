"""ASGI integration tests for the Amazon Connect external A2A bridge."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from collections.abc import Mapping

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect, WebSocketState

from voice_agent.app import ReadinessState, SecretValue
from voice_agent.session.contracts import AudioChunk, AudioFormat, EngineResult, SessionContext, SessionOutcome
from voice_agent.transports.connect_a2a import (
    EXTENSION_HEADER,
    EXTENSION_URI,
    EVENT_TYPE_KEY,
    ConnectA2AAdapter,
)
from voice_agent.transports.connect_a2a_asgi import handle_connect_a2a_websocket


def _ready_state() -> ReadinessState:
    state = ReadinessState("test-a2a-asgi")
    state.begin_startup()
    state.set_check("core", ok=True, critical=True)
    state.mark_started()
    return state


def _request(
    *,
    request_id: str,
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
                "contextId": "context-1",
                "role": "ROLE_USER",
                "parts": [dict(part)],
                "metadata": metadata,
            }
        },
    }


def _init_request() -> dict[str, object]:
    return _request(
        request_id="init-1",
        event_type="INIT_SESSION",
        part={
            "data": {
                "initSession": {
                    "instanceArn": "arn:aws:connect:us-east-1:111111111111:instance/abc",
                    "contactArn": "arn:aws:connect:us-east-1:111111111111:instance/abc/contact/contact-1",
                    "supportedFinishTypes": ["COMPLETE", "ESCALATE"],
                    "subscribeToTracingEvents": False,
                    "audioInputConfiguration": {
                        "encoding": "LINEAR_PCM",
                        "sampleRateHertz": 8000,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                    },
                    "audioOutputConfiguration": {
                        "encoding": "LINEAR_PCM",
                        "sampleRateHertz": 16000,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                    },
                }
            },
            "mediaType": "application/json",
        },
    )


def _ready_request() -> dict[str, object]:
    return _request(
        request_id="ready-1",
        event_type="CHANNEL_STATE",
        part={"data": {"channelState": {"state": "READY"}}, "mediaType": "application/json"},
    )


def _audio_request(pcm: bytes) -> dict[str, object]:
    return _request(
        request_id="audio-1",
        event_type=None,
        part={"raw": base64.b64encode(pcm).decode("ascii"), "mediaType": "audio/lpcm"},
    )


class _AudioEngine:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.received: bytes | None = None

    async def run(self, context: SessionContext) -> EngineResult:
        self.started.set()
        try:
            async for chunk in context.ports.inbound_audio:
                self.received = chunk.data
                output_format = AudioFormat(sample_rate_hz=24000)
                await context.ports.outbound_audio.write(AudioChunk(b"\x03\x00", output_format, 0))
                return EngineResult(SessionOutcome.COMPLETE, "done")
            return EngineResult(SessionOutcome.REMOTE_DISCONNECT, "socket closed")
        finally:
            self.stopped.set()


def _client(
    *,
    api_key: str = "api-key",
    engine: _AudioEngine | None = None,
    initial_frame_timeout_s: float = 10.0,
) -> tuple[TestClient, _AudioEngine]:
    app = FastAPI()
    adapter = ConnectA2AAdapter(api_key=SecretValue(api_key), readiness=_ready_state())
    selected_engine = engine or _AudioEngine()

    @app.websocket("/connect/a2a")
    async def endpoint(websocket: WebSocket):
        await handle_connect_a2a_websocket(
            websocket,
            adapter=adapter,
            engine_factory=lambda _: selected_engine,
            initial_frame_timeout_s=initial_frame_timeout_s,
        )

    return TestClient(app), selected_engine


def _headers(*, token: str = "api-key", extension: bool = True) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {token}"}
    if extension:
        headers[EXTENSION_HEADER] = EXTENSION_URI
    return headers


def test_a2a_endpoint_rejects_missing_bearer_token():
    client, _ = _client()
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/connect/a2a", headers={EXTENSION_HEADER: EXTENSION_URI}) as socket:
            socket.send_text(json.dumps(_init_request()))
            socket.receive_text()
    assert exc.value.code == 1008


def test_a2a_endpoint_rejects_missing_required_extension():
    client, _ = _client()
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/connect/a2a", headers=_headers(extension=False)) as socket:
            socket.send_text(json.dumps(_init_request()))
            socket.receive_text()
    assert exc.value.code == 1002


def test_a2a_endpoint_starts_engine_after_ready_and_routes_pcm():
    client, engine = _client()
    with client.websocket_connect("/connect/a2a", headers=_headers()) as socket:
        socket.send_text(json.dumps(_init_request()))
        init_response = json.loads(socket.receive_text())
        init = init_response["result"]["message"]["parts"][0]["data"]["initSessionResponse"]
        assert init["audioInputConfiguration"]["sampleRateHertz"] == 16000
        assert init["audioOutputConfiguration"]["sampleRateHertz"] == 24000

        socket.send_text(json.dumps(_ready_request()))
        assert json.loads(socket.receive_text())["id"] == "ready-1"
        assert engine.started.wait(timeout=1.0)

        pcm = b"\x01\x00\x02\x00"
        socket.send_text(json.dumps(_audio_request(pcm)))
        messages = [json.loads(socket.receive_text()) for _ in range(3)]

    assert engine.received == pcm
    assert any(message.get("id") == "audio-1" for message in messages)
    assert any("artifactUpdate" in message.get("result", {}) for message in messages)
    assert any("statusUpdate" in message.get("result", {}) for message in messages)


def test_a2a_endpoint_starts_engine_only_once_for_repeated_ready():
    class _CountingEngine(_AudioEngine):
        def __init__(self) -> None:
            super().__init__()
            self.run_calls = 0

        async def run(self, context: SessionContext) -> EngineResult:
            self.run_calls += 1
            return await super().run(context)

    client, engine = _client(engine=_CountingEngine())
    with client.websocket_connect("/connect/a2a", headers=_headers()) as socket:
        socket.send_text(json.dumps(_init_request()))
        socket.receive_text()
        socket.send_text(json.dumps(_ready_request()))
        socket.receive_text()
        assert engine.started.wait(timeout=1.0)
        socket.send_text(json.dumps(_ready_request()))
        socket.receive_text()

    assert engine.started.is_set()
    assert engine.run_calls == 1
    assert engine.stopped.wait(timeout=1.0)


def test_a2a_endpoint_closes_idle_authenticated_socket_before_init():
    client, _ = _client(initial_frame_timeout_s=0.01)
    with client.websocket_connect("/connect/a2a", headers=_headers()) as socket:
        with pytest.raises(WebSocketDisconnect) as exc:
            socket.receive_text()
    assert exc.value.code == 1008


class _ConcurrentSendWebSocket:
    """Minimal ASGI WebSocket double that detects overlapping send_text calls."""

    def __init__(self) -> None:
        self.headers = _headers()
        self.application_state = WebSocketState.CONNECTING
        self._messages = iter(
            [
                json.dumps(_init_request()),
                json.dumps(_ready_request()),
                json.dumps(_audio_request(b"\x01\x00")),
            ]
        )
        self.sent: list[dict[str, object]] = []
        self._active_sends = 0
        self.max_concurrent_sends = 0

    async def accept(self, *, headers):
        self.application_state = WebSocketState.CONNECTED

    async def receive_text(self) -> str:
        try:
            message = next(self._messages)
        except StopIteration:
            await asyncio.sleep(0.03)
            raise WebSocketDisconnect(code=1000)
        # Let the engine task race its artifact output with the next inbound
        # acknowledgement send; the bridge lock must serialize both.
        await asyncio.sleep(0.002)
        return message

    async def send_text(self, payload: str) -> None:
        self._active_sends += 1
        self.max_concurrent_sends = max(self.max_concurrent_sends, self._active_sends)
        try:
            await asyncio.sleep(0.01)
            self.sent.append(json.loads(payload))
        finally:
            self._active_sends -= 1

    async def close(self, *, code: int) -> None:
        self.application_state = WebSocketState.DISCONNECTED


class _ConcurrentOutputEngine:
    async def run(self, context: SessionContext) -> EngineResult:
        output_format = AudioFormat(sample_rate_hz=24000)
        await context.ports.outbound_audio.write(AudioChunk(b"\x03\x00", output_format, 0))
        await context.ports.outbound_audio.write(AudioChunk(b"\x04\x00", output_format, 1))
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_a2a_bridge_serializes_engine_output_and_request_acknowledgements():
    socket = _ConcurrentSendWebSocket()
    adapter = ConnectA2AAdapter(api_key=SecretValue("api-key"), readiness=_ready_state())

    await handle_connect_a2a_websocket(
        socket,
        adapter=adapter,
        engine_factory=lambda _: _ConcurrentOutputEngine(),
    )

    assert socket.max_concurrent_sends == 1
    assert any("artifactUpdate" in frame.get("result", {}) for frame in socket.sent)
    assert any(frame.get("id") == "audio-1" for frame in socket.sent)


def test_a2a_endpoint_allows_one_reconnect_then_rejects_a_third_attempt():
    class _ReconnectEngine(_AudioEngine):
        def __init__(self) -> None:
            super().__init__()
            self.run_calls = 0

        async def run(self, context: SessionContext) -> EngineResult:
            self.run_calls += 1
            return await super().run(context)

    client, engine = _client(engine=_ReconnectEngine())
    for _ in range(2):
        with client.websocket_connect("/connect/a2a", headers=_headers()) as socket:
            socket.send_text(json.dumps(_init_request()))
            socket.receive_text()
            socket.send_text(json.dumps(_ready_request()))
            socket.receive_text()
            assert engine.started.wait(timeout=1.0)

    assert engine.run_calls == 2
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/connect/a2a", headers=_headers()) as socket:
            socket.send_text(json.dumps(_init_request()))
            socket.receive_text()
    assert exc.value.code == 1008
