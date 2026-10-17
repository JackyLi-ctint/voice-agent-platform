"""Unit tests for the transport-independent Nova Sonic SessionEngine."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from voice_agent.engines.nova_sonic import NovaSonicEngineConfig, NovaSonicSessionEngine
from voice_agent.session.contracts import AudioFormat, EngineResult, SessionContext, SessionIdentity, SessionOutcome
from voice_agent.transports.memory import MemorySession


class _FakeLlm:
    def __init__(self) -> None:
        self.functions = {}

    def register_function(self, name, function) -> None:
        self.functions[name] = function


class _FakeTask:
    def __init__(self, *args, **kwargs) -> None:
        self.handlers = {}
        self.queued = []

    def event_handler(self, name):
        def decorate(function):
            self.handlers[name] = function
            return function
        return decorate

    async def queue_frames(self, frames) -> None:
        self.queued.extend(frames)


class _FakeRunner:
    task = None

    def __init__(self, **kwargs) -> None:
        pass

    async def run(self, task) -> None:
        _FakeRunner.task = task
        await task.handlers["on_pipeline_started"](task, object())
        await task.handlers["on_pipeline_finished"](task, type("Frame", (), {"reason": "transfer requested"})())


def _context() -> SessionContext:
    memory = MemorySession()
    return SessionContext(
        identity=SessionIdentity("session-1", "test", "attempt-1"),
        ports=memory.ports,
    )


@pytest.mark.asyncio
async def test_nova_engine_queues_greeting_and_maps_transfer_to_escalate():
    llm = _FakeLlm()
    config = NovaSonicEngineConfig(
        region="us-east-1",
        access_key_id="key",
        secret_access_key="secret",
        session_token=None,
        voice="tiffany",
        system_instruction="system",
        greeting="hello",
    )
    engine = NovaSonicSessionEngine(
        config,
        llm_factory=lambda _: llm,
        transport_factory=lambda **kwargs: type("Transport", (), {"input": lambda self: object(), "output": lambda self: object()})(),
        pipeline_factory=lambda processors: processors,
        task_factory=_FakeTask,
        runner_factory=_FakeRunner,
    )

    result = await engine.run(_context())

    assert result == EngineResult(SessionOutcome.ESCALATE, "transfer requested")
    assert _FakeRunner.task.queued


@pytest.mark.asyncio
async def test_nova_engine_closes_owned_mcp_clients_on_error():
    class _Client:
        closed = False

        async def close(self):
            self.closed = True

    client = _Client()
    config = NovaSonicEngineConfig(
        region="us-east-1",
        access_key_id="key",
        secret_access_key="secret",
        session_token=None,
        voice="tiffany",
        system_instruction="system",
        greeting="hello",
        mcp_clients=(client,),
    )
    engine = NovaSonicSessionEngine(config, llm_factory=lambda _: (_ for _ in ()).throw(RuntimeError("boom")))

    result = await engine.run(_context())

    assert result.outcome is SessionOutcome.ERROR
    assert "RuntimeError" in result.reason
    assert client.closed is True


@pytest.mark.asyncio
async def test_user_speaking_observer_emits_telemetry_without_flushing_audio():
    from pipecat.frames.frames import UserStartedSpeakingFrame
    from pipecat.observers.base_observer import FramePushed
    from pipecat.processors.frame_processor import FrameDirection

    from voice_agent.engines.nova_sonic import SessionEventObserver

    context = _context()
    observer = SessionEventObserver(context, start_time=0.0)
    await observer.on_push_frame(
        FramePushed(
            source=None,
            destination=None,
            frame=UserStartedSpeakingFrame(),
            direction=FrameDirection.DOWNSTREAM,
            timestamp=0,
        )
    )

    assert context.ports.events.events[0].name == "user_speaking"
    assert context.ports.events.events[0].payload["value"] is True
    assert context.ports.outbound_audio.interruptions == []
