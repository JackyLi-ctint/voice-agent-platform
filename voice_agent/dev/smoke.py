"""Run deterministic, no-AWS Phase 1 session checks.

Usage from the repository root::

    python -m voice_agent.dev.smoke
"""

from __future__ import annotations

import asyncio
import json

from ..observability.logging import configure_logging
from ..session.contracts import (
    AudioChunk,
    AudioFormat,
    EngineResult,
    LifecyclePhase,
    SessionContext,
    SessionEvent,
    SessionIdentity,
    SessionOutcome,
    SessionPorts,
)
from ..session.runner import SessionRunner
from ..transports.memory import MemoryAudioSink, MemorySession


class StepClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class DeterministicEngine:
    def __init__(self, clock: StepClock) -> None:
        self._clock = clock

    async def run(self, context: SessionContext) -> EngineResult:
        async for chunk in context.ports.inbound_audio:
            await context.ports.events.emit(
                SessionEvent(
                    name="audio_received",
                    timestamp_s=self._clock(),
                    payload={"sequence": chunk.sequence, "bytes": len(chunk.data)},
                )
            )
            await context.ports.outbound_audio.write(chunk)
            self._clock.advance(0.02)
        return EngineResult(SessionOutcome.COMPLETE, "phase1_smoke_complete")


class ReturnedErrorEngine:
    async def run(self, context: SessionContext) -> EngineResult:
        return EngineResult(SessionOutcome.ERROR, "expected_engine_error")


class CancelledEngine:
    async def run(self, context: SessionContext) -> EngineResult:
        raise asyncio.CancelledError("expected_cancel")


class FailingCloseSink(MemoryAudioSink):
    async def aclose(self) -> None:
        self.closed = True
        raise OSError("expected_close_failure")


def _identity(suffix: str) -> SessionIdentity:
    return SessionIdentity.for_connect(
        attempt_id=f"attempt-{suffix}",
        context_id=f"context-{suffix}",
        contact_id=f"contact-{suffix}",
        caller_id="anonymous",
    )


async def _verify_success(clock: StepClock) -> dict[str, object]:
    audio_format = AudioFormat(sample_rate_hz=16000)
    chunks = [
        AudioChunk(data=b"\x01\x00\x02\x00", format=audio_format, sequence=0),
        AudioChunk(data=b"\x03\x00\x04\x00", format=audio_format, sequence=1),
    ]
    memory = MemorySession(chunks)
    result = await SessionRunner(clock=clock).run(
        identity=_identity("success"),
        ports=memory.ports,
        engine=DeterministicEngine(clock),
    )

    phases = [event.phase for event in memory.lifecycle.events]
    expected_phases = [
        LifecyclePhase.CREATED,
        LifecyclePhase.STARTING,
        LifecyclePhase.READY,
        LifecyclePhase.ACTIVE,
        LifecyclePhase.FINISHING,
        LifecyclePhase.CLOSED,
    ]
    if phases != expected_phases:
        raise RuntimeError(f"unexpected lifecycle: {phases!r}")
    if [chunk.data for chunk in memory.outbound_audio.chunks] != [chunk.data for chunk in chunks]:
        raise RuntimeError("outbound audio did not match deterministic input")
    if result.outcome is not SessionOutcome.COMPLETE:
        raise RuntimeError(f"unexpected outcome: {result.outcome}")
    if not memory.inbound_audio.closed or not memory.outbound_audio.closed:
        raise RuntimeError("session ports were not closed")
    if len(memory.traces.spans) != 1 or memory.traces.spans[0].status.value != "ok":
        raise RuntimeError("success trace was not completed")

    return {
        "session_id": result.identity.logical_session_id,
        "outcome": result.outcome.value,
        "lifecycle": [phase.value for phase in phases],
        "input_chunks": len(chunks),
        "output_chunks": len(memory.outbound_audio.chunks),
        "events": len(memory.events.events),
        "traces": len(memory.traces.spans),
    }


async def _verify_returned_error(clock: StepClock) -> None:
    memory = MemorySession()
    result = await SessionRunner(clock=clock).run(
        identity=_identity("returned-error"),
        ports=memory.ports,
        engine=ReturnedErrorEngine(),
    )
    if result.outcome is not SessionOutcome.ERROR:
        raise RuntimeError("returned engine error was not preserved")
    if memory.lifecycle.events[-1].outcome is not SessionOutcome.ERROR:
        raise RuntimeError("returned engine error did not reach lifecycle sink")
    if memory.traces.spans[-1].status.value != "error":
        raise RuntimeError("returned engine error did not reach trace sink")


async def _verify_cleanup_failure(clock: StepClock) -> None:
    memory = MemorySession()
    failing_sink = FailingCloseSink()
    ports = SessionPorts(
        inbound_audio=memory.inbound_audio,
        outbound_audio=failing_sink,
        events=memory.events,
        lifecycle=memory.lifecycle,
        traces=memory.traces,
    )
    try:
        await SessionRunner(clock=clock).run(
            identity=_identity("cleanup-error"),
            ports=ports,
            engine=DeterministicEngine(clock),
        )
    except RuntimeError as error:
        if "cleanup" not in str(error):
            raise
    else:
        raise RuntimeError("media cleanup failure did not fail the session")

    terminal = memory.lifecycle.events[-2:]
    if any(event.outcome is not SessionOutcome.ERROR for event in terminal):
        raise RuntimeError("cleanup failure published a successful terminal outcome")
    if memory.traces.spans[-1].status.value != "error":
        raise RuntimeError("cleanup failure published a successful trace")


async def _verify_cancellation(clock: StepClock) -> None:
    memory = MemorySession()
    try:
        await SessionRunner(clock=clock).run(
            identity=_identity("cancel"),
            ports=memory.ports,
            engine=CancelledEngine(),
        )
    except asyncio.CancelledError:
        pass
    else:
        raise RuntimeError("engine cancellation was not propagated")
    if memory.lifecycle.events[-1].outcome is not SessionOutcome.ERROR:
        raise RuntimeError("cancellation did not publish an error lifecycle")
    if not memory.inbound_audio.closed or not memory.outbound_audio.closed:
        raise RuntimeError("canceled session ports were not closed")


async def run_smoke() -> dict[str, object]:
    clock = StepClock()
    success = await _verify_success(clock)
    await _verify_returned_error(clock)
    await _verify_cleanup_failure(clock)
    await _verify_cancellation(clock)
    return {"ok": True, **success, "failure_paths": 3}


def main() -> None:
    configure_logging(level="INFO", json_output=False)
    print(json.dumps(asyncio.run(run_smoke()), indent=2))


if __name__ == "__main__":
    main()
