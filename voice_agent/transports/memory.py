"""Deterministic in-memory ports for local development and protocol harnesses."""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Iterable

from ..session.contracts import (
    AudioChunk,
    LifecycleEvent,
    SessionEvent,
    SessionPorts,
    TraceSpan,
)


class MemoryAudioSource:
    def __init__(self, chunks: Iterable[AudioChunk] = ()) -> None:
        self._chunks = deque(chunks)
        self.closed = False

    def __aiter__(self) -> AsyncIterator[AudioChunk]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[AudioChunk]:
        while self._chunks and not self.closed:
            yield self._chunks.popleft()

    async def aclose(self) -> None:
        self.closed = True
        self._chunks.clear()


class MemoryAudioSink:
    def __init__(self) -> None:
        self.chunks: list[AudioChunk] = []
        self.interruptions: list[str] = []
        self.closed = False

    async def write(self, chunk: AudioChunk) -> None:
        if self.closed:
            raise RuntimeError("outbound audio sink is closed")
        self.chunks.append(chunk)

    async def interrupt(self, reason: str) -> None:
        if self.closed:
            raise RuntimeError("outbound audio sink is closed")
        self.interruptions.append(reason)

    async def aclose(self) -> None:
        self.closed = True


class MemoryEventSink:
    def __init__(self) -> None:
        self.events: list[SessionEvent] = []

    async def emit(self, event: SessionEvent) -> None:
        self.events.append(event)


class MemoryLifecycleSink:
    def __init__(self) -> None:
        self.events: list[LifecycleEvent] = []

    async def transition(self, event: LifecycleEvent) -> None:
        if self.events and self.events[-1].phase == event.phase:
            return
        self.events.append(event)


class MemoryTraceSink:
    def __init__(self) -> None:
        self.spans: list[TraceSpan] = []

    async def emit(self, span: TraceSpan) -> None:
        self.spans.append(span)


class MemorySession:
    """Own all in-memory ports and expose the composed SessionPorts value."""

    def __init__(self, chunks: Iterable[AudioChunk] = ()) -> None:
        self.inbound_audio = MemoryAudioSource(chunks)
        self.outbound_audio = MemoryAudioSink()
        self.events = MemoryEventSink()
        self.lifecycle = MemoryLifecycleSink()
        self.traces = MemoryTraceSink()
        self.ports = SessionPorts(
            inbound_audio=self.inbound_audio,
            outbound_audio=self.outbound_audio,
            events=self.events,
            lifecycle=self.lifecycle,
            traces=self.traces,
        )
