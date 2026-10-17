"""Unit tests for the Pipecat SessionPorts transport bridge."""

from __future__ import annotations

import asyncio

import pytest

from pipecat.frames.frames import InputAudioRawFrame, InterruptionFrame, OutputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection

from voice_agent.pipecat.ports_transport import SessionPortsTransport
from voice_agent.session.contracts import AudioChunk, AudioFormat
from voice_agent.transports.memory import MemorySession


@pytest.mark.asyncio
async def test_input_pump_preserves_negotiated_pcm_bytes_and_rate(monkeypatch):
    input_format = AudioFormat(sample_rate_hz=16000)
    memory = MemorySession([AudioChunk(b"\x01\x00\x02\x00", input_format, 0)])
    transport = SessionPortsTransport(
        ports=memory.ports,
        input_format=input_format,
        output_format=AudioFormat(sample_rate_hz=24000),
    )
    captured = []

    async def push(frame):
        captured.append(frame)

    monkeypatch.setattr(transport.input(), "push_audio_frame", push)
    await transport.input()._pump_ports()

    assert len(captured) == 1
    assert isinstance(captured[0], InputAudioRawFrame)
    assert captured[0].audio == b"\x01\x00\x02\x00"
    assert captured[0].sample_rate == 16000
    assert captured[0].num_channels == 1


@pytest.mark.asyncio
async def test_source_eof_terminates_a_real_pipeline_task():
    """A finite generic source must end the live Pipecat task, not just a pump."""
    input_format = AudioFormat(sample_rate_hz=16000)
    memory = MemorySession([AudioChunk(b"\x01\x00", input_format, 0)])
    transport = SessionPortsTransport(
        ports=memory.ports,
        input_format=input_format,
        output_format=AudioFormat(sample_rate_hz=24000),
    )
    task = PipelineTask(Pipeline([transport.input(), transport.output()]))

    await asyncio.wait_for(PipelineRunner().run(task), timeout=1.0)


@pytest.mark.asyncio
async def test_output_bridge_writes_sequenced_chunks_to_session_ports():
    output_format = AudioFormat(sample_rate_hz=24000)
    memory = MemorySession()
    transport = SessionPortsTransport(
        ports=memory.ports,
        input_format=AudioFormat(sample_rate_hz=16000),
        output_format=output_format,
    )

    ok = await transport.output().write_audio_frame(
        OutputAudioRawFrame(audio=b"\x03\x00", sample_rate=24000, num_channels=1)
    )

    assert ok is True
    assert memory.outbound_audio.chunks[0].data == b"\x03\x00"
    assert memory.outbound_audio.chunks[0].format == output_format
    assert memory.outbound_audio.chunks[0].sequence == 0


@pytest.mark.asyncio
async def test_output_interruption_is_forwarded_once_to_the_session_port(monkeypatch):
    from pipecat.transports.base_output import BaseOutputTransport

    async def ignore_downstream_processing(self, frame, direction):
        return None

    monkeypatch.setattr(BaseOutputTransport, "process_frame", ignore_downstream_processing)
    memory = MemorySession()
    transport = SessionPortsTransport(
        ports=memory.ports,
        input_format=AudioFormat(sample_rate_hz=16000),
        output_format=AudioFormat(sample_rate_hz=24000),
    )

    await transport.output().process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

    assert memory.outbound_audio.interruptions == ["pipecat_interruption"]


@pytest.mark.asyncio
async def test_output_bridge_rejects_wrong_negotiated_rate():
    memory = MemorySession()
    transport = SessionPortsTransport(
        ports=memory.ports,
        input_format=AudioFormat(sample_rate_hz=16000),
        output_format=AudioFormat(sample_rate_hz=24000),
    )

    with pytest.raises(ValueError, match="sample rate"):
        await transport.output().write_audio_frame(
            OutputAudioRawFrame(audio=b"\x03\x00", sample_rate=16000, num_channels=1)
        )
