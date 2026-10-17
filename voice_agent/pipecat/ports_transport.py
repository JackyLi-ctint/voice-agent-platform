"""Pipecat transport backed by :class:`voice_agent.session.SessionPorts`.

This bridge deliberately contains no ASGI, WebSocket, SIP, Chime, or Connect
protocol code. Pipecat sees its normal input/output transports while the caller
owns durable audio ports and their lifecycle.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress

from pipecat.frames.frames import CancelFrame, EndFrame, ErrorFrame, Frame, InputAudioRawFrame, InterruptionFrame, OutputAudioRawFrame, StartFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from ..session.contracts import AudioChunk, AudioFormat, SessionPorts


class _PortsInputTransport(BaseInputTransport):
    def __init__(self, transport: "SessionPortsTransport", params: TransportParams, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self._transport = transport
        self._ports_task: asyncio.Task | None = None

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        await self.set_transport_ready(frame)
        if self._ports_task is None:
            self._ports_task = self.create_task(self._pump_ports())

    async def stop(self, frame: EndFrame) -> None:
        await self._stop_pump()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame) -> None:
        await self._stop_pump()
        await super().cancel(frame)

    async def _stop_pump(self) -> None:
        if self._ports_task is not None:
            self._ports_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._ports_task
            self._ports_task = None

    async def _pump_ports(self) -> None:
        source_exhausted = False
        try:
            async for chunk in self._transport.ports.inbound_audio:
                if chunk.format != self._transport.input_format:
                    raise ValueError("inbound audio does not match negotiated input format")
                await self.push_audio_frame(
                    InputAudioRawFrame(
                        audio=chunk.data,
                        sample_rate=chunk.format.sample_rate_hz,
                        num_channels=chunk.format.channels,
                    )
                )
            source_exhausted = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.push_frame(ErrorFrame(error=f"ports input failed: {exc}", fatal=True))
        finally:
            if source_exhausted and self._pipeline_worker is not None:
                # PipelineTask wires the public pipeline_worker property before
                # starting transports. The guard keeps direct pump unit tests
                # independent from Pipecat task construction.
                await self.pipeline_worker.queue_frame(EndFrame(reason="session input closed"))


class _PortsOutputTransport(BaseOutputTransport):
    def __init__(self, transport: "SessionPortsTransport", params: TransportParams, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self._transport = transport
        self._sequence = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, InterruptionFrame):
            # The bound A2A output port maps this to a non-terminal Connect
            # playback flush; other ports may record it as an interruption.
            await self._transport.ports.outbound_audio.interrupt("pipecat_interruption")
        await super().process_frame(frame, direction)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        if frame.sample_rate != self._transport.output_format.sample_rate_hz:
            raise ValueError("Pipecat output audio sample rate does not match negotiated output")
        if frame.num_channels != self._transport.output_format.channels:
            raise ValueError("Pipecat output audio channels do not match negotiated output")
        chunk = AudioChunk(
            data=bytes(frame.audio),
            format=self._transport.output_format,
            sequence=self._sequence,
        )
        self._sequence += 1
        await self._transport.ports.outbound_audio.write(chunk)
        return True


class SessionPortsTransport(BaseTransport):
    """Bidirectional Pipecat transport over a generic :class:`SessionPorts`."""

    def __init__(
        self,
        *,
        ports: SessionPorts,
        input_format: AudioFormat,
        output_format: AudioFormat,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, input_name="SessionPortsInput", output_name="SessionPortsOutput")
        self.ports = ports
        self.input_format = input_format
        self.output_format = output_format
        params = TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=input_format.sample_rate_hz,
            audio_in_channels=input_format.channels,
            audio_in_passthrough=True,
            audio_out_enabled=True,
            audio_out_sample_rate=output_format.sample_rate_hz,
            audio_out_channels=output_format.channels,
            audio_out_auto_silence=False,
            audio_out_end_silence_secs=0,
        )
        self._input = _PortsInputTransport(self, params, name=self._input_name)
        self._output = _PortsOutputTransport(self, params, name=self._output_name)

    def input(self) -> _PortsInputTransport:
        return self._input

    def output(self) -> _PortsOutputTransport:
        return self._output

    async def cleanup(self) -> None:
        await self._input.cleanup()
        await self._output.cleanup()
