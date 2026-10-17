"""Nova Sonic SessionEngine over transport-independent session ports.

This module is additive. It deliberately does not replace ``bot.py``'s legacy
FastAPI WebSocket builders; the future A2A ASGI bridge will construct this
engine after it has resolved the same runtime configuration, prompts, tools,
and MCP clients.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    EndTaskFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    LLMTextFrame,
    StartFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from ..pipecat.ports_transport import SessionPortsTransport
from ..session.contracts import (
    AudioFormat,
    EngineResult,
    SessionContext,
    SessionEngine,
    SessionEvent,
    SessionOutcome,
)


@dataclass(frozen=True, slots=True)
class NovaSonicEngineConfig:
    """Fully resolved immutable Nova Sonic configuration for one session."""

    region: str
    access_key_id: str
    secret_access_key: str
    session_token: str | None
    voice: str
    system_instruction: str
    greeting: str
    input_format: AudioFormat = AudioFormat(sample_rate_hz=16000)
    output_format: AudioFormat = AudioFormat(sample_rate_hz=24000)
    tools: Any | None = None
    seed_messages: tuple[dict[str, object], ...] = ()
    vad_analyzer: Any | None = None
    mcp_clients: tuple[Any, ...] = ()
    mcp_functions: tuple[tuple[str, Callable[..., Awaitable[None]]], ...] = ()

    def __post_init__(self) -> None:
        if self.input_format.sample_rate_hz != 16000:
            raise ValueError("Nova Sonic requires 16 kHz PCM input")
        if self.output_format.sample_rate_hz != 24000:
            raise ValueError("Nova Sonic engine emits 24 kHz PCM output")


class SessionEventObserver(BaseObserver):
    """Translate Pipecat state frames to generic SessionEvent values."""

    _watched = (
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
        TranscriptionFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
        LLMFullResponseEndFrame,
        TTSStartedFrame,
        TTSStoppedFrame,
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
    )

    def __init__(self, context: SessionContext, *, start_time: float) -> None:
        super().__init__()
        self._context = context
        self._start_time = start_time
        self._seen: set[int] = set()
        self._llm_text = ""

    async def on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame
        if not isinstance(frame, self._watched) or frame.id in self._seen:
            return
        self._seen.add(frame.id)
        now = time.monotonic()
        payload: dict[str, object]
        if isinstance(frame, UserStartedSpeakingFrame):
            # The output transport handles the corresponding InterruptionFrame
            # and owns the single non-terminal A2A playback flush.
            payload = {"name": "user_speaking", "value": True}
        elif isinstance(frame, UserStoppedSpeakingFrame):
            payload = {"name": "user_speaking", "value": False}
        elif isinstance(frame, TranscriptionFrame):
            payload = {"name": "asr_final", "text": frame.text}
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._llm_text = ""
            payload = {"name": "llm_start"}
        elif isinstance(frame, LLMTextFrame):
            self._llm_text += frame.text
            payload = {"name": "llm_delta", "text": frame.text}
        elif isinstance(frame, LLMFullResponseEndFrame):
            payload = {"name": "llm_end", "text": self._llm_text}
            self._llm_text = ""
        elif isinstance(frame, TTSStartedFrame):
            payload = {"name": "tts_start"}
        elif isinstance(frame, TTSStoppedFrame):
            payload = {"name": "tts_end"}
        elif isinstance(frame, BotStartedSpeakingFrame):
            payload = {"name": "bot_speaking", "value": True}
        else:
            payload = {"name": "bot_speaking", "value": False}
        await self._context.ports.events.emit(
            SessionEvent(payload.pop("name"), now, {**payload, "t": round(now - self._start_time, 3)})
        )


class NovaSonicSessionEngine(SessionEngine):
    """Run one Nova Sonic Pipecat graph over a generic SessionContext."""

    def __init__(
        self,
        config: NovaSonicEngineConfig,
        *,
        llm_factory: Callable[[NovaSonicEngineConfig], Any] | None = None,
        transport_factory: Callable[..., SessionPortsTransport] = SessionPortsTransport,
        pipeline_factory: Callable[[list[Any]], Any] = Pipeline,
        task_factory: Callable[..., PipelineTask] = PipelineTask,
        runner_factory: Callable[..., PipelineRunner] = PipelineRunner,
    ) -> None:
        self._config = config
        self._llm_factory = llm_factory or self._build_llm
        self._transport_factory = transport_factory
        self._pipeline_factory = pipeline_factory
        self._task_factory = task_factory
        self._runner_factory = runner_factory

    async def run(self, context: SessionContext) -> EngineResult:
        start_time = time.monotonic()
        terminal = {"outcome": SessionOutcome.COMPLETE, "reason": "pipeline ended"}
        mcp_clients = list(self._config.mcp_clients)
        try:
            transport = self._transport_factory(
                ports=context.ports,
                input_format=self._config.input_format,
                output_format=self._config.output_format,
                name="NovaSonicSessionPorts",
            )
            llm = self._llm_factory(self._config)
            for name, function in self._config.mcp_functions:
                llm.register_function(name, function)

            llm_context = LLMContext(tools=self._config.tools) if self._config.tools else LLMContext()
            for message in self._config.seed_messages:
                llm_context.add_message(dict(message))
            user_agg, assistant_agg = self._build_aggregators(llm_context)
            pipeline = self._pipeline_factory(
                [transport.input(), user_agg, llm, transport.output(), assistant_agg]
            )
            observer = SessionEventObserver(context, start_time=start_time)
            task = self._task_factory(
                pipeline,
                params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
                observers=[observer],
            )

            @task.event_handler("on_pipeline_started")
            async def _on_started(_task, _frame):
                llm_context.add_message({"role": "developer", "content": self._config.greeting})
                await task.queue_frames([LLMRunFrame()])

            @task.event_handler("on_pipeline_error")
            async def _on_error(_task, frame):
                terminal["outcome"] = SessionOutcome.ERROR
                terminal["reason"] = getattr(frame, "error", "Pipecat pipeline error")

            @task.event_handler("on_pipeline_finished")
            async def _on_finished(_task, frame):
                reason = getattr(frame, "reason", "") or terminal["reason"]
                if "transfer" in reason.lower():
                    terminal["outcome"] = SessionOutcome.ESCALATE
                terminal["reason"] = reason

            await self._runner_factory(handle_sigint=False).run(task)
            return EngineResult(terminal["outcome"], str(terminal["reason"]))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return EngineResult(SessionOutcome.ERROR, f"{type(exc).__name__}: {exc}"[:500])
        finally:
            for client in mcp_clients:
                try:
                    await client.close()
                except Exception:
                    pass

    def _build_aggregators(self, llm_context: LLMContext):
        params: dict[str, Any] = {
            "user_turn_strategies": UserTurnStrategies(
                start=[VADUserTurnStartStrategy(enable_interruptions=True, enable_user_speaking_frames=True)],
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.4)],
            )
        }
        if self._config.vad_analyzer is not None:
            params["vad_analyzer"] = self._config.vad_analyzer
        return LLMContextAggregatorPair(llm_context, user_params=LLMUserAggregatorParams(**params))

    @staticmethod
    def _build_llm(config: NovaSonicEngineConfig) -> AWSNovaSonicLLMService:
        kwargs: dict[str, Any] = {
            "access_key_id": config.access_key_id,
            "secret_access_key": config.secret_access_key,
            "session_token": config.session_token,
            "region": config.region,
            "settings": AWSNovaSonicLLMService.Settings(
                model="amazon.nova-2-sonic-v1:0",
                voice=config.voice,
                system_instruction=config.system_instruction,
            ),
        }
        if config.tools is not None:
            kwargs["tools"] = config.tools
        return AWSNovaSonicLLMService(**kwargs)
