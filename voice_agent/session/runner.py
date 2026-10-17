"""Deterministic owner of session lifecycle, tracing, and port cleanup."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from ..observability.logging import session_logger
from .contracts import (
    EngineResult,
    LifecycleEvent,
    LifecyclePhase,
    SessionContext,
    SessionEngine,
    SessionIdentity,
    SessionOutcome,
    SessionPorts,
    SessionResult,
    TraceSpan,
    TraceStatus,
)


class SessionRunner:
    """Run one engine with ordered lifecycle and guaranteed media cleanup.

    Transport adapters own wire protocol and engines own conversation behavior.
    The runner decides the final outcome only after required media ports close,
    so lifecycle and trace sinks never publish success followed by a fatal media
    cleanup error. Trace and terminal notification failures are diagnostic and
    are logged rather than rewriting an already-decided media outcome.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        logger: logging.Logger | None = None,
    ) -> None:
        self._clock = clock
        self._logger = logger

    async def run(
        self,
        *,
        identity: SessionIdentity,
        ports: SessionPorts,
        engine: SessionEngine,
    ) -> SessionResult:
        started_at = self._clock()
        log = session_logger(identity, self._logger)
        context = SessionContext(identity=identity, ports=ports)
        engine_result = EngineResult(SessionOutcome.ERROR, "session did not start")
        failure: BaseException | None = None

        async def transition(
            phase: LifecyclePhase,
            *,
            outcome: SessionOutcome | None = None,
            reason: str = "",
        ) -> None:
            await ports.lifecycle.transition(
                LifecycleEvent(
                    phase=phase,
                    timestamp_s=self._clock(),
                    outcome=outcome,
                    reason=reason,
                )
            )

        try:
            await transition(LifecyclePhase.CREATED)
            await transition(LifecyclePhase.STARTING)
            await transition(LifecyclePhase.READY)
            await transition(LifecyclePhase.ACTIVE)
            log.info("session active")

            candidate = await engine.run(context)
            if not isinstance(candidate, EngineResult):
                raise TypeError("session engine must return EngineResult")
            engine_result = candidate
        except BaseException as exc:
            failure = exc
            engine_result = EngineResult(
                SessionOutcome.ERROR,
                f"{type(exc).__name__}: {exc}"[:500],
            )
            log.error("session failed", exc_info=(type(exc), exc, exc.__traceback__))
        finally:
            cleanup_errors: list[BaseException] = []
            for close in (ports.inbound_audio.aclose, ports.outbound_audio.aclose):
                try:
                    await close()
                except BaseException as close_error:
                    cleanup_errors.append(close_error)

            if cleanup_errors:
                cleanup_summary = "; ".join(
                    f"{type(error).__name__}: {error}" for error in cleanup_errors
                )[:400]
                if failure is None:
                    failure = RuntimeError(f"session media cleanup failed: {cleanup_summary}")
                    engine_result = EngineResult(
                        SessionOutcome.ERROR,
                        f"media cleanup failed: {cleanup_summary}"[:500],
                    )
                else:
                    engine_result = EngineResult(
                        SessionOutcome.ERROR,
                        f"{engine_result.reason}; media cleanup failed: {cleanup_summary}"[:500],
                    )
                log.error("session media cleanup failed: %s", cleanup_summary)

            # Required media cleanup has completed, so this is now the single
            # authoritative terminal outcome for lifecycle and trace consumers.
            try:
                await transition(
                    LifecyclePhase.FINISHING,
                    outcome=engine_result.outcome,
                    reason=engine_result.reason,
                )
            except BaseException as transition_error:
                log.error(
                    "terminal lifecycle notification failed",
                    exc_info=(
                        type(transition_error),
                        transition_error,
                        transition_error.__traceback__,
                    ),
                )
                if failure is None:
                    failure = transition_error
                    engine_result = EngineResult(
                        SessionOutcome.ERROR,
                        f"lifecycle notification failed: {type(transition_error).__name__}"[:500],
                    )

            ended_at = self._clock()
            try:
                await ports.traces.emit(
                    TraceSpan(
                        name="voice_agent.session",
                        trace_id=identity.logical_session_id,
                        span_id=identity.attempt_id,
                        started_at_s=started_at,
                        ended_at_s=ended_at,
                        status=(
                            TraceStatus.ERROR
                            if engine_result.outcome is SessionOutcome.ERROR
                            else TraceStatus.OK
                        ),
                        attributes={
                            "transport": identity.transport,
                            "outcome": engine_result.outcome.value,
                            "reason": engine_result.reason,
                        },
                        error_type=type(failure).__name__ if failure else None,
                    )
                )
            except BaseException as trace_error:
                log.error(
                    "session trace emission failed",
                    exc_info=(type(trace_error), trace_error, trace_error.__traceback__),
                )

            try:
                await transition(
                    LifecyclePhase.CLOSED,
                    outcome=engine_result.outcome,
                    reason=engine_result.reason,
                )
            except BaseException as transition_error:
                log.error(
                    "closed lifecycle notification failed",
                    exc_info=(
                        type(transition_error),
                        transition_error,
                        transition_error.__traceback__,
                    ),
                )

        ended_at = self._clock()
        if failure is not None:
            raise failure.with_traceback(failure.__traceback__)

        log.info("session closed with outcome=%s", engine_result.outcome.value)
        return SessionResult(
            identity=identity,
            outcome=engine_result.outcome,
            reason=engine_result.reason,
            started_at_s=started_at,
            ended_at_s=ended_at,
        )
