"""ASGI WebSocket bridge for Amazon Connect external A2A voice sessions.

The bridge deliberately owns only WebSocket mechanics. Authentication, A2A
JSON-RPC parsing, audio negotiation, bounded queues, artifacts, terminal
statuses, and reconnect policy remain in :mod:`connect_a2a`.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from ..session.contracts import SessionEngine, SessionOutcome
from .connect_a2a import (
    ConnectA2AAdapter,
    ConnectA2AAuthenticationError,
    ConnectA2AError,
    ConnectA2ANotReadyError,
    ConnectA2AProtocolError,
    ConnectA2ASession,
)

EngineFactory = Callable[[ConnectA2ASession], SessionEngine | Awaitable[SessionEngine]]


def _accept_headers(adapter: ConnectA2AAdapter) -> list[tuple[bytes, bytes]]:
    return [
        (name.encode("latin-1"), value.encode("latin-1"))
        for name, value in adapter.extension_response_headers.items()
    ]


async def _close_if_open(websocket: WebSocket, *, code: int) -> None:
    if websocket.application_state is not WebSocketState.DISCONNECTED:
        with suppress(RuntimeError):
            await websocket.close(code=code)


async def _resolve_engine(
    factory: EngineFactory,
    session: ConnectA2ASession,
) -> SessionEngine:
    candidate = factory(session)
    if inspect.isawaitable(candidate):
        candidate = await candidate
    return candidate


async def handle_connect_a2a_websocket(
    websocket: WebSocket,
    *,
    adapter: ConnectA2AAdapter,
    engine_factory: EngineFactory,
    logger: logging.Logger | None = None,
    initial_frame_timeout_s: float = 10.0,
) -> None:
    """Serve one authenticated Connect A2A collaboration over ``websocket``.

    The engine starts only after the first successful ``CHANNEL_STATE READY``.
    Outbound acknowledgements and engine-driven A2A envelopes share one lock so
    Starlette never receives concurrent ``send_text`` calls for one socket.
    """
    log = logger or logging.getLogger(__name__)
    send_lock = asyncio.Lock()
    session: ConnectA2ASession | None = None
    engine_task: asyncio.Task[object] | None = None

    async def send(envelope: dict[str, object]) -> None:
        async with send_lock:
            if websocket.application_state is WebSocketState.DISCONNECTED:
                return
            await websocket.send_text(json.dumps(envelope, separators=(",", ":")))

    if initial_frame_timeout_s <= 0:
        raise ValueError("initial_frame_timeout_s must be positive")

    try:
        # Reject unauthenticated, unsupported, or unavailable peers before
        # upgrading them to a long-lived socket. INIT_SESSION must then arrive
        # promptly so an otherwise valid idle peer cannot consume a worker.
        adapter.validate_handshake_headers(websocket.headers)
        await websocket.accept(headers=_accept_headers(adapter))
        initial_message = await asyncio.wait_for(
            websocket.receive_text(),
            timeout=initial_frame_timeout_s,
        )
        session, init_response = await adapter.initialize(
            headers=websocket.headers,
            raw_request=initial_message,
            send=send,
        )
        await send(init_response)

        while True:
            raw_message = await websocket.receive_text()
            request = adapter.parse_request(raw_message)
            was_ready = session.ready
            acknowledgement = await session.handle_request(request)
            await send(acknowledgement)

            if request.event_type == "CHANNEL_STATE" and session.ready and not was_ready:
                engine = await _resolve_engine(engine_factory, session)
                engine_task = asyncio.create_task(
                    session.run_engine(engine),
                    name=f"connect-a2a-engine:{session.identity.logical_session_id}",
                )
    except WebSocketDisconnect:
        log.info("Connect A2A WebSocket disconnected")
    except asyncio.TimeoutError:
        log.warning("Connect A2A WebSocket initialization timed out")
        await _close_if_open(websocket, code=1008)
    except (ConnectA2AAuthenticationError, ConnectA2ANotReadyError, ConnectA2AProtocolError) as exc:
        close_code = 1002 if isinstance(exc, ConnectA2AProtocolError) else exc.close_code
        log.warning("Connect A2A WebSocket rejected: %s", type(exc).__name__)
        await _close_if_open(websocket, code=close_code)
    except ConnectA2AError as exc:
        log.warning("Connect A2A session rejected: %s", type(exc).__name__)
        await _close_if_open(websocket, code=exc.close_code)
    except Exception:
        log.exception("Connect A2A WebSocket failed")
        if session is not None and not session.terminal:
            with suppress(Exception):
                await session.finish(SessionOutcome.ERROR, "A2A engine initialization failed")
        await _close_if_open(websocket, code=1011)
    finally:
        if session is not None:
            with suppress(Exception):
                await session.disconnect(unexpected=True)
        if engine_task is not None:
            with suppress(asyncio.CancelledError, Exception):
                await engine_task
