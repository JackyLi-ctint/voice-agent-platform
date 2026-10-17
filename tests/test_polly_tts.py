"""Regression tests for the custom Polly TTS aiobotocore integration."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def bot_mod(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "x")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    for mod in list(sys.modules):
        if mod in ("bot", "runtime_config", "demo_loader", "mcp_config"):
            del sys.modules[mod]
    return importlib.import_module("bot")


class _AudioStream:
    async def read(self) -> bytes:
        return b"\x01\x00" * 320  # 20 ms of 16 kHz PCM16 mono


class _PollyClient:
    async def synthesize_speech(self, **kwargs) -> dict:
        self.kwargs = kwargs
        return {"AudioStream": _AudioStream()}


class _ClientContext:
    def __init__(self, client: _PollyClient) -> None:
        self.client = client

    async def __aenter__(self) -> _PollyClient:
        return self.client

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None


class _Session:
    def __init__(self) -> None:
        self.client = _PollyClient()
        self.calls: list[tuple[str, dict]] = []

    def create_client(self, service_name: str, **kwargs) -> _ClientContext:
        self.calls.append((service_name, kwargs))
        return _ClientContext(self.client)


@pytest.mark.asyncio
async def test_simple_polly_uses_aiobotocore_create_client(bot_mod, monkeypatch):
    """The custom service must use AioSession.create_client, not aioboto3's API."""
    session = _Session()
    monkeypatch.setattr(bot_mod, "get_session", lambda: session)

    service = bot_mod.SimplePollyTTSService(
        voice="Joanna",
        engine="neural",
        region="us-east-1",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
    )
    monkeypatch.setattr(service, "start_tts_usage_metrics", AsyncMock())
    monkeypatch.setattr(service, "stop_ttfb_metrics", AsyncMock())

    frames = [frame async for frame in service.run_tts("hello", "ctx-1")]

    assert session.calls == [
        (
            "polly",
            {
                "aws_access_key_id": "test-key",
                "aws_secret_access_key": "test-secret",
                "aws_session_token": None,
                "region_name": "us-east-1",
            },
        )
    ]
    assert session.client.kwargs["Text"] == "hello"
    assert session.client.kwargs["OutputFormat"] == "pcm"
    assert len(frames) == 1
    assert frames[0].audio == b"\x01\x00" * 320
    assert frames[0].sample_rate == 16000
    assert frames[0].num_channels == 1
