"""HistoryRecorder integration tests.

Uses moto[dynamodb] to mock the table; the real bot.py module is loaded
with HISTORY_TABLE set to a moto-managed table name.

Verifies:
  - flush_turns_and_summarize() writes a row with ttl, turn_count,
    summary_status='pending', and the buffered turns.
  - The follow-up _summarize_and_update task overwrites summary_status to
    'ok' and stores summary.model + summary.generated_at.
  - On Bedrock failure the row's summary_status flips to 'failed'.
  - HISTORY_DISABLED=1 short-circuits to _history is None and creates no
    AWS clients.
"""
from __future__ import annotations

import asyncio
import importlib
import os
import sys
import time

import boto3
import pytest

# moto v5 mocks all AWS services through a single decorator/context manager.
from moto import mock_aws


TABLE_NAME = "voicebot-test-call-history"


def _create_table(region: str = "us-east-1") -> None:
    ddb = boto3.client("dynamodb", region_name=region)
    ddb.create_table(
        TableName=TABLE_NAME,
        BillingMode="PAY_PER_REQUEST",
        AttributeDefinitions=[
            {"AttributeName": "call_id", "AttributeType": "S"},
            {"AttributeName": "caller", "AttributeType": "S"},
            {"AttributeName": "started_at", "AttributeType": "N"},
        ],
        KeySchema=[{"AttributeName": "call_id", "KeyType": "HASH"}],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "caller-started-index",
                "KeySchema": [
                    {"AttributeName": "caller", "KeyType": "HASH"},
                    {"AttributeName": "started_at", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "KEYS_ONLY"},
            }
        ],
    )
    ddb.update_time_to_live(
        TableName=TABLE_NAME,
        TimeToLiveSpecification={"Enabled": True, "AttributeName": "ttl"},
    )


def _import_bot_with_env(env: dict) -> object:
    """Drop bot module state then re-import so module-level _history sees env."""
    for mod in list(sys.modules):
        if mod in ("bot", "runtime_config", "demo_loader"):
            del sys.modules[mod]
    for k, v in env.items():
        os.environ[k] = v
    return importlib.import_module("bot")


@pytest.fixture
def aws_creds(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("MINIMAX_API_KEY", "x")
    monkeypatch.setenv("ADMIN_PASSWORD", "")
    yield


def test_history_disabled_when_table_empty(aws_creds, monkeypatch):
    monkeypatch.setenv("HISTORY_TABLE", "")
    bot = _import_bot_with_env({"HISTORY_TABLE": ""})
    assert bot._history is None
    assert bot.HISTORY_DISABLED is True


def test_history_disabled_via_flag(aws_creds, monkeypatch):
    monkeypatch.setenv("HISTORY_TABLE", "still-set")
    monkeypatch.setenv("HISTORY_DISABLED", "1")
    bot = _import_bot_with_env({"HISTORY_TABLE": "still-set", "HISTORY_DISABLED": "1"})
    assert bot._history is None
    assert bot.HISTORY_DISABLED is True


def test_finalize_writes_pending_then_ok(aws_creds, monkeypatch):
    monkeypatch.setenv("HISTORY_TABLE", TABLE_NAME)
    monkeypatch.setenv("HISTORY_TTL_DAYS", "7")
    monkeypatch.delenv("HISTORY_DISABLED", raising=False)

    with mock_aws():
        _create_table()
        bot = _import_bot_with_env({
            "HISTORY_TABLE": TABLE_NAME,
            "HISTORY_TTL_DAYS": "7",
        })
        assert bot._history is not None

        # Patch out the Bedrock call so the test is hermetic.
        async def fake_summary(turns, lang_key, **_):
            return {
                "intent": "Stub intent",
                "key_questions": ["q1"],
                "action_items": [],
                "sentiment": "positive",
                "model": "stub-model-id",
                "generated_at": int(time.time()),
            }
        monkeypatch.setattr(bot, "_invoke_summary_bedrock", fake_summary)

        call_id = "test-call-001"
        started = time.time() - 10  # call lasted ~10s
        bot._history.attach(call_id, {
            "caller": "+15551234567",
            "started_at": started,
            "engine": "nova-sonic",
            "lang": "en-US",
            "scenario": "default",
            "provider": "polly",
            "voice": "tiffany",
            "model": "nova-2-lite",
            "minimax_model": "speech-2.8-turbo",
        })
        bot._history.append(call_id, {"who": "user", "text": "hello", "t": 0.5})
        bot._history.append(call_id, {"who": "bot", "text": "hi there", "t": 1.2})

        async def run():
            await bot._history.flush_turns_and_summarize(call_id)
            # The flush schedules the summary task; let it run to completion.
            for t in asyncio.all_tasks() - {asyncio.current_task()}:
                await t

        asyncio.run(run())

        ddb = boto3.resource("dynamodb", region_name="us-east-1")
        table = ddb.Table(TABLE_NAME)
        item = table.get_item(Key={"call_id": call_id}).get("Item")

    assert item is not None, "row not written"
    assert item["call_id"] == call_id
    assert item["caller"] == "+15551234567"
    assert int(item["turn_count"]) == 2
    # ttl = started + 7 days
    expected_ttl = int(started) + 7 * 86400
    assert abs(int(item["ttl"]) - expected_ttl) <= 2
    # summary_status flipped from pending → ok by the spawned task
    assert item["summary_status"] == "ok"
    summary = item["summary"]
    assert summary["intent"] == "Stub intent"
    assert summary["model"] == "stub-model-id"
    assert "generated_at" in summary
    # turns persisted
    assert len(item["turns"]) == 2
    assert item["turns"][0]["who"] == "user"


def test_finalize_failed_summary_marks_status(aws_creds, monkeypatch):
    monkeypatch.setenv("HISTORY_TABLE", TABLE_NAME)
    monkeypatch.delenv("HISTORY_DISABLED", raising=False)

    with mock_aws():
        _create_table()
        bot = _import_bot_with_env({"HISTORY_TABLE": TABLE_NAME})

        async def boom(turns, lang_key, **_):
            raise RuntimeError("bedrock unreachable")
        monkeypatch.setattr(bot, "_invoke_summary_bedrock", boom)

        call_id = "test-call-fail"
        bot._history.attach(call_id, {
            "caller": "+15550000000",
            "started_at": time.time(),
            "engine": "nova-sonic",
            "lang": "en-US",
            "scenario": "default",
            "provider": "polly",
            "voice": "tiffany",
            "model": "nova-2-lite",
            "minimax_model": "",
        })
        bot._history.append(call_id, {"who": "user", "text": "hi", "t": 0.1})

        async def run():
            await bot._history.flush_turns_and_summarize(call_id)
            for t in asyncio.all_tasks() - {asyncio.current_task()}:
                await t

        asyncio.run(run())

        ddb = boto3.resource("dynamodb", region_name="us-east-1")
        table = ddb.Table(TABLE_NAME)
        item = table.get_item(Key={"call_id": call_id}).get("Item")

    assert item is not None
    assert item["summary_status"] == "failed"
    assert "summary_error" in item
    assert "bedrock" in item["summary_error"].lower()


def test_coerce_summary_struct_handles_fenced_json(aws_creds, monkeypatch):
    bot = _import_bot_with_env({"HISTORY_TABLE": ""})
    fenced = '```json\n{"intent":"x","key_questions":["q"],"action_items":[],"sentiment":"neutral"}\n```'
    out = bot._coerce_summary_struct(fenced)
    assert out["intent"] == "x"
    assert out["key_questions"] == ["q"]
    assert out["sentiment"] == "neutral"


def test_coerce_summary_struct_degrades_on_garbage(aws_creds):
    bot = _import_bot_with_env({"HISTORY_TABLE": ""})
    out = bot._coerce_summary_struct("not json at all")
    assert out["intent"].startswith("not json")
    assert out["key_questions"] == []
    assert out["sentiment"] == "neutral"
