"""Tests for demo_store.DemoStore (T3) — DDB-backed demo store that preserves
DemoLoader's interface.

Coverage:
  - put -> rescan -> get/list reads back the SAME dict, including the
    ``tool_ids`` key (NOT ``tools``), kb_chars summary, and a Decimal->float
    round-trip for filler.probability.
  - rescan() scans the whole (paginated) table into the in-memory cache.
  - Table-missing degrades: get()->None, list()->[], last_skipped recorded,
    no raise.
  - The pure normalization helper (demo_loader.normalize_demo_dict) yields a
    dict whose tool list key is ``tool_ids``.

DynamoDB is mocked with moto (mock_aws). If moto were unavailable, the
table-missing degrade path and the pure-helper assertion still run standalone
(they need no live table) — see the module-level skip guard.
"""

from __future__ import annotations

import asyncio
import os

import boto3
import pytest

try:
    from moto import mock_aws
    _HAVE_MOTO = True
except Exception:  # pragma: no cover — env without moto
    _HAVE_MOTO = False

from demo_loader import normalize_demo_dict
from demo_store import DemoStore, _from_ddb

DEMOS_TABLE = "voicebot-test-demos"


# A demo dict that exercises every key shape DemoLoader.get produces:
# per-lang system/greeting/kb_intro/kb_ack maps, kb_body as a per-lang map,
# tool_ids (NOT tools), mcp_servers, a filler block with a FLOAT probability,
# and tags.
SAMPLE_DEMO = {
    "id": "acme-demo",
    "label": "Acme Demo",
    "lang": "zh-HK",
    "system": {"zh-HK": "你係客服", "en-US": "you are support"},
    "greeting": {"zh-HK": "你好", "en-US": "hello"},
    "kb_intro": {"zh-HK": "以下係資料", "en-US": "here is info"},
    "kb_ack": {"zh-HK": "明白", "en-US": "got it"},
    "kb_body": {"zh-HK": "知識庫內容" * 10, "en-US": "kb body here" * 10},
    "tool_ids": ["end_call"],
    "mcp_servers": ["weather-mcp"],
    "filler": {"enabled": True, "timeout_ms": 1200, "probability": 0.35},
    "tags": ["support", "demo"],
}


@pytest.fixture
def ddb_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("DEMOS_TABLE", DEMOS_TABLE)
    yield


def _create_demos_table(region: str = "us-east-1") -> None:
    ddb = boto3.client("dynamodb", region_name=region)
    ddb.create_table(
        TableName=DEMOS_TABLE,
        BillingMode="PAY_PER_REQUEST",
        AttributeDefinitions=[{"AttributeName": "id", "AttributeType": "S"}],
        KeySchema=[{"AttributeName": "id", "KeyType": "HASH"}],
    )
    ddb.get_waiter("table_exists").wait(TableName=DEMOS_TABLE)


# ---------------------------------------------------------------------------
# Pure helper — runs WITHOUT moto / any live table.
# ---------------------------------------------------------------------------

def test_normalize_helper_produces_tool_ids_key():
    """The shared normalizer's tool-list key is `tool_ids`, never `tools`,
    regardless of whether the raw dict used `tool_ids` or `tools`."""
    out = normalize_demo_dict(SAMPLE_DEMO, validate_tools=False)
    assert "tool_ids" in out
    assert "tools" not in out
    assert out["tool_ids"] == ["end_call"]

    # Raw manifest style (key `tools`) is also accepted and normalized to
    # `tool_ids`.
    raw = dict(SAMPLE_DEMO)
    raw.pop("tool_ids")
    raw["tools"] = ["end_call", "transfer_to_human"]
    out2 = normalize_demo_dict(raw, validate_tools=False)
    assert "tools" not in out2
    assert out2["tool_ids"] == ["end_call", "transfer_to_human"]


def test_from_ddb_decimal_conversion():
    """_from_ddb turns Decimals into int (integral) / float (fractional)."""
    from decimal import Decimal

    src = {"a": Decimal("0.35"), "b": Decimal("1200"), "c": [Decimal("0.5")]}
    out = _from_ddb(src)
    assert out["a"] == 0.35 and isinstance(out["a"], float)
    assert out["b"] == 1200 and isinstance(out["b"], int)
    assert isinstance(out["c"][0], float)


# ---------------------------------------------------------------------------
# Table-missing degrade — needs moto only to scope the (empty) AWS env, but
# the point is that NO table exists.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAVE_MOTO, reason="moto unavailable")
def test_missing_table_degrades(ddb_env):
    """No table created -> rescan degrades (no raise), get()->None, list()->[],
    last_skipped records the reason; cache stays empty."""
    async def run():
        store = DemoStore(table_name=DEMOS_TABLE)
        n = await store.rescan()
        assert n == 0
        assert store.get("anything") is None
        assert store.list() == []
        assert store.last_skipped
        assert "missing" in store.last_skipped[0]["reason"]

    with mock_aws():
        asyncio.run(run())


# ---------------------------------------------------------------------------
# Empty-cache tolerance BEFORE any rescan (no prewarm) — pure, no AWS.
# ---------------------------------------------------------------------------

def test_empty_cache_before_prewarm_is_safe(ddb_env):
    """get()/list() must not raise on a freshly constructed store (cache empty,
    rescan not yet awaited) — get()->None, list()->[]."""
    store = DemoStore(table_name=DEMOS_TABLE)
    assert store.get("acme-demo") is None
    assert store.list() == []


# ---------------------------------------------------------------------------
# Full round-trip — put -> rescan -> get/list (requires moto table).
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAVE_MOTO, reason="moto unavailable")
def test_put_get_list_roundtrip_with_tool_ids_and_decimal(ddb_env):
    async def run():
        store = DemoStore(table_name=DEMOS_TABLE)
        await store.put(SAMPLE_DEMO)
        n = await store.rescan()
        assert n == 1

        # --- get(): full dict, byte-identical key shape to DemoLoader ---
        demo = store.get("acme-demo")
        assert demo is not None
        # CRITICAL: tool list key is `tool_ids`, not `tools`.
        assert "tool_ids" in demo
        assert "tools" not in demo
        assert demo["tool_ids"] == ["end_call"]
        # Canonical key set remains aligned with DemoLoader's single shared
        # normalizer, including later-added optional model/ASR fields.
        assert set(demo.keys()) == set(
            normalize_demo_dict(SAMPLE_DEMO, validate_tools=False).keys()
        )
        # Per-lang maps survive.
        assert demo["system"]["zh-HK"] == "你係客服"
        assert demo["greeting"]["en-US"] == "hello"
        assert demo["kb_intro"]["zh-HK"] == "以下係資料"
        assert demo["kb_ack"]["en-US"] == "got it"
        assert isinstance(demo["kb_body"], dict)
        assert demo["mcp_servers"] == ["weather-mcp"]
        assert demo["tags"] == ["support", "demo"]

        # --- Decimal -> float round-trip: filler.probability is a float ---
        assert demo["filler"]["probability"] == 0.35
        assert isinstance(demo["filler"]["probability"], float)
        # timeout_ms (integral) comes back as int, not Decimal.
        assert demo["filler"]["timeout_ms"] == 1200
        assert isinstance(demo["filler"]["timeout_ms"], int)
        assert demo["filler"]["enabled"] is True

        # --- list(): summary shape {id,label,lang,kb_chars} ---
        summaries = store.list()
        assert len(summaries) == 1
        s = summaries[0]
        assert set(s.keys()) == {"id", "label", "lang", "kb_chars"}
        assert s["id"] == "acme-demo"
        assert s["label"] == "Acme Demo"
        assert s["lang"] == "zh-HK"
        # kb_chars = sum over the per-lang kb_body variants.
        expected_kb = len(SAMPLE_DEMO["kb_body"]["zh-HK"]) + len(
            SAMPLE_DEMO["kb_body"]["en-US"]
        )
        assert s["kb_chars"] == expected_kb

    with mock_aws():
        _create_demos_table()
        asyncio.run(run())


@pytest.mark.skipif(not _HAVE_MOTO, reason="moto unavailable")
def test_rescan_scans_all_items(ddb_env):
    """rescan loads every demo in the table into the cache."""
    async def run():
        store = DemoStore(table_name=DEMOS_TABLE)
        for i in range(5):
            d = dict(SAMPLE_DEMO)
            d["id"] = f"demo-{i}"
            d["label"] = f"Demo {i}"
            await store.put(d)
        n = await store.rescan()
        assert n == 5
        ids = sorted(x["id"] for x in store.list())
        assert ids == [f"demo-{i}" for i in range(5)]
        for i in range(5):
            assert store.get(f"demo-{i}") is not None

    with mock_aws():
        _create_demos_table()
        asyncio.run(run())


@pytest.mark.skipif(not _HAVE_MOTO, reason="moto unavailable")
def test_rescan_degrade_keeps_prior_cache(ddb_env):
    """If a later rescan hits a missing table, the prior cache is KEPT (not
    wiped) and the previous count is returned."""
    async def run():
        store = DemoStore(table_name=DEMOS_TABLE)
        await store.put(SAMPLE_DEMO)
        assert await store.rescan() == 1
        # Point the store at a non-existent table and rescan again.
        store._table = None
        store._table_name = "no-such-demos-table"
        n = await store.rescan()
        assert n == 1  # prior count preserved
        assert store.get("acme-demo") is not None  # cache intact
        assert store.last_skipped

    with mock_aws():
        _create_demos_table()
        asyncio.run(run())
