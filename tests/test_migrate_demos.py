"""Tests for scripts/migrate_demos_to_ddb.py (T4) — one-shot idempotent
migration of disk demos (data/) into the DynamoDB DemosTable via DemoStore.

Coverage:
  - Full migration of a sample data dir -> DemosTable items match
    DemoLoader.get output INCLUDING the ``tool_ids`` key (NOT ``tools``) and
    system / greeting / kb_body / kb_intro / kb_ack / mcp_servers / filler /
    tags.
  - Idempotency: a second run skips ids already present (no duplicate insert,
    no mutation).
  - ``--overwrite`` (overwrite=True) force-refreshes an existing id.
  - moto unavailable -> fall back to the pure "demo dict -> ddb item"
    conversion (demo_loader.normalize_demo_dict + bot._to_ddb) asserting the
    ``tool_ids`` key is preserved.

DynamoDB is mocked with moto (mock_aws). The pure-conversion fallback runs
standalone (no live table) so this file is meaningful even without moto.
"""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap

import boto3
import pytest

try:
    from moto import mock_aws
    _HAVE_MOTO = True
except Exception:  # pragma: no cover — env without moto
    _HAVE_MOTO = False

# Make the project root + scripts importable regardless of cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
for _p in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from demo_loader import DemoLoader, normalize_demo_dict  # noqa: E402
from demo_store import DemoStore  # noqa: E402
import migrate_demos_to_ddb  # noqa: E402

DEMOS_TABLE = "voicebot-test-migrate-demos"


# ---------------------------------------------------------------------------
# Sample on-disk data dir (a fixture demo with tool_ids + mcp_servers + filler
# + per-lang kb + tags) so the assertions don't depend on the real data/.
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data_root(tmp_path):
    """Build a tmp data/ with one rich demo (alpha) + one minimal demo (beta)."""
    root = tmp_path / "data"
    # --- alpha: exercises every field shape ---
    alpha = root / "alpha"
    alpha.mkdir(parents=True)
    (alpha / "kb.zh.md").write_text("知識庫內容" * 20, encoding="utf-8")
    (alpha / "kb.en.md").write_text("kb body here" * 20, encoding="utf-8")
    (alpha / "manifest.yaml").write_text(
        textwrap.dedent(
            """\
            id: alpha
            label: Alpha Demo
            lang: zh-HK
            tags: [support, demo]
            tools: [end_call]
            mcp_servers: [weather-mcp]
            kb_path:
              zh-HK: kb.zh.md
              en-US: kb.en.md
            system:
              zh-HK: 你係客服
              en-US: you are support
            greeting:
              zh-HK: 你好
              en-US: hello
            kb_intro:
              zh-HK: 以下係資料
              en-US: here is info
            kb_ack:
              zh-HK: 明白
              en-US: got it
            filler:
              enabled: true
              timeout_ms: 1200
              probability: 0.35
            """
        ),
        encoding="utf-8",
    )
    # --- beta: minimal (no tools / kb / filler) ---
    beta = root / "beta"
    beta.mkdir(parents=True)
    (beta / "manifest.yaml").write_text(
        textwrap.dedent(
            """\
            id: beta
            label: Beta Demo
            lang: en-US
            system:
              en-US: minimal system
            greeting:
              en-US: hi
            """
        ),
        encoding="utf-8",
    )
    return str(root)


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


CANONICAL_KEYS = set(
    normalize_demo_dict({"id": "schema"}, validate_tools=False).keys()
)


# ---------------------------------------------------------------------------
# Pure conversion — runs WITHOUT moto / any live table.
# ---------------------------------------------------------------------------

def test_pure_demo_dict_to_ddb_item_preserves_tool_ids():
    """The demo dict -> ddb item path (normalize_demo_dict + bot._to_ddb) keeps
    the ``tool_ids`` key (never ``tools``)."""
    from bot import _to_ddb

    raw = {
        "id": "alpha",
        "label": "Alpha",
        "lang": "zh-HK",
        "tools": ["end_call"],  # raw manifest key
        "filler": {"enabled": True, "timeout_ms": 1200, "probability": 0.35},
    }
    canonical = normalize_demo_dict(raw, validate_tools=False)
    assert "tool_ids" in canonical and "tools" not in canonical
    assert canonical["tool_ids"] == ["end_call"]

    item = _to_ddb(canonical)
    assert "tool_ids" in item and "tools" not in item
    assert item["tool_ids"] == ["end_call"]


# ---------------------------------------------------------------------------
# Full migration round-trip (requires moto table).
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAVE_MOTO, reason="moto unavailable")
def test_migrate_writes_all_fields_matching_loader(ddb_env, sample_data_root):
    """Migrate sample data/ -> DemosTable items match DemoLoader.get output,
    including tool_ids + system/greeting/kb_body/kb_intro/kb_ack/mcp_servers/
    filler/tags."""
    async def run():
        store = DemoStore(table_name=DEMOS_TABLE)
        summary = await migrate_demos_to_ddb.migrate(
            data_root=sample_data_root, store=store, overwrite=False
        )
        assert summary["total"] == 2
        assert sorted(summary["migrated"]) == ["alpha", "beta"]
        assert summary["overwritten"] == []
        assert summary["skipped"] == []

        # Read back through the store cache and compare to DemoLoader.get.
        await store.rescan()
        loader = DemoLoader(sample_data_root)

        for demo_id in ("alpha", "beta"):
            stored = store.get(demo_id)
            expected = loader.get(demo_id)
            assert stored is not None
            assert expected is not None
            # DemoLoader.get adds a filesystem-only `_dir` key; the canonical
            # store shape does not have it. Compare on the canonical key set.
            assert set(stored.keys()) == CANONICAL_KEYS
            for k in CANONICAL_KEYS:
                assert stored[k] == expected[k], f"{demo_id}.{k} mismatch"

        # Spotlight the BLOCKER-#1 invariants on the rich demo.
        alpha = store.get("alpha")
        assert "tool_ids" in alpha and "tools" not in alpha
        assert alpha["tool_ids"] == ["end_call"]
        assert alpha["mcp_servers"] == ["weather-mcp"]
        assert alpha["tags"] == ["support", "demo"]
        assert alpha["system"]["zh-HK"] == "你係客服"
        assert alpha["greeting"]["en-US"] == "hello"
        assert alpha["kb_intro"]["zh-HK"] == "以下係資料"
        assert alpha["kb_ack"]["en-US"] == "got it"
        assert isinstance(alpha["kb_body"], dict)
        assert alpha["kb_body"]["zh-HK"] == "知識庫內容" * 20
        # filler.probability is a float (Decimal round-trip), not a Decimal.
        assert alpha["filler"]["probability"] == 0.35
        assert isinstance(alpha["filler"]["probability"], float)
        assert alpha["filler"]["timeout_ms"] == 1200

    with mock_aws():
        _create_demos_table()
        asyncio.run(run())


@pytest.mark.skipif(not _HAVE_MOTO, reason="moto unavailable")
def test_migrate_is_idempotent_skips_existing(ddb_env, sample_data_root):
    """A second run skips ids already in the table — no duplicate insert."""
    async def run():
        store = DemoStore(table_name=DEMOS_TABLE)
        first = await migrate_demos_to_ddb.migrate(
            data_root=sample_data_root, store=store, overwrite=False
        )
        assert sorted(first["migrated"]) == ["alpha", "beta"]

        second = await migrate_demos_to_ddb.migrate(
            data_root=sample_data_root, store=store, overwrite=False
        )
        assert second["migrated"] == []
        assert second["overwritten"] == []
        assert sorted(s["id"] for s in second["skipped"]) == ["alpha", "beta"]
        assert all(s["reason"] == "exists" for s in second["skipped"])

        # Table still has exactly 2 items (no dup rows under same key).
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        scanned = ddb.scan(TableName=DEMOS_TABLE)
        assert scanned["Count"] == 2

    with mock_aws():
        _create_demos_table()
        asyncio.run(run())


@pytest.mark.skipif(not _HAVE_MOTO, reason="moto unavailable")
def test_migrate_overwrite_refreshes_existing(ddb_env, sample_data_root):
    """--overwrite (overwrite=True) force-replaces an existing id with the
    current disk content."""
    async def run():
        store = DemoStore(table_name=DEMOS_TABLE)
        await migrate_demos_to_ddb.migrate(
            data_root=sample_data_root, store=store, overwrite=False
        )
        # migrate() does not refresh the cache after its writes; rescan to read.
        await store.rescan()

        # Mutate the stored alpha item directly so we can prove the overwrite
        # actually rewrites it from disk.
        tampered = dict(store.get("alpha"))
        tampered["label"] = "TAMPERED"
        await store.put(tampered)
        await store.rescan()
        assert store.get("alpha")["label"] == "TAMPERED"

        # Overwrite run rewrites alpha (and beta) from disk -> label restored.
        result = await migrate_demos_to_ddb.migrate(
            data_root=sample_data_root, store=store, overwrite=True
        )
        assert result["migrated"] == []
        assert sorted(result["overwritten"]) == ["alpha", "beta"]
        assert result["skipped"] == []

        await store.rescan()
        assert store.get("alpha")["label"] == "Alpha Demo"
        # Still exactly 2 items.
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        assert ddb.scan(TableName=DEMOS_TABLE)["Count"] == 2

    with mock_aws():
        _create_demos_table()
        asyncio.run(run())


@pytest.mark.skipif(not _HAVE_MOTO, reason="moto unavailable")
def test_main_cli_entrypoint_runs(ddb_env, sample_data_root):
    """The argparse main() runs end-to-end against a sample dir and exits 0."""
    with mock_aws():
        _create_demos_table()
        rc = migrate_demos_to_ddb.main(
            ["--path", sample_data_root, "--table", DEMOS_TABLE]
        )
        assert rc == 0
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        assert ddb.scan(TableName=DEMOS_TABLE)["Count"] == 2
