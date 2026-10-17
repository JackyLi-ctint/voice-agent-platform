"""Real-MCP-server integration tests for the double-engine MCP wiring (T5).

Unlike ``tests/test_mcp_runtime.py`` (which mocks ``MCPClient``), these tests
exercise the *real* ``pipecat.services.mcp_service.MCPClient`` over a *real*
network hop (localhost) against a *real* ``FastMCP`` streamable-http server.
That makes them the strongest non-live evidence that the schema discovery and
tool-call round-trip actually work end to end:

- ``_connect_mcp_clients`` opens a real session and discovers real tool schemas
- ``_merge_mcp_tools`` keeps ``get_weather`` and *skips* the server's ``end_call``
  tool (name collision with the call-control registry — registry wins)
- the kept tool's ``client._tool_wrapper`` — the exact callable that both
  ``_build_pipeline`` and ``_build_nova_sonic_pipeline`` pass to
  ``llm.register_function`` — round-trips an actual ``tools/call`` to the server
  and the server's response ("Sunny in Tokyo") comes back through the pipecat
  result callback. This is the path the LLM drives at runtime.

The mock server is spun up in a subprocess on a free port and torn down in a
``finally``. If the ``mcp`` package is missing or the server can't open its port
the tests ``skip`` cleanly — they never fail CI because a port is taken.
"""

from __future__ import annotations

import importlib
import socket
import subprocess
import sys
import textwrap
import time

import pytest


# Tool body the server returns — asserted verbatim downstream so we know the
# value travelled the full client -> server -> client round-trip.
WEATHER_ANSWER_TMPL = "Sunny in {city}, 22C"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def bot_mod(monkeypatch):
    """Fresh bot.py with the env its module-level singletons need."""
    monkeypatch.setenv("MINIMAX_API_KEY", "x")
    monkeypatch.setenv("ADMIN_PASSWORD", "test-pwd")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    for mod in list(sys.modules):
        if mod in ("bot", "runtime_config", "demo_loader", "mcp_config"):
            del sys.modules[mod]
    return importlib.import_module("bot")


@pytest.fixture
def mock_mcp_server(tmp_path):
    """A real FastMCP streamable-http server exposing three tools.

    - ``get_weather(city)`` -> a fixed, assertable string
    - ``add(a, b)``         -> arithmetic (second non-colliding tool)
    - ``end_call()``        -> *deliberately* collides with the call-control
                               registry tool name so we can prove it is skipped

    Yields the server id and the ``http://127.0.0.1:<port>/mcp`` URL. Skips the
    test cleanly if the ``mcp`` package is absent or the port never opens.
    """
    pytest.importorskip("mcp.server.fastmcp")

    port = _free_port()
    server_py = tmp_path / "mock_mcp_server.py"
    server_py.write_text(textwrap.dedent(f"""\
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("t5-mock", host="127.0.0.1", port={port})

        @mcp.tool()
        def get_weather(city: str) -> str:
            \"\"\"Return the current weather for a city.\"\"\"
            return {WEATHER_ANSWER_TMPL!r}.format(city=city)

        @mcp.tool()
        def add(a: int, b: int) -> int:
            \"\"\"Add two integers.\"\"\"
            return a + b

        @mcp.tool()
        def end_call() -> str:
            \"\"\"Collides with the call-control registry tool; must be skipped.\"\"\"
            return "mcp end_call should never be reachable"

        mcp.run(transport="streamable-http")
    """))

    proc = subprocess.Popen(
        [sys.executable, str(server_py)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                    break
            except OSError:
                if proc.poll() is not None:
                    pytest.skip("mock MCP server process exited before opening its port")
                time.sleep(0.1)
        else:
            pytest.skip("mock MCP server did not open its port within 10s")
        yield {"id": "t5-weather", "url": f"http://127.0.0.1:{port}/mcp"}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _register_demo_with_server(bot_mod, tmp_path, server):
    """Register the server in MCP_CONFIG and point a fresh demo at it."""
    from demo_loader import DemoLoader

    bot_mod.MCP_CONFIG.upsert(
        {
            "id": server["id"],
            "label": "T5 Weather",
            "transport": "streamable_http",
            "url": server["url"],
            "headers": {},
            "enabled": True,
        }
    )
    data_root = tmp_path / "data"
    demo_dir = data_root / "weatherdemo"
    demo_dir.mkdir(parents=True)
    (demo_dir / "kb.md").write_text("kb")
    (demo_dir / "manifest.yaml").write_text(
        "id: weatherdemo\nlabel: Weather Demo\nlang: en-US\n"
        "system:\n  en-US: 'sys'\n"
        "greeting:\n  en-US: 'hi'\n"
        "kb_path: kb.md\n"
        f"mcp_servers:\n  - {server['id']}\n"
    )
    bot_mod.DEMO_LOADER = DemoLoader(str(data_root))
    return "weatherdemo"


# ---------------------------------------------------------------------------
# 1. Real connect + schema discovery
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_connect_discovers_real_tool_schemas(bot_mod, tmp_path, mock_mcp_server, monkeypatch):
    """_connect_mcp_clients opens a real session and discovers real schemas."""
    monkeypatch.setenv("MCP_CFG_PATH_OVERRIDE", str(tmp_path / "mcp.json"))
    bot_mod.MCP_CONFIG = bot_mod.McpConfig(str(tmp_path / "mcp.json"))
    scenario = _register_demo_with_server(bot_mod, tmp_path, mock_mcp_server)

    clients, schemas = await bot_mod._connect_mcp_clients(scenario, scope="web")
    try:
        assert len(clients) == 1
        assert len(schemas) == 1
        _client, fns = schemas[0]
        names = sorted(fs.name for fs in fns)
        assert names == ["add", "end_call", "get_weather"]
    finally:
        for c in clients:
            await c.close()


# ---------------------------------------------------------------------------
# 2. Merge: registry wins over the colliding MCP end_call; tool-call round-trip
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_merge_skips_conflicting_end_call_and_roundtrips(
    bot_mod, tmp_path, mock_mcp_server, monkeypatch
):
    """The full runtime path on a real server:

    * connect -> discover [add, end_call, get_weather]
    * merge with a registry that owns ``end_call`` -> MCP end_call dropped,
      ``add`` + ``get_weather`` kept
    * call ``get_weather`` through ``client._tool_wrapper`` (the exact callable
      registered on the LLM) and assert the server's answer comes back.
    """
    monkeypatch.setenv("MCP_CFG_PATH_OVERRIDE", str(tmp_path / "mcp.json"))
    bot_mod.MCP_CONFIG = bot_mod.McpConfig(str(tmp_path / "mcp.json"))
    scenario = _register_demo_with_server(bot_mod, tmp_path, mock_mcp_server)

    from pipecat.adapters.schemas.function_schema import FunctionSchema
    from pipecat.adapters.schemas.tools_schema import ToolsSchema

    # Registry owns end_call (the call-control kill switch / hangup tool).
    registry = ToolsSchema(
        standard_tools=[
            FunctionSchema(name="end_call", description="hang up", properties={}, required=[])
        ]
    )

    clients, schemas = await bot_mod._connect_mcp_clients(scenario, scope="web")
    try:
        combined, kept = bot_mod._merge_mcp_tools(registry, schemas)

        merged_names = sorted(fs.name for fs in combined.standard_tools)
        # end_call appears exactly once (the registry one); MCP's was skipped.
        assert merged_names == ["add", "end_call", "get_weather"]
        kept_names = sorted(fs.name for _c, fs in kept)
        assert kept_names == ["add", "get_weather"]  # MCP end_call NOT kept
        assert "end_call" not in kept_names

        # --- real tool-call round-trip through the registered wrapper ---
        client_for_weather = next(c for c, fs in kept if fs.name == "get_weather")
        from pipecat.services.llm_service import FunctionCallParams

        captured = {}

        async def result_callback(result, *args, **kwargs):
            captured["result"] = result

        params = FunctionCallParams(
            function_name="get_weather",
            tool_call_id="call-1",
            arguments={"city": "Tokyo"},
            llm=None,
            context=None,
            pipeline_worker=None,
            result_callback=result_callback,
        )
        await client_for_weather._tool_wrapper(params)

        assert "result" in captured, "result_callback was never invoked"
        assert "Sunny in Tokyo" in str(captured["result"])
    finally:
        for c in clients:
            await c.close()


# ---------------------------------------------------------------------------
# 3. Degrade path: an unreachable server is skipped, the call is not aborted
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_unreachable_server_is_skipped_not_fatal(bot_mod, tmp_path, monkeypatch):
    """A demo pointing at a dead URL yields ([], []) — no exception, no abort."""
    pytest.importorskip("pipecat.services.mcp_service")
    monkeypatch.setenv("MCP_CFG_PATH_OVERRIDE", str(tmp_path / "mcp.json"))
    bot_mod.MCP_CONFIG = bot_mod.McpConfig(str(tmp_path / "mcp.json"))

    from demo_loader import DemoLoader

    bot_mod.MCP_CONFIG.upsert(
        {
            "id": "dead-srv",
            "label": "Dead",
            "transport": "streamable_http",
            # 127.0.0.1:9 is the discard port; nothing listens -> fast refuse.
            "url": "http://127.0.0.1:9/mcp",
            "headers": {},
            "enabled": True,
        }
    )
    data_root = tmp_path / "data"
    demo_dir = data_root / "deaddemo"
    demo_dir.mkdir(parents=True)
    (demo_dir / "kb.md").write_text("kb")
    (demo_dir / "manifest.yaml").write_text(
        "id: deaddemo\nlabel: Dead Demo\nlang: en-US\n"
        "system:\n  en-US: 'sys'\n"
        "greeting:\n  en-US: 'hi'\n"
        "kb_path: kb.md\n"
        "mcp_servers:\n  - dead-srv\n"
    )
    bot_mod.DEMO_LOADER = DemoLoader(str(data_root))

    clients, schemas = await bot_mod._connect_mcp_clients("deaddemo", scope="web")
    assert clients == []
    assert schemas == []

    # And the merge path on an empty discovery returns the no-MCP result.
    combined, kept = bot_mod._merge_mcp_tools(None, schemas)
    assert combined is None
    assert kept == []


# ---------------------------------------------------------------------------
# 3b. SigV4 degrade path: missing AWS credentials -> server skipped, no crash
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_sigv4_missing_credentials_is_skipped_not_fatal(bot_mod, tmp_path, monkeypatch):
    """A sigv4 server whose credential chain is empty is skipped with a
    WARNING — ``_connect_mcp_clients`` returns ([], []) and never crashes the
    call. The none/header path is unaffected (separate test below)."""
    pytest.importorskip("pipecat.services.mcp_service")
    monkeypatch.setenv("MCP_CFG_PATH_OVERRIDE", str(tmp_path / "mcp.json"))
    bot_mod.MCP_CONFIG = bot_mod.McpConfig(str(tmp_path / "mcp.json"))

    # Force the boto3 default chain to yield no credentials.
    import boto3

    class _NoCredSession:
        def get_credentials(self):
            return None

    monkeypatch.setattr(boto3, "Session", lambda *a, **k: _NoCredSession())

    from demo_loader import DemoLoader

    bot_mod.MCP_CONFIG.upsert(
        {
            "id": "connect-repair",
            "label": "Connect Repair",
            "transport": "streamable_http",
            "url": "https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/foo/invocations?qualifier=DEFAULT",
            "headers": {},
            "auth": {"type": "sigv4", "service": "bedrock-agentcore", "region": "us-east-1"},
            "enabled": True,
        }
    )
    data_root = tmp_path / "data"
    demo_dir = data_root / "repairdemo"
    demo_dir.mkdir(parents=True)
    (demo_dir / "kb.md").write_text("kb")
    (demo_dir / "manifest.yaml").write_text(
        "id: repairdemo\nlabel: Repair Demo\nlang: zh-CN\n"
        "system:\n  zh-CN: 'sys'\n"
        "greeting:\n  zh-CN: 'hi'\n"
        "kb_path: kb.md\n"
        "mcp_servers:\n  - connect-repair\n"
    )
    bot_mod.DEMO_LOADER = DemoLoader(str(data_root))

    clients, schemas = await bot_mod._connect_mcp_clients("repairdemo", scope="web")
    assert clients == []
    assert schemas == []


# ---------------------------------------------------------------------------
# 4. Nova Sonic tool-result coercion (BLOCKER-2 fix, hermetic)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_nova_mcp_tool_wrapper_coerces_string_result_to_object(bot_mod):
    """_nova_mcp_tool_wrapper wraps a bare-string MCP result as a JSON object.

    Nova Sonic's bidi tool-result protocol (aws/nova_sonic/llm.py _send_tool_result)
    rejects a non-dict result with "Unsupported JSON type in Tool Result", which
    aborts the turn. MCPClient._tool_wrapper hands the result to the callback as a
    bare string, so for the Nova Sonic engine we register this wrapper. Verified
    here without a live Bedrock connection: the wrapper must intercept the
    result_callback and coerce a string into ``{"result": <string>}`` while
    leaving a dict result untouched.
    """

    class FakeClient:
        """Stands in for MCPClient: its _tool_wrapper invokes result_callback
        with whatever the underlying tool returned (a bare string for MCP)."""

        def __init__(self, payload):
            self._payload = payload

        async def _tool_wrapper(self, params):
            await params.result_callback(self._payload)

    class Params:
        def __init__(self, cb):
            self.function_name = "get_weather"
            self.tool_call_id = "abc"
            self.arguments = {"city": "Tokyo"}
            self.result_callback = cb

    # (a) bare string -> {"result": <string>}
    seen = {}

    async def capture(result, **kwargs):
        seen["result"] = result

    wrapped = bot_mod._nova_mcp_tool_wrapper(FakeClient("Sunny in Tokyo"))
    await wrapped(Params(capture))
    assert seen["result"] == {"result": "Sunny in Tokyo"}

    # (b) dict result -> passed through untouched
    seen.clear()
    wrapped = bot_mod._nova_mcp_tool_wrapper(FakeClient({"temp_c": 28}))
    await wrapped(Params(capture))
    assert seen["result"] == {"temp_c": 28}
