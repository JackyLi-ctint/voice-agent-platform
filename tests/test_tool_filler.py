"""Tests for tool_filler (proposal c4890047, T1).

Two layers:

* **Unit (logic):** ``filler_text`` resolution (per-tool hit / ``_default``
  fallback / language→en-US fallback); ``with_filler`` ordering with a mock
  inner + fake params (filler pushed BEFORE inner, with the right
  ``TTSSpeakFrame.text`` and ``append_to_context is False``); disabled →
  identity; filler push raising → inner still called.

* **Integration (AC#4, the round-1 BLOCKER):** drive the ``with_filler``
  wrapper against a **real** pipecat ``LLMService`` wired into a **real**
  pipeline whose sink records the frames it actually receives — ``push_frame``
  is NOT mocked. Asserts the downstream processor really gets a
  ``TTSSpeakFrame`` (correct text / ``append_to_context``) and that it arrives
  before the inner (MCP) call runs — proving the ``FunctionCallParams.llm`` +
  ``push_frame`` cross-module contract holds for real.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipecat.frames.frames import Frame, TextFrame, TTSSpeakFrame  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402
from pipecat.processors.frame_processor import FrameDirection  # noqa: E402
from pipecat.services.llm_service import FunctionCallParams, LLMService  # noqa: E402
from pipecat.services.settings import LLMSettings  # noqa: E402

import tool_filler  # noqa: E402
from tool_filler import (  # noqa: E402
    DEFAULT_LANG_FALLBACK,
    TOOL_ACKS,
    filler_enabled,
    filler_text,
    with_filler,
)


# --------------------------------------------------------------------------
# filler_text — three resolution paths (AC#1)
# --------------------------------------------------------------------------

def test_filler_text_per_tool_hit():
    # verifyCustomer in zh-CN → its dedicated text, not the _default.
    assert filler_text("verifyCustomer", "zh-CN") == "好的，正在为您核验身份…"
    assert filler_text("verifyCustomer", "zh-CN") != TOOL_ACKS["zh-CN"]["_default"]
    # all six connect-repair tools resolve to a non-default, language-correct string
    for tool in (
        "verifyCustomer",
        "verifyCustomerByPhoneAndName",
        "requestRepair",
        "trackRepair",
        "cancelRepair",
        "faqSearch",
    ):
        assert filler_text(tool, "en-US") == TOOL_ACKS["en-US"][tool]


def test_filler_text_unknown_tool_falls_back_to_lang_default():
    # tool not in the table → that language's _default (NOT en-US).
    assert filler_text("noSuchTool", "zh-CN") == TOOL_ACKS["zh-CN"]["_default"]
    assert filler_text("noSuchTool", "zh-HK") == TOOL_ACKS["zh-HK"]["_default"]


def test_filler_text_unknown_lang_falls_back_to_en_us_table():
    # language absent entirely → en-US table is used.
    assert filler_text("verifyCustomer", "ja-JP") == TOOL_ACKS[DEFAULT_LANG_FALLBACK][
        "verifyCustomer"
    ]
    # unknown tool AND unknown lang → en-US _default.
    assert filler_text("noSuchTool", "ja-JP") == TOOL_ACKS[DEFAULT_LANG_FALLBACK][
        "_default"
    ]


def test_tool_acks_covers_three_langs_and_six_tools():
    assert set(TOOL_ACKS) == {"zh-CN", "zh-HK", "en-US"}
    six = {
        "verifyCustomer",
        "verifyCustomerByPhoneAndName",
        "requestRepair",
        "trackRepair",
        "cancelRepair",
        "faqSearch",
    }
    for lang, table in TOOL_ACKS.items():
        assert "_default" in table, lang
        assert six.issubset(set(table)), lang


# --------------------------------------------------------------------------
# filler_enabled — env gate (AC#3)
# --------------------------------------------------------------------------

def test_filler_enabled_default_true(monkeypatch):
    monkeypatch.delenv("TOOL_FILLER_ENABLED", raising=False)
    assert filler_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "False", "no", "off", "", " OFF "])
def test_filler_enabled_falsy_values_disable(monkeypatch, val):
    monkeypatch.setenv("TOOL_FILLER_ENABLED", val)
    assert filler_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "True", "yes", "on", "anything"])
def test_filler_enabled_truthy_values_enable(monkeypatch, val):
    monkeypatch.setenv("TOOL_FILLER_ENABLED", val)
    assert filler_enabled() is True


# --------------------------------------------------------------------------
# with_filler — ordering / payload with a mock inner + fake params (AC#2)
# --------------------------------------------------------------------------

def _fake_params(calls):
    """Fake FunctionCallParams whose llm.push_frame records into `calls`."""

    async def _push_frame(frame, *a, **k):
        calls.append(("push", frame))

    llm = SimpleNamespace(push_frame=_push_frame)
    return SimpleNamespace(llm=llm, result_callback=AsyncMock())


def test_with_filler_pushes_ttsspeak_before_inner(monkeypatch):
    monkeypatch.setenv("TOOL_FILLER_ENABLED", "true")
    calls = []

    async def inner(params):
        calls.append(("inner", params))

    wrapped = with_filler(inner, "verifyCustomer", "zh-CN")
    assert wrapped is not inner  # enabled → a new wrapper

    params = _fake_params(calls)
    asyncio.run(wrapped(params))

    # ordering: filler push first, inner (MCP) second.
    assert [c[0] for c in calls] == ["push", "inner"]
    pushed_frame = calls[0][1]
    assert isinstance(pushed_frame, TTSSpeakFrame)
    assert pushed_frame.text == "好的，正在为您核验身份…"
    assert pushed_frame.append_to_context is False


def test_with_filler_disabled_returns_inner_unchanged_no_push(monkeypatch):
    monkeypatch.setenv("TOOL_FILLER_ENABLED", "false")
    calls = []

    async def inner(params):
        calls.append(("inner", params))

    wrapped = with_filler(inner, "verifyCustomer", "zh-CN")
    assert wrapped is inner  # disabled → identity, no wrapping

    params = _fake_params(calls)
    asyncio.run(wrapped(params))
    # inner ran, nothing was pushed.
    assert [c[0] for c in calls] == ["inner"]


def test_with_filler_push_failure_does_not_block_inner(monkeypatch):
    monkeypatch.setenv("TOOL_FILLER_ENABLED", "true")
    calls = []

    async def boom(frame, *a, **k):
        calls.append(("push-attempt", frame))
        raise RuntimeError("tts down")

    async def inner(params):
        calls.append(("inner", params))

    params = SimpleNamespace(llm=SimpleNamespace(push_frame=boom))
    wrapped = with_filler(inner, "requestRepair", "en-US")
    # must NOT raise — filler failure is swallowed.
    asyncio.run(wrapped(params))

    # the push was attempted and failed, but inner still ran afterwards.
    assert [c[0] for c in calls] == ["push-attempt", "inner"]


# --------------------------------------------------------------------------
# INTEGRATION — real pipecat LLMService + real downstream link (AC#4)
# --------------------------------------------------------------------------

class _SentinelFrame(Frame):
    """A do-nothing frame used to trigger the wrapper from inside the pipeline."""


class _DriverLLMService(LLMService):
    """Smallest real pipecat LLMService that can sit in a real pipeline.

    No inference — we only exercise the real ``push_frame`` path. Settings
    mirror the pattern in pipecat's own tests (tests/test_app_resources.py,
    tests/test_llm_service.py).

    When it processes a ``_SentinelFrame`` it runs the ``with_filler``-wrapped
    wrapper, building a REAL ``FunctionCallParams`` whose ``.llm`` is *this*
    in-pipeline service. So both the filler ``push_frame`` and the inner (MCP)
    marker frame are pushed downstream by the real linked processor — proving
    the cross-module contract end-to-end.
    """

    def __init__(self, *, tool_name: str, lang_key: str, inner_order: dict, **kwargs):
        settings = LLMSettings(
            model="test-model",
            system_instruction=None,
            temperature=None,
            max_tokens=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=None,
            user_turn_completion_config=None,
        )
        super().__init__(settings=settings, **kwargs)
        self._tool_name = tool_name
        self._lang_key = lang_key
        self._inner_order = inner_order

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, _SentinelFrame):
            inner_order = self._inner_order

            async def inner(params):
                inner_order["ran"] = True
                # inner (the real MCP call) pushes a distinct marker downstream
                # via the SAME real in-pipeline llm.
                await params.llm.push_frame(TextFrame(text="__INNER_MARKER__"))

            wrapped = with_filler(inner, self._tool_name, self._lang_key)
            params = FunctionCallParams(
                function_name=self._tool_name,
                tool_call_id="call-1",
                arguments={},
                llm=self,  # REAL in-pipeline LLMService — push_frame not mocked
                context=LLMContext(),
                pipeline_worker=None,
                result_callback=AsyncMock(),
            )
            await wrapped(params)
        else:
            await self.push_frame(frame, direction)


def test_integration_real_push_frame_reaches_downstream_before_inner(monkeypatch):
    """Drive the ``with_filler`` wrapper with a REAL FunctionCallParams whose
    ``.llm`` is a REAL pipecat LLMService linked (via a real pipeline + the
    official run_test harness) to a recording sink. ``push_frame`` is NOT
    mocked. Assert the sink ACTUALLY receives the filler TTSSpeakFrame (correct
    text / append_to_context) and that it arrives BEFORE the inner (MCP) marker
    frame — proving the FunctionCallParams.llm + push_frame cross-module
    contract holds for real."""
    monkeypatch.setenv("TOOL_FILLER_ENABLED", "true")

    from pipecat.tests.utils import run_test  # noqa: E402

    inner_order: dict = {}
    llm = _DriverLLMService(
        tool_name="verifyCustomer", lang_key="zh-CN", inner_order=inner_order
    )

    async def _run():
        # run_test wraps `llm` as source -> llm -> sink in a REAL pipeline and
        # returns exactly the frames the downstream sink received. The sentinel
        # triggers the wrapper from inside the running pipeline.
        down, _up = await run_test(
            llm,
            frames_to_send=[_SentinelFrame()],
            ignore_start=True,
        )
        return down

    frames = asyncio.run(asyncio.wait_for(_run(), timeout=30))

    assert inner_order.get("ran") is True, "inner (MCP) wrapper never ran"

    tts = [f for f in frames if isinstance(f, TTSSpeakFrame)]
    markers = [
        i for i, f in enumerate(frames)
        if isinstance(f, TextFrame) and getattr(f, "text", None) == "__INNER_MARKER__"
    ]
    assert len(tts) == 1, f"expected exactly one TTSSpeakFrame, got {frames!r}"
    assert tts[0].text == "好的，正在为您核验身份…"
    assert tts[0].append_to_context is False
    assert markers, "inner marker frame never reached the sink"

    # ordering: the filler TTSSpeakFrame arrives at the sink BEFORE the inner
    # (MCP) marker frame.
    tts_idx = frames.index(tts[0])
    assert tts_idx < markers[0], (
        f"filler did not arrive before inner: {[type(f).__name__ for f in frames]}"
    )
