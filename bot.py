"""WebSocket-based voice bot.

Client pushes 16 kHz PCM16 mono, server streams 24 kHz PCM16 mono back.
Pipeline: ws_in -> AWS Transcribe -> Bedrock Nova 2 Lite -> MiniMax TTS -> ws_out.
Silero VAD runs in the transport and gates STT / provides barge-in.

Run:
    python bot.py                # http://localhost:7860/
"""

import argparse
import asyncio
import base64
import contextlib
import csv
import io
import json
import os
import time
import xml.sax.saxutils as saxutils
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import aiohttp
import boto3
import secrets as _secrets
import uvicorn
from dotenv import load_dotenv
from fastapi import Cookie, Depends, FastAPI, HTTPException, Query, Request, WebSocket, status
from fastapi.responses import FileResponse, RedirectResponse, Response
from loguru import logger
from starlette.websockets import WebSocketState

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.services.aws.llm import AWSBedrockLLMService
from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService
from pipecat.services.aws.stt import AWSTranscribeSTTService
from pipecat.services.minimax.tts import MiniMaxHttpTTSService, MiniMaxTTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.services.settings import TTSSettings
from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame
from pipecat.utils.tracing.service_decorators import traced_tts
from aiobotocore.session import get_session
from asr_filter import (
    TranscriptHallucinationFilter,
    _env_bool,
    _env_float,
    _env_int,
)
from asr_multilang import build_transcribe_presigned_url
from timeout_filler import TimeoutFillerObserver
from tool_filler import with_filler
from collections.abc import AsyncGenerator
from dataclasses import dataclass
import json as _mm_json
from pipecat.transcriptions.language import Language
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from demo_store import DemoStore
from voice_store import VoiceStore
from activity_store import ActivityStore
import twilio_sig  # X-Twilio-Signature verification (T1; stdlib-only)
from latency_observer import LatencyObserver  # pure-observation latency (T1/T2)
from voice_agent.app import ApplicationSettings, ReadinessState
from voice_agent.engines.nova_sonic import NovaSonicEngineConfig, NovaSonicSessionEngine
from voice_agent.transports.connect_a2a import ConnectA2AAdapter, ConnectA2ASession
from voice_agent.transports.connect_a2a_asgi import handle_connect_a2a_websocket

load_dotenv(override=True)

# Module-level demo store singleton (DDB-backed; replaced the disk-scanning
# DemoLoader in T3). Construction is cheap (no AWS call); the in-memory cache
# is prewarmed by `await DEMO_LOADER.rescan()` in the FastAPI lifespan startup
# hook below. get()/list() stay synchronous + non-blocking so every per-call
# site is unchanged, and tolerate an empty cache (before prewarm / on degrade):
# get() -> None, list() -> []. PATCH (T5) calls rescan() to pick up writes.
DEMO_LOADER = DemoStore()

# On-disk demo seed root (data/<id>/manifest.yaml). As of T5, DynamoDB is the
# single source of truth for BOTH reads and writes: the PATCH endpoint merges
# onto DEMO_LOADER.get() and persists via DEMO_LOADER.put() + rescan() — it no
# longer writes manifest.yaml. This root is now only a one-time migration seed
# (scripts/migrate_demos_to_ddb.py reads it); kept as a module constant so the
# migration path and tests can point it at a chosen data dir.
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Module-level voice registry singleton (DDB-backed; Part A of the
# voice-registry feature). Construction is cheap (lazy table accessor, no AWS
# call); the FastAPI lifespan hook prewarms it via `await VOICE_STORE.rescan()`
# + `seed_if_empty(...)`. Until the VoicesTable exists the store stays NOT live,
# so `voices_for(provider)` (below, after the constant dicts) returns the
# in-code MINIMAX_VOICES / POLLY_VOICES / NOVA_SONIC_VOICES — byte-identical to
# before. See voice_store.py.
VOICE_STORE = VoiceStore()

# Module-level operations-audit singleton (DDB-backed; Part B / T3). Like
# VOICE_STORE, construction is cheap — the boto3 Table is built lazily on first
# use (N3), never eagerly — so building it at import time never touches AWS and
# is safe when the ActivityTable is absent / creds aren't ready. ``log`` is
# best-effort (swallows all failures incl. table-missing), so shipping this is
# zero-impact on prod until the table exists + ACTIVITY_TABLE env is set. The
# lifespan hook does NOT need to prewarm it (no cache). See activity_store.py.
ACTIVITY_STORE = ActivityStore()

# --- Auth ----------------------------------------------------------------
# JWT-session auth (DynamoDB user table + bcrypt) replaced the old shared
# SITE_PASSWORD / ADMIN_PASSWORD Basic Auth. See user_store.py.
#
# Auth deps (bcrypt via user_store, PyJWT here) are imported in a way that a
# missing package degrades only the web auth path — it must NEVER break
# `import bot` or the unauthenticated PSTN /phone/ws bridge. PyJWT is imported
# lazily inside the JWT helpers; a missing module surfaces as 503 on auth
# routes, not an import-time crash.
COOKIE_NAME = "vb_session"

# Session JWT secret. If AUTH_SECRET is unset we generate a random one at
# startup (with a WARNING) so the app still boots — but every restart then
# invalidates all sessions, so prod MUST set AUTH_SECRET explicitly.
AUTH_SECRET = os.environ.get("AUTH_SECRET", "").strip()
if not AUTH_SECRET:
    AUTH_SECRET = _secrets.token_urlsafe(48)
    logger.warning(
        "AUTH_SECRET not set; generated an ephemeral one. All sessions will be "
        "invalidated on restart. Set AUTH_SECRET in the environment for prod."
    )
AUTH_TTL_HOURS = int(os.environ.get("AUTH_TTL_HOURS", "12"))
_JWT_ALG = "HS256"
# Short-lived token used to authorize a browser WebSocket handshake (the WS
# handshake cannot carry the HttpOnly cookie reliably through CloudFront).
_WS_TOKEN_TTL = 60.0


def _jwt_module():
    """Lazy import of PyJWT. Returns None if unavailable (auth disabled)."""
    try:
        import jwt  # type: ignore
        return jwt
    except Exception as e:  # pragma: no cover - exercised only without PyJWT
        logger.warning(f"PyJWT not installed: {e}; web auth disabled")
        return None


def _issue_jwt(user: dict, ttl_hours: int | None = None, extra: dict | None = None) -> str:
    """Sign a session JWT for ``user`` (sub=username, role, exp)."""
    jwt = _jwt_module()
    if jwt is None:
        raise HTTPException(status_code=503, detail="auth unavailable (PyJWT missing)")
    import time as _t
    ttl = AUTH_TTL_HOURS if ttl_hours is None else ttl_hours
    payload = {
        "sub": user["username"],
        "role": user.get("role") or "user",
        "exp": int(_t.time()) + int(ttl * 3600),
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, AUTH_SECRET, algorithm=_JWT_ALG)


def _decode_jwt(token: str) -> dict | None:
    """Decode + verify a session JWT. Returns claims or None (bad/expired)."""
    if not token:
        return None
    jwt = _jwt_module()
    if jwt is None:
        return None
    try:
        return jwt.decode(token, AUTH_SECRET, algorithms=[_JWT_ALG])
    except Exception:
        return None


def _issue_ws_token(username: str, role: str = "user") -> str:
    """Mint a short-TTL JWT (60s) that binds a WebSocket to ``username``."""
    return _issue_jwt(
        {"username": username, "role": role},
        ttl_hours=_WS_TOKEN_TTL / 3600.0,
        extra={"ws": True},
    )


# --- Guest experience links -------------------------------------------------
# An admin mints a short-lived guest JWT (default 1h, ≤24h hard cap) so an
# external person can try the Talk page with no account / password. The token
# is a normal session JWT (signed with AUTH_SECRET) marked guest=True; it is
# stateless (no UserStore row) and self-expiring (exp is the kill switch).
GUEST_TTL_DEFAULT_MIN = 60
GUEST_TTL_MAX_MIN = 1440  # 24h hard cap


def _issue_guest_token(
    ttl_minutes: int,
    scenario: str | None = None,
    *,
    lang: str | None = None,
    engine: str | None = None,
    voice: str | None = None,
    provider: str | None = None,
) -> tuple[str, int]:
    """Return (jwt, exp_epoch). Guest JWT: sub='guest', role='guest', guest=True,
    plus the optional launch-config subset (scenario/lang/engine/voice/provider).

    Each field is embedded into ``extra`` ONLY when truthy, so an old
    scenario-only (or no-arg) token is a strict subset and ``_decode_jwt``
    round-trips exactly what was present. Signed with AUTH_SECRET like every
    other session token."""
    extra = {"guest": True}
    for k, v in (
        ("scenario", scenario),
        ("lang", lang),
        ("engine", engine),
        ("voice", voice),
        ("provider", provider),
    ):
        if v:
            extra[k] = v
    tok = _issue_jwt(
        {"username": "guest", "role": "guest"},
        ttl_hours=ttl_minutes / 60.0,
        extra=extra,
    )
    exp = int(time.time()) + ttl_minutes * 60
    return tok, exp


def _ws_token_claims(websocket: WebSocket) -> dict | None:
    """Validate the ?token=... WS token; return its claims or None.

    Browsers cannot reliably send the HttpOnly cookie / Authorization on a WS
    handshake (CloudFront strips them), so the SPA fetches a short-TTL token
    from GET /api/ws-token (cookie-authenticated) and passes it as ?token=.
    """
    claims = _decode_jwt(websocket.query_params.get("token", ""))
    if not claims or not claims.get("ws"):
        return None
    return claims


def _ws_token_username(websocket: WebSocket) -> str | None:
    """Validate the WS token and return its bound username, or None."""
    claims = _ws_token_claims(websocket)
    return claims.get("sub") if claims else None


INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
# PSTN/Chime phone uplink wire rate: voice-server sends native G.711-decoded
# 8 kHz PCM (no upsample). The pipeline/Transcribe phone path consumes it
# directly; the nova-sonic phone path upsamples 8 kHz → INPUT_SAMPLE_RATE.
PHONE_WIRE_SAMPLE_RATE = 8000

# Languages selectable from the UI. Each entry maps to a Pipecat Language
# enum (for Transcribe) and a localized system instruction for the LLM.
#
# Each entry also carries an ``engines`` list naming which conversation
# engines can serve that language:
#   - "pipeline":   STT (Transcribe) + LLM (Bedrock) + TTS — requires a non-None
#                   ``stt`` Language enum.
#   - "nova-sonic": end-to-end Nova Sonic v2 S2S — does NOT read ``stt`` (the
#                   model handles its own ASR), so Nova-only languages set
#                   ``stt=None``. Nova supports en-US/GB/AU/IN, fr-FR, it-IT,
#                   de-DE, es-US, pt-BR (NOT Cantonese/Mandarin/Japanese).
# ``stt`` is therefore typed ``Language | None``; every pipeline code path that
# reads it must tolerate / never reach None (validation blocks pipeline + a
# stt=None language; build_pipeline_task is additionally defensive).
LANGUAGES: dict[str, dict] = {
    "zh-HK": {
        "label": "粵語 (Cantonese)",
        "stt": Language.ZH_HK,
        "engines": ["pipeline"],
        "prompt": (
            "你係一個友善嘅粵語語音助手。回覆會被朗讀出嚟,唔好用 emoji、列表"
            "或者任何書面格式。答嘢要簡短、自然、口語化,最好一兩句。"
            "一定要用粵語(廣東話)口語回答,唔好用普通話書面語。"
        ),
        "greeting": "用粵語同用戶打個招呼,介紹吓你係邊個,一句話。",
    },
    "zh-CN": {
        "label": "普通话 (Mandarin)",
        "stt": Language.ZH_CN,
        "engines": ["pipeline"],
        "prompt": (
            "你是一个友好的中文语音助手。回复会被朗读出来,不要用 emoji、列表"
            "或任何书面格式。回答要简短、自然、口语化,最好一两句。"
        ),
        "greeting": "用普通话跟用户打个招呼,一句话介绍你是谁。",
    },
    "en-US": {
        "label": "English (US)",
        "stt": Language.EN,
        "engines": ["pipeline", "nova-sonic"],
        "prompt": (
            "You are a friendly voice assistant. Your replies will be spoken aloud, "
            "so avoid emojis, bullet points, or any written formatting. Keep answers "
            "short, natural, and conversational — one or two sentences."
        ),
        "greeting": "Greet the user in English and briefly introduce yourself in one sentence.",
    },
    "ja-JP": {
        "label": "日本語 (Japanese)",
        "stt": Language.JA,
        "engines": ["pipeline"],
        "prompt": (
            "あなたはフレンドリーな音声アシスタントです。返答は読み上げられるので、"
            "絵文字や箇条書き、書き言葉は使わないでください。短く、自然な話し言葉で、"
            "1〜2文で答えてください。"
        ),
        "greeting": "日本語でユーザーに挨拶し、一文で自己紹介してください。",
    },
    "en-SG": {
        "label": "English (Singapore)",
        # Pragmatic: AWS Transcribe streaming has no en-SG LanguageCode, so we
        # route STT through en-US (Singlish vocabulary/accents are within en-US's
        # robustness envelope). Prompt + TTS keep the Singapore English flavor.
        # Nova Sonic has no en-SG locale, so pipeline-only.
        "stt": Language.EN,
        "engines": ["pipeline"],
        "prompt": (
            "You are a friendly voice assistant speaking Singapore English (Singlish). "
            "Your replies will be spoken aloud, so avoid emojis, bullet points, or any "
            "written formatting. Keep answers short, natural, and conversational — one or two sentences. "
            "Light Singlish particles (\"lah\", \"lor\") are fine but not required."
        ),
        "greeting": "Greet the user in Singapore English in one sentence and briefly introduce yourself.",
    },
    # --- Nova Sonic-only languages (stt=None, engines=["nova-sonic"]) -------
    # These exist purely to widen Nova's selectable language set; they never
    # enter the pipeline path (no Transcribe LanguageCode). prompt/greeting are
    # written in-language, matching the voice-assistant tone of the 5 above.
    "en-GB": {
        "label": "English (UK)",
        "stt": None,
        "engines": ["nova-sonic"],
        "prompt": (
            "You are a friendly voice assistant speaking British English. Your replies "
            "will be spoken aloud, so avoid emojis, bullet points, or any written "
            "formatting. Keep answers short, natural, and conversational — one or two sentences."
        ),
        "greeting": "Greet the user in British English and briefly introduce yourself in one sentence.",
    },
    "en-AU": {
        "label": "English (Australia)",
        "stt": None,
        "engines": ["nova-sonic"],
        "prompt": (
            "You are a friendly voice assistant speaking Australian English. Your replies "
            "will be spoken aloud, so avoid emojis, bullet points, or any written "
            "formatting. Keep answers short, natural, and conversational — one or two sentences."
        ),
        "greeting": "Greet the user in Australian English and briefly introduce yourself in one sentence.",
    },
    "en-IN": {
        "label": "English (India)",
        "stt": None,
        "engines": ["nova-sonic"],
        "prompt": (
            "You are a friendly voice assistant speaking Indian English. Your replies "
            "will be spoken aloud, so avoid emojis, bullet points, or any written "
            "formatting. Keep answers short, natural, and conversational — one or two sentences."
        ),
        "greeting": "Greet the user in Indian English and briefly introduce yourself in one sentence.",
    },
    "fr-FR": {
        "label": "Français",
        "stt": None,
        "engines": ["nova-sonic"],
        "prompt": (
            "Vous êtes un assistant vocal amical. Vos réponses seront lues à voix haute, "
            "donc évitez les emojis, les listes à puces ou toute mise en forme écrite. "
            "Restez bref, naturel et conversationnel — une ou deux phrases. Répondez en français."
        ),
        "greeting": "Saluez l'utilisateur en français et présentez-vous brièvement en une phrase.",
    },
    "it-IT": {
        "label": "Italiano",
        "stt": None,
        "engines": ["nova-sonic"],
        "prompt": (
            "Sei un assistente vocale amichevole. Le tue risposte verranno lette ad alta voce, "
            "quindi evita emoji, elenchi puntati o qualsiasi formattazione scritta. "
            "Mantieni le risposte brevi, naturali e colloquiali — una o due frasi. Rispondi in italiano."
        ),
        "greeting": "Saluta l'utente in italiano e presentati brevemente in una frase.",
    },
    "de-DE": {
        "label": "Deutsch",
        "stt": None,
        "engines": ["nova-sonic"],
        "prompt": (
            "Du bist ein freundlicher Sprachassistent. Deine Antworten werden vorgelesen, "
            "vermeide daher Emojis, Aufzählungspunkte oder jegliche schriftliche Formatierung. "
            "Halte die Antworten kurz, natürlich und gesprächig — ein oder zwei Sätze. Antworte auf Deutsch."
        ),
        "greeting": "Begrüße den Benutzer auf Deutsch und stelle dich kurz in einem Satz vor.",
    },
    "es-US": {
        "label": "Español (US)",
        "stt": None,
        "engines": ["nova-sonic"],
        "prompt": (
            "Eres un asistente de voz amigable. Tus respuestas se leerán en voz alta, "
            "así que evita emojis, listas con viñetas o cualquier formato escrito. "
            "Mantén las respuestas breves, naturales y conversacionales — una o dos frases. Responde en español."
        ),
        "greeting": "Saluda al usuario en español y preséntate brevemente en una frase.",
    },
    "pt-BR": {
        "label": "Português (BR)",
        "stt": None,
        "engines": ["nova-sonic"],
        "prompt": (
            "Você é um assistente de voz amigável. Suas respostas serão lidas em voz alta, "
            "então evite emojis, listas com marcadores ou qualquer formatação escrita. "
            "Mantenha as respostas curtas, naturais e coloquiais — uma ou duas frases. Responda em português."
        ),
        "greeting": "Cumprimente o usuário em português e apresente-se brevemente em uma frase.",
    },
}


def _languages_for_engine(engine: str) -> list[str]:
    """Return the LANGUAGES keys whose ``engines`` list includes ``engine``.

    Order follows LANGUAGES insertion order. Used both for filtering options
    payloads and as the source of truth for engine↔lang validation.
    """
    return [k for k, v in LANGUAGES.items() if engine in v.get("engines", [])]

# LLMs selectable from the UI. Values are Bedrock inference-profile IDs.
# Anthropic Claude on Bedrock requires inference profiles (us.* / global.*) —
# bare anthropic.* IDs return ValidationException ("on-demand throughput isn't
# supported"). Stack lives in us-east-1, so we use us.* for lowest latency.
MODELS: dict[str, str] = {
    "nova-2-lite":       "us.amazon.nova-2-lite-v1:0",
    "nova-lite":         "us.amazon.nova-lite-v1:0",
    "nova-micro":        "us.amazon.nova-micro-v1:0",
    "claude-haiku-4.5":  "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-sonnet-4.5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-sonnet-4.6": "us.anthropic.claude-sonnet-4-6",
}


def _is_claude_model_id(model_id: str) -> bool:
    return ".anthropic." in model_id
DEFAULT_LANG = "en-US"
DEFAULT_MODEL = "nova-2-lite"
# Demo prompt translation (admin one-click translate) is a quality-sensitive
# task: the conversational DEFAULT_MODEL (nova-2-lite) tends to echo long
# system-prompt fields back untranslated, and claude-haiku mis-targets the
# language. Default the translate path to a stronger instruction-follower.
# Falls back to DEFAULT_MODEL only if this key is somehow absent from MODELS.
TRANSLATE_MODEL = "claude-sonnet-4.5"

# MiniMax preset voices. Sourced via the MiniMax `get_voice` API against this
# account on 2026-05-12 (filtered to the 4 languages we expose). Critical:
# some Cantonese voice IDs use a FULL-WIDTH left paren `（` together with a
# half-width right paren `)` — copying from the public FAQ as straight ASCII
# yields HTTP 200 + "voice id not exist". The strings below are exactly what
# the API accepts.
_BOOST_YUE = "Chinese,Yue"
_BOOST_ZH = "Chinese"
_BOOST_EN = "English"
_BOOST_JA = "Japanese"

MINIMAX_VOICES: dict[str, dict] = {
    # Cantonese (zh-HK) — note full-width "（" in ProfessionalHost IDs.
    "Cantonese_GentleLady":          {"label": "Gentle Lady",          "gender": "F",   "language": "zh-HK", "boost": _BOOST_YUE},
    "Cantonese_KindWoman":           {"label": "Kind Woman",           "gender": "F",   "language": "zh-HK", "boost": _BOOST_YUE},
    "Cantonese_CuteGirl":            {"label": "Cute Girl",            "gender": "F",   "language": "zh-HK", "boost": _BOOST_YUE},
    "Cantonese_PlayfulMan":          {"label": "Playful Man",          "gender": "M",   "language": "zh-HK", "boost": _BOOST_YUE},
    "Cantonese_ProfessionalHost（F)":{"label": "Professional Host","gender": "F",   "language": "zh-HK", "boost": _BOOST_YUE},
    "Cantonese_ProfessionalHost（M)":{"label": "Professional Host","gender": "M",   "language": "zh-HK", "boost": _BOOST_YUE},

    # Mandarin (zh-CN) — verified against `get_voice` API
    "Chinese (Mandarin)_Gentleman":            {"label": "Gentleman",             "gender": "M",   "language": "zh-CN", "boost": _BOOST_ZH},
    "Chinese (Mandarin)_News_Anchor":          {"label": "News Anchor",           "gender": "F",   "language": "zh-CN", "boost": _BOOST_ZH},
    "Chinese (Mandarin)_Reliable_Executive":   {"label": "Reliable Executive",    "gender": "M",   "language": "zh-CN", "boost": _BOOST_ZH},
    "Chinese (Mandarin)_Male_Announcer":       {"label": "Male Announcer",        "gender": "M",   "language": "zh-CN", "boost": _BOOST_ZH},
    "Chinese (Mandarin)_Radio_Host":           {"label": "Radio Host (M)",        "gender": "M",   "language": "zh-CN", "boost": _BOOST_ZH},
    "Chinese (Mandarin)_Lyrical_Voice":        {"label": "Lyrical Voice (M)",     "gender": "M",   "language": "zh-CN", "boost": _BOOST_ZH},
    "Chinese (Mandarin)_Humorous_Elder":       {"label": "Humorous Elder",        "gender": "M",   "language": "zh-CN", "boost": _BOOST_ZH},
    "Chinese (Mandarin)_Kind-hearted_Elder":   {"label": "Kind-hearted Elder",    "gender": "F",   "language": "zh-CN", "boost": _BOOST_ZH},
    "Chinese (Mandarin)_Gentle_Senior":        {"label": "Gentle Senior",         "gender": "F",   "language": "zh-CN", "boost": _BOOST_ZH},
    "Chinese (Mandarin)_Sincere_Adult":        {"label": "Sincere Adult",         "gender": "M",   "language": "zh-CN", "boost": _BOOST_ZH},
    "Chinese (Mandarin)_Mature_Woman":         {"label": "Mature Woman",          "gender": "F",   "language": "zh-CN", "boost": _BOOST_ZH},
    "Chinese (Mandarin)_Wise_Women":           {"label": "Wise Woman",            "gender": "F",   "language": "zh-CN", "boost": _BOOST_ZH},
    "Chinese (Mandarin)_Warm_Bestie":          {"label": "Warm Bestie",           "gender": "F",   "language": "zh-CN", "boost": _BOOST_ZH},
    "Chinese (Mandarin)_Sweet_Lady":           {"label": "Sweet Lady",            "gender": "F",   "language": "zh-CN", "boost": _BOOST_ZH},
    "Chinese (Mandarin)_Warm_Girl":            {"label": "Warm Girl",             "gender": "F",   "language": "zh-CN", "boost": _BOOST_ZH},
    "Chinese (Mandarin)_Crisp_Girl":           {"label": "Crisp Girl",            "gender": "F",   "language": "zh-CN", "boost": _BOOST_ZH},
    "Chinese (Mandarin)_Soft_Girl":            {"label": "Soft Girl",             "gender": "F",   "language": "zh-CN", "boost": _BOOST_ZH},
    "Chinese (Mandarin)_Cute_Spirit":          {"label": "Cute Spirit",           "gender": "F",   "language": "zh-CN", "boost": _BOOST_ZH},
    "Chinese (Mandarin)_Kind-hearted_Antie":   {"label": "Kind-hearted Auntie",   "gender": "F",   "language": "zh-CN", "boost": _BOOST_ZH},
    "Chinese (Mandarin)_HK_Flight_Attendant":  {"label": "HK Flight Attendant",   "gender": "F",   "language": "zh-CN", "boost": _BOOST_ZH},
    "Chinese (Mandarin)_Stubborn_Friend":      {"label": "Stubborn Friend",       "gender": "M",   "language": "zh-CN", "boost": _BOOST_ZH},
    "Chinese (Mandarin)_Gentle_Youth":         {"label": "Gentle Youth",          "gender": "M",   "language": "zh-CN", "boost": _BOOST_ZH},
    "Chinese (Mandarin)_Southern_Young_Man":   {"label": "Southern Young Man",    "gender": "M",   "language": "zh-CN", "boost": _BOOST_ZH},
    "Chinese (Mandarin)_Unrestrained_Young_Man":{"label": "Unrestrained Young Man","gender": "M",  "language": "zh-CN", "boost": _BOOST_ZH},
    "Chinese (Mandarin)_Straightforward_Boy":  {"label": "Straightforward Boy",   "gender": "M",   "language": "zh-CN", "boost": _BOOST_ZH},
    "Chinese (Mandarin)_Pure-hearted_Boy":     {"label": "Pure-hearted Boy",      "gender": "M",   "language": "zh-CN", "boost": _BOOST_ZH},

    # English (en-US) — verified against `get_voice` API (only 6 enabled)
    "English_Trustworthy_Man":   {"label": "Trustworthy Man",  "gender": "M", "language": "en-US", "boost": _BOOST_EN},
    "English_Graceful_Lady":     {"label": "Graceful Lady",    "gender": "F", "language": "en-US", "boost": _BOOST_EN},
    "English_Aussie_Bloke":      {"label": "Aussie Bloke",     "gender": "M", "language": "en-US", "boost": _BOOST_EN},
    "English_Whispering_girl":   {"label": "Whispering Girl",  "gender": "F", "language": "en-US", "boost": _BOOST_EN},
    "English_Diligent_Man":      {"label": "Diligent Man",     "gender": "M", "language": "en-US", "boost": _BOOST_EN},
    "English_Gentle-voiced_man": {"label": "Gentle-voiced Man","gender": "M", "language": "en-US", "boost": _BOOST_EN},

    # Japanese (ja-JP)
    "Japanese_IntellectualSenior":  {"label": "Intellectual Senior",   "gender": "M",   "language": "ja-JP", "boost": _BOOST_JA},
    "Japanese_GentleButler":        {"label": "Gentle Butler",         "gender": "M",   "language": "ja-JP", "boost": _BOOST_JA},
    "Japanese_LoyalKnight":         {"label": "Loyal Knight",          "gender": "M",   "language": "ja-JP", "boost": _BOOST_JA},
    "Japanese_DominantMan":         {"label": "Dominant Man",          "gender": "M",   "language": "ja-JP", "boost": _BOOST_JA},
    "Japanese_SeriousCommander":    {"label": "Serious Commander",     "gender": "M",   "language": "ja-JP", "boost": _BOOST_JA},
    "Japanese_GenerousIzakayaOwner":{"label": "Izakaya Owner",         "gender": "M",   "language": "ja-JP", "boost": _BOOST_JA},
    "Japanese_InnocentBoy":         {"label": "Innocent Boy",          "gender": "M",   "language": "ja-JP", "boost": _BOOST_JA},
    "Japanese_KindLady":            {"label": "Kind Lady",             "gender": "F",   "language": "ja-JP", "boost": _BOOST_JA},
    "Japanese_CalmLady":            {"label": "Calm Lady",             "gender": "F",   "language": "ja-JP", "boost": _BOOST_JA},
    "Japanese_DependableWoman":     {"label": "Dependable Woman",      "gender": "F",   "language": "ja-JP", "boost": _BOOST_JA},
    "Japanese_DecisivePrincess":    {"label": "Decisive Princess",     "gender": "F",   "language": "ja-JP", "boost": _BOOST_JA},
    "Japanese_ColdQueen":           {"label": "Cold Queen",            "gender": "F",   "language": "ja-JP", "boost": _BOOST_JA},
    "Japanese_GracefulMaiden":      {"label": "Graceful Maiden",       "gender": "F",   "language": "ja-JP", "boost": _BOOST_JA},
    "Japanese_OptimisticYouth":     {"label": "Optimistic Youth",      "gender": "M/F", "language": "ja-JP", "boost": _BOOST_JA},
    "Japanese_SportyStudent":       {"label": "Sporty Student",        "gender": "M/F", "language": "ja-JP", "boost": _BOOST_JA},

    # Japanese — custom cloned voices (1222 batch)
    "jap_female_1222_1":            {"label": "JP Female 1222-1",      "gender": "F",   "language": "ja-JP", "boost": _BOOST_JA},
    "jap_female_1222_3":            {"label": "JP Female 1222-3",      "gender": "F",   "language": "ja-JP", "boost": _BOOST_JA},
    "jap_female_1222_4":            {"label": "JP Female 1222-4",      "gender": "F",   "language": "ja-JP", "boost": _BOOST_JA},
    "jap_female_1222_6":            {"label": "JP Female 1222-6",      "gender": "F",   "language": "ja-JP", "boost": _BOOST_JA},
    "jap_male_1222_1":              {"label": "JP Male 1222-1",        "gender": "M",   "language": "ja-JP", "boost": _BOOST_JA},
    "jap_male_1222_2":              {"label": "JP Male 1222-2",        "gender": "M",   "language": "ja-JP", "boost": _BOOST_JA},

    # Japanese — custom cloned voices (voice_agent batch)
    "jap_female_voice_agent_001":   {"label": "JP Female Agent 001",   "gender": "F",   "language": "ja-JP", "boost": _BOOST_JA},
    "jap_female_voice_agent_002":   {"label": "JP Female Agent 002",   "gender": "F",   "language": "ja-JP", "boost": _BOOST_JA},
    "jap_female_voice_agent_003":   {"label": "JP Female Agent 003",   "gender": "F",   "language": "ja-JP", "boost": _BOOST_JA},
    "jap_female_voice_agent_004":   {"label": "JP Female Agent 004",   "gender": "F",   "language": "ja-JP", "boost": _BOOST_JA},
    "jap_female_voice_agent_005":   {"label": "JP Female Agent 005",   "gender": "F",   "language": "ja-JP", "boost": _BOOST_JA},
    "jap_female_voice_agent_006":   {"label": "JP Female Agent 006",   "gender": "F",   "language": "ja-JP", "boost": _BOOST_JA},
    "jap_female_voice_agent_007":   {"label": "JP Female Agent 007",   "gender": "F",   "language": "ja-JP", "boost": _BOOST_JA},
    "jap_female_voice_agent_008":   {"label": "JP Female Agent 008",   "gender": "F",   "language": "ja-JP", "boost": _BOOST_JA},
    "jap_male_voice_agent_001":     {"label": "JP Male Agent 001",     "gender": "M",   "language": "ja-JP", "boost": _BOOST_JA},
    "jap_male_voice_agent_002":     {"label": "JP Male Agent 002",     "gender": "M",   "language": "ja-JP", "boost": _BOOST_JA},
    "jap_male_voice_agent_004":     {"label": "JP Male Agent 004",     "gender": "M",   "language": "ja-JP", "boost": _BOOST_JA},
    "jap_male_voice_agent_005":     {"label": "JP Male Agent 005",     "gender": "M",   "language": "ja-JP", "boost": _BOOST_JA},
    "jap_male_voice_agent_006":     {"label": "JP Male Agent 006",     "gender": "M",   "language": "ja-JP", "boost": _BOOST_JA},
    "jap_male_voice_agent_007":     {"label": "JP Male Agent 007",     "gender": "M",   "language": "ja-JP", "boost": _BOOST_JA},
    "jap_male_voice_agent_008":     {"label": "JP Male Agent 008",     "gender": "M",   "language": "ja-JP", "boost": _BOOST_JA},
    "jap_male_voice_agent_009":     {"label": "JP Male Agent 009",     "gender": "M",   "language": "ja-JP", "boost": _BOOST_JA},
    "jap_male_voice_agent_010":     {"label": "JP Male Agent 010",     "gender": "M",   "language": "ja-JP", "boost": _BOOST_JA},

    # Singapore English (en-SG) — user-supplied MiniMax IVC custom voices.
    # These are account-scoped IVC IDs, not public preset voices, so they're
    # only validated by hitting t2a_v2 (see T3 pre-deploy IVC API smoke).
    "mm_singlish_female_5_ivc_v3":  {"label": "Singlish Female 5",     "gender": "F",   "language": "en-SG", "boost": _BOOST_EN},
    "mm_singlish_male_2_ivc":       {"label": "Singlish Male 2",       "gender": "M",   "language": "en-SG", "boost": _BOOST_EN},
    "mm_singlish_male_1_ivc":       {"label": "Singlish Male 1",       "gender": "M",   "language": "en-SG", "boost": _BOOST_EN},
}
DEFAULT_MINIMAX_VOICE = "Cantonese_GentleLady"

# Per-language default voice override. Used when a session has lang=X but no
# explicit voice — without this, en-SG would fall back to Cantonese_GentleLady,
# which speaks Cantonese (wrong language).
DEFAULT_MINIMAX_VOICE_BY_LANG: dict[str, str] = {
    "en-SG": "mm_singlish_female_5_ivc_v3",
}


class SimpleMiniMaxTTSService(MiniMaxHttpTTSService):
    """MiniMax HTTP TTS using SSE streaming for low TTFB.

    Pipecat 1.1.0's bundled MiniMax service had problems surfacing audio frames
    in our pipeline. This subclass keeps the same wire protocol (``stream=true``,
    SSE ``data:`` chunks with hex-encoded PCM) but yields ``TTSAudioRawFrame``
    directly, which avoids Pipecat's internal audio-context bookkeeping that
    occasionally swallowed frames in the bundled service. Also fixes:

    - The ``?GroupId=`` parent-class URL bug (empty GroupId is treated as an
      unfunded account by MiniMax, returning "insufficient balance").
    - Surface ``base_resp.status_code != 0`` as a real ErrorFrame instead of
      silently producing zero-length audio.
    """

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator:
        logger.debug(f"{self}: MiniMax synth [{text}]")

        # Empty GroupId in URL == different (unfunded) account on MiniMax's side.
        url = self._base_url
        if not (self._group_id and self._group_id.strip()):
            url = url.split("?", 1)[0]

        headers = {
            "accept": "application/json, text/plain, */*",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        voice_setting = {
            "voice_id": self._settings.voice,
            "speed": self._settings.speed,
            "vol": self._settings.volume,
            "pitch": self._settings.pitch,
        }
        if self._settings.emotion is not None:
            voice_setting["emotion"] = self._settings.emotion
        payload: dict = {
            "stream": True,
            "model": self._settings.model,
            "text": text,
            "voice_setting": voice_setting,
            "audio_setting": {
                "bitrate": self._audio_bitrate,
                "format": self._audio_format,
                "channel": self._audio_channel,
                "sample_rate": self._audio_sample_rate,
            },
        }
        if self._settings.language_boost is not None:
            payload["language_boost"] = self._settings.language_boost
            # Cantonese 「返」homograph fix: force faan1 (虚词「再做一次」) over the
            # default faan2 (回去) reading. MiniMax T2A v2 pronunciation_dict.tone
            # accepts Jyutping 1-6, format "原字/(jyutping)".
            if self._settings.language_boost == "Chinese,Yue":
                payload["pronunciation_dict"] = {"tone": ["返/(faan1)"]}

        try:
            resp_cm = self._session.post(url, headers=headers, json=payload)
        except Exception as e:
            yield ErrorFrame(error=f"MiniMax request failed: {e}")
            return

        usage_started = False
        try:
            async with resp_cm as resp:
                if resp.status != 200:
                    body = await resp.text()
                    yield ErrorFrame(error=f"MiniMax HTTP {resp.status}: {body[:200]}")
                    return

                buffer = bytearray()
                CHUNK = 8192
                async for raw in resp.content.iter_chunked(CHUNK):
                    if not raw:
                        continue
                    buffer.extend(raw)
                    # Walk all complete SSE blocks ("data: ..." separated).
                    while b"data:" in buffer:
                        start = buffer.find(b"data:")
                        nxt = buffer.find(b"data:", start + 5)
                        if nxt == -1:
                            # Keep current block in buffer; need more bytes.
                            if start > 0:
                                buffer = buffer[start:]
                            break
                        block = bytes(buffer[start:nxt])
                        buffer = buffer[nxt:]
                        try:
                            data = _mm_json.loads(block[5:].decode("utf-8"))
                        except (_mm_json.JSONDecodeError, UnicodeDecodeError):
                            continue

                        # First chunk of a stream may be base_resp / extra_info only.
                        base = (data or {}).get("base_resp") or {}
                        if base.get("status_code", 0) != 0:
                            yield ErrorFrame(error=f"MiniMax error: {base.get('status_msg', data)}")
                            return
                        if "extra_info" in data:
                            # Final chunk; loop will end naturally.
                            continue

                        audio_hex = ((data or {}).get("data") or {}).get("audio") or ""
                        if not audio_hex:
                            continue
                        try:
                            pcm = bytes.fromhex(audio_hex)
                        except ValueError:
                            continue
                        if not pcm:
                            continue

                        if not usage_started:
                            await self.start_tts_usage_metrics(text)
                            usage_started = True
                        await self.stop_ttfb_metrics()

                        # Re-slice incoming PCM into Pipecat's preferred chunk size
                        # (20 ms @ 24 kHz = 960 bytes) so downstream pacing stays smooth.
                        chunk_bytes = self.chunk_size or 960
                        for i in range(0, len(pcm), chunk_bytes):
                            yield TTSAudioRawFrame(
                                audio=pcm[i : i + chunk_bytes],
                                sample_rate=self._audio_sample_rate,
                                num_channels=1,
                                context_id=context_id,
                            )
        except Exception as e:
            yield ErrorFrame(error=f"MiniMax stream failed: {e}")
            return

# ---- Minimal AWS Polly TTS service -------------------------------------
# Pipecat's bundled AWSPollyTTSService was not yielding audio frames in our
# pipeline (run_tts started but never reached yield; resampler interaction
# suspected). This thin replacement calls Polly directly, streams PCM out
# at Polly's native rate, and lets downstream transport handle playback.


@dataclass
class _PollyTTSSettings(TTSSettings):
    engine: str | None = None


class SimplePollyTTSService(TTSService):
    """Minimal Polly TTS. Skips SSML/resample — outputs native 16 kHz PCM16."""

    Settings = _PollyTTSSettings
    _settings: Settings

    POLLY_SAMPLE_RATE = 16000  # Polly neural PCM is 16 kHz

    def __init__(
        self,
        *,
        voice: str,
        engine: str = "neural",
        region: str,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        sample_rate: int | None = None,
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=self.Settings(model=None, voice=voice, engine=engine),
            **kwargs,
        )
        self._aws_kwargs = {
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
            "aws_session_token": aws_session_token,
            "region_name": region,
        }
        self._aws_session = get_session()

    def can_generate_metrics(self) -> bool:
        return True

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator:
        logger.debug(f"{self}: Polly synth [{text}]")
        try:
            async with self._aws_session.create_client("polly", **self._aws_kwargs) as polly:
                resp = await polly.synthesize_speech(
                    Text=text,
                    TextType="text",
                    OutputFormat="pcm",
                    VoiceId=self._settings.voice,
                    Engine=self._settings.engine,
                    SampleRate=str(self.POLLY_SAMPLE_RATE),
                )
                pcm = await resp["AudioStream"].read()

            if not pcm:
                yield ErrorFrame(error="Polly returned empty audio")
                return

            await self.start_tts_usage_metrics(text)

            # Emit in 20ms chunks at Polly's native rate. Transport will resample
            # to self.sample_rate via its own audio pipeline.
            chunk_bytes = 640  # 20ms @ 16kHz PCM16 mono
            for i in range(0, len(pcm), chunk_bytes):
                await self.stop_ttfb_metrics()
                yield TTSAudioRawFrame(
                    audio=pcm[i : i + chunk_bytes],
                    sample_rate=self.POLLY_SAMPLE_RATE,
                    num_channels=1,
                    context_id=context_id,
                )
        except Exception as e:
            logger.exception("Polly TTS failed")
            yield ErrorFrame(error=f"Polly TTS error: {e}")
        finally:
            await self.stop_ttfb_metrics()


# Polly voices, sourced from
# https://docs.aws.amazon.com/polly/latest/dg/available-voices.html
# Each entry: label, gender, language (matches our LANGUAGES keys), engine.
# Engine prefers "generative" where Polly supports it (better quality), else "neural".
POLLY_VOICES: dict[str, dict] = {
    # Cantonese (zh-HK) — Polly's only Cantonese voice
    "Hiujin":   {"label": "Hiujin",   "gender": "F", "language": "zh-HK", "engine": "neural"},

    # Mandarin (zh-CN)
    "Zhiyu":    {"label": "Zhiyu",    "gender": "F", "language": "zh-CN", "engine": "neural"},

    # English (en-US)
    "Joanna":     {"label": "Joanna",    "gender": "F", "language": "en-US", "engine": "generative"},
    "Matthew":    {"label": "Matthew",   "gender": "M", "language": "en-US", "engine": "generative"},
    "Ruth":       {"label": "Ruth",      "gender": "F", "language": "en-US", "engine": "generative"},
    "Stephen":    {"label": "Stephen",   "gender": "M", "language": "en-US", "engine": "generative"},
    "Danielle":   {"label": "Danielle",  "gender": "F", "language": "en-US", "engine": "generative"},
    "Gregory":    {"label": "Gregory",   "gender": "M", "language": "en-US", "engine": "long-form"},
    "Patrick":    {"label": "Patrick",   "gender": "M", "language": "en-US", "engine": "long-form"},
    "Kendra":     {"label": "Kendra",    "gender": "F", "language": "en-US", "engine": "neural"},
    "Kimberly":   {"label": "Kimberly",  "gender": "F", "language": "en-US", "engine": "neural"},
    "Salli":      {"label": "Salli",     "gender": "F", "language": "en-US", "engine": "neural"},
    "Joey":       {"label": "Joey",      "gender": "M", "language": "en-US", "engine": "neural"},
    "Justin":     {"label": "Justin",    "gender": "M (child)", "language": "en-US", "engine": "neural"},
    "Kevin":      {"label": "Kevin",     "gender": "M (child)", "language": "en-US", "engine": "neural"},
    "Ivy":        {"label": "Ivy",       "gender": "F (child)", "language": "en-US", "engine": "neural"},

    # Japanese (ja-JP)
    "Takumi":   {"label": "Takumi",   "gender": "M", "language": "ja-JP", "engine": "neural"},
    "Kazuha":   {"label": "Kazuha",   "gender": "F", "language": "ja-JP", "engine": "neural"},
    "Tomoko":   {"label": "Tomoko",   "gender": "F", "language": "ja-JP", "engine": "neural"},
    "Mizuki":   {"label": "Mizuki",   "gender": "F", "language": "ja-JP", "engine": "standard"},
}
DEFAULT_POLLY_VOICE = "Zhiyu"

TTS_PROVIDERS = {"minimax": "MiniMax", "polly": "AWS Polly"}
DEFAULT_PROVIDER = "minimax"

# MiniMax TTS models. Newer "speech-2.x-turbo" gives better quality and is the
# default; older "speech-02-turbo" stays as a fallback. Override via env
# MINIMAX_MODEL_DEFAULT.
MINIMAX_MODELS: dict[str, str] = {
    "speech-2.8-turbo": "speech-2.8-turbo (faster, recommended)",
    "speech-2.8-hd":    "speech-2.8-hd (higher quality, slower)",
}
DEFAULT_MINIMAX_MODEL = os.environ.get("MINIMAX_MODEL_DEFAULT", "speech-2.8-turbo")

# Nova Sonic v2 voices (lowercase). See:
# https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-language-support.html
NOVA_SONIC_VOICES: dict[str, dict] = {
    # locale en-US — tiffany/matthew are polyglot (can speak all 7 supported languages)
    "tiffany":  {"label": "Tiffany",  "gender": "F", "locale": "en-US", "lang_label": "English (US)", "polyglot": True},
    "matthew":  {"label": "Matthew",  "gender": "M", "locale": "en-US", "lang_label": "English (US)", "polyglot": True},
    "amy":      {"label": "Amy",      "gender": "F", "locale": "en-GB", "lang_label": "English (UK)"},
    "olivia":   {"label": "Olivia",   "gender": "F", "locale": "en-AU", "lang_label": "English (AU)"},
    "kiara":    {"label": "Kiara",    "gender": "F", "locale": "en-IN", "lang_label": "English (IN) / Hindi"},
    "arjun":    {"label": "Arjun",    "gender": "M", "locale": "en-IN", "lang_label": "English (IN) / Hindi"},
    "ambre":    {"label": "Ambre",    "gender": "F", "locale": "fr-FR", "lang_label": "Français"},
    "florian":  {"label": "Florian",  "gender": "M", "locale": "fr-FR", "lang_label": "Français"},
    "beatrice": {"label": "Beatrice", "gender": "F", "locale": "it-IT", "lang_label": "Italiano"},
    "lorenzo":  {"label": "Lorenzo",  "gender": "M", "locale": "it-IT", "lang_label": "Italiano"},
    "tina":     {"label": "Tina",     "gender": "F", "locale": "de-DE", "lang_label": "Deutsch"},
    "lennart":  {"label": "Lennart",  "gender": "M", "locale": "de-DE", "lang_label": "Deutsch"},
    "lupe":     {"label": "Lupe",     "gender": "F", "locale": "es-US", "lang_label": "Español (US)"},
    "carlos":   {"label": "Carlos",   "gender": "M", "locale": "es-US", "lang_label": "Español (US)"},
    "carolina": {"label": "Carolina", "gender": "F", "locale": "pt-BR", "lang_label": "Português (BR)"},
    "leo":      {"label": "Leo",      "gender": "M", "locale": "pt-BR", "lang_label": "Português (BR)"},
}
# Migration map for old voice ids that are NOT in the official Nova 2 Sonic list
# (these may still live in runtime.json or bookmarked /ws?voice= links — keep them working).
NOVA_SONIC_VOICE_ALIASES = {"marie": "ambre", "sofia": "lupe", "ana": "carolina"}
DEFAULT_NOVA_SONIC_VOICE = "tiffany"

# Conversation engine:
#   - "pipeline": STT (Transcribe) + LLM (Bedrock) + TTS (MiniMax/Polly)
#   - "nova-sonic": one end-to-end speech-to-speech model (Nova 2 Sonic)
ENGINES: dict[str, str] = {
    "pipeline":   "Pipeline (STT+LLM+TTS)",
    "nova-sonic": "Nova Sonic v2 (S2S)",
}
DEFAULT_ENGINE = "nova-sonic"

# Pipeline mode is strictly turn-based: the LLM only runs when a user turn
# arrives. If the bot ends its reply announcing it will continue ("請稍等" /
# "please hold"), both sides wait forever. After the bot stops speaking and
# the user stays silent for IDLE_NUDGE_TIMEOUT seconds, we re-run the LLM on
# the existing context so it continues naturally. At most IDLE_NUDGE_MAX
# consecutive nudges without a real user turn, so an absent user doesn't
# trigger an endless monologue. Nova Sonic path unaffected (server-side turn
# management).
IDLE_NUDGE_TIMEOUT = 4.0
IDLE_NUDGE_MAX = 2

# Developer-role instruction injected before each idle re-run. A bare
# LLMRunFrame is not enough: with the context ending in an assistant message,
# nova-2-lite considers its turn finished and returns empty (~2 tokens), which
# also never re-arms the idle timer (no bot speech -> no BotStoppedSpeaking).
# Same mechanism as the greeting flow (developer message + LLMRunFrame).
IDLE_NUDGE_PROMPTS = {
    "zh-HK": "用戶暫時冇出聲。如果你頭先講咗會繼續（例如「請稍等」），而家直接繼續講落去；否則簡短問下用戶仲喺唔喺度。只講一兩句。",
    "zh-CN": "用户暂时没有说话。如果你刚才说过会继续（例如「请稍等」），现在直接继续；否则简短问一下用户是否还在。只说一两句。",
    "en-US": "The user hasn't spoken for a while. If your last message promised to continue (e.g. \"please hold\"), continue now; otherwise briefly check if they are still there. One or two sentences.",
    "ja-JP": "ユーザーがしばらく話していません。直前に「お待ちください」など続きを約束した場合は、そのまま続けてください。そうでなければ、まだいらっしゃるか短く確認してください。1〜2文で。",
}


def _resolve_minimax_voice(voice_key: str, lang: str | None = None) -> tuple[str, str]:
    # Read through the registry (live table if loaded, else in-code constants).
    voices = voices_for("minimax")
    # Known voice key wins outright — per-lang fallback only fires when the
    # caller didn't pick (or picked something we don't have).
    if voice_key in voices:
        return voice_key, voices[voice_key]["boost"]
    if lang and lang in DEFAULT_MINIMAX_VOICE_BY_LANG:
        fb = DEFAULT_MINIMAX_VOICE_BY_LANG[lang]
        if fb in voices:
            return fb, voices[fb]["boost"]
    # DEFAULT_MINIMAX_VOICE is also seeded as a row, so it normally resolves
    # against the live registry; the constant boost is the last-resort fallback
    # if even the default id is absent (deleted/dangling), so we never crash.
    if DEFAULT_MINIMAX_VOICE in voices:
        return DEFAULT_MINIMAX_VOICE, voices[DEFAULT_MINIMAX_VOICE]["boost"]
    return DEFAULT_MINIMAX_VOICE, MINIMAX_VOICES[DEFAULT_MINIMAX_VOICE]["boost"]


def _resolve_polly_voice(voice_key: str) -> tuple[str, str]:
    voices = voices_for("polly")
    if voice_key in voices:
        return voice_key, voices[voice_key]["engine"]
    if DEFAULT_POLLY_VOICE in voices:
        return DEFAULT_POLLY_VOICE, voices[DEFAULT_POLLY_VOICE]["engine"]
    return DEFAULT_POLLY_VOICE, POLLY_VOICES[DEFAULT_POLLY_VOICE]["engine"]


def _voice_display(v: dict) -> str:
    """Human label for a voice dropdown entry: "Warm Bestie (F)".

    ``label`` is defaulted by _validate_voice_body for every admin-created
    voice, but a row written out-of-band (manual DDB put / future migration)
    could still lack one — fall back to the id, then "?", so rendering a voice
    list never KeyErrors and 500s /api/config app-wide."""
    label = v.get("label") or v.get("voice_id") or v.get("id") or "?"
    g = v.get("gender")
    return f"{label} ({g})" if g else label


# --- Voice registry accessor (Part A) ------------------------------------
# The three hardcoded dicts above stay in code as (1) the seed source and
# (2) the table-missing fallback. ``voices_for(provider)`` is the SINGLE read
# path that every resolver / endpoint goes through: when VOICE_STORE is live
# (its DDB table exists + a scan succeeded) it returns the live registry,
# otherwise the in-code constant dict — so with no VoicesTable the behaviour is
# byte-identical to before (zero-impact ship).

_VOICE_CONSTANTS: dict[str, dict[str, dict]] = {
    "minimax": MINIMAX_VOICES,
    "polly": POLLY_VOICES,
    "nova-sonic": NOVA_SONIC_VOICES,
}


def voices_for(provider: str) -> dict[str, dict]:
    """Return ``{voice_id: attrs}`` for ``provider`` from the live registry if
    VOICE_STORE is live, else the in-code constant dict. Unknown provider -> {}.

    Never raises and never returns None, so every call site can treat the result
    like the old constant dict (``v in voices_for(p)`` / iterate ``.items()``)."""
    if VOICE_STORE.live:
        live = VOICE_STORE.list(provider)
        # Defensive: if the live table somehow has no rows for this provider
        # (e.g. an admin deleted them all) fall back to the constants so the
        # provider's resolver still has its DEFAULT_* voice to land on.
        if live:
            return live
    return _VOICE_CONSTANTS.get(provider, {})


# --- Activity log hook (Part B / T3) -------------------------------------
# Real audit-log helper backed by ACTIVITY_STORE (DDB). Thin wrapper that
# resolves the actor from the authed user (or 'phone'/'anon'), then calls the
# best-effort ``ACTIVITY_STORE.log``. WHOLE PATH is try/except'd so a logging
# failure can NEVER break the operation being audited — the call sites also
# wrap their call defensively, this is belt-and-suspenders. ``detail`` is a
# CURATED, already-safe map (changed field NAMES for mutating endpoints, never
# raw bodies/passwords); the store's default-deny ``_sanitize_detail`` is the
# final backstop. (Replaces the T2 no-op stub; the voice-CRUD endpoints already
# call this — do NOT duplicate those call sites.)
async def log_activity(
    actor: "str | dict | None" = None,
    actor_role: str | None = None,
    type: str = "",
    target: str | None = None,
    detail: dict | None = None,
    status: str = "success",
    error: str | None = None,
) -> None:
    """Best-effort audit log. ``actor`` may be a username string, an authed
    user dict ({'username','role'}), or None → resolved to 'anon'. Never
    raises into the caller."""
    try:
        # Resolve actor: a user dict → its username/role; a string → as-is; a
        # falsy value → 'anon' (e.g. an unauthenticated path).
        if isinstance(actor, dict):
            actor_role = actor_role or actor.get("role")
            actor = actor.get("username")
        actor = actor or "anon"
        await ACTIVITY_STORE.log(
            actor=actor,
            actor_role=actor_role,
            type=type,
            target=target,
            detail=detail,
            status=status,
            error=error,
        )
    except Exception as e:  # noqa: BLE001 — auditing must never break the op
        # NB: the `type` PARAMETER shadows the builtin here, so use the class
        # attribute directly — `type(e)` would raise "str object is not
        # callable" and defeat the never-break-the-caller guarantee.
        logger.debug(f"log_activity dropped (type={type!r}): {e.__class__.__name__}: {e}")


# Demo presets used to live in two in-code dicts that were unioned with
# disk-based DEMO_LOADER demos. Both dicts have been removed in T4 in favor
# of a single source of truth: ``data/<demo>/manifest.yaml`` loaded by
# DEMO_LOADER. The only scenario sentinel left in code is the "default" id,
# which means "no demo selected — use the per-language default prompt from
# LANGUAGES[lang]". See proposal §3.3.
DEFAULT_DEMO_ID = "default"
DEFAULT_SCENARIO = DEFAULT_DEMO_ID  # back-compat alias for legacy call sites




def _load_kb(path: str) -> str:
    """Load a knowledge-base file from the project directory. Returns '' on error."""
    here = os.path.dirname(os.path.abspath(__file__))
    full = os.path.join(here, path)
    try:
        with open(full, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.warning(f"failed to load KB {full}: {e}")
        return ""


def _kb_seed_messages(scenario_key: str, lang_key: str) -> list[dict] | None:
    """Return the user/assistant message pair that seeds the KB into chat history.

    The pair looks like:
        user:      "Here is the document...: <KB BODY>"
        assistant: "Got it, I've read the document and I'm ready to help."

    Source: ``DEMO_LOADER`` (data/<demo>/manifest.yaml). Returns None when
    the demo has no KB body for the requested language. (T4 removed the
    legacy in-code fallback dict.)
    """
    demo = DEMO_LOADER.get(scenario_key)
    if not demo:
        return None
    kb = demo.get("kb_body")
    # Per-language KB: dict[lang -> str]; legacy single-file KB: str.
    if isinstance(kb, dict):
        body = kb.get(lang_key) or kb.get(DEFAULT_LANG) or next(iter(kb.values()), "")
    else:
        body = kb or ""
    if not body:
        return None
    intro_map = demo.get("kb_intro") or {}
    ack_map = demo.get("kb_ack") or {}
    intro = intro_map.get(lang_key) or intro_map.get(DEFAULT_LANG) or ""
    ack = ack_map.get(lang_key) or ack_map.get(DEFAULT_LANG) or "OK."
    return [
        {"role": "user", "content": intro + body},
        {"role": "assistant", "content": ack},
    ]


# Summary prompts — per target language. Kept concise; the transcript is
# injected as the single user message.
SUMMARY_PROMPTS: dict[str, str] = {
    "zh-HK": (
        "你係一個專業嘅對話總結助手。請用粵語(廣東話)書面語,簡潔地總結以下對話嘅"
        "主要內容、關鍵信息同結論。用 3-5 個要點列出,每點一行。"
    ),
    "zh-CN": (
        "你是一个专业的对话总结助手。请用普通话简洁地总结以下对话的主要内容、"
        "关键信息和结论。用 3-5 个要点列出,每点一行。"
    ),
    "en-US": (
        "You are a professional conversation summarizer. Summarize the following "
        "conversation in English concisely, covering the main topics, key "
        "information, and conclusions. Use 3-5 bullet points, one per line."
    ),
    "ja-JP": (
        "あなたはプロの対話要約アシスタントです。以下の会話の主な内容、重要な情報、"
        "結論を日本語で簡潔に要約してください。3〜5つの箇条書きで、1行に1つずつ示してください。"
    ),
}


from collections.abc import Awaitable, Callable

EmitFn = Callable[[dict], Awaitable[None]]


# --- Multi-listener session registry ------------------------------------
# Tracks active phone sessions so monitor websockets can subscribe to events.
# Keys are call_id strings; values describe the session and hold a list of
# emit callbacks (one per attached monitor + the phone's own ws).
_SESSIONS_LOCK = asyncio.Lock()
ACTIVE_SESSIONS: dict[str, dict] = {}


def _ws_emit(websocket: WebSocket) -> EmitFn:
    """Build an EmitFn that sends a JSON event to one websocket, no-op if closed."""
    async def emit(payload: dict) -> None:
        if websocket.application_state is not WebSocketState.CONNECTED:
            return
        try:
            await websocket.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            logger.debug(f"event send failed: {e}")
    return emit


def _multi_emit(get_emits: Callable[[], list[EmitFn]]) -> EmitFn:
    """Fan out one event to every emit function returned by ``get_emits()``.

    The list is fetched fresh per event so newly attached monitors see future
    events without needing to restart the broadcaster.
    """
    async def emit(payload: dict) -> None:
        for fn in get_emits():
            try:
                await fn(payload)
            except Exception:
                pass  # one bad listener shouldn't stop the others
    return emit


async def session_register(
    call_id: str,
    *,
    caller: str | None = None,
    primary_emit: EmitFn,
) -> dict:
    """Register a new session. Returns the session dict (also stored in registry)."""
    async with _SESSIONS_LOCK:
        sess = {
            "call_id": call_id,
            "caller": caller,
            "started": time.time(),
            "emits": [primary_emit],
            "monitors": [],  # subset of emits that came from /monitor/ws
        }
        ACTIVE_SESSIONS[call_id] = sess
        logger.info(f"session registered: call_id={call_id} caller={caller}")
        return sess


async def session_unregister(call_id: str) -> None:
    async with _SESSIONS_LOCK:
        sess = ACTIVE_SESSIONS.pop(call_id, None)
        if sess:
            logger.info(f"session unregistered: call_id={call_id}")


async def session_attach_monitor(call_id: str, monitor_emit: EmitFn) -> bool:
    """Attach a monitor's emit fn. Returns False if call_id not found."""
    async with _SESSIONS_LOCK:
        sess = ACTIVE_SESSIONS.get(call_id)
        if not sess:
            return False
        sess["emits"].append(monitor_emit)
        sess["monitors"].append(monitor_emit)
        return True


async def session_detach_monitor(call_id: str, monitor_emit: EmitFn) -> None:
    async with _SESSIONS_LOCK:
        sess = ACTIVE_SESSIONS.get(call_id)
        if not sess:
            return
        for lst_name in ("emits", "monitors"):
            try:
                sess[lst_name].remove(monitor_emit)
            except ValueError:
                pass


def session_emits(call_id: str) -> list[EmitFn]:
    """Return current emits list for a session (snapshot, fine for fan-out)."""
    sess = ACTIVE_SESSIONS.get(call_id)
    return list(sess["emits"]) if sess else []


# --- Call history persistence (DynamoDB) ----------------------------------
# Phone calls only. Browser /ws does not persist. HISTORY_TABLE empty or
# HISTORY_DISABLED=1 ⇒ _history is None and every call site is a no-op.
HISTORY_TABLE = os.environ.get("HISTORY_TABLE", "").strip()
HISTORY_TTL_DAYS = int(os.environ.get("HISTORY_TTL_DAYS", "30"))
HISTORY_DISABLED = os.environ.get("HISTORY_DISABLED", "0") == "1" or not HISTORY_TABLE
HISTORY_BUFFERS: dict[str, list[dict]] = {}

# Phone-leg LLM tool registration (end_call / transfer_to_human). Default ON.
# Disable by setting PHONE_TOOLS_ENABLED=0 in /opt/voicebot/.env and restarting.
PHONE_TOOLS_ENABLED = os.environ.get("PHONE_TOOLS_ENABLED", "1").strip() not in ("", "0", "false", "False")


def _to_ddb(value):
    """Recursively convert Python floats to Decimal so the DynamoDB resource
    layer accepts them. Lists / dicts are walked in place; other scalars are
    returned unchanged."""
    from decimal import Decimal

    if isinstance(value, float):
        # str() round-trips through Decimal cleanly; passing the float directly
        # produces noisy artifacts like 0.10000000000000000555…
        return Decimal(str(value))
    if isinstance(value, list):
        return [_to_ddb(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_ddb(v) for k, v in value.items()}
    return value


class HistoryRecorder:
    """Buffers final transcript turns + meta per call, flushes to DynamoDB
    on finalize(), then schedules an async Bedrock summary that updates the
    same row in place.

    Methods:
        attach(call_id, session_meta) — initialize buffer + meta snapshot.
        append(call_id, turn)         — append one final turn (sync).
        async finalize(call_id)       — drain + put_item, schedule summary.
    """

    def __init__(self, table_name: str, region: str, ttl_days: int):
        self._table_name = table_name
        self._region = region
        self._ttl_days = ttl_days
        self._meta: dict[str, dict] = {}
        # Sync boto3 resource invoked via asyncio.to_thread for two reasons:
        # (1) moto's test fixture patches botocore (sync) cleanly but breaks
        # against aiobotocore in this version pair; (2) post-call DDB writes
        # are infrequent enough that the thread-pool hop is negligible.
        self._table = boto3.resource("dynamodb", region_name=region).Table(table_name)

    def attach(self, call_id: str, session_meta: dict) -> None:
        HISTORY_BUFFERS.setdefault(call_id, [])
        self._meta[call_id] = dict(session_meta)

    def append(self, call_id: str, turn: dict) -> None:
        HISTORY_BUFFERS.setdefault(call_id, []).append(turn)

    def append_event(self, call_id: str, kind: str, data: dict) -> None:
        """Record a non-transcript event (e.g. tool_call) onto the same buffer
        so the persisted call history reflects in-call signals like hangups."""
        HISTORY_BUFFERS.setdefault(call_id, []).append(
            {"kind": kind, "data": data, "t": time.monotonic()}
        )

    def mark_transfer(self, call_id: str, topic: str) -> None:
        """Flag this call as needing human follow-up. Persisted as top-level
        ``transfer_requested`` / ``transfer_topic`` columns by ``finalize`` so a
        callback queue can Query the table without scanning the turns array."""
        meta = self._meta.get(call_id)
        if meta is None:
            return
        meta["transfer_requested"] = True
        meta["transfer_topic"] = (topic or "")[:500]

    async def write_outcome_row(self, call_id: str) -> bool:
        """Persist the minimal fields a downstream Connect Flow Lambda needs
        (caller, transfer_requested, transfer_topic, timestamps, TTL).

        Awaited synchronously by the LLM tool handlers BEFORE BYE goes out so
        the row exists by the time Connect resumes its Flow. Idempotent via
        meta["_outcome_written"]; safe to call again from
        flush_turns_and_summarize for the remote-hangup path.
        """
        meta = self._meta.get(call_id)
        if meta is None:
            return False
        if meta.get("_outcome_written"):
            return True
        started_at = float(meta.get("started_at") or time.time())
        ended_at = time.time()
        ttl = int(started_at + self._ttl_days * 86400)
        item = {
            "call_id": call_id,
            "caller": meta.get("caller") or "unknown",
            "started_at": int(started_at),
            "ended_at": int(ended_at),
            "duration_s": int(max(0, ended_at - started_at)),
            "summary_status": "pending",
            "ttl": ttl,
            # Always written so downstream consumers (e.g. Connect Flow
            # Lambda) can read item["transfer_requested"] without a fallback.
            "transfer_requested": bool(meta.get("transfer_requested")),
            "transfer_topic": meta.get("transfer_topic") or "",
            # web_user attributes a browser /ws call to the logged-in user so
            # GET /api/history can scope results per-user. Phone calls leave it
            # empty (Chime callers are not logged-in users).
            "web_user": meta.get("web_user") or "",
        }
        t0 = time.monotonic()
        try:
            await asyncio.to_thread(self._table.put_item, Item=_to_ddb(item))
        except Exception as e:
            logger.warning(f"history outcome put_item failed for {call_id}: {e}")
            return False
        meta["_outcome_written"] = True
        latency_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            f"history outcome row written call_id={call_id} "
            f"caller={item['caller']} transfer_requested={item['transfer_requested']} "
            f"latency_ms={latency_ms}"
        )
        return True

    async def flush_turns_and_summarize(self, call_id: str) -> None:
        """Post-BYE flush: ensure the outcome row exists (remote-hangup path),
        UPDATE it with turns + scenario meta, then schedule the Bedrock
        summary in the background. Uses update_item (SET) so it never
        overwrites the outcome fields a Connect Lambda may already be reading.
        """
        # Cover the remote-hangup path where the LLM tool was never called.
        # write_outcome_row is idempotent so the tool path pays no extra cost.
        await self.write_outcome_row(call_id)

        turns = HISTORY_BUFFERS.pop(call_id, [])
        meta = self._meta.pop(call_id, None)
        if meta is None:
            return
        update_kwargs = {
            "Key": {"call_id": call_id},
            "UpdateExpression": (
                "SET turns = :t, turn_count = :n, lang = :lang, engine = :eng, "
                "scenario = :sc, provider = :prov, voice = :v, model = :m, "
                "minimax_model = :mm"
            ),
            "ExpressionAttributeValues": _to_ddb({
                ":t": turns,
                ":n": len(turns),
                ":lang": meta.get("lang") or "",
                ":eng": meta.get("engine") or "",
                ":sc": meta.get("scenario") or "",
                ":prov": meta.get("provider") or "",
                ":v": meta.get("voice") or "",
                ":m": meta.get("model") or "",
                ":mm": meta.get("minimax_model") or "",
            }),
        }
        t0 = time.monotonic()
        try:
            await asyncio.to_thread(self._table.update_item, **update_kwargs)
        except Exception as e:
            logger.warning(f"history turns update_item failed for {call_id}: {e}")
            return
        latency_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            f"history turns flushed call_id={call_id} "
            f"turns={len(turns)} latency_ms={latency_ms}"
        )
        asyncio.create_task(self._summarize_and_update(call_id, turns, meta))

    async def _summarize_and_update(
        self, call_id: str, turns: list[dict], meta: dict
    ) -> None:
        lang = meta.get("lang") or DEFAULT_LANG
        try:
            struct = await asyncio.wait_for(
                _invoke_summary_bedrock(turns, lang), timeout=30.0
            )
            update_kwargs = {
                "Key": {"call_id": call_id},
                "UpdateExpression": "SET summary = :s, summary_status = :ok",
                "ExpressionAttributeValues": {":s": _to_ddb(struct), ":ok": "ok"},
            }
        except Exception as e:
            logger.warning(f"history summary failed for {call_id}: {e}")
            update_kwargs = {
                "Key": {"call_id": call_id},
                "UpdateExpression": "SET summary_status = :f, summary_error = :e",
                "ExpressionAttributeValues": {
                    ":f": "failed",
                    ":e": str(e)[:500],
                },
            }
        try:
            await asyncio.to_thread(self._table.update_item, **update_kwargs)
        except Exception as e:
            logger.warning(f"history update_item failed for {call_id}: {e}")


_history: HistoryRecorder | None = (
    None
    if HISTORY_DISABLED
    else HistoryRecorder(
        table_name=HISTORY_TABLE,
        # DynamoDB lives in the DEPLOY region, which can differ from AWS_REGION
        # (the Bedrock region). DDB_REGION falls back to AWS_REGION, so a
        # single-region deploy (the common case) is unchanged.
        region=os.environ.get("DDB_REGION") or os.environ.get("AWS_REGION", "us-east-1"),
        ttl_days=HISTORY_TTL_DAYS,
    )
)
if _history is None:
    logger.info(
        f"history persistence disabled (HISTORY_TABLE={HISTORY_TABLE!r} "
        f"HISTORY_DISABLED={os.environ.get('HISTORY_DISABLED','0')})"
    )
else:
    logger.info(
        f"history persistence enabled: table={HISTORY_TABLE} ttl_days={HISTORY_TTL_DAYS}"
    )


class BotPlayoutTracker:
    """Server-side model of when the bot's current audio turn finishes
    *playing* at the far end.

    pipecat emits ``BotStoppedSpeakingFrame`` when TTS finishes *generating*
    (base_output.py on ``TTSStoppedFrame``), not when the audio finishes
    *playing*: the output write loop has no per-chunk real-time pacing, so it
    flushes audio as fast as the socket accepts it. On a long opening sentence
    that BotStopped lands 2-3 s before the caller has actually heard the bot
    finish. The idle-nudge timer keyed off that early BotStopped then fires
    while the bot is still audible, so the bot talks over itself / repeats.

    We rebuild the true playout-finish time on the server: the bot's audio is
    PCM16, so its duration is ``bytes / (sample_rate * 2 * channels)``. Adding
    that to the turn's start monotonic time gives the estimated instant the
    last sample reaches the listener. The idle handler uses this as the timer's
    zero point instead of the (early) BotStopped event.

    Not thread-safe; all mutation happens from the single pipeline event loop.
    """

    def __init__(self, out_sample_rate: int, channels: int = 1):
        self._rate = out_sample_rate
        self._channels = channels
        self.tts_started_at: float | None = None
        self.bot_audio_bytes = 0
        # True once a bot turn's audio has begun and not yet ended. The first
        # audio frame of a NEW turn (turn_active False) resets the byte
        # counter + start clock; subsequent frames accumulate.
        self._turn_active = False

    def on_bot_turn_start(self, now: float | None = None) -> None:
        """Anchor the playout clock at the start of a bot turn.

        In the pipecat pipeline the first ``TTSAudioRawFrame`` is observed
        slightly before transport.output emits ``BotStartedSpeakingFrame``, so
        the start may already have been anchored (and bytes counted) by
        ``add_audio``. We therefore only (re)anchor the clock when no audio has
        landed yet this turn — never zeroing already-counted bytes."""
        if not self._turn_active:
            self.tts_started_at = time.monotonic() if now is None else now
            self.bot_audio_bytes = 0
            self._turn_active = True

    def on_bot_turn_end(self) -> None:
        """Mark the current bot turn finished so the next audio frame starts a
        fresh measurement. Called on BotStoppedSpeaking / user interruption."""
        self._turn_active = False

    def add_audio(self, n_bytes: int, now: float | None = None) -> None:
        """Accumulate output-audio bytes for the current bot turn. The first
        frame of a new turn (no active turn yet) anchors the start clock and
        resets the counter, so a turn is measured from its own first sample
        regardless of whether BotStartedSpeakingFrame was seen first."""
        if not self._turn_active:
            self.tts_started_at = time.monotonic() if now is None else now
            self.bot_audio_bytes = 0
            self._turn_active = True
        self.bot_audio_bytes += n_bytes

    @property
    def bot_audio_secs(self) -> float:
        """Cumulative playout duration of the current bot turn, in seconds."""
        denom = self._rate * 2 * self._channels
        return self.bot_audio_bytes / denom if denom else 0.0

    def playout_finish_at(self) -> float | None:
        """Estimated monotonic time the current turn's audio finishes playing,
        or None if no bot audio has been seen this turn."""
        if self.tts_started_at is None:
            return None
        return self.tts_started_at + self.bot_audio_secs

    def remaining(self, now: float | None = None) -> float:
        """Seconds of bot audio estimated still unplayed (<=0 means done)."""
        fin = self.playout_finish_at()
        if fin is None:
            return 0.0
        return fin - (time.monotonic() if now is None else now)


class EventBroadcaster(BaseObserver):
    """Forward interesting pipeline events to a fan-out emit callable.

    Each event becomes a JSON-serializable dict that the ``emit`` callback
    decides what to do with (send to a single ws, fan out to multiple, log,
    etc.). Deduplicates per ``frame.id`` so each frame is emitted once even
    though observers see it cross many processor boundaries.
    """

    _WATCHED = (
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
        InterimTranscriptionFrame,
        TranscriptionFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
        LLMFullResponseEndFrame,
        TTSStartedFrame,
        TTSStoppedFrame,
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
    )

    def __init__(
        self,
        emit,
        start_time: float,
        history_recorder: "HistoryRecorder | None" = None,
        call_id: str | None = None,
        playout_tracker: "BotPlayoutTracker | None" = None,
    ):
        """Args: emit: async callable taking the event dict (already enriched
        with `t`); start_time: monotonic clock zero for relative timestamps;
        history_recorder + call_id: optional sink for transcript persistence;
        playout_tracker: optional sink fed bot-output audio bytes + turn
        boundaries so the idle-nudge timer can estimate true playout-finish."""
        super().__init__()
        self._emit = emit
        self._start = start_time
        self._seen: set[int] = set()
        self._llm_buffer = ""
        self._recorder = history_recorder
        self._call_id = call_id
        self._playout = playout_tracker

    async def emit_external(self, payload: dict) -> None:
        """Public emit for non-frame events (e.g. LLM tool calls). Stamps `t`
        the same way ``_send`` does so monitor consumers see consistent
        timestamps."""
        payload["t"] = round(time.monotonic() - self._start, 3)
        try:
            await self._emit(payload)
        except Exception as e:
            logger.debug(f"emit_external failed: {e}")

    async def _send(self, payload: dict) -> None:
        payload["t"] = round(time.monotonic() - self._start, 3)
        try:
            await self._emit(payload)
        except Exception as e:
            logger.debug(f"event emit failed: {e}")
        # Tee final turns to history. Failures here must never break the
        # live monitor stream, so we swallow exceptions at debug level.
        if self._recorder is not None and self._call_id is not None:
            try:
                ptype = payload.get("type")
                if ptype == "asr_final":
                    self._recorder.append(
                        self._call_id,
                        {"who": "user", "text": payload.get("text", ""), "t": payload["t"]},
                    )
                elif ptype == "llm_end":
                    self._recorder.append(
                        self._call_id,
                        {"who": "bot", "text": payload.get("text", ""), "t": payload["t"]},
                    )
            except Exception as e:
                logger.debug(f"history tee failed: {e}")

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame

        # Feed the playout tracker (pipeline path). TTSAudioRawFrame is the
        # bot's generated output audio; we sum its bytes per bot turn so the
        # idle-nudge handler can estimate when the audio actually finishes
        # PLAYING (not when TTS finishes GENERATING). Done before the _WATCHED
        # gate below since audio frames are intentionally not broadcast as
        # monitor events. Dedup by frame.id so a frame crossing multiple
        # processor boundaries is counted once.
        if self._playout is not None and isinstance(frame, TTSAudioRawFrame):
            if frame.id not in self._seen:
                self._seen.add(frame.id)
                self._playout.add_audio(len(frame.audio))

        if not isinstance(frame, self._WATCHED):
            return

        # Pipecat assigns each Frame a monotonic `id` via obj_id(); use that
        # instead of Python's id(), which can reuse memory addresses after GC.
        fid = frame.id
        if fid in self._seen:
            return
        self._seen.add(fid)

        # A new bot speaking turn starts: anchor the playout clock and reset
        # the per-turn byte counter so bot_audio_secs reflects only this turn.
        if self._playout is not None and isinstance(frame, BotStartedSpeakingFrame):
            self._playout.on_bot_turn_start()

        # End the playout measurement when the bot stops (TTSStopped-derived
        # BotStopped) or when the user barges in. The next bot audio frame then
        # starts a fresh per-turn measurement.
        if self._playout is not None and isinstance(
            frame, (BotStoppedSpeakingFrame, UserStartedSpeakingFrame)
        ):
            self._playout.on_bot_turn_end()

        if isinstance(frame, UserStartedSpeakingFrame):
            await self._send({"type": "user_speaking", "value": True})
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._send({"type": "user_speaking", "value": False})
        elif isinstance(frame, InterimTranscriptionFrame):
            await self._send({"type": "asr_partial", "text": frame.text})
        elif isinstance(frame, TranscriptionFrame):
            await self._send({"type": "asr_final", "text": frame.text})
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._llm_buffer = ""
            await self._send({"type": "llm_start"})
        elif isinstance(frame, LLMTextFrame):
            self._llm_buffer += frame.text
            await self._send({"type": "llm_delta", "text": frame.text})
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._send({"type": "llm_end", "text": self._llm_buffer})
            self._llm_buffer = ""
        elif isinstance(frame, TTSStartedFrame):
            await self._send({"type": "tts_start"})
        elif isinstance(frame, TTSStoppedFrame):
            await self._send({"type": "tts_end"})
        elif isinstance(frame, BotStartedSpeakingFrame):
            await self._send({"type": "bot_speaking", "value": True})
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._send({"type": "bot_speaking", "value": False})


class RawPCMSerializer(FrameSerializer):
    """Pass PCM16 mono bytes unchanged in both directions."""

    def __init__(self, input_sample_rate: int, output_sample_rate: int):
        super().__init__()
        self._input_sample_rate = input_sample_rate
        self._output_sample_rate = output_sample_rate

    async def serialize(self, frame: Frame) -> bytes | None:
        if self.should_ignore_frame(frame):
            return None
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if not isinstance(data, (bytes, bytearray)):
            return None
        if not data:
            return None
        return InputAudioRawFrame(
            audio=bytes(data),
            sample_rate=self._input_sample_rate,
            num_channels=1,
        )


def _upsample_pcm16_linear(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolation upsample of mono PCM16LE from src_rate to dst_rate.

    Mirrors voice-server's audio-utils.resample. Used only on the Nova Sonic
    phone path (8 kHz wire → 16 kHz for the S2S model); the pipeline/Transcribe
    path deliberately does NOT upsample (Transcribe gets native 8 kHz). Kept
    simple on purpose — Nova Sonic is end-to-end and far less sensitive to
    upsample artifacts than a discrete ASR, and the phone-nova combo is an edge
    case (nova-sonic can't do zh-HK/ja anyway). src==dst returns input unchanged.
    """
    if src_rate == dst_rate or not pcm:
        return pcm
    import array

    src = array.array("h")
    src.frombytes(pcm[: (len(pcm) // 2) * 2])
    n_src = len(src)
    if n_src == 0:
        return b""
    n_dst = (n_src * dst_rate) // src_rate
    out = array.array("h", bytes(n_dst * 2))
    for i in range(n_dst):
        pos = (i * src_rate) / dst_rate
        idx = int(pos)
        frac = pos - idx
        s0 = src[min(idx, n_src - 1)]
        s1 = src[min(idx + 1, n_src - 1)]
        val = int(round(s0 + frac * (s1 - s0)))
        out[i] = max(-32768, min(32767, val))
    return out.tobytes()


class UpsamplingPCMSerializer(RawPCMSerializer):
    """RawPCMSerializer that upsamples inbound wire audio before it enters the
    pipeline. Nova Sonic v2 requires 16 kHz input, but voice-server now sends the
    phone uplink as native 8 kHz — so on the nova-sonic phone path we upsample
    8 kHz → 16 kHz here and tag the frame at the frame (pipeline) rate.

    ``wire_rate`` = the sample rate of bytes arriving on the WebSocket (8000).
    ``input_sample_rate`` (from RawPCMSerializer) = the pipeline/frame rate the
    upsampled audio is tagged with (16000). Outbound (serialize) is unchanged.
    """

    def __init__(self, wire_rate: int, input_sample_rate: int, output_sample_rate: int):
        super().__init__(input_sample_rate, output_sample_rate)
        self._wire_rate = wire_rate

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if not isinstance(data, (bytes, bytearray)) or not data:
            return None
        pcm = _upsample_pcm16_linear(bytes(data), self._wire_rate, self._input_sample_rate)
        return InputAudioRawFrame(
            audio=pcm,
            sample_rate=self._input_sample_rate,
            num_channels=1,
        )


def _build_tts(
    provider: str,
    voice_key: str,
    *,
    region: str,
    frozen,
    aio_session: aiohttp.ClientSession,
    minimax_model: str | None = None,
    lang_key: str | None = None,
):
    """Return a Pipecat TTS service for the chosen provider."""
    if provider == "polly":
        voice_id, engine = _resolve_polly_voice(voice_key)
        logger.info(f"TTS: Polly voice={voice_id} engine={engine} (simple impl)")
        return SimplePollyTTSService(
            voice=voice_id,
            engine=engine,
            region=region,
            aws_access_key_id=frozen.access_key,
            aws_secret_access_key=frozen.secret_key,
            aws_session_token=frozen.token,
            sample_rate=OUTPUT_SAMPLE_RATE,
        )

    voice_id, language_boost = _resolve_minimax_voice(voice_key, lang_key)
    model_id = minimax_model if minimax_model in MINIMAX_MODELS else DEFAULT_MINIMAX_MODEL
    logger.info(f"TTS: MiniMax voice={voice_id} boost={language_boost} model={model_id}")
    tts_settings = MiniMaxTTSSettings(
        model=model_id,
        voice=voice_id,
    )
    tts_settings.language_boost = language_boost
    return SimpleMiniMaxTTSService(
        api_key=os.environ["MINIMAX_API_KEY"],
        group_id=os.environ.get("MINIMAX_GROUP_ID", ""),
        base_url=os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.chat/v1/t2a_v2"),
        aiohttp_session=aio_session,
        sample_rate=OUTPUT_SAMPLE_RATE,
        stream=True,
        settings=tts_settings,
    )


def _resolve_demo_prompts(scenario_key: str, lang_key: str) -> tuple[str, str] | None:
    """Return ``(system, greeting)`` from the matching disk demo, or None.

    The KB body itself is *not* in the system prompt — it is injected into
    the chat context as a synthetic user/assistant pair (see
    ``_kb_seed_messages``) so Nova Sonic v2 fits within its chat-history
    limit.
    """
    demo = DEMO_LOADER.get(scenario_key)
    if not demo:
        return None
    sys_map = demo.get("system") or {}
    system = sys_map.get(lang_key) or sys_map.get(DEFAULT_LANG) or ""
    if not system:
        return None
    greet_map = demo.get("greeting") or {}
    greeting = greet_map.get(lang_key) or greet_map.get(DEFAULT_LANG) or ""
    return system, greeting


def _resolve_demo_tools(demo_id: str, scope: str) -> list:
    """Resolve the LLM tool definitions enabled for ``demo_id`` on ``scope``.

    Returns an empty list — and skips registry lookup entirely — when:

    - ``PHONE_TOOLS_ENABLED`` is OFF (global kill switch retained from the
      phone-only era; despite the name it now applies to both channels).
    - ``demo_id`` is :data:`DEFAULT_DEMO_ID` ("default" — no demo
      selected, so no opt-in tools).
    - ``demo_id`` is unknown to :data:`DEMO_LOADER`.

    Otherwise returns ``tools.registry.get_tool_defs(demo["tool_ids"], scope)``,
    which itself filters out unknown ids and ids whose registered ``scope``
    does not include the requested channel. The result is a list of
    :class:`tools.registry.ToolDefinition`; both pipeline construction
    and ``_resolve_system_greeting`` consume it.

    Always logs exactly one ``[tools] demo=… scope=… registered=[…]`` line
    so smoke tests can assert the active tool set.
    """
    from tools.registry import get_tool_defs

    if not PHONE_TOOLS_ENABLED:
        logger.info(f"[tools] demo={demo_id} scope={scope} registered=[] (kill switch)")
        return []
    if demo_id == DEFAULT_DEMO_ID:
        logger.info(f"[tools] demo={demo_id} scope={scope} registered=[] (default)")
        return []
    demo = DEMO_LOADER.get(demo_id)
    if not demo:
        logger.info(f"[tools] demo={demo_id} scope={scope} registered=[] (unknown demo)")
        return []
    defs = get_tool_defs(demo.get("tool_ids") or [], scope=scope)
    logger.info(
        f"[tools] demo={demo_id} scope={scope} registered={[d.id for d in defs]}"
    )
    return defs


def _resolve_system_greeting(
    lang_key: str,
    scenario_key: str,
    system_override: str | None,
    greeting_override: str | None,
    tool_defs: list | None = None,
) -> tuple[str, str]:
    """Resolve which system prompt and greeting to use.

    Precedence (system): explicit user override > demo's system map >
    per-language default. When ``tool_defs`` is non-empty *and* no
    ``system_override`` is in effect, the matching multi-language hangup
    policy blurb (from ``tools.registry.assemble_policy_blurb``) is
    appended to the resolved system text with a blank-line separator.
    """
    from tools.registry import assemble_policy_blurb

    lang = LANGUAGES.get(lang_key) or LANGUAGES[DEFAULT_LANG]
    demo_pair = _resolve_demo_prompts(scenario_key, lang_key)
    if demo_pair:
        sys_sc, greet_sc = demo_pair
    else:
        sys_sc, greet_sc = None, None
    override_text = (system_override or "").strip()
    system_prompt = override_text or sys_sc or lang["prompt"]
    greeting = (greeting_override or "").strip() or greet_sc or lang["greeting"]
    if tool_defs and not override_text:
        blurb = assemble_policy_blurb(tool_defs, lang_key)
        if blurb:
            system_prompt = f"{system_prompt}\n\n{blurb}"
    return system_prompt, greeting


async def _connect_mcp_clients(scenario_key: str, scope: str) -> tuple[list, list]:
    """Connect the MCP servers a demo declares and collect their tool schemas.

    Returns ``(clients, schemas)`` where ``clients`` is the list of started
    :class:`MCPClient` instances (to be closed in the ws/phone ``finally`` via
    ``task._mcp_clients``) and ``schemas`` is ``[(client, [FunctionSchema, ...]), ...]``.

    Zero-overhead path: a demo with no ``mcp_servers`` returns ``([], [])`` on
    the first line — no imports, no awaits beyond this coroutine's own. A single
    server failing (missing/disabled config, connect timeout, bad transport) is
    logged at WARNING and skipped; it never aborts the call. If the python
    ``mcp`` package is unavailable the whole feature is disabled gracefully.
    """
    ids = (DEMO_LOADER.get(scenario_key) or {}).get("mcp_servers") or []
    if not ids:
        return [], []
    try:
        from pipecat.services.mcp_service import MCPClient
        from mcp.client.session_group import (
            SseServerParameters,
            StreamableHttpParameters,
        )
    except ImportError as e:
        logger.warning(f"[mcp] python 'mcp' package unavailable, MCP disabled: {e}")
        return [], []
    clients: list = []
    schemas: list = []
    for sid in ids:
        cfg = MCP_CONFIG.get(sid)
        if not cfg or not cfg.get("enabled"):
            logger.warning(f"[mcp] server {sid!r} missing/disabled, skipping")
            continue
        auth = cfg.get("auth") or {}
        if auth.get("type") == "sigv4":
            # SigV4-signed streamable-http (e.g. Bedrock AgentCore). Build a
            # StreamableHttpParameters subclass that injects an httpx SigV4 auth
            # into model_dump(), so pipecat's MCPClient signs every request with
            # the instance's IAM credentials — no pipecat patch, full reuse of
            # its SSE/double-parse/FunctionSchema handling. Missing credentials
            # degrade gracefully: skip this server with a WARNING, never crash.
            try:
                import mcp_sigv4
                params = mcp_sigv4.make_sigv4_streamable_params(
                    cfg["url"],
                    service=auth.get("service") or "bedrock-agentcore",
                    region=auth.get("region") or "us-east-1",
                    headers=cfg.get("headers") or None,
                )
            except mcp_sigv4.MissingCredentialsError as e:
                logger.warning(
                    f"[mcp] server {sid!r} sigv4 needs AWS credentials but none "
                    f"available, skipping: {e}"
                )
                continue
            except Exception as e:  # noqa: BLE001 — sigv4 setup must not crash the call
                logger.warning(f"[mcp] server {sid!r} sigv4 setup failed, skipping: {e!r}")
                continue
        else:
            params_cls = (
                SseServerParameters if cfg.get("transport") == "sse"
                else StreamableHttpParameters
            )
            params = params_cls(url=cfg["url"], headers=cfg.get("headers") or {})
        client = MCPClient(server_params=params)
        try:
            await asyncio.wait_for(client.start(), timeout=3.0)
            ts = await asyncio.wait_for(client.get_tools_schema(), timeout=3.0)
        # NOTE: catch BaseException, not Exception. An unreachable URL / the 3s
        # asyncio.wait_for timeout unwinds the anyio task group inside
        # streamablehttp_client, and client.start() surfaces the failure as a
        # bare asyncio.CancelledError (and/or a BaseExceptionGroup) — both of
        # which are BaseException, NOT Exception. A plain `except Exception`
        # lets them escape and aborts the whole call, regressing the contract
        # that an unreachable MCP server must degrade to a WARNING and never
        # crash the call. We deliberately do NOT re-raise CancelledError here:
        # in this connect helper it means "this server's connect attempt was
        # torn down", not "our task is being cancelled" — re-raising would
        # abort the call. client.close() is best-effort (its own anyio teardown
        # may log cross-task noise on a failed connect; that's harmless).
        except BaseException as e:  # noqa: BLE001 — see NOTE above
            logger.warning(f"[mcp] server {sid!r} connect failed, skipping: {e!r}")
            with contextlib.suppress(BaseException):
                await client.close()
            continue
        clients.append(client)
        schemas.append((client, ts.standard_tools))
        logger.info(f"[mcp] server {sid!r} connected, {len(ts.standard_tools)} tools")
    return clients, schemas


def _merge_mcp_tools(tools_schema, mcp_schemas):
    """Merge registry tool schemas with discovered MCP tool schemas.

    Registry tools win on name collisions (skipped + WARNING). Among MCP
    servers, first-seen name wins (later duplicates skipped + WARNING).

    ``tools_schema`` is the registry :class:`ToolsSchema` (or ``None``).
    ``mcp_schemas`` is ``[(client, [FunctionSchema, ...]), ...]``.

    Returns ``(combined_schema, kept_mcp)`` where ``combined_schema`` is a
    :class:`ToolsSchema` (or ``None`` when there are no tools at all — i.e.
    byte-identical to the pre-MCP no-tools path) and ``kept_mcp`` is the list
    of ``(client, FunctionSchema)`` pairs that survived merging, for
    ``llm.register_function``.
    """
    registry_schemas = list(tools_schema.standard_tools) if tools_schema else []
    registry_names = {fs.name for fs in registry_schemas}
    merged = list(registry_schemas)
    seen_mcp: set = set()
    kept_mcp: list = []
    for client, fns in mcp_schemas:
        for fs in fns:
            if fs.name in registry_names or fs.name in seen_mcp:
                logger.warning(
                    f"[mcp] tool {fs.name!r} conflicts, skipping (registry wins)"
                )
                continue
            merged.append(fs)
            seen_mcp.add(fs.name)
            kept_mcp.append((client, fs))
    if not merged:
        return None, []
    from pipecat.adapters.schemas.tools_schema import ToolsSchema
    return ToolsSchema(standard_tools=merged), kept_mcp


def _make_vad_analyzer(is_phone: bool) -> SileroVADAnalyzer:
    """Build the Silero VAD analyzer for a session.

    Web (browser) callers get the pipecat defaults (confidence=0.7,
    start_secs=0.2, stop_secs=0.2, min_volume=0.6) — browsers run their own
    echo cancellation / noise suppression, so the snappy defaults are fine.

    Phone (PSTN) callers have NO echo cancellation: the bot's own TTS leaks
    back over the line and a short start window lets that echo (and line
    noise) trip the VAD, which broadcasts an interruption and cuts the bot off
    mid-sentence. Since barge-in interruptions are ENABLED for phone, the
    thresholds are tightened so only sustained, louder, higher-confidence
    speech counts as the caller actually barging in (echo of the bot's own
    voice should not clear this bar):
      - start_secs 0.2 -> 0.5  (must hear ~0.5 s of speech before "user spoke")
      - min_volume 0.6 -> 0.8  (echo of the bot is usually quieter than the
                                caller speaking directly into the handset)
      - confidence 0.7 -> 0.8  (stricter voiced-speech probability)
    stop_secs stays 0.2 so the bot still responds promptly once the caller
    actually finishes.
    """
    if is_phone:
        return SileroVADAnalyzer(
            params=VADParams(
                confidence=0.8,
                start_secs=0.5,
                stop_secs=0.2,
                min_volume=0.8,
            )
        )
    return SileroVADAnalyzer()


def _nova_mcp_tool_wrapper(client):
    """Adapt an MCPClient tool proxy for the Nova Sonic result protocol.

    MCPClient._tool_wrapper delivers its result to the LLM as a *bare string*
    (mcp_service._call_tool concatenates the tool's text content and calls
    result_callback(<str>)). The Bedrock pipeline LLM tolerates that — it wraps
    the string as a {"text": ...} content block. Nova Sonic does NOT:
    its _send_tool_result (aws/nova_sonic/llm.py:902-911) only json-encodes the
    result when it's a dict, otherwise it ships the raw string, and Bedrock
    bidi rejects it with "Unsupported JSON type in Tool Result. Please provide
    the Tool Result as a JSON object" — which aborts the turn (verified live in
    T5 before this fix).

    So for the Nova Sonic engine we register a wrapper that swaps in a
    result_callback which coerces a non-dict result into a JSON object
    ({"result": <str>}) before it reaches the context / _send_tool_result. dict
    results (rare for MCP, but possible) pass through untouched. The Bedrock
    pipeline keeps using the raw _tool_wrapper (object-wrapping there would only
    add a needless nesting level).
    """

    async def _wrapper(params):
        orig_cb = params.result_callback

        async def _coercing_cb(result, **kwargs):
            if not isinstance(result, dict):
                result = {"result": result}
            await orig_cb(result, **kwargs)

        params.result_callback = _coercing_cb
        await client._tool_wrapper(params)

    return _wrapper


async def _build_nova_sonic_pipeline(
    websocket: WebSocket,
    lang_key: str,
    voice_key: str,
    system_override: str | None,
    greeting_override: str | None,
    scenario_key: str = DEFAULT_SCENARIO,
    event_emit: EmitFn | None = None,
    input_sample_rate: int = INPUT_SAMPLE_RATE,
    history_recorder: "HistoryRecorder | None" = None,
    call_id: str | None = None,
    is_phone: bool = False,
    serializer: FrameSerializer | None = None,
) -> PipelineTask:
    """End-to-end speech-to-speech via Nova Sonic v2.

    The pipeline skips Transcribe/TTS entirely — Nova Sonic streams audio in
    and streams audio out over a bidirectional Bedrock connection.
    """
    _scope = "phone" if is_phone else "web"
    _tool_defs = _resolve_demo_tools(scenario_key, scope=_scope)
    system_prompt, greeting = _resolve_system_greeting(
        lang_key, scenario_key, system_override, greeting_override,
        tool_defs=_tool_defs,
    )
    # Apply the alias migration FIRST (old bookmarked ids -> current), THEN
    # check membership against the registry (live table if loaded, else the
    # in-code NOVA_SONIC_VOICES). Don't drop the alias step.
    voice = NOVA_SONIC_VOICE_ALIASES.get(voice_key, voice_key)
    voice = voice if voice in voices_for("nova-sonic") else DEFAULT_NOVA_SONIC_VOICE

    logger.info(
        f"Building Nova Sonic pipeline: lang={lang_key} voice={voice} scenario={scenario_key} scope={_scope}"
    )

    region = os.environ.get("AWS_REGION", "us-east-1")
    frozen = boto3.Session(region_name=region).get_credentials().get_frozen_credentials()

    serializer = serializer or RawPCMSerializer(input_sample_rate, OUTPUT_SAMPLE_RATE)
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=input_sample_rate,
            audio_out_sample_rate=OUTPUT_SAMPLE_RATE,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    # LLM tools (end_call / transfer_to_human / ...). Resolved by
    # _resolve_demo_tools above; _tool_defs is empty when the demo
    # declares no tools, the registry has no scope match, or the global
    # kill switch is off.
    _tools_schema = None
    if _tool_defs:
        from tools.registry import assemble_tools_schema
        _tools_schema = assemble_tools_schema(_tool_defs)

    # MCP tool discovery: connect the demo's MCP servers (if any) and merge
    # their tools with the registry tools. Registry wins on name collisions
    # (security red line — a malicious/buggy MCP server cannot shadow
    # end_call / transfer_to_human / ...). Must complete before LLMContext is
    # constructed because tools are injected at context-construction time.
    _mcp_clients, _mcp_schemas = await _connect_mcp_clients(scenario_key, _scope)
    _combined_schema, _kept_mcp = _merge_mcp_tools(_tools_schema, _mcp_schemas)

    nova_kwargs = dict(
        access_key_id=frozen.access_key,
        secret_access_key=frozen.secret_key,
        session_token=frozen.token,
        region=region,
        settings=AWSNovaSonicLLMService.Settings(
            model="amazon.nova-2-sonic-v1:0",
            voice=voice,
            system_instruction=system_prompt,
        ),
    )
    if _combined_schema is not None:
        nova_kwargs["tools"] = _combined_schema
    llm = AWSNovaSonicLLMService(**nova_kwargs)

    # Register the kept MCP tools' call proxies on the LLM (registry tools are
    # registered separately via register_call_control_handlers below). Nova
    # Sonic needs the tool result delivered as a JSON object, so we wrap the
    # MCPClient proxy (see _nova_mcp_tool_wrapper) — the raw string the proxy
    # produces is rejected by the Nova Sonic bidi tool-result protocol.
    for _client, _fs in _kept_mcp:
        llm.register_function(_fs.name, with_filler(_nova_mcp_tool_wrapper(_client), _fs.name, lang_key))

    context = LLMContext(tools=_combined_schema) if _combined_schema is not None else LLMContext()
    # Seed KB scenario: inject the document as a synthetic user/assistant turn
    # so the model has it in chat history without inflating the system prompt.
    kb_seed = _kb_seed_messages(scenario_key, lang_key)
    if kb_seed:
        for m in kb_seed:
            context.add_message(m)
        logger.info(f"KB scenario {scenario_key}: seeded {len(kb_seed[0]['content'])} chars into context")

    # SileroVADAnalyzer fires UserStartedSpeakingFrame the instant the caller
    # opens their mouth — that's what voice-server uses to mute outbound TTS
    # for barge-in. Without it the only interruption signal is Nova Sonic's
    # server-side user-transcription-started event, which arrives too late
    # (or not at all while the bot is still speaking).
    #
    # Replace Pipecat's default SmartTurn turn-stop strategy with a simple
    # speech-timeout. SmartTurn runs an ONNX model that adds 1-1.5 s of
    # "is the user really done" wait and produced visible 3 s round-trips on
    # phone calls. Nova Sonic does its own server-side turn finalization, so
    # a 400 ms VAD-silence timeout is enough on this side.
    #
    # Turn-START: local VAD broadcasts interruptions so the caller can barge in
    # on the bot (BOTH web and phone). An earlier version disabled interruptions
    # here on the theory that Nova Sonic's server-side barge-in would take over —
    # but logs showed it does NOT cut the bot off, so the net effect was "can't
    # interrupt at all". Re-enabled. The PSTN echo false-trigger risk (the bot's
    # own TTS leaking back over the line with no AEC, tripping a volume-based VAD
    # interruption) is mitigated by the tightened phone VAD thresholds in
    # _make_vad_analyzer(is_phone), not by killing interruptions outright.
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=_make_vad_analyzer(is_phone),
            user_turn_strategies=UserTurnStrategies(
                start=[
                    VADUserTurnStartStrategy(
                        enable_interruptions=True,
                        enable_user_speaking_frames=True,
                    ),
                ],
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.4)],
            ),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            user_agg,
            llm,
            transport.output(),
            assistant_agg,
        ]
    )

    _primary_emit = event_emit or _ws_emit(websocket)
    observer = EventBroadcaster(
        _primary_emit,
        start_time=time.monotonic(),
        history_recorder=history_recorder,
        call_id=call_id,
    )
    lat = LatencyObserver(
        emit=_primary_emit,
        call_id=call_id,
        engine="nova-sonic",
        scenario=scenario_key,
        lang=lang_key,
    )
    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=[observer, lat],
    )

    @transport.event_handler("on_client_connected")
    async def on_connected(_t, _client):
        logger.info("Client connected (nova-sonic)")
        context.add_message({"role": "developer", "content": greeting})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(_t, _client):
        logger.info("Client disconnected")
        await task.cancel()

    if _tools_schema is not None:
        from call_control_tools import register_call_control_handlers
        if _scope == "phone":
            _hist_append = (
                (lambda kind, data: history_recorder.append_event(call_id, kind, data))
                if history_recorder is not None and call_id is not None
                else None
            )
            _mark_transfer = (
                (lambda topic: history_recorder.mark_transfer(call_id, topic))
                if history_recorder is not None and call_id is not None
                else None
            )
            _write_outcome = (
                (lambda: history_recorder.write_outcome_row(call_id))
                if history_recorder is not None and call_id is not None
                else None
            )
        else:
            # Web scope: no DDB writes, no SIP BYE — handlers must be None
            # so call_control_tools._handle_* skip those side-effects.
            _hist_append = None
            _mark_transfer = None
            _write_outcome = None
        register_call_control_handlers(
            llm, task,
            call_id=call_id or "unknown",
            emit=observer.emit_external,
            scope=_scope,
            history_append=_hist_append,
            mark_transfer=_mark_transfer,
            write_outcome=_write_outcome,
        )

    task._aio_session = None  # nothing to close for Nova Sonic
    task._mcp_clients = _mcp_clients  # closed in ws_endpoint / phone finally
    return task


async def _build_connect_a2a_nova_engine(session: ConnectA2ASession) -> NovaSonicSessionEngine:
    """Resolve the existing phone persona into a Connect-safe Nova engine.

    The external A2A path intentionally enables only MCP tools that it
    registers itself. Legacy phone call-control schemas require a PipelineTask
    callback contract and are therefore not advertised on this new path.
    """
    defaults = RUNTIME_CONFIG.get_phone_defaults()
    if defaults.get("engine", DEFAULT_ENGINE) != "nova-sonic":
        raise RuntimeError("Amazon Connect A2A requires PHONE_ENGINE=nova-sonic")

    lang_key = defaults.get("lang", DEFAULT_LANG)
    scenario_key = defaults.get("scenario", DEFAULT_SCENARIO)
    voice_key = defaults.get("voice", DEFAULT_NOVA_SONIC_VOICE)
    voice = NOVA_SONIC_VOICE_ALIASES.get(voice_key, voice_key)
    if voice not in voices_for("nova-sonic"):
        voice = DEFAULT_NOVA_SONIC_VOICE
    system_prompt, greeting = _resolve_system_greeting(
        lang_key,
        scenario_key,
        None,
        None,
        tool_defs=[],
    )

    region = os.environ.get("AWS_REGION", "us-east-1")
    credentials = boto3.Session(region_name=region).get_credentials()
    if credentials is None:
        raise RuntimeError("AWS credentials are unavailable")
    frozen = credentials.get_frozen_credentials()

    mcp_clients: list[object] = []
    try:
        mcp_clients, mcp_schemas = await _connect_mcp_clients(scenario_key, "phone")
        tools_schema, kept_mcp = _merge_mcp_tools(None, mcp_schemas)
        mcp_functions = tuple(
            (
                function_schema.name,
                with_filler(
                    _nova_mcp_tool_wrapper(client),
                    function_schema.name,
                    lang_key,
                ),
            )
            for client, function_schema in kept_mcp
        )
        return NovaSonicSessionEngine(
            NovaSonicEngineConfig(
                region=region,
                access_key_id=frozen.access_key,
                secret_access_key=frozen.secret_key,
                session_token=frozen.token,
                voice=voice,
                system_instruction=system_prompt,
                greeting=greeting,
                input_format=session.input_format,
                output_format=session.output_format,
                tools=tools_schema,
                seed_messages=tuple(_kb_seed_messages(scenario_key, lang_key) or ()),
                vad_analyzer=_make_vad_analyzer(is_phone=True),
                mcp_clients=tuple(mcp_clients),
                mcp_functions=mcp_functions,
            )
        )
    except Exception:
        for client in mcp_clients:
            with contextlib.suppress(Exception):
                await client.close()
        raise


async def _build_pipeline(
    websocket: WebSocket,
    lang_key: str,
    model_key: str,
    voice_key: str,
    provider: str,
    system_override: str | None = None,
    greeting_override: str | None = None,
    scenario_key: str = DEFAULT_SCENARIO,
    minimax_model: str | None = None,
    event_emit: EmitFn | None = None,
    input_sample_rate: int = INPUT_SAMPLE_RATE,
    history_recorder: "HistoryRecorder | None" = None,
    call_id: str | None = None,
    is_phone: bool = False,
    serializer: FrameSerializer | None = None,
) -> PipelineTask:
    lang = LANGUAGES.get(lang_key) or LANGUAGES[DEFAULT_LANG]
    # Defensive: Nova-only languages have stt=None and must never drive the
    # pipeline (the PUT validation blocks pipeline + a stt=None lang). If one
    # somehow reaches here (stale runtime.json, direct call), fall back to the
    # default language so Transcribe gets a valid LanguageCode instead of None.
    if lang.get("stt") is None:
        logger.warning(
            "build_pipeline_task: lang=%s has no STT (nova-only); "
            "falling back to %s for the pipeline path",
            lang_key, DEFAULT_LANG,
        )
        lang = LANGUAGES[DEFAULT_LANG]
    model_id = MODELS.get(model_key) or MODELS[DEFAULT_MODEL]
    provider = provider if provider in TTS_PROVIDERS else DEFAULT_PROVIDER

    _scope = "phone" if is_phone else "web"
    _tool_defs = _resolve_demo_tools(scenario_key, scope=_scope)
    system_prompt, greeting = _resolve_system_greeting(
        lang_key, scenario_key, system_override, greeting_override,
        tool_defs=_tool_defs,
    )
    logger.info(
        f"Building pipeline: lang={lang_key} model={model_id} "
        f"tts={provider} voice={voice_key} scenario={scenario_key} "
        f"minimax_model={minimax_model} in_sr={input_sample_rate} scope={_scope}"
    )

    region = os.environ.get("AWS_REGION", "us-east-1")
    frozen = boto3.Session(region_name=region).get_credentials().get_frozen_credentials()

    serializer = serializer or RawPCMSerializer(input_sample_rate, OUTPUT_SAMPLE_RATE)
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=input_sample_rate,
            audio_out_sample_rate=OUTPUT_SAMPLE_RATE,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    stt = AWSTranscribeSTTService(
        region=region,
        aws_access_key_id=frozen.access_key,
        api_key=frozen.secret_key,
        aws_session_token=frozen.token,
        sample_rate=input_sample_rate,
        settings=AWSTranscribeSTTService.Settings(language=lang["stt"]),
    )

    llm_settings = AWSBedrockLLMService.Settings(
        model=model_id,
        system_instruction=system_prompt,
        max_tokens=256,
    )
    if _is_claude_model_id(model_id):
        # Claude on Bedrock honors cachePoint markers; cuts TTFT ~85% on multi-turn
        # conversations where the system prompt (KB seed) stays constant. Nova
        # rejects cachePoint with a 400, so we keep this Claude-only.
        llm_settings.enable_prompt_caching = True
    llm = AWSBedrockLLMService(
        aws_region=region,
        aws_access_key=frozen.access_key,
        aws_secret_key=frozen.secret_key,
        aws_session_token=frozen.token,
        settings=llm_settings,
    )

    # Always create an aiohttp session; MiniMax uses it, Polly just ignores it.
    aio_session = aiohttp.ClientSession()
    tts = _build_tts(
        provider, voice_key,
        region=region, frozen=frozen, aio_session=aio_session,
        minimax_model=minimax_model,
        lang_key=lang_key,
    )

    # LLM tools (end_call / transfer_to_human / ...). Resolved by
    # _resolve_demo_tools above; _tool_defs is empty when the demo
    # declares no tools, the registry has no scope match, or the global
    # kill switch is off.
    _tools_schema = None
    if _tool_defs:
        from tools.registry import assemble_tools_schema
        _tools_schema = assemble_tools_schema(_tool_defs)

    # MCP tool discovery (see _build_nova_sonic_pipeline for the rationale).
    # Registry tools win on name collisions. Must complete before LLMContext.
    _mcp_clients, _mcp_schemas = await _connect_mcp_clients(scenario_key, _scope)
    _combined_schema, _kept_mcp = _merge_mcp_tools(_tools_schema, _mcp_schemas)
    for _client, _fs in _kept_mcp:
        llm.register_function(_fs.name, with_filler(_client._tool_wrapper, _fs.name, lang_key))

    context = LLMContext(tools=_combined_schema) if _combined_schema is not None else LLMContext()
    kb_seed = _kb_seed_messages(scenario_key, lang_key)
    if kb_seed:
        for m in kb_seed:
            context.add_message(m)
        logger.info(f"KB scenario {scenario_key}: seeded {len(kb_seed[0]['content'])} chars into context")

    # Same turn-stop tweak as the Nova Sonic path: replace Pipecat's default
    # SmartTurn ONNX (which adds 1-1.5 s waiting for "is the user really
    # done") with a fixed 400 ms VAD-silence timeout. Cuts ~1 s off every
    # caller turn in the STT+LLM+TTS pipeline.
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=_make_vad_analyzer(is_phone),
            user_turn_strategies=UserTurnStrategies(
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.4)],
            ),
            user_idle_timeout=IDLE_NUDGE_TIMEOUT,
        ),
    )

    # Drop AWS Transcribe hallucinations (short, low-confidence filler words like
    # うん/は/そう that it invents on near-silent/noise segments VAD mis-triggers
    # on) BEFORE they reach the aggregator/LLM and derail the conversation. Sits
    # only on this STT-based pipeline; Nova Sonic has no TranscriptionFrame.
    # Per-demo filler config (tech_design §3): read the current demo's
    # ``filler`` block None-safely and pass each field through. A field that is
    # absent stays None, so the processor's existing _env_* fallback applies
    # (demo > env > built-in default). A demo with no ``filler`` block → all
    # None → full env fallback (default OFF) → zero behaviour change.
    _demo = DEMO_LOADER.get(scenario_key) or {}
    _filler_cfg = _demo.get("filler") or {}
    # Resolve the ASR hallucination filter config for THIS call's scope: phone
    # calls read phone defaults, web calls read web defaults (tech_design §3).
    # Per-field precedence demo > segment > env > built-in(OFF); maps config key
    # names max_chars/max_words → ctor names max_cjk_chars/max_latin_words. With
    # no override anywhere → enabled=False → the always-inserted processor no-ops.
    _asr_seg = (
        RUNTIME_CONFIG.get_phone_defaults() if is_phone
        else RUNTIME_CONFIG.get_web_defaults()
    )
    _asr_filter_cfg = _resolve_asr_filter(_demo, _asr_seg)
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            TranscriptHallucinationFilter(**_asr_filter_cfg),
            user_agg,
            llm,
            tts,
            transport.output(),
            assistant_agg,
        ]
    )

    # Tracks bot output-audio bytes per turn so the idle-nudge timer can model
    # when the audio actually finishes PLAYING (see BotPlayoutTracker). Channels
    # is 1 (mono PCM16) for both web and phone output.
    playout = BotPlayoutTracker(out_sample_rate=OUTPUT_SAMPLE_RATE, channels=1)

    _primary_emit = event_emit or _ws_emit(websocket)
    observer = EventBroadcaster(
        _primary_emit,
        start_time=time.monotonic(),
        history_recorder=history_recorder,
        call_id=call_id,
        playout_tracker=playout,
    )
    lat = LatencyObserver(
        emit=_primary_emit,
        call_id=call_id,
        engine="pipeline",
        scenario=scenario_key,
        lang=lang_key,
    )
    # Timeout filler is now an OBSERVER (was an in-pipeline processor): it must
    # see the downstream TTSStartedFrame to cancel before injecting, which an
    # upstream processor cannot (tech_design §0-§1). PipelineTask takes its
    # observers list at construction and has no post-construction add API, so
    # build the observer first with task=None, include it in observers=[...],
    # then inject the task reference right after the task is constructed.
    filler_obs = TimeoutFillerObserver(
        task=None,
        lang_key=lang_key,
        enabled=_filler_cfg.get("enabled"),
        timeout_ms=_filler_cfg.get("timeout_ms"),
        probability=_filler_cfg.get("probability"),
        phrases=_filler_cfg.get("phrases"),  # per-demo override pool (§4.1)
    )
    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=[observer, lat, filler_obs],
    )
    filler_obs.task = task

    # Idle nudge: when the bot stops speaking and the user stays silent for
    # IDLE_NUDGE_TIMEOUT, re-run the LLM so it continues on its own (fixes the
    # "請稍等" deadlock where both sides wait for each other). A real user turn
    # resets the counter; after IDLE_NUDGE_MAX consecutive nudges we go quiet.
    #
    # Timing fix: pipecat fires BotStoppedSpeaking (which arms this idle timer)
    # when TTS finishes GENERATING, not when the far end finishes PLAYING — the
    # output write loop has no real-time pacing, so on a long opening sentence
    # BotStopped (and thus the 4 s idle window) starts 2-3 s early and the bot
    # nudges itself mid-utterance. We therefore treat IDLE_NUDGE_TIMEOUT as
    # silence measured from the *estimated playout-finish* time. When the timer
    # fires but the bot is (estimated) still audible, we don't nudge and don't
    # count it; instead we schedule a one-shot re-check shortly after playout
    # finishes. Any UserStartedSpeaking cancels the pending re-check.
    idle_nudge_count = 0
    pending_recheck: "asyncio.Task | None" = None
    RECHECK_MARGIN = 0.3

    def _cancel_pending_recheck() -> None:
        nonlocal pending_recheck
        if pending_recheck is not None and not pending_recheck.done():
            pending_recheck.cancel()
        pending_recheck = None

    async def _do_nudge() -> None:
        nonlocal idle_nudge_count
        idle_nudge_count += 1
        logger.info(
            f"[idle-nudge] call_id={call_id} count={idle_nudge_count}/{IDLE_NUDGE_MAX} "
            f"user silent {IDLE_NUDGE_TIMEOUT}s after bot playout finished, re-running LLM"
        )
        nudge_prompt = IDLE_NUDGE_PROMPTS.get(lang_key) or IDLE_NUDGE_PROMPTS["en-US"]
        context.add_message({"role": "developer", "content": nudge_prompt})
        await task.queue_frames([LLMRunFrame()])

    async def _recheck_after_playout(delay: float) -> None:
        # Wait until the bot's audio is estimated done, then nudge iff the user
        # is still silent (no UserStartedSpeaking cancelled us) and we're truly
        # past playout. Cancellation (user spoke / cleanup) just exits.
        nonlocal pending_recheck
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        pending_recheck = None
        remaining = playout.remaining()
        if remaining > 0:
            # Bot produced more audio while we waited; defer again.
            pending_recheck = asyncio.create_task(
                _recheck_after_playout(remaining + RECHECK_MARGIN)
            )
            return
        if idle_nudge_count >= IDLE_NUDGE_MAX:
            logger.info(
                f"[idle-nudge] call_id={call_id} max consecutive nudges reached "
                f"({IDLE_NUDGE_MAX}) at re-check, staying quiet until the user speaks"
            )
            return
        await _do_nudge()

    @user_agg.event_handler("on_user_turn_started")
    async def _user_started_cancels_recheck(_agg, _strategy):
        # The user spoke: kill any deferred nudge so we never talk over them.
        # NOTE: pipecat 1.3.0 emits on_user_turn_started as (aggregator, strategy)
        # — 2 args. A third param here raises "missing 1 required positional
        # argument" in the handler and breaks the turn/interruption path.
        _cancel_pending_recheck()

    @user_agg.event_handler("on_user_turn_stopped")
    async def _reset_idle_nudge(_agg, _strategy, _message):
        nonlocal idle_nudge_count
        idle_nudge_count = 0
        _cancel_pending_recheck()

    @user_agg.event_handler("on_user_turn_idle")
    async def _on_user_turn_idle(_agg):
        nonlocal pending_recheck
        # The bot may (estimated) still be playing: the 4 s idle window began at
        # the early TTS-generation-done BotStopped, not at playout-finish.
        remaining = playout.remaining()
        if remaining > 0:
            # Don't nudge, don't count it. Re-check just after playout finishes;
            # only then does the IDLE_NUDGE_TIMEOUT silence truly start. The
            # controller fires on_user_turn_idle only once per BotStopped, so we
            # must actively reschedule rather than wait for another fire.
            logger.info(
                f"[idle-nudge] call_id={call_id} idle fired but bot still playing "
                f"(~{remaining:.2f}s left); deferring nudge until playout finishes"
            )
            _cancel_pending_recheck()
            pending_recheck = asyncio.create_task(
                _recheck_after_playout(remaining + RECHECK_MARGIN)
            )
            return
        if idle_nudge_count >= IDLE_NUDGE_MAX:
            logger.info(
                f"[idle-nudge] call_id={call_id} max consecutive nudges reached "
                f"({IDLE_NUDGE_MAX}), staying quiet until the user speaks"
            )
            return
        await _do_nudge()

    @transport.event_handler("on_client_connected")
    async def on_connected(_t, _client):
        logger.info("Client connected")
        context.add_message({"role": "developer", "content": greeting})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(_t, _client):
        logger.info("Client disconnected")
        _cancel_pending_recheck()
        await task.cancel()

    if _tools_schema is not None:
        from call_control_tools import register_call_control_handlers
        if _scope == "phone":
            _hist_append = (
                (lambda kind, data: history_recorder.append_event(call_id, kind, data))
                if history_recorder is not None and call_id is not None
                else None
            )
            _mark_transfer = (
                (lambda topic: history_recorder.mark_transfer(call_id, topic))
                if history_recorder is not None and call_id is not None
                else None
            )
            _write_outcome = (
                (lambda: history_recorder.write_outcome_row(call_id))
                if history_recorder is not None and call_id is not None
                else None
            )
        else:
            # Web scope: no DDB writes, no SIP BYE — handlers must be None
            # so call_control_tools._handle_* skip those side-effects.
            _hist_append = None
            _mark_transfer = None
            _write_outcome = None
        register_call_control_handlers(
            llm, task,
            call_id=call_id or "unknown",
            emit=observer.emit_external,
            scope=_scope,
            history_append=_hist_append,
            mark_transfer=_mark_transfer,
            write_outcome=_write_outcome,
        )

    task._aio_session = aio_session  # cleaned up in ws_endpoint finally
    task._mcp_clients = _mcp_clients  # closed in ws_endpoint / phone finally
    return task


app = FastAPI()
APP_READINESS = ReadinessState(
    os.environ.get("SERVICE_NAME", "voice-agent-ac").strip() or "voice-agent-ac"
)
APPLICATION_SETTINGS = ApplicationSettings.from_env()
CONNECT_A2A_ADAPTER = ConnectA2AAdapter(
    api_key=APPLICATION_SETTINGS.connect_a2a_api_key,
    readiness=APP_READINESS,
)


@app.get("/health/live", include_in_schema=False)
async def health_live() -> dict[str, object]:
    """Unauthenticated process liveness for load balancers and supervisors."""
    return APP_READINESS.liveness()


@app.get("/health/ready", include_in_schema=False)
async def health_ready() -> Response:
    """Unauthenticated cached startup readiness; performs no network probes."""
    payload = APP_READINESS.snapshot()
    return Response(
        content=json.dumps(payload),
        status_code=200 if payload["ok"] else 503,
        media_type="application/json",
    )


@app.websocket("/connect/a2a")
async def connect_a2a_websocket_endpoint(websocket: WebSocket) -> None:
    """Amazon Connect external A2A voice collaborator endpoint."""
    await handle_connect_a2a_websocket(
        websocket,
        adapter=CONNECT_A2A_ADAPTER,
        engine_factory=_build_connect_a2a_nova_engine,
    )


STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


# --- Runtime config + Admin (T3 — config/demo separation) ----------------
# Defaults captured at import time from existing module constants + PHONE_* env.
# These remain the authoritative fallback if config/runtime.json is missing or
# any field is absent. Per-call hot-reload: WS endpoints read RUNTIME_CONFIG at
# entry, in-flight calls keep their captured snapshot.

from runtime_config import RuntimeConfig

_RUNTIME_FALLBACK = {
    "web": {
        "lang": DEFAULT_LANG,
        "engine": DEFAULT_ENGINE,
        "scenario": DEFAULT_SCENARIO,
        "model": DEFAULT_MODEL,
        "provider": DEFAULT_PROVIDER,
        "voice": DEFAULT_MINIMAX_VOICE,
        "minimax_model": DEFAULT_MINIMAX_MODEL,
    },
    "phone": {
        "engine": os.environ.get("PHONE_ENGINE", DEFAULT_ENGINE),
        "lang": os.environ.get("PHONE_LANG", DEFAULT_LANG),
        "scenario": os.environ.get("PHONE_SCENARIO", DEFAULT_SCENARIO),
        "voice": os.environ.get("PHONE_VOICE", DEFAULT_MINIMAX_VOICE),
        "provider": os.environ.get("PHONE_PROVIDER", DEFAULT_PROVIDER),
        "model": os.environ.get("PHONE_MODEL", DEFAULT_MODEL),
        "minimax_model": os.environ.get("PHONE_MINIMAX_MODEL", DEFAULT_MINIMAX_MODEL),
    },
}
RUNTIME_CONFIG = RuntimeConfig(
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "runtime.json"),
    fallback=_RUNTIME_FALLBACK,
)

# --- MCP server registry (global; demos mount servers via manifest
# `mcp_servers: [id, ...]`). File is gitignored — headers may hold secrets.
from mcp_config import McpConfig, mask_headers as _mask_mcp_headers

MCP_CONFIG = McpConfig(
    path=os.environ.get("MCP_CFG_PATH_OVERRIDE")
    or os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "mcp_servers.json"),
)

# ADMIN_PASSWORD is no longer a Basic-Auth gate — it survives ONLY as the
# first-boot seed for the bootstrap "admin" account (see _seed_admin below).
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "").strip()

# --- User store + role-based auth dependencies --------------------------
from botocore.exceptions import ClientError
from user_store import UserStore, AuthUnavailable

USER_STORE = UserStore()


async def require_user(vb_session: str | None = Cookie(default=None)) -> dict:
    """FastAPI dependency: resolve the logged-in user from the session cookie.

    Decodes the ``vb_session`` JWT, loads the user from the store, and rejects
    disabled / unknown / expired sessions with 401. Returns the safe user view
    ({username, role, created_at, disabled}).
    """
    claims = _decode_jwt(vb_session or "")
    if not claims or not claims.get("sub") or claims.get("ws"):
        raise HTTPException(status_code=401, detail="not authenticated")
    if claims.get("guest"):
        # Stateless guest session: a temporary experience link, no UserStore
        # row. role='guest' → require_admin 403s every /api/admin/* for free.
        return {
            "username": "guest",
            "role": "guest",
            "created_at": None,
            "disabled": False,
            "scenario": claims.get("scenario"),
        }
    user = await USER_STORE.get(claims["sub"])
    if user is None or user.get("disabled"):
        raise HTTPException(status_code=401, detail="not authenticated")
    return user


async def require_admin(user: dict = Depends(require_user)) -> dict:
    """FastAPI dependency: like require_user but requires role == 'admin'."""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="admin role required")
    return user


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/",
        max_age=AUTH_TTL_HOURS * 3600,
    )


async def _needs_setup() -> bool:
    """True when the user table holds no accounts (fresh deploy → first-run
    setup wizard). A missing table degrades to [] in USER_STORE.list(), so
    that also reports needs_setup=True."""
    return len(await USER_STORE.list()) == 0


async def _seed_admin() -> None:
    """Deprecated no-op. The ADMIN_PASSWORD first-boot seed has been removed:
    a CFN heredoc escaping bug corrupted special-character passwords before
    bcrypt-hashing them, so seeded admins could not log in. First-run setup is
    now handled interactively via POST /api/auth/setup. ADMIN_PASSWORD may
    still be present in the environment but is no longer used to create a user."""
    return None


# --- Auth API (public: login / setup-status / setup; cookie-auth: logout / me) ---------

@app.post("/api/auth/login")
async def auth_login(payload: dict, response: Response) -> dict:
    """Verify {username, password}, mint a session JWT, set the HttpOnly
    cookie, and return {username, role}. 401 on bad credentials."""
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    if not username or not password:
        # Detail carries ONLY the username (never the password) — the store's
        # default-deny allowlist for type 'login' is {username} regardless.
        await log_activity(
            actor=username or "anon", type="login", status="failure",
            target=username or None, detail={"username": username},
            error="missing credentials",
        )
        raise HTTPException(status_code=401, detail="invalid credentials")
    try:
        user = await USER_STORE.verify(username, password)
    except AuthUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    if user is None:
        await log_activity(
            actor=username, type="login", status="failure",
            target=username, detail={"username": username},
            error="invalid credentials",
        )
        raise HTTPException(status_code=401, detail="invalid credentials")
    token = _issue_jwt(user)
    _set_session_cookie(response, token)
    await log_activity(
        actor=user["username"], actor_role=user.get("role"), type="login",
        status="success", target=user["username"],
        detail={"username": user["username"]},
    )
    return {"username": user["username"], "role": user["role"]}


@app.get("/api/auth/setup-status")
async def auth_setup_status() -> dict:
    """PUBLIC (no auth dependency): report whether the deploy still needs
    first-run setup. Returns ONLY the bool — never any username/count — so it
    leaks nothing about existing accounts. On an initialized deploy this is
    permanently false."""
    try:
        return {"needs_setup": await _needs_setup()}
    except Exception as e:
        # USER_STORE.list() now re-raises real errors (AccessDenied, throttling)
        # instead of masquerading as an empty table. Surface a clean 503 so the
        # frontend guard fails open to the login flow rather than the wizard.
        logger.error(f"setup-status: user store unavailable: {e}")
        raise HTTPException(status_code=503, detail="user store unavailable")


@app.post("/api/auth/setup")
async def auth_setup(payload: dict, response: Response) -> dict:
    """PUBLIC, self-closing first-run bootstrap. Creates the first admin
    account when (and only when) the user table is empty, then auto-logs the
    caller in (same JWT + cookie pattern as /api/auth/login).

    Self-closing: once any user exists this endpoint is permanently 409 and can
    never reset or overwrite an existing account. It is the only public WRITE
    endpoint, so the guard is doubled: the _needs_setup() pre-check plus the
    conditional write inside USER_STORE.create — under a concurrent race only
    one create wins; the loser's DynamoDB ConditionalCheckFailedException is
    mapped to 409 (never 500)."""
    if not await _needs_setup():
        raise HTTPException(status_code=409, detail="already initialized")

    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    # Validate non-empty up front so empty input → 400 cleanly, distinct from
    # the "already exists" ValueError (→ 409) that create can raise.
    if not username or not password:
        raise HTTPException(status_code=400, detail="username and password are required")

    try:
        user = await USER_STORE.create(username, password, role="admin")
    except AuthUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        # create raises ValueError("...must be a non-empty string") for empty
        # input (already guarded above → 400) and ValueError("user ... already
        # exists") for a duplicate, including the normalized concurrent-write
        # collision. Any ValueError here means the table is no longer empty
        # (lost the race) → 409, never 500.
        raise HTTPException(status_code=409, detail="already initialized")
    except ClientError as e:
        # Defense-in-depth: even if create's normalization is ever removed, a
        # raw ConditionalCheckFailedException from the concurrent loser must
        # still surface as 409, never 500.
        if e.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            raise HTTPException(status_code=409, detail="already initialized")
        raise

    token = _issue_jwt(user)
    _set_session_cookie(response, token)
    return {"username": user["username"], "role": user["role"]}


@app.post("/api/auth/logout")
async def auth_logout(
    response: Response, vb_session: str | None = Cookie(default=None)
) -> dict:
    """Clear the session cookie. Always 200 (idempotent)."""
    response.delete_cookie(key=COOKIE_NAME, path="/")
    # Best-effort: decode the (about-to-be-cleared) session to attribute the
    # logout to a username. No auth is required — logout stays idempotent.
    claims = _decode_jwt(vb_session or "")
    actor = (claims or {}).get("sub") or "anon"
    await log_activity(
        actor=actor, type="logout", target=actor if actor != "anon" else None,
        detail={"username": actor},
    )
    return {"ok": True}


@app.get("/api/auth/me")
async def auth_me(user: dict = Depends(require_user)) -> dict:
    return {"username": user["username"], "role": user["role"]}


@app.post("/api/admin/guest-links")
async def admin_create_guest_link(body: dict, admin: dict = Depends(require_admin)) -> dict:
    """Mint a temporary guest experience link (admin only). Validates ttl /
    scenario, returns the token ONCE to the minting admin, and audit-logs the
    parameters (never the token)."""
    ttl = body.get("ttl_minutes", GUEST_TTL_DEFAULT_MIN)
    # int only — reject bool (int subclass) and float; enforce 1..max range.
    if not isinstance(ttl, int) or isinstance(ttl, bool) or not (1 <= ttl <= GUEST_TTL_MAX_MIN):
        raise HTTPException(
            status_code=400, detail=f"ttl_minutes must be an int in 1..{GUEST_TTL_MAX_MIN}"
        )
    scenario = body.get("scenario")
    if scenario is not None:
        if not isinstance(scenario, str):
            raise HTTPException(status_code=400, detail="scenario must be a string")
        valid = {DEFAULT_DEMO_ID} | {d["id"] for d in DEMO_LOADER.list()}
        if scenario not in valid:
            raise HTTPException(status_code=400, detail="unknown scenario")
    # Optional launch-config overrides (all optional). Validated together so the
    # guest experiences the admin's configured variant; nova-sonic+zh-HK → 400.
    lang = body.get("lang")
    engine = body.get("engine")
    voice = body.get("voice")
    provider = body.get("provider")
    clean = _validate_launch_config(lang=lang, engine=engine, voice=voice, provider=provider)
    token, exp = _issue_guest_token(
        ttl, scenario,
        lang=clean.get("lang"), engine=clean.get("engine"),
        voice=clean.get("voice"), provider=clean.get("provider"),
    )
    # NOTE: the token is NEVER logged — only the link parameters.
    await log_activity(
        actor=admin, type="guest-link-create", target="guest",
        detail={
            "ttl_minutes": ttl,
            **({"scenario": scenario} if scenario else {}),
            **clean,  # lang/engine/voice/provider, only those present
        },
    )
    return {
        "token": token,
        "ttl_seconds": ttl * 60,
        "expires_at": exp,
        "scenario": scenario,
        "lang": clean.get("lang"),
        "engine": clean.get("engine"),
        "voice": clean.get("voice"),
        "provider": clean.get("provider"),
        "path": "/guest",
    }


@app.post("/api/auth/guest-login")
async def auth_guest_login(payload: dict, response: Response) -> dict:
    """PUBLIC (no auth dep): redeem a guest link token. A tampered / expired /
    non-guest token → 401. On success set the same HttpOnly vb_session cookie
    used by /api/auth/login (the JWT's own exp is the real authority)."""
    claims = _decode_jwt((payload.get("token") or "").strip())
    if not claims or not claims.get("guest"):
        raise HTTPException(status_code=401, detail="invalid or expired guest link")
    _set_session_cookie(response, payload["token"])
    scenario = claims.get("scenario")
    await log_activity(
        actor="guest", actor_role="guest", type="guest-login",
        status="success", target="guest",
        detail={"scenario": scenario} if scenario else None,
    )
    # Forward the full launch-config subset the token carries (null when absent
    # → guest falls back to global defaults; old scenario-only tokens unchanged).
    return {
        "username": "guest",
        "role": "guest",
        "scenario": scenario,
        "lang": claims.get("lang"),
        "engine": claims.get("engine"),
        "voice": claims.get("voice"),
        "provider": claims.get("provider"),
    }


# --- User management API (admin-only) ------------------------------------

@app.get("/api/admin/users")
async def admin_users_list(_: dict = Depends(require_admin)) -> dict:
    return {"users": await USER_STORE.list()}


@app.post("/api/admin/users")
async def admin_users_create(payload: dict, admin: dict = Depends(require_admin)) -> dict:
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    role = payload.get("role") or "user"
    try:
        user = await USER_STORE.create(username, password, role=role)
    except AuthUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Detail: role only — NEVER the password (store allowlist for user-* is
    # {role,disabled}, so a stray password key would be masked regardless).
    await log_activity(
        actor=admin, type="user-create", target=username,
        detail={"role": role},
    )
    return {"user": user}


@app.patch("/api/admin/users/{username}")
async def admin_users_patch(
    username: str, payload: dict, admin: dict = Depends(require_admin)
) -> dict:
    """Change role / reset password / enable-disable a user. Any subset of
    {role, password, disabled} may be supplied."""
    did_something = False
    try:
        if "role" in payload and payload["role"] is not None:
            if not await USER_STORE.set_role(username, payload["role"]):
                raise HTTPException(status_code=404, detail=f"user not found: {username}")
            did_something = True
        if "password" in payload and payload["password"]:
            if not await USER_STORE.set_password(username, payload["password"]):
                raise HTTPException(status_code=404, detail=f"user not found: {username}")
            did_something = True
        if "disabled" in payload and payload["disabled"] is not None:
            # Guard: an admin cannot disable their own account out from under
            # themselves (avoid lock-out).
            if username == admin["username"] and bool(payload["disabled"]):
                raise HTTPException(status_code=400, detail="cannot disable your own account")
            if not await USER_STORE.set_disabled(username, bool(payload["disabled"])):
                raise HTTPException(status_code=404, detail=f"user not found: {username}")
            did_something = True
    except AuthUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not did_something:
        raise HTTPException(
            status_code=400, detail="body must include at least one of: role, password, disabled"
        )
    user = await USER_STORE.get(username)
    if user is None:
        raise HTTPException(status_code=404, detail=f"user not found: {username}")
    # Detail: the allowlisted changed fields ({role,disabled}) by value, and
    # the FACT a password was reset by NAME only (password is never in the
    # user-* allowlist → masked to "***changed***" by the store backstop).
    changed_detail: dict = {}
    if "role" in payload and payload["role"] is not None:
        changed_detail["role"] = payload["role"]
    if "disabled" in payload and payload["disabled"] is not None:
        changed_detail["disabled"] = bool(payload["disabled"])
    if "password" in payload and payload["password"]:
        changed_detail["password"] = payload["password"]  # masked by store
    await log_activity(
        actor=admin, type="user-update", target=username, detail=changed_detail,
    )
    return {"user": user}


@app.delete("/api/admin/users/{username}")
async def admin_users_delete(username: str, admin: dict = Depends(require_admin)) -> dict:
    if username == admin["username"]:
        raise HTTPException(status_code=400, detail="cannot delete your own account")
    if not await USER_STORE.delete(username):
        raise HTTPException(status_code=404, detail=f"user not found: {username}")
    await log_activity(actor=admin, type="user-delete", target=username)
    return {"deleted": username}


# Admin SPA dist location. As of the single-page merge (tech_design §2) the
# admin SPA serves at the site ROOT "/" — that StaticFiles mount lives at the
# very bottom of this file (it is a catch-all and must come AFTER every /api/*
# + WS route). Here we only keep the legacy /admin path working by redirecting
# it to / (the SPA now lives there). The actual root mount is added later.
ADMIN_DIST_DIR = os.path.join(STATIC_DIR, "admin", "dist")


@app.get("/admin")
@app.get("/admin/")
async def _admin_legacy_redirect() -> Response:
    """Back-compat: the admin SPA moved from /admin to the site root /."""
    return RedirectResponse(url="/", status_code=302)


# --- Admin REST endpoints ------------------------------------------------

def _demo_options() -> list[dict]:
    """Return the demo selector list rendered in admin/SPA dropdowns.

    Layout:

    - First entry is always ``{id: "default", label: "Default (no demo)",
      kind: "default"}`` — the sentinel for "no demo selected".
    - Subsequent entries come from :data:`DEMO_LOADER`.list() in loader
      order, each tagged ``kind: "demo"``. The legacy ``"scenario"`` and
      ``"kb"`` kinds are gone — every non-default entry is a disk demo
      (single source of truth, see proposal §3.3).

    A demo whose own id is ``"default"`` is dropped to keep the sentinel
    unique.
    """
    out: list[dict] = [
        {"id": DEFAULT_DEMO_ID, "label": "Default (no demo)", "kind": "default"},
    ]
    for d in DEMO_LOADER.list():
        if d["id"] == DEFAULT_DEMO_ID:
            continue
        out.append({"id": d["id"], "label": d["label"], "kind": "demo"})
    return out


def _scenario_options() -> list[dict]:
    """Deprecated: kept as a thin wrapper around :func:`_demo_options` for
    callers internal to bot.py that still pass through this name. New code
    should import :func:`_demo_options` directly.
    """
    return _demo_options()


def _admin_options_payload() -> dict:
    demo_opts = _demo_options()
    return {
        "languages": [
            {"id": k, "label": v["label"], "engines": list(v.get("engines", []))}
            for k, v in LANGUAGES.items()
        ],
        "engines": [{"id": k, "label": v} for k, v in ENGINES.items()],
        "providers": [{"id": k, "label": v} for k, v in TTS_PROVIDERS.items()],
        "models": [{"id": k, "label": k, "bedrock_id": v} for k, v in MODELS.items()],
        "minimax_models": [{"id": k, "label": v} for k, v in MINIMAX_MODELS.items()],
        "demos": demo_opts,
        "scenarios": demo_opts,
        "voices_by_provider": {
            "minimax": [
                {"id": k, "label": _voice_display(v), "language": v.get("language")}
                for k, v in voices_for("minimax").items()
            ],
            "polly": [
                {"id": k, "label": _voice_display(v), "language": v.get("language")}
                for k, v in voices_for("polly").items()
            ],
        },
        "nova_sonic_voices": [
            {
                "id": k,
                "label": v["label"],
                "gender": v.get("gender"),
                "locale": v.get("locale"),
                "lang_label": v.get("lang_label"),
                "polyglot": bool(v.get("polyglot")),
            }
            for k, v in voices_for("nova-sonic").items()
        ],
        # Parity with /api/config: expose the true default Nova voice so the
        # admin DefaultsForm normalizes voice-on-engine-switch to the real
        # DEFAULT_NOVA_SONIC_VOICE rather than falling back to the first id.
        "default_nova_sonic_voice": DEFAULT_NOVA_SONIC_VOICE,
        "mcp_servers": [
            {"id": s["id"], "label": s["label"], "enabled": s["enabled"]}
            for s in MCP_CONFIG.list_servers()
        ],
    }


# Pydantic models for partial PUT bodies (any field optional)
from pydantic import BaseModel, Field


class WebUpdate(BaseModel):
    lang: str | None = None
    engine: str | None = None
    scenario: str | None = None
    demo: str | None = None
    model: str | None = None
    provider: str | None = None
    voice: str | None = None
    minimax_model: str | None = None
    asr_filter: dict | None = None


class PhoneUpdate(BaseModel):
    lang: str | None = None
    engine: str | None = None
    scenario: str | None = None
    demo: str | None = None
    voice: str | None = None
    provider: str | None = None
    model: str | None = None
    minimax_model: str | None = None
    asr_filter: dict | None = None


def _normalize_segment(updates: dict) -> dict:
    """Collapse the new ``demo`` alias onto the legacy ``scenario`` key.

    Either, neither, or both may be present. When both are present, ``demo``
    wins (it's what the new SPA sends). The returned dict no longer
    contains a ``demo`` key — :func:`_validate_segment` expects only
    ``scenario``.
    """
    if "demo" in updates:
        demo_val = updates.pop("demo")
        if demo_val is not None:
            updates["scenario"] = demo_val
    return updates


def _validate_segment(updates: dict, current: dict | None = None) -> None:
    """Reject obviously invalid values; let unknown fields pass (Pydantic
    already strips them) so we can extend the schema without lockstep deploy.

    ``updates`` is the (already normalized) partial PUT body. ``current`` is the
    currently-stored segment (e.g. ``RUNTIME_CONFIG.get_web_defaults()``); the
    engine↔lang cross-check is run against the MERGED effective state
    (``current`` overlaid with ``updates``). This matters for partial updates:
    changing only ``lang`` while the stored ``engine`` is incompatible must be
    rejected — looking at ``updates`` alone would miss it. ``current`` defaults
    to ``{}`` so old callers (full-segment validation) still work.
    """
    if "engine" in updates and updates["engine"] not in ENGINES:
        raise HTTPException(status_code=400, detail=f"invalid engine: {updates['engine']}")
    if "lang" in updates and updates["lang"] not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"invalid lang: {updates['lang']}")
    if "scenario" in updates:
        ids = {x["id"] for x in _scenario_options()}
        if updates["scenario"] not in ids:
            raise HTTPException(status_code=400, detail=f"invalid scenario: {updates['scenario']}")
    if "provider" in updates and updates["provider"] not in TTS_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"invalid provider: {updates['provider']}")
    if "model" in updates and updates["model"] not in MODELS:
        raise HTTPException(status_code=400, detail=f"invalid model: {updates['model']}")

    # Engine↔lang compatibility, evaluated on the merged effective state so
    # partial updates (only lang, or only engine) are judged correctly.
    merged = dict(current or {})
    merged.update(updates)
    eff_engine = merged.get("engine")
    eff_lang = merged.get("lang")
    if eff_engine in ENGINES and eff_lang in LANGUAGES:
        lang_entry = LANGUAGES[eff_lang]
        if eff_engine not in lang_entry.get("engines", []):
            raise HTTPException(
                status_code=400,
                detail=f"invalid lang for engine: {eff_lang} not available on {eff_engine}",
            )
        if eff_engine == "pipeline" and lang_entry.get("stt") is None:
            raise HTTPException(
                status_code=400,
                detail=f"language not available for pipeline (no STT): {eff_lang}",
            )


@app.get("/api/admin/health")
async def admin_health(_: dict = Depends(require_admin)) -> dict:
    return {"ok": True, "service": "voice-bot-admin"}


def _mirror_demo_alias(segment: dict) -> dict:
    """Stored under legacy ``scenario`` key; expose as ``demo`` too so the
    SPA (which binds ``form.demo``) reads back what it saved."""
    if "scenario" in segment and "demo" not in segment:
        segment = dict(segment)
        segment["demo"] = segment["scenario"]
    return segment


@app.get("/api/admin/config")
async def admin_get_config(_: dict = Depends(require_admin)) -> dict:
    return {
        "web": _mirror_demo_alias(RUNTIME_CONFIG.get_web_defaults()),
        "phone": _mirror_demo_alias(RUNTIME_CONFIG.get_phone_defaults()),
    }


@app.put("/api/admin/config/web")
async def admin_put_web(body: WebUpdate, admin: dict = Depends(require_admin)) -> dict:
    updates = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None}
    updates = _normalize_segment(updates)
    _validate_segment(updates, RUNTIME_CONFIG.get_web_defaults())
    # asr_filter sub-block: runtime_config._update_segment is SHALLOW, so a raw
    # {enabled:...} would REPLACE the whole asr_filter dict and clobber sibling
    # thresholds. Validate + read-merge-write the sub-block ourselves so a
    # single-field edit preserves previously-set siblings (tech_design §6).
    if "asr_filter" in updates:
        clean = _validate_asr_filter_patch(updates["asr_filter"])
        existing = dict(RUNTIME_CONFIG.get_web_defaults().get("asr_filter") or {})
        existing.update(clean)
        updates["asr_filter"] = existing
    result = _mirror_demo_alias(RUNTIME_CONFIG.update_web(updates))
    # Detail: the changed allowlisted keys by value (config-* allowlist =
    # {engine,lang,scenario,provider,voice,minimax_model,model}); anything else
    # is masked to name-only by the store backstop.
    await log_activity(actor=admin, type="config-web", detail=dict(updates))
    return {"web": result}


@app.put("/api/admin/config/phone")
async def admin_put_phone(body: PhoneUpdate, admin: dict = Depends(require_admin)) -> dict:
    updates = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None}
    updates = _normalize_segment(updates)
    _validate_segment(updates, RUNTIME_CONFIG.get_phone_defaults())
    # asr_filter sub-block read-merge-write (see admin_put_web; §6).
    if "asr_filter" in updates:
        clean = _validate_asr_filter_patch(updates["asr_filter"])
        existing = dict(RUNTIME_CONFIG.get_phone_defaults().get("asr_filter") or {})
        existing.update(clean)
        updates["asr_filter"] = existing
    result = _mirror_demo_alias(RUNTIME_CONFIG.update_phone(updates))
    await log_activity(actor=admin, type="config-phone", detail=dict(updates))
    return {"phone": result}


@app.get("/api/admin/options")
async def admin_options(_: dict = Depends(require_admin)) -> dict:
    return _admin_options_payload()


# Chime Voice Connector region is fixed: VCs only exist in us-east-1 (see
# CLAUDE.md). This is independent of BedrockRegion / the deploy region.
CHIME_REGION = "us-east-1"


def _list_chime_vc_numbers() -> dict:
    """Return Chime Voice Connectors joined with their associated phone numbers.

    READ-ONLY: calls only ``ListVoiceConnectors`` + ``ListPhoneNumbers`` (both
    paginated). There is no per-VC "list associated numbers" API, so we read all
    account numbers and join each number's
    ``Associations[Name=='VoiceConnectorId'].Value`` back to the VC.

    Shape::

        {"voice_connectors": [
            {"voice_connector_id", "voice_connector_name", "e164", "status"}
         ], "error": None}

    Rows: one per (VC, associated number); a VC with no number → one row with
    ``e164=None``; a number associated to no VC → one row with both VC fields
    ``None``. Best-effort: any failure (AccessDenied / timeout / unreachable)
    returns ``{"voice_connectors": [], "error": "<reason>"}`` with NO exception,
    so the admin page never 500s on a Chime hiccup or missing IAM permission.
    """
    from botocore.config import Config

    try:
        client = boto3.client(
            "chime-sdk-voice",
            region_name=CHIME_REGION,
            config=Config(
                connect_timeout=3,
                read_timeout=5,
                retries={"max_attempts": 2},
            ),
        )

        # 1. VCs (paginated) → {id: name}
        vc_names: dict[str, str] = {}
        token = None
        while True:
            kwargs = {"MaxResults": 99}
            if token:
                kwargs["NextToken"] = token
            resp = client.list_voice_connectors(**kwargs)
            for vc in resp.get("VoiceConnectors", []):
                vc_names[vc.get("VoiceConnectorId")] = vc.get("Name")
            token = resp.get("NextToken")
            if not token:
                break

        # 2. Numbers (paginated) → rows joined to their VC.
        rows: list[dict] = []
        seen_vc_ids: set[str] = set()
        token = None
        while True:
            kwargs = {"MaxResults": 99}
            if token:
                kwargs["NextToken"] = token
            resp = client.list_phone_numbers(**kwargs)
            for num in resp.get("PhoneNumbers", []):
                e164 = num.get("E164PhoneNumber")
                status = num.get("Status")
                vc_id = None
                for assoc in num.get("Associations", []):
                    if assoc.get("Name") == "VoiceConnectorId":
                        vc_id = assoc.get("Value")
                        break
                if vc_id is not None:
                    seen_vc_ids.add(vc_id)
                rows.append(
                    {
                        "voice_connector_id": vc_id,
                        # name resolved from the VC list; unknown/absent → None
                        "voice_connector_name": vc_names.get(vc_id) if vc_id else None,
                        "e164": e164,
                        "status": status,
                    }
                )
            token = resp.get("NextToken")
            if not token:
                break

        # 3. Also surface VCs that have NO associated number, so the admin can
        #    see a VC exists but isn't wired to a number yet.
        for vc_id, name in vc_names.items():
            if vc_id not in seen_vc_ids:
                rows.append(
                    {
                        "voice_connector_id": vc_id,
                        "voice_connector_name": name,
                        "e164": None,
                        "status": None,
                    }
                )

        return {"voice_connectors": rows, "error": None}
    except Exception as e:  # noqa: BLE001 — best-effort: never 500 the admin page
        logger.warning(f"_list_chime_vc_numbers failed (degraded to empty): {e}")
        return {"voice_connectors": [], "error": str(e)}


@app.get("/api/admin/phone-numbers")
async def admin_phone_numbers(_: dict = Depends(require_admin)) -> dict:
    """Admin-only, read-only view of Chime Voice Connector ↔ phone-number
    associations (real-time from chime-sdk-voice). See _list_chime_vc_numbers."""
    return _list_chime_vc_numbers()


@app.get("/api/admin/metrics")
async def admin_metrics(_: dict = Depends(require_admin)):
    """Real-time admin dashboard metrics.

    Thin delegate over :func:`_collect_metrics` (T1). Admin-only via
    ``Depends(require_admin)`` (the legacy ``admin_path_guard`` middleware was
    removed when the auth model moved to JWT sessions).

    On any exception inside the aggregator, return ``{"error": "..."}`` with
    HTTP 500 so the front-end can render a toast without parsing FastAPI's
    default ``{"detail": ...}`` envelope.
    """
    from fastapi.responses import JSONResponse as _JR
    try:
        return await _collect_metrics()
    except Exception as exc:  # pragma: no cover - safety net; _collect_metrics swallows scan errors
        logger.exception("/api/admin/metrics failed")
        return _JR(status_code=500, content={"error": str(exc) or exc.__class__.__name__})


# --- Admin call-history endpoints (T3) ----------------------------------
#
# These mirror the user-scoped /api/history* routes but return the FULL view
# across all calls. Admin-only via Depends(require_admin) on each route (the
# legacy admin_path_guard middleware was removed when auth moved to JWT
# sessions).
#
# Persisted columns reference (set by HistoryRecorder, see bot.py around
# line 880+): call_id, caller, started_at(int), ended_at(int), duration_s,
# outcome, summary_status('pending'|'ok'|'failed'), ttl, transfer_requested,
# transfer_topic, turns[], turn_count, lang, engine, scenario, provider,
# voice, model, minimax_model, summary, summary_error.

# Admin list-card projection — keep it lean (no turns / no summary blob).
# The detail endpoint (GET /api/admin/history/{call_id}) is where clients
# fetch the heavy fields.
_ADMIN_LIST_FIELDS = (
    "call_id",
    "caller",
    "started_at",
    "ended_at",
    "duration_s",
    "outcome",
    "engine",
    "scenario",
    "lang",
    "summary_status",
    "transfer_requested",
    "turn_count",
)

# Outcome buckets the admin list filter accepts. Anything else is treated as
# unknown and rejected (we don't want to silently scan with a bogus value).
_ADMIN_OUTCOME_VALUES = {
    "user_requested",
    "task_completed",
    "transferred",
    "timeout",
    "error",
    "unknown",
}

# Hard cap for the streaming CSV export. Above this we set X-Truncated: 1
# so the front-end can prompt the operator to refine filters.
_ADMIN_CSV_MAX_ROWS = 5000


def _admin_history_filter_clause(
    *,
    outcome: str | None,
    engine: str | None,
    scenario: str | None,
    start_after: int | None,
    start_before: int | None,
    skip_started_at: bool = False,
) -> tuple[str, dict, dict]:
    """Build a DDB FilterExpression covering the admin-list query knobs.

    Returns ``(filter_expr, expr_attr_names, expr_attr_values)``. Empty filter
    is signalled by an empty string for ``filter_expr``. Most column names are
    DDB-reserved or look-reserved so we always go through ExpressionAttributeNames
    aliases.

    ``skip_started_at=True`` is used by the GSI Query path where the time-range
    is folded into the KeyConditionExpression instead of the FilterExpression.
    """
    clauses: list[str] = []
    names: dict[str, str] = {}
    values: dict[str, object] = {}

    if outcome:
        names["#oc"] = "outcome"
        values[":oc"] = outcome
        clauses.append("#oc = :oc")
    if engine:
        names["#en"] = "engine"
        values[":en"] = engine
        clauses.append("#en = :en")
    if scenario:
        names["#sc"] = "scenario"
        values[":sc"] = scenario
        clauses.append("#sc = :sc")
    if not skip_started_at:
        if start_after is not None:
            names["#sa"] = "started_at"
            values[":sa_after"] = int(start_after)
            clauses.append("#sa >= :sa_after")
        if start_before is not None:
            names["#sa"] = "started_at"
            values[":sa_before"] = int(start_before)
            clauses.append("#sa <= :sa_before")

    return (" AND ".join(clauses), names, values)


def _project_admin_card(item: dict) -> dict:
    """Strip a full DDB row to the columns surfaced by the admin list view."""
    out = {k: item.get(k) for k in _ADMIN_LIST_FIELDS}
    # transfer_requested is stored as a bool but DDB hands it back as Decimal in
    # some old rows; normalize.
    tr = out.get("transfer_requested")
    if isinstance(tr, (int, float)):
        out["transfer_requested"] = bool(tr)
    return out


@app.get("/api/admin/history")
async def admin_list_history(
    limit: int = 50,
    cursor: str | None = None,
    caller: str | None = None,
    outcome: str | None = None,
    engine: str | None = None,
    demo: str | None = None,
    scenario: str | None = None,
    start_after: int | None = None,
    start_before: int | None = None,
    _: dict = Depends(require_admin),
) -> dict:
    """Admin call-history list with cursor pagination + filters.

    - When ``caller`` is provided, queries the ``caller-started-index`` GSI.
    - Otherwise scans the table and applies a FilterExpression covering the
      remaining knobs.
    - Heavy columns (``turns``, ``summary``) are stripped via ProjectionExpression
      so the list payload is bounded even for chatty calls.
    """
    if limit <= 0 or limit > 200:
        limit = 50
    if outcome and outcome not in _ADMIN_OUTCOME_VALUES:
        raise HTTPException(status_code=400, detail=f"unknown outcome: {outcome!r}")
    # demo and scenario are aliases — accept either, but the on-disk column
    # name is `scenario`.
    scenario_filter = scenario or demo

    if HISTORY_DISABLED or _history is None:
        return {"items": [], "next_cursor": None}

    start_key = _decode_cursor(cursor)
    table = _history_table()

    # Projection used by both the Scan and the Query path. We alias every
    # column to dodge DDB reserved-word collisions (started_at, outcome, ...).
    proj_aliases = {
        "#cid": "call_id",
        "#caller": "caller",
        "#sa": "started_at",
        "#ea": "ended_at",
        "#dur": "duration_s",
        "#oc": "outcome",
        "#en": "engine",
        "#sc": "scenario",
        "#lang": "lang",
        "#ss": "summary_status",
        "#tr": "transfer_requested",
        "#tc": "turn_count",
    }
    projection = (
        "#cid, #caller, #sa, #ea, #dur, #oc, #en, #sc, #lang, #ss, #tr, #tc"
    )

    try:
        if caller:
            # Caller path → Query the GSI. We can fold start_after/start_before
            # into the KeyConditionExpression via the sort key (started_at).
            key_cond = "caller = :c"
            expr_values: dict[str, object] = {":c": caller}
            names = dict(proj_aliases)
            if start_after is not None and start_before is not None:
                key_cond += " AND #sa BETWEEN :sa_after AND :sa_before"
                expr_values[":sa_after"] = int(start_after)
                expr_values[":sa_before"] = int(start_before)
            elif start_after is not None:
                key_cond += " AND #sa >= :sa_after"
                expr_values[":sa_after"] = int(start_after)
            elif start_before is not None:
                key_cond += " AND #sa <= :sa_before"
                expr_values[":sa_before"] = int(start_before)
            # outcome / engine / scenario fold into a separate FilterExpression.
            filter_expr, fnames, fvalues = _admin_history_filter_clause(
                outcome=outcome,
                engine=engine,
                scenario=scenario_filter,
                start_after=None,
                start_before=None,
                skip_started_at=True,
            )
            names.update(fnames)
            expr_values.update(fvalues)
            query_kwargs = {
                "IndexName": "caller-started-index",
                "KeyConditionExpression": key_cond,
                "ExpressionAttributeNames": names,
                "ExpressionAttributeValues": expr_values,
                "ScanIndexForward": False,
                "Limit": limit,
                "ProjectionExpression": projection,
            }
            if filter_expr:
                query_kwargs["FilterExpression"] = filter_expr
            if start_key:
                query_kwargs["ExclusiveStartKey"] = start_key
            resp = await asyncio.to_thread(table.query, **query_kwargs)
        else:
            # Non-caller path → Scan with a FilterExpression that covers
            # outcome/engine/scenario/start_after/start_before.
            filter_expr, fnames, fvalues = _admin_history_filter_clause(
                outcome=outcome,
                engine=engine,
                scenario=scenario_filter,
                start_after=start_after,
                start_before=start_before,
            )
            names = dict(proj_aliases)
            names.update(fnames)
            scan_kwargs: dict = {
                "ExpressionAttributeNames": names,
                "ProjectionExpression": projection,
                "Limit": limit,
            }
            if filter_expr:
                scan_kwargs["FilterExpression"] = filter_expr
            if fvalues:
                scan_kwargs["ExpressionAttributeValues"] = fvalues
            if start_key:
                scan_kwargs["ExclusiveStartKey"] = start_key
            resp = await asyncio.to_thread(table.scan, **scan_kwargs)
    except HTTPException:
        raise
    except Exception:
        logger.exception("admin history list failed")
        raise HTTPException(status_code=500, detail="history backend unavailable")

    items_raw = resp.get("Items") or []
    items = [_decimal_to_native(_project_admin_card(it)) for it in items_raw]
    # GSI Query already returns ScanIndexForward=False (newest first). Scan
    # is unordered, so sort the page in-memory by started_at desc — the only
    # column users sort by from the UI.
    if not caller:
        items.sort(key=lambda r: r.get("started_at") or 0, reverse=True)
    last_key = resp.get("LastEvaluatedKey")
    next_cursor = _encode_cursor(last_key) if last_key else None
    return {"items": items, "next_cursor": next_cursor}


@app.get("/api/admin/history/{call_id}")
async def admin_get_history_item(call_id: str, _: dict = Depends(require_admin)) -> dict:
    """Full admin row including ``turns[]`` and ``summary``."""
    if HISTORY_DISABLED or _history is None:
        raise HTTPException(status_code=404, detail="not found")
    try:
        table = _history_table()
        resp = await asyncio.to_thread(table.get_item, Key={"call_id": call_id})
    except HTTPException:
        raise
    except Exception:
        logger.exception("admin history get_item failed")
        raise HTTPException(status_code=500, detail="history backend unavailable")
    item = resp.get("Item")
    if not item:
        raise HTTPException(status_code=404, detail="not found")
    return _decimal_to_native(item)


def _format_summary_for_md(summary, status: str | None) -> str:
    """Render the persisted ``summary`` field for the export.md view.

    HistoryRecorder writes ``summary`` as a structured dict (intent /
    key_questions / action_items / sentiment / model / generated_at). When
    that's the case we prefer the human-readable shape (intent + bulleted
    items + sentiment); we fall back to a JSON pretty-print for anything
    unexpected. If ``summary_status`` isn't 'ok' (or summary is missing),
    we say so explicitly so operators don't mistake silence for ``intent=""``.
    """
    if status and status != "ok":
        return f"[summary not available: {status}]"
    if not summary:
        return "[summary not available: missing]"
    if isinstance(summary, dict):
        # Prefer a readable composition over a raw JSON dump.
        lines: list[str] = []
        intent = summary.get("intent")
        if intent:
            lines.append(f"**Intent:** {intent}")
        for q in summary.get("key_questions") or []:
            lines.append(f"- Q: {q}")
        for a in summary.get("action_items") or []:
            lines.append(f"- Action: {a}")
        sentiment = summary.get("sentiment")
        if sentiment:
            lines.append(f"_Sentiment:_ {sentiment}")
        text = summary.get("text")
        if text:
            lines.append(text)
        if lines:
            return "\n\n".join(lines)
        # Empty-but-present dict → fall back to JSON so something is visible.
        return "```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```"
    if isinstance(summary, str):
        return summary
    # Anything exotic — pretty-print so the operator sees structure.
    try:
        return "```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```"
    except Exception:
        return str(summary)


def _format_iso_utc(epoch_seconds) -> str:
    """epoch-seconds → 'YYYY-MM-DDTHH:MM:SSZ'. Tolerates None / Decimal."""
    if not epoch_seconds:
        return ""
    try:
        ts = int(epoch_seconds)
    except (TypeError, ValueError):
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@app.get("/api/admin/history/{call_id}/export.md")
async def admin_export_history_md(call_id: str, _: dict = Depends(require_admin)) -> Response:
    """Server-rendered Markdown for a single call. Triggers a file download."""
    if HISTORY_DISABLED or _history is None:
        raise HTTPException(status_code=404, detail="not found")
    try:
        table = _history_table()
        resp = await asyncio.to_thread(table.get_item, Key={"call_id": call_id})
    except HTTPException:
        raise
    except Exception:
        logger.exception("admin history export.md get_item failed")
        raise HTTPException(status_code=500, detail="history backend unavailable")
    raw = resp.get("Item")
    if not raw:
        raise HTTPException(status_code=404, detail="not found")
    item = _decimal_to_native(raw)

    started_iso = _format_iso_utc(item.get("started_at"))
    duration_s = item.get("duration_s") or 0
    transferred = bool(item.get("transfer_requested"))
    topic = item.get("transfer_topic") or ""
    summary_md = _format_summary_for_md(item.get("summary"), item.get("summary_status"))

    lines: list[str] = []
    lines.append(f"# Call {call_id}")
    lines.append(f"- Caller: {item.get('caller') or 'unknown'}")
    lines.append(f"- Time: {started_iso}  ({duration_s} s)")
    lines.append(
        f"- Engine: {item.get('engine') or ''} · Demo: {item.get('scenario') or ''} "
        f"· Lang: {item.get('lang') or ''}"
    )
    lines.append(
        f"- Outcome: {item.get('outcome') or ''}  "
        f"(transferred={str(transferred).lower()}, topic={topic})"
    )
    lines.append("")
    lines.append("## Summary")
    lines.append(summary_md)
    lines.append("")
    lines.append("## Transcript")
    for turn in item.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        who = turn.get("who") or "bot"
        lines.append(f"- **{who}**: {text}")

    body = "\n".join(lines) + "\n"
    return Response(
        content=body,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{call_id}.md"',
        },
    )


def _scan_history_for_csv(
    *,
    caller: str | None,
    outcome: str | None,
    engine: str | None,
    scenario_filter: str | None,
    start_after: int | None,
    start_before: int | None,
):
    """Synchronous generator yielding raw DDB items page by page.

    The CSV endpoint runs this in a thread via ``asyncio.to_thread`` so the
    event loop isn't blocked by DDB scan latency. It pages through
    ``LastEvaluatedKey`` until the table is drained or the caller stops
    iterating (the truncation guard relies on cooperative early-exit).
    """
    table = _history_table()
    proj_aliases = {
        "#cid": "call_id",
        "#caller": "caller",
        "#sa": "started_at",
        "#ea": "ended_at",
        "#dur": "duration_s",
        "#oc": "outcome",
        "#en": "engine",
        "#sc": "scenario",
        "#lang": "lang",
        "#ss": "summary_status",
        "#tr": "transfer_requested",
        "#tt": "transfer_topic",
        "#tc": "turn_count",
    }
    projection = (
        "#cid, #caller, #sa, #ea, #dur, #oc, #en, #sc, #lang, "
        "#ss, #tr, #tt, #tc"
    )
    last_key: dict | None = None
    while True:
        if caller:
            key_cond = "caller = :c"
            expr_values: dict[str, object] = {":c": caller}
            names = dict(proj_aliases)
            if start_after is not None and start_before is not None:
                key_cond += " AND #sa BETWEEN :sa_after AND :sa_before"
                expr_values[":sa_after"] = int(start_after)
                expr_values[":sa_before"] = int(start_before)
            elif start_after is not None:
                key_cond += " AND #sa >= :sa_after"
                expr_values[":sa_after"] = int(start_after)
            elif start_before is not None:
                key_cond += " AND #sa <= :sa_before"
                expr_values[":sa_before"] = int(start_before)
            filter_expr, fnames, fvalues = _admin_history_filter_clause(
                outcome=outcome,
                engine=engine,
                scenario=scenario_filter,
                start_after=None,
                start_before=None,
                skip_started_at=True,
            )
            names.update(fnames)
            expr_values.update(fvalues)
            kwargs = {
                "IndexName": "caller-started-index",
                "KeyConditionExpression": key_cond,
                "ExpressionAttributeNames": names,
                "ExpressionAttributeValues": expr_values,
                "ScanIndexForward": False,
                "ProjectionExpression": projection,
            }
            if filter_expr:
                kwargs["FilterExpression"] = filter_expr
            if last_key:
                kwargs["ExclusiveStartKey"] = last_key
            resp = table.query(**kwargs)
        else:
            filter_expr, fnames, fvalues = _admin_history_filter_clause(
                outcome=outcome,
                engine=engine,
                scenario=scenario_filter,
                start_after=start_after,
                start_before=start_before,
            )
            names = dict(proj_aliases)
            names.update(fnames)
            kwargs = {
                "ExpressionAttributeNames": names,
                "ProjectionExpression": projection,
            }
            if filter_expr:
                kwargs["FilterExpression"] = filter_expr
            if fvalues:
                kwargs["ExpressionAttributeValues"] = fvalues
            if last_key:
                kwargs["ExclusiveStartKey"] = last_key
            resp = table.scan(**kwargs)
        for it in resp.get("Items") or []:
            yield it
        last_key = resp.get("LastEvaluatedKey")
        if not last_key:
            break


_ADMIN_CSV_COLUMNS = (
    "call_id",
    "caller",
    "started_utc",
    "duration_s",
    "outcome",
    "engine",
    "scenario",
    "lang",
    "summary_status",
    "transfer_requested",
    "transfer_topic",
    "turn_count",
)


def _row_to_csv_dict(item: dict) -> dict:
    """Coerce a (possibly Decimal-laced) DDB row into the flat CSV column set."""
    native = _decimal_to_native(item)
    return {
        "call_id": native.get("call_id") or "",
        "caller": native.get("caller") or "",
        "started_utc": _format_iso_utc(native.get("started_at")),
        "duration_s": native.get("duration_s") or 0,
        "outcome": native.get("outcome") or "",
        "engine": native.get("engine") or "",
        "scenario": native.get("scenario") or "",
        "lang": native.get("lang") or "",
        "summary_status": native.get("summary_status") or "",
        "transfer_requested": "true" if native.get("transfer_requested") else "false",
        "transfer_topic": native.get("transfer_topic") or "",
        "turn_count": native.get("turn_count") or 0,
    }


@app.get("/api/admin/history.csv")
async def admin_export_history_csv(
    caller: str | None = None,
    outcome: str | None = None,
    engine: str | None = None,
    demo: str | None = None,
    scenario: str | None = None,
    start_after: int | None = None,
    start_before: int | None = None,
    _: dict = Depends(require_admin),
):
    """Stream the filtered call history as CSV.

    Memory bound: yields one CSV row per item. Hard cap at
    ``_ADMIN_CSV_MAX_ROWS``; if reached we add ``X-Truncated: 1`` so the
    front-end can prompt the operator to refine filters.

    Implementation note on the truncation header: ``StreamingResponse``
    sends headers before the body, so we have to know the truncation flag
    *before* streaming. We pre-fetch up to ``max_rows + 1`` rows into memory,
    decide the flag, then stream the (capped) buffer back out. This trades
    a small in-memory buffer (≤5001 lean rows ≈ 0.5 MB) for a correct header.
    """
    from fastapi.responses import StreamingResponse

    if outcome and outcome not in _ADMIN_OUTCOME_VALUES:
        raise HTTPException(status_code=400, detail=f"unknown outcome: {outcome!r}")
    scenario_filter = scenario or demo

    headers: dict[str, str] = {
        "Content-Disposition": 'attachment; filename="history.csv"',
    }
    if HISTORY_DISABLED or _history is None:
        # Empty CSV with header row still — easier on the front-end which
        # always opens the file in a viewer.
        empty = io.StringIO()
        writer = csv.DictWriter(empty, fieldnames=_ADMIN_CSV_COLUMNS)
        writer.writeheader()
        return StreamingResponse(
            iter([empty.getvalue()]),
            media_type="text/csv; charset=utf-8",
            headers=headers,
        )

    max_rows = _ADMIN_CSV_MAX_ROWS

    def collect():
        out: list[dict] = []
        try:
            it = _scan_history_for_csv(
                caller=caller,
                outcome=outcome,
                engine=engine,
                scenario_filter=scenario_filter,
                start_after=start_after,
                start_before=start_before,
            )
            for raw in it:
                if len(out) >= max_rows + 1:
                    break
                out.append(_row_to_csv_dict(raw))
        except Exception:
            logger.exception("admin history.csv scan failed")
            raise
        return out

    try:
        rows = await asyncio.to_thread(collect)
    except Exception:
        raise HTTPException(status_code=500, detail="history backend unavailable")

    truncated = len(rows) > max_rows
    if truncated:
        rows = rows[:max_rows]
        headers["X-Truncated"] = "1"

    def stream():
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=_ADMIN_CSV_COLUMNS)
        writer.writeheader()
        yield buf.getvalue()
        for row in rows:
            buf.seek(0)
            buf.truncate(0)
            writer.writerow(row)
            yield buf.getvalue()

    return StreamingResponse(
        stream(),
        media_type="text/csv; charset=utf-8",
        headers=headers,
    )


@app.post("/api/admin/history/{call_id}/summarize")
async def admin_resummarize_history(call_id: str, _: dict = Depends(require_admin)):
    """Re-run the Bedrock summarizer over an existing call's transcript.

    Re-uses the module-level :func:`_invoke_summary_bedrock` so we don't
    drift from how end-of-call summarization is done. On success, persists
    ``summary`` + ``summary_status='ok'`` and clears ``summary_error``. On
    failure, persists ``summary_status='failed'`` + ``summary_error`` and
    returns a 500 with ``{"error": "..."}`` so the SPA can render a toast.
    """
    from fastapi.responses import JSONResponse as _JR

    if HISTORY_DISABLED or _history is None:
        raise HTTPException(status_code=404, detail="not found")
    try:
        table = _history_table()
        resp = await asyncio.to_thread(table.get_item, Key={"call_id": call_id})
    except HTTPException:
        raise
    except Exception:
        logger.exception("admin summarize get_item failed")
        raise HTTPException(status_code=500, detail="history backend unavailable")
    raw = resp.get("Item")
    if not raw:
        raise HTTPException(status_code=404, detail="not found")
    item = _decimal_to_native(raw)
    turns = item.get("turns") or []
    lang = item.get("lang") or DEFAULT_LANG

    try:
        struct = await _invoke_summary_bedrock(turns, lang)
    except Exception as e:
        logger.exception("admin summarize Bedrock invocation failed")
        # Persist the failure shape so the row reflects reality.
        try:
            await asyncio.to_thread(
                table.update_item,
                Key={"call_id": call_id},
                UpdateExpression="SET summary_status = :f, summary_error = :e",
                ExpressionAttributeValues={
                    ":f": "failed",
                    ":e": str(e)[:500],
                },
            )
        except Exception:
            logger.exception("admin summarize failure persist failed")
        return _JR(status_code=500, content={"error": f"{type(e).__name__}: {e}"})

    # Success — persist + clear summary_error so we don't leave stale failure
    # bread-crumbs from a previous attempt.
    try:
        await asyncio.to_thread(
            table.update_item,
            Key={"call_id": call_id},
            UpdateExpression=(
                "SET summary = :s, summary_status = :ok REMOVE summary_error"
            ),
            ExpressionAttributeValues={
                ":s": _to_ddb(struct),
                ":ok": "ok",
            },
        )
    except Exception as e:
        logger.exception("admin summarize update_item failed")
        return _JR(
            status_code=500,
            content={"error": f"persist failed: {type(e).__name__}: {e}"},
        )

    return {
        "call_id": call_id,
        "summary": struct,
        "summary_status": "ok",
    }


# --- Tool registry helpers (T6) -----------------------------------------
#
# REGISTRY entries hold a ``frozenset`` for ``scope``; FastAPI / Pydantic
# does NOT serialize frozenset to JSON. Always convert via
# ``sorted(list(scope))`` so the wire format is a deterministic JSON list
# (e.g. ``["phone", "web"]``). The dedicated helper below is the single
# source of truth for the wire shape so /api/admin/tools and the
# augmentation embedded in /api/admin/demos[/{id}] stay in lockstep.
def _tool_def_to_dict(tool_id: str, defn) -> dict:
    """Render a :class:`ToolDefinition` to the JSON shape the admin SPA reads.

    Keys (per task spec):
      - ``id``: stable tool id
      - ``label``: human-friendly admin label
      - ``description``: short summary (description_short, falls back to
        the FunctionSchema description)
      - ``scope``: ``sorted(list(scope))`` — explicit JSON list, never
        ``frozenset``, never ``null``
      - ``default_enabled``: hardcoded ``False`` (no per-tool default
        state implemented yet — enabling is per-demo via manifest.tools)
      - ``supported_langs``: ``sorted(list(policy_blurb.keys()))``
      - ``hangup_blurb_keys``: same list, kept as a separate key for
        documentation / future divergence (today the two are identical
        because every blurb in the registry is a hangup/transfer policy)
    """
    schema_desc = getattr(defn.schema, "description", "") or ""
    description = (defn.description_short or "").strip() or schema_desc
    langs = sorted(list((defn.policy_blurb or {}).keys()))
    return {
        "id": tool_id,
        "label": defn.label,
        "description": description,
        "scope": sorted(list(defn.scope)),
        "default_enabled": False,
        "supported_langs": langs,
        "hangup_blurb_keys": langs,
    }


def _augment_demo_with_tools(demo: dict) -> dict:
    """Return a shallow-copied demo dict with admin-only ``tools`` +
    ``tool_defs`` keys derived from ``tool_ids`` + tools.registry.REGISTRY.

    Unknown ids are dropped silently here — demo_loader already logs a
    warning when it parses the manifest, so re-warning at REST time
    would just be noise.
    """
    from tools.registry import REGISTRY as _TOOLS

    tool_ids = list(demo.get("tool_ids") or [])
    out = dict(demo)
    out["tools"] = tool_ids
    out["tool_defs"] = [
        _tool_def_to_dict(tid, _TOOLS[tid]) for tid in tool_ids if tid in _TOOLS
    ]
    return out


def _demo_summary_with_tools(demo: dict) -> dict:
    """Build the summary dict the legacy /api/admin/demos list returns
    (id, label, lang, kb_chars) plus ``tools`` + ``tool_defs``."""
    from tools.registry import REGISTRY as _TOOLS

    kb = demo.get("kb_body")
    if isinstance(kb, dict):
        kb_chars = sum(len(v or "") for v in kb.values())
    else:
        kb_chars = len(kb or "")
    tool_ids = list(demo.get("tool_ids") or [])
    return {
        "id": demo["id"],
        "label": demo["label"],
        "lang": demo["lang"],
        "kb_chars": kb_chars,
        "tools": tool_ids,
        "tool_defs": [
            _tool_def_to_dict(tid, _TOOLS[tid])
            for tid in tool_ids
            if tid in _TOOLS
        ],
        "mcp_servers": list(demo.get("mcp_servers") or []),
    }


def _demo_localized_fields(demo: dict) -> list[str]:
    """Return the demo's inline localized manifest fields that are present
    as non-empty per-language dicts.

    WHITELIST-ANCHORED (Round-1 BLOCKER): candidates come ONLY from
    ``demo_loader.LOCALIZED_REQUIRED + LOCALIZED_OPTIONAL``
    (= ``system`` / ``greeting`` / ``kb_intro`` / ``kb_ack``). These are the
    fields stored *inline* in manifest.yaml as ``{lang: text}`` maps and are
    therefore safe to translate and write back.

    ``kb_body`` is DELIBERATELY and explicitly excluded even though
    ``_load_one`` injects it into the demo dict (demo_loader.py:312) and it
    may *look* like a ``{lang: text}`` dict: when the manifest uses
    ``kb_path``, ``kb_body`` is read from a separate KB file on disk and the
    manifest has no ``kb_body:`` key at all. Translating it would (a) translate
    KB prose (out of scope) and (b) write a phantom ``kb_body:`` map into the
    manifest, breaking the ``kb_path`` contract. We therefore anchor on the
    whitelist and never use a "value looks like a lang dict" heuristic.
    """
    from demo_loader import LOCALIZED_REQUIRED, LOCALIZED_OPTIONAL

    candidates = tuple(LOCALIZED_REQUIRED) + tuple(LOCALIZED_OPTIONAL)
    out: list[str] = []
    for field in candidates:
        if field == "kb_body":  # belt-and-suspenders — never in the whitelist
            continue
        val = demo.get(field)
        if isinstance(val, dict) and any(
            isinstance(v, str) and v.strip() for v in val.values()
        ):
            out.append(field)
    return out


# The single source of truth for which manifest fields may be translated /
# written back via the localized PATCH path. Mirrors _demo_localized_fields'
# candidate set; kb_body is NOT here (see that helper's docstring).
def _localized_field_whitelist() -> tuple[str, ...]:
    from demo_loader import LOCALIZED_REQUIRED, LOCALIZED_OPTIONAL

    return tuple(LOCALIZED_REQUIRED) + tuple(LOCALIZED_OPTIONAL)


def _demo_detail_with_tools(demo: dict) -> dict:
    """Build the detail dict GET /api/admin/demos/{id} returns. Returns the
    FULL ``kb_body`` (str OR per-lang ``{lang: str}``) alongside ``kb_chars``
    (length, for display), plus the ``tools`` + ``tool_defs`` keys and
    ``present_langs`` / ``missing_langs`` derived from the demo's ``system``
    per-lang map.

    kb_body is now returned in full (demos live in DynamoDB, kb_body <= ~26KB,
    well under DDB/HTTP limits) so the SPA editor can seed from the full text
    and round-trip it without truncation. (It used to be popped and replaced
    by a 500-char kb_preview, which truncated long KBs on save.)"""
    out = {k: v for k, v in demo.items() if k != "_dir"}
    body = out.get("kb_body")
    if isinstance(body, dict):
        out["kb_chars"] = {lang: len(text or "") for lang, text in body.items()}
    else:
        out["kb_chars"] = len(body or "")
    # kb_body is intentionally NOT popped — full text round-trips to the editor.

    # present_langs / missing_langs are computed against the demo's ``system``
    # map (the required localized field every demo has). Order follows the
    # LANGUAGES declaration order so the SPA renders deterministically.
    system_map = demo.get("system")
    present = set(system_map.keys()) if isinstance(system_map, dict) else set()
    out["present_langs"] = [k for k in LANGUAGES if k in present]
    out["missing_langs"] = [k for k in LANGUAGES if k not in present]

    out = _augment_demo_with_tools(out)
    return out


@app.get("/api/admin/tools")
async def admin_tools(_: dict = Depends(require_admin)) -> dict:
    """List every tool registered in :data:`tools.registry.REGISTRY`.

    See :func:`_tool_def_to_dict` for the wire shape. The list is in
    REGISTRY iteration order (declaration order) so the admin UI can
    rely on it for stable rendering.
    """
    from tools.registry import REGISTRY as _TOOLS
    return {"tools": [_tool_def_to_dict(tid, defn) for tid, defn in _TOOLS.items()]}


@app.get("/api/admin/demos")
async def admin_demos(_: dict = Depends(require_admin)) -> dict:
    return {
        "demos": [
            _demo_summary_with_tools(d)
            for d in sorted(
                (d for d in (DEMO_LOADER.get(s["id"]) for s in DEMO_LOADER.list()) if d),
                key=lambda x: x["id"],
            )
        ]
    }


async def _rescan_demo_loader() -> int:
    """Call DEMO_LOADER.rescan(), awaiting it when it is async.

    DEMO_LOADER is the DDB-backed DemoStore in production (async rescan), but
    several test suites inject a plain disk-backed DemoLoader (sync rescan) as
    ``bot.DEMO_LOADER``. Tolerating both keeps the admin endpoints correct under
    either backend without each call site caring which is wired in."""
    import inspect

    result = DEMO_LOADER.rescan()
    if inspect.isawaitable(result):
        return await result
    return result


@app.post("/api/admin/demos/rescan")
async def admin_demos_rescan(_: dict = Depends(require_admin)) -> dict:
    n = await _rescan_demo_loader()
    demos = [
        _demo_summary_with_tools(d)
        for d in sorted(
            (d for d in (DEMO_LOADER.get(s["id"]) for s in DEMO_LOADER.list()) if d),
            key=lambda x: x["id"],
        )
    ]
    # last_skipped is a list of {id, reason}; surface it so admins can
    # diagnose why a manifest was rejected.
    return {
        "count": n,
        "demos": demos,
        "last_skipped": list(DEMO_LOADER.last_skipped),
    }


@app.get("/api/admin/demos/{demo_id}")
async def admin_demo_detail(demo_id: str, _: dict = Depends(require_admin)) -> dict:
    demo = DEMO_LOADER.get(demo_id)
    if not demo:
        raise HTTPException(status_code=404, detail=f"demo not found: {demo_id}")
    return _demo_detail_with_tools(demo)


def _resolve_asr_filter(demo_cfg: dict | None, defaults_segment: dict | None) -> dict:
    """Per-field: demo.asr_filter > defaults_segment.asr_filter > env ASR_FILTER_* > built-in(OFF).

    ``defaults_segment`` is the runtime_config segment for the CALL'S SCOPE —
    ``RUNTIME_CONFIG.get_web_defaults()`` for web /ws calls,
    ``RUNTIME_CONFIG.get_phone_defaults()`` for phone calls (passed explicitly by
    the caller — NOT hardwired to web). Returns
    ``{enabled, min_confidence, max_cjk_chars, max_latin_words}`` ready to splat
    into ``TranscriptHallucinationFilter(**...)``. Maps the stored/config/env key
    names ``max_chars`` / ``max_words`` to the ctor names ``max_cjk_chars`` /
    ``max_latin_words`` (a required, tested behaviour — tech_design §2).
    """
    demo_af = (demo_cfg or {}).get("asr_filter") or {}
    global_af = (defaults_segment or {}).get("asr_filter") or {}

    def pick(cfg_key, env_reader, env_name, default):
        if cfg_key in demo_af and demo_af[cfg_key] is not None:
            return demo_af[cfg_key]
        if cfg_key in global_af and global_af[cfg_key] is not None:
            return global_af[cfg_key]
        return env_reader(env_name, default)

    return {
        "enabled":         pick("enabled",        _env_bool,  "ASR_FILTER_ENABLED",        False),
        "min_confidence":  pick("min_confidence", _env_float, "ASR_FILTER_MIN_CONFIDENCE", 0.5),
        "max_cjk_chars":   pick("max_chars",      _env_int,   "ASR_FILTER_MAX_CHARS",      4),   # NOTE name map
        "max_latin_words": pick("max_words",      _env_int,   "ASR_FILTER_MAX_WORDS",      1),   # NOTE name map
    }


def _validate_asr_filter_patch(d: dict) -> dict:
    """Validate a PATCH-body / config ``asr_filter`` block; return the cleaned subset.

    Mirror of :func:`_validate_filler_patch`: each sub-key validated independently
    so a partial body is accepted; only present+valid keys returned (caller merges
    them onto the existing block without clobbering siblings). ``bool`` is an
    ``int`` subclass so the numeric fields explicitly reject bool. Shared by the
    per-demo PATCH (§5) and the global config update (§6) → no drift.

    Rules:
      * ``enabled``        — must be ``bool``
      * ``min_confidence`` — ``int``/``float`` (not bool) in ``0.0..1.0``; stored ``float``
      * ``max_chars``      — ``int`` (not bool) ``>= 0``
      * ``max_words``      — ``int`` (not bool) ``>= 0``
    """
    if not isinstance(d, dict):
        raise HTTPException(status_code=400, detail="asr_filter must be an object")
    clean: dict = {}
    if "enabled" in d:
        if not isinstance(d["enabled"], bool):
            raise HTTPException(status_code=400, detail="asr_filter.enabled must be bool")
        clean["enabled"] = d["enabled"]
    if "min_confidence" in d:
        v = d["min_confidence"]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not (0.0 <= v <= 1.0):
            raise HTTPException(
                status_code=400, detail="asr_filter.min_confidence must be a number in 0..1"
            )
        clean["min_confidence"] = float(v)
    if "max_chars" in d:
        v = d["max_chars"]
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise HTTPException(status_code=400, detail="asr_filter.max_chars must be int >= 0")
        clean["max_chars"] = v
    if "max_words" in d:
        v = d["max_words"]
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise HTTPException(status_code=400, detail="asr_filter.max_words must be int >= 0")
        clean["max_words"] = v
    return clean


def _validate_filler_patch(f: dict) -> dict:
    """Validate a PATCH-body ``filler`` block and return the cleaned subset.

    Each sub-key is validated independently so a partial body (e.g. only
    ``enabled``) is accepted; only present+valid keys are returned, so the
    caller can merge them into an existing manifest ``filler`` block without
    clobbering siblings. ``bool`` is an ``int`` subclass, so ``timeout_ms`` /
    ``probability`` explicitly reject bool. Any invalid value raises
    ``HTTPException(400)`` before the caller writes anything to disk.

    Rules:
      * ``enabled``     — must be ``bool``
      * ``timeout_ms``  — must be ``int`` (not bool) and ``> 0``
      * ``probability`` — must be ``int``/``float`` (not bool) in ``0.0..1.0``
      * ``phrases``     — per-demo override pool (tech_design §4.2): must be a
        ``list`` of ≤20 ``str`` items, each ``.strip()``-ed (blanks dropped) and
        ≤40 chars. ``[]`` (or an all-blank list) is allowed and means "clear the
        override → fall back to the global pool".
    """
    if not isinstance(f, dict):
        raise HTTPException(status_code=400, detail="filler must be an object")
    clean: dict = {}
    if "enabled" in f:
        if not isinstance(f["enabled"], bool):
            raise HTTPException(status_code=400, detail="filler.enabled must be bool")
        clean["enabled"] = f["enabled"]
    if "timeout_ms" in f:
        v = f["timeout_ms"]
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            raise HTTPException(status_code=400, detail="filler.timeout_ms must be int > 0")
        clean["timeout_ms"] = v
    if "probability" in f:
        v = f["probability"]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not (0.0 <= v <= 1.0):
            raise HTTPException(status_code=400, detail="filler.probability must be a number in 0..1")
        clean["probability"] = float(v)
    if "phrases" in f:
        v = f["phrases"]
        if not isinstance(v, list):
            raise HTTPException(status_code=400, detail="filler.phrases must be a list")
        if len(v) > 20:
            raise HTTPException(status_code=400, detail="filler.phrases: at most 20 items")
        cleaned: list[str] = []
        for item in v:
            if not isinstance(item, str):
                raise HTTPException(
                    status_code=400, detail="filler.phrases items must be strings"
                )
            s = item.strip()
            if not s:
                continue  # drop blanks
            if len(s) > 40:
                raise HTTPException(
                    status_code=400, detail="filler.phrases item too long (max 40 chars)"
                )
            cleaned.append(s)
        clean["phrases"] = cleaned  # [] allowed → clears the override (global pool)
    return clean


def _validate_engine_voice_patch(
    body,
    fields,
    *,
    demo_engine: str | None = None,
    demo_provider: str | None = None,
) -> dict:
    """Validate the engine/provider/voice subset of a PATCH body and return the
    cleaned subset (parity with :func:`_validate_filler_patch`).

    Only keys present in ``fields`` are processed. ``fields`` is the body's
    ``model_fields_set`` (so an explicit ``null`` IS present and clears the
    field — distinct from an omitted key). A ``None`` value is recorded as
    ``None`` in the result so the caller merges it as a clear. Any invalid value
    raises ``HTTPException(400)`` BEFORE the caller writes anything.

    Rules (only for keys in ``fields``):
      * ``engine``   — ``None`` or ``in ENGINES`` ({pipeline, nova-sonic}); else 400.
      * ``provider`` — ``None`` or ``in TTS_PROVIDERS`` ({minimax, polly}); else 400.
      * ``voice``    — ``None`` allowed (clear). A non-null voice must resolve
        against the EFFECTIVE engine after merge:
          - effective engine == nova-sonic →
            ``NOVA_SONIC_VOICE_ALIASES.get(v, v) in NOVA_SONIC_VOICES`` else 400.
          - effective engine == pipeline →
            exact match in the EFFECTIVE provider's table
            (``_resolve_minimax_voice(v)[0] == v`` for minimax, ``v in
            POLLY_VOICES`` for polly — exact so a typo isn't swallowed by the
            resolver's default fallback) else 400.
          - effective engine UNSET (neither body nor demo sets it) → lenient:
            accept a voice that resolves in EITHER the nova table OR a pipeline
            provider table; the real engine is decided at launch.

    ``demo_engine`` / ``demo_provider`` are the demo's CURRENT values, used when
    the body does not set engine / provider so the effective value is correct
    whether a field changes in this PATCH or was set previously.

    Note: Nova Sonic supports tools/MCP (the builder registers registry +
    MCP tools into the bidi LLM), so there is NO tools→engine constraint here.
    The only Nova Sonic constraint is language (no zh/ja), enforced separately
    by the engine×language cross-filter.
    """
    clean: dict = {}

    # ---- engine ----------------------------------------------------------
    if "engine" in fields:
        v = getattr(body, "engine", None)
        if v is not None and v not in ENGINES:
            raise HTTPException(
                status_code=400,
                detail=f"unknown engine: {v!r} (allowed: {sorted(ENGINES)})",
            )
        clean["engine"] = v

    # ---- provider --------------------------------------------------------
    if "provider" in fields:
        v = getattr(body, "provider", None)
        if v is not None and v not in TTS_PROVIDERS:
            raise HTTPException(
                status_code=400,
                detail=f"unknown provider: {v!r} (allowed: {sorted(TTS_PROVIDERS)})",
            )
        clean["provider"] = v

    # ---- effective engine / provider (after merge) -----------------------
    # engine: body value if engine is in this PATCH (incl. explicit null), else
    # the demo's current engine, else DEFAULT_ENGINE — BUT keep UNSET as a
    # distinct state when neither body nor demo sets it, so a voice-only PATCH on
    # an engine-unset demo is validated leniently (real engine decided at launch).
    if "engine" in fields:
        eff_engine = clean["engine"]  # may be None (explicit clear)
    else:
        eff_engine = demo_engine  # may be None (demo never set it)

    # provider: body value if provider is in this PATCH, else demo's current,
    # else DEFAULT_PROVIDER (only consulted on the pipeline path).
    if "provider" in fields:
        eff_provider = clean["provider"]
    else:
        eff_provider = demo_provider
    if eff_provider is None:
        eff_provider = DEFAULT_PROVIDER

    # ---- voice -----------------------------------------------------------
    if "voice" in fields:
        v = getattr(body, "voice", None)
        if v is None:
            clean["voice"] = None  # explicit clear
        else:
            if eff_engine == "nova-sonic":
                # Alias migration BEFORE the registry membership check (N1).
                if NOVA_SONIC_VOICE_ALIASES.get(v, v) not in voices_for("nova-sonic"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"voice {v!r} is not a Nova Sonic voice",
                    )
            elif eff_engine == "pipeline":
                if not _voice_in_provider_table(v, eff_provider):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"voice {v!r} does not resolve in provider "
                            f"{eff_provider!r}'s table"
                        ),
                    )
            else:
                # Engine UNSET (neither body nor demo): lenient — accept if the
                # voice resolves in the nova table OR any pipeline provider
                # table. The real engine is decided at launch.
                nova_ok = NOVA_SONIC_VOICE_ALIASES.get(v, v) in voices_for("nova-sonic")
                pipe_ok = any(
                    _voice_in_provider_table(v, p) for p in TTS_PROVIDERS
                )
                if not (nova_ok or pipe_ok):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"voice {v!r} does not resolve in any engine's "
                            f"voice table"
                        ),
                    )
            clean["voice"] = v

    # ---- model (per-demo LLM override) -----------------------------------
    # None allowed (explicit clear → inherit DEFAULT_MODEL at launch); a non-null
    # value must be a known key in MODELS (parity with the global config path's
    # `if model not in MODELS: 400`). Engine-agnostic: only the pipeline engine
    # consumes it, but storing it on a nova-sonic demo is harmless (ignored).
    if "model" in fields:
        v = getattr(body, "model", None)
        if v is not None and v not in MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"invalid model: {v!r} (allowed: {sorted(MODELS)})",
            )
        clean["model"] = v

    return clean


def _voice_in_provider_table(v: str, provider: str | None) -> bool:
    """True iff voice ``v`` is an EXACT member of ``provider``'s voice table.

    Exact match (not the resolver's default fallback) so a typo can't be
    silently swallowed: ``_resolve_minimax_voice(v)[0] == v`` for minimax,
    ``v in POLLY_VOICES`` for polly (case-sensitive — Polly keys are
    case-sensitive). An unknown provider → False."""
    if provider == "minimax":
        return _resolve_minimax_voice(v)[0] == v
    if provider == "polly":
        return v in voices_for("polly")
    return False


def _validate_launch_config(
    *,
    lang: str | None = None,
    engine: str | None = None,
    voice: str | None = None,
    provider: str | None = None,
) -> dict:
    """Validate an optional launch-config quadruple (used by the guest-link mint)
    and return the cleaned subset. All fields optional; any invalid field raises
    ``HTTPException(400)``; all-None → ``{}``.

    This COMPOSES the same primitives the demo/config validators use — it does
    NOT extract from / modify ``_validate_engine_voice_patch`` (which has no lang
    param and no engine×lang check). Shared primitives:
      * ``engine``   — None or ``in ENGINES``.
      * ``provider`` — None or ``in TTS_PROVIDERS``.
      * ``lang``     — None or ``in LANGUAGES``.
      * engine×lang  — if BOTH lang and engine are given and ``engine`` is not in
        ``LANGUAGES[lang]['engines']`` → 400 (THIS is what rejects
        nova-sonic + zh-HK).
      * ``voice``    — None or resolves against the EFFECTIVE engine/provider
        (``engine or DEFAULT_ENGINE`` / ``provider or DEFAULT_PROVIDER``): for
        nova-sonic via ``NOVA_SONIC_VOICE_ALIASES`` + ``voices_for('nova-sonic')``,
        else via ``_voice_in_provider_table``.
    """
    clean: dict = {}
    if engine is not None:
        if engine not in ENGINES:
            raise HTTPException(status_code=400, detail=f"invalid engine: {engine!r}")
        clean["engine"] = engine
    if provider is not None:
        if provider not in TTS_PROVIDERS:
            raise HTTPException(status_code=400, detail=f"invalid provider: {provider!r}")
        clean["provider"] = provider
    if lang is not None:
        if lang not in LANGUAGES:
            raise HTTPException(status_code=400, detail=f"invalid lang: {lang!r}")
        clean["lang"] = lang
        # engine×language: if an engine is also given, it must serve this lang.
        if engine and engine not in (LANGUAGES[lang].get("engines") or []):
            raise HTTPException(
                status_code=400,
                detail=f"engine {engine!r} not available for language {lang!r}",
            )
    if voice is not None:
        eff_engine = engine or DEFAULT_ENGINE
        eff_provider = provider or DEFAULT_PROVIDER
        if eff_engine == "nova-sonic":
            if NOVA_SONIC_VOICE_ALIASES.get(voice, voice) not in voices_for("nova-sonic"):
                raise HTTPException(status_code=400, detail=f"invalid nova-sonic voice: {voice!r}")
        else:
            if not _voice_in_provider_table(voice, eff_provider):
                raise HTTPException(status_code=400, detail=f"invalid pipeline voice: {voice!r}")
        clean["voice"] = voice
    return clean


class DemoPatchBody(BaseModel):
    """Body of PATCH /api/admin/demos/{id}.

    Full-field editor (T5): the DynamoDB-backed DemoStore is the single
    source of truth, so the admin UI can now edit every demo field, not
    just tools/mcp_servers. The endpoint merges the provided fields onto
    the CURRENT demo dict (``DEMO_LOADER.get(id)``), then ``put()``s the
    merged dict to DDB and ``rescan()``s. ``manifest.yaml`` is NO LONGER
    written — the repo ``data/`` tree is a one-time migration seed only.

    Editable fields (all optional; an omitted field is left untouched):

      * ``label``      — non-empty str (the human display name)
      * ``lang``       — str, must be a key of ``LANGUAGES`` (default lang)
      * ``tags``       — list[str]
      * ``system`` / ``greeting`` / ``kb_intro`` / ``kb_ack`` — each a
        per-language ``{lang: str}`` map; every key must be in ``LANGUAGES``
        and every value a str. A provided map REPLACES the field wholesale
        (full set semantics).
      * ``kb_body``    — either a single str OR a per-language ``{lang: str}``
        map (KB prose per language).
      * ``tools``      — list[str] of registered tool ids (validated against
        ``tools.registry.REGISTRY``); persisted under the demo's internal
        ``tool_ids`` key.
      * ``mcp_servers`` — list[str] of MCP server ids (shape-validated).
      * ``filler``     — per-demo override of the global FILLER_* config
        (``enabled`` / ``timeout_ms`` / ``probability`` / ``phrases``);
        partial sub-field merge (see :func:`_validate_filler_patch`).
        ``phrases`` is a ``list[str]`` that fully replaces the language pool
        when non-empty; ``[]`` clears the override.

    ``id`` is NOT editable — it is the DDB partition key. An ``id`` in the
    body is rejected (400) so a client never silently believes it renamed a
    demo.

    ``localized`` is the legacy fine-grained ``{field: {lang: text}}`` write
    path (confirmed translations from the translate endpoint). It coexists
    with the direct per-lang field edits above: where a direct ``system``
    map replaces wholesale, ``localized["system"]`` merges individual langs
    (and still honours the ``overwrite`` conflict guard). Field names are
    validated against the whitelist (``kb_body`` or anything else → 400) and
    each lang must be in ``LANGUAGES``.
    """

    # Existing fields (preserved).
    tools: list[str] | None = None
    mcp_servers: list[str] | None = None
    localized: dict[str, dict[str, str]] | None = None
    filler: dict | None = None
    asr_filter: dict | None = None
    overwrite: bool = False
    # New full-field editor surface (T5).
    label: str | None = None
    lang: str | None = None
    tags: list[str] | None = None
    system: dict[str, str] | None = None
    greeting: dict[str, str] | None = None
    kb_intro: dict[str, str] | None = None
    kb_ack: dict[str, str] | None = None
    kb_body: str | dict[str, str] | None = None
    # Optional per-demo engine/voice/provider overrides. All Optional; an
    # explicit ``null`` (present in ``model_fields_set``) clears the field, an
    # omitted key leaves it untouched. Validated by _validate_engine_voice_patch.
    engine: str | None = None
    provider: str | None = None
    voice: str | None = None
    # Optional per-demo LLM (Bedrock model) override. None/omitted → inherit the
    # global DEFAULT_MODEL at launch, exactly like engine/provider/voice. An
    # explicit ``null`` (present in ``model_fields_set``) clears it. Validated by
    # _validate_engine_voice_patch (must be in MODELS or null).
    model: str | None = None
    # ``id`` is accepted into the model purely so we can detect and reject an
    # attempt to change it — it is never applied to the merged demo.
    id: str | None = None


@app.patch("/api/admin/demos/{demo_id}")
async def admin_demo_patch(demo_id: str, body: DemoPatchBody, admin: dict = Depends(require_admin)) -> dict:
    """Full-field edit of a demo: validate the provided fields, merge them
    onto the CURRENT demo dict, persist to DynamoDB via ``DEMO_LOADER.put``,
    then ``rescan`` so the in-memory cache (and the next call) sees it.

    ``manifest.yaml`` is intentionally NOT written — DynamoDB is the single
    source of truth (the repo ``data/`` tree was a one-time migration seed).
    All validation runs BEFORE the single ``put``, so an invalid body 400s
    without ever mutating the store. ``id`` is the partition key and is not
    editable.
    """
    from tools.registry import REGISTRY as _TOOLS

    demo = DEMO_LOADER.get(demo_id)
    if not demo:
        raise HTTPException(status_code=404, detail=f"demo not found: {demo_id}")

    fields = body.model_dump(exclude_unset=True)

    # ``id`` is the partition key — never editable. Reject an attempt to change
    # it (a no-op id == demo_id is tolerated as harmless), so a client can't
    # silently believe it renamed a demo.
    if "id" in fields and body.id != demo_id:
        raise HTTPException(
            status_code=400,
            detail="id is not editable (it is the demo's primary key)",
        )

    # ``overwrite`` / ``id`` are modifiers, not content. A body carrying only
    # those is "did you ask to change anything?"-empty. Every editable field
    # (incl. all the T5 full-field additions) counts as content so e.g. a
    # label-only or system-only body is NOT rejected.
    _content_keys = (
        "tools", "mcp_servers", "localized", "filler", "asr_filter",
        "label", "lang", "tags",
        "system", "greeting", "kb_intro", "kb_ack", "kb_body",
        "engine", "provider", "voice", "model",
    )
    has_content = any(k in fields for k in _content_keys)
    if not has_content:
        raise HTTPException(
            status_code=400,
            detail=f"body must include at least one of: {', '.join(_content_keys)}",
        )

    # ---- per-field validation (all BEFORE any write) ----------------------

    requested = None
    if "tools" in fields:
        requested = list(body.tools or [])
        unknown = [t for t in requested if t not in _TOOLS]
        if unknown:
            # Surface every unknown id, sorted, in a stable shape so the
            # admin SPA can render a precise error.
            raise HTTPException(
                status_code=400,
                detail=f"unknown tool ids: {sorted(unknown)}",
            )

    requested_mcp = None
    if "mcp_servers" in fields:
        requested_mcp = list(body.mcp_servers or [])
        # Existence against MCP_CONFIG is deliberately NOT enforced —
        # servers can be registered after the demo (unknown ids are
        # skipped with a WARNING at pipeline-build time). We only reject
        # entries that could never be a valid server id.
        from mcp_config import SERVER_ID_RE as _MCP_ID_RE
        bad = [m for m in requested_mcp if not isinstance(m, str) or not _MCP_ID_RE.match(m)]
        if bad:
            raise HTTPException(
                status_code=400,
                detail=f"invalid mcp server ids: {sorted(map(str, bad))}",
            )

    # label — non-empty string.
    if "label" in fields:
        if not isinstance(body.label, str) or not body.label.strip():
            raise HTTPException(status_code=400, detail="label must be a non-empty string")

    # lang — must be a known language key.
    if "lang" in fields:
        if not isinstance(body.lang, str) or body.lang not in LANGUAGES:
            raise HTTPException(status_code=400, detail=f"unknown language: {body.lang!r}")

    # tags — list of strings.
    if "tags" in fields:
        if not isinstance(body.tags, list) or any(not isinstance(t, str) for t in body.tags):
            raise HTTPException(status_code=400, detail="tags must be a list of strings")

    # system / greeting / kb_intro / kb_ack — each a {lang: str} map whose keys
    # are all in LANGUAGES and whose values are all str. A provided map REPLACES
    # the field wholesale (full set semantics).
    PER_LANG_FIELDS = ("system", "greeting", "kb_intro", "kb_ack")
    for fname in PER_LANG_FIELDS:
        if fname not in fields:
            continue
        val = getattr(body, fname)
        if not isinstance(val, dict):
            raise HTTPException(
                status_code=400,
                detail=f"{fname} must be a {{lang: text}} object",
            )
        for lang, text in val.items():
            if lang not in LANGUAGES:
                raise HTTPException(status_code=400, detail=f"unknown language: {lang!r}")
            if not isinstance(text, str):
                raise HTTPException(
                    status_code=400,
                    detail=f"{fname}[{lang!r}] must be a string",
                )

    # kb_body — either a single str OR a {lang: str} map.
    if "kb_body" in fields:
        kb = body.kb_body
        if isinstance(kb, dict):
            for lang, text in kb.items():
                if lang not in LANGUAGES:
                    raise HTTPException(status_code=400, detail=f"unknown language: {lang!r}")
                if not isinstance(text, str):
                    raise HTTPException(
                        status_code=400,
                        detail=f"kb_body[{lang!r}] must be a string",
                    )
        elif not isinstance(kb, str):
            raise HTTPException(
                status_code=400,
                detail="kb_body must be a string or a {lang: text} object",
            )

    # filler validation (per-demo override of the global FILLER_* env). Runs
    # BEFORE any write so an invalid value 400s without touching the store.
    # Extracted to a module-level helper so the validation logic is unit-testable
    # without the bcrypt-backed auth path (see tests/test_timeout_filler.py).
    filler_clean: dict | None = None
    if "filler" in fields and body.filler is not None:
        filler_clean = _validate_filler_patch(body.filler)

    # asr_filter validation (per-demo override of the global ASR_FILTER_* env /
    # config). Same partial-merge contract as filler; shared validator with the
    # global config path so the rules never drift. Runs BEFORE any write.
    asr_filter_clean: dict | None = None
    if "asr_filter" in fields and body.asr_filter is not None:
        asr_filter_clean = _validate_asr_filter_patch(body.asr_filter)

    # engine / provider / voice / model validation (per-demo override). Uses
    # ``model_fields_set`` for presence so an explicit ``null`` is detected and
    # clears the field (distinct from an omitted key). Runs BEFORE any write.
    # ``model`` is included so a model-only PATCH still generates ``clean_ev``.
    ev_fields = body.model_fields_set
    clean_ev: dict = {}
    if any(k in ev_fields for k in ("engine", "provider", "voice", "model")):
        clean_ev = _validate_engine_voice_patch(
            body,
            ev_fields,
            demo_engine=demo.get("engine"),
            demo_provider=demo.get("provider"),
        )

    # localized write-back validation (independent of the translate endpoint —
    # this is the second whitelist guard mandated by the Round-1 BLOCKER).
    localized = body.localized if "localized" in fields else None
    if localized is not None:
        whitelist = set(_localized_field_whitelist())
        if not isinstance(localized, dict):
            raise HTTPException(
                status_code=400,
                detail="localized must be an object of {field: {lang: text}}",
            )
        for field_name, lang_map in localized.items():
            # GUARD: reject kb_body or any non-whitelist key outright. This is
            # deliberately independent of _demo_localized_fields so a buggy /
            # malicious client cannot smuggle a phantom kb_body: map in and
            # break the kb_path contract.
            if field_name not in whitelist:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"field not allowed for localized write-back: "
                        f"{field_name!r}; allowed: {sorted(whitelist)}"
                    ),
                )
            if not isinstance(lang_map, dict) or not lang_map:
                raise HTTPException(
                    status_code=400,
                    detail=f"localized[{field_name!r}] must be a non-empty {{lang: text}} object",
                )
            for lang in lang_map:
                if lang not in LANGUAGES:
                    raise HTTPException(
                        status_code=400,
                        detail=f"unknown language: {lang!r}",
                    )

    # ---- merge onto the CURRENT demo dict ---------------------------------
    # Start from a copy of the cached demo (canonical key shape, incl. the
    # internal ``tool_ids`` key) and apply each provided field. ``put`` re-runs
    # normalize_demo_dict, so dropping the non-canonical ``_dir`` here is purely
    # tidy. We deliberately do NOT carry ``id`` from the body — id stays pinned
    # to the path param / partition key.
    merged: dict = {k: v for k, v in demo.items() if k != "_dir"}
    merged["id"] = demo_id

    if "label" in fields:
        merged["label"] = body.label
    if "lang" in fields:
        merged["lang"] = body.lang
    if "tags" in fields:
        merged["tags"] = list(body.tags)
    for fname in PER_LANG_FIELDS:
        if fname in fields:
            merged[fname] = dict(getattr(body, fname))
    if "kb_body" in fields:
        merged["kb_body"] = dict(body.kb_body) if isinstance(body.kb_body, dict) else body.kb_body

    # tools persist under the internal ``tool_ids`` key (BLOCKER #1) — never a
    # ``tools`` key. normalize_demo_dict in put() also accepts ``tool_ids``, so
    # writing it here keeps the key shape single-sourced and KeyError-free at
    # pipeline-build time.
    if requested is not None:
        merged["tool_ids"] = requested
    if requested_mcp is not None:
        merged["mcp_servers"] = requested_mcp

    if filler_clean is not None:
        # Partial sub-field merge: only the validated keys are overwritten, so
        # sending {enabled: ...} alone leaves an existing timeout_ms /
        # probability intact. An empty {} filler body is a no-op.
        existing_filler = merged.get("filler")
        if not isinstance(existing_filler, dict):
            existing_filler = {}
        else:
            existing_filler = dict(existing_filler)
        existing_filler.update(filler_clean)
        merged["filler"] = existing_filler

    if asr_filter_clean is not None:
        # Partial sub-field merge: only validated keys overwrite, so sending
        # {enabled: ...} alone leaves an existing min_confidence / max_chars /
        # max_words intact. An empty {} asr_filter body is a no-op.
        existing_asr = merged.get("asr_filter")
        if not isinstance(existing_asr, dict):
            existing_asr = {}
        else:
            existing_asr = dict(existing_asr)
        existing_asr.update(asr_filter_clean)
        merged["asr_filter"] = existing_asr

    # engine / provider / voice / model merge — only keys present in clean_ev
    # (i.e. in the PATCH) are applied; a value of None clears the field.
    # normalize_demo_dict in put() keeps the keys. ``model`` MUST be in this loop
    # or a validated model would never reach ``merged`` → never persist to DDB.
    for k in ("engine", "provider", "voice", "model"):
        if k in clean_ev:
            merged[k] = clean_ev[k]

    if localized is not None:
        # First pass: reject any (field, lang) that already exists unless
        # overwrite=True. Checking fully before mutating means a single
        # conflicting lang aborts the whole PATCH with no partial write. The
        # "existing" view is the merged dict, so a direct ``system`` map set in
        # THIS same body is what localized merges onto (full-set + fine-grained
        # coexist).
        for field_name, lang_map in localized.items():
            existing = merged.get(field_name)
            if isinstance(existing, dict):
                for lang in lang_map:
                    if lang in existing and not body.overwrite:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"{field_name}[{lang}] already exists; "
                                f"set overwrite=true to replace it"
                            ),
                        )
        # Second pass: apply onto a copy of the field map (fine-grained merge,
        # preserving sibling langs).
        for field_name, lang_map in localized.items():
            target = merged.get(field_name)
            target = dict(target) if isinstance(target, dict) else {}
            for lang, text in lang_map.items():
                target[lang] = text
            merged[field_name] = target

    # ---- persist: write DDB, then refresh the cache -----------------------
    await DEMO_LOADER.put(merged)
    await _rescan_demo_loader()
    refreshed = DEMO_LOADER.get(demo_id)
    if not refreshed:
        # Should never happen — we just put a valid demo — but guard so a
        # corrupted store state surfaces as a 500 instead of a 404 mid-flight.
        raise HTTPException(
            status_code=500,
            detail=f"demo {demo_id} disappeared after rescan",
        )
    # Detail: changed field NAMES only (never the field bodies) — type
    # 'demo-edit' allowlists only {changed}.
    await log_activity(
        actor=admin, type="demo-edit", target=demo_id,
        detail={"changed": sorted(k for k in fields if k not in ("id", "overwrite"))},
    )
    return _demo_detail_with_tools(refreshed)


@app.get("/api/admin/scenarios")
async def admin_scenarios(_: dict = Depends(require_admin)) -> dict:
    """Legacy alias kept for the old admin SPA.

    Returns the SAME shape as /api/admin/demos. Both routes are stable
    contracts — the new SPA reads /api/admin/demos, the legacy SPA reads
    /api/admin/scenarios. We don't transform field names because the
    legacy SPA already accepts the demo shape (``id`` / ``label`` /
    ``lang`` / ``kb_chars``).
    """
    return await admin_demos()


# --- MCP server registry endpoints ----------------------------------------
# Auth: admin-only via Depends(require_admin) on each route (the legacy
# admin_path_guard middleware was removed). Headers may contain secrets, so GET responses mask every
# header value as "***"; POST treats a "***" value as "keep the stored
# value for this key" (see mcp_config.HEADER_MASK).

class McpServerBody(BaseModel):
    """Body of POST /api/admin/mcp-servers (upsert by id)."""

    id: str
    label: str | None = None
    transport: str
    url: str
    headers: dict[str, str] | None = None
    enabled: bool = True
    # auth object {type: none|header|sigv4, service?, region?}. Optional and
    # validated/normalized by McpConfig.upsert -> validate_auth (missing = none).
    # Without this field Pydantic would silently drop the SigV4 auth the admin
    # UI sends, so a sigv4 server would save as auth=none and 403 at connect.
    auth: dict[str, Any] | None = None


def _demos_referencing_mcp_server(server_id: str) -> list[str]:
    """Return ids of demos whose manifest mounts ``server_id``."""
    out = []
    for summary in DEMO_LOADER.list():
        demo = DEMO_LOADER.get(summary["id"])
        if demo and server_id in (demo.get("mcp_servers") or []):
            out.append(demo["id"])
    return sorted(out)


@app.get("/api/admin/mcp-servers")
async def admin_mcp_servers_list(_: dict = Depends(require_admin)) -> dict:
    return {"servers": [_mask_mcp_headers(s) for s in MCP_CONFIG.list_servers()]}


@app.post("/api/admin/mcp-servers")
async def admin_mcp_servers_upsert(body: McpServerBody, admin: dict = Depends(require_admin)) -> dict:
    try:
        stored = MCP_CONFIG.upsert(body.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Detail: id/label/transport/enabled ONLY — NEVER url/headers/auth (the
    # mcp-* allowlist masks those even if a caller passed them).
    await log_activity(
        actor=admin, type="mcp-upsert", target=stored.get("id"),
        detail={k: stored.get(k) for k in ("id", "label", "transport", "enabled") if k in stored},
    )
    return {"server": _mask_mcp_headers(stored)}


@app.delete("/api/admin/mcp-servers/{server_id}")
async def admin_mcp_servers_delete(server_id: str, admin: dict = Depends(require_admin)) -> dict:
    refs = _demos_referencing_mcp_server(server_id)
    if refs:
        raise HTTPException(
            status_code=409,
            detail={
                "message": f"mcp server {server_id!r} is referenced by demos",
                "demos": refs,
            },
        )
    if not MCP_CONFIG.delete(server_id):
        raise HTTPException(status_code=404, detail=f"mcp server not found: {server_id}")
    await log_activity(
        actor=admin, type="mcp-delete", target=server_id, detail={"id": server_id},
    )
    return {"deleted": server_id}


@app.post("/api/admin/mcp-servers/{server_id}/test")
async def admin_mcp_servers_test(server_id: str, _: dict = Depends(require_admin)) -> dict:
    """Connect to the server (3s timeout), list its tools, disconnect.

    Always returns 200 with ``{ok, tools, error}`` — connection problems
    are data, not transport errors, so the SPA can render them inline.
    """
    cfg = MCP_CONFIG.get(server_id)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"mcp server not found: {server_id}")

    # Lazy imports: the `mcp` package is optional. A missing package must
    # not break bot.py startup — only this endpoint reports it.
    try:
        from pipecat.services.mcp_service import MCPClient
        from mcp.client.session_group import SseServerParameters, StreamableHttpParameters
    except Exception as e:
        logger.warning(f"[mcp] python 'mcp' package unavailable: {e}")
        return {"ok": False, "tools": [], "error": "mcp package not installed"}

    auth = cfg.get("auth") or {}
    if auth.get("type") == "sigv4":
        try:
            import mcp_sigv4
            params = mcp_sigv4.make_sigv4_streamable_params(
                cfg["url"],
                service=auth.get("service") or "bedrock-agentcore",
                region=auth.get("region") or "us-east-1",
                headers=cfg.get("headers") or None,
            )
        except Exception as e:  # MissingCredentialsError or any setup failure
            logger.warning(f"[mcp] /test {server_id!r} sigv4 setup failed: {e!r}")
            return {"ok": False, "tools": [], "error": f"sigv4 setup failed: {e}"}
    else:
        params_cls = SseServerParameters if cfg["transport"] == "sse" else StreamableHttpParameters
        params = params_cls(url=cfg["url"], headers=cfg.get("headers") or {})

    async def _probe() -> list[str]:
        # start / get_tools_schema / close MUST all run in the same asyncio
        # task: MCPClient holds anyio cancel scopes (AsyncExitStack) that
        # blow up with "exit cancel scope in a different task" otherwise —
        # asyncio.wait_for wraps its awaitable in a separate task, so the
        # whole lifecycle lives inside this coroutine and wait_for wraps it
        # as a unit. The client is always closed (finally), even on cancel.
        client = MCPClient(server_params=params)
        try:
            await client.start()
            tools_schema = await client.get_tools_schema()
            return [fs.name for fs in (tools_schema.standard_tools or [])]
        finally:
            try:
                await client.close()
            except BaseException:
                pass

    # Run the probe in its OWN task and time it out with asyncio.wait
    # instead of asyncio.wait_for: the MCP transports use anyio cancel
    # scopes that mangle wait_for's cancellation (a bare CancelledError /
    # "Cancelled via cancel scope" can escape instead of TimeoutError).
    # Containing the probe in a task lets us reap any such wreckage here.
    probe_task = asyncio.ensure_future(_probe())
    done, pending = await asyncio.wait({probe_task}, timeout=3.0)
    if pending:
        probe_task.cancel()
        try:
            await probe_task
        except BaseException:
            pass
        return {"ok": False, "tools": [], "error": "connection timed out (3s)"}
    try:
        names = probe_task.result()
        return {"ok": True, "tools": names, "error": None}
    except asyncio.CancelledError:
        # The MCP transports' anyio cancel scopes can surface a connect
        # failure as a bare CancelledError on the (already-finished) probe
        # task. We were not cancelled ourselves — the task is done — so
        # report it as a connection failure rather than propagating.
        return {"ok": False, "tools": [], "error": "connection failed (transport cancelled)"}
    except BaseExceptionGroup as eg:
        # anyio task-group failures (connect errors inside the MCP client's
        # HTTP transport) surface as BaseExceptionGroup, which `except
        # Exception` does NOT catch. Flatten to the first leaf for a
        # readable error message.
        leaf: BaseException = eg
        while isinstance(leaf, BaseExceptionGroup) and leaf.exceptions:
            leaf = leaf.exceptions[0]
        return {"ok": False, "tools": [], "error": f"{type(leaf).__name__}: {leaf}"}
    except Exception as e:
        return {"ok": False, "tools": [], "error": f"{type(e).__name__}: {e}"}


# --- Voice registry admin API (Part A / A.5) -----------------------------
# Admin-editable voice registry. Auth: admin-only via Depends(require_admin) on
# each route. Writes/deletes go to VOICE_STORE (DDB), then `await
# VOICE_STORE.rescan()` so the in-memory cache + /api/config reflect the change
# immediately. Each mutation emits a voice-* activity row via log_activity (a
# best-effort no-op until T3 wires the real helper).

_VOICE_PROVIDERS = ("minimax", "polly", "nova-sonic")


def _validate_voice_body(provider: str, body: dict) -> dict:
    """Validate + normalize a voice attr dict for ``provider``; return the
    cleaned attrs (NO provider/voice_id keys). Invalid -> HTTPException(400).

    Per-provider required fields (A.5):
      * all: ``id`` (== voice_id) non-empty; provider ∈ {minimax,polly,nova-sonic}.
      * minimax: ``language`` (str) + numeric ``boost``.
      * polly:   ``language`` (str) + ``engine`` (str).
      * nova-sonic: ``locale`` present; ``polyglot`` bool (default False);
        ``lang_label`` optional.
    ``label``/``gender`` are accepted for every provider (optional)."""
    if provider not in _VOICE_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown provider: {provider!r} (allowed: {sorted(_VOICE_PROVIDERS)})",
        )
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

    voice_id = body.get("id") or body.get("voice_id")
    if not isinstance(voice_id, str) or not voice_id.strip():
        raise HTTPException(status_code=400, detail="voice id must be a non-empty string")

    # Common optional fields.
    attrs: dict = {}
    # label is rendered by _voice_display (v["label"]) on every /api/config +
    # /api/admin/options fetch, for ALL providers — so a row MUST always carry a
    # label. Default it to the voice id when omitted (an admin POST without a
    # label is valid) so a label-less voice can never 500 the config endpoints.
    label = body.get("label")
    attrs["label"] = str(label) if label is not None and str(label).strip() else voice_id
    if body.get("gender") is not None:
        attrs["gender"] = str(body["gender"])

    if provider == "minimax":
        lang = body.get("language")
        if not isinstance(lang, str) or not lang.strip():
            raise HTTPException(status_code=400, detail="minimax voice requires a 'language' string")
        boost = body.get("boost")
        if not isinstance(boost, (int, float, str)) or isinstance(boost, bool):
            raise HTTPException(status_code=400, detail="minimax voice requires a numeric/string 'boost'")
        attrs["language"] = lang
        attrs["boost"] = boost
    elif provider == "polly":
        lang = body.get("language")
        if not isinstance(lang, str) or not lang.strip():
            raise HTTPException(status_code=400, detail="polly voice requires a 'language' string")
        engine = body.get("engine")
        if not isinstance(engine, str) or not engine.strip():
            raise HTTPException(status_code=400, detail="polly voice requires an 'engine' string")
        attrs["language"] = lang
        attrs["engine"] = engine
    else:  # nova-sonic
        locale = body.get("locale")
        if not isinstance(locale, str) or not locale.strip():
            raise HTTPException(status_code=400, detail="nova-sonic voice requires a 'locale'")
        polyglot = body.get("polyglot", False)
        if not isinstance(polyglot, bool):
            raise HTTPException(status_code=400, detail="nova-sonic 'polyglot' must be a boolean")
        attrs["locale"] = locale
        attrs["polyglot"] = polyglot
        if body.get("lang_label") is not None:
            attrs["lang_label"] = str(body["lang_label"])

    return attrs


def _voice_safe_detail(provider: str, voice_id: str, attrs: dict | None) -> dict:
    """A curated, secret-free detail map for the activity log (Part B B.3):
    provider + voice_id + the changed/safe voice attrs only (no raw body)."""
    detail = {"provider": provider, "voice_id": voice_id}
    for k in ("label", "gender", "language", "locale", "polyglot", "engine", "boost"):
        if attrs and k in attrs:
            detail[k] = attrs[k]
    return detail


@app.get("/api/admin/voices")
async def admin_voices_list(provider: str | None = None, _: dict = Depends(require_admin)) -> dict:
    """List registry voices. ``?provider=`` filters; otherwise all providers.

    Reads through ``voices_for`` so the response is identical whether the live
    table exists (registry) or not (in-code constants)."""
    if provider is not None and provider not in _VOICE_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown provider: {provider!r} (allowed: {sorted(_VOICE_PROVIDERS)})",
        )
    providers = [provider] if provider else list(_VOICE_PROVIDERS)
    out: dict[str, list] = {}
    for p in providers:
        out[p] = [
            {"id": vid, "provider": p, **attrs}
            for vid, attrs in voices_for(p).items()
        ]
    return {"voices": out, "live": VOICE_STORE.live}


@app.post("/api/admin/voices")
async def admin_voices_create(body: dict, admin: dict = Depends(require_admin)) -> dict:
    """Create/overwrite a voice (upsert by (provider, voice_id))."""
    provider = body.get("provider")
    voice_id = (body.get("id") or body.get("voice_id") or "").strip()
    attrs = _validate_voice_body(provider, body)
    await VOICE_STORE.put(provider, voice_id, attrs)
    await VOICE_STORE.rescan()
    try:
        await log_activity(
            actor=admin.get("username"),
            actor_role=admin.get("role"),
            type="voice-create",
            target=f"{provider}/{voice_id}",
            detail=_voice_safe_detail(provider, voice_id, attrs),
        )
    except Exception:  # pragma: no cover — logging never breaks the op
        pass
    return {"voice": {"id": voice_id, "provider": provider, **(VOICE_STORE.get(provider, voice_id) or attrs)}}


@app.patch("/api/admin/voices/{provider}/{voice_id}")
async def admin_voices_update(
    provider: str, voice_id: str, body: dict, admin: dict = Depends(require_admin)
) -> dict:
    """Update one voice. The merged dict is re-validated as a whole, so the
    result always satisfies the per-provider required-field contract."""
    if provider not in _VOICE_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown provider: {provider!r} (allowed: {sorted(_VOICE_PROVIDERS)})",
        )
    existing = VOICE_STORE.get(provider, voice_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"voice not found: {provider}/{voice_id}")
    merged = {**existing, **{k: v for k, v in body.items() if k not in ("id", "voice_id", "provider")}}
    merged["id"] = voice_id
    attrs = _validate_voice_body(provider, merged)
    await VOICE_STORE.put(provider, voice_id, attrs)
    await VOICE_STORE.rescan()
    try:
        await log_activity(
            actor=admin.get("username"),
            actor_role=admin.get("role"),
            type="voice-update",
            target=f"{provider}/{voice_id}",
            detail=_voice_safe_detail(provider, voice_id, attrs),
        )
    except Exception:  # pragma: no cover
        pass
    return {"voice": {"id": voice_id, "provider": provider, **(VOICE_STORE.get(provider, voice_id) or attrs)}}


@app.delete("/api/admin/voices/{provider}/{voice_id}")
async def admin_voices_delete(
    provider: str, voice_id: str, admin: dict = Depends(require_admin)
) -> dict:
    """Delete one voice. Deleting a voice that a demo/runtime config still
    references is allowed — the resolvers fall back to the provider default, so
    a dangling reference never crashes a call (A.4)."""
    if provider not in _VOICE_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown provider: {provider!r} (allowed: {sorted(_VOICE_PROVIDERS)})",
        )
    if VOICE_STORE.get(provider, voice_id) is None:
        raise HTTPException(status_code=404, detail=f"voice not found: {provider}/{voice_id}")
    await VOICE_STORE.delete(provider, voice_id)
    await VOICE_STORE.rescan()
    try:
        await log_activity(
            actor=admin.get("username"),
            actor_role=admin.get("role"),
            type="voice-delete",
            target=f"{provider}/{voice_id}",
            detail=_voice_safe_detail(provider, voice_id, None),
        )
    except Exception:  # pragma: no cover
        pass
    return {"deleted": f"{provider}/{voice_id}"}


# --- Activity audit log API (admin-only) ---------------------------------
# GET /api/admin/activity → paginated, newest-first operations audit rows
# (Part B B.5). Reads through ACTIVITY_STORE.query, which degrades to an empty
# result when the ActivityTable is absent (zero-impact ship). All detail maps
# were already sanitized (default-deny) at write time.
@app.get("/api/admin/activity")
async def admin_activity_list(
    from_: str | None = Query(default=None, alias="from"),
    to: str | None = Query(default=None, alias="to"),
    actor: str | None = None,
    type: str | None = None,
    limit: int = 100,
    cursor: str | None = None,
    _: dict = Depends(require_admin),
) -> dict:
    """List audit rows, newest-first, filterable by ``from``/``to`` (UTC
    YYYY-MM-DD), ``actor``, ``type``. Paginate via the opaque ``cursor`` echoed
    back in the response. Table absent → ``{"items": [], "cursor": None}``."""
    result = await ACTIVITY_STORE.query(
        day_from=from_, day_to=to, actor=actor, type=type, limit=limit, cursor=cursor,
    )
    return {"items": result["items"], "cursor": result["cursor"]}


# NB: GET / is handled by the demo SPA mount near the bottom of this file.
# That mount must come *after* every @app.* endpoint definition, otherwise
# StaticFiles shadowing would 404 our /api/* and WS routes.


@app.get("/api/ws-token")
async def ws_token(user: dict = Depends(require_user)) -> dict:
    """Short-TTL (60s) token binding a browser WebSocket to the logged-in user.

    The WS handshake can't reliably carry the HttpOnly cookie through
    CloudFront, so the SPA fetches this token (cookie-authenticated) and passes
    it as ?token=… on /ws — the WS handler decodes the bound username for
    history attribution. require_user means an anonymous caller never gets a
    token (the old `if not SITE_PASSWORD: return {"token": ""}` short-circuit is
    gone — a logged-in user is now always required)."""
    return {
        "token": _issue_ws_token(user["username"], role=user.get("role") or "user"),
        "ttl": int(_WS_TOKEN_TTL),
    }


@app.get("/api/config")
async def config(_: dict = Depends(require_user)) -> dict:
    web = RUNTIME_CONFIG.get_web_defaults()
    return {
        "languages": [
            {"id": k, "label": v["label"], "engines": list(v.get("engines", []))}
            for k, v in LANGUAGES.items()
        ],
        "models": [{"id": k, "label": k, "bedrock_id": v} for k, v in MODELS.items()],
        "providers": [{"id": k, "label": v} for k, v in TTS_PROVIDERS.items()],
        "minimax_models": [{"id": k, "label": v} for k, v in MINIMAX_MODELS.items()],
        "default_minimax_model": web.get("minimax_model", DEFAULT_MINIMAX_MODEL),
        "engines": [{"id": k, "label": v} for k, v in ENGINES.items()],
        "demos": _demo_options(),
        "scenarios": _scenario_options(),
        "default_demo": web.get("scenario", DEFAULT_SCENARIO),
        "default_scenario": web.get("scenario", DEFAULT_SCENARIO),
        "voices_by_provider": {
            "minimax": [
                {"id": k, "label": _voice_display(v), "language": v.get("language")}
                for k, v in voices_for("minimax").items()
            ],
            "polly": [
                {"id": k, "label": _voice_display(v), "language": v.get("language")}
                for k, v in voices_for("polly").items()
            ],
        },
        "nova_sonic_voices": [
            {
                "id": k,
                "label": v["label"],
                "gender": v.get("gender"),
                "locale": v.get("locale"),
                "lang_label": v.get("lang_label"),
                "polyglot": bool(v.get("polyglot")),
            }
            for k, v in voices_for("nova-sonic").items()
        ],
        "default_language": web.get("lang", DEFAULT_LANG),
        "default_model": web.get("model", DEFAULT_MODEL),
        "default_provider": web.get("provider", DEFAULT_PROVIDER),
        "default_engine": web.get("engine", DEFAULT_ENGINE),
        "default_nova_sonic_voice": DEFAULT_NOVA_SONIC_VOICE,
        "default_voices": {
            "minimax": web.get("voice", DEFAULT_MINIMAX_VOICE),
            "polly":   DEFAULT_POLLY_VOICE,
        },
        "default_prompts": {
            k: {
                "system": v["prompt"],
                "greeting": v["greeting"],
                "summary": SUMMARY_PROMPTS.get(k, ""),
            }
            for k, v in LANGUAGES.items()
        },
    }


# JSON-only system prompt appended to the language-specific summary prompt.
# The structured schema is what /api/history persists into the DDB row's
# `summary` Map; the schema is identical across languages so callers (and the
# UI) can rely on the same field names.
_SUMMARY_JSON_INSTRUCTION = (
    "\n\nRespond with a single JSON object, no prose, no markdown fences. "
    "Schema: {"
    '"intent": string (one sentence summarising why the caller called), '
    '"key_questions": array of strings (questions the caller asked, may be empty), '
    '"action_items": array of strings (commitments or next steps, may be empty), '
    '"sentiment": one of \"neutral\"|\"positive\"|\"negative\"|\"mixed\"'
    "}."
)


def _coerce_summary_struct(raw_text: str) -> dict:
    """Parse Bedrock output into the canonical summary dict.

    Tolerates ``` fences, leading/trailing prose, and complete parse failures
    by returning a degraded dict so the persisted row never blocks on a
    cosmetically broken model response.
    """
    s = (raw_text or "").strip()
    if s.startswith("```"):
        # strip ``` fences (optionally with a "json" tag).
        s = s.lstrip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    try:
        data = json.loads(s)
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
    except Exception:
        return {
            "intent": (raw_text or "")[:500],
            "key_questions": [],
            "action_items": [],
            "sentiment": "neutral",
        }
    return {
        "intent": str(data.get("intent") or "")[:1000],
        "key_questions": [str(x) for x in (data.get("key_questions") or []) if x],
        "action_items": [str(x) for x in (data.get("action_items") or []) if x],
        "sentiment": str(data.get("sentiment") or "neutral"),
    }


def _humanize_summary_struct(struct: dict, lang_key: str) -> str:
    """Render the structured summary as a multi-line bullet string for
    backwards compatibility with the existing /api/summary frontend."""
    is_zh = lang_key.startswith("zh")
    is_ja = lang_key == "ja-JP"
    if is_zh:
        labels = ("意图", "关键问题", "后续事项", "情绪")
    elif is_ja:
        labels = ("意図", "重要な質問", "アクション", "感情")
    else:
        labels = ("Intent", "Key questions", "Action items", "Sentiment")
    lines: list[str] = []
    if struct.get("intent"):
        lines.append(f"- {labels[0]}: {struct['intent']}")
    for q in struct.get("key_questions") or []:
        lines.append(f"- {labels[1]}: {q}")
    for a in struct.get("action_items") or []:
        lines.append(f"- {labels[2]}: {a}")
    if struct.get("sentiment"):
        lines.append(f"- {labels[3]}: {struct['sentiment']}")
    return "\n".join(lines)


async def _invoke_summary_bedrock(
    turns: list[dict],
    lang_key: str,
    *,
    model_key: str = DEFAULT_MODEL,
    system_override: str | None = None,
) -> dict:
    """Run a Bedrock Converse summarization and return the canonical
    summary dict (intent / key_questions / action_items / sentiment / model /
    generated_at). Always returns a dict — never raises for parse errors.
    Underlying Bedrock errors do propagate so callers can decide how to
    surface them (the recorder writes summary_status=failed)."""
    transcript = "\n".join(
        f"{'用戶/User' if t.get('who') == 'user' else '助手/Assistant'}: {t.get('text', '')}"
        for t in turns
        if t.get("text")
    )
    if not transcript.strip():
        return {
            "intent": "",
            "key_questions": [],
            "action_items": [],
            "sentiment": "neutral",
            "model": MODELS.get(model_key) or MODELS[DEFAULT_MODEL],
            "generated_at": int(time.time()),
        }

    base_prompt = (system_override or "").strip() or SUMMARY_PROMPTS.get(
        lang_key, SUMMARY_PROMPTS[DEFAULT_LANG]
    )
    system_instruction = base_prompt + _SUMMARY_JSON_INSTRUCTION
    model_id = MODELS.get(model_key) or MODELS[DEFAULT_MODEL]

    region = os.environ.get("AWS_REGION", "us-east-1")
    frozen = boto3.Session(region_name=region).get_credentials().get_frozen_credentials()
    summary_settings = AWSBedrockLLMService.Settings(
        model=model_id,
        system_instruction=system_instruction,
        max_tokens=512,
    )
    if _is_claude_model_id(model_id):
        summary_settings.enable_prompt_caching = True
    llm = AWSBedrockLLMService(
        aws_region=region,
        aws_access_key=frozen.access_key,
        aws_secret_key=frozen.secret_key,
        aws_session_token=frozen.token,
        settings=summary_settings,
    )
    ctx = LLMContext()
    ctx.add_message({"role": "user", "content": transcript})
    text = await llm.run_inference(ctx)
    struct = _coerce_summary_struct(text or "")
    struct["model"] = model_id
    struct["generated_at"] = int(time.time())
    return struct


# --- Demo localization / one-click translate -------------------------------

_TRANSLATE_SYSTEM_TEMPLATE = (
    "You are a professional localization translator for a VOICE assistant. "
    "Translate the natural-language narration in the provided fields from "
    "{source_label} into {target_label}.\n\n"
    "STRICTLY PRESERVE (do NOT translate, do NOT alter):\n"
    "- English technical identifiers verbatim: verifyCustomer, requestRepair, "
    "woNumber, customerId (and any similar camelCase / snake_case API names).\n"
    "- Enum / tier values verbatim: smart version, premium version, "
    "elite version.\n"
    "- Error codes verbatim: CUSTOMER_NOT_FOUND, IDENTITY_EXPIRED (and any "
    "ALL_CAPS_UNDERSCORE code).\n"
    "- 【】 section-header structure and the bracketed header text format.\n"
    "- Digit-by-digit number reading rules, placeholders, and any formatting "
    "instructions.\n"
    "Translate ONLY the natural-language narration; never change business "
    "logic or structure. The output is a spoken voice-assistant prompt, so the "
    "translation must be colloquial and natural to read aloud.\n\n"
    "Respond with a SINGLE JSON object, no prose and no markdown fences, "
    "mapping each input field name to its translated string: "
    '{{"field": "translated text", ...}}. Include every field you were given.'
)


def _coerce_translate_struct(raw_text: str, expected_fields: list[str]) -> dict:
    """Parse the translator's JSON output robustly. Mirrors
    :func:`_coerce_summary_struct` (strip ``` fence, take first ``{`` … last
    ``}``, ``json.loads``). On failure returns a diagnostic dict
    ``{"_error": ...}`` rather than raising — the endpoint turns that into a
    502 and never writes disk."""
    s = (raw_text or "").strip()
    if s.startswith("```"):
        s = s.lstrip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    # Be tolerant of leading/trailing prose: slice the outermost braces.
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        s = s[first : last + 1]
    try:
        data = json.loads(s)
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
    except Exception as e:
        return {"_error": f"translate parse failed: {type(e).__name__}: {e}"}
    out: dict[str, str] = {}
    for field in expected_fields:
        if field in data and isinstance(data[field], str) and data[field].strip():
            out[field] = data[field]
    if not out:
        return {"_error": "translate output had no usable fields"}
    return out


async def _invoke_translate_bedrock(
    texts: dict[str, str],
    target_lang: str,
    source_lang: str,
    *,
    model_key: str | None = None,
) -> dict:
    """Translate a ``{field: source_text}`` map into ``target_lang`` via a
    one-shot Bedrock Converse call. Mirrors :func:`_invoke_summary_bedrock`
    (same Settings / LLMContext / run_inference shape, InstanceRole creds).

    Returns ``{field: translated_text}`` on success, or ``{"_error": ...}`` on
    a parse failure (never raises for parse problems). Underlying Bedrock
    errors propagate so the caller can surface them."""
    source_label = (LANGUAGES.get(source_lang) or {}).get("label", source_lang)
    target_label = (LANGUAGES.get(target_lang) or {}).get("label", target_lang)
    system_instruction = _TRANSLATE_SYSTEM_TEMPLATE.format(
        source_label=source_label, target_label=target_label
    )
    # Translate defaults to TRANSLATE_MODEL (a strong instruction-follower);
    # an explicit caller model_key still wins. Two-stage fallback keeps it
    # robust if either key is missing from MODELS.
    effective_key = model_key or TRANSLATE_MODEL
    model_id = MODELS.get(effective_key) or MODELS.get(TRANSLATE_MODEL) or MODELS[DEFAULT_MODEL]

    region = os.environ.get("AWS_REGION", "us-east-1")
    frozen = boto3.Session(region_name=region).get_credentials().get_frozen_credentials()
    translate_settings = AWSBedrockLLMService.Settings(
        model=model_id,
        system_instruction=system_instruction,
        max_tokens=4096,
    )
    if _is_claude_model_id(model_id):
        translate_settings.enable_prompt_caching = True
    llm = AWSBedrockLLMService(
        aws_region=region,
        aws_access_key=frozen.access_key,
        aws_secret_key=frozen.secret_key,
        aws_session_token=frozen.token,
        settings=translate_settings,
    )
    ctx = LLMContext()
    # Hand the model a JSON object of the source fields so it can echo back the
    # same keys with translated values.
    ctx.add_message({"role": "user", "content": json.dumps(texts, ensure_ascii=False)})
    text = await llm.run_inference(ctx)
    return _coerce_translate_struct(text or "", list(texts.keys()))


class DemoTranslateBody(BaseModel):
    """Body of POST /api/admin/demos/{id}/translate."""

    target_lang: str
    source_lang: str | None = None
    model: str | None = None


@app.post("/api/admin/demos/{demo_id}/translate")
async def admin_demo_translate(
    demo_id: str, body: DemoTranslateBody, _: dict = Depends(require_admin)
) -> dict:
    """Preview-translate a demo's inline localized fields into
    ``target_lang``. Never writes disk — the confirmed translation is written
    back through PATCH /api/admin/demos/{id} with a ``localized`` body."""
    demo = DEMO_LOADER.get(demo_id)
    if not demo:
        raise HTTPException(status_code=404, detail=f"demo not found: {demo_id}")

    target_lang = body.target_lang
    if target_lang not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"unknown target_lang: {target_lang!r}")

    fields = _demo_localized_fields(demo)
    if not fields:
        raise HTTPException(
            status_code=400,
            detail=f"demo {demo_id} has no localized content to translate",
        )

    # Per-field source selection with fallback. Preference order:
    #   body.source_lang (if given) > zh-CN > DEFAULT_LANG > any lang the field
    # actually has. We pick PER FIELD, so a field missing the preferred source
    # falls back to one of its own available langs independently.
    preference: list[str] = []
    if body.source_lang:
        if body.source_lang not in LANGUAGES:
            raise HTTPException(
                status_code=400, detail=f"unknown source_lang: {body.source_lang!r}"
            )
        preference.append(body.source_lang)
    for cand in ("zh-CN", DEFAULT_LANG):
        if cand not in preference:
            preference.append(cand)

    texts: dict[str, str] = {}
    source_used: dict[str, str] = {}
    for field in fields:
        lang_map = demo.get(field) or {}
        chosen_lang = None
        for cand in preference:
            val = lang_map.get(cand)
            if isinstance(val, str) and val.strip():
                chosen_lang = cand
                break
        if chosen_lang is None:
            # Fall back to any lang this field has non-empty text for. Iterate
            # in LANGUAGES order for determinism, then any remaining keys.
            ordered = [k for k in LANGUAGES if k in lang_map] + [
                k for k in lang_map if k not in LANGUAGES
            ]
            for cand in ordered:
                val = lang_map.get(cand)
                if isinstance(val, str) and val.strip():
                    chosen_lang = cand
                    break
        if chosen_lang is None:
            continue  # no usable source text for this field; skip it
        texts[field] = lang_map[chosen_lang]
        source_used[field] = chosen_lang

    if not texts:
        raise HTTPException(
            status_code=400,
            detail=f"demo {demo_id} has no usable source text to translate",
        )

    # body.model is optional; None lets _invoke_translate_bedrock apply
    # TRANSLATE_MODEL (nova-2-lite is too weak for prompt translation).
    try:
        result = await _invoke_translate_bedrock(
            texts, target_lang, body.source_lang or source_used.get(fields[0], DEFAULT_LANG),
            model_key=body.model,
        )
    except Exception as e:
        logger.exception("translate failed")
        raise HTTPException(
            status_code=502, detail=f"translation failed: {type(e).__name__}: {e}"
        )

    if isinstance(result, dict) and result.get("_error"):
        raise HTTPException(
            status_code=502, detail=f"translation failed: {result['_error']}"
        )

    # already_exists: which translated fields already have target_lang on disk
    # (so the SPA can warn the admin that writing back needs overwrite=true).
    already_exists = {
        field: (
            isinstance(demo.get(field), dict)
            and isinstance(demo[field].get(target_lang), str)
            and bool(demo[field][target_lang].strip())
        )
        for field in result
    }

    return {
        "target_lang": target_lang,
        "source_used": source_used,
        "fields": result,
        "already_exists": already_exists,
    }


@app.post("/api/summary")
async def summarize(payload: dict, _: dict = Depends(require_user)) -> dict:
    """One-shot conversation summary via Bedrock Converse.

    Expects JSON:
        {
            "language": "zh-HK" | "zh-CN" | "en-US" | "ja-JP",
            "model":    "<MODELS key>",          # optional
            "turns":    [{"who": "user"|"bot", "text": "..."}]
        }

    Returns both a structured `summary_struct` (intent / key_questions /
    action_items / sentiment / model / generated_at) and a human-readable
    `summary` string for backwards compatibility with the existing frontend.
    """
    lang_key = payload.get("language", DEFAULT_LANG)
    model_key = payload.get("model", DEFAULT_MODEL)
    turns = payload.get("turns") or []

    if not turns:
        return {"error": "no turns to summarize"}

    transcript_has_text = any(t.get("text") for t in turns)
    if not transcript_has_text:
        return {"error": "empty transcript"}

    override = payload.get("system_prompt")
    try:
        struct = await _invoke_summary_bedrock(
            turns, lang_key, model_key=model_key, system_override=override
        )
    except Exception as e:
        logger.exception("summary failed")
        return {"error": f"{type(e).__name__}: {e}"}

    return {
        "summary": _humanize_summary_struct(struct, lang_key),
        "summary_struct": struct,
        "language": lang_key,
        "model": struct.get("model"),
        "turns": len(turns),
    }


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    # Browser web calls must present a user-bound ?token=… minted by
    # GET /api/ws-token (which itself requires a logged-in user). The bound
    # username is recorded for history attribution (web_user).
    web_user = _ws_token_username(websocket)
    if not web_user:
        await websocket.close(code=1008, reason="unauthorized")
        return
    await websocket.accept()
    # Per-call snapshot of runtime web defaults; in-flight calls keep this view.
    web_defaults = RUNTIME_CONFIG.get_web_defaults()
    qp = websocket.query_params
    engine   = qp.get("engine",   web_defaults.get("engine",   DEFAULT_ENGINE))
    lang     = qp.get("lang",     web_defaults.get("lang",     DEFAULT_LANG))
    scenario = qp.get("scenario", web_defaults.get("scenario", DEFAULT_SCENARIO))
    valid_scenarios = {DEFAULT_DEMO_ID} | {d["id"] for d in DEMO_LOADER.list()}
    if scenario not in valid_scenarios:
        scenario = web_defaults.get("scenario", DEFAULT_SCENARIO)
    system_override   = qp.get("system")
    greeting_override = qp.get("greeting")

    # Web calls register in ACTIVE_SESSIONS so dashboard active_calls counts
    # them, and attach to HistoryRecorder so they show up in /admin/history.
    call_id = qp.get("call_id") or f"web-{int(time.time() * 1000)}"
    caller = qp.get("caller") or "web"
    primary = _ws_emit(websocket)
    sess = await session_register(call_id, caller=caller, primary_emit=primary)
    fan_emit = _multi_emit(lambda: session_emits(call_id))

    if engine == "nova-sonic":
        voice = qp.get("voice", DEFAULT_NOVA_SONIC_VOICE)
        # Alias migration BEFORE the registry membership check (N1).
        voice = NOVA_SONIC_VOICE_ALIASES.get(voice, voice)
        if voice not in voices_for("nova-sonic"):
            voice = DEFAULT_NOVA_SONIC_VOICE
        attach_meta = {
            "caller": caller,
            "started_at": sess.get("started", time.time()),
            "engine": engine,
            "lang": lang,
            "scenario": scenario,
            "provider": "",
            "voice": voice,
            "model": "",
            "minimax_model": "",
            "web_user": web_user,
        }
    else:
        model    = qp.get("model",    web_defaults.get("model",    DEFAULT_MODEL))
        provider = qp.get("provider", web_defaults.get("provider", DEFAULT_PROVIDER))
        default_voice = (
            web_defaults.get("voice")
            if web_defaults.get("voice")
            else (DEFAULT_POLLY_VOICE if provider == "polly" else DEFAULT_MINIMAX_VOICE)
        )
        voice    = qp.get("voice", default_voice)
        minimax_model = qp.get("minimax_model", web_defaults.get("minimax_model", DEFAULT_MINIMAX_MODEL))
        attach_meta = {
            "caller": caller,
            "started_at": sess.get("started", time.time()),
            "engine": engine,
            "lang": lang,
            "scenario": scenario,
            "provider": provider,
            "voice": voice,
            "model": model,
            "minimax_model": minimax_model,
            "web_user": web_user,
        }

    if _history is not None:
        try:
            _history.attach(call_id, attach_meta)
        except Exception as e:
            logger.warning(f"web history attach failed: {e}")

    # Audit: web call start. actor = the ws-token-bound web user; target =
    # call_id; detail = non-secret runtime descriptors (engine/lang/scenario/
    # voice). Best-effort — never blocks the call.
    await log_activity(
        actor=web_user, type="call-start", target=call_id,
        detail={"engine": engine, "lang": lang, "scenario": scenario, "voice": voice},
    )

    if engine == "nova-sonic":
        task = await _build_nova_sonic_pipeline(
            websocket, lang, voice, system_override, greeting_override,
            scenario_key=scenario,
            event_emit=fan_emit,
            history_recorder=_history,
            call_id=call_id,
            is_phone=False,
        )
    else:
        task = await _build_pipeline(
            websocket, lang, model, voice, provider,
            system_override=system_override,
            greeting_override=greeting_override,
            scenario_key=scenario,
            minimax_model=minimax_model,
            event_emit=fan_emit,
            history_recorder=_history,
            call_id=call_id,
            is_phone=False,
        )
    try:
        await PipelineRunner(handle_sigint=False).run(task)
    finally:
        session = getattr(task, "_aio_session", None)
        if session and not session.closed:
            await session.close()
        for c in getattr(task, "_mcp_clients", None) or []:
            try:
                await c.close()
            except Exception:
                pass
        await session_unregister(call_id)
        if _history is not None:
            try:
                asyncio.create_task(_history.flush_turns_and_summarize(call_id))
            except Exception as e:
                logger.warning(f"web history flush schedule failed: {e}")
        await log_activity(
            actor=web_user, type="call-end", target=call_id,
            detail={"engine": engine, "lang": lang, "scenario": scenario, "voice": voice},
        )
        logger.info(f"web WS finished: call_id={call_id}")


# --- Isolated ASR multi-language A/B test harness ------------------------
# /asr-test/ws is a measurement-only endpoint, COMPLETELY SEPARATE from /ws.
# It fans one mic's PCM into TWO standalone AWS Transcribe streams (no Pipeline,
# no VAD/LLM/TTS) so the user can compare single-language zh-HK (A) vs
# multi-language code-switching zh-HK+en-US (B) on identical audio, BEFORE any
# prod call path changes. Nothing is persisted; no transcripts are logged.
# A single git revert removes this block + the frontend page cleanly.

# The pipecat version our multi-language seam-patch (below) is validated
# against. CFN UserData pins pipecat-ai to this exact version. If the installed
# version drifts, stream B degrades to safe single-language (see
# _MultiLangTranscribeSTT._connect_websocket).
PINNED_PIPECAT = "1.4.0"


class _MultiLangTranscribeSTT(AWSTranscribeSTTService):
    """Stock-pipecat-compatible multi-language Transcribe STT for stream B.

    Stock ``AWSTranscribeSTTService`` (PyPI ``pipecat-ai``) only supports a
    single fixed ``language``; its ``Settings`` has NO identify-* fields. To get
    multi-language identification WITHOUT editing the installed pipecat, we:

    1. Construct stock with a dummy ``Settings(language=Language.ZH_HK)`` so the
       stock ctor + connect path are satisfied (we never pass an identify field
       to stock Settings — it would raise on stock).
    2. Override ``_connect_websocket`` to monkeypatch the MODULE-LEVEL name
       ``pipecat.services.aws.stt.get_presigned_url`` (stt.py imports it into its
       namespace and calls it bare) with a wrapper that swaps in our first-party
       ``build_transcribe_presigned_url`` (adding identify-* + language-options +
       preferred-language). We then call ``super()._connect_websocket()`` so the
       INSTALLED version's connect + receive tail runs unchanged — we couple only
       to the stable ``get_presigned_url(...)`` call-site, never the private tail.
       The patch is restored in ``finally``.

    Version guard: if the installed ``pipecat-ai`` != ``PINNED_PIPECAT`` we log
    loudly and fall back to plain single-language zh-HK (stock ``super()``),
    rather than patch a possibly-changed signature. Safe degrade, never silently
    wrong.
    """

    def __init__(
        self,
        *,
        language_options,
        preferred_language=None,
        mode="multiple",
        partial_results_stability=None,
        **kwargs,
    ):
        super().__init__(
            settings=AWSTranscribeSTTService.Settings(language=Language.ZH_HK), **kwargs
        )
        self._ml = {
            "language_options": language_options,
            "preferred_language": preferred_language,
            "mode": mode,
            # None → keep whatever stock passes at the call-site (its "high"); a
            # "low"/"medium"/"high" string overrides it for latency tuning.
            "partial_results_stability": partial_results_stability,
        }

    async def _connect_websocket(self):
        import importlib.metadata
        import pipecat.services.aws.stt as _stt

        ver = importlib.metadata.version("pipecat-ai")
        if ver != PINNED_PIPECAT:
            logger.warning(
                f"[asr-test] pipecat-ai {ver} != pinned {PINNED_PIPECAT}; stream B "
                f"falls back to single-language zh-HK (multi-language seam not applied)"
            )
            return await super()._connect_websocket()

        ml = self._ml

        def _patched_get_presigned_url(**kw):
            # Stock stt.py calls get_presigned_url(region=, credentials=,
            # language_code=, media_encoding=, sample_rate=, number_of_channels=,
            # enable_partial_results_stabilization=, partial_results_stability=,
            # show_speaker_label=, enable_channel_identification=). Our signer
            # takes a subset; drop language-code (identify-* mode omits it) and
            # the params our signer doesn't model (all at their inert defaults
            # in this path: single channel, no speaker label, no channel id).
            kw.pop("language_code", None)
            for drop in ("number_of_channels", "show_speaker_label",
                         "enable_channel_identification"):
                kw.pop(drop, None)
            # Latency tuning: override stock's hardcoded partial-results-stability
            # ("high") when the endpoint requested a specific level. Lower =
            # partials surface earlier. None → leave stock's value untouched.
            if ml["partial_results_stability"] is not None:
                kw["partial_results_stability"] = ml["partial_results_stability"]
            return build_transcribe_presigned_url(
                **kw,
                identify_multiple_languages=(ml["mode"] != "single"),
                identify_language=(ml["mode"] == "single"),
                language_options=ml["language_options"],
                preferred_language=ml["preferred_language"],
            )

        orig = _stt.get_presigned_url
        _stt.get_presigned_url = _patched_get_presigned_url
        try:
            await super()._connect_websocket()
        finally:
            _stt.get_presigned_url = orig


class _AsrTestSink(FrameProcessor):
    """Downstream sink that captures an STT stream's transcript frames.

    Each AWSTranscribeSTTService egresses TranscriptionFrame (final) and
    InterimTranscriptionFrame (partial) DOWNSTREAM. We link one of these sinks
    after each service and forward {stream, text, is_final, lang?} to a queue
    that the WS writer drains. We never log the transcript text.
    """

    def __init__(self, stream_label: str, out_queue: "asyncio.Queue"):
        super().__init__()
        self._stream_label = stream_label
        self._out_queue = out_queue

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        try:
            if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
                lang = getattr(frame, "language", None)
                # In identify-* mode AWS returns the detected LanguageCode in the
                # raw result; surface it as the per-message lang when present.
                result = getattr(frame, "result", None)
                if isinstance(result, dict) and result.get("LanguageCode"):
                    lang = result.get("LanguageCode")
                self._out_queue.put_nowait(
                    {
                        "stream": self._stream_label,
                        "text": frame.text,
                        "is_final": isinstance(frame, TranscriptionFrame),
                        "lang": str(lang) if lang is not None else None,
                    }
                )
        except Exception:
            # A capture hiccup must never break the stream or the endpoint.
            pass
        await self.push_frame(frame, direction)


class _AsrTestErrorCatcher(FrameProcessor):
    """Upstream catcher wired as the STT service's ``_prev``.

    AWS Transcribe pushes ``ErrorFrame`` UPSTREAM (toward the source) on
    connect/auth/protocol failure. By sitting upstream of the STT we intercept
    those errors and surface them in the stream's own pane as
    ``{stream, error}`` — keeping one stream's failure isolated to its pane and
    never crashing the other stream or the endpoint.
    """

    def __init__(self, stream_label: str, out_queue: "asyncio.Queue"):
        super().__init__()
        self._stream_label = stream_label
        self._out_queue = out_queue

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        try:
            if isinstance(frame, ErrorFrame):
                self._out_queue.put_nowait(
                    {"stream": self._stream_label, "error": str(frame.error)}
                )
        except Exception:
            pass
        await self.push_frame(frame, direction)


class _AsrTestStream:
    """One isolated STT stream (A or B) driven inside a real minimal Pipeline.

    Wraps construction / start / per-frame feed / teardown so that an error in
    this stream is reported in its own pane and never crosses over to the other
    stream or crashes the endpoint.

    The STT service runs inside Pipeline([error_catcher, stt, sink]) +
    PipelineTask + PipelineRunner — the exact pattern bot.py's real call paths
    use — so pipecat constructs the FrameProcessorSetup itself with whatever
    arguments the INSTALLED pipecat version requires. This deliberately avoids
    hand-wiring FrameProcessorSetup, whose signature differs between the local
    editable pipecat and prod's stock 1.4.0 (the bug that crashed this endpoint
    on connect).
    """

    def __init__(
        self,
        label: str,
        stt: "AWSTranscribeSTTService",
        *,
        out_queue: "asyncio.Queue",
    ):
        self.label = label
        self._out_queue = out_queue
        self._failed = False
        self._task: "PipelineTask | None" = None
        self._runner_task: "asyncio.Task | None" = None
        # The STT service is constructed by the caller so stream A can be a plain
        # single-language AWSTranscribeSTTService and stream B can be the
        # multi-language _MultiLangTranscribeSTT subclass.
        self.stt = stt
        self.sink = _AsrTestSink(label, out_queue)
        self.error_catcher = _AsrTestErrorCatcher(label, out_queue)

    async def _report_error(self, msg: str):
        self._failed = True
        try:
            self._out_queue.put_nowait({"stream": self.label, "error": msg})
        except Exception:
            pass

    async def _await_connected(self, timeout: float = 5.0):
        """Poll until the STT service's underlying websocket is connected.

        The connect happens as the StartFrame propagates inside the runner, so
        there is no awaitable handle to it — poll the service's `_websocket`
        attribute (and its OPEN state if exposed) until present or timeout. Used
        to synchronize the A-then-B ordering: A must be fully connected (signed
        with the unpatched global signer) before B opens its monkeypatch window.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ws = getattr(self.stt, "_websocket", None)
            if ws is not None:
                state = getattr(ws, "state", None)
                if state is None or getattr(state, "name", str(state)).upper().endswith("OPEN"):
                    return
            await asyncio.sleep(0.02)

    async def start(self):
        try:
            # error_catcher upstream of stt so upstream-pushed ErrorFrames still
            # reach it; sink downstream for Transcription/InterimTranscription.
            pipeline = Pipeline([self.error_catcher, self.stt, self.sink])
            self._task = PipelineTask(
                pipeline,
                params=PipelineParams(audio_in_sample_rate=INPUT_SAMPLE_RATE),
                # This is an eval stream fed only when the user speaks; don't let
                # pipecat's idle-timeout cancel it during silence.
                cancel_on_idle_timeout=False,
            )
            # The runner drives StartFrame propagation (which triggers the
            # service's websocket connect). Run it in the background; we then
            # wait for the connect so callers can sequence A before B.
            self._runner_task = asyncio.create_task(
                PipelineRunner(handle_sigint=False).run(self._task)
            )
            await self._await_connected()
        except Exception as e:
            await self._report_error(f"start failed: {e}")

    async def feed(self, pcm: bytes):
        if self._failed or self._task is None:
            return
        try:
            # Queue the same frame transport.input would push; the pipeline
            # drives stt.run_stt itself — we never call run_stt manually.
            await self._task.queue_frames(
                [
                    InputAudioRawFrame(
                        audio=pcm,
                        sample_rate=INPUT_SAMPLE_RATE,
                        num_channels=1,
                    )
                ]
            )
        except Exception as e:
            await self._report_error(f"audio error: {e}")

    async def stop(self):
        # Cancel the PipelineTask (stock teardown — propagates EndFrame/cleanup),
        # then make sure the background runner coroutine is finished.
        try:
            if self._task is not None:
                await self._task.cancel()
        except Exception:
            pass
        try:
            if self._runner_task is not None:
                self._runner_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._runner_task
        except Exception:
            pass


@app.websocket("/asr-test/ws")
async def asr_test_ws_endpoint(websocket: WebSocket) -> None:
    """Token-gated A/B Transcribe harness. Binary in (16 kHz PCM16 mono),
    JSON out ({stream,text,is_final,lang} | {stream,error}). Separate from /ws;
    no Pipeline / VAD / LLM / TTS; nothing persisted or logged."""
    # Same user-bound ws-token gate as /ws (minted by GET /api/ws-token, which
    # requires a logged-in user). Missing/invalid → close 1008.
    if not _ws_token_username(websocket):
        await websocket.close(code=1008, reason="unauthorized")
        return
    await websocket.accept()

    # ?mode=single → B uses identify-language (single dominant); default/multiple
    # → B uses identify-multiple-languages (code-switching).
    mode = websocket.query_params.get("mode", "multiple")
    # ?stability=low|medium|high → B's partial-results-stability. Lower = partials
    # surface earlier (lower latency) at the cost of more partial rewrites; "high"
    # (AWS + stock-pipecat default) is the most stable but slowest. Only stream B
    # is tunable; A always mirrors stock's hardcoded "high". Unknown value → high.
    stability = websocket.query_params.get("stability", "high")
    if stability not in ("low", "medium", "high"):
        stability = "high"

    region = os.environ.get("AWS_REGION", "us-east-1")
    frozen = boto3.Session(region_name=region).get_credentials().get_frozen_credentials()

    out_queue: "asyncio.Queue" = asyncio.Queue()

    # Stream A: plain stock single-language zh-HK — today's exact prod config.
    stt_a = AWSTranscribeSTTService(
        region=region,
        aws_access_key_id=frozen.access_key,
        api_key=frozen.secret_key,
        aws_session_token=frozen.token,
        sample_rate=INPUT_SAMPLE_RATE,
        settings=AWSTranscribeSTTService.Settings(language=Language.ZH_HK),
    )
    # Stream B: multi-language via the first-party signer seam-patch. mode=single
    # → identify-language (single dominant); default → identify-multiple-languages
    # (code-switching). Passes only language= to the stock Settings ctor.
    stt_b = _MultiLangTranscribeSTT(
        region=region,
        aws_access_key_id=frozen.access_key,
        api_key=frozen.secret_key,
        aws_session_token=frozen.token,
        sample_rate=INPUT_SAMPLE_RATE,
        language_options="zh-HK,en-US",
        preferred_language="zh-HK",
        mode=("single" if mode == "single" else "multiple"),
        partial_results_stability=stability,
    )

    stream_a = _AsrTestStream("A", stt_a, out_queue=out_queue)
    stream_b = _AsrTestStream("B", stt_b, out_queue=out_queue)

    # CRITICAL ORDERING: connect A FULLY (await its start) BEFORE starting B.
    # B's _connect_websocket monkeypatches the PROCESS-GLOBAL module function
    # pipecat.services.aws.stt.get_presigned_url for the duration of its own
    # super()._connect_websocket(); stream A uses that SAME module function. If A
    # connected inside B's patch window it would sign with B's multi-language
    # wrapper instead of plain single-language zh-HK. Sequencing A fully first
    # (its connect completes here) guarantees A signs single-language and B's
    # tiny, finally-restored patch window cannot overlap A. Best-effort: a failed
    # stream reports into its own pane, the other keeps going.
    await stream_a.start()
    await stream_b.start()

    async def _writer():
        """Drain captured transcript/error messages to the client."""
        while True:
            msg = await out_queue.get()
            if websocket.application_state is not WebSocketState.CONNECTED:
                break
            try:
                await websocket.send_text(json.dumps(msg, ensure_ascii=False))
            except Exception:
                break

    writer_task = asyncio.create_task(_writer())

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            pcm = message.get("bytes")
            if pcm is None:
                # Ignore non-binary frames (e.g. stray text/keepalive).
                continue
            # Fan identical audio to BOTH streams; isolation is inside feed().
            await asyncio.gather(stream_a.feed(pcm), stream_b.feed(pcm))
    except Exception:
        # Disconnect / receive errors end the session quietly.
        pass
    finally:
        await stream_a.stop()
        await stream_b.stop()
        writer_task.cancel()
        try:
            await writer_task
        except (asyncio.CancelledError, Exception):
            pass
        try:
            if websocket.application_state is WebSocketState.CONNECTED:
                await websocket.close()
        except Exception:
            pass


# --- Inbound phone (PSTN via Chime SDK Voice Connector) -----------------
# voice-server/ (Node) terminates the SIP/RTP leg from Chime, decodes the
# G.711 μ-law to 16 kHz PCM, and connects here as a plain WebSocket binary
# stream — same wire format as /ws but with no SITE_PASSWORD gating (Chime
# can't send Basic Auth).
#
# Defaults are now read per-call from RUNTIME_CONFIG.get_phone_defaults() at
# /phone/ws entry. The previous PHONE_* module-level constants are obsolete —
# the corresponding env vars are still consulted *once at import time* via
# _RUNTIME_FALLBACK above, so they keep working as bootstrap defaults for the
# initial seed of config/runtime.json. Edits made via Admin UI win after that.


@app.get("/api/calls")
async def list_calls(_: dict = Depends(require_admin)) -> dict:
    """List currently active phone sessions, for the Monitor UI.

    Admin-only: monitoring others' live calls is an admin action (NOTE-1)."""
    snapshot = []
    for cid, sess in ACTIVE_SESSIONS.items():
        snapshot.append({
            "call_id": cid,
            "caller": sess.get("caller"),
            "started": sess.get("started"),
            "monitors": len(sess.get("monitors", [])),
        })
    return {"calls": snapshot}


# --- Call history HTTP API (DynamoDB-backed) -----------------------------
# These endpoints expose the rows that HistoryRecorder writes after each
# phone call. HISTORY_DISABLED ⇒ they short-circuit to empty results without
# constructing any boto3 client.
#
# Pagination encoding: cursor is base64(JSON(LastEvaluatedKey)). The Python
# bytes/Decimal values that DDB returns aren't JSON-serializable as-is, so
# we coerce Decimal→int/float and drop binary types before encoding.
import base64 as _b64
from decimal import Decimal as _Decimal


def _decimal_to_native(obj):
    """Recursively walk DDB output and turn Decimal into int / float so the
    payload is plain-JSON serializable."""
    if isinstance(obj, _Decimal):
        # ints round-trip exactly, floats may lose precision (acceptable for
        # ttl/turn_count style fields used here).
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    if isinstance(obj, list):
        return [_decimal_to_native(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _decimal_to_native(v) for k, v in obj.items()}
    return obj


def _encode_cursor(last_key: dict | None) -> str | None:
    if not last_key:
        return None
    payload = json.dumps(_decimal_to_native(last_key), ensure_ascii=False)
    return _b64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")


def _decode_cursor(cursor: str | None) -> dict | None:
    if not cursor:
        return None
    try:
        payload = _b64.urlsafe_b64decode(cursor.encode("ascii")).decode("utf-8")
        return json.loads(payload)
    except Exception as e:
        logger.warning(f"invalid history cursor: {e}")
        raise HTTPException(status_code=400, detail="invalid cursor")


_LIST_FIELDS = (
    "call_id", "caller", "started_at", "ended_at", "duration_s",
    "turn_count", "summary_status",
)


def _project_card(item: dict) -> dict:
    """Project a full DDB row down to the list-card shape."""
    out = {k: item.get(k) for k in _LIST_FIELDS}
    summary = item.get("summary") or {}
    if isinstance(summary, dict) and summary.get("intent"):
        out["intent"] = summary["intent"]
    return out


def _history_table():
    """Return the DDB Table object for history. Caller must have already
    verified `_history is not None` to avoid creating boto3 clients in the
    disabled path."""
    return _history._table  # type: ignore[union-attr]


@app.get("/api/history")
async def list_history(
    limit: int = 50,
    cursor: str | None = None,
    user: dict = Depends(require_user),
) -> dict:
    """List the CALLER'S OWN web calls in started_at desc order.

    Scoped to ``web_user == <logged-in username>`` so a regular user only ever
    sees their own history. (Admins use /api/admin/history for the full view.)

    Implementation note: the table has no partition that lets DynamoDB sort
    rows by started_at globally — call_id is the hash key. So we Scan a few
    pages, sort in-memory, and slice. This is fine while the table is small
    (< ~10k rows). Once the table grows, swap this for a GSI on a constant
    hash + started_at range.
    """
    if limit <= 0 or limit > 200:
        limit = 50
    if HISTORY_DISABLED or _history is None:
        return {"items": [], "next_cursor": None}

    username = user["username"]
    start_key = _decode_cursor(cursor)
    page_limit = min(max(limit * 4, 50), 200)
    target = limit * 4
    aggregated: list[dict] = []
    last_key: dict | None = start_key
    pages = 0
    try:
        table = _history_table()
        while True:
            kwargs = {
                "Limit": page_limit,
                "FilterExpression": "web_user = :u",
                "ExpressionAttributeValues": {":u": username},
            }
            if last_key:
                kwargs["ExclusiveStartKey"] = last_key
            resp = await asyncio.to_thread(table.scan, **kwargs)
            aggregated.extend(resp.get("Items", []))
            last_key = resp.get("LastEvaluatedKey")
            pages += 1
            # Stop once we have at least 3 pages OR enough items, OR the
            # table is fully scanned. Bound at 5 pages so a malicious key
            # can't drive us into an unbounded loop.
            if last_key is None:
                break
            if pages >= 3 and len(aggregated) >= target:
                break
            if pages >= 5:
                break
    except Exception:
        logger.exception("history scan failed")
        raise HTTPException(status_code=500, detail="history backend unavailable")

    aggregated.sort(key=lambda x: int(x.get("started_at") or 0), reverse=True)
    page = aggregated[:limit]
    items = [_decimal_to_native(_project_card(it)) for it in page]
    # If we cut the result short, encode the next ExclusiveStartKey so the
    # caller can continue from the last DDB scan position. Note: this is a
    # cursor over Scan order, NOT over our in-memory sorted view — clients
    # may see slight reorderings around page boundaries, which is acceptable
    # for a low-QPS history list.
    next_cursor = _encode_cursor(last_key) if last_key else None
    return {"items": items, "next_cursor": next_cursor}


@app.get("/api/history/by-caller")
async def list_history_by_caller(
    caller: str,
    limit: int = 50,
    cursor: str | None = None,
    _: dict = Depends(require_admin),
) -> dict:
    """List a single caller's calls, newest first, via the
    `caller-started-index` GSI. The GSI is KEYS_ONLY so we follow the Query
    with a BatchGetItem to fetch the projected fields from the main table.
    """
    if limit <= 0 or limit > 200:
        limit = 50
    if not caller:
        raise HTTPException(status_code=400, detail="caller is required")
    if HISTORY_DISABLED or _history is None:
        return {"items": [], "next_cursor": None}

    start_key = _decode_cursor(cursor)
    try:
        table = _history_table()
        query_kwargs = {
            "IndexName": "caller-started-index",
            "KeyConditionExpression": "caller = :c",
            "ExpressionAttributeValues": {":c": caller},
            "ScanIndexForward": False,
            "Limit": limit,
        }
        if start_key:
            query_kwargs["ExclusiveStartKey"] = start_key
        resp = await asyncio.to_thread(table.query, **query_kwargs)
        keys = [{"call_id": it["call_id"]} for it in resp.get("Items", [])]
        next_cursor = _encode_cursor(resp.get("LastEvaluatedKey"))

        if not keys:
            return {"items": [], "next_cursor": next_cursor}

        # BatchGetItem returns up to 100 items per request — limit ≤ 200 so
        # in the worst case we make two batches.
        ddb = boto3.resource("dynamodb", region_name=os.environ.get("DDB_REGION") or os.environ.get("AWS_REGION", "us-east-1"))

        def _batch_get(batch_keys: list[dict]) -> list[dict]:
            req = {table.name: {"Keys": batch_keys}}
            out: list[dict] = []
            while req:
                bresp = ddb.batch_get_item(RequestItems=req)
                out.extend(bresp.get("Responses", {}).get(table.name, []))
                req = bresp.get("UnprocessedKeys") or None
            return out

        items: list[dict] = []
        for i in range(0, len(keys), 100):
            chunk = keys[i : i + 100]
            items.extend(await asyncio.to_thread(_batch_get, chunk))
    except HTTPException:
        raise
    except Exception:
        logger.exception("history by-caller failed")
        raise HTTPException(status_code=500, detail="history backend unavailable")

    # BatchGetItem does NOT preserve key order — re-sort by started_at desc
    # so the response matches the GSI's order.
    items.sort(key=lambda x: int(x.get("started_at") or 0), reverse=True)
    cards = [_decimal_to_native(_project_card(it)) for it in items]
    return {"items": cards, "next_cursor": next_cursor}


@app.get("/api/history/{call_id}")
async def get_history_item(
    call_id: str,
    user: dict = Depends(require_user),
) -> dict:
    """Return a full call row including transcript turns + summary.

    A non-admin may only fetch rows attributed to them (``web_user`` ==
    their username); anything else 404s so call ids can't be enumerated."""
    if HISTORY_DISABLED or _history is None:
        raise HTTPException(status_code=404, detail="not found")
    try:
        table = _history_table()
        resp = await asyncio.to_thread(table.get_item, Key={"call_id": call_id})
    except Exception:
        logger.exception("history get_item failed")
        raise HTTPException(status_code=500, detail="history backend unavailable")
    item = resp.get("Item")
    if not item:
        raise HTTPException(status_code=404, detail="not found")
    if user.get("role") != "admin" and item.get("web_user") != user["username"]:
        raise HTTPException(status_code=404, detail="not found")
    return _decimal_to_native(item)


# --- Admin metrics aggregation -------------------------------------------
# Builds the dashboard payload consumed by /api/admin/metrics (added in T2).
# Pure function; no FastAPI route here so it stays unit-testable without
# spinning up a TestClient. Cached for ``_METRICS_CACHE_TTL`` seconds so a
# 5-10s front-end poll loop can't fan out into per-poll DDB scans.

_METRICS_CACHE_TTL = 10.0  # seconds
_METRICS_CACHE: dict = {"ts": 0.0, "value": None}
_METRICS_LOCK = asyncio.Lock()

# Outcome buckets we always emit (with zero) so the front-end has stable keys.
_METRICS_OUTCOME_KEYS = (
    "user_requested",
    "task_completed",
    "transferred",
    "timeout",
    "error",
    "unknown",
)


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Return the ``pct`` (0-100) percentile via nearest-rank on a list that's
    already sorted ascending. Empty list ⇒ 0 (matches the empty-table contract
    in AC #4)."""
    if not sorted_values:
        return 0
    if pct <= 0:
        return sorted_values[0]
    if pct >= 100:
        return sorted_values[-1]
    # Nearest-rank: ceil(pct/100 * N) - 1 (0-indexed). Avoids the off-by-one
    # that "len * pct / 100" hits at pct=50 with even N.
    import math
    rank = math.ceil(pct / 100.0 * len(sorted_values)) - 1
    rank = max(0, min(rank, len(sorted_values) - 1))
    return sorted_values[rank]


def _peak_concurrent(rows: list[dict]) -> int:
    """Sweep-line peak concurrent calls.

    Each row contributes ``(started_at, +1)`` and ``(ended_at, -1)``. Sorting
    by (timestamp, delta) — with -1 before +1 at the same instant — means a
    call that ends and another that starts in the same second don't double-
    count, matching how DDB second-resolution timestamps land in practice.
    """
    events: list[tuple[int, int]] = []
    for r in rows:
        s = int(r.get("started_at") or 0)
        e = int(r.get("ended_at") or 0)
        if e <= 0 or e < s:
            # Row hasn't been finalized yet (no ended_at) — skip; the live
            # session is already covered by ACTIVE_SESSIONS / active_calls.
            continue
        events.append((s, +1))
        events.append((e, -1))
    if not events:
        return 0
    # Sort ascending by time; ties: -1 before +1.
    events.sort(key=lambda x: (x[0], x[1]))
    peak = 0
    cur = 0
    for _, delta in events:
        cur += delta
        if cur > peak:
            peak = cur
    return peak


def _utc_midnight(now_ts: float) -> int:
    """Return the UTC midnight epoch <= now_ts. Pure arithmetic, no datetime
    so the helper is timezone-safe regardless of TZ env vars."""
    return int(now_ts) - (int(now_ts) % 86400)


async def _scan_recent_history(now_ts: float) -> list[dict]:
    """Scan the history table for rows with ``started_at >= now-86400``.

    Pages through ``LastEvaluatedKey`` so we don't miss rows in tables that
    cross the 1 MB scan-page boundary. ProjectionExpression limits columns
    so we never pay the read cost of turns/summary on the metrics path
    (AC #2).
    """
    if HISTORY_DISABLED or _history is None:
        return []
    cutoff = int(now_ts) - 86400
    table = _history_table()
    items: list[dict] = []
    last_key: dict | None = None
    # `started_at`, `outcome`, `engine`, `scenario`, `duration_s`, `ended_at`
    # are reserved or near-reserved in DynamoDB FilterExpression land, so we
    # alias every column through ExpressionAttributeNames to be safe.
    name_aliases = {
        "#cid": "call_id",
        "#sa": "started_at",
        "#ea": "ended_at",
        "#dur": "duration_s",
        "#oc": "outcome",
        "#tr": "transfer_requested",
        "#sc": "scenario",
        "#en": "engine",
    }
    scan_kwargs_base = {
        "FilterExpression": "#sa >= :cutoff",
        "ProjectionExpression": "#cid, #sa, #ea, #dur, #oc, #tr, #sc, #en",
        "ExpressionAttributeNames": name_aliases,
        "ExpressionAttributeValues": {":cutoff": cutoff},
    }
    pages = 0
    while True:
        kwargs = dict(scan_kwargs_base)
        if last_key:
            kwargs["ExclusiveStartKey"] = last_key
        resp = await asyncio.to_thread(table.scan, **kwargs)
        items.extend(resp.get("Items", []))
        last_key = resp.get("LastEvaluatedKey")
        pages += 1
        if last_key is None:
            break
        # Hard cap so a giant table can't stall the dashboard. Real prod
        # table size is ~k rows over 30d so this is comfortably above the
        # working set; if we ever hit it the front-end still gets a value.
        if pages >= 20:
            logger.warning(
                f"_scan_recent_history page cap hit (pages={pages}); "
                f"truncating at {len(items)} rows"
            )
            break
    return [_decimal_to_native(it) for it in items]


def _aggregate_metrics(now_ts: float, rows: list[dict], active_calls: int) -> dict:
    """Pure aggregator — no IO. Split out so unit tests can feed synthetic
    rows without mocking DDB."""
    midnight = _utc_midnight(now_ts)

    # Today (UTC calendar day) duration percentiles.
    today_durations: list[float] = []
    for r in rows:
        if int(r.get("started_at") or 0) >= midnight:
            d = r.get("duration_s")
            if d is None:
                continue
            try:
                today_durations.append(float(d))
            except (TypeError, ValueError):
                continue
    today_durations.sort()
    today_total = len(today_durations)
    today_avg = (sum(today_durations) / today_total) if today_total else 0
    today_p50 = _percentile(today_durations, 50)
    today_p95 = _percentile(today_durations, 95)

    # 24h outcome distribution. Always emit the canonical bucket set so the
    # front-end can render zeros without conditional rendering.
    outcome_24h = {k: 0 for k in _METRICS_OUTCOME_KEYS}
    transferred_count = 0
    for r in rows:
        oc = (r.get("outcome") or "unknown") or "unknown"
        if oc not in outcome_24h:
            outcome_24h[oc] = 0
        outcome_24h[oc] += 1
        if r.get("transfer_requested"):
            transferred_count += 1
    total_24h = len(rows)
    transfer_rate_24h = (transferred_count / total_24h) if total_24h else 0

    # 24h demo + engine distribution. Empty/missing → bucketed as "unknown".
    demo_24h: dict[str, int] = {}
    engine_24h: dict[str, int] = {}
    for r in rows:
        sc = r.get("scenario") or "unknown"
        en = r.get("engine") or "unknown"
        demo_24h[sc] = demo_24h.get(sc, 0) + 1
        engine_24h[en] = engine_24h.get(en, 0) + 1

    return {
        "as_of": int(now_ts),
        "active_calls": active_calls,
        "today": {
            "total": today_total,
            "avg_duration_s": round(today_avg, 2),
            "p50_duration_s": today_p50,
            "p95_duration_s": today_p95,
        },
        "outcome_24h": outcome_24h,
        "transfer_rate_24h": round(transfer_rate_24h, 4),
        "demo_distribution_24h": demo_24h,
        "engine_distribution_24h": engine_24h,
        "peak_concurrent_24h": _peak_concurrent(rows),
    }


async def _collect_metrics() -> dict:
    """Build the admin dashboard payload, with a 10 s in-memory cache.

    The lock serialises the *first* concurrent miss so we don't fan out N
    parallel DDB scans when the front-end first loads. Subsequent misses
    after the TTL just take the lock again — there's no thundering-herd
    risk because the scan finishes well under 10 s on the prod table size.
    """
    now = time.time()
    cached = _METRICS_CACHE.get("value")
    if cached is not None and (now - _METRICS_CACHE.get("ts", 0.0)) < _METRICS_CACHE_TTL:
        # Overlay live active_calls so the dashboard reflects in-flight calls
        # immediately even when the rest of the payload is served from cache.
        return {**cached, "active_calls": len(ACTIVE_SESSIONS), "as_of": int(now)}
    async with _METRICS_LOCK:
        # Re-check after acquiring the lock — another coroutine may have
        # populated the cache while we were waiting.
        now = time.time()
        cached = _METRICS_CACHE.get("value")
        if cached is not None and (now - _METRICS_CACHE.get("ts", 0.0)) < _METRICS_CACHE_TTL:
            return {**cached, "active_calls": len(ACTIVE_SESSIONS), "as_of": int(now)}
        try:
            rows = await _scan_recent_history(now)
        except Exception:
            logger.exception("_collect_metrics scan failed")
            rows = []
        active = len(ACTIVE_SESSIONS)
        result = _aggregate_metrics(now, rows, active)
        _METRICS_CACHE["value"] = result
        _METRICS_CACHE["ts"] = now
        return result


@app.websocket("/phone/ws")
async def phone_ws_endpoint(websocket: WebSocket) -> None:
    """Inbound phone-call audio bridge. Voice-server (Node) connects here.

    Wire protocol: client sends binary PCM frames at the PSTN-native
    8 kHz (PHONE_WIRE_SAMPLE_RATE); the backend adapts per engine in
    _run_phone_session (pipeline feeds 8 kHz straight to Transcribe; nova-sonic
    upsamples to 16 kHz). Server sends binary 24 kHz PCM frames + JSON event
    text frames (downlink unchanged).
    """
    await websocket.accept()
    qp = websocket.query_params
    call_id = qp.get("call_id") or f"phone-{int(time.time() * 1000)}"
    caller = qp.get("caller")
    logger.info(f"phone WS connected: call_id={call_id} caller={caller}")

    primary = _ws_emit(websocket)
    await session_register(call_id, caller=caller, primary_emit=primary)

    await _run_phone_session(websocket, call_id=call_id, caller=caller)


async def _run_phone_session(
    websocket: WebSocket,
    *,
    call_id: str,
    caller: str | None,
    serializer: FrameSerializer | None = None,
) -> None:
    """Shared phone-call session kernel.

    Runs the full inbound-call lifecycle for an already-accepted, already
    session-registered websocket: read per-call runtime config, attach history,
    build the engine-appropriate pipeline (threading ``serializer`` through to
    the builder — ``None`` falls back to ``RawPCMSerializer`` for byte-identical
    Chime/phone behavior), run it, and clean up the session/MCP/history in a
    ``finally`` block.

    Both the Chime ``/phone/ws`` endpoint (serializer=None) and the Twilio
    media-stream endpoint (serializer=TwilioFrameSerializer) call this.
    """
    sess = ACTIVE_SESSIONS.get(call_id) or {}

    fan_emit = _multi_emit(lambda: session_emits(call_id))

    # Uplink wire format (Chime path): voice-server now sends the PSTN audio as
    # NATIVE 8 kHz PCM (it no longer upsamples). We adapt per engine below:
    #   - pipeline (Transcribe): feed native 8 kHz directly (Transcribe has a
    #     telephony/narrowband model) → RawPCMSerializer(8000) + in_sr=8000.
    #   - nova-sonic: requires 16 kHz, so use UpsamplingPCMSerializer(8000→16000)
    #     and keep the frame/pipeline rate at 16 kHz (no regression).
    # Twilio path passes its own serializer (!= None) and is left untouched.
    #
    # Per-call hot-reload: read RUNTIME_CONFIG once at endpoint entry; mid-call
    # config edits via Admin UI do not affect this in-flight call.
    p = RUNTIME_CONFIG.get_phone_defaults()
    p_engine   = p.get("engine",   DEFAULT_ENGINE)
    p_lang     = p.get("lang",     DEFAULT_LANG)
    p_scenario = p.get("scenario", DEFAULT_SCENARIO)
    p_voice    = p.get("voice",    DEFAULT_MINIMAX_VOICE)
    p_provider = p.get("provider", DEFAULT_PROVIDER)
    p_model    = p.get("model",    DEFAULT_MODEL)
    p_mmmodel  = p.get("minimax_model", DEFAULT_MINIMAX_MODEL)
    logger.info(
        f"phone WS using runtime config: engine={p_engine} lang={p_lang} "
        f"scenario={p_scenario} voice={p_voice}"
    )

    # Audit: phone call start. Phone callers are not logged-in users → actor
    # 'phone'; target = call_id; detail = non-secret runtime descriptors.
    await log_activity(
        actor="phone", type="call-start", target=call_id,
        detail={"engine": p_engine, "lang": p_lang, "scenario": p_scenario, "voice": p_voice},
    )

    # Snapshot meta for history persistence; in-flight call keeps this view
    # even if Admin UI mutates RUNTIME_CONFIG mid-call.
    if _history is not None:
        try:
            _history.attach(call_id, {
                "caller": caller,
                "started_at": sess.get("started", time.time()),
                "engine": p_engine,
                "lang": p_lang,
                "scenario": p_scenario,
                "provider": p_provider,
                "voice": p_voice,
                "model": p_model,
                "minimax_model": p_mmmodel,
            })
        except Exception as e:
            logger.warning(f"history attach failed: {e}")

    # Per-engine uplink sample-rate adaptation for the Chime path (voice-server
    # sends native 8 kHz). Only applied when the caller did NOT pass a serializer
    # (i.e. Chime, serializer is None); Twilio brings its own serializer and is
    # left untouched.
    phone_serializer = serializer
    nova_in_sr = INPUT_SAMPLE_RATE       # nova frame rate stays 16 kHz
    pipeline_in_sr = INPUT_SAMPLE_RATE   # default; overridden to 8000 for Chime
    if serializer is None:
        if p_engine == "nova-sonic":
            # 8 kHz wire → upsample to 16 kHz for the S2S model (no regression).
            phone_serializer = UpsamplingPCMSerializer(
                PHONE_WIRE_SAMPLE_RATE, INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE
            )
        else:
            # pipeline/Transcribe: feed native 8 kHz directly (telephony model).
            pipeline_in_sr = PHONE_WIRE_SAMPLE_RATE

    if p_engine == "nova-sonic":
        task = await _build_nova_sonic_pipeline(
            websocket, p_lang, p_voice,
            system_override=None,
            greeting_override=None,
            scenario_key=p_scenario,
            event_emit=fan_emit,
            input_sample_rate=nova_in_sr,
            history_recorder=_history,
            call_id=call_id,
            is_phone=True,
            serializer=phone_serializer,
        )
    else:
        task = await _build_pipeline(
            websocket, p_lang, p_model, p_voice, p_provider,
            scenario_key=p_scenario,
            minimax_model=p_mmmodel,
            event_emit=fan_emit,
            input_sample_rate=pipeline_in_sr,
            history_recorder=_history,
            call_id=call_id,
            is_phone=True,
            serializer=phone_serializer,
        )
    try:
        await PipelineRunner(handle_sigint=False).run(task)
    finally:
        sessobj = getattr(task, "_aio_session", None)
        if sessobj and not sessobj.closed:
            await sessobj.close()
        for c in getattr(task, "_mcp_clients", None) or []:
            try:
                await c.close()
            except Exception:
                pass
        await session_unregister(call_id)
        if _history is not None:
            try:
                asyncio.create_task(_history.flush_turns_and_summarize(call_id))
            except Exception as e:
                logger.warning(f"history flush schedule failed: {e}")
        await log_activity(
            actor="phone", type="call-end", target=call_id,
            detail={"engine": p_engine, "lang": p_lang, "scenario": p_scenario, "voice": p_voice},
        )
        logger.info(f"phone WS finished: call_id={call_id}")


# ---------------------------------------------------------------------------
# Twilio Media Streams ingress (coexists with Chime SIP/RTP — see tech_design
# 2f6d1455). Audio / μ-law / resampling / interruption-clear are ALL handled by
# pipecat's built-in TwilioFrameSerializer; we only do the webhook handshake +
# signature gate and hand the websocket to the shared _run_phone_session kernel.
#
# env / gate: read fresh per request (not cached at import) so the bot starts
# fine when Twilio is unconfigured and so config can be toggled without a
# restart. TWILIO_ENABLED requires BOTH an auth token (for signature
# verification) AND a public base URL (Twilio-visible https origin, used for the
# signed full_url AND the TwiML wss host — CloudFront rewrites Host so we can't
# trust the inbound request host). TWILIO_ACCOUNT_SID is optional (only needed
# for the serializer's auto-hangup, which also needs call_sid + auth_token).
# ---------------------------------------------------------------------------
def _twilio_config() -> dict:
    """Snapshot Twilio env into a dict. ``enabled`` is the master gate."""
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
    public_base = os.environ.get("TWILIO_PUBLIC_BASE_URL", "").strip()
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
    return {
        "auth_token": auth_token,
        "public_base": public_base,
        "account_sid": account_sid,
        "enabled": bool(auth_token and public_base),
    }


def _twilio_public_host(public_base: str) -> str:
    """Strip scheme (and any trailing slash) from the public base to get the
    bare host[:port][/path] used in the TwiML ``wss://`` Stream URL."""
    host = public_base.strip()
    if "://" in host:
        host = host.split("://", 1)[1]
    return host.rstrip("/")


@app.post("/twilio/incoming-call")
async def twilio_incoming_call(request: Request) -> Response:
    """Twilio voice webhook. Verifies X-Twilio-Signature, returns Connect/Stream
    TwiML that points Twilio's Media Stream at our /twilio/media-stream WS.

    503 when Twilio is not configured; 403 on a missing / invalid signature.
    """
    cfg = _twilio_config()
    if not cfg["enabled"]:
        return Response(
            content="Twilio integration not configured",
            status_code=503,
            media_type="text/plain",
        )

    # full_url MUST be the external URL Twilio actually signed: public base +
    # request path + original raw query (NOT the CloudFront-rewritten host).
    raw_query = request.url.query or ""
    full_url = twilio_sig.build_signed_url(
        cfg["public_base"], request.url.path, raw_query
    )

    form = await request.form()
    post_params = {k: str(v) for k, v in form.items()}
    header_sig = request.headers.get("X-Twilio-Signature", "")

    if not twilio_sig.verify_twilio_signature(
        cfg["auth_token"], full_url, post_params, header_sig
    ):
        logger.warning("twilio incoming-call rejected: bad signature")
        return Response(
            content="invalid signature", status_code=403, media_type="text/plain"
        )

    caller = post_params.get("From", "")
    public_host = _twilio_public_host(cfg["public_base"])
    caller_attr = saxutils.quoteattr(caller)  # safe XML attribute quoting
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response><Connect>"
        f'<Stream url="wss://{public_host}/twilio/media-stream">'
        f'<Parameter name="caller" value={caller_attr}/>'
        "</Stream></Connect></Response>"
    )
    return Response(content=twiml, media_type="text/xml")


# How long we wait for Twilio to send its `start` event before giving up on an
# unauthenticated / abusive WS connection (abuse guard, see tech_design §3).
TWILIO_START_TIMEOUT = float(os.environ.get("TWILIO_START_TIMEOUT", "10"))


async def _twilio_read_start(websocket: WebSocket) -> dict | None:
    """Read WS JSON frames until Twilio's ``start`` event.

    Before ``start`` only ``connected`` is expected; anything else, a malformed
    frame, a timeout, or a disconnect → return ``None`` (caller closes). On
    success returns the parsed ``start`` message dict.
    """
    while True:
        try:
            raw = await asyncio.wait_for(
                websocket.receive_text(), timeout=TWILIO_START_TIMEOUT
            )
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"twilio media-stream: no start frame ({type(e).__name__})")
            return None
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            logger.warning("twilio media-stream: non-JSON frame before start")
            return None
        event = msg.get("event")
        if event == "connected":
            continue  # benign handshake frame, keep waiting for start
        if event == "start":
            return msg
        # Any other event (incl. media before start) is illegal here.
        logger.warning(f"twilio media-stream: unexpected pre-start event {event!r}")
        return None


@app.websocket("/twilio/media-stream")
async def twilio_media_stream(websocket: WebSocket) -> None:
    """Twilio Media Streams WS. Reads the handshake up to the ``start`` event,
    builds a pipecat TwilioFrameSerializer, then hands the websocket to the
    shared phone-session kernel (which constructs the FastAPIWebsocketTransport
    that reads all subsequent media frames). Disabled → refuse the connection.
    """
    cfg = _twilio_config()
    if not cfg["enabled"]:
        # Gracefully refuse: accept then close so Twilio sees a clean shutdown.
        await websocket.accept()
        await websocket.close(code=1011)
        return

    await websocket.accept()

    start = await _twilio_read_start(websocket)
    if start is None:
        await websocket.close(code=1008)  # policy violation / abuse guard
        return

    start_data = start.get("start") or {}
    stream_sid = start.get("streamSid") or start_data.get("streamSid")
    call_sid = start.get("callSid") or start_data.get("callSid")
    caller = (start_data.get("customParameters") or {}).get("caller")
    if not stream_sid:
        logger.warning("twilio media-stream: start without streamSid")
        await websocket.close(code=1008)
        return

    from pipecat.serializers.twilio import TwilioFrameSerializer

    # auto_hang_up REQUIRES call_sid + account_sid + auth_token; disable it
    # unless all three are present, otherwise the constructor raises ValueError.
    have_hangup_creds = bool(call_sid and cfg["account_sid"] and cfg["auth_token"])
    serializer = TwilioFrameSerializer(
        stream_sid,
        call_sid=call_sid,
        account_sid=cfg["account_sid"] or None,
        auth_token=cfg["auth_token"] or None,
        params=TwilioFrameSerializer.InputParams(auto_hang_up=have_hangup_creds),
    )

    call_id = f"twilio-{stream_sid}"
    logger.info(f"twilio media-stream start: call_id={call_id} caller={caller}")

    # Mirror /phone/ws: register the session before handing off to the kernel
    # (the kernel's finally unregisters it).
    primary = _ws_emit(websocket)
    await session_register(call_id, caller=caller, primary_emit=primary)

    await _run_phone_session(
        websocket, call_id=call_id, caller=caller, serializer=serializer
    )


@app.websocket("/monitor/ws")
async def monitor_ws_endpoint(websocket: WebSocket) -> None:
    """Read-only event stream for an active call. Multi-attach is fine.

    Admin-only (NOTE-1): monitoring others' live calls is an admin action, so
    the ws-token must carry role=='admin'."""
    claims = _ws_token_claims(websocket)
    if not claims or claims.get("role") != "admin":
        await websocket.close(code=1008, reason="unauthorized")
        return
    await websocket.accept()
    call_id = websocket.query_params.get("call_id", "")
    if not call_id:
        await websocket.send_text(json.dumps({"type": "error", "text": "call_id required"}))
        await websocket.close()
        return

    monitor_emit = _ws_emit(websocket)
    attached = await session_attach_monitor(call_id, monitor_emit)
    if not attached:
        await websocket.send_text(json.dumps({"type": "error", "text": f"call_id {call_id!r} not active"}))
        await websocket.close()
        return

    await websocket.send_text(json.dumps({"type": "monitor_attached", "call_id": call_id, "t": 0}))
    try:
        # Keep the socket alive; we only read so we notice client disconnects.
        while True:
            try:
                await websocket.receive_text()
            except Exception:
                break
    finally:
        await session_detach_monitor(call_id, monitor_emit)


@asynccontextmanager
async def lifespan(app: FastAPI):
    APP_READINESS.begin_startup()
    APP_READINESS.set_check(
        "session_foundation",
        ok=True,
        critical=True,
        detail="transport-independent contracts available",
    )

    # Runtime config is critical for accepting a session. Snapshot both segments
    # once during startup without performing any external network call.
    try:
        RUNTIME_CONFIG.get_web_defaults()
        RUNTIME_CONFIG.get_phone_defaults()
        APP_READINESS.set_check(
            "runtime_config", ok=True, critical=True, detail="defaults loaded"
        )
    except Exception as e:
        APP_READINESS.set_check(
            "runtime_config", ok=False, critical=True, detail=type(e).__name__
        )
        logger.exception("lifespan: runtime config initialization failed")

    # Prewarm the DDB-backed demo cache so the first call doesn't pay a scan.
    # This remains optional: the store already degrades to an empty cache.
    try:
        n = await DEMO_LOADER.rescan()
        skipped = len(DEMO_LOADER.last_skipped)
        APP_READINESS.set_check(
            "demo_store",
            ok=skipped == 0,
            critical=False,
            detail=f"{n} demos cached; {skipped} skipped",
        )
        logger.info(f"lifespan: prewarmed demo cache ({n} demos)")
    except Exception as e:  # pragma: no cover — rescan already degrades internally
        APP_READINESS.set_check(
            "demo_store", ok=False, critical=False, detail=type(e).__name__
        )
        logger.warning(f"lifespan: demo cache prewarm failed (continuing): {e}")

    # Prewarm + one-time seed of the optional voice registry. If its table is
    # unavailable the in-code registry remains usable and readiness is degraded,
    # not failed.
    try:
        seeded = await VOICE_STORE.seed_if_empty(_VOICE_CONSTANTS)
        APP_READINESS.set_check(
            "voice_store",
            ok=bool(VOICE_STORE.live),
            critical=False,
            detail=(
                f"live={VOICE_STORE.live} seeded={seeded} "
                f"voices={sum(len(v) for v in VOICE_STORE.list().values())}"
            ),
        )
        logger.info(
            f"lifespan: voice registry live={VOICE_STORE.live} seeded={seeded} "
            f"({sum(len(v) for v in VOICE_STORE.list().values())} voices)"
        )
    except Exception as e:  # pragma: no cover — seed/rescan degrade internally
        APP_READINESS.set_check(
            "voice_store", ok=False, critical=False, detail=type(e).__name__
        )
        logger.warning(f"lifespan: voice registry prewarm failed (continuing): {e}")

    # Activity audit store needs no prewarm (no cache) — it is a best-effort
    # write path with a lazy table.
    APP_READINESS.set_check(
        "activity_store", ok=True, critical=False, detail="lazy best-effort store"
    )
    logger.info(
        f"lifespan: activity store ready (table={ACTIVITY_STORE._table_name!r} "
        f"ttl_days={ACTIVITY_STORE._ttl_days})"
    )

    APP_READINESS.mark_started()
    try:
        yield
    finally:
        APP_READINESS.mark_stopping()


# Attach the lifespan to the already-constructed `app` (created at import time,
# above, before this function is defined — so it couldn't be passed to the
# FastAPI() constructor). Setting lifespan_context here keeps a SINGLE lifespan
# (no separate @app.on_event) and runs the demo-cache prewarm on startup.
app.router.lifespan_context = lifespan


# --- Root SPA mount (must come AFTER every @app.* endpoint above) ---------
# Single-page merge (tech_design §2): the Vue 3 + Naive UI ADMIN SPA now serves
# at the site root "/". The static mount itself is PUBLIC by design —
# StaticFiles cannot take a Depends and the login page must load without auth.
# Access control lives on the /api/* + WS routes (JWT session cookie via
# require_user / require_admin); the SPA router guard redirects unauthenticated
# users to /login.
#
# This mount is a catch-all, so it MUST stay below every /api/* + WS route or
# it would shadow them. StaticFiles' html=True serves index.html for unknown
# paths under the mount, enabling client-side hash routing (#/dashboard,
# #/login, ...) without server help.
#
# NOTE: the legacy demo SPA was removed in T4 (single-page merge into admin);
# only this admin SPA is mounted at the root now.
try:
    if os.path.isdir(ADMIN_DIST_DIR) and os.path.isfile(os.path.join(ADMIN_DIST_DIR, "index.html")):
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=ADMIN_DIST_DIR, html=True), name="spa")
        logger.info(f"mounted admin SPA at / from {ADMIN_DIST_DIR}")
    else:
        logger.warning(f"admin SPA dist not found at {ADMIN_DIST_DIR}; / routes 404")
except Exception as _e:
    logger.warning(f"failed to mount admin SPA: {_e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
