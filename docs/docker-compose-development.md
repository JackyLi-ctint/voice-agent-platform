# Docker Compose Development Environment

Use this workflow when the Kiro Dev Containers extension is unavailable. Kiro stays on Windows and Docker runs the Linux/Python 3.12 test environment.

## Prerequisites

1. Install Docker Desktop.
2. Select Linux containers and enable the WSL 2 backend.
3. Start Docker Desktop and wait until it reports that the engine is running.
4. From the repository root, confirm:

```powershell
docker version
docker compose version
```

The first command must print both Client and Server information. A missing `dockerDesktopLinuxEngine` pipe means Docker Desktop is not running yet.

## Build the environment

```powershell
docker compose build app
```

The image uses Debian GNU/Linux 12 (Bookworm) with Python 3.12, installs `ffmpeg`, and installs the tested transitive dependency graph in `requirements-dev.lock`. `requirements-dev.txt` records the direct pinned inputs that produce that lock.

## Run checks

Run the Phase 1 deterministic check first:

```powershell
docker compose run --rm app python -m voice_agent.dev.smoke
```

Then run the focused regression group:

```powershell
docker compose run --rm app python -m pytest -q tests/test_phone_session.py tests/test_phone_sample_rate.py tests/test_twilio_endpoints.py tests/test_twilio_integration.py tests/test_latency.py tests/test_latency_log.py tests/test_runtime_config.py
```

Finally run the complete Python suite:

```powershell
docker compose run --rm app python -m pytest -q tests
```

## Interactive shell

```powershell
docker compose run --rm app bash
```

The working tree is mounted at `/workspace`, and `PYTHONPATH=/workspace` permits existing flat-module imports such as `import bot` while the Phase 1 `voice_agent` package is introduced.

The Compose service masks a developer's host `.env` with the tracked, empty `.docker/compose.env` because `bot.py` calls `load_dotenv(override=True)`. This preserves the dummy credentials declared in `compose.yaml` and prevents unit tests from silently using a developer's AWS credentials. Do not put secrets in `.docker/compose.env`; for an intentional live AWS smoke test, pass credentials at runtime through an approved mechanism rather than storing them in `compose.yaml`.

The custom Polly service uses the `aiobotocore` session supplied by Pipecat's AWS extra and calls `create_client(...)` on that session. Do not add `aioboto3`: currently available releases pin an incompatible `aiobotocore 2.x` range.

MCP support is supplied by Pipecat's `mcp` extra; do not pin an older standalone MCP package alongside it.

The image installs `requirements-dev.lock`, the tested transitive dependency graph. `requirements-dev.txt` remains the human-maintained direct dependency manifest. After changing it, build a candidate image, run the full suite, then regenerate the lock from that tested container before committing both files.
