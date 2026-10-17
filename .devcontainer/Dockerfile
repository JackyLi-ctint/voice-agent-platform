FROM mcr.microsoft.com/devcontainers/python@sha256:589b65db720b03c96d721706005f176ff467ee879f92c2d137c04a537feffa56

USER root

ARG PIP_INDEX_URL=https://pypi.org/simple

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY --chown=vscode:vscode requirements-dev.lock /tmp/requirements-dev.lock
RUN python -m pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" \
    -r /tmp/requirements-dev.lock

USER vscode
WORKDIR /workspace
