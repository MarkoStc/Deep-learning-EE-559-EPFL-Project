# syntax=docker/dockerfile:1.7
#
# Two runtime targets from one file:
#   docker build --target cpu  -t red-queen:cpu  .   (default target)
#   docker build --target cuda -t red-queen:cuda .   (CUDA 12.6, run with --gpus all)
#
# Builder stages install a uv-managed CPython and the locked dependencies into
# /app/.venv. Runtime stages copy only the interpreter and the venv, and run as
# an unprivileged user.

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.12.5

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ── builders ──────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS builder-base
ARG PYTHON_VERSION
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/python \
    UV_PYTHON_PREFERENCE=only-managed
RUN uv python install "${PYTHON_VERSION}"
WORKDIR /app

FROM builder-base AS builder-cpu
# Dependencies first, in their own layer, so source edits do not reinstall torch.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project --extra cpu
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --extra cpu

FROM builder-base AS builder-cuda
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project --extra cu126
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --extra cu126

# ── runtimes ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.6.3-base-ubuntu24.04 AS cuda
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app \
    && install -d -o app -g app /app /app/.cache
COPY --from=builder-cuda /python /python
COPY --from=builder-cuda --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app configs /app/configs
ENV PATH=/app/.venv/bin:$PATH \
    HF_HOME=/app/.cache/huggingface \
    NLTK_DATA=/app/.cache/nltk \
    PYTHONUNBUFFERED=1
WORKDIR /app
USER app
ENTRYPOINT ["red-queen"]
CMD ["--help"]

FROM debian:bookworm-slim AS cpu
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app \
    && install -d -o app -g app /app /app/.cache
COPY --from=builder-cpu /python /python
COPY --from=builder-cpu --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app configs /app/configs
ENV PATH=/app/.venv/bin:$PATH \
    HF_HOME=/app/.cache/huggingface \
    NLTK_DATA=/app/.cache/nltk \
    PYTHONUNBUFFERED=1
WORKDIR /app
USER app
ENTRYPOINT ["red-queen"]
CMD ["--help"]
