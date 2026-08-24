FROM python:3.13-slim AS builder

ARG UV_VERSION=0.7.20

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    MODEL_CACHE_DIR=/opt/models \
    HF_HOME=/opt/models \
    SENTENCE_TRANSFORMERS_HOME=/opt/models \
    PYTHONPATH=/app/src \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

RUN python -m pip install --upgrade pip \
    && python -m pip install "uv==${UV_VERSION}"

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts

RUN uv sync --locked --no-dev

FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_CACHE_DIR=/opt/models \
    HF_HOME=/opt/models \
    SENTENCE_TRANSFORMERS_HOME=/opt/models \
    PYTHONPATH=/app/src \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY scripts ./scripts

EXPOSE 8080

ENTRYPOINT ["sh", "/app/scripts/entrypoint.sh"]
