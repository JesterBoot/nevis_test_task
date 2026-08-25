#!/bin/sh

set -eu

MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/opt/models}"
MODEL_READY_FILE="${MODEL_READY_FILE:-${MODEL_CACHE_DIR}/.model-ready}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
API_HOST="${API_HOST:-0.0.0.0}"
API_CONTAINER_PORT="${API_CONTAINER_PORT:-8080}"
DEBUG="${DEBUG:-0}"

mkdir -p "${MODEL_CACHE_DIR}"

model_ready=0
if [ -f "${MODEL_READY_FILE}" ] \
    && [ "$(cat "${MODEL_READY_FILE}")" = "${EMBEDDING_MODEL}" ]; then
    model_ready=1
fi

if [ "${model_ready}" -eq 0 ]; then
    echo "Downloading embedding model ${EMBEDDING_MODEL} into ${MODEL_CACHE_DIR}"
    HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0 \
    /app/.venv/bin/python /app/scripts/download_model.py
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

/app/.venv/bin/alembic \
    -c /app/src/alembic.ini \
    upgrade head

if [ "${DEBUG}" = "1" ]; then
    exec /app/.venv/bin/uvicorn \
        main:app \
        --host "${API_HOST}" \
        --port "${API_CONTAINER_PORT}" \
        --reload \
        --reload-dir /app/src
fi

exec /app/.venv/bin/uvicorn \
    main:app \
    --host "${API_HOST}" \
    --port "${API_CONTAINER_PORT}"
