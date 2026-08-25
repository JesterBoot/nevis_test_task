# Nevis Backend Home Task

This repository contains a focused FastAPI MVP for the Nevis backend home
task. It supports client creation, document ingestion with local embeddings, and
combined client/document search.

## What is included

- `POST /clients`
- `POST /clients/{id}/documents`
- `GET /search`
- `/health/live`, `/health/ready`, and `/health/startup`
- PostgreSQL with pgvector as the only database backend
- Local `sentence-transformers/all-MiniLM-L6-v2` embeddings
- CPU-only PyTorch in the Linux Docker runtime; CUDA, NVIDIA, and Triton
  runtime packages are intentionally excluded
- Docker Compose with the API exposed on port `8080`
- Alembic migrations and generated FastAPI OpenAPI/Swagger documentation

Production features such as authentication, summaries, asynchronous indexing,
multi-tenancy, and operational security controls are intentionally out of scope
for this MVP.

## Repository layout

Application code lives directly under `src/`. Runtime and delivery files remain
at the repository root.

```text
src/
|-- main.py
|-- api/
|-- core/
|-- db/
|-- models/
|-- schemas/
|-- search/
|-- services/
`-- migrations/
    |-- env.py
    `-- versions/
```

Alembic is configured by `src/alembic.ini` and points to `src/migrations`.

## Quick start

Use Python 3.13 and Docker Compose.
`uv sync` is only required for local development and test commands. Docker Compose provides the complete reviewer runtime.

```bash
cp .env.example .env
uv sync  # optional
docker compose up --build -d
```

The first API container startup downloads the configured MiniLM model into a
named Docker volume. After the model is present, the container switches Hugging
Face and Transformers to offline mode before starting Uvicorn.

Open Swagger UI at:

```text
http://localhost:8080/docs
```

Remove the database and model volumes when you want a fully fresh Docker run:

```bash
docker compose down -v
docker compose up --build -d
```

## Configuration

The defaults in `.env.example` are enough for local Docker review. It is kept
intentionally short and contains only reviewer-facing values:

- `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` for PostgreSQL;
- `API_PORT` for the host port, defaulting to `8080`;
- `DEBUG=1` for Uvicorn reload, FastAPI debug mode, and SQLAlchemy SQL logging;
- `EMBEDDING_MODEL` and `EMBEDDING_DIMENSION`;
- `MAX_DOCUMENT_CHARS`, `MAX_CHUNKS`, `CHUNK_SIZE`, and `CHUNK_OVERLAP`;
- `SEARCH_LIMIT_DEFAULT`, `SEARCH_LIMIT_MAX`, `FTS_BOOST`, and
  `SEMANTIC_SIMILARITY_THRESHOLD`.

Stable Docker details such as container ports, healthcheck timings, model cache
paths, and PostgreSQL data paths live directly in `docker-compose.yml`.
Docker Compose mounts `./src` into `/app/src`, so `DEBUG=1` enables hot reload
without rebuilding the image. Model weights remain in the separate named model
volume and are not re-downloaded on source changes.

## Health and migrations

The API exposes:

- `GET /health/live` for process liveness;
- `GET /health/ready` for readiness after startup dependency checks;
- `GET /health/startup` for an explicit database dependency check.

The container entrypoint runs `alembic upgrade head` before Uvicorn starts.
Docker Compose uses `/health/ready` as the API healthcheck.

Manual migration commands:

```bash
make migrate
make downgrade-migration
make show-heads
```

## API examples

Create a client:

```bash
curl -X POST http://localhost:8080/clients \
  -H 'Content-Type: application/json' \
  -d '{
    "first_name": "Anton",
    "last_name": "Batiaev",
    "email": "anton.batiaev@neviswealth.com",
    "countryOfResidence": "GB"
  }'
```

Example response:

```json
{
  "id": "081c0705-e062-40b4-98c1-dbbf2087b900",
  "first_name": "Anton",
  "last_name": "Batiaev",
  "email": "anton.batiaev@neviswealth.com",
  "countryOfResidence": "GB"
}
```

Create a document for that client:

```bash
CLIENT_ID="081c0705-e062-40b4-98c1-dbbf2087b900"
curl -X POST http://localhost:8080/clients/${CLIENT_ID}/documents \
  -H 'Content-Type: application/json' \
  -d '{
    "title": "Proof of address",
    "content": "Utility bill issued in August."
  }'
```

Example response:

```json
{
  "id": "11aa804b-5431-4f42-8f9b-172eedf691f0",
  "client_id": "081c0705-e062-40b4-98c1-dbbf2087b900",
  "title": "Proof of address",
  "content": "Utility bill issued in August.",
  "created_at": "2026-08-25T14:59:43.165372Z"
}
```

Search by company phrase:

```bash
curl --get http://localhost:8080/search \
  --data-urlencode 'q=Nevis Wealth'
```

Example response:

```json
[
  {
    "type": "client",
    "id": "081c0705-e062-40b4-98c1-dbbf2087b900",
    "first_name": "Anton",
    "last_name": "Batiaev",
    "email": "anton.batiaev@neviswealth.com"
  }
]
```

Search by semantic document concept:

```bash
curl --get http://localhost:8080/search \
  --data-urlencode 'q=address proof'
```

Example response:

```json
[
  {
    "type": "document",
    "id": "11aa804b-5431-4f42-8f9b-172eedf691f0",
    "client_id": "081c0705-e062-40b4-98c1-dbbf2087b900",
    "title": "Proof of address",
    "snippet": "Utility bill issued in August."
  }
]
```

Search returns a direct JSON array. It does not expose `score`,
`ranking_score`, or embedding fields.

## Tests

Install dependencies:

```bash
uv sync
```

Run the regular non-MiniLM regression suite:

```bash
make test       # fast unit/API/integration tests
make test-all   # complete non-semantic pytest suite
make test-semantic  # real MiniLM acceptance test in Docker
make check     # lint + compile + test
```

`make test-semantic` executes `scripts/semantic_spike.py` in the API container
and uses the Docker model volume. Local `uv run pytest` does not download model
weights or run MiniLM inference by default.

## Docker smoke check

Use this path for reviewer verification:

```bash
cp .env.example .env
docker compose config
docker compose up --build -d
docker compose ps
docker compose exec api /app/.venv/bin/alembic -c /app/src/alembic.ini current
make test-semantic
curl http://localhost:8080/health/ready
```

Then run the curl examples above and stop the stack:

```bash
docker compose down
```

## Design decisions and trade-offs

- FastAPI keeps the HTTP layer small and provides OpenAPI/Swagger without custom
  documentation generation.
- PostgreSQL stores relational data, full-text indexes, and pgvector embeddings,
  avoiding a separate vector database for this small MVP.
- Local embeddings avoid external API dependencies and data transfer, at the cost
  of higher CPU usage, model footprint, and first-start provisioning time.
- The Docker runtime uses CPU-only PyTorch. An ONNX Runtime migration is
  explicitly out of scope; the SentenceTransformers API and MiniLM model remain
  unchanged.
- Model weights are downloaded only during initial provisioning; subsequent
  inference runs offline from the Docker volume.
- Chunking is deterministic: documents up to `1000` characters use one chunk;
  larger documents use `1000`-character windows with `100` characters of overlap.
- Search keeps client and document relevance separate. Client results are ordered
  first, followed by document results.
- Document ranking applies the semantic threshold before the FTS boost. A lexical
  match cannot resurrect a semantically weak document.
- Optional summaries, authentication, operational security controls, background
  indexing, and provider routing are documented as Future Work rather than MVP
  implementation scope.

Verify the runtime uses CPU-only PyTorch:

```bash
docker compose exec api /app/.venv/bin/python -c \
  "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The expected CUDA availability output is `False`. Model weights are stored in
the named `model_cache` volume and are not embedded in the API image.
