# Nevis Backend MVP Architecture

## 1. Purpose and scope

This document is the implementation source of truth for the Nevis Backend home task.
The service creates clients and their documents, then searches across both entity types.

The MVP supports:

- company-oriented client discovery from corporate email domains, for example
  `Nevis Wealth` matching `anton.batiaev@neviswealth.com`;
- semantic document retrieval, for example `address proof` matching a document that
  contains `utility bill`;
- a direct, JSON-array search response compatible with the supplied API shape;
- liveness and dependency health checks under `/health`.

The MVP intentionally optimizes for a small, reproducible deployment and clear
trade-offs. It does not implement authentication, summaries, audit logging,
encryption policy, retention policy, provider routing, Redis, Celery, Qdrant, or
multi-tenant authorization.

## 2. Technology choices

| Concern         | MVP decision                                                           |
|-----------------|------------------------------------------------------------------------|
| API             | Python, FastAPI, and generated OpenAPI documentation                   |
| Persistence     | PostgreSQL with SQLModel on SQLAlchemy and Alembic                     |
| Vector storage  | pgvector in the same PostgreSQL database                               |
| Lexical search  | PostgreSQL full-text search plus normalized email-domain matching      |
| Embeddings      | Local `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, CPU    |
| ML runtime      | CPU-only PyTorch on Linux Docker builds; no CUDA, NVIDIA, or Triton    |
| Deployment      | Docker Compose; API exposed on port `8080`                             |
| Background work | None; indexing is synchronous                                          |

PostgreSQL is both the source of truth and the persistent vector store. A separate
vector database is intentionally avoided because the expected dataset and deployment
topology do not justify another stateful service.

The MiniLM model is the initial fixed MVP choice, subject to the required semantic
acceptance test. The implementation must verify that the real fixture ranks a
`utility bill` document for the query `address proof`. If the fixture fails, the
model or retrieval approach must be reconsidered before the search implementation is
treated as complete.

Model weights are downloaded on the first API container startup into a named Docker
volume mounted at `/opt/models`. The startup script writes a marker containing the
model name after a successful load. Subsequent starts reuse the volume without a
network request and force offline model loading before launching the application.
The API creates the embedding provider lazily on the first ingestion request, then
reuses one provider and model instance per application process.

The Linux Docker dependency resolution uses the PyTorch CPU index so the runtime
does not install CUDA, NVIDIA, or Triton packages. The inference backend remains
PyTorch through the existing SentenceTransformers provider. Migrating to ONNX
Runtime is explicitly out of scope for this MVP.

`DEBUG=1` is the single development switch. It enables Uvicorn reload, FastAPI
debug mode, and SQLAlchemy SQL logging. Docker Compose bind-mounts `./src` into
`/app/src` so source changes are visible to the reload process, while model
weights remain in the separate named model volume. `DEBUG=0` disables reload and
SQL query logging.

The example environment file is intentionally small. It keeps credentials and
reviewer-facing application knobs configurable, while stable Docker details such
as internal ports, cache paths, data paths, and healthcheck timings live directly
in `docker-compose.yml`.

## 3. Data model and lifecycle

### Client

The `Client` table contains:

- `id` - UUID primary key;
- `first_name`;
- `last_name`;
- `email` - original normalized email value;
- `normalized_email` - lowercase comparison value with surrounding whitespace removed;
- `email_domain` - complete normalized domain, such as `neviswealth.com`;
- `email_domain_label` - lightweight comparison label derived by removing the
  final DNS label, such as `neviswealth` from `neviswealth.com`. This intentionally
  does not implement public-suffix or registrable-domain parsing;
- `country_of_residence`;
- `created_at`.

The normalized email is unique. The complete domain and lightweight domain label
are stored as derived fields so company matching remains deterministic and
inexpensive. The label is not a public-suffix-aware registrable-domain value.

### Document

The `Document` table contains:

- `id` - UUID primary key;
- `client_id` - foreign key to `Client`;
- `title`;
- `content`;
- `created_at`.

The MVP does not expose document update or delete endpoints.

### DocumentChunk

The `DocumentChunk` table contains:

- `id` - UUID primary key;
- `document_id` - foreign key to `Document`;
- `position` - zero-based order within the document;
- `content`;
- `embedding` - pgvector column with 384 dimensions.

Chunks are owned by their parent document. The foreign key uses `ON DELETE CASCADE`
to prevent orphaned vector records. Cascade behaviour is tested at the persistence
and service layers even though public document deletion is outside the supplied API
contract.

Hashes, embedding-model versions, chunking versions, summary records, and cache
metadata are intentionally excluded from the MVP. They can be introduced together
with document updates and re-indexing if a later requirement needs them.

## 4. Ingestion flow

Document ingestion is deliberately split into preparation and persistence:

```text
validate request
      |
require no caller-owned transaction
      |
owned client-existence lookup
      |
finish lookup transaction
      |
deterministic chunking
      |
generate all local embeddings in one batch from title + chunk text
      |
owned persistence transaction
      |
persist document, chunks, and vectors
      |
COMMIT
```

`create_document()` requires a session without an already active caller-owned
transaction. It owns and finishes the client-existence lookup transaction, then
runs chunking and CPU-bound embedding inference with no database transaction open.
It owns a separate persistence transaction for the document and chunks. The service
never commits or rolls back an external transaction.

If validation, chunking, or inference fails, no document write has started. If
persistence fails after the owned transaction begins, the transaction is rolled
back and no partial document or chunk state remains.

Client existence is checked early to avoid unnecessary embedding work. The document
foreign-key constraint remains the final client/document integrity boundary if a
concurrent deletion were ever introduced.

Chunking is intentionally simple and deterministic:

- content up to `1000` characters becomes one chunk;
- larger content uses fixed-size `1000` character chunks;
- adjacent chunks overlap by `100` characters;
- the maximum document size is `50_000` characters;
- the maximum chunk count is `100`.

These values are configuration defaults, not a semantic chunking system. Their
purpose is to keep document size from becoming an architectural constraint without
adding unnecessary complexity to the home task.

## 5. Search design

### Client/company matching

Email matching uses the complete corporate domain rather than invented sub-tokens.
For example:

```text
email: anton.batiaev@neviswealth.com
domain: neviswealth.com
label:  neviswealth
```

The query `Nevis Wealth` is lowercased, normalized, and compacted to `neviswealth`
for comparison. The implementation does not automatically split `neviswealth.com`
into `nevis` and `wealth`.

Exact normalized domain/label matches receive the strongest client ranking. PostgreSQL
full-text search over client names, email, and domain fields provides a fallback for
less exact queries.

### Document/content matching

Each document chunk is embedded locally with the same MiniLM model used for search
queries. The stored chunk content remains the source content chunk, while the
embedding input combines the document title with that chunk. This keeps the B2
semantic fixture aligned with real ingestion, so a title such as `Proof of address`
can help the content chunk `Utility bill issued in August.` rank for `address
proof`. PostgreSQL/pgvector provides cosine-distance retrieval. PostgreSQL full-text
search supplies bounded lexical candidates and a deterministic configurable boost.

The document pipeline is:

1. embed the query once;
2. retrieve up to `candidate_limit` vector chunk candidates;
3. retrieve up to `candidate_limit` lexical document candidates;
4. fetch chunks belonging to lexical documents and union chunk IDs;
5. calculate raw cosine similarity in one PostgreSQL query;
6. discard chunks below `SEMANTIC_SIMILARITY_THRESHOLD`;
7. group surviving chunks by document and keep the maximum raw cosine;
8. determine the lexical match at document level;
9. apply `FTS_BOOST` once per matching document;
10. rank documents deterministically.

The candidate limit is the bounded implementation heuristic
`min(max(limit * 5, 20), 50)`. It retrieves more candidates than the final result
limit without introducing a generic retrieval framework. It is not presented as an
optimal or benchmark-derived value.

The internal document score is:

```text
best_raw_cosine
    + FTS_BOOST when the document has a lexical match
```

The threshold is applied before the boost. A lexical match cannot resurrect a
document whose raw semantic similarity is below the threshold. The lexical match
may be on a different chunk from the chunk that provides the best semantic score.
Multiple chunks from one document are collapsed to one document-level result.
The public snippet comes from the surviving chunk with the highest raw cosine
similarity. Equal similarities use chunk position and then chunk UUID as
deterministic tie-breakers.

The internal score is not exposed by the public API. Search results contain only
safe entity metadata and a bounded document snippet.

`FTS_BOOST=0.10` and `SEMANTIC_SIMILARITY_THRESHOLD=0.30` are configurable initial
B6 fixture-derived values. The threshold is experimental and is not a calibrated
global relevance boundary.

Client and document relevance values are not compared. Client results use exact
domain/label matching followed by PostgreSQL FTS fallback. Document results use
cosine similarity plus the document-level lexical boost. The public response
combines the two result types with clients first, followed by documents.

PostgreSQL with pgvector is the only supported database backend. Search and
persistence tests use the same PostgreSQL implementation rather than maintaining a
second database-specific code path.

## 6. Public API contracts

All endpoints return JSON. No authentication header is required by the MVP.

### Health checks

The service exposes operational health endpoints:

- `GET /health/live` returns `200` when the process is running;
- `GET /health/ready` returns `200` only after the application startup dependency
  check succeeds, otherwise `503`;
- `GET /health/startup` performs an explicit database dependency check and returns
  `503` when the database is unavailable.

The startup check is executed by the FastAPI lifespan. It records the result in
`app.state.startup_ok` and does not run during module import.
Docker Compose uses `/health/ready` for the API container healthcheck.

### Create client

**Method:** `POST`  
**Path:** `/clients`

Request:

```json
{
  "first_name": "Anton",
  "last_name": "Batiaev",
  "email": "anton.batiaev@neviswealth.com",
  "countryOfResidence": "GB"
}
```

Response (`201 Created`):

```json
{
  "id": "8f245e2c-3a0d-44cc-a56e-7be5e091406a",
  "first_name": "Anton",
  "last_name": "Batiaev",
  "email": "anton.batiaev@neviswealth.com",
  "countryOfResidence": "GB"
}
```

Errors:

- `422` - invalid request body or email;
- `409` - duplicate normalized email.

### Create document

**Method:** `POST`  
**Path:** `/clients/{id}/documents`

Request:

```json
{
  "title": "Proof of address",
  "content": "Utility bill issued in August."
}
```

Response (`201 Created`):

```json
{
  "id": "408f3d32-6200-4a1d-a7c3-57cdac6560e4",
  "client_id": "8f245e2c-3a0d-44cc-a56e-7be5e091406a",
  "title": "Proof of address",
  "content": "Utility bill issued in August.",
  "created_at": "2026-08-23T10:00:00Z"
}
```

Errors:

- `422` - invalid request body, whitespace-only title/content, or configured
  document limit exceeded;
- `404` - client not found.

### Search clients and documents

**Method:** `GET`  
**Path:** `/search?q={query}&limit={limit}`

`limit` is optional, defaults to `10`, and is capped at `50`.

The response is a direct JSON array, not an object containing a `results` property:

```json
[
  {
    "type": "client",
    "id": "8f245e2c-3a0d-44cc-a56e-7be5e091406a",
    "first_name": "Anton",
    "last_name": "Batiaev",
    "email": "anton.batiaev@neviswealth.com"
  },
  {
    "type": "document",
    "id": "408f3d32-6200-4a1d-a7c3-57cdac6560e4",
    "client_id": "8f245e2c-3a0d-44cc-a56e-7be5e091406a",
    "title": "Proof of address",
    "snippet": "Utility bill issued in August."
  }
]
```

Errors:

- `422` - missing, empty, or invalid `q`/`limit`.

The public response does not expose a relevance score. The search service may use
an internal `ranking_score` for ordering, but it is not a calibrated probability
or public semantic-similarity value.

## 7. Test strategy

The regression suite uses the same boundaries as the MVP implementation:

- unit tests cover deterministic chunking, embedding-provider invariants, query
  normalization, and ranking aggregation;
- API and integration tests run against PostgreSQL with pgvector using schema
  created by the current Alembic migrations;
- normal API and integration tests use deterministic fake embeddings so they do
  not load MiniLM implicitly;
- the real MiniLM semantic acceptance check runs `scripts/semantic_spike.py`
  inside the Docker API container against the Docker model volume;
- local `uv run pytest` and `make test-all` do not download model weights or run
  MiniLM inference by default;
- the lifecycle regression creates a client, creates a document, verifies vector
  chunks were persisted, and searches both the company/domain and semantic
  document paths.

If PostgreSQL/pgvector or the Docker model volume is unavailable, the affected
checks report that explicitly; the relevant task is not considered complete until
those checks pass in an environment with the required dependencies.

Docker image optimization is measured with:

```bash
docker compose build --no-cache api
docker compose images api
docker history "$(docker compose images -q api)"
```

The model weights are intentionally absent from the image and are provisioned in
the named `model_cache` volume at first API startup.

## 8. Future work

The following are deliberately excluded from the MVP:

- authentication and organization-scoped authorization;
- document update and public deletion endpoints;
- summary generation through an LLM provider;
- audit events and compliance workflows;
- field-level encryption, TLS policy, backup retention, and provider policy checks;
- asynchronous indexing and operational metrics;
- model/versioned derived-artifact invalidation.

At MVP scale, vector retrieval uses exact pgvector search rather than an ANN index.
This keeps the implementation simple and avoids tuning an index before corpus size
and latency requirements are known. HNSW/IVFFlat and indexed PostgreSQL FTS are
natural next steps if the corpus grows.
