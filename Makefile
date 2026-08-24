.PHONY: install test test-db compile check run migrate downgrade-migration \
	make-migration show-heads docker-config docker-build docker-up docker-down \
	docker-logs

UV ?= uv
DC ?= docker compose
API_SERVICE ?= api
APP_MODULE ?= main:app
APP_HOST ?= 0.0.0.0
APP_PORT ?= 8080
ALEMBIC ?= $(UV) run alembic
ALEMBIC_CONFIG ?= src/alembic.ini

install:
	$(UV) sync

test:
	$(UV) run pytest

test-db:
	$(UV) run pytest tests/test_database.py

compile:
	$(UV) run python -m compileall -q src tests

check:
	$(UV) lock --check
	$(UV) run ruff check src tests scripts
	$(MAKE) compile
	$(MAKE) test

run:
	$(UV) run uvicorn --app-dir src $(APP_MODULE) \
		--host $(APP_HOST) \
		--port $(APP_PORT) \
		--reload

migrate:
	$(DC) exec $(API_SERVICE) /app/.venv/bin/alembic \
		-c /app/src/alembic.ini \
		upgrade head

downgrade-migration:
	$(DC) exec $(API_SERVICE) /app/.venv/bin/alembic \
		-c /app/src/alembic.ini \
		downgrade -1

make-migration:
	$(ALEMBIC) -c $(ALEMBIC_CONFIG) \
		revision --autogenerate -m "$(or $(name),migration)"

show-heads:
	$(ALEMBIC) -c $(ALEMBIC_CONFIG) heads

docker-config:
	$(DC) config

docker-build:
	$(DC) build

docker-up:
	$(DC) up -d

docker-down:
	$(DC) down

docker-logs:
	$(DC) logs -f $(API_SERVICE)
