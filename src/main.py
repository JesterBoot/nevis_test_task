from fastapi import FastAPI

from core.custom_logging import configure_logging

configure_logging()


def create_app() -> FastAPI:
    return FastAPI(
        title="Nevis Backend API",
        version="0.1.0",
    )


app = create_app()
