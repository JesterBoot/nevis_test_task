from fastapi import APIRouter

from api.endpoints import (
    clients,
    documents,
    health,
    search,
)

api_router = APIRouter()
api_router.include_router(clients.router, prefix="/clients", tags=["Clients"])
api_router.include_router(
    documents.router,
    prefix="/clients/{id}/documents",
    tags=["Documents"],
)
api_router.include_router(search.router)
api_router.include_router(health.router)
