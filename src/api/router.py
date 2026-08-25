from fastapi import APIRouter

from api.endpoints import clients, health

api_router = APIRouter()
api_router.include_router(clients.router, prefix="/clients", tags=["Clients"])
api_router.include_router(health.router)
