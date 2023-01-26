from fastapi import APIRouter
from app.api.api_v1.endpoints import model, storage


api_router = APIRouter()
api_router.include_router(model.router, prefix="/model", tags=["model"])
api_router.include_router(storage.router, prefix="/storage", tags=["storage"])
