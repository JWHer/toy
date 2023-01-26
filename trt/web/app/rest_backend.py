from pathlib import Path

from fastapi import FastAPI

from app.api.api_v1.api import api_router
from app.core.config import settings


def init_rest_api() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
    )
    
    BASE_PATH = Path(__file__).resolve().parent
    print("BASE_PATH {0}".format(BASE_PATH))
    print("BASE_PATH %s"%BASE_PATH)
    app.include_router(api_router, prefix=settings.API_V1_STR)
    return app