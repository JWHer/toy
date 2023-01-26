import asyncio
import logging
import matplotlib
import uvicorn

from app.db.session import SessionLocal
from app.rest_backend import init_rest_api
from app.core.config import settings
from app.service.base.worker import worker
from app.service.collector_service import collector_service

# get logger
logger = logging.getLogger(__name__)

def initialize() -> None:
    try:
        db = SessionLocal()
        # Try to create session to check if DB is awake
        db.execute("SELECT 1")

    except Exception as e:
        logger.error("DB is not initialized!")
        raise e

initialize()
app = init_rest_api()

@app.on_event('startup')
def run_on_startup():
    logger.info('Start up tasks')
    matplotlib.pyplot.set_loglevel("error")
    asyncio.create_task(worker.loop())
    asyncio.create_task(collector_service.loop())

if __name__ == "__main__":
    uvicorn.run(app, host=settings.HOST, port=int(settings.PORT), loop='asyncio')

