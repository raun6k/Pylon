import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from pylon.api.routes import router
from pylon.config import get_config
from pylon.engine import Engine
from pylon.frontend import TextGenerator
from pylon.model.tokenizer import CheckpointTokenizer

logger = logging.getLogger("pylon")


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    logger.info("pylon_startup model=%s", config.model_id)
    engine = Engine(config)
    try:
        tokenizer = CheckpointTokenizer(engine.snapshot, engine.stop_token_ids)
        generator = TextGenerator(tokenizer, engine)
        logger.info("pylon_warmup_started")
        generator.warm_up()
        logger.info("pylon_warmup_completed")
        app.state.generator = generator
        logger.info(
            "pylon_ready model=%s revision=%s",
            generator.model_id,
            engine.model_revision,
        )
        yield
    finally:
        engine.close()


def create_app() -> FastAPI:
    app = FastAPI(title="Pylon", lifespan=lifespan)
    app.include_router(router)
    return app
