from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import router
from app.config import get_settings
from app.db import init_db, mark_processing_questions_as_failed
from app.storage import ensure_storage_dirs


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        ensure_storage_dirs(settings)
        init_db(settings.database_url)
        mark_processing_questions_as_failed("Service restarted before completion.")
        yield

    app = FastAPI(
        title="DOCX Q&A API",
        description="Ask questions about uploaded DOCX files with OpenRouter-backed LLM.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()

