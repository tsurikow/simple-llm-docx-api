import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import router
from app.config import get_settings
from app.db import (
    init_db,
    mark_documents_as_failed,
    mark_questions_as_failed,
)
from app.models import DocumentStatus, QuestionStatus
from app.storage import ensure_storage_dirs


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        ensure_storage_dirs(settings)
        await init_db(settings.database_url)
        await mark_documents_as_failed(
            status=DocumentStatus.PENDING,
            reason="Service restarted before background task started.",
        )
        await mark_documents_as_failed(
            status=DocumentStatus.PROCESSING,
            reason="Service restarted before document indexing completed.",
        )
        await mark_questions_as_failed(
            status=QuestionStatus.PENDING,
            reason="Service restarted before background task started.",
        )
        await mark_questions_as_failed(
            status=QuestionStatus.PROCESSING,
            reason="Service restarted before completion.",
        )
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
