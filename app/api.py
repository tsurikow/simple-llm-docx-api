import asyncio
from functools import lru_cache
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.models import Document, DocumentStatus, Question, QuestionStatus, utcnow
from app.schemas import (
    CreateQuestionRequest,
    CreateQuestionResponse,
    DocumentStatusResponse,
    QuestionStatusResponse,
    UploadDocumentResponse,
)
from app.services.document_service import DocumentService
from app.services.embeddings import EmbeddingService
from app.services.qa_service import QAService
from app.storage import get_document_dir

router = APIRouter()


def _ensure_docx(filename: str) -> None:
    if not filename.lower().endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are supported.")


def _ensure_upload_size(raw_bytes: bytes, max_upload_mb: int) -> None:
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw_bytes) > max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File is too large. Max size is {max_upload_mb} MB.",
        )


def _ensure_document_ready(document: Document) -> None:
    if document.status == DocumentStatus.READY:
        return
    if document.status == DocumentStatus.FAILED:
        raise HTTPException(
            status_code=409,
            detail=document.error or "Document indexing failed.",
        )
    raise HTTPException(status_code=409, detail="Document is still being indexed.")


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    settings = get_settings()
    return EmbeddingService(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        model=settings.openrouter_embedding_model,
        batch_size=settings.embedding_batch_size,
    )


@lru_cache(maxsize=1)
def get_document_service() -> DocumentService:
    settings = get_settings()
    return DocumentService(settings=settings, embeddings=get_embedding_service())


@lru_cache(maxsize=1)
def get_qa_service() -> QAService:
    settings = get_settings()
    return QAService(settings=settings, embeddings=get_embedding_service())


def reset_service_caches() -> None:
    get_embedding_service.cache_clear()
    get_document_service.cache_clear()
    get_qa_service.cache_clear()


@router.post("/documents", response_model=UploadDocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
) -> UploadDocumentResponse:
    settings = get_settings()
    filename = file.filename or ""
    _ensure_docx(filename)

    raw_bytes = await file.read()
    _ensure_upload_size(raw_bytes, settings.max_upload_mb)

    document_id = str(uuid4())
    document_dir = get_document_dir(settings, document_id)
    source_path = document_dir / "source.docx"

    try:
        await asyncio.to_thread(document_dir.mkdir, parents=True, exist_ok=False)
        await asyncio.to_thread(source_path.write_bytes, raw_bytes)
        record = Document(
            id=document_id,
            filename=filename,
            stored_path=str(source_path),
            status=DocumentStatus.PENDING,
            chunks_path=None,
            embeddings_path=None,
            error=None,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        session.add(record)
        await session.commit()
        background_tasks.add_task(get_document_service().process_document, document_id)
        return UploadDocumentResponse(document_id=document_id, status=DocumentStatus.PENDING)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"Failed to process document: {exc}"
        ) from exc


@router.get("/documents/{document_id}", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: str,
    session: AsyncSession = Depends(get_session),
) -> DocumentStatusResponse:
    record = await session.get(Document, document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document was not found.")
    return DocumentStatusResponse(
        document_id=record.id,
        status=record.status,
        error=record.error if record.status == DocumentStatus.FAILED else None,
    )


@router.post("/questions", response_model=CreateQuestionResponse)
async def create_question(
    payload: CreateQuestionRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> CreateQuestionResponse:
    normalized_question = payload.question.strip()
    if not normalized_question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    document = await session.get(Document, payload.document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document was not found.")
    _ensure_document_ready(document)

    question_id = str(uuid4())
    record = Question(
        id=question_id,
        document_id=payload.document_id,
        question=normalized_question,
        status=QuestionStatus.PENDING,
        created_at=utcnow(),
        updated_at=utcnow(),
    )
    session.add(record)
    await session.commit()

    background_tasks.add_task(get_qa_service().process_question, question_id)
    return CreateQuestionResponse(question_id=question_id)


@router.get("/questions/{question_id}", response_model=QuestionStatusResponse)
async def get_question_status(
    question_id: str,
    session: AsyncSession = Depends(get_session),
) -> QuestionStatusResponse:
    record = await session.get(Question, question_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Question was not found.")
    return QuestionStatusResponse(
        question_id=record.id,
        status=record.status,
        answer=record.answer if record.status == QuestionStatus.COMPLETED else None,
        error=record.error if record.status == QuestionStatus.FAILED else None,
    )
