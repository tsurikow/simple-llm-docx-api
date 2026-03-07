from functools import lru_cache
import shutil
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlmodel import Session

from app.config import get_settings
from app.db import get_session
from app.models import Document, Question, QuestionStatus, utcnow
from app.schemas import (
    CreateQuestionRequest,
    CreateQuestionResponse,
    QuestionStatusResponse,
    UploadDocumentResponse,
)
from app.services.document_service import DocumentService
from app.services.embeddings import EmbeddingService
from app.services.qa_service import QAService
from app.storage import get_document_dir

router = APIRouter()


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    settings = get_settings()
    return EmbeddingService(settings.embedding_model_name)


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
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> UploadDocumentResponse:
    settings = get_settings()
    filename = file.filename or ""
    if not filename.lower().endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are supported.")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    max_size_bytes = settings.max_upload_mb * 1024 * 1024
    if len(raw_bytes) > max_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File is too large. Max size is {settings.max_upload_mb} MB.",
        )

    document_id = str(uuid4())
    document_dir = get_document_dir(settings, document_id)
    source_path = document_dir / "source.docx"

    try:
        document_dir.mkdir(parents=True, exist_ok=False)
        source_path.write_bytes(raw_bytes)
        artifacts = get_document_service().index_document(
            document_id=document_id,
            source_path=source_path,
            document_dir=document_dir,
        )

        record = Document(
            id=document_id,
            filename=filename,
            stored_path=str(source_path),
            chunks_path=str(artifacts.chunks_path),
            embeddings_path=str(artifacts.embeddings_path),
            created_at=utcnow(),
        )
        session.add(record)
        session.commit()
        return UploadDocumentResponse(document_id=document_id)
    except ValueError as exc:
        if document_dir.exists():
            shutil.rmtree(document_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        if document_dir.exists():
            shutil.rmtree(document_dir, ignore_errors=True)
        raise
    except Exception as exc:  # noqa: BLE001
        if document_dir.exists():
            shutil.rmtree(document_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to process document: {exc}"
        ) from exc


@router.post("/questions", response_model=CreateQuestionResponse)
def create_question(
    payload: CreateQuestionRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
) -> CreateQuestionResponse:
    normalized_question = payload.question.strip()
    if not normalized_question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    document = session.get(Document, payload.document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document was not found.")

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
    session.commit()

    background_tasks.add_task(get_qa_service().process_question, question_id)
    return CreateQuestionResponse(question_id=question_id)


@router.get("/questions/{question_id}", response_model=QuestionStatusResponse)
def get_question_status(
    question_id: str,
    session: Session = Depends(get_session),
) -> QuestionStatusResponse:
    record = session.get(Question, question_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Question was not found.")
    return QuestionStatusResponse(
        question_id=record.id,
        status=record.status,
        answer=record.answer if record.status == QuestionStatus.COMPLETED else None,
        error=record.error if record.status == QuestionStatus.FAILED else None,
    )
