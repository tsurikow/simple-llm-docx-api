from app.models import DocumentStatus, QuestionStatus
from pydantic import BaseModel


class UploadDocumentResponse(BaseModel):
    document_id: str
    status: DocumentStatus


class DocumentStatusResponse(BaseModel):
    document_id: str
    status: DocumentStatus
    error: str | None = None


class CreateQuestionRequest(BaseModel):
    document_id: str
    question: str


class CreateQuestionResponse(BaseModel):
    question_id: str


class QuestionStatusResponse(BaseModel):
    question_id: str
    status: QuestionStatus
    answer: str | None = None
    error: str | None = None
