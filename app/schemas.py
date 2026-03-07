from app.models import QuestionStatus
from pydantic import BaseModel


class UploadDocumentResponse(BaseModel):
    document_id: str


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

