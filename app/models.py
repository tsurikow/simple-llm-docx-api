from datetime import datetime, UTC
from enum import Enum

from sqlalchemy import Column, DateTime
from sqlmodel import Field, SQLModel


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class QuestionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


def utcnow() -> datetime:
    return datetime.now(UTC)


class Document(SQLModel, table=True):
    __tablename__ = "documents"

    id: str = Field(primary_key=True)
    filename: str
    stored_path: str
    status: DocumentStatus = Field(default=DocumentStatus.PENDING, index=True)
    chunks_path: str | None = Field(default=None)
    embeddings_path: str | None = Field(default=None)
    error: str | None = Field(default=None)
    created_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class Question(SQLModel, table=True):
    __tablename__ = "questions"

    id: str = Field(primary_key=True)
    document_id: str = Field(foreign_key="documents.id", index=True)
    question: str
    status: QuestionStatus = Field(default=QuestionStatus.PENDING, index=True)
    answer: str | None = Field(default=None)
    error: str | None = Field(default=None)
    created_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
