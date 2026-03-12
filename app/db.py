from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models import Document, DocumentStatus, Question, QuestionStatus, utcnow

_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_db(database_url: str) -> None:
    global _engine, _session_factory
    _engine = create_async_engine(database_url, echo=False)
    _session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with _engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


def reset_engine() -> None:
    global _engine, _session_factory
    _engine = None
    _session_factory = None


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("Database session factory is not initialized.")
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with get_session_factory()() as session:
        yield session


async def mark_questions_as_failed(
    status: QuestionStatus,
    reason: str,
) -> int:
    async with get_session_factory()() as session:
        statement = select(Question).where(Question.status == status)
        records = (await session.exec(statement)).all()
        now = utcnow()
        for item in records:
            item.status = QuestionStatus.FAILED
            item.error = reason
            item.updated_at = now
            session.add(item)
        await session.commit()
        return len(records)


async def mark_documents_as_failed(
    status: DocumentStatus,
    reason: str,
) -> int:
    async with get_session_factory()() as session:
        statement = select(Document).where(Document.status == status)
        records = (await session.exec(statement)).all()
        now = utcnow()
        for item in records:
            item.status = DocumentStatus.FAILED
            item.error = reason
            item.updated_at = now
            session.add(item)
        await session.commit()
        return len(records)
