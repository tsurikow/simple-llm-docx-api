from collections.abc import Generator

from sqlmodel import Session, SQLModel, create_engine, select

from app.models import Question, QuestionStatus, utcnow

_engine = None


def init_db(database_url: str) -> None:
    global _engine
    _engine = create_engine(database_url, connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(_engine)


def reset_engine() -> None:
    global _engine
    _engine = None


def get_engine():
    if _engine is None:
        raise RuntimeError("Database engine is not initialized.")
    return _engine


def get_session() -> Generator[Session, None, None]:
    with Session(get_engine()) as session:
        yield session


def mark_processing_questions_as_failed(reason: str) -> int:
    with Session(get_engine()) as session:
        statement = select(Question).where(Question.status == QuestionStatus.PROCESSING)
        records = session.exec(statement).all()
        now = utcnow()
        for item in records:
            item.status = QuestionStatus.FAILED
            item.error = reason
            item.updated_at = now
            session.add(item)
        session.commit()
        return len(records)

