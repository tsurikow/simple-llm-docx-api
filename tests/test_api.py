import asyncio
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from app.db import get_session_factory, mark_documents_as_failed, mark_questions_as_failed
from app.models import Document, DocumentStatus, Question, QuestionStatus, utcnow
from app.services.document_service import DocumentArtifacts, DocumentService
from app.services.qa_service import QAService
from app.storage import save_chunks, save_embeddings


async def _mock_index_document(
    _: DocumentService, document_id: str, source_path: Path, document_dir: Path
) -> DocumentArtifacts:
    _ = document_id, source_path
    chunks_path = document_dir / "chunks.json"
    embeddings_path = document_dir / "embeddings.npy"
    save_chunks(chunks_path, ["Это тестовый контекст по документу."])
    save_embeddings(embeddings_path, np.array([[1.0, 0.0]], dtype=np.float32))
    return DocumentArtifacts(chunks_path=chunks_path, embeddings_path=embeddings_path)


def _upload_document(client) -> dict:
    response = client.post(
        "/documents",
        files={
            "file": (
                "sample.docx",
                b"fake-docx-content",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert response.status_code == 200
    return response.json()


def _ask_question(client, document_id: str, question: str) -> dict:
    response = client.post(
        "/questions",
        json={"document_id": document_id, "question": question},
    )
    assert response.status_code == 200
    return response.json()


async def _get_document(document_id: str) -> Document | None:
    async with get_session_factory()() as session:
        return await session.get(Document, document_id)


async def _get_question(question_id: str) -> Question | None:
    async with get_session_factory()() as session:
        return await session.get(Question, question_id)


async def _insert_question(status: QuestionStatus) -> str:
    async with get_session_factory()() as session:
        document = Document(
            id=str(uuid4()),
            filename="sample.docx",
            stored_path="/tmp/source.docx",
            status=DocumentStatus.READY,
            chunks_path="/tmp/chunks.json",
            embeddings_path="/tmp/embeddings.npy",
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        question = Question(
            id=str(uuid4()),
            document_id=document.id,
            question="Вопрос",
            status=status,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        session.add(document)
        session.add(question)
        await session.commit()
        return question.id


async def _insert_document(status: DocumentStatus) -> str:
    async with get_session_factory()() as session:
        document = Document(
            id=str(uuid4()),
            filename="sample.docx",
            stored_path="/tmp/source.docx",
            status=status,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        session.add(document)
        await session.commit()
        return document.id


def test_upload_rejects_non_docx(client):
    test_client, _ = client
    response = test_client.post(
        "/documents",
        files={"file": ("bad.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_happy_path_upload_poll_ask_poll(client, monkeypatch):
    test_client, data_dir = client
    monkeypatch.setattr(DocumentService, "index_document", _mock_index_document)
    monkeypatch.setattr(
        QAService,
        "retrieve_context",
        lambda *_args, **_kwargs: asyncio.sleep(0, result="контекст"),
    )
    monkeypatch.setattr(
        QAService,
        "_generate_answer",
        lambda *_args, **_kwargs: asyncio.sleep(0, result="Готовый ответ"),
    )

    payload = _upload_document(test_client)
    document_id = payload["document_id"]

    assert payload["status"] == DocumentStatus.PENDING
    assert (data_dir / "documents" / document_id / "source.docx").exists()
    assert test_client.get(f"/documents/{document_id}").json()["status"] == DocumentStatus.READY

    question_id = _ask_question(test_client, document_id, "О чем документ?")["question_id"]
    result = test_client.get(f"/questions/{question_id}").json()

    assert result["status"] == QuestionStatus.COMPLETED
    assert result["answer"] == "Готовый ответ"

    stored = asyncio.run(_get_document(document_id))
    assert stored is not None
    assert stored.status == DocumentStatus.READY


def test_question_requires_ready_document(client, monkeypatch):
    test_client, _ = client

    async def _noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(DocumentService, "process_document", _noop)
    payload = _upload_document(test_client)

    response = test_client.post(
        "/questions",
        json={"document_id": payload["document_id"], "question": "Что в документе?"},
    )

    assert response.status_code == 409
    assert "still being indexed" in response.json()["detail"]


def test_llm_error_sets_failed_status(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setattr(DocumentService, "index_document", _mock_index_document)
    monkeypatch.setattr(
        QAService,
        "retrieve_context",
        lambda *_args, **_kwargs: asyncio.sleep(0, result="контекст"),
    )

    async def _raise(*_args, **_kwargs) -> str:
        raise RuntimeError("OpenRouter error")

    monkeypatch.setattr(QAService, "_generate_answer", _raise)

    document_id = _upload_document(test_client)["document_id"]
    question_id = _ask_question(test_client, document_id, "Сформулируй вывод.")["question_id"]
    payload = test_client.get(f"/questions/{question_id}").json()

    assert payload["status"] == QuestionStatus.FAILED
    assert "OpenRouter error" in (payload["error"] or "")


@pytest.mark.parametrize(
    ("kind", "status", "marker", "reason"),
    [
        ("question", QuestionStatus.PENDING, mark_questions_as_failed, "Service restarted before background task started."),
        ("question", QuestionStatus.PROCESSING, mark_questions_as_failed, "Service restarted before completion."),
        ("document", DocumentStatus.PENDING, mark_documents_as_failed, "Service restarted before background task started."),
        ("document", DocumentStatus.PROCESSING, mark_documents_as_failed, "Service restarted before completion."),
    ],
)
def test_restart_recovery_marks_in_flight_records_failed(client, kind, status, marker, reason):
    if kind == "question":
        record_id = asyncio.run(_insert_question(status))
        changed = asyncio.run(marker(status, reason))
        stored = asyncio.run(_get_question(record_id))
        assert stored is not None
        assert stored.status == QuestionStatus.FAILED
    else:
        record_id = asyncio.run(_insert_document(status))
        changed = asyncio.run(marker(status, reason))
        stored = asyncio.run(_get_document(record_id))
        assert stored is not None
        assert stored.status == DocumentStatus.FAILED

    assert changed == 1
    assert stored.error == reason
