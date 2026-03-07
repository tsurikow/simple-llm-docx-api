from pathlib import Path
from uuid import uuid4

import numpy as np
from sqlmodel import Session, select

from app.db import get_engine, mark_processing_questions_as_failed
from app.models import Document, Question, QuestionStatus, utcnow
from app.services.document_service import DocumentArtifacts, DocumentService
from app.services.qa_service import QAService
from app.storage import save_chunks, save_embeddings


def _mock_index_document(
    _: DocumentService, document_id: str, source_path: Path, document_dir: Path
) -> DocumentArtifacts:
    _ = document_id
    _ = source_path
    chunks_path = document_dir / "chunks.json"
    embeddings_path = document_dir / "embeddings.npy"
    save_chunks(chunks_path, ["Это тестовый контекст по документу."])
    save_embeddings(embeddings_path, np.array([[1.0, 0.0]], dtype=np.float32))
    return DocumentArtifacts(chunks_path=chunks_path, embeddings_path=embeddings_path)


def _upload_document(client) -> str:
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
    return response.json()["document_id"]


def test_upload_docx_success(client, monkeypatch):
    test_client, data_dir = client
    monkeypatch.setattr(DocumentService, "index_document", _mock_index_document)

    document_id = _upload_document(test_client)
    document_dir = data_dir / "documents" / document_id
    assert (document_dir / "source.docx").exists()
    assert (document_dir / "chunks.json").exists()
    assert (document_dir / "embeddings.npy").exists()

    with Session(get_engine()) as session:
        stored = session.get(Document, document_id)
        assert stored is not None


def test_upload_rejects_non_docx(client):
    test_client, _ = client
    response = test_client.post(
        "/documents",
        files={"file": ("bad.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_question_for_unknown_document_returns_404(client):
    test_client, _ = client
    response = test_client.post(
        "/questions",
        json={"document_id": str(uuid4()), "question": "Что в документе?"},
    )
    assert response.status_code == 404


def test_happy_path_upload_ask_poll(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setattr(DocumentService, "index_document", _mock_index_document)
    monkeypatch.setattr(QAService, "retrieve_context", lambda *_: "контекст")
    monkeypatch.setattr(QAService, "_generate_answer", lambda *_: "Готовый ответ")

    document_id = _upload_document(test_client)
    ask_response = test_client.post(
        "/questions",
        json={"document_id": document_id, "question": "О чем документ?"},
    )
    assert ask_response.status_code == 200
    question_id = ask_response.json()["question_id"]

    status_response = test_client.get(f"/questions/{question_id}")
    assert status_response.status_code == 200
    payload = status_response.json()
    assert payload["status"] == QuestionStatus.COMPLETED
    assert payload["answer"] == "Готовый ответ"


def test_llm_error_sets_failed_status(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setattr(DocumentService, "index_document", _mock_index_document)
    monkeypatch.setattr(QAService, "retrieve_context", lambda *_: "контекст")

    def _raise(_: QAService, __: str, ___: str) -> str:
        raise RuntimeError("OpenRouter error")

    monkeypatch.setattr(QAService, "_generate_answer", _raise)

    document_id = _upload_document(test_client)
    ask_response = test_client.post(
        "/questions",
        json={"document_id": document_id, "question": "Сформулируй вывод."},
    )
    assert ask_response.status_code == 200
    question_id = ask_response.json()["question_id"]

    status_response = test_client.get(f"/questions/{question_id}")
    assert status_response.status_code == 200
    payload = status_response.json()
    assert payload["status"] == QuestionStatus.FAILED
    assert "OpenRouter error" in (payload["error"] or "")


def test_restart_marks_processing_as_failed(client):
    _, _ = client
    with Session(get_engine()) as session:
        document = Document(
            id=str(uuid4()),
            filename="sample.docx",
            stored_path="/tmp/source.docx",
            chunks_path="/tmp/chunks.json",
            embeddings_path="/tmp/embeddings.npy",
            created_at=utcnow(),
        )
        session.add(document)
        session.add(
            Question(
                id=str(uuid4()),
                document_id=document.id,
                question="Вопрос",
                status=QuestionStatus.PROCESSING,
                created_at=utcnow(),
                updated_at=utcnow(),
            )
        )
        session.commit()

        processing_question = session.exec(
            select(Question).where(Question.status == QuestionStatus.PROCESSING)
        ).first()
        assert processing_question is not None
        question_id = processing_question.id

    changed = mark_processing_questions_as_failed("Service restarted.")
    assert changed == 1

    with Session(get_engine()) as session:
        stored = session.get(Question, question_id)
        assert stored is not None
        assert stored.status == QuestionStatus.FAILED
        assert stored.error == "Service restarted."
