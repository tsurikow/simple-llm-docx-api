import asyncio
from pathlib import Path

import numpy as np

from app.config import Settings
from app.models import Document
from app.services.qa_service import QAService
from app.storage import save_chunks, save_embeddings


class _DummyEmbeddings:
    async def embed_query_async(self, _: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


def _make_service(tmp_path: Path, top_k: int) -> QAService:
    settings = Settings(
        app_data_dir=tmp_path,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        openrouter_api_key="test",
        openrouter_model="qwen/qwen3-8b",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_embedding_model="baai/bge-m3",
        embedding_batch_size=64,
        max_upload_mb=10,
        chunk_size=900,
        chunk_overlap=150,
        retrieval_top_k=top_k,
    )
    return QAService(settings=settings, embeddings=_DummyEmbeddings())  # type: ignore[arg-type]


def _retrieve(
    tmp_path: Path, *, top_k: int, chunks: list[str], vectors: np.ndarray, question: str
) -> str:
    chunks_path = tmp_path / "chunks.json"
    embeddings_path = tmp_path / "embeddings.npy"
    save_chunks(chunks_path, chunks)
    save_embeddings(embeddings_path, vectors)
    document = Document(
        id="doc",
        filename="doc.docx",
        stored_path=str(tmp_path / "source.docx"),
        chunks_path=str(chunks_path),
        embeddings_path=str(embeddings_path),
    )
    return asyncio.run(_make_service(tmp_path, top_k).retrieve_context(document, question))


def test_hybrid_retrieval_prefers_lexical_match_when_dense_scores_are_equal(tmp_path: Path):
    context = _retrieve(
        tmp_path,
        top_k=1,
        chunks=[
            "Оплата производится до 10 числа месяца.",
            "Штраф за просрочку поставки составляет 0.1% в день.",
            "Срок действия договора составляет 12 месяцев.",
        ],
        vectors=np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        question="Какие штрафы за просрочку?",
    )

    assert context == "Штраф за просрочку поставки составляет 0.1% в день."


def test_metadata_question_includes_first_chunk(tmp_path: Path):
    context = _retrieve(
        tmp_path,
        top_k=2,
        chunks=[
            "Договор № TEST-001/2025",
            "Дата подписания: 15.02.2025",
            "Ответственность сторон и неустойка.",
        ],
        vectors=np.array([[0.1, 0.0], [1.0, 0.0], [0.2, 0.0]], dtype=np.float32),
        question="Какой номер и дата договора?",
    )

    assert context.split("\n\n")[0] == "Договор № TEST-001/2025"


def test_metadata_retrieval_prefers_matching_appendix_date(tmp_path: Path):
    context = _retrieve(
        tmp_path,
        top_k=2,
        chunks=[
            "Договор № TEST-001/2025",
            "Общие положения договора.",
            "Приложение №1\nк договору № ALT-999/2024\nот 10.01.2024",
            "Приложение №2\nк договору № TEST-001/2025\nот «15» февраля 2025 г.",
        ],
        vectors=np.array([[0.2, 0.0], [1.0, 0.0], [0.3, 0.0], [0.1, 0.0]], dtype=np.float32),
        question="Какой номер и дата договора?",
    )

    assert context.split("\n\n")[1] == "Приложение №2\nк договору № TEST-001/2025\nот «15» февраля 2025 г."
