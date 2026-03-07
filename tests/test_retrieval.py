from pathlib import Path

import numpy as np

from app.config import Settings
from app.models import Document
from app.services.qa_service import QAService
from app.storage import save_chunks, save_embeddings


class _DummyEmbeddings:
    def embed_query(self, _: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


def _make_settings(tmp_path: Path, top_k: int) -> Settings:
    return Settings(
        app_data_dir=tmp_path,
        openrouter_api_key="test",
        openrouter_model="qwen/qwen3-8b",
        openrouter_base_url="https://openrouter.ai/api/v1",
        max_upload_mb=10,
        chunk_size=900,
        chunk_overlap=150,
        retrieval_top_k=top_k,
        embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )


def _make_service(tmp_path: Path, top_k: int) -> QAService:
    settings = _make_settings(tmp_path=tmp_path, top_k=top_k)
    service = QAService(settings=settings, embeddings=_DummyEmbeddings())  # type: ignore[arg-type]
    return service


def _make_document(tmp_path: Path, chunks: list[str], vectors: np.ndarray) -> Document:
    chunks_path = tmp_path / "chunks.json"
    embeddings_path = tmp_path / "embeddings.npy"
    save_chunks(chunks_path, chunks)
    save_embeddings(embeddings_path, vectors)
    return Document(
        id="doc",
        filename="doc.docx",
        stored_path=str(tmp_path / "source.docx"),
        chunks_path=str(chunks_path),
        embeddings_path=str(embeddings_path),
    )


def _retrieve(
    tmp_path: Path, *, top_k: int, chunks: list[str], vectors: np.ndarray, question: str
) -> str:
    service = _make_service(tmp_path=tmp_path, top_k=top_k)
    document = _make_document(tmp_path=tmp_path, chunks=chunks, vectors=vectors)
    return service.retrieve_context(document, question)


def test_hybrid_retrieval_uses_bm25_when_dense_scores_are_equal(tmp_path: Path):
    chunks = [
        "Оплата производится до 10 числа месяца.",
        "Штраф за просрочку поставки составляет 0.1% в день.",
        "Срок действия договора составляет 12 месяцев.",
    ]
    vectors = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    context = _retrieve(
        tmp_path,
        top_k=2,
        chunks=chunks,
        vectors=vectors,
        question="Какие штрафы за просрочку?",
    )
    first_chunk = context.split("\n\n")[0]
    assert "Штраф за просрочку поставки" in first_chunk
    assert len(context.split("\n\n")) == 2


def test_hybrid_retrieval_respects_existing_top_k(tmp_path: Path):
    chunks = [
        "Предмет договора: разработка ПО.",
        "Ответственность сторон и неустойка.",
    ]
    vectors = np.array(
        [
            [0.6, 0.0],
            [0.5, 0.0],
        ],
        dtype=np.float32,
    )
    context = _retrieve(
        tmp_path,
        top_k=1,
        chunks=chunks,
        vectors=vectors,
        question="Какой предмет договора?",
    )
    assert "\n\n" not in context


def test_contract_meta_question_always_includes_first_chunk(tmp_path: Path):
    chunks = [
        "Договор № PC-P/1232-1231-3/18",
        "Дата подписания: 12.10.2024",
        "Ответственность сторон и неустойка.",
    ]
    vectors = np.array(
        [
            [0.1, 0.0],
            [1.0, 0.0],
            [0.2, 0.0],
        ],
        dtype=np.float32,
    )
    context = _retrieve(
        tmp_path,
        top_k=2,
        chunks=chunks,
        vectors=vectors,
        question="Какой номер и дата договора?",
    )
    parts = context.split("\n\n")
    assert parts[0] == "Договор № PC-P/1232-1231-3/18"
    assert len(parts) == 2


def test_contract_meta_question_with_top_k_one_returns_only_first_chunk(tmp_path: Path):
    chunks = [
        "Договор № PC-P/1232-1231-3/18",
        "Дата подписания: 12.10.2024",
    ]
    vectors = np.array(
        [
            [0.1, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    context = _retrieve(
        tmp_path,
        top_k=1,
        chunks=chunks,
        vectors=vectors,
        question="Какая дата договора?",
    )
    assert context == "Договор № PC-P/1232-1231-3/18"


def test_non_meta_question_does_not_force_first_chunk(tmp_path: Path):
    chunks = [
        "Договор № PC-P/1232-1231-3/18",
        "Ответственность сторон и неустойка.",
    ]
    vectors = np.array(
        [
            [0.1, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    context = _retrieve(
        tmp_path,
        top_k=1,
        chunks=chunks,
        vectors=vectors,
        question="Какая ответственность сторон?",
    )
    assert context == "Ответственность сторон и неустойка."
