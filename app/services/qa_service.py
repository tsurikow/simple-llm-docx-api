import math
from pathlib import Path

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi
from razdel import tokenize
from sqlmodel import Session

from app.config import Settings
from app.db import get_engine
from app.models import Document, Question, QuestionStatus, utcnow
from app.services.embeddings import EmbeddingService
from app.storage import load_chunks, load_embeddings


class QAService:
    _DENSE_WEIGHT = 0.7
    _BM25_WEIGHT = 0.3

    def __init__(self, settings: Settings, embeddings: EmbeddingService):
        self._settings = settings
        self._embeddings = embeddings
        self._llm: ChatOpenAI | None = None

    def process_question(self, question_id: str) -> None:
        question, document = self._mark_processing_and_get_entities(question_id)
        if question is None or document is None:
            self._mark_failed(question_id, "Question or document was not found.")
            return

        try:
            context = self.retrieve_context(document, question.question)
            answer = self._generate_answer(question.question, context)
            self._mark_completed(question_id, answer)
        except Exception as exc:
            self._mark_failed(question_id, str(exc))

    def retrieve_context(self, document: Document, user_question: str) -> str:
        chunks = load_chunks(Path(document.chunks_path))
        vectors = load_embeddings(Path(document.embeddings_path))
        if not chunks:
            raise ValueError("No chunks available for this document.")
        if vectors.ndim != 2:
            raise ValueError("Invalid embeddings matrix shape.")
        if vectors.shape[0] != len(chunks):
            raise ValueError("Embeddings count does not match chunk count.")

        question_vector = self._embeddings.embed_query(user_question)
        if question_vector.ndim != 1:
            raise ValueError("Invalid question embedding shape.")
        if vectors.shape[1] != question_vector.shape[0]:
            raise ValueError("Embedding dimensions mismatch.")

        dense_scores = vectors @ question_vector
        bm25_scores = self._bm25_scores(chunks=chunks, query=user_question)
        scores = self._hybrid_scores(dense_scores=dense_scores, bm25_scores=bm25_scores)
        top_k = max(1, min(self._settings.retrieval_top_k, len(chunks)))
        ranked_indices = np.argsort(scores)[::-1]
        q = user_question.lower()
        is_contract_meta = ("договор" in q) and (
            ("номер" in q) or ("дата" in q) or ("№" in user_question)
        )

        if is_contract_meta and chunks:
            selected = [0]
            for index in ranked_indices:
                idx = int(index)
                if idx == 0:
                    continue
                selected.append(idx)
                if len(selected) >= top_k:
                    break
            top_indices = np.array(selected[:top_k], dtype=int)
        else:
            top_indices = ranked_indices[:top_k]

        context_chunks = [chunks[index] for index in top_indices]
        context = "\n\n".join(context_chunks).strip()
        if not context:
            raise ValueError("No relevant context was found.")
        return context

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        return [item.text.lower() for item in tokenize(text)]

    @classmethod
    def _bm25_scores(cls, chunks: list[str], query: str) -> np.ndarray:
        query_tokens = cls._tokenize(query)
        if not query_tokens:
            return np.zeros(len(chunks), dtype=np.float32)
        tokenized_chunks = [cls._tokenize(chunk) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        scores = bm25.get_scores(query_tokens)
        return np.asarray(scores, dtype=np.float32)

    @classmethod
    def _normalize_scores(cls, scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        if math.isclose(min_score, max_score):
            return np.zeros_like(scores, dtype=np.float32)
        normalized = (scores - min_score) / (max_score - min_score)
        return normalized.astype(np.float32)

    @classmethod
    def _hybrid_scores(
        cls, dense_scores: np.ndarray, bm25_scores: np.ndarray
    ) -> np.ndarray:
        dense_norm = cls._normalize_scores(dense_scores)
        bm25_norm = cls._normalize_scores(bm25_scores)
        return cls._DENSE_WEIGHT * dense_norm + cls._BM25_WEIGHT * bm25_norm

    def _generate_answer(self, user_question: str, context: str) -> str:
        llm = self._get_llm()
        system_prompt = (
            "Ты помощник по вопросам к документу. Отвечай только на русском языке и "
            "только на основе переданного контекста. Если в контексте нет данных для "
            "точного ответа, прямо скажи, что информации недостаточно."
        )
        human_prompt = (
            f"Контекст документа:\n{context}\n\n"
            f"Вопрос пользователя:\n{user_question}\n\n"
            "Сформируй короткий и точный ответ."
        )
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )
        if isinstance(response.content, str):
            text = response.content.strip()
            if text:
                return text
        raise RuntimeError("Model returned empty response.")

    def _get_llm(self) -> ChatOpenAI:
        if self._llm is not None:
            return self._llm
        if not self._settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured.")
        self._llm = ChatOpenAI(
            model=self._settings.openrouter_model,
            api_key=self._settings.openrouter_api_key,
            base_url=self._settings.openrouter_base_url,
            temperature=0.0,
        )
        return self._llm

    def _mark_processing_and_get_entities(
        self, question_id: str
    ) -> tuple[Question | None, Document | None]:
        with Session(get_engine()) as session:
            question = session.get(Question, question_id)
            if question is None:
                return None, None
            document = session.get(Document, question.document_id)
            question.status = QuestionStatus.PROCESSING
            question.error = None
            question.updated_at = utcnow()
            session.add(question)
            session.commit()
            session.refresh(question)
            if document is not None:
                session.refresh(document)
            return question, document

    def _mark_completed(self, question_id: str, answer: str) -> None:
        with Session(get_engine()) as session:
            question = session.get(Question, question_id)
            if question is None:
                return
            question.status = QuestionStatus.COMPLETED
            question.answer = answer
            question.error = None
            question.updated_at = utcnow()
            session.add(question)
            session.commit()

    def _mark_failed(self, question_id: str, message: str) -> None:
        with Session(get_engine()) as session:
            question = session.get(Question, question_id)
            if question is None:
                return
            question.status = QuestionStatus.FAILED
            question.answer = None
            question.error = message[:500]
            question.updated_at = utcnow()
            session.add(question)
            session.commit()
