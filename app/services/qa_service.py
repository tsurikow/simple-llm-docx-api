import asyncio
import logging
import threading
from pathlib import Path
from time import perf_counter

import numpy as np
from langchain_openai import ChatOpenAI
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config import Settings
from app.db import get_session_factory
from app.logging_utils import elapsed_since
from app.models import Document, Question, QuestionStatus, utcnow
from app.services.embeddings import EmbeddingService
from app.services.prompts import build_messages
from app.services.retrieval import is_contract_metadata_question, score_top_indices
from app.storage import load_chunks, load_embeddings

logger = logging.getLogger(__name__)


class QAService:
    def __init__(self, settings: Settings, embeddings: EmbeddingService):
        self._settings = settings
        self._embeddings = embeddings
        self._llm: ChatOpenAI | None = None
        self._llm_lock = threading.Lock()

    async def process_question(self, question_id: str) -> None:
        started_at = perf_counter()
        question, document = await self._mark_processing_and_get_entities(question_id)
        if question is None or document is None:
            await self._mark_failed(question_id, "Question or document was not found.")
            return

        try:
            metadata_intent = is_contract_metadata_question(question.question)
            context = await self.retrieve_context(document, question.question)
            answer = await self._generate_answer(question.question, context, metadata_intent)
            await self._set_question_state(
                question_id,
                status=QuestionStatus.COMPLETED,
                answer=answer,
                error=None,
            )
            logger.info(
                "Question processed: question_id=%s document_id=%s elapsed=%.3fs",
                question_id,
                document.id,
                elapsed_since(started_at),
            )
        except Exception as exc:
            logger.exception(
                "Question processing failed: question_id=%s document_id=%s elapsed=%.3fs",
                question_id,
                document.id,
                elapsed_since(started_at),
            )
            await self._mark_failed(question_id, str(exc))

    async def retrieve_context(self, document: Document, user_question: str) -> str:
        started_at = perf_counter()
        chunks, vectors, load_elapsed = await self._load_document_index(document)
        question_vector, embed_elapsed = await self._embed_question(user_question, vectors)

        scoring_started_at = perf_counter()
        top_indices = await asyncio.to_thread(
            score_top_indices,
            chunks,
            vectors,
            question_vector,
            user_question,
            max(1, min(self._settings.retrieval_top_k, len(chunks))),
        )
        scoring_elapsed = elapsed_since(scoring_started_at)

        context = "\n\n".join(chunks[index] for index in top_indices).strip()
        if not context:
            raise ValueError("No relevant context was found.")

        logger.info(
            "Context retrieved: document_id=%s chunks=%s top_k=%s metadata=%s load=%.3fs embed=%.3fs rerank=%.3fs total=%.3fs",
            document.id,
            len(chunks),
            len(top_indices),
            is_contract_metadata_question(user_question),
            load_elapsed,
            embed_elapsed,
            scoring_elapsed,
            elapsed_since(started_at),
        )
        return context

    async def _generate_answer(
        self, user_question: str, context: str, metadata_intent: bool
    ) -> str:
        started_at = perf_counter()
        response = await self._get_llm().ainvoke(
            build_messages(user_question, context, metadata_intent)
        )
        logger.info(
            "LLM answered: model=%s metadata=%s context_chars=%s elapsed=%.3fs",
            self._settings.openrouter_model,
            metadata_intent,
            len(context),
            elapsed_since(started_at),
        )
        if isinstance(response.content, str):
            text = response.content.strip()
            if text:
                return text
        raise RuntimeError("Model returned empty response.")

    def _get_llm(self) -> ChatOpenAI:
        if self._llm is not None:
            return self._llm
        with self._llm_lock:
            if self._llm is None:
                if not self._settings.openrouter_api_key:
                    raise RuntimeError("OPENROUTER_API_KEY is not configured.")
                self._llm = ChatOpenAI(
                    model=self._settings.openrouter_model,
                    api_key=self._settings.openrouter_api_key,
                    base_url=self._settings.openrouter_base_url,
                    temperature=0.0,
                    extra_body={"provider": {"sort": {"by": "latency"}}},
                )
        return self._llm

    async def _load_document_index(
        self, document: Document
    ) -> tuple[list[str], np.ndarray, float]:
        started_at = perf_counter()
        chunks, vectors = await asyncio.gather(
            asyncio.to_thread(load_chunks, Path(document.chunks_path or "")),
            asyncio.to_thread(load_embeddings, Path(document.embeddings_path or "")),
        )
        if not chunks:
            raise ValueError("No chunks available for this document.")
        if vectors.ndim != 2:
            raise ValueError("Invalid embeddings matrix shape.")
        if vectors.shape[0] != len(chunks):
            raise ValueError("Embeddings count does not match chunk count.")
        return chunks, vectors, elapsed_since(started_at)

    async def _embed_question(
        self, user_question: str, vectors: np.ndarray
    ) -> tuple[np.ndarray, float]:
        started_at = perf_counter()
        question_vector = await self._embeddings.embed_query_async(user_question)
        if question_vector.ndim != 1:
            raise ValueError("Invalid question embedding shape.")
        if vectors.shape[1] != question_vector.shape[0]:
            raise ValueError("Embedding dimensions mismatch.")
        return question_vector, elapsed_since(started_at)

    async def _mark_processing_and_get_entities(
        self, question_id: str
    ) -> tuple[Question | None, Document | None]:
        async with get_session_factory()() as session:
            question = await session.get(Question, question_id)
            if question is None:
                return None, None
            document = await session.get(Document, question.document_id)
            await self._apply_question_state(
                session,
                question,
                status=QuestionStatus.PROCESSING,
                answer=None,
                error=None,
                refresh=True,
            )
            return question, document

    async def _mark_failed(self, question_id: str, message: str) -> None:
        await self._set_question_state(
            question_id,
            status=QuestionStatus.FAILED,
            answer=None,
            error=message[:500],
        )

    async def _set_question_state(
        self,
        question_id: str,
        *,
        status: QuestionStatus,
        answer: str | None,
        error: str | None,
    ) -> None:
        async with get_session_factory()() as session:
            question = await session.get(Question, question_id)
            if question is not None:
                await self._apply_question_state(
                    session,
                    question,
                    status=status,
                    answer=answer,
                    error=error,
                )

    @staticmethod
    async def _apply_question_state(
        session: AsyncSession,
        question: Question,
        *,
        status: QuestionStatus,
        answer: str | None,
        error: str | None,
        refresh: bool = False,
    ) -> None:
        question.status = status
        question.answer = answer
        question.error = error
        question.updated_at = utcnow()
        session.add(question)
        await session.commit()
        if refresh:
            await session.refresh(question)
