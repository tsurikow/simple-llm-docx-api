import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path
from time import perf_counter

import numpy as np
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config import Settings
from app.db import get_session_factory
from app.logging_utils import elapsed_since
from app.models import Document, DocumentStatus, utcnow
from app.services.embeddings import EmbeddingsClient
from app.storage import save_chunks, save_embeddings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentArtifacts:
    chunks_path: Path
    embeddings_path: Path


class DocumentService:
    def __init__(self, settings: Settings, embeddings: EmbeddingsClient):
        self._embeddings = embeddings
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def _extract_chunks(self, source_path: Path) -> list[str]:
        pages = Docx2txtLoader(str(source_path)).load()
        full_text = "\n\n".join(
            page.page_content.strip()
            for page in pages
            if page.page_content and page.page_content.strip()
        ).strip()
        if not full_text:
            raise ValueError("Document has no extractable text.")

        chunks = [chunk.strip() for chunk in self._splitter.split_text(full_text)]
        chunks = [chunk for chunk in chunks if chunk]
        if not chunks:
            raise ValueError("Document has no chunks after splitting.")
        return chunks

    @staticmethod
    def _save_artifacts(
        document_dir: Path, chunks: list[str], vectors: np.ndarray
    ) -> DocumentArtifacts:
        chunks_path = document_dir / "chunks.json"
        embeddings_path = document_dir / "embeddings.npy"
        save_chunks(chunks_path, chunks)
        save_embeddings(embeddings_path, vectors)
        return DocumentArtifacts(chunks_path=chunks_path, embeddings_path=embeddings_path)

    async def index_document(
        self, document_id: str, source_path: Path, document_dir: Path
    ) -> DocumentArtifacts:
        started_at = perf_counter()
        chunks = await asyncio.to_thread(self._extract_chunks, source_path)
        prepare_elapsed = elapsed_since(started_at)

        embed_started_at = perf_counter()
        vectors = await self._embeddings.embed_texts_async(chunks)
        embed_elapsed = elapsed_since(embed_started_at)
        if vectors.size == 0:
            raise RuntimeError("Failed to build embeddings for document.")

        save_started_at = perf_counter()
        artifacts = await asyncio.to_thread(self._save_artifacts, document_dir, chunks, vectors)
        save_elapsed = elapsed_since(save_started_at)
        logger.info(
            "Document indexed: document_id=%s chunks=%s prepare=%.3fs embed=%.3fs save=%.3fs total=%.3fs",
            document_id,
            len(chunks),
            prepare_elapsed,
            embed_elapsed,
            save_elapsed,
            elapsed_since(started_at),
        )
        return artifacts

    async def process_document(self, document_id: str) -> None:
        started_at = perf_counter()
        document = await self._set_document_state(
            document_id,
            status=DocumentStatus.PROCESSING,
            error=None,
            refresh=True,
        )
        if document is None:
            logger.error("Document indexing failed: document_id=%s reason=not_found", document_id)
            return

        try:
            artifacts = await self.index_document(
                document_id,
                Path(document.stored_path),
                Path(document.stored_path).parent,
            )
            await self._set_document_state(
                document_id,
                status=DocumentStatus.READY,
                chunks_path=artifacts.chunks_path,
                embeddings_path=artifacts.embeddings_path,
                error=None,
            )
            logger.info(
                "Background document indexing completed: document_id=%s elapsed=%.3fs",
                document_id,
                elapsed_since(started_at),
            )
        except Exception as exc:
            await self._set_document_state(
                document_id,
                status=DocumentStatus.FAILED,
                error=str(exc)[:500],
            )
            logger.exception(
                "Background document indexing failed: document_id=%s elapsed=%.3fs",
                document_id,
                elapsed_since(started_at),
            )

    async def _set_document_state(
        self,
        document_id: str,
        *,
        status: DocumentStatus,
        error: str | None,
        chunks_path: Path | None = None,
        embeddings_path: Path | None = None,
        refresh: bool = False,
    ) -> Document | None:
        async with get_session_factory()() as session:
            document = await session.get(Document, document_id)
            if document is None:
                return None
            await self._apply_document_state(
                session,
                document,
                status=status,
                error=error,
                chunks_path=chunks_path,
                embeddings_path=embeddings_path,
                refresh=refresh,
            )
            return document

    @staticmethod
    async def _apply_document_state(
        session: AsyncSession,
        document: Document,
        *,
        status: DocumentStatus,
        error: str | None,
        chunks_path: Path | None = None,
        embeddings_path: Path | None = None,
        refresh: bool = False,
    ) -> None:
        document.status = status
        document.error = error
        document.updated_at = utcnow()
        if chunks_path is not None:
            document.chunks_path = str(chunks_path)
        if embeddings_path is not None:
            document.embeddings_path = str(embeddings_path)
        session.add(document)
        await session.commit()
        if refresh:
            await session.refresh(document)
