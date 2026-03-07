from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.services.embeddings import EmbeddingService
from app.storage import save_chunks, save_embeddings


@dataclass(frozen=True)
class DocumentArtifacts:
    chunks_path: Path
    embeddings_path: Path


class DocumentService:
    def __init__(self, settings: Settings, embeddings: EmbeddingService):
        self._settings = settings
        self._embeddings = embeddings
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def index_document(
        self, document_id: str, source_path: Path, document_dir: Path
    ) -> DocumentArtifacts:
        loader = Docx2txtLoader(str(source_path))
        pages = loader.load()
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

        vectors = self._embeddings.embed_texts(chunks)
        if vectors.size == 0:
            raise RuntimeError("Failed to build embeddings for document.")

        chunks_path = document_dir / "chunks.json"
        embeddings_path = document_dir / "embeddings.npy"
        save_chunks(chunks_path, chunks)
        save_embeddings(embeddings_path, vectors)
        return DocumentArtifacts(
            chunks_path=chunks_path,
            embeddings_path=embeddings_path,
        )

