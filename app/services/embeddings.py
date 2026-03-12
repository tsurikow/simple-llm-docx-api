import logging
from typing import Protocol

import numpy as np
from openai import APIError, AsyncOpenAI


logger = logging.getLogger(__name__)


class EmbeddingsClient(Protocol):
    async def embed_texts_async(self, texts: list[str]) -> np.ndarray: ...

    async def embed_query_async(self, query: str) -> np.ndarray: ...


class EmbeddingService:
    def __init__(self, api_key: str, base_url: str, model: str, batch_size: int):
        self._model = model
        self._batch_size = max(1, batch_size)
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    async def embed_texts_async(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = await self._create_embeddings(batch)
            vectors.extend(item.embedding for item in response.data)

        return np.asarray(vectors, dtype=np.float32)

    async def embed_query_async(self, query: str) -> np.ndarray:
        response = await self._create_embeddings([query])
        return np.asarray(response.data[0].embedding, dtype=np.float32)

    async def _create_embeddings(self, inputs: list[str]):
        try:
            return await self._client.embeddings.create(
                model=self._model,
                input=inputs,
                encoding_format="float",
            )
        except APIError as exc:
            message = getattr(exc, "message", None) or str(exc)
            logger.warning("OpenRouter embeddings request failed: %s", message)
            raise RuntimeError(f"Embedding request failed: {message}") from exc
