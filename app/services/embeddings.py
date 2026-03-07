import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        values = self._get_model().encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(values, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        value = self._get_model().encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(value, dtype=np.float32)

