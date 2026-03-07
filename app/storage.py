import json
from pathlib import Path

import numpy as np

from app.config import Settings


def ensure_storage_dirs(settings: Settings) -> None:
    settings.app_data_dir.mkdir(parents=True, exist_ok=True)
    (settings.app_data_dir / "documents").mkdir(parents=True, exist_ok=True)


def get_document_dir(settings: Settings, document_id: str) -> Path:
    return settings.app_data_dir / "documents" / document_id


def save_chunks(path: Path, chunks: list[str]) -> None:
    path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")


def load_chunks(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError("Invalid chunks file format.")
    return [str(item) for item in value]


def save_embeddings(path: Path, embeddings: np.ndarray) -> None:
    np.save(path, embeddings.astype(np.float32))


def load_embeddings(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)

