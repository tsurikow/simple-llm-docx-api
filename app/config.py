from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    app_data_dir: Path
    openrouter_api_key: str
    openrouter_model: str
    openrouter_base_url: str
    max_upload_mb: int
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    embedding_model_name: str

    @property
    def database_url(self) -> str:
        db_path = self.app_data_dir / "app.db"
        return f"sqlite:///{db_path}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = Path(os.getenv("APP_DATA_DIR", "data")).resolve()
    return Settings(
        app_data_dir=data_dir,
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "qwen/qwen3-8b"),
        openrouter_base_url=os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ),
        max_upload_mb=_env_int("MAX_UPLOAD_MB", 10),
        chunk_size=_env_int("CHUNK_SIZE", 800),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 160),
        retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 4),
        embedding_model_name=os.getenv(
            "EMBEDDING_MODEL_NAME",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
    )


def reset_settings_cache() -> None:
    get_settings.cache_clear()
