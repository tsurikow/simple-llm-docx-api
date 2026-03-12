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
    database_url: str
    openrouter_api_key: str
    openrouter_model: str
    openrouter_base_url: str
    openrouter_embedding_model: str
    embedding_batch_size: int
    max_upload_mb: int
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = Path(os.getenv("APP_DATA_DIR", "data")).resolve()
    default_db_url = f"sqlite+aiosqlite:///{(data_dir / 'app.db').as_posix()}"
    return Settings(
        app_data_dir=data_dir,
        database_url=os.getenv("DATABASE_URL", default_db_url),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "qwen/qwen3-8b"),
        openrouter_base_url=os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ),
        openrouter_embedding_model=os.getenv(
            "OPENROUTER_EMBEDDING_MODEL", "baai/bge-m3"
        ),
        embedding_batch_size=_env_int("EMBEDDING_BATCH_SIZE", 64),
        max_upload_mb=_env_int("MAX_UPLOAD_MB", 10),
        chunk_size=_env_int("CHUNK_SIZE", 750),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 200),
        retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 6),
    )


def reset_settings_cache() -> None:
    get_settings.cache_clear()
