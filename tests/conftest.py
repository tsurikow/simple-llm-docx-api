from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("APP_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "qwen/qwen3-8b")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    from app.api import reset_service_caches
    from app.config import reset_settings_cache
    from app.db import reset_engine
    from app.main import create_app

    reset_settings_cache()
    reset_service_caches()
    reset_engine()
    app = create_app()

    with TestClient(app) as test_client:
        yield test_client, data_dir
