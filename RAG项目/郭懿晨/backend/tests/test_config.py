from pathlib import Path

import pytest

from backend.app.config import AppSettings, ensure_project_path


def test_settings_defaults_use_constitution_paths():
    settings = AppSettings()

    assert settings.data_dir == Path("data")
    assert settings.qdrant_path == Path("data/qdrant")
    assert settings.qdrant_collection == "rag_documents"
    assert settings.ollama_model == "deepseek-r1:7b"


def test_ensure_project_path_allows_data_dir(tmp_path):
    root = tmp_path
    target = root / "data" / "file.pdf"

    assert ensure_project_path(root, target) == target.resolve()


def test_ensure_project_path_rejects_outside_path(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.pdf"

    with pytest.raises(ValueError, match="项目目录之外"):
        ensure_project_path(root, outside)


def test_settings_rerank_defaults():
    settings = AppSettings()

    assert settings.rerank_enabled is True
    assert settings.retrieval_candidate_k == 30
    assert settings.coarse_top_k == 10
    assert settings.final_top_k == 4
    assert settings.coarse_scorer == "dense_cosine"
    assert settings.min_retrieval_score == 0.2
