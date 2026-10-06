from pathlib import Path

import pytest

from backend.app.config import AppSettings
from backend.app.models import Chunk
from backend.app.rerank import FineReranker, _load_flag_reranker, _resolve_reranker_dir
from backend.app.vector_store import SearchResult


def _skip_if_model_missing(settings: AppSettings) -> None:
    if not settings.reranker_model_path.exists():
        pytest.skip("bge-reranker-v2-m3 模型不存在，跳过真实模型测试")


def test_resolve_real_model_path_finds_config():
    settings = AppSettings()
    _skip_if_model_missing(settings)

    resolved = _resolve_reranker_dir(settings.reranker_model_path)

    assert (resolved / "config.json").exists()
    assert (resolved / "model.safetensors").exists()


def test_real_model_loads_and_scores():
    settings = AppSettings()
    _skip_if_model_missing(settings)

    fine = FineReranker(settings.reranker_model_path, batch_size=8, top_k=1)
    result = SearchResult(
        chunk=Chunk(
            chunk_id="c1",
            document_id="doc-1",
            page=7,
            category="正文",
            text="本标准规定了电气设备中六氟化硫（SF6）气体的现场检测、回收、净化、回充全流程循环再利用方法。",
            source_span="page=7:block=1",
        ),
        score=0.0,
        dense=None,
    )

    kept = fine.rerank("本标准适用于什么场景？", [result])

    assert len(kept) == 1
    assert 0.0 <= kept[0].score <= 1.0
