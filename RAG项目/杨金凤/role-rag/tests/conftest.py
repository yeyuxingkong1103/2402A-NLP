"""pytest 共享 fixture：mock embedder / Chroma collection / reranker / BM25。

全部用 unittest.mock.MagicMock，不加载任何真实模型；常量通过 monkeypatch
固定为默认值，隔离 .env 影响，保证测试确定性。
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# 默认强制 Chroma 模式，隔离 .env 的 USE_MILVUS=true（先覆盖 env 再 import，模块级常量才正确）。
os.environ["USE_MILVUS"] = "false"

# 项目根加入 sys.path，保证 `import rag` / `import ingest` 可用（模型均为懒加载）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402
import retrieval  # noqa: E402

EMBED_DIM = 1024   # BGE-m3 输出维度
N_RESULTS = 20    # fake_collection 固定返回的召回条数


@pytest.fixture(autouse=True)
def _pin_constants(monkeypatch):
    """固定检索常量（RECALL_K/RERANK_TOP_K/RRF_K），隔离 .env。"""
    monkeypatch.setattr(retrieval, "RECALL_K", 20)
    monkeypatch.setattr(retrieval, "RERANK_TOP_K", 4)
    monkeypatch.setattr(retrieval, "RRF_K", 60)
    monkeypatch.setattr(retrieval, "SIMILARITY_THRESHOLD", 0.0)


@pytest.fixture
def fake_embedder():
    """mock embedder：encode 返回固定 512 维假向量，.tolist() 自然可用。"""
    embedder = MagicMock()
    embedder.encode.return_value = np.array([[0.5] * EMBED_DIM])
    return embedder


@pytest.fixture
def fake_collection():
    """mock Chroma collection：query 返回固定 20 条假结果（余弦距离递增）。"""
    ids = [f"p{i}_c0" for i in range(1, N_RESULTS + 1)]
    documents = [f"指南第{i}页内容片段" for i in range(1, N_RESULTS + 1)]
    metadatas = [{"page": i} for i in range(1, N_RESULTS + 1)]
    distances = [round(0.05 * i, 4) for i in range(1, N_RESULTS + 1)]
    collection = MagicMock()
    collection.query.return_value = {
        "ids": [ids],
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }
    return collection


@pytest.fixture
def fake_reranker():
    """mock reranker：predict 返回可控 logits（测试内自行设 return_value）。"""
    return MagicMock()


@pytest.fixture
def fake_keyword_index():
    """mock BM25：返回 (bm25, chunks)，bm25.get_scores 由测试设 return_value。"""
    bm25 = MagicMock()
    chunks = [
        {"id": f"p{i}_c0", "content": f"片段{i}", "page": i}
        for i in range(1, N_RESULTS + 1)
    ]
    return bm25, chunks


@pytest.fixture
def patch_retrieve_deps(
    monkeypatch, fake_embedder, fake_collection, fake_reranker, fake_keyword_index
):
    """把 retrieve 依赖的 4 个懒加载函数替换为 fake，返回各 fake 引用供断言。"""
    bm25, chunks = fake_keyword_index
    monkeypatch.setattr(retrieval, "load_embedder", lambda: fake_embedder)
    monkeypatch.setattr(retrieval, "get_collection", lambda *a, **k: fake_collection)
    monkeypatch.setattr(retrieval, "load_reranker", lambda: fake_reranker)
    monkeypatch.setattr(retrieval, "load_keyword_index", lambda *a, **k: (bm25, chunks))
    # 分词也 mock 掉，避免真实加载 jieba 词典拖慢测试。
    monkeypatch.setattr(retrieval, "_tokenize", lambda text: ["query", "tokens"])
    return {
        "embedder": fake_embedder,
        "collection": fake_collection,
        "reranker": fake_reranker,
        "bm25": bm25,
        "chunks": chunks,
    }


@pytest.fixture
def patch_retrieve_deps_milvus(
    monkeypatch, fake_embedder, fake_reranker, fake_keyword_index
):
    """把 retrieve 依赖替换为 fake，并切到 Milvus 分支（mock milvus_search 而非 get_collection）。"""
    bm25, chunks = fake_keyword_index
    monkeypatch.setattr(retrieval, "USE_MILVUS", True)
    monkeypatch.setattr(retrieval, "load_embedder", lambda: fake_embedder)
    monkeypatch.setattr(retrieval, "load_reranker", lambda: fake_reranker)
    monkeypatch.setattr(retrieval, "load_keyword_index", lambda *a, **k: (bm25, chunks))
    monkeypatch.setattr(retrieval, "_tokenize", lambda text: ["query", "tokens"])
    # Milvus 分支的向量召回：与 fake_collection 等价的归一化 vec_hits。
    vec_hits = [
        {
            "id": f"p{i}_c0",
            "content": f"指南第{i}页内容片段",
            "page": i,
            "similarity": round(1 - 0.05 * i, 4),
        }
        for i in range(1, N_RESULTS + 1)
    ]
    milvus_search = MagicMock(return_value=vec_hits)
    monkeypatch.setattr(retrieval, "milvus_search", milvus_search)
    return {
        "embedder": fake_embedder,
        "reranker": fake_reranker,
        "bm25": bm25,
        "chunks": chunks,
        "milvus_search": milvus_search,
    }
