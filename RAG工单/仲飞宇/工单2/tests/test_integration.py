"""集成测试：真实 Ollama（embedding + LLM）+ 临时 Milvus/SQL 的端到端链路。

不同于其它单测（LLM/Embedding 用 dummy），这里走真实的 bge-m3 向量化与 Milvus，
验证「解析 → 分块 → 向量化 → 入库 → 混合检索召回」整条离线+在线链路。
Ollama 不可达时自动 skip，不阻塞默认 `pytest -q`。
"""
from __future__ import annotations

import os
import shutil
import socket
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import pytest

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1").removesuffix("/v1")


def _ollama_up() -> bool:
    """只探 TCP 能不能建连，不打任何 HTTP 接口：这里只想确定「有东西在听」，
    模型有没有拉下来、鉴权对不对，是这条用例自己的事（失败时该红，而不是被 skip 掉）。"""
    u = urlparse(OLLAMA_BASE)
    try:
        s = socket.create_connection((u.hostname or "localhost", u.port or 11434), 3)
        s.close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _ollama_up(), reason="Ollama 不可达，跳过集成测试")
def test_end_to_end_ingest_and_retrieve(monkeypatch):
    tmp = Path(tempfile.mkdtemp(prefix="rag-int-"))
    # 走 monkeypatch 而不是直接构造 Settings：load_settings() 每次现读环境变量，所以必须在它
    # 之前设；用这个夹具还能保证用例结束后环境复原——否则后面的用例会跟着连到这个临时库。
    monkeypatch.setenv("MILVUS_DB_URI", str(tmp / "milvus.db"))
    monkeypatch.setenv("SQL_URL", f"sqlite:///{tmp / 'app.db'}")

    from app.core.config import load_settings
    from app.core.document import Chunker, DocumentParser
    from app.core.embedding import EmbeddingClient
    from app.core.retrieve.hybrid_retriever import HybridRetriever
    from app.core.store.milvus_store import MilvusStore
    from app.core.store.sql_store import SQLStore

    settings = load_settings()

    # 用一条现有知识库里没有的「独特事实」，召回它即证明整条链路真实跑通
    doc_file = tmp / "water.md"
    doc_file.write_text("# 饮水\n高血压患者每日饮水量宜控制在 1500 毫升。\n", encoding="utf-8")

    doc = DocumentParser().parse_file(doc_file)
    texts = Chunker().chunk(doc["text"])
    assert texts, "解析后无有效文本"

    embedding = EmbeddingClient(settings)
    milvus = MilvusStore(settings)
    sql = SQLStore(settings)
    sql.connect()

    vecs = embedding.embed_texts(texts)
    milvus.insert(
        "psychologist",
        [
            {"text": t, "title": doc["title"], "source": doc["source"], "chunk_index": i, "summary": "", "vector": v}
            for i, (t, v) in enumerate(zip(texts, vecs))
        ],
    )

    retriever = HybridRetriever(settings, embedding, milvus)
    hits = retriever.retrieve("高血压患者每天应该喝多少水", "psychologist", top_k=3)
    assert hits, "没有召回任何内容"
    assert any("1500" in h["text"] for h in hits), "没召回入库的独特事实"

    shutil.rmtree(tmp, ignore_errors=True)
