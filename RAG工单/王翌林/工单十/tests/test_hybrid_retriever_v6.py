# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
tests/test_hybrid_retriever_v6.py —— 工单六 统一混合检索器单元测试

注入 mock vector_store / embedder 与内存全文索引，不依赖 Milvus/模型，
验证三种检索模式、两种融合、三种重排器的编排逻辑。
"""
import pytest

from src.retrieval.fulltext_retriever import FulltextRetriever
from src.retrieval.hybrid_retriever_v6 import HybridRetrieverV6
from src.retrieval.retrieval_config import (
    FUSION_WEIGHTED, MODE_FULLTEXT, MODE_HYBRID, MODE_VECTOR,
    RERANKER_ADAPTIVE, RERANKER_TFIDF, RetrievalConfig,
)

WORK_ORDER = "人工智能NLP-RAG-混合检索任务"


class FakeEmbedder:
    def encode(self, texts, **kw):
        return [[0.1] * 8 for _ in texts]


class FakeVectorStore:
    """工单六：内存向量库 mock（按 content 含词返回伪余弦分）"""

    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, qvec, top_k=24, doc_id=None, output_fields=None):
        out = []
        for c in self.chunks:
            if doc_id and c["doc_id"] != doc_id:
                continue
            out.append({"doc_id": c["doc_id"], "chunk_id": c["chunk_id"],
                        "content": c["content"], "page": c["page"],
                        "metadata": {}, "score": c.get("vec_score", 0.5)})
        return out[:top_k]


CHUNKS = [
    {"doc_id": "招股说明书1", "chunk_id": "c1", "page": 1,
     "content": "公司来自军用领域的收入为18780万元", "vec_score": 0.81},
    {"doc_id": "招股说明书1", "chunk_id": "c2", "page": 2,
     "content": "本次发行募集资金总额40584万元", "vec_score": 0.72},
    {"doc_id": "招股说明书2", "chunk_id": "c3", "page": 3,
     "content": "武汉力源销售部大客户销售处遍布全国", "vec_score": 0.63},
]


@pytest.fixture
def retriever():
    cfg = RetrievalConfig(reranker=RERANKER_TFIDF)
    r = HybridRetrieverV6(config=cfg, vector_store=FakeVectorStore(CHUNKS),
                          embedder=FakeEmbedder())
    r._fulltext = FulltextRetriever(chunks=CHUNKS)
    return r


class TestModes:
    def test_vector_mode(self, retriever):
        """工单六：vector 模式只走向量通道"""
        out = retriever.retrieve("军用收入", config=RetrievalConfig(
            mode=MODE_VECTOR, reranker=RERANKER_TFIDF))
        assert out["mode"] == "vector"
        assert out["vector_hits"] == 3
        assert out["fulltext_hits"] == 0
        assert len(out["results"]) > 0

    def test_fulltext_mode(self, retriever):
        """工单六：fulltext 模式只走全文通道"""
        out = retriever.retrieve("军用 收入", config=RetrievalConfig(
            mode=MODE_FULLTEXT, reranker=RERANKER_TFIDF, match="and"))
        assert out["mode"] == "fulltext"
        assert out["vector_hits"] == 0
        assert out["fulltext_hits"] >= 1
        assert out["results"][0]["chunk_id"] == "c1"

    def test_hybrid_mode_union(self, retriever):
        """工单六：hybrid 模式两路并集召回"""
        out = retriever.retrieve("军用收入", config=RetrievalConfig(
            mode=MODE_HYBRID, reranker=RERANKER_TFIDF))
        assert out["mode"] == "hybrid"
        assert out["vector_hits"] == 3
        assert out["fulltext_hits"] >= 1

    def test_doc_id_filter_vector(self, retriever):
        out = retriever.retrieve("军用收入", doc_id="招股说明书1",
                                 config=RetrievalConfig(
                                     mode=MODE_VECTOR,
                                     reranker=RERANKER_TFIDF))
        assert all(h["doc_id"] == "招股说明书1" for h in out["results"])


class TestFusionAndReranker:
    def test_weighted_fusion(self, retriever):
        out = retriever.retrieve("军用收入", config=RetrievalConfig(
            mode=MODE_HYBRID, fusion=FUSION_WEIGHTED,
            reranker=RERANKER_TFIDF, vector_weight=0.7,
            fulltext_weight=0.3))
        assert out["fusion"] == "weighted"
        assert len(out["results"]) > 0

    def test_adaptive_reranker_runs(self, retriever, tmp_path, monkeypatch):
        """工单六：自适应重排器在混合模式下可运行（无反馈冷启动）"""
        monkeypatch.chdir(tmp_path)
        out = retriever.retrieve("军用收入", config=RetrievalConfig(
            mode=MODE_HYBRID, reranker=RERANKER_ADAPTIVE))
        assert out["reranker"] == "adaptive"
        assert len(out["results"]) > 0


class TestMeta:
    def test_result_has_timing(self, retriever):
        out = retriever.retrieve("军用收入", config=RetrievalConfig(
            mode=MODE_HYBRID, reranker=RERANKER_TFIDF))
        assert out["elapsed_ms"] >= 0
        assert out["vector_weight"] + out["fulltext_weight"] == pytest.approx(1.0)
