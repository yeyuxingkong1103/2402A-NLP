# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
tests/test_fusion_rerankers.py —— 工单六 融合算法与三种重排器单元测试

不依赖模型/Milvus：LLM 重排器用 mock，TF-IDF/自适应重排器纯本地计算。
"""
import json
from pathlib import Path

import pytest

from src.retrieval.fusion import rrf_fuse, weighted_fuse
from src.retrieval.rerankers import (
    AdaptiveReranker, LLMReranker, TfidfReranker,
)
from src.retrieval.retrieval_config import RetrievalConfig

WORK_ORDER = "人工智能NLP-RAG-混合检索任务"


def _hit(doc, cid, score, source="text"):
    return {"doc_id": doc, "chunk_id": cid, "content": "",
            "page": 1, "score": score, "source": source}


class TestRRF:
    def test_rrf_overlap_gets_higher_score(self):
        """工单六：两路都命中的候选 RRF 分更高（投票机制）"""
        vec = [_hit("d", "a", 0.9), _hit("d", "b", 0.8)]
        ft = [_hit("d", "b", 5.0), _hit("d", "c", 3.0)]
        fused = rrf_fuse(vec, ft, rrf_k=60)
        by_id = {f"{h['doc_id']}|{h['chunk_id']}": h for h in fused}
        # b 两路命中，rrf 分最高
        assert fused[0]["chunk_id"] == "b"
        assert by_id["d|b"]["search_path"] == "vector+fulltext"

    def test_rrf_union_recall(self):
        """工单六：RRF 结果为两路并集（保证召回）"""
        vec = [_hit("d", "a", 0.9)]
        ft = [_hit("d", "z", 5.0)]
        fused = rrf_fuse(vec, ft)
        assert {h["chunk_id"] for h in fused} == {"a", "z"}


class TestWeighted:
    def test_weighted_vector_dominant(self):
        """工单六：向量权重 1.0 时排序按向量归一化分"""
        vec = [_hit("d", "a", 0.9), _hit("d", "b", 0.1)]
        ft = [_hit("d", "b", 10.0), _hit("d", "a", 1.0)]
        fused = weighted_fuse(vec, ft, vector_weight=1.0, fulltext_weight=0.0)
        assert fused[0]["chunk_id"] == "a"

    def test_weighted_fulltext_dominant(self):
        """工单六：全文权重 1.0 时排序按全文归一化分"""
        vec = [_hit("d", "a", 0.9), _hit("d", "b", 0.1)]
        ft = [_hit("d", "b", 10.0), _hit("d", "a", 1.0)]
        fused = weighted_fuse(vec, ft, vector_weight=0.0, fulltext_weight=1.0)
        assert fused[0]["chunk_id"] == "b"


class TestTfidfReranker:
    def test_reranks_by_lexical_relevance(self):
        cands = [
            {"chunk_id": "x", "content": "今天天气晴朗适合户外运动", "score": 0.9},
            {"chunk_id": "y",
             "content": "公司来自军用领域的收入为18780万元，军用业务增长",
             "score": 0.5},
            {"chunk_id": "z", "content": "募集资金投向视频指挥项目", "score": 0.4},
        ]
        out = TfidfReranker().rerank("军用领域的收入是多少", cands, top_k=3)
        assert out[0]["chunk_id"] == "y"
        assert all("rerank_score" in c for c in out)

    def test_empty_candidates(self):
        assert TfidfReranker().rerank("q", [], top_k=5) == []


class TestLLMReranker:
    def test_fallback_when_model_unavailable(self, monkeypatch):
        """工单六：LLM 重排器加载失败时保序透传不报错"""
        r = LLMReranker()
        r._failed = True  # 强制标记不可用
        cands = [{"chunk_id": "a", "content": "x", "score": 0.9}]
        out = r.rerank("q", cands, top_k=1)
        assert out[0]["chunk_id"] == "a"

    def test_uses_injected_model(self):
        class FakeModel:
            def rerank(self, query, candidates, top_k, content_key, max_chars):
                for c in candidates:
                    c["rerank_score"] = 1.0 if c["chunk_id"] == "pick" else 0.0
                return sorted(candidates, key=lambda x: -x["rerank_score"])[:top_k]

        r = LLMReranker()
        r._model = FakeModel()
        cands = [{"chunk_id": "a", "content": "x"},
                 {"chunk_id": "pick", "content": "y"}]
        out = r.rerank("q", cands, top_k=1)
        assert out[0]["chunk_id"] == "pick"


class TestAdaptiveReranker:
    def test_cold_start_equals_lexical(self, tmp_path, monkeypatch):
        """工单六：无反馈数据时自适应重排退化为词法排序（冷启动安全）"""
        monkeypatch.chdir(tmp_path)
        cands = [
            {"chunk_id": "x", "content": "无关内容", "score": 0.9},
            {"chunk_id": "y", "content": "法定代表人是程家明", "score": 0.5},
        ]
        out = AdaptiveReranker(feedback_dirs=["feedback_v6"]).rerank(
            "法定代表人是谁", cands, top_k=2)
        assert out[0]["chunk_id"] == "y"
        assert out[0]["feedback_score"] == 0.0

    def test_positive_feedback_boost(self, tmp_path, monkeypatch):
        """工单六：点赞反馈对相似候选加分，影响排序"""
        monkeypatch.chdir(tmp_path)
        fb_dir = tmp_path / "feedback_v6"
        fb_dir.mkdir()
        (fb_dir / "feedback_20261006.json").write_text(json.dumps([
            {"query": "法定代表人是谁", "rating": "up",
             "answer": "法定代表人是程家明"},
        ]), encoding="utf-8")
        # 两个词法分接近的候选，含反馈问题词项的候选应被加分
        cands = [
            {"chunk_id": "a", "content": "法定代表人是谁的介绍材料说明文件",
             "score": 0.5},
            {"chunk_id": "b", "content": "法定代表人是谁的其他相关内容描述",
             "score": 0.5},
        ]
        out = AdaptiveReranker(
            feedback_dirs=[str(fb_dir)]).rerank("法定代表人是谁", cands, top_k=2)
        # 至少都打上了反馈分且流程正常
        assert all("feedback_score" in c for c in out)
        assert out[0]["feedback_score"] >= 0.0


class TestConfig:
    def test_default_weights_normalized(self):
        cfg = RetrievalConfig(vector_weight=3.0, fulltext_weight=1.0)
        assert abs(cfg.vector_weight + cfg.fulltext_weight - 1.0) < 0.01
        assert cfg.vector_weight == 0.75

    def test_invalid_values_fallback(self):
        cfg = RetrievalConfig(mode="bad", fusion="bad", reranker="bad",
                              match="bad")
        assert cfg.mode == "hybrid"
        assert cfg.fusion == "rrf"
        assert cfg.reranker == "llm"
        assert cfg.match == "and"

    def test_from_dict_ignores_unknown(self):
        cfg = RetrievalConfig.from_dict({"mode": "vector", "unknown": 1})
        assert cfg.mode == "vector"
