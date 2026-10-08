# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
tests/test_fulltext_retriever.py —— 工单六 全文检索器单元测试（新增文件）

用注入 chunks 构建内存倒排索引，不依赖 Milvus/模型。
覆盖：AND/OR 布尔、短语、模糊匹配、多字段权重、doc_id 过滤、TF-IDF 排序。
"""
import pytest

from src.retrieval.fulltext_retriever import FulltextRetriever, tokenize
from src.retrieval.retrieval_config import (
    FIELD_CONTENT, FIELD_SUMMARY, FIELD_TITLE,
)

WORK_ORDER = "人工智能NLP-RAG-混合检索任务"


@pytest.fixture
def retriever():
    chunks = [
        {"doc_id": "招股说明书1", "chunk_id": "c1", "page": 1,
         "content": "公司来自军用领域的收入持续增长，2018年军用收入18780万元",
         "metadata": {}},
        {"doc_id": "招股说明书1", "chunk_id": "c2", "page": 2,
         "content": "本次发行募集资金总额40584万元，投向军用视频指挥项目",
         "metadata": {}},
        {"doc_id": "招股说明书2", "chunk_id": "c3", "page": 3,
         "content": "武汉力源信息技术股份有限公司销售部下设大客户销售部",
         "metadata": {}},
        {"doc_id": "招股说明书2", "chunk_id": "c4", "page": 4,
         "content": "2008年中国IC市场应用结构中汽车电子增长率14.0%",
         "metadata": {}},
    ]
    return FulltextRetriever(chunks=chunks)


class TestTokenizer:
    def test_tokenize_keeps_business_terms(self):
        toks = tokenize("军用领域的收入是多少")
        assert "军用" in toks and "收入" in toks

    def test_tokenize_keeps_numbers(self):
        toks = tokenize("收入18780万元")
        assert any("18780" in t for t in toks)


class TestBooleanMatch:
    def test_and_match(self, retriever):
        """工单六：AND —— 所有关键词共现才召回"""
        hits = retriever.search("军用 收入", top_k=10, match="and")
        ids = [h["chunk_id"] for h in hits]
        assert "c1" in ids
        # c2 含"军用"和"募集"但不含"收入"，AND 不召回
        assert "c2" not in ids

    def test_or_match(self, retriever):
        """工单六：OR —— 任一词命中即召回，召回面更广"""
        # c1 仅含"军用"，c2 同时含"军用/募集"：AND 只命中 c2，OR 命中 c1+c2
        hits_and = retriever.search("军用 募集", top_k=10, match="and")
        hits_or = retriever.search("军用 募集", top_k=10, match="or")
        ids_and = {h["chunk_id"] for h in hits_and}
        ids_or = {h["chunk_id"] for h in hits_or}
        assert ids_and == {"c2"}
        assert {"c1", "c2"} <= ids_or
        assert len(ids_or) >= len(ids_and)

    def test_boolean_not_operator(self, retriever):
        """工单六：'-词' 排除语法"""
        hits = retriever.search("军用 -募集", top_k=10, match="and")
        ids = [h["chunk_id"] for h in hits]
        assert "c1" in ids and "c2" not in ids


class TestPhraseMatch:
    def test_phrase_quoted(self, retriever):
        """工单六：引号短语连续匹配"""
        hits = retriever.search('"汽车电子增长率"', top_k=10, match="and")
        ids = [h["chunk_id"] for h in hits]
        assert "c4" in ids

    def test_phrase_not_broken(self, retriever):
        """工单六：不连续的词序不构成短语命中"""
        hits = retriever.search('"汽车销售部"', top_k=10, match="and")
        assert all(h["chunk_id"] != "c4" for h in hits)


class TestFuzzyMatch:
    def test_fuzzy_tolerates_variant(self, retriever):
        """工单六：模糊匹配容忍近义/错字（bigram Jaccard）"""
        hits = retriever.search("军用领域收入增长情况", top_k=10,
                                match="fuzzy")
        ids = [h["chunk_id"] for h in hits]
        assert "c1" in ids


class TestMultiField:
    def test_title_field_match(self, retriever):
        """工单六：标题字段（doc_id）可单独命中"""
        hits = retriever.search("招股说明书2", top_k=10, match="and",
                                fields=[FIELD_TITLE])
        ids = {h["chunk_id"] for h in hits}
        assert ids == {"c3", "c4"}

    def test_doc_id_filter(self, retriever):
        """工单六：doc_id 多文档隔离过滤"""
        hits = retriever.search("军用 收入", top_k=10, match="and",
                                doc_id="招股说明书1")
        assert all(h["doc_id"] == "招股说明书1" for h in hits)
        assert any(h["chunk_id"] == "c1" for h in hits)

    def test_field_weights_affect_order(self, retriever):
        """工单六：标题权重提升时，标题命中的文档排前"""
        # 仅标题字段 + 高权重
        hits = retriever.search("招股说明书2", top_k=2, match="and",
                                fields=[FIELD_TITLE, FIELD_CONTENT],
                                field_weights={FIELD_TITLE: 5.0,
                                               FIELD_CONTENT: 0.1,
                                               FIELD_SUMMARY: 0.1})
        assert hits[0]["doc_id"] == "招股说明书2"

    def test_empty_index(self):
        r = FulltextRetriever(chunks=[])
        assert r.search("测试", top_k=5) == []
        assert r.size == 0
