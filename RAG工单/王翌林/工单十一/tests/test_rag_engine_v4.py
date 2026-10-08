# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
tests/test_rag_engine_v4.py —— 工单四融合引擎单测（Mock 三路检索与 LLM，不依赖模型/Milvus）
"""
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"

# ---------------- 工单四：Mock 组件 ----------------
MOCK_TEXT = [{"source": "text", "doc_id": "招股说明书2", "page": 38,
              "content": "公司设有销售部，下设大客户销售部等4个部门。",
              "chunk_id": "c1", "rrf_score": 0.9}]
MOCK_TABLE = [{"source": "table", "doc_id": "招股说明书2", "page": 100,
               "table_id": "tbl_001", "content": "年度 营业收入",
               "rrf_score": 0.8}]
MOCK_IMAGES = [{"image_id": "img_008", "doc_id": "招股说明书2", "page": 39,
                "path": "data/images/招股说明书2/page_039_draw_1.png",
                "caption": "组织结构图", "ocr_text": "销售部 大客户销售部",
                "vqa_text": "销售部由4个部门构成；大客户销售部下设6个销售处",
                "rrf": 0.85}]


class FakeV3:
    """工单四：Mock RAGEngineV3（table_retriever.retrieve + ask_llm）"""
    top_k = 5
    max_context_chars = 9000

    class _TR:
        def retrieve(self, query, top_k=5, doc_id=None, company=None, route=None):
            return {"fused": MOCK_TEXT + MOCK_TABLE, "elapsed_ms": 12.5}

    table_retriever = _TR()

    def ask_llm(self, query):
        return {"mode": "pure_llm", "query": query, "answer": "纯LLM答案",
                "references": [], "retrieved_text_chunks": [],
                "retrieved_tables": [], "latency_ms": 5.0, "token_usage": {}}


@pytest.fixture()
def engine(monkeypatch):
    """工单四：构造被测引擎（Mock 图像检索 + Mock LLM chat）"""
    import src.rag_engine_v4 as m
    eng = m.RAGEngineV4(v3_engine=FakeV3(), image_retriever=MagicMock(),
                        top_k=5)
    ir = MagicMock()
    ir.retrieve.return_value = MOCK_IMAGES           # bge-m3 通道命中组织结构图
    eng._image_retriever = ir
    # 工单四：屏蔽 CLIP 通道（模型未就绪场景）
    monkeypatch.setattr(eng, "retrieve_images",
                        lambda q, d, top_k=None: MOCK_IMAGES, raising=False)
    monkeypatch.setattr(m, "chat", MagicMock(return_value={
        "content": "销售部由4个部门构成[图1]，大客户销售部下设6个销售处。",
        "token_usage": {"total_tokens": 100}}))
    return eng


# ---------------- 工单四：路由与检索 ----------------
def test_image_route_priority():
    """工单四：id5 类问题路由 image_first（组织结构强信号）"""
    from src.image_parser.image_query_router import route_query
    r = route_query("武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成？")
    assert r["mode"] == "image_first" and r["confidence"] >= 0.7


def test_ask_includes_images_and_answer(engine):
    """工单四：融合问答返回图像引用与三路明细"""
    r = engine.ask("武汉力源组织结构图中，销售部有几个部门构成？", doc_id="招股说明书2")
    assert r["mode"] == "rag_v4"
    assert "4个部门" in r["answer"] and "6个销售处" in r["answer"]
    assert len(r["retrieved_images"]) == 1
    assert r["retrieved_images"][0]["image_id"] == "img_008"
    assert any(x["type"] == "text" for x in r["references"])
    assert any(x["type"] == "table" for x in r["references"])
    assert any(x["type"] == "image" and x["ref_id"] == "图1"
               for x in r["references"])
    assert r["latency_ms"] > 0 and "retrieve_ms" in r["breakdown"]


def test_ask_without_image_switch(engine):
    """工单四：use_image=False 时不检索图像（对照）"""
    calls = {}

    def fake_retrieve_images(q, d, top_k=None):
        calls["called"] = True
        return []

    import src.rag_engine_v4 as m
    m.chat  # noqa  确保模块可引用
    engine.retrieve_images = fake_retrieve_images
    r = engine.ask("武汉力源2018年营业收入是多少？", doc_id="招股说明书2",
                   use_image=False)
    assert "called" not in calls
    assert r["retrieved_images"] == []
    assert any(x["type"] == "text" for x in r["references"])


# ---------------- 工单四：上下文组装 ----------------
def test_multimodal_prompt_contains_three_sources():
    """工单四：三源上下文——图像块置顶（caption/VQA/OCR）+ 表格 + 文本"""
    from src.llm_client_v4 import build_multimodal_prompt
    prompt = build_multimodal_prompt("销售部构成？", MOCK_TEXT, MOCK_TABLE,
                                     MOCK_IMAGES)
    assert prompt.index("【图1】") < prompt.index("【表1】") < prompt.index("【资料1】")
    assert "组织结构图" in prompt and "6个销售处" in prompt
    assert "OCR" in prompt and "VQA" in prompt
    assert prompt.startswith("问题：")


def test_llm_empty_response_retry(engine, monkeypatch):
    """工单四：LLM 空响应自动重试一次"""
    import src.rag_engine_v4 as m
    resp = [{"content": "", "token_usage": {}},
            {"content": "重试后的答案", "token_usage": {}}]
    monkeypatch.setattr(m, "chat", MagicMock(side_effect=resp))
    r = engine.ask("组织结构图 销售部", doc_id="招股说明书2")
    assert r["answer"] == "重试后的答案"
    assert m.chat.call_count == 2


def test_retrieval_failure_graceful(engine):
    """工单四：三路全空时优雅返回未找到（不抛异常）"""
    engine.retrieve_images = lambda q, d, top_k=None: []
    engine._v3.table_retriever.retrieve = MagicMock(
        return_value={"fused": [], "elapsed_ms": 1.0})
    r = engine.ask("不相干问题", doc_id=None)
    assert "未能" in r["answer"] and r["references"] == []
