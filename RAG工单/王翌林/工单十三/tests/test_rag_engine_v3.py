# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
tests/test_rag_engine_v3.py —— 工单三 RAG 引擎单测

覆盖：
  1. build_table_aware_prompt：上下文组装（text + table 分号）
  2. RAGEngineV3.ask_rag：路由驱动检索 + LLM 生成
  3. RAGEngineV3.ask_llm：纯 LLM 模式
  4. 集成：力源发行股数（T08）
  5. 集成：兴图军用领域收入（T05）
"""
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.llm_client_v3 import build_table_aware_prompt
from src.rag_engine_v3 import RAGEngineV3
from src.table_parser.query_router import route_query

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HAS_LLM = bool(os.getenv("DEEPSEEK_API_KEY")
                 and not os.getenv("DEEPSEEK_API_KEY", "").startswith("your_key"))


# ================= 1. prompt 组装 =================
class TestTableAwarePrompt:
    def test_text_only(self):
        chunks = [{"content": "公司主营业务…", "page": 10, "doc_id": "d1"}]
        prompt = build_table_aware_prompt("公司简介？", chunks, [])
        assert "【文本资料】" in prompt
        assert "[资料1]" in prompt
        assert "【表格资料】" not in prompt

    def test_table_only(self):
        tables = [{"table_text": "发行股数 1,670万股", "page": 2,
                   "doc_id": "d2", "table_id": "tbl_003"}]
        prompt = build_table_aware_prompt("发行股数？", [], tables)
        assert "【表格资料】" in prompt
        assert "[表1]" in prompt
        assert "1,670万股" in prompt

    def test_hybrid(self):
        chunks = [{"content": "文本A", "page": 5, "doc_id": "d1"}]
        tables = [{"table_text": "表格B", "page": 6, "doc_id": "d1"}]
        prompt = build_table_aware_prompt("问题？", chunks, tables)
        assert "【文本资料】" in prompt
        assert "【表格资料】" in prompt


# ================= 2. 引擎单测（不依赖 LLM） =================
@pytest.fixture(scope="module")
def engine():
    """工单三：RAGEngineV3（关 reranker 加速单测）"""
    return RAGEngineV3(use_rerank=False, top_k=3)


def test_route_table_only(engine):
    r = route_query("发行股数是多少？")
    assert r.route in ("table_only", "hybrid")


def test_ask_rag_no_results_graceful(engine):
    """工单三：无候选时返回友好提示（不崩）"""
    # 用一个无关 doc_id 触发空结果
    r = engine.ask_rag("发行股数", doc_id="nonexistent_doc_12345")
    assert r["mode"] == "rag_v3"
    # 空结果应返回"未找到"
    if not r["retrieved_tables"] and not r["retrieved_text_chunks"]:
        assert "未能在知识库" in r["answer"]


def test_ask_rag_returns_structure(engine):
    """工单三：ask_rag 输出结构完整"""
    r = engine.ask_rag("发行股数是多少？", doc_id="招股说明书2")
    assert r["mode"] == "rag_v3"
    assert "answer" in r
    assert "references" in r
    assert "latency_ms" in r
    assert "retrieved_text_chunks" in r
    assert "retrieved_tables" in r
    assert "breakdown" in r
    assert "retrieve_ms" in r["breakdown"]
    # 表格问题应有表格候选
    if r["route"]["route"] in ("table_only", "hybrid"):
        assert len(r["retrieved_tables"]) >= 1


# ================= 3. 纯 LLM 模式 =================
@pytest.mark.skipif(not _HAS_LLM, reason="无 DEEPSEEK_API_KEY")
def test_ask_llm(engine):
    r = engine.ask_llm("什么是招股说明书？")
    assert r["mode"] == "pure_llm"
    assert len(r["answer"]) > 0
    assert r["latency_ms"] > 0


# ================= 4. 集成：力源发行股数（T08） =================
@pytest.mark.skipif(not _HAS_LLM, reason="无 DEEPSEEK_API_KEY")
def test_integration_liyuan_shares(engine):
    """工单三 T08：武汉力源本次发行股数"""
    query = ("武汉力源信息技术股份有限公司本次发行股数是多少，"
             "占发行后总股本的比例是多少？")
    r = engine.ask_rag(query, doc_id="招股说明书2")
    assert r["mode"] == "rag_v3"
    assert r["route"]["route"] in ("table_only", "hybrid")
    assert len(r["retrieved_tables"]) >= 1
    # 答案应含"万股"或"股数"
    assert ("万股" in r["answer"] or "股" in r["answer"]
            or "shares" in r["answer"].lower()), \
        f"答案应含股数单位，实际: {r['answer'][:200]}"


# ================= 5. 集成：兴图军用领域收入（T05） =================
@pytest.mark.skipif(not _HAS_LLM, reason="无 DEEPSEEK_API_KEY")
def test_integration_xingtu_military_revenue(engine):
    """工单三 T05：武汉兴图军用领域收入"""
    query = ("报告期内，武汉兴图新科电子股份有限公司"
             "来自军用领域的收入分别是多少？")
    r = engine.ask_rag(query, doc_id="招股说明书1")
    assert r["mode"] == "rag_v3"
    assert r["route"]["route"] in ("table_only", "hybrid")
    # 应有候选（文本或表格）
    assert (len(r["retrieved_tables"]) + len(r["retrieved_text_chunks"])) >= 1
