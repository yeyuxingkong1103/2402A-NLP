# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
tests/test_multilingual_v3.py —— 工单三中英文问答单测

覆盖：
  1. 语言检测：中文/英文/混合
  2. 翻译路由：英文→中文
  3. system prompt 按语言构建
  4. 集成：中英文注册资本问题
"""
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.multilingual_v3 import (
    detect_language, build_system_prompt, translate_to_chinese,
    bilingual_ask_rag,
)

load_dotenv()

_HAS_LLM = bool(os.getenv("DEEPSEEK_API_KEY")
                 and not os.getenv("DEEPSEEK_API_KEY", "").startswith("your_key"))


# ================= 1. 语言检测 =================
class TestDetectLanguage:
    def test_chinese(self):
        assert detect_language("武汉兴图新科电子股份有限公司注册资本是多少？") == "zh"

    def test_english(self):
        assert detect_language("What is the registered capital of Wuhan Xingtu?") == "en"

    def test_mixed_chinese_dominant(self):
        assert detect_language("武汉Xingtu的registered capital是多少？") == "zh"

    def test_empty(self):
        assert detect_language("") == "zh"

    def test_numbers_only(self):
        assert detect_language("12345") == "zh"


# ================= 2. system prompt =================
class TestBuildSystemPrompt:
    def test_zh(self):
        p = build_system_prompt("zh")
        assert "中文" in p or "投资分析师" in p

    def test_en(self):
        p = build_system_prompt("en")
        assert "English" in p or "investment analyst" in p

    def test_override(self):
        p = build_system_prompt("zh", lang_override="en")
        assert "English" in p or "investment analyst" in p


# ================= 3. 翻译 =================
@pytest.mark.skipif(not _HAS_LLM, reason="无 DEEPSEEK_API_KEY")
def test_translate_to_chinese():
    out = translate_to_chinese("What is the registered capital?")
    # 翻译结果应含中文
    assert any("\u4e00" <= c <= "\u9fff" for c in out)


# ================= 4. 集成：中英文注册资本 =================
@pytest.fixture(scope="module")
def engine():
    """工单三：RAGEngineV3 实例"""
    from src.rag_engine_v3 import RAGEngineV3
    return RAGEngineV3(top_k=3, use_rerank=False)


@pytest.mark.skipif(not _HAS_LLM, reason="无 DEEPSEEK_API_KEY")
def test_integration_chinese_registered_capital(engine):
    """工单三：中文注册资本问题"""
    query = "武汉兴图新科电子股份有限公司注册资本是多少？"
    r = bilingual_ask_rag(engine, query, doc_id="招股说明书1")
    assert r["lang"] == "zh"
    assert r["answer_lang"] == "zh"
    assert len(r["answer"]) > 0
    # 答案应含"万"或"元"（注册资本金额单位）
    assert "万" in r["answer"] or "元" in r["answer"], \
        f"答案应含金额单位，实际: {r['answer'][:200]}"


@pytest.mark.skipif(not _HAS_LLM, reason="无 DEEPSEEK_API_KEY")
def test_integration_english_registered_capital(engine):
    """工单三：英文注册资本问题"""
    query = "What is the registered capital of Wuhan Xingtu Xinke?"
    r = bilingual_ask_rag(engine, query, doc_id="招股说明书1")
    assert r["lang"] == "en"
    assert r["answer_lang"] == "en"
    assert r["translated"] is True
    assert len(r["answer"]) > 0
    # 英文答案应含数字或金额相关词
    assert any(c.isdigit() for c in r["answer"]) or \
           "CNY" in r["answer"] or "yuan" in r["answer"].lower(), \
        f"答案应含数字/金额，实际: {r['answer'][:200]}"
