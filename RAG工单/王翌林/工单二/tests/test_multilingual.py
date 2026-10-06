# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
tests/test_multilingual.py —— 工单二中英文问答单元测试

策略：翻译与 LLM 生成用 monkeypatch 模拟，不依赖真实 API；
真实双语链路由 _tmp_bilingual_demo 脚本验证（注册资本中英问题）。
"""
import pytest

import src.multilingual as ml
from src.multilingual import (bilingual_answer, bilingual_retrieve, build_system_prompt,
                              detect_language, translate_to_chinese)


class FakeRetriever:
    """记录检索查询的假检索器（验证翻译路由是否生效）"""

    def __init__(self):
        self.calls = []

    def retrieve(self, query, top_k=5):
        self.calls.append(query)
        return {"results": [{"chunk_id": "c0", "content": "注册资本：6,000万元", "score": 0.9,
                             "source": "招股说明书1.pdf#page=8", "page": 8,
                             "chunk_type": "text", "heading": "第一节", "parent_id": "p0"}],
                "rewritten": {}}


@pytest.fixture
def fake_llm(monkeypatch):
    """模拟 LLM：英文输入返回固定中文译文；answer 返回固定答案"""
    def fake_translate(text):
        return "武汉兴图新科电子股份有限公司的注册资本是多少？"
    monkeypatch.setattr(ml, "translate_to_chinese", fake_translate)
    return fake_translate


# ---------- 1. 语言检测（人工智能NLP-RAG-基于PDF文档的问答系统优化） ----------
def test_detect_zh():
    assert detect_language("武汉兴图新科电子股份有限公司注册资本是多少？") == "zh"


def test_detect_en():
    assert detect_language("What is the registered capital of Wuhan Xingtu Xinke?") == "en"


def test_detect_mixed_majority_zh():
    assert detect_language("公司的registered capital是多少？") == "zh"


def test_detect_mixed_majority_en():
    assert detect_language("公司 registered capital 是多少 how much total") == "en"


def test_detect_empty_fallback_zh():
    assert detect_language("") == "zh"


# ---------- 2. 检索路由（人工智能NLP-RAG-基于PDF文档的问答系统优化） ----------
def test_zh_direct_retrieve(fake_llm):
    ret = FakeRetriever()
    out = bilingual_retrieve(ret, "注册资本是多少")
    assert out["lang"] == "zh" and not out["translated"]
    assert ret.calls == ["注册资本是多少"]  # 中文不翻译直接检索


def test_en_translated_retrieve(fake_llm):
    ret = FakeRetriever()
    out = bilingual_retrieve(ret, "What is the registered capital?")
    assert out["lang"] == "en" and out["translated"] is True
    assert ret.calls == ["武汉兴图新科电子股份有限公司的注册资本是多少？"]  # 翻译后检索


def test_en_translate_fail_fallback(monkeypatch):
    """翻译失败 → 回退原文跨语言检索（不抛异常）"""
    def boom(text):
        raise RuntimeError("LLM timeout")
    monkeypatch.setattr(ml, "translate_to_chinese", boom)
    ret = FakeRetriever()
    out = bilingual_retrieve(ret, "What is the registered capital?")
    assert out["translated"] is False
    assert ret.calls == ["What is the registered capital?"]


def test_lang_override_forces_retrieval_path(fake_llm):
    ret = FakeRetriever()
    out = bilingual_retrieve(ret, "What is the registered capital?", lang_override="zh")
    assert out["lang"] == "zh"


# ---------- 3. 回答语言一致（人工智能NLP-RAG-基于PDF文档的问答系统优化） ----------
def test_system_prompt_language():
    assert "中文" in build_system_prompt("zh")
    assert "English" in build_system_prompt("en")
    # override 优先于提问语言
    assert "中文" in build_system_prompt("en", lang_override="zh")
    assert "English" in build_system_prompt("zh", lang_override="en")


def test_bilingual_answer_en_uses_english_prompt(fake_llm, monkeypatch):
    captured = {}

    def fake_generate(prompt, system=None, **kw):
        captured["prompt"] = prompt
        captured["system"] = system
        return "The registered capital is 60 million RMB."

    monkeypatch.setattr("src.llm_client.simple_generate", fake_generate)
    ret = FakeRetriever()
    out = bilingual_answer("What is the registered capital?", ret)
    assert out["lang"] == "en"
    assert out["answer"].startswith("The registered capital")
    assert "Answer the question in English" in captured["prompt"]
    assert "English" in captured["system"]


def test_bilingual_answer_zh_keeps_chinese(fake_llm, monkeypatch):
    captured = {}

    def fake_generate(prompt, system=None, **kw):
        captured["prompt"] = prompt
        return "注册资本为6,000万元。"

    monkeypatch.setattr("src.llm_client.simple_generate", fake_generate)
    ret = FakeRetriever()
    out = bilingual_answer("注册资本是多少？", ret)
    assert out["lang"] == "zh" and not out["translated"]
    assert "中文回答" in captured["prompt"]
    assert out["references"][0]["page"] == 8


def test_translate_helper_rejects_non_chinese(monkeypatch):
    """翻译输出无中文时回退原文"""
    monkeypatch.setattr("src.llm_client.simple_generate", lambda *a, **k: "sorry no idea")
    assert translate_to_chinese("What is X?") == "What is X?"
