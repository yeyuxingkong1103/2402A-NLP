# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Query理解优化
#          （工单 PDF 原文：人工智能NLP-RAG-Query 理解优化任务）
"""答案语言（工单 PDF P6「多语言支持：支持中文和英文的问答」）。

缺陷背景：答案语言原本取 `analysis["lang"]`，而它是 `query_rewrite.detect_language`
的**纯 ASCII 占比**口径。问句外壳是英文、里面嵌了中文公司全称时占比被稀释
（`Who is the legal representative of 武汉力源信息技术股份有限公司?` → 0.58 ≤ 0.6），
被判成 zh，于是走了中文 system prompt，**英文提问回中文答案**，且语言守卫
（`wrong_lang` 要求 `lang == "en"` 才触发）根本不启动。
"""
from __future__ import annotations

import pytest

from src import rag

LIYUAN_EN = "Who is the legal representative of 武汉力源信息技术股份有限公司?"
LIYUAN_ZH = "武汉力源信息技术股份有限公司的法定代表人是谁？"


def _stub_pipeline(monkeypatch, analysis_lang: str, seen: dict) -> None:
    """把检索与生成换成桩，只观察**最终送给 LLM 的 system prompt**。"""
    def fake_retrieve(question, *a, **k):
        return ([{"text": "x"}], {"lang": analysis_lang, "resolved": question,
                                  "intent": "fact"}, {})

    def fake_context(hits):
        return ("【片段1】招股说明书2.pdf 第15页\n法定代表人：赵马克。",
                [], [{"page_idx": 14}], {})

    def fake_chat(messages, **kwargs):
        seen["system"] = messages[0]["content"]
        seen["user"] = messages[1]["content"]
        return {"answer": "The legal representative is 赵马克。"}

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag, "build_qa_context", fake_context)
    monkeypatch.setattr(rag.llm, "chat", fake_chat)


def test_english_question_with_chinese_entity_uses_english_prompt(monkeypatch):
    """英文提问（句子里含中文公司全称）→ 必须用英文 system prompt。

    这正是旧口径判成 zh 的那类句子：占比 0.58，`analysis["lang"]` = "zh"。
    """
    seen: dict = {}
    _stub_pipeline(monkeypatch, "zh", seen)          # 旧口径给出的语言
    r = rag.ask(LIYUAN_EN, use_cache=False)

    assert r.answer
    assert "Answer ONLY from the provided [Document Fragments]" in seen["system"], \
        "英文提问必须走英文 system prompt，不能沿用 analysis['lang'] 的 zh"
    assert r.analysis["lang"] == "en", "留痕里的 lang 也要是用户提问语言"


def test_chinese_question_still_uses_chinese_prompt(monkeypatch):
    """反向：中文提问仍走中文 prompt（防止矫枉过正）。"""
    seen: dict = {}
    _stub_pipeline(monkeypatch, "zh", seen)
    r = rag.ask(LIYUAN_ZH, use_cache=False)

    assert r.answer
    assert "【文档片段】" in seen["user"]
    assert "Answer ONLY from the provided" not in seen["system"]
    assert r.analysis["lang"] == "zh"


@pytest.mark.slow
def test_english_followup_really_answers_in_english():
    """端到端（真模型）：英文追问的答案必须含英文，而不是整句中文。

    对应多轮评测里 d5 轮（`Who is the legal representative of this company?`
    → 旧行为回「法定代表人：赵马克。[1]」）。
    """
    import re

    from src import dialogue

    st = dialogue.get_state("test-answer-lang")
    dialogue.reset_state("test-answer-lang")
    st = dialogue.get_state("test-answer-lang")

    rag.ask_turn("武汉力源信息技术股份有限公司的法定代表人是谁？", st)
    r, _t = rag.ask_turn("Who is the legal representative of this company?", st)

    assert r.answer
    assert re.search(r"[A-Za-z]{3,}", r.answer), \
        f"英文提问的答案必须至少含一个英文词，实际：{r.answer!r}"
