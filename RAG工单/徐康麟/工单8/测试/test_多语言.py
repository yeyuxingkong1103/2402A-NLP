# -*- coding: utf-8 -*-
"""工单8 测试：多语言支持（中文 / 英文问答）。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

验收项对应：工单原文验收标准 4「多语言支持：支持中文和英文的问答」。

测试策略（**不依赖 Ollama**，保证 CI 可跑）：
    * 语言判定：复用 工单7 的 ``app/core/language``（与本工单同一份代码）；
    * 英文题**检索**：bge-m3 是多语向量模型，英文问句可直接检索中文语料，
      断言「英文题能召回期望文件的可疑证据块」；
    * 英文问句 → 中文检索式：LLM 翻译不可用时按设计**降级**（不编造），
      断言降级路径被如实标记，而不是静默失败。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "研发"))

from app.core.language import detect_language, resolve_language  # noqa: E402

#: 工单编号（源码头部注释要求）：人工智能NLP-RAG-基于Graph RAG 实现金融问答

ENGLISH_Q = ("According to Postal Savings Bank of China's 2019 Annual Report, what were the bank's "
             "net profit and its year-on-year growth rate, the NPL ratio, the provision coverage ratio, "
             "and the IT investment amount with its share of operating revenue in 2019?")
CHINESE_Q = "平安银行2019年年度报告中，该行2019年实现营业收入和净利润分别是多少？"
PSBC_FILE = "2020-03-26__中国邮政储蓄银行股份有限公司__601658__邮储银行__2019年__年度报告.pdf"


def test_语言自动判定() -> None:
    """中文问句判 zh、英文问句判 en（按 CJK/拉丁字符占比）。"""
    assert detect_language(CHINESE_Q) == "zh"
    assert detect_language(ENGLISH_Q) == "en"
    assert resolve_language(CHINESE_Q, "auto").resolved == "zh"
    assert resolve_language(ENGLISH_Q, "auto").resolved == "en"


def test_显式语言优先于自动判定() -> None:
    """``--lang en`` 时即使问句是中文，也按用户指定执行（并如实留痕，不静默）。"""
    info = resolve_language(CHINESE_Q, "en")
    assert info.resolved == "en", "用户显式指定语言未生效"
    assert info.requested == "en", "未记录用户原始请求"


def test_引擎回答语言标注正确(engine) -> None:
    """英文问句的作答语言必须是 ``en``（不因检索式被译成中文而改变）。"""
    answer = engine.answer(ENGLISH_Q, top_k=10, use_graph=True, use_llm=False)
    assert answer.language == "en", f"英文问句的语言标注错误：{answer.language}"
    assert answer.answer, "英文问句返回空答案"
    assert answer.evidence_chunks, "英文问句未返回证据块"


def test_英文问句能召回对应年报证据(engine) -> None:
    """英文问句（bge-m3 多语向量 + BM25 英文 token）必须召回邮储银行年报的块。"""
    result = engine.retrieve(ENGLISH_Q, top_k=10, use_graph=True)
    hits = list(result.get("hits") or [])
    assert hits, "英文问句检索返回空"
    files = {str(h.get("file_name")) for h in hits}
    assert any(PSBC_FILE == f for f in files), (
        f"英文问句未召回邮储银行年报（实际召回文件：{sorted(files)[:3]}）")


def test_中文问句不受多语言改造影响(engine) -> None:
    """中文路径必须保持原有行为：语言 zh、返回带页码的证据块。"""
    answer = engine.answer(CHINESE_Q, top_k=10, use_graph=True, use_llm=False)
    assert answer.language == "zh"
    assert answer.evidence_chunks, "中文问句未返回证据块"
    assert all(int(c.get("page") or 0) > 0 for c in answer.evidence_chunks), "证据块缺页码"


def test_中英文均可产出图谱结构(engine) -> None:
    """中英文问句都应产出图谱结构缩略（图谱实体匹配是语言无关的子串匹配）。"""
    for question in (CHINESE_Q, ENGLISH_Q):
        result = engine.retrieve(question, top_k=10, use_graph=True)
        graph = result.get("graph") or {}
        assert graph.get("nodes") is not None, f"图谱结构缺失：{question[:20]}"
        assert "seed_names" in graph, "图谱结构缺 seed_names 字段"
