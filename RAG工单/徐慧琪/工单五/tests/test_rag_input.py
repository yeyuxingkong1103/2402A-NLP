# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Query理解优化
#          （工单 PDF 原文：人工智能NLP-RAG-Query 理解优化任务）
"""输入侧容错（工单 PDF「3 系统稳定性 / 容错机制：能够处理常见的异常情况，如用户输入错误」）。

只覆盖**不加载模型就能走到**的路径：空输入在 `rag.ask` 入口即返回，
不会碰检索与生成，因此本文件是 `not slow` 的纯逻辑测试。
"""
from __future__ import annotations

import pytest

from src import rag


@pytest.mark.parametrize("bad", ["", "   ", "\t\n", None])
def test_blank_input_returns_hint_without_raising(bad):
    """空/空白/None 输入必须给出提示而不是抛异常或检索瞎答。"""
    r = rag.ask(bad)
    assert r.answer == "请输入问题。"
    assert r.mode == "extractive"
    assert r.citations == []


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_blank_input_with_dialogue_state_also_short_circuits(bad):
    """多轮入口同样要挡住在最前面，不能把空串喂给 coref。"""
    from src import dialogue
    st = dialogue.get_state("input-tolerance")
    r = rag.ask(bad, state=st)
    assert r.answer == "请输入问题。"
    assert st.turn_count == 0, "空输入不该在对话历史里留下轮次"
