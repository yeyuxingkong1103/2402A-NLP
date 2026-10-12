# -*- coding: utf-8 -*-
"""
query_understanding.py — Query 理解模块
工单编号：人工智能NLP-RAG-基于PDF 文档的问答系统

实现基础的 Query 理解能力：
  1. 意图识别：判断用户问题的核心意图（检索类 / 闲聊 / 其他）
  2. 消歧：对多义、模糊表述做规范化
  3. 分解与抽象：将复杂问题拆分为子问题，抽取关键信息
"""
from __future__ import annotations

import re
from typing import List, Optional

import rag_core
from rag_core import BaseLLM

# 意图关键词表（简单规则实现；接入 LLM 时可替换为分类提示词）
INTENT_PATTERNS = {
    "数值查询": ["多少", "占比", "比例", "金额", "数量", "注册资本", "股数", "募集"],
    "事实查询": ["是谁", "哪个", "哪些", "什么", "何时", "何时", "制定", "参与", "荣获", "涉及", "成为"],
    "分析查询": ["分析", "对比", "原因", "为什么", "如何", "趋势"],
}


def detect_intent(query: str) -> str:
    """意图识别：基于关键词规则返回意图类别。"""
    for intent, kws in INTENT_PATTERNS.items():
        if any(k in query for k in kws):
            return intent
    return "其他"


def normalize(query: str) -> str:
    """消歧与规范化：去除语气词、统一标点、压缩空白。"""
    q = query.strip()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[。？?！!]+$", "", q)
    return q


def decompose(query: str, llm: Optional[BaseLLM] = None) -> List[str]:
    """分解与抽象：复杂问题拆为子问题；LLM 不可用时退化为原问题。"""
    # 以句号 / 逗号切分的多子句，直接作为子问题
    parts = [p for p in re.split(r"[，,；;]", query) if len(p.strip()) > 3]
    if len(parts) <= 1:
        return [normalize(query)]
    return [normalize(p) for p in parts]


def understand(query: str, llm: Optional[BaseLLM] = None) -> dict:
    """Query 理解统一入口。"""
    q = normalize(query)
    return {
        "query": q,
        "intent": detect_intent(q),
        "sub_queries": decompose(q, llm),
    }
