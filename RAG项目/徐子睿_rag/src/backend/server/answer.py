# -*- coding: utf-8 -*-
"""server/answer.py —— 把检索结果变成带引用的答案。

在链路中的位置：
    routes_search.py 的 /api/ask 与 /api/ask/stream → 【本文件】 → Ollama 生成

本文件是"答案可溯源 + 无依据不编造"这两条主张的落地点：
    SYSTEM_PROMPT 三条硬约束   只能依据片段回答、末尾标注 [来源:文件名 第X页]、依据不足就固定拒答
    generate_answer 的三道早退  知识库为空、检索无结果、模型自述找不到依据（此时清空引用）
"""
from __future__ import annotations

from typing import Any

import requests

try:
    from ..pipeline import COLLECTION
    from ..retrieval import retrieve_with_trace
    from ..vector_store import collection_count
except ImportError:
    from pipeline import COLLECTION
    from retrieval import retrieve_with_trace
    from vector_store import collection_count

from .config import LLM_MODEL, LLM_URL, REFUSAL

# 生成答案的系统提示词。三条硬约束对应本项目的核心主张：
#   1. "只能依据【知识库片段】回答，不得编造" —— 防幻觉
#   2. "答案末尾按 [来源:文件名 第X页] 标注引用" —— 可溯源
#   3. "片段不足以回答时只回复：知识库中未找到相关依据" —— 明确的拒答出口，
#      这句话必须与下面的 REFUSAL 常量完全一致，否则拒答时引用清空逻辑匹配不上
SYSTEM_PROMPT = (
    "你是基于知识库回答问题的 RAG 助手。只能依据【知识库片段】回答，不得编造；"
    "答案末尾按 [来源:文件名 第X页] 标注引用；片段不足以回答时只回复：知识库中未找到相关依据。"
)

def generate_answer(question: str, bundle: dict[str, Any]) -> dict[str, Any]:
    """把检索结果拼成提示词，调大模型生成带引用的答案。

    参数：
        question: 用户问题
        bundle:   retrieve_with_trace 的返回
    返回：
        {"answer": 答案文本, "citations": [{"source","page","text"}, ...]}

    三道早退（都不调大模型，省时间也避免无依据时的幻觉）：
        1. 知识库为空 -> 提示先上传
        2. 检索没结果 -> 直接返回拒答文案
        3. 模型自己说"未找到相关依据" -> 清空引用（见函数末尾）

    上下文格式 [片段 p{页码} {章节}] 是特意设计给模型看的：
        让模型在组织答案时能直接读到页码和章节，它标注 [来源:文件名 第X页] 才准。
    """
    if collection_count(COLLECTION) == 0:
        return {"answer": "知识库为空，请先上传 PDF 构建。", "citations": []}

    # 优先用 organize_context 精选并截断过的 context_docs，退化时才用未整理的 results
    hits = bundle["context_docs"] or bundle["results"]
    if not hits:
        return {"answer": REFUSAL, "citations": []}

    context = "\n\n".join(
        f"[片段 p{hit['page']} {hit['section']}]\n{hit.get('snippet') or hit['text']}"
        for hit in hits
    )
    response = requests.post(
        LLM_URL,
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"【知识库片段】\n{context}\n\n【问题】{question}\n请依据片段回答。"},
            ],
            "stream": False,
            "options": {"temperature": 0.2},  # 低温度：问答要的是忠实复述依据，不是创造性发挥
        },
        timeout=180,
    )
    response.raise_for_status()
    answer = response.json()["message"]["content"].strip()
    # 引用取每条片段的前 60 字做摘要，前端点开能核对"这句话出自哪段原文"
    citations = [
        {"source": hit["source"], "page": hit["page"], "text": (hit.get("snippet") or hit["text"])[:60]}
        for hit in hits
    ]
    if REFUSAL in answer or "未找到相关依据" in answer:
        # 模型拒答了却还显示一堆引用，会让人误以为答案有依据 —— 这里必须清干净
        citations = []
    return {"answer": answer, "citations": citations}

def retrieve_and_answer(question: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """检索 + 生成的一步封装（/api/ask 与 SSE 流式接口共用）。

    参数：
        question: 用户问题
    返回：
        (检索 bundle, 生成结果)
    说明：
        这里的 15/5/3600 与 retrieval 的默认值一致，显式写出来是为了让接口层的
        行为一眼可查，不用翻到 retrieval.py 才知道用的是什么参数。
    """
    bundle = retrieve_with_trace(question, candidate_k=15, final_k=5, char_budget=3600)
    return bundle, generate_answer(question, bundle)
