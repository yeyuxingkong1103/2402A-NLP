# -*- coding: utf-8 -*-
"""
问答引擎模块
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统

功能:
    1. 整合 Query 理解 → 检索 → LLM 生成
    2. 同时输出 RAG 与 纯 LLM 两种结果以便对比
    3. 记录响应时间用于性能验收
"""
import time
import logging
from typing import Dict

import config
import query_understanding
import vector_retrieval
import llm_client

logger = logging.getLogger(__name__)

# 全局单例 (避免重复加载索引)
_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = vector_retrieval.get_retriever()
    return _retriever


def answer_question(question: str, top_k: int = None) -> Dict:
    """
    对用户问题进行完整 RAG 问答

    Returns:
        {
            "question": str,
            "understanding": {...},
            "retrieval": [...],
            "rag_answer": str,
            "llm_answer": str,           # 纯 LLM 对比基线
            "response_time": float,      # 秒
            "within_time_limit": bool,
        }
    """
    start = time.time()

    # 1. Query 理解
    understanding = query_understanding.understand_query(question)
    enriched_query = understanding["enriched_query"]

    # 2. 向量检索
    retriever = _get_retriever()
    retrieval = retriever.search(enriched_query, top_k=top_k or config.TOP_K)
    contexts = [r["text"] for r in retrieval]

    # 3. RAG 生成
    rag_answer = llm_client.generate_with_rag(question, contexts)

    # 4. 纯 LLM 生成 (对比基线)
    llm_answer = llm_client.generate_without_rag(question)

    elapsed = time.time() - start

    return {
        "question": question,
        "understanding": understanding,
        "retrieval": retrieval,
        "rag_answer": rag_answer,
        "llm_answer": llm_answer,
        "response_time": round(elapsed, 3),
        "within_time_limit": elapsed <= config.MAX_RESPONSE_TIME,
    }


def answer_simple(question: str) -> str:
    """简化版: 仅返回 RAG 答案"""
    result = answer_question(question)
    return result["rag_answer"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    q = "武汉兴图新科电子股份有限公司的注册资本是多少?"
    r = answer_question(q)
    print(f"响应时间: {r['response_time']}s (限 {config.MAX_RESPONSE_TIME}s)")
    print(f"检索片段数: {len(r['retrieval'])}")
    print("\n--- RAG 答案 ---")
    print(r["rag_answer"])
    print("\n--- 纯 LLM 答案 ---")
    print(r["llm_answer"])
