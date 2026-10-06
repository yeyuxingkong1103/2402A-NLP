# -*- coding: utf-8 -*-
"""
问答引擎 V2 - 整合优化模块
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统优化

核心改进 (对比 V1):
    1. 查询增强 (同义词扩展 + 多语言)
    2. 混合检索 (BM25 + TF-IDF + Rerank)
    3. V2 Prompt 模板 (表格格式化 + 结构约束)
    4. 保留 V1 API 的 retriever 单例
"""
import time
import logging
from typing import Dict

import config_v2 as config
import query_enhancement
import vector_retrieval_v2
import llm_client_v2

logger = logging.getLogger(__name__)

_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = vector_retrieval_v2.get_hybrid_retriever()
    return _retriever


def answer_question(question: str) -> Dict:
    """
    V2 完整 RAG 问答

    Returns:
        {
            "question": str,
            "enhancement": {...},
            "retrieval": [...],
            "rag_answer": str,
            "llm_answer": str,
            "response_time": float,
            "within_time_limit": bool,
        }
    """
    start = time.time()

    # 1. 查询增强 (V2 新增)
    enhancement = query_enhancement.enhance_query(question)
    enhanced_query = enhancement["expanded"]
    top_k = enhancement["top_k"]
    raw_k = enhancement["raw_k"]

    # 2. 混合检索 (BM25 + TF-IDF + Rerank)
    retriever = _get_retriever()
    retrieval = retriever.search(enhanced_query, top_k=top_k, raw_k=raw_k)

    # 3. 准备 context 与 type
    contexts = [r["text"] for r in retrieval]
    context_types = [r.get("type", "text") for r in retrieval]

    # 4. V2 RAG 生成
    rag_answer = llm_client_v2.generate_with_rag_v2(
        question, contexts, context_types)

    # 5. 纯 LLM 基线
    llm_answer = llm_client_v2.generate_without_rag(question)

    elapsed = time.time() - start

    return {
        "question": question,
        "enhancement": enhancement,
        "retrieval": retrieval,
        "rag_answer": rag_answer,
        "llm_answer": llm_answer,
        "response_time": round(elapsed, 3),
        "within_time_limit": elapsed <= config.MAX_RESPONSE_TIME,
    }


def answer_simple(question: str) -> str:
    """简化版"""
    return answer_question(question)["rag_answer"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    q = "武汉兴图新科电子股份有限公司的注册资本是多少?"
    r = answer_question(q)
    print(f"意图: {r['enhancement']['intent']}")
    print(f"扩展查询: {r['enhancement']['expanded']}")
    print(f"响应时间: {r['response_time']}s (限 {config.MAX_RESPONSE_TIME}s)")
    print(f"检索: {len(r['retrieval'])} 个")
    print(f"\n--- V2 RAG 答案 ---\n{r['rag_answer']}")
    print(f"\n--- 纯 LLM 答案 ---\n{r['llm_answer']}")
