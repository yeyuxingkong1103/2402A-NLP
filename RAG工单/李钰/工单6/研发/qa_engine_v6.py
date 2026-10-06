# -*- coding: utf-8 -*-
"""
问答引擎 V6 - 混合检索整合
工单编号: 人工智能 NLP-RAG-混合检索任务

核心: 配置化检索策略 (vector / fulltext / hybrid) + 3 种重排 + 3 种融合
"""
import os, sys, time, logging, json
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config_v6 as config
import hybrid_retriever

logger = logging.getLogger(__name__)

# 引入 V5 多轮对话 + LLM
_V5_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "工单5", "研发"))
if _V5_DIR not in sys.path:
    sys.path.insert(0, _V5_DIR)

_retriever = None


def _get_retriever() -> hybrid_retriever.HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = hybrid_retriever.HybridRetriever(
            vector_weight=config.VECTOR_WEIGHT,
            fulltext_weight=config.FULLTEXT_WEIGHT,
            fusion_method=config.FUSION_METHOD,
        )
        # 尝试加载已有索引
        if not _retriever.load_index():
            logger.info("索引不存在, 请先调用 build_index()")
    return _retriever


def build_index_from_chunks(chunks):
    """从 chunks 构建索引并持久化"""
    global _retriever
    _retriever = hybrid_retriever.HybridRetriever(
        vector_weight=config.VECTOR_WEIGHT,
        fulltext_weight=config.FULLTEXT_WEIGHT,
        fusion_method=config.FUSION_METHOD,
    )
    _retriever.build_index(chunks)
    _retriever.save_index()
    return _retriever


def answer_question(question: str, strategy: str = None) -> Dict:
    """
    V6 问答主入口

    Args:
        question: 用户问题
        strategy: vector / fulltext / hybrid (None → 用 config)
    """
    start = time.time()
    retriever = _get_retriever()
    if retriever is None:
        return {"error": "索引未构建, 请先 build_index_from_chunks()"}

    strategy = strategy or config.RETRIEVAL_STRATEGY
    results = retriever.search_by_strategy(question, strategy=strategy)

    # 准备 LLM 上下文
    contexts = []
    for i, r in enumerate(results[:5]):
        text = r.get("text", "") or str(r.get("chunk", ""))
        contexts.append(f"【片段{i+1}】{text[:300]}")
    context = "\n\n".join(contexts) if contexts else ""

    # 生成答案 (尝试复用 V5/V4 的 LLM)
    rag_answer = _generate_answer(question, context)
    llm_answer = _generate_without_rag(question)

    elapsed = time.time() - start

    return {
        "question": question,
        "strategy": strategy,
        "fusion_method": retriever.fusion_method,
        "embedding_model": retriever.vector_retriever.embedding_model,
        "reranker": retriever.vector_retriever.reranker_name,
        "retrieval_results": [{
            "id": r.get("id"),
            "text": (r.get("text") or "")[:150],
            "score": r.get("fused_score", r.get("rerank_score", r.get("score", 0))),
            "v_score": r.get("v_score"),
            "f_score": r.get("f_score"),
            "votes": r.get("votes"),
            "source": r.get("source"),
            "hit_terms": r.get("hit_terms"),
        } for r in results],
        "rag_answer": rag_answer or context[:500],
        "llm_answer": llm_answer,
        "response_time": round(elapsed, 3),
        "within_time_limit": elapsed <= 3.0,
    }


def _generate_answer(question: str, context: str) -> str:
    if not context:
        return ""
    try:
        sys.path.insert(0, _V5_DIR)
        from llm_client_v2 import _call_openai_api
        messages = [
            {"role": "system", "content": "你是金融文档问答助手, 仅根据提供的片段回答。如果片段没有相关信息, 回答'文档中未找到相关信息'。"},
            {"role": "user", "content": f"片段:\n{context}\n\n问题: {question}"},
        ]
        return _call_openai_api(messages) or ""
    except Exception as e:
        logger.warning(f"LLM 调用不可用: {e}")
        return ""


def _generate_without_rag(question: str) -> str:
    try:
        sys.path.insert(0, _V5_DIR)
        from llm_client_v2 import generate_without_rag
        return generate_without_rag(question)
    except Exception:
        return "[纯LLM模式未启用]"


def record_feedback(chunk_id: str, is_positive: bool):
    """记录用户反馈 (FeedbackReranker)"""
    try:
        from rerankers import get_reranker
        fb = get_reranker("feedback")
        fb.record_feedback(chunk_id, is_positive)
        return {"ok": True, "feedback": "positive" if is_positive else "negative"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_config() -> Dict:
    """获取当前配置"""
    return {
        "strategy": config.RETRIEVAL_STRATEGY,
        "vector_weight": config.VECTOR_WEIGHT,
        "fulltext_weight": config.FULLTEXT_WEIGHT,
        "embedding_model": config.EMBEDDING_MODEL,
        "reranker": config.RERANKER,
        "fusion_method": config.FUSION_METHOD,
        "top_k_recall": config.TOP_K_RECALL,
        "top_k_final": config.TOP_K_FINAL,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 模拟 chunks
    chunks = [
        {"id": 1, "text": "武汉兴图新科电子股份有限公司注册资本7360万元"},
        {"id": 2, "text": "武汉兴图新科法定代表人XXX"},
        {"id": 3, "text": "本次发行募集资金4亿元用于补充流动资金"},
        {"id": 4, "text": "武汉力源信息技术股份有限公司注册资本8000万元"},
        {"id": 5, "text": "报告期内军用领域收入分别为6464.51 14414.16 18780.67 4627.14万元"},
    ]
    _retriever = hybrid_retriever.HybridRetriever()
    _retriever.build_index(chunks)

    q = "武汉兴图新科的注册资本是多少"
    print(f"\n=== V6 问答: {q} ===")
    for s in ["vector", "fulltext", "hybrid"]:
        r = answer_question(q, strategy=s)
        print(f"\n[{s}] 融合={r['fusion_method']} 嵌入={r['embedding_model']} 重排={r['reranker']}")
        print(f"  Top-1: {r['retrieval_results'][0]['text'][:40]} "
              f"score={r['retrieval_results'][0]['score']:.3f}")
