"""在线检索编排（services 层）：Query 改写 → 调用 rag.retriever 执行检索。

分层说明（G3 修复）：Query 改写依赖 LLM（services/llm_service），按 CLAUDE.md §3
的分层约定 services → rag 合法、rag → services 违规，故改写逻辑上移到本模块；
纯检索执行（向量化 / Milvus / 重排 / 阈值）仍在 rag/retriever.py。

阶段 4 优化：
- 改写调用使用短超时（LLM_REWRITE_TIMEOUT，默认 3s），失败回退原问题不拖慢首 token；
- 检索结果 Redis 缓存（RETRIEVAL_CACHE_TTL，默认 300s，0=禁用），知识库变更时由
  knowledge_service 调 clear_retrieval_cache 失效。

被谁调用：rag_service（正式问答）、knowledge_service.search（管理端调试/评测）。
依赖：llm_service（改写）、rag.retriever（真正检索）、db.redis（结果缓存）。
"""
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import settings
from src.core.logging import get_logger
from src.db import redis as redis_db
from src.rag import prompt as prompt_utils
from src.rag.retriever import retrieve
from src.services import llm_service

logger = get_logger("rag.retrieval")


def rewrite_query(question: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    """Query 改写/扩写；失败时回退原问题。"""
    if not settings.query_rewrite_enabled:
        return question
    history_text = "\n".join(
        f"{'用户' if m.get('role') == 'user' else '咨询师'}：{m.get('content', '')}"
        for m in (history or [])[-4:]
    ) or "（无）"
    # temperature 0.2：改写属于"确定性"任务，低温可避免模型自由发挥偏离原意；
    # 短超时 + 短输出上限是为了不拖慢首 token（失败即回退原问题）。
    rewritten = llm_service.simple_complete(
        prompt_utils.QUERY_REWRITE_PROMPT.format(history=history_text, question=question),
        temperature=0.2, max_tokens=128, timeout=settings.llm_rewrite_timeout,
    ).strip().replace("\n", " ")
    if not rewritten or len(rewritten) > 300:
        return question
    # 改写结果与原文合并检索，兼顾召回与精度
    merged = rewritten if rewritten == question else f"{rewritten} {question}"
    logger.info("Query 改写：%s -> %s", question, merged)
    return merged


def _cache_fingerprint(question: str, history: Optional[List[Dict[str, str]]],
                       top_k: Optional[int], rerank_top_n: Optional[int]) -> str:
    """缓存指纹：问题 + 最近 2 条历史 + 检索参数（历史变动视为新查询）。"""
    tail = "|".join((h.get("content") or "") for h in (history or [])[-2:])
    raw = f"{question}|{tail}|{top_k}|{rerank_top_n}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def search_with_query_rewrite(persona_id: int, question: str,
                              history: Optional[List[Dict[str, str]]] = None,
                              top_k: Optional[int] = None,
                              rerank_top_n: Optional[int] = None
                              ) -> Tuple[List[Dict[str, Any]], str]:
    """改写后检索，返回（重排片段, 改写后的 query）。命中缓存时跳过改写+检索+重排。"""
    # TTL<=0 视为禁用缓存（调试/评测时需要每次都真实检索）
    if settings.retrieval_cache_ttl > 0:
        fingerprint = _cache_fingerprint(question, history, top_k, rerank_top_n)
        cached = redis_db.get_retrieval_cache(persona_id, fingerprint)
        if cached is not None:
            logger.info("检索缓存命中 persona=%s query=%s...", persona_id, question[:30])
            return cached["hits"], cached["rewritten"]

    rewritten = rewrite_query(question, history)
    hits = retrieve(persona_id, rewritten, top_k=top_k, rerank_top_n=rerank_top_n)

    if settings.retrieval_cache_ttl > 0:
        # 回填缓存：下次同样的问题（历史也相同）可直接命中，省掉一次 LLM 改写 + 向量检索 + 重排
        redis_db.cache_retrieval(
            persona_id, fingerprint,
            {"hits": hits, "rewritten": rewritten},
        )
    return hits, rewritten
