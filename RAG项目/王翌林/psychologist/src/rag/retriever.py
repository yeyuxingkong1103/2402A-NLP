"""在线检索执行：BGE-M3 向量化 → Milvus 混合检索 → 重排序 → 阈值过滤。

这四步构成一条「由宽到窄」的检索漏斗，每一步都在压缩候选数量：
    ① 向量化   —— 1 条 query → 1 个 1024 维归一化向量（数量不变，形态变了）
    ② 混合检索 —— 稠密+关键词双路召回 top_k 条候选（如 20 条），目标是"高召回、不漏"
    ③ 重排序   —— 交叉编码器逐条精排后截断到 rerank_top_n 条（如 5 条），目标是"高精度、不滥"
    ④ 阈值过滤 —— 用相似度阈值砍掉弱相关噪声，只留真正相关的
之所以要"先宽后窄"，是因为单靠向量召回精度不足，而逐条精排又太贵，
只能先用便宜的召回保证不漏，再用昂贵的精排保证不滥。

Query 改写已上移到 services/retrieval_service.py（G3 分层修复，本模块不依赖 services）。
"""
from typing import Any, Dict, List, Optional

from src.core.config import settings
from src.core.logging import get_logger
from src.db import milvus as milvus_db
from src.rag.embedder import get_embedder
from src.rag.reranker import get_reranker

logger = get_logger("rag.retriever")


def retrieve(persona_id: int, query: str, top_k: Optional[int] = None,
             rerank_top_n: Optional[int] = None,
             threshold: Optional[float] = None,
             use_hybrid: bool = True) -> List[Dict[str, Any]]:
    """执行检索全流程，返回重排后的知识片段。

    检索漏斗四阶段，候选数量逐级收窄（top_k → rerank_top_n → 阈值过滤）：
      1. 向量化：把 query 编码成归一化向量（供 Milvus COSINE 检索）。
      2. 混合检索：Milvus 稠密 + 稀疏双路召回 top_k 个候选。混合检索结合
         语义匹配与关键词匹配，比纯向量更鲁棒，尤其对专有名词、医学术语等
         词汇敏感场景；混合路不可用（返回空）时自动降级为纯稠密检索。
      3. 重排：用交叉编码器对候选逐条精排，得到 (0,1) 的相关度分数，
         再截断到 rerank_top_n 条。
      4. 阈值过滤：只保留分数 ≥ threshold 的片段，过滤掉弱相关噪声。

    参数 use_hybrid=False 可强制走纯稠密检索（调试或稀疏索引缺失时使用）。
    """
    # 三个"数量/阈值"参数都允许调用方覆盖，缺省时回落到全局配置（便于按场景调参）
    top_k = top_k or settings.retrieve_top_k              # 召回条数（漏斗最宽处）
    rerank_top_n = rerank_top_n or settings.rerank_top_n  # 精排后保留条数（漏斗中段）
    threshold = settings.similarity_threshold if threshold is None else threshold  # 阈值过滤线（漏斗最窄处）

    # ① 向量化：query → 归一化稠密向量（BGE-M3 无需指令前缀）
    embedder = get_embedder()
    query_vector = embedder.encode_query(query)

    # ② 混合检索：稠密+稀疏双路召回。稀疏路对关键词/专有名词更敏感，与稠密路互补；
    #    若混合路拿不到结果（稀疏索引未建 / 服务异常 / 路由失败），则降级为纯稠密检索，
    #    保证"至少不断链"，代价是失去关键词精确匹配能力。
    if use_hybrid:
        candidates = milvus_db.hybrid_search_knowledge(persona_id, query_vector, query, top_k=top_k)
        if not candidates:
            # 混合检索不可用时降级为稠密检索
            candidates = milvus_db.dense_search_knowledge(persona_id, query_vector, top_k=top_k)
    else:
        candidates = milvus_db.dense_search_knowledge(persona_id, query_vector, top_k=top_k)

    if not candidates:
        return []  # 召回为空：直接返回空，交给上层用 NO_KNOWLEDGE_HINT 兜底提示词

    # ③ 重排：交叉编码器精排 + 截断到 rerank_top_n，把"宽召回"收敛为"高精度候选"
    reranker = get_reranker()
    ranked = reranker.rerank(query, candidates, top_n=rerank_top_n)

    # ④ 阈值过滤：rerank_score 已是 (0,1) 的 sigmoid 值，与 threshold 同量纲，可直接比较
    filtered = [r for r in ranked if r.get("rerank_score", 0) >= threshold]
    if not filtered:
        # 全部低于阈值时保留最高分片段，避免无依据回答
        # （完全空着会让 LLM 被迫自由发挥，更易幻觉；给一条最高分至少提供一点依据）
        #
        # ⚠ 已知缺陷：rerank_score 在 reranker.py 里做过 sigmoid 归一化，恒 > 0。
        #   因此下面的 `> 0` 判断只要 ranked 非空就恒为真，兜底必然命中一条。
        #   后果：无论阈值设多高、候选多不相关，本函数都至少返回 1 条，
        #   阈值过滤永远无法得到"空结果"，即阈值实际只能"减少"条数、不能"归零"。
        #   若要真正生效，应改用原始 logit 判断（logit 可为负）或干脆去掉 >0 兜底。
        filtered = ranked[:1] if ranked and ranked[0].get("rerank_score", 0) > 0 else []
    logger.info("检索 persona=%s 召回=%d 重排后=%d 通过阈值=%d",
                persona_id, len(candidates), len(ranked), len(filtered))
    return filtered
