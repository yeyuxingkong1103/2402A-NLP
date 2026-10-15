# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-15-优化技术图纸与文本的跨模态检索流程
# 模块职责：跨模态重排器。用 bge-reranker-base（Cross-Encoder）对"问题-候选块"逐对打分，
#           替代纯向量相似度做精细化排序，解决故障 2（检索到页面但无法关联图与查询词）。

import config                                                    # 配置
from logger import get_logger                                    # 日志

logger = get_logger("rag15.rerank")

_MODEL = None                                                    # Cross-Encoder 单例


def load_reranker():
    """惰性加载 bge-reranker-base。max_length 限制在 320 以保证 CPU 上的重排耗时可控。"""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import CrossEncoder           # 延迟导入
        logger.info("加载重排模型 %s ...", config.RERANK_MODEL_PATH.split("\\")[-1])
        _MODEL = CrossEncoder(config.RERANK_MODEL_PATH,         # 本地加载
                              max_length=config.RERANK_MAX_LENGTH)
        logger.info("重排模型加载完成")
    return _MODEL


def score_pairs(query, texts):
    """返回 query 与每个 text 的相关性分数。只用文本前 RERANK_TEXT_CHARS 字，控制耗时。"""
    model = load_reranker()                                      # 单例
    pairs = [(query, t[:config.RERANK_TEXT_CHARS]) for t in texts]  # 构造问题-文本对
    return [float(s) for s in model.predict(pairs, batch_size=8)]   # 逐对打分


def rerank(query, candidates, chunks, top_k=None):
    """对候选 (块下标, 融合分) 做重排，返回新的 (块下标, 综合分) 列表。

    综合分 = RERANK_WEIGHT × 归一化重排分 + FUSION_WEIGHT × 归一化融合分，
    使跨模态重排结果与多路召回结果都能体现，而不是完全丢弃召回信息。
    """
    top_k = top_k or config.RERANK_TOP_K                         # 重排后保留条数
    if not candidates:                                           # 空候选
        return []
    idxs = [i for i, _ in candidates]                            # 候选下标
    texts = [chunks[i]["text"] for i in idxs]                    # 候选文本
    raw = score_pairs(query, texts)                              # 重排原始分

    def _norm(vals):                                             # 归一到 [0,1]
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        return [(v - lo) / span for v in vals]

    n_rerank = _norm(raw)                                        # 归一化重排分
    n_fusion = _norm([s for _, s in candidates])                 # 归一化融合分
    fused = []                                                   # 综合结果
    for k, idx in enumerate(idxs):                               # 逐候选
        combined = (config.RERANK_WEIGHT * n_rerank[k]
                    + config.FUSION_WEIGHT * n_fusion[k])        # 加权综合
        fused.append((idx, round(combined, 6)))                  # 收集
    fused.sort(key=lambda x: -x[1])                              # 降序
    logger.info("重排完成：%d 条候选 → 前 %d 条（最高分 %.4f）",
                len(candidates), min(top_k, len(fused)), fused[0][1] if fused else 0.0)
    return fused[:top_k]
