# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-14-修复低质量工业PDF的解析与信息丢失
# 模块职责：混合检索——向量召回 + BM25 召回 → RRF 融合 → 最终排序，返回可溯源的知识片段。

import numpy as np                                               # 向量点积
import config                                                    # 检索超参
from logger import get_logger                                    # 日志
from indexer import encode                                       # 复用嵌入编码

logger = get_logger("rag14.retriever")


def vector_search(query, embeddings, top_k=None, allowed=None):
    """向量召回：查询与块向量均已 L2 归一化，点积即余弦相似度。"""
    top_k = top_k or config.VECTOR_TOP_K                         # 召回条数
    q = encode([query])[0]                                       # 查询向量
    sims = embeddings @ q                                        # 余弦相似度
    if allowed is not None:                                      # 限定候选集合（如只看图形块）
        mask = np.full(len(sims), -1.0, dtype=np.float32)        # 掩码
        mask[list(allowed)] = sims[list(allowed)]                # 仅保留允许项
        sims = mask
    idx = np.argsort(-sims)[:top_k]                              # 取 top-k
    return [(int(i), float(sims[i])) for i in idx if sims[i] > -1.0]


def bm25_search(query, bm25, top_k=None):
    """全文召回：jieba 分词 + BM25 打分。"""
    top_k = top_k or config.BM25_TOP_K                           # 召回条数
    return [(int(i), float(s)) for i, s in bm25.search(query, top_k)]


def rrf_fuse(rankings, k=None):
    """倒数排名融合（Reciprocal Rank Fusion）：多路候选按 1/(k+rank) 累加。"""
    k = k or config.RRF_K                                        # 平滑常数
    scores = {}                                                  # 文档下标 -> 融合分
    for ranking in rankings:                                     # 遍历每一路召回
        for rank, (idx, _score) in enumerate(ranking):           # 按名次累加
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])           # 降序


def weighted_fuse(rankings, weights):
    """加权分数融合：先归一到 [0,1] 再按权重相加（两路分数尺度不同，必须归一）。"""
    normed = []                                                  # 归一化后的各路结果
    for ranking in rankings:
        if not ranking:                                          # 空路跳过
            normed.append({})
            continue
        vals = [s for _, s in ranking]                           # 该路分数
        lo, hi = min(vals), max(vals)                            # 极值
        span = (hi - lo) or 1.0                                  # 极差（防除零）
        normed.append({i: (s - lo) / span for i, s in ranking})  # 归一到 [0,1]
    scores = {}                                                  # 融合分
    for w, d in zip(weights, normed):                            # 按权重累加
        for i, s in d.items():
            scores[i] = scores.get(i, 0.0) + w * s
    return sorted(scores.items(), key=lambda x: -x[1])


def retrieve(query, index, top_k=None, mode="rrf"):
    """混合检索主入口：返回排序后的 (块下标, 融合分) 列表。"""
    top_k = top_k or config.FUSION_TOP_K                         # 进入最终排序的条数
    vec = vector_search(query, index["embeddings"])              # 向量召回
    bm = bm25_search(query, index["bm25"])                       # 全文召回
    if mode == "rrf":                                            # RRF 融合
        fused = rrf_fuse([vec, bm])
    else:                                                        # 加权融合
        fused = weighted_fuse([vec, bm], [config.VECTOR_WEIGHT, config.BM25_WEIGHT])
    logger.info("检索完成：向量 %d 条 / BM25 %d 条 → 融合 %d 条",
                len(vec), len(bm), len(fused))
    return fused[:top_k]


def pick_top_chunks(query, index, final_k=None, mode="rrf"):
    """返回最终送入 LLM 的块对象列表（含来源、页码、类型）。"""
    final_k = final_k or config.FINAL_TOP_K                      # 上下文块数
    fused = retrieve(query, index, mode=mode)                    # 融合排序
    chunks = index["chunks"]                                     # 块表
    picked = []                                                  # 结果
    for idx, score in fused[:final_k]:                           # 取前 k
        c = dict(chunks[idx])                                    # 复制，避免污染缓存
        c["score"] = round(score, 6)                             # 记录分数
        picked.append(c)                                         # 收集
    return picked


def assemble_context(picked, max_chars=None):
    """把检索块拼成带来源标注的上下文，超长时截断。"""
    max_chars = max_chars or config.CONTEXT_MAX_CHARS            # 上限
    parts, total = [], 0                                         # 片段与累计长度
    type_label = {"figure": "图纸", "biblio": "著录项", "text": "正文"}  # 块类型的中文标签
    for c in picked:                                             # 逐块
        head = (f"[{c['chunk_id']} | 来源 {c['doc']} 第{c['page_no']}页 | "
                f"类型 {type_label.get(c['type'], '正文')}]")      # 来源抬头
        piece = head + "\n" + c["text"]                          # 抬头 + 正文
        if total + len(piece) > max_chars:                       # 超长则截断
            piece = piece[: max(0, max_chars - total)]           # 截到上限
        parts.append(piece)                                      # 收集
        total += len(piece)                                      # 累计
        if total >= max_chars:                                   # 已满则停
            break
    return "\n\n".join(parts)
