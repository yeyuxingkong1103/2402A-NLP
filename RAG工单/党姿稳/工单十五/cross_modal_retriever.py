# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-15-优化技术图纸与文本的跨模态检索流程
# 模块职责：跨模态混合检索主流程——多路召回（文本向量 / 图像增强向量 / 全文 / 图号精确匹配）
#           → 加权融合 → 跨模态重排 → 组装上下文。对应工单"任务二 优化混合检索策略"。

import time                                                      # 分阶段耗时统计
import numpy as np                                               # 向量点积
import config                                                    # 超参
from logger import get_logger                                    # 日志
from indexer import encode                                       # 复用嵌入编码
from query_understanding import parse_visual_refs, locate_figures, build_figure_query
from reranker import rerank                                      # 跨模态重排
from bm25 import tokenize                                        # 是否走 BM25 的判断复用分词

logger = get_logger("rag15.retriever")


def vector_search(query, embeddings, top_k):
    """单路向量召回：查询与块向量均已 L2 归一化，点积即余弦相似度。"""
    q = encode([query])[0]                                       # 查询向量
    sims = embeddings @ q                                        # 相似度
    idx = np.argsort(-sims)[:top_k]                              # top-k
    return [(int(i), float(sims[i])) for i in idx]


def bm25_search(query, bm25, top_k):
    """全文召回。"""
    return [(int(i), float(s)) for i, s in bm25.search(query, top_k)]


def label_route(refs, chunks, figure_hits, base=1.0):
    """图号/页码精确匹配路：把被引用到的图纸块直接抬进候选，避免被向量检索漏掉。"""
    route = []                                                   # 结果
    for i in figure_hits:                                        # 命中的图纸块给高分
        route.append((i, base))
    for i, c in enumerate(chunks):                               # 图纸块的说明书描述页也一同抬高
        if c["type"] == "text" and any(lb in c["text"] for lb in refs["labels"]):
            route.append((i, base * 0.6))                        # 次高分
    return route


def normalize_route(route):
    """把一路召回的分数归一到 [0,1]，便于异源分数加权相加。"""
    if not route:
        return {}
    vals = [s for _, s in route]                                 # 该路分数
    lo, hi = min(vals), max(vals)                                # 极值
    span = (hi - lo) or 1.0                                      # 极差
    return {i: (s - lo) / span for i, s in route}                # 归一化字典


def fuse(routes, weights):
    """加权分数融合：各路先归一化再按权重相加。"""
    fused = {}                                                   # 融合结果
    for name, route in routes.items():                           # 逐路
        w = weights.get(name, 0.0)                               # 该路权重
        for i, s in normalize_route(route).items():              # 归一化后累加
            fused[i] = fused.get(i, 0.0) + w * s
    return sorted(fused.items(), key=lambda x: -x[1])            # 降序


def retrieve_baseline(query, index):
    """优化前的基线检索：仅文本向量 + BM25 简单融合，不含查询理解与重排。

    用于工单"任务三"要求的优化前后对比，说明老流程为何会漏掉图纸页。
    """
    t0 = time.time()                                             # 计时
    routes = {"text_vector": vector_search(query, index["embeddings"], config.VECTOR_TOP_K),
              "bm25": bm25_search(query, index["bm25"], config.BM25_TOP_K)}
    fused = fuse(routes, {"text_vector": 0.5, "bm25": 0.5})      # 等权融合
    return fused[: config.FUSION_TOP_K], {"refs": {"has_visual": False, "labels": [],
                                                   "pages": [], "keywords": []},
                                          "figure_hits": [], "enhanced_query_used": False,
                                          "route_sizes": {k: len(v) for k, v in routes.items()},
                                          "figure_desc_chars": 0,
                                          "timings": {"查询理解": 0.0, "多路召回": 0.0,
                                                      "加权融合": 0.0, "跨模态重排": 0.0,
                                                      "检索合计": round(time.time() - t0, 2)}}


def retrieve(query, index, optimized=True):
    """跨模态检索主入口，返回 (候选列表, 诊断信息)。optimized=False 走基线流程。"""
    if not optimized:                                            # 基线（优化前）
        return retrieve_baseline(query, index)
    chunks = index["chunks"]                                     # 块表
    t0 = time.time()                                             # 查询理解计时
    refs = parse_visual_refs(query)                              # 1) 查询理解：识别视觉引用
    figure_hits = locate_figures(refs, chunks) if refs["has_visual"] else []  # 2) 定位图纸
    enhanced, desc = build_figure_query(query, refs, chunks, figure_hits)     # 3) 构造图像增强查询

    t1 = time.time()                                             # 召回计时
    routes = {}                                                  # 多路召回结果
    routes["text_vector"] = vector_search(query, index["embeddings"], config.VECTOR_TOP_K)
    routes["bm25"] = bm25_search(query, index["bm25"], config.BM25_TOP_K)
    if enhanced != query:                                        # 有图像增强查询才走该路
        routes["figure_vector"] = vector_search(enhanced, index["embeddings"], config.FIGURE_TOP_K)
    if figure_hits:                                              # 图号精确匹配路
        routes["figure_label"] = label_route(refs, chunks, figure_hits)
    t2 = time.time()                                             # 融合计时

    fused = fuse(routes, config.ROUTE_WEIGHTS)                   # 4) 加权融合
    fused = fused[:config.FUSION_TOP_K]                          # 截断候选
    t3 = time.time()                                             # 重排计时
    if config.RERANK_ENABLED:                                    # 5) 跨模态重排
        ranked = rerank(query, fused, chunks, config.RERANK_TOP_K)
    else:
        ranked = fused
    ranked = _apply_label_boost(ranked, chunks, figure_hits)     # 6) 被引用图纸块再加成
    diag = {"refs": refs, "figure_hits": figure_hits,            # 诊断信息（供报告与界面）
            "enhanced_query_used": enhanced != query,
            "route_sizes": {k: len(v) for k, v in routes.items()},
            "figure_desc_chars": len(desc),
            "timings": {"查询理解": round(t1 - t0, 2),            # 各阶段耗时（秒）
                        "多路召回": round(t2 - t1, 2),
                        "加权融合": round(t3 - t2, 2),
                        "跨模态重排": round(time.time() - t3, 2),
                        "检索合计": round(time.time() - t0, 2)}}
    return ranked, diag


def _apply_label_boost(ranked, chunks, figure_hits):
    """对命中图号/页码的图纸块额外加成，确保 Top-3 内出现被引用的图纸。"""
    if not figure_hits:
        return ranked
    boost = config.FIGURE_LABEL_BOOST                           # 加成值
    hit_set = set(figure_hits)                                   # 命中集合
    out = [(i, round(s + (boost if i in hit_set else 0.0), 6)) for i, s in ranked]
    out.sort(key=lambda x: -x[1])                                # 重新排序
    return out


def pick_top_chunks(query, index, final_k=None, optimized=True):
    """返回最终送入 LLM 的块列表（附综合分）与诊断信息。"""
    final_k = final_k or config.FINAL_TOP_K                      # 上下文块数
    ranked, diag = retrieve(query, index, optimized=optimized)   # 检索
    chunks = index["chunks"]                                     # 块表
    picked = []                                                  # 结果
    for idx, score in ranked[:final_k]:                          # 取前 k
        c = dict(chunks[idx])                                    # 复制
        c["score"] = score                                       # 综合分
        picked.append(c)
    diag["top_chunks"] = [c["chunk_id"] for c in picked]         # 记录 Top 块
    return picked, diag


TYPE_LABEL = {"figure": "图纸", "biblio": "著录项", "text": "正文"}  # 类型中文名


def assemble_context(picked, max_chars=None):
    """组装上下文：图纸块置顶并显式标注，配合 Prompt 引导模型结合图纸推理。"""
    max_chars = max_chars or config.CONTEXT_MAX_CHARS            # 上限
    ordered = sorted(picked, key=lambda c: c["type"] != "figure")  # 图纸块排在最前
    parts, total = [], 0                                         # 片段与累计长度
    for c in ordered:                                            # 逐块
        head = (f"[{c['chunk_id']} | 来源 {c['doc']} 第{c['page_no']}页 | "
                f"类型 {TYPE_LABEL.get(c['type'], '正文')}]")     # 抬头
        piece = head + "\n" + c["text"]                          # 抬头 + 正文
        if total + len(piece) > max_chars:                       # 超长截断
            piece = piece[: max(0, max_chars - total)]
        parts.append(piece)                                      # 收集
        total += len(piece)                                      # 累计
        if total >= max_chars:                                   # 满则停
            break
    return "\n\n".join(parts)
