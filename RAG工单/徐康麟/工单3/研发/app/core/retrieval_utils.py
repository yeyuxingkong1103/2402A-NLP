# -*- coding: utf-8 -*-
"""工单3 检索工具：RRF 融合、加权、按文件过滤、命中判定（设计/接口设计.md §3.11 冻结）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

要点：
    * ``rrf_fuse``：``score = w_v/(k+rank_v) + w_b/(k+rank_b)``，缺一路则该项为 0；
    * ``apply_boosts``：融合分 × 系数（表格块 / 含数字块 / 关键词覆盖块，可叠加），系数原样写回；
    * ``filter_ids_by_file``：把 ``file_names`` 映射成 id 集合，供**检索前**硬过滤；
    * ``is_evidence_hit``：**命中判定的唯一入口**，委托 ``text_utils.evidence_contains``，
      判据是「证据原文是否落在返回的 chunk 里」，**不是**「引用页 == evidence_pages」。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .chunker import Chunk
from .errors import RetrievalError
from .text_utils import (chunk_field, evidence_contains, extract_numbers, keyword_coverage, text_digest,
                         tokenize)

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"

# 「单路高分救援」常数：RRF 只看名次，会出现「只在 BM25 排第 4、向量没进前 100」的块
# 被大量「两路都中等」的块挤到 40 名开外（实测 PDF1 物理 129/152 的证据块正是这种情况）。
# 因此对「任一路前 N 名」的块额外加 rescue_weight/(RESCUE_K + 最佳名次)。
RESCUE_K = 10


def _lazy_logger(logger: Any, module: str = "retrieval_utils") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


def rrf_fuse(
    vector_hits: Sequence[tuple[str, float]],
    bm25_hits: Sequence[tuple[str, float]],
    *,
    k: int = 60,
    weights: tuple[float, float] = (1.0, 1.0),
    rescue_weight: float = 0.0,
) -> list[tuple[str, float]]:
    """RRF 融合两路召回（输入按各自排名，score 不参与计算，只用名次）。

    参数扩展说明（设计 §9 允许新增带默认值的参数）：``rescue_weight`` 默认 0.0 = **纯 RRF（冻结语义不变）**；
    取正值时启用「单路高分救援」——
    ``score += rescue_weight / (RESCUE_K + min(向量名次, BM25 名次))``。

    为什么需要它（实测证据，T5）：题 33/260 的证据块 ``招股说明书1_p0129_x311`` 在 BM25 排第 4，
    但向量前 100 都没有它 → 纯 RRF 下只拿到 ``1/(60+4)=0.0156``，被大量「两路都中等」的块挤到第 42 名；
    题 34 的证据块同理（BM25 第 2 / 向量第 87 → RRF 第 10，取 top-5 时丢掉）。
    救援项让「某一路非常靠前」的块不被另一路的缺席淹没。
    """
    kk = max(int(k), 1)
    w_vector, w_bm25 = float(weights[0]), float(weights[1])
    fused: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for rank, (chunk_id, _score) in enumerate(vector_hits, start=1):
        fused[chunk_id] = fused.get(chunk_id, 0.0) + w_vector / (kk + rank)
        best_rank[chunk_id] = min(best_rank.get(chunk_id, 10 ** 9), rank)
    for rank, (chunk_id, _score) in enumerate(bm25_hits, start=1):
        fused[chunk_id] = fused.get(chunk_id, 0.0) + w_bm25 / (kk + rank)
        best_rank[chunk_id] = min(best_rank.get(chunk_id, 10 ** 9), rank)
    if float(rescue_weight) > 0.0:
        for chunk_id, rank in best_rank.items():
            fused[chunk_id] += float(rescue_weight) / (RESCUE_K + rank)
    ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return ranked


def keyword_coverage_of(query_tokens: Sequence[str], chunk: Chunk) -> float:
    """查询词在该块中的覆盖率（供加权与重排共用）。"""
    return keyword_coverage(query_tokens, tokenize(chunk.content))


def apply_boosts(
    candidates: Sequence[tuple[str, float]],
    chunk_map: Mapping[str, Chunk],
    *,
    table_boost: float,
    numeric_boost: float,
    keyword_boost: float,
    query_tokens: Sequence[str],
    query_expects_numeric: bool | None = None,
) -> list[tuple[str, float, dict[str, float]]]:
    """在融合分上乘系数：表格块 / 含数字块 / 有查询词覆盖的块（三者可叠加）。

    参数扩展说明：``query_expects_numeric`` 默认 ``None`` = **旧语义**（只要块内含数字就加权）；
    显式传 ``False`` 时**不对数字加分**（问题本身不是在问数字，例如「上游涉及哪些企业」）。

    为什么需要它（T5 实测）：题 34「电子信息行业的上游涉及哪些企业」的证据是**无数字的正文段**，
    而大量含数字的表格/财务段落会被 ``numeric_boost`` 抬上去，把证据段挤出 top-5；
    题 207/260 的正文证据同理被数字加权相对压制。

    返回 ``[(chunk_id, boosted_score, boosts)]``，``boosts`` 里三个键**始终存在**：
    生效时写实际系数，未生效写 ``1.0``（便于日志与 T9 归因，不用「缺键」表达未生效）。
    """
    out: list[tuple[str, float, dict[str, float]]] = []
    for chunk_id, score in candidates:
        chunk = chunk_map.get(chunk_id)
        if chunk is None:
            continue                                     # 元数据缺失的候选直接跳过（上层会记 WARN 计数）
        boosts = {"table": 1.0, "numeric": 1.0, "keyword": 1.0}
        factor = 1.0
        if chunk.type == "table":
            boosts["table"] = float(table_boost)
            factor *= float(table_boost)
        has_numbers = bool(extract_numbers(chunk.content))
        numeric_enabled = True if query_expects_numeric is None else bool(query_expects_numeric)
        if has_numbers and numeric_enabled:
            boosts["numeric"] = float(numeric_boost)
            factor *= float(numeric_boost)
        coverage = keyword_coverage_of(query_tokens, chunk)
        if coverage > 0.0:
            boosts["keyword"] = float(keyword_boost)
            factor *= float(keyword_boost)
        boosts["keyword_coverage"] = round(coverage, 4)
        out.append((chunk_id, float(score) * factor, boosts))
    out.sort(key=lambda item: (-item[1], item[0]))
    return out


def filter_ids_by_file(chunk_map: Mapping[str, Chunk],
                       file_names: Sequence[str] | None) -> set[str] | None:
    """把 ``file_names`` 映射为 chunk_id 集合（``None`` 表示不过滤）。

    * ``file_names`` 为空/None → 返回 ``None``（不过滤）；
    * 给了具体文件名 → 返回这些文件的 id 集合；**空的集合**代表该过滤条件下无候选，
      由调用方直接返回空列表（禁止退化成「不过滤」）。
    """
    if not file_names:
        return None
    wanted = {str(name) for name in file_names if str(name).strip()}
    if not wanted:
        return None
    return {chunk_id for chunk_id, chunk in chunk_map.items() if chunk.file_name in wanted}


def merge_unique(*ranked: Sequence[tuple[str, float]], top_k: int) -> list[tuple[str, float]]:
    """按顺序合并多路结果并去重（保留首个出现的较高分），截断到 ``top_k``。"""
    merged: dict[str, float] = {}
    for series in ranked:
        for chunk_id, score in series:
            if chunk_id not in merged or score > merged[chunk_id]:
                merged[chunk_id] = float(score)
    ordered = sorted(merged.items(), key=lambda item: (-item[1], item[0]))
    return ordered[: max(int(top_k), 1)]


def is_evidence_hit(chunks: Sequence[Any], evidence: str, *, threshold: float = 0.90,
                    strict: bool = False, logger: Any = None) -> bool:
    """证据原文是否落在任一返回块中（**唯一实现**，委托 ``text_utils.evidence_contains``）。

    ``chunks`` 兼容三种形态：``RetrievedChunk`` / ``Chunk`` 对象、**dict/Mapping**（JSONL 回读、
    HTTP 响应、评估产物回读都是这种）、纯文本字符串。

    t14 修复（设计 §17，实测缺陷）：
        * 旧实现写 ``getattr(chunk, "content", None)`` → 传 dict 时取到 ``None``，循环走完
          **静默返回 False**，把「形状不匹配」伪装成「真实未命中」；
        * 现在统一走 ``text_utils.chunk_field``；元素既非 Mapping 也无 ``content`` 时**不再静默跳过**——
          记 ``retrieval_utils.is_evidence_hit.degrade`` WARN 事件（含形状与原因），``strict=True`` 时抛错。
    """
    log = _lazy_logger(logger)
    if not str(evidence or "").strip():
        return False
    skipped = 0
    with log.enter("is_evidence_hit", {"chunks": len(chunks), "evidence_chars": len(str(evidence))}) as span:
        for index, chunk in enumerate(chunks):
            content = chunk_field(chunk, "content")
            if content is None:
                skipped += 1
                shape = type(chunk).__name__
                log.log_event("retrieval_utils.is_evidence_hit.degrade", level="WARNING", index=index,
                              shape=shape, keys=(sorted(chunk) if isinstance(chunk, Mapping) else None),
                              reason="块既非 Mapping 也没有 content 字段，无法参与命中判定",
                              fallback="该块跳过；结果不得解释为「真实未命中」")
                if strict:
                    raise RetrievalError(
                        f"第 {index} 个块形状不支持命中判定（shape={shape}）；strict=True 时不静默跳过",
                        detail={"index": index, "shape": shape})
                continue
            if evidence_contains(str(content), evidence, threshold=threshold):
                log.log_event("retrieval_utils.is_evidence_hit", hit=True, index=index,
                              evidence_digest=text_digest(evidence, limit=40))
                span.set_output({"hit": True, "index": index, "skipped": skipped})
                return True
        if skipped:
            log.log_event("retrieval.hit_shape_mismatch", level="WARNING", skipped=skipped,
                          total=len(chunks),
                          note="存在形状不匹配的块被跳过：命中结果只代表可解析的块")
        span.set_output({"hit": False, "skipped": skipped})
        return False


def summarize_hits(chunks: Sequence[Any], *, limit: int = 5) -> list[dict[str, Any]]:
    """检索片段摘要（写日志/报告用）：chunk_id、文件名、页码、类型、分数、内容摘要。"""
    summary: list[dict[str, Any]] = []
    for rank, chunk in enumerate(list(chunks)[: max(int(limit), 1)], start=1):
        summary.append({
            "rank": rank,
            "chunk_id": getattr(chunk, "chunk_id", ""),
            "file_name": getattr(chunk, "file_name", ""),
            "page": getattr(chunk, "page", 0),
            "type": getattr(chunk, "type", ""),
            "score": round(float(getattr(chunk, "score", 0.0) or 0.0), 6),
            "content_digest": text_digest(str(getattr(chunk, "content", "") or ""), limit=80),
        })
    return summary
