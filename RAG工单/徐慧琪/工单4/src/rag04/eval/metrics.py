# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""检索与答案质量指标。

相关性判定口径：召回块的页码 ∈ 该题 gold_pages，即视为相关。
precision 按「命中块数 / 实际参与计分的块数」统计（同页多块各计一次）；
recall 与 ndcg 按「页」去重，避免同一页被多次召回刷分。
注意：页码在两本文档间重叠，调用方（runner）必须先按 doc_id 过滤召回块。
"""
from __future__ import annotations

import math

from rag04.schema import Hit


def is_relevant(hit: Hit, gold_pages: list[int]) -> bool:
    return hit.page in set(gold_pages or [])


def _relevant_flags(hits: list[Hit], gold_pages: list[int], k: int) -> list[int]:
    """取前 k 条，首次出现的相关页记 1，重复页记 0。"""
    gold = set(gold_pages or [])
    seen: set[int] = set()
    flags: list[int] = []
    for h in hits[:k]:
        if h.page in gold and h.page not in seen:
            seen.add(h.page)
            flags.append(1)
        else:
            flags.append(0)
    return flags


def precision_at_k(hits: list[Hit], gold_pages: list[int], k: int) -> float:
    """命中相关块数 / min(k, 实际召回数)。"""
    if not hits or k <= 0:
        return 0.0
    take = hits[:k]
    relevant = sum(1 for h in take if is_relevant(h, gold_pages))
    return relevant / len(take)


def recall_at_k(hits: list[Hit], gold_pages: list[int], k: int) -> float:
    """命中相关页数 / 总相关页数。"""
    gold = set(gold_pages or [])
    if not gold or k <= 0:
        return 0.0
    got = {h.page for h in hits[:k] if h.page in gold}
    return len(got) / len(gold)


def hit_rate(results: list[bool]) -> float:
    """至少命中一个相关块的题目占比。"""
    if not results:
        return 0.0
    return sum(1 for r in results if r) / len(results)


def mrr(hits_list: list[list[Hit]], gold_pages_list: list[list[int]]) -> float:
    """第一个相关块排名倒数的均值。"""
    if not hits_list:
        return 0.0
    total = 0.0
    for hits, gold in zip(hits_list, gold_pages_list):
        rr = 0.0
        for rank, h in enumerate(hits, start=1):
            if is_relevant(h, gold):
                rr = 1.0 / rank
                break
        total += rr
    return total / len(hits_list)


def ndcg_at_k(hits: list[Hit], gold_pages: list[int], k: int) -> float:
    """NDCG@k（二值相关性，重复页去重）。"""
    if not hits or k <= 0:
        return 0.0
    flags = _relevant_flags(hits, gold_pages, k)
    dcg = sum(f / math.log2(i + 2) for i, f in enumerate(flags))
    n_ideal = min(len(set(gold_pages or [])), k)
    if n_ideal == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))
    return dcg / idcg if idcg > 0 else 0.0


def verdict(cov: float, threshold: float, strict: bool,
            refused: bool = False) -> bool:
    """答案正确性判定（全局约束 10）。

    严格口径（工单对 id 5 的要求）：4 个部门 + 6 个销售处共 10 个要点必须
    全部命中，缺一不可；一般口径：要点覆盖率 >= threshold。

    RC6：拒答一律不计正确。覆盖率的字面重合会把「复述问句后拒答」误判为对
    （id531 实测：答案含「程家明」但结论是"无法确定"），故先看 refused。
    """
    if refused:
        return False
    if strict:
        return cov >= 1.0
    return cov >= threshold
