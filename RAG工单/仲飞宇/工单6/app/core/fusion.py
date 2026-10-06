# 工单编号：人工智能NLP-RAG-混合检索任务
# 工单06 - 混合检索（向量检索 / 全文检索 / 混合检索）
"""
融合：把向量路与关键词路的候选合成一份排序。工单点名了两种融合算法。

    rrf       倒数排名融合（Reciprocal Rank Fusion）—— 就是工单说的"投票机制"
    weighted  加权平均（先把两路分数各自归一化，再按权重相加）

【为什么默认是 RRF 而不是加权平均】
两路的分数**量纲根本不同**：COSINE ∈ [-1,1]，BM25 无上界（实测 0~20+），
Milvus 的 WeightedRanker 又是另一套。要线性加权就得先归一化，而归一化方式
（min-max / z-score / 除以最大值）会**直接改变排序** —— 那是"调参调出来的效果"，
不是融合本身的贡献，也很难在文档里讲清。

RRF 只用**排名**、不用分数：`score(d) = Σ_src w_src / (k + rank_src(d))`，
天然免疫尺度差异，且对单路返回条数的变化不敏感。所以它是默认。
加权平均作为**对照**保留（工单明确点名了它），但报告里会标注它的不稳。

【一个必须小心的细节】RRF 的分数只有 1/(60+1) ≈ 0.016 量级。
如果后续还有代码对它做 `(score+1)/2` 那种"COSINE 归一化"，就会被压成
近乎常数 → 排序悄悄退化成"只按另一个信号排"，而且**不报错**。
所以每条命中都带 `score_kind`，归一化前必须先看它（见 rerank.lexical_score）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from app.core.vectorstore import SearchHit

# RRF 的平滑常数（原论文取 60）。越大越"平"，排名靠后的也能拿到份量。
RRF_K = 60


@dataclass
class FusionResult:
    hits: list[SearchHit]
    method: str = "rrf"
    n_dense: int = 0
    n_keyword: int = 0
    n_shared: int = 0          # 两路都召回到的块数
    n_total: int = 0           # 去重后候选总数

    def as_dict(self) -> dict:
        return {"fusion": self.method, "n_dense": self.n_dense,
                "n_keyword": self.n_keyword, "n_shared": self.n_shared,
                "n_fused": self.n_total}


def minmax(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return lo, lo + 1.0        # 全同分时不放大噪声，退化成常数
    return lo, hi


def _merge_by_chunk(*lists: Sequence[SearchHit]) -> dict[int, SearchHit]:
    """按 chunk_id 去重合并，保留**第一次**出现的那份（便于溯源来自哪一路）。"""
    out: dict[int, SearchHit] = {}
    for lst in lists:
        for h in lst:
            out.setdefault(h.chunk_id, h)
    return out


def _tag(hit: SearchHit, *, kind: str, src: str, score: float,
         base_score: float, base_kind: str) -> SearchHit:
    """复制一份并改写分数与来源标记（SearchHit 是 dataclass，直接改会污染调用方）。

    【base_score / base_kind 是给重排器用的】融合分是"融合器"的尺度，
    重排做归一化时要知道这条命中**原本**来自哪把尺子（见 SearchHit 上的说明）。
    关键词路独有的命中没有 COSINE 分，按 `base_kind="none"`（分量为 0）处理 ——
    它们是"向量没召回到、关键词补上的"，不该在稠密分那一项上占便宜。
    """
    from dataclasses import replace as _replace
    return _replace(hit, score=score, score_kind=kind, retrieval_src=src,
                    base_score=base_score, base_kind=base_kind)


def _base_tag(hit: SearchHit, fused_score: float, from_dense: bool,
              *, kind: str = "rank", kw_base: float | None = None) -> SearchHit:
    """打上融合分，同时把**融合前的原始分**留一份给重排器做归一化。

    来自稠密路的命中保留原 COSINE 分（重排时 `(cos+1)/2`，与工单02 一致）。
    只在关键词路出现的命中，其 BM25 分先被**映射进稠密路归一化后的区间**
    （`kw_base`，由调用方算好），标成 `unit` 直接使用。

    【为什么不是记 0 分】记 0 的话，关键词路独有的候选在"稠密分"那一项上永远
    垫底，只在覆盖率极高时才可能翻盘 —— 实测结果是**权重怎么调都不影响结果**
    （w 从 0 到 0.6 页召回一模一样），等于混合检索没生效。
    映射到同一区间后，两路才是真的在同一把尺子上竞争。
    """
    if from_dense:
        return _tag(hit, kind=kind, src="fused", score=fused_score,
                    base_score=hit.score, base_kind=hit.score_kind)
    return _tag(hit, kind=kind, src="fused", score=fused_score,
                base_score=(kw_base if kw_base is not None else 0.0), base_kind="unit")


def _kw_bases_in_dense_range(dense: Sequence[SearchHit],
                             keyword: Sequence[SearchHit]) -> dict[int, float]:
    """把关键词路的 BM25 分**映射进稠密路归一化后的区间** `[d_lo, d_hi]`。

    两路的原始分不可比（COSINE 挤在 0.75~0.83，BM25 可到 20+）。但重排时的
    "稠密分量"是一个 [0,1] 的归一化值 —— 只要把 BM25 也拉到同一个区间，
    两路就能在同一把尺子上竞争：关键词路最强的候选 ≈ 稠密路最强的候选。
    """
    if not dense or not keyword:
        return {}
    d_norm = [(h.score + 1.0) / 2.0 for h in dense]
    d_lo, d_hi = minmax(d_norm)
    k_lo, k_hi = minmax([h.score for h in keyword])
    span = (k_hi - k_lo) or 1.0
    return {h.chunk_id: d_lo + (d_hi - d_lo) * ((h.score - k_lo) / span)
            for h in keyword}


def rrf_fuse(dense: Sequence[SearchHit], keyword: Sequence[SearchHit], *,
             w_keyword: float = 0.5, k: int = RRF_K) -> FusionResult:
    """倒数排名融合（投票机制）。

    【权重为 0 的那一路必须**整条不入候选**】否则它的命中会带着 0 分留在候选集里
    （`setdefault` 会照样登记），再被 min-max 归一化放大 —— 实测 `w_keyword=0`
    本应完全等价于纯稠密路，却因为泄漏而把页召回从 60.4% 拖到 38.5%。
    这类"权重明明设了 0 却还在起作用"的错误不会报错，只能靠断言/测试发现。
    """
    w_dense = 1.0 - w_keyword
    scores: dict[int, float] = {}
    base: dict[int, SearchHit] = {}

    for lst, w in ((dense, w_dense), (keyword, w_keyword)):
        if w <= 0.0:
            continue                            # 见上：零权重一路整体排除
        for rank, h in enumerate(lst):          # rank 从 0 起，故用 k + rank + 1
            scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + w / (k + rank + 1)
            base.setdefault(h.chunk_id, h)

    dense_ids = {h.chunk_id for h in dense}
    shared = len(dense_ids & {h.chunk_id for h in keyword})
    kw_bases = _kw_bases_in_dense_range(dense, keyword)
    ordered = sorted(scores, key=lambda cid: -scores[cid])
    hits = [_base_tag(base[cid], scores[cid], cid in dense_ids,
                      kw_base=kw_bases.get(cid)) for cid in ordered]
    return FusionResult(hits=hits, method="rrf", n_dense=len(dense),
                        n_keyword=len(keyword), n_shared=shared,
                        n_total=len(hits))


def weighted_fuse(dense: Sequence[SearchHit], keyword: Sequence[SearchHit], *,
                  w_keyword: float = 0.5) -> FusionResult:
    """加权平均（分数各自 min-max 归一化后相加）。

    【必须写进文档的限度】min-max 依赖**候选池的组成** —— 换个 pool_size，
    同一个块的归一化分就会变，排序可能跟着变。所以它只作为对照，
    且报告里会与 RRF 并列，不会用它当交付默认。
    """
    w_dense = 1.0 - w_keyword
    dense = list(dense) if w_dense > 0 else []
    keyword = list(keyword) if w_keyword > 0 else []     # 零权重一路整体排除
    all_hits = _merge_by_chunk(dense, keyword)

    d_lo, d_hi = minmax([h.score for h in dense])
    k_lo, k_hi = minmax([h.score for h in keyword])

    dense_map = {h.chunk_id: h for h in dense}
    kw_map = {h.chunk_id: h for h in keyword}

    scores: dict[int, float] = {}
    for cid in all_hits:
        v = 0.0
        if cid in dense_map:
            v += w_dense * (dense_map[cid].score - d_lo) / (d_hi - d_lo)
        if cid in kw_map:
            v += w_keyword * (kw_map[cid].score - k_lo) / (k_hi - k_lo)
        scores[cid] = v

    shared = len(set(dense_map) & set(kw_map))
    ordered = sorted(scores, key=lambda cid: -scores[cid])
    hits = [_base_tag(all_hits[cid], scores[cid], cid in dense_map, kind="fused")
            for cid in ordered]
    return FusionResult(hits=hits, method="weighted", n_dense=len(dense),
                        n_keyword=len(keyword), n_shared=shared,
                        n_total=len(hits))


def fuse(dense: Sequence[SearchHit], keyword: Sequence[SearchHit], *,
         method: str = "rrf", w_keyword: float = 0.5) -> FusionResult:
    """按名字选融合方式。未知名字 → 抛错（不静默回退，见 profiles 的同类说明）。"""
    if method == "rrf":
        return rrf_fuse(dense, keyword, w_keyword=w_keyword)
    if method == "weighted":
        return weighted_fuse(dense, keyword, w_keyword=w_keyword)
    raise ValueError(f"未知融合方式 {method!r}（可选：rrf / weighted）")
