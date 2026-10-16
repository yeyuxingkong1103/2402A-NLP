# -*- coding: utf-8 -*-
"""工单6 混合检索 —— 多通道结果融合算法（加权平均 / RRF / 投票）。

工单编号：人工智能NLP-RAG-混合检索任务

本模块把「向量通道」与「全文通道」的候选结果合并成一份有序候选表，提供三种融合算法：

======================  ==========================================================
算法                     口径与适用场景
======================  ==========================================================
``weighted_average``      **加权平均**（工单最低需求必须实现）：两路分数各自 min-max
                          归一化到 ``[0,1]`` 后，按 ``vector_weight`` / ``fulltext_weight``
                          线性加权求和。直观、权重可解释；缺点是受单路分数分布影响
                          （长尾被压缩）。
``rrf``                   **倒数排名融合**（Reciprocal Rank Fusion）：``Σ w / (k + rank)``。
                          只用名次、不用原始分数 ⇒ **天然免疫两路分数的量纲差异**。
``voting``                **投票机制**：两路各自 Top-N 提名，按「是否被两路同时提名」
                          与名次计 Borda 票；两路都命中的块获得 bonus，用于压制单路噪声。
======================  ==========================================================

设计纪律（沿用 工单4 ``研发/README.md`` §2「融合纪律」）：
    向量余弦分（bge-m3，通常 0.5~0.85）与 BM25 分（无上界，实测 0~60+）属**不同语义空间**，
    **禁止把原始相似度直接相加** —— 必须相加时先各自归一化（``weighted_average`` 即如此），
    否则 BM25 的大数值会把向量分完全淹没。

对外契约：
    ``fuse(algorithm, vector_hits, fulltext_hits, *, vector_weight, fulltext_weight,
    rrf_k) -> FusionResult``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

WORK_ORDER = "人工智能NLP-RAG-混合检索任务"

#: 融合算法标识（与 ``retrieval_strategy.FusionAlgorithm`` 同字面量，避免循环 import 故用裸字符串）
ALGORITHM_WEIGHTED_AVERAGE = "weighted_average"
ALGORITHM_RRF = "rrf"
ALGORITHM_VOTING = "voting"

ALGORITHMS: tuple[str, ...] = (ALGORITHM_WEIGHTED_AVERAGE, ALGORITHM_RRF, ALGORITHM_VOTING)

#: 单通道模式的算法标识（向量-only / 全文-only 没有「融合」，但共用 ``FusionResult`` 载体）
SINGLE_CHANNEL = "single_channel"

#: 算法中文名（用于界面下拉框与报告）
ALGORITHM_LABELS: dict[str, str] = {
    ALGORITHM_WEIGHTED_AVERAGE: "加权平均（归一化后线性加权）",
    ALGORITHM_RRF: "RRF 倒数排名融合",
    ALGORITHM_VOTING: "投票机制（Borda 计票 + 双路命中加成）",
    # 单通道模式（vector-only / fulltext-only）也用 FusionResult 承载，
    # 必须给中文标签，否则界面与报告的 ``algorithm_label`` 会退化成裸 id。
    SINGLE_CHANNEL: "单通道（未融合）",
}

#: RRF 平滑常数默认值（与 工单4 ``RetrievalSettings.rrf_k`` 一致）
DEFAULT_RRF_K = 60

#: 投票算法中「两路同时命中」的加成系数（Borda 票的倍数）
VOTING_BOTH_CHANNEL_BONUS = 1.5


@dataclass(slots=True)
class FusionResult:
    """一次融合的结果。

    ``scores``：``chunk_id -> 融合分``（降序排列由调用方决定）；
    ``detail``：``chunk_id -> {算法相关字段}``，用于界面/报告展示「这个块为什么排在前面」。
    """

    algorithm: str
    scores: dict[str, float] = field(default_factory=dict)
    detail: dict[str, dict[str, Any]] = field(default_factory=dict)
    vector_weight: float = 1.0
    fulltext_weight: float = 1.0
    notes: list[str] = field(default_factory=list)

    def ranked(self) -> list[tuple[str, float]]:
        """``[(chunk_id, score)]`` 降序（同分按 chunk_id 稳定排序）。"""
        return sorted(self.scores.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "algorithm_label": ALGORITHM_LABELS.get(self.algorithm, self.algorithm),
            "vector_weight": self.vector_weight,
            "fulltext_weight": self.fulltext_weight,
            "candidates": len(self.scores),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# 归一化工具
# ---------------------------------------------------------------------------
def minmax_normalize(hits: Sequence[tuple[str, float]]) -> dict[str, float]:
    """把一路检索结果**按分数** min-max 归一化到 ``[0,1]``。

    边界纪律：
        * 空列表 → ``{}``；
        * 只有一个候选，或所有候选同分 → 全部给 ``1.0``
          （不是 0.5：该路只有唯一证据时不应被主观腰斩，且与「相对最优」语义一致）；
        * 归一化以**该路返回的 Top-K 内部**为基准，不是全局基准 —— 这是加权平均的已知口径，
          已写入 ``notes`` 供报告如实说明。
    """
    if not hits:
        return {}
    values = [float(score) for _cid, score in hits]
    lo, hi = min(values), max(values)
    span = hi - lo
    if span <= 1e-12:
        return {str(cid): 1.0 for cid, _score in hits}
    return {str(cid): (float(score) - lo) / span for cid, score in hits}


def _weight_pair(vector_weight: float, fulltext_weight: float) -> tuple[float, float, str | None]:
    """权重清洗：负值截断为 0；两者同时为 0 时回退到 ``(1.0, 1.0)``。

    返回 ``(生效向量权重, 生效全文权重, 说明或 None)``。

    **说明必须回传并写进 notes**（评审修正）：早期版本只有 ``weighted_average`` 记录清洗动作，
    ``rrf`` / ``voting`` 清洗后不留痕 —— 与 工单4「回退动作必须写日志、绝不静默」的纪律相悖。
    现在三个算法共用本函数，清洗一定留痕。

    ``(0, 0)`` 的兜底与 ``StrategyConfig.validate()`` 的「权重不得同时为 0」并不矛盾：
    ``validate()`` 是**面向用户输入的第一道闸**（直接拒配），本函数是**面向内部调用的兜底**
    （保证即便绕过校验也不会产出 NaN/空结果）。两层都保留，且都留痕。
    """
    raw_v, raw_f = float(vector_weight), float(fulltext_weight)
    vw, fw = max(0.0, raw_v), max(0.0, raw_f)
    notes: list[str] = []
    if vw != raw_v or fw != raw_f:
        notes.append(f"负权重已截断为 0：传入 ({raw_v}, {raw_f}) → ({vw}, {fw})")
    if vw <= 0.0 and fw <= 0.0:
        vw = fw = 1.0
        notes.append("两路权重同时为 0，已兜底为 (1.0, 1.0) 以保证结果非空")
    return vw, fw, ("；".join(notes) if notes else None)


def _rank_map(hits: Sequence[tuple[str, float]]) -> dict[str, int]:
    """``chunk_id -> 1-based 名次``。"""
    return {str(cid): rank for rank, (cid, _s) in enumerate(hits, 1)}


def _score_map(hits: Sequence[tuple[str, float]]) -> dict[str, float]:
    """``chunk_id -> 原始分``（同 id 重复出现时保留首次即最高分）。"""
    out: dict[str, float] = {}
    for cid, score in hits:
        out.setdefault(str(cid), float(score))
    return out


# ---------------------------------------------------------------------------
# 算法 ①：加权平均
# ---------------------------------------------------------------------------
def weighted_average_fusion(vector_hits: Sequence[tuple[str, float]],
                            fulltext_hits: Sequence[tuple[str, float]], *,
                            vector_weight: float = 0.5, fulltext_weight: float = 0.5,
                            rrf_k: int = DEFAULT_RRF_K) -> FusionResult:
    """加权平均融合：两路分数各自 min-max 归一化后线性加权求和。

    公式：``score(c) = w_v · norm_v(c) + w_f · norm_f(c)``；
    仅被一路召回的块，另一路按 ``0.0`` 计入（即缺项补零，**不是**忽略该项后重新归一化权重）。

    权重口径：工单要求「向量权重 0.5、全文权重 0.5」这类可调配置 ——
    本实现直接使用传入的两个权重（默认 0.5/0.5），并在 ``detail`` 中回写生效权重。
    """
    vw, fw, weight_note = _weight_pair(vector_weight, fulltext_weight)
    v_norm = minmax_normalize(vector_hits)
    f_norm = minmax_normalize(fulltext_hits)
    result = FusionResult(algorithm=ALGORITHM_WEIGHTED_AVERAGE,
                          vector_weight=vw, fulltext_weight=fw)
    if weight_note:
        result.notes.append(weight_note)
    if not v_norm and not f_norm:
        result.notes.append("两路均无候选")
        return result

    v_rank, f_rank = _rank_map(vector_hits), _rank_map(fulltext_hits)
    v_raw, f_raw = _score_map(vector_hits), _score_map(fulltext_hits)
    for cid in set(v_norm) | set(f_norm):
        nv = float(v_norm.get(cid, 0.0))
        nf = float(f_norm.get(cid, 0.0))
        result.scores[cid] = round(vw * nv + fw * nf, 8)
        result.detail[cid] = {
            "vector_norm": round(nv, 6), "fulltext_norm": round(nf, 6),
            "vector_score": v_raw.get(cid), "fulltext_score": f_raw.get(cid),
            "vector_rank": v_rank.get(cid), "fulltext_rank": f_rank.get(cid),
            "in_vector": cid in v_norm, "in_fulltext": cid in f_norm,
        }
    result.notes.append("两路分数各自在**本路 Top-K 内部**做 min-max 归一化，缺项补 0")
    return result


# ---------------------------------------------------------------------------
# 算法 ②：RRF 倒数排名融合
# ---------------------------------------------------------------------------
def rrf_fusion(vector_hits: Sequence[tuple[str, float]],
               fulltext_hits: Sequence[tuple[str, float]], *,
               vector_weight: float = 0.5, fulltext_weight: float = 0.5,
               rrf_k: int = DEFAULT_RRF_K) -> FusionResult:
    """RRF 融合：``score(c) = Σ_channel w_channel / (rrf_k + rank_channel(c))``。

    只用名次不用原始分 ⇒ 免疫量纲差异，是 工单4 既有 ``hybrid_search`` 的默认口径。
    ``rrf_k`` 越大，名次差异被压得越平（越强调「两路都出现」而非「某路第一名」）。
    """
    vw, fw, weight_note = _weight_pair(vector_weight, fulltext_weight)
    k = max(1, int(rrf_k))
    result = FusionResult(algorithm=ALGORITHM_RRF, vector_weight=vw, fulltext_weight=fw)
    if weight_note:
        result.notes.append(weight_note)
    if not vector_hits and not fulltext_hits:
        result.notes.append("两路均无候选")
        return result

    v_rank, f_rank = _rank_map(vector_hits), _rank_map(fulltext_hits)
    v_raw, f_raw = _score_map(vector_hits), _score_map(fulltext_hits)
    contributions: dict[str, float] = {}
    for cid, rank in v_rank.items():
        contributions[cid] = contributions.get(cid, 0.0) + vw / (k + rank)
    for cid, rank in f_rank.items():
        contributions[cid] = contributions.get(cid, 0.0) + fw / (k + rank)
    for cid, value in contributions.items():
        result.scores[cid] = round(value, 10)
        result.detail[cid] = {
            "vector_rank": v_rank.get(cid), "fulltext_rank": f_rank.get(cid),
            "vector_score": v_raw.get(cid), "fulltext_score": f_raw.get(cid),
            "rrf_vector": round(vw / (k + v_rank[cid]), 10) if cid in v_rank else 0.0,
            "rrf_fulltext": round(fw / (k + f_rank[cid]), 10) if cid in f_rank else 0.0,
            "in_vector": cid in v_rank, "in_fulltext": cid in f_rank,
        }
    result.notes.append(f"rrf_k={k}；不使用原始分，仅用名次")
    return result


# ---------------------------------------------------------------------------
# 算法 ③：投票机制
# ---------------------------------------------------------------------------
def voting_fusion(vector_hits: Sequence[tuple[str, float]],
                  fulltext_hits: Sequence[tuple[str, float]], *,
                  vector_weight: float = 0.5, fulltext_weight: float = 0.5,
                  rrf_k: int = DEFAULT_RRF_K,
                  both_channel_bonus: float = VOTING_BOTH_CHANNEL_BONUS) -> FusionResult:
    """投票融合（Borda 计票 + 双路命中加成）。

    计票口径（把每一路看成一位「投票人」）：
        1. 每路的 ``N`` 个候选按名次给 Borda 分 ``N - rank + 1``（第一名得 N 分，末名得 1 分）；
        2. 该路的总 Borda 分按 ``w_channel`` 缩放后累加 —— 权重因此直接体现为「投票人话语权」；
        3. **两路同时提名**的块额外乘 ``both_channel_bonus``（默认 1.5）——
           这是投票机制相对加权平均的核心优势：一致同意的证据被显式奖励，
           单路高频噪声（如 BM25 只命中一次的生僻词）拿不到这个加成。

    归一化：最终分数 = 累加票 / ``(w_v·N_v + w_f·N_f)``，落在 ``[0, both_channel_bonus]``。
    """
    vw, fw, weight_note = _weight_pair(vector_weight, fulltext_weight)
    result = FusionResult(algorithm=ALGORITHM_VOTING, vector_weight=vw, fulltext_weight=fw)
    if weight_note:
        result.notes.append(weight_note)
    nv, nf = len(vector_hits), len(fulltext_hits)
    if nv == 0 and nf == 0:
        result.notes.append("两路均无候选")
        return result

    v_rank, f_rank = _rank_map(vector_hits), _rank_map(fulltext_hits)
    v_raw, f_raw = _score_map(vector_hits), _score_map(fulltext_hits)
    borda_v: dict[str, float] = {cid: float(nv - rank + 1) for cid, rank in v_rank.items()}
    borda_f: dict[str, float] = {cid: float(nf - rank + 1) for cid, rank in f_rank.items()}
    total_votes = vw * nv + fw * nf
    denom = total_votes if total_votes > 0 else 1.0
    for cid in set(borda_v) | set(borda_f):
        raw_votes = vw * borda_v.get(cid, 0.0) + fw * borda_f.get(cid, 0.0)
        both = cid in borda_v and cid in borda_f
        bonus = float(both_channel_bonus) if both else 1.0
        result.scores[cid] = round(raw_votes * bonus / denom, 8)
        result.detail[cid] = {
            "borda_vector": borda_v.get(cid, 0.0), "borda_fulltext": borda_f.get(cid, 0.0),
            "votes": round(raw_votes, 6), "both_channels": both, "bonus": bonus,
            "vector_rank": v_rank.get(cid), "fulltext_rank": f_rank.get(cid),
            "vector_score": v_raw.get(cid), "fulltext_score": f_raw.get(cid),
            "in_vector": cid in borda_v, "in_fulltext": cid in borda_f,
        }
    result.notes.append(f"Borda 计票 + 双路命中加成 ×{both_channel_bonus}")
    return result


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
def fuse(algorithm: str, vector_hits: Sequence[tuple[str, float]],
         fulltext_hits: Sequence[tuple[str, float]], *,
         vector_weight: float = 0.5, fulltext_weight: float = 0.5,
         rrf_k: int = DEFAULT_RRF_K) -> FusionResult:
    """按 ``algorithm`` 分派到具体融合算法。

    未知算法**不静默回退**：抛 ``ValueError`` 并由调用方（``retrieval_strategy``）
    转成用户可读错误 —— 与 工单4「回退动作必须写日志、绝不静默」的纪律一致。
    """
    name = str(algorithm or "").strip().lower()
    if name == ALGORITHM_WEIGHTED_AVERAGE:
        return weighted_average_fusion(vector_hits, fulltext_hits,
                                       vector_weight=vector_weight,
                                       fulltext_weight=fulltext_weight, rrf_k=rrf_k)
    if name == ALGORITHM_RRF:
        return rrf_fusion(vector_hits, fulltext_hits, vector_weight=vector_weight,
                          fulltext_weight=fulltext_weight, rrf_k=rrf_k)
    if name == ALGORITHM_VOTING:
        return voting_fusion(vector_hits, fulltext_hits, vector_weight=vector_weight,
                             fulltext_weight=fulltext_weight, rrf_k=rrf_k)
    raise ValueError(f"未知融合算法：{algorithm!r}（可选：{', '.join(ALGORITHMS)}）")


def describe_algorithms() -> list[dict[str, Any]]:
    """给界面/文档用的算法清单。"""
    return [
        {"id": ALGORITHM_WEIGHTED_AVERAGE, "label": ALGORITHM_LABELS[ALGORITHM_WEIGHTED_AVERAGE],
         "formula": "w_v·minmax(v) + w_f·minmax(f)", "needs_weights": True},
        {"id": ALGORITHM_RRF, "label": ALGORITHM_LABELS[ALGORITHM_RRF],
         "formula": "Σ w/(k+rank)", "needs_weights": True},
        {"id": ALGORITHM_VOTING, "label": ALGORITHM_LABELS[ALGORITHM_VOTING],
         "formula": "(w_v·Borda_v + w_f·Borda_f)·bonus / Σ", "needs_weights": True},
    ]


__all__ = [
    "WORK_ORDER", "ALGORITHMS", "ALGORITHM_LABELS", "DEFAULT_RRF_K", "VOTING_BOTH_CHANNEL_BONUS",
    "FusionResult", "minmax_normalize", "weighted_average_fusion", "rrf_fusion",
    "voting_fusion", "fuse", "describe_algorithms",
]
