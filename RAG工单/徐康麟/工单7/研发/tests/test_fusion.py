# -*- coding: utf-8 -*-
"""工单：人工智能NLP-RAG-混合检索任务（工单6）

工单编号：人工智能NLP-RAG-混合检索任务

``app/core/fusion.py`` 的**离线单元测试**（纯内存构造，不联网、不读索引、不调 Ollama）：

1. ``minmax_normalize`` 的四个边界（空 / 单元素 / 全同分 / 正常 min-max）；
2. 三种融合算法的排序合理性 —— 同时被两路命中的块必须排第一；
3. ``weighted_average`` 的权重口径（(1,0) 只看向量、(0,1) 只看全文、任意权重不出 NaN）；
4. ``rrf`` 的名次单调性与 ``rrf_k`` 的压平效应；
5. ``voting`` 的双路命中加成（等 Borda 票时加成生效）；
6. ``fuse()`` 对未知算法**抛 ValueError**（不静默回退）与权重清洗（负数不产生负分）。

运行：
    pwsh -NoProfile -File run_py.ps1 -m pytest 研发/tests/test_fusion.py -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "研发"))

from app.core import fusion as F  # noqa: E402

WORK_ORDER = "人工智能NLP-RAG-混合检索任务"

#: 向量路候选（分数：余弦相似度口径）
VECTOR_HITS: list[tuple[str, float]] = [("v1", 0.90), ("both", 0.80), ("v3", 0.50)]
#: 全文路候选（分数：BM25 口径，量纲与向量完全不同 —— 这正是必须先归一化的原因）
FULLTEXT_HITS: list[tuple[str, float]] = [("f1", 30.0), ("both", 20.0), ("f3", 5.0)]


# ---------------------------------------------------------------------------
# ① minmax_normalize
# ---------------------------------------------------------------------------
def test_minmax_normalize_empty_returns_empty_dict() -> None:
    assert F.minmax_normalize([]) == {}


def test_minmax_normalize_single_element_is_one() -> None:
    assert F.minmax_normalize([("only", 5.0)]) == {"only": 1.0}


def test_minmax_normalize_all_equal_scores_are_one() -> None:
    assert F.minmax_normalize([("a", 5.0), ("b", 5.0), ("c", 5.0)]) == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_minmax_normalize_normal_case_within_unit_range() -> None:
    got = F.minmax_normalize([("a", 3.0), ("b", 1.0), ("c", 2.0)])
    assert got == {"a": 1.0, "b": 0.0, "c": 0.5}
    assert max(got.values()) == 1.0
    assert min(got.values()) == 0.0
    assert all(0.0 <= value <= 1.0 for value in got.values())


def test_minmax_normalize_keys_are_stringified() -> None:
    """倒排/向量结果里的 id 可能是 int，归一化后必须统一成 str（下游按字符串比较）。"""
    got = F.minmax_normalize([(1, 1.0), (2, 3.0)])  # type: ignore[list-item]
    assert got == {"1": 0.0, "2": 1.0}


# ---------------------------------------------------------------------------
# ② 三种算法的排序合理性：两路都命中的块排第一
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("algorithm", list(F.ALGORITHMS))
def test_both_channel_hit_ranks_first_in_every_algorithm(algorithm: str) -> None:
    result = F.fuse(algorithm, VECTOR_HITS, FULLTEXT_HITS)
    ranked = result.ranked()
    assert ranked[0][0] == "both", f"{algorithm} 未把两路同时命中的块排在第一：{ranked}"
    assert ranked[0][1] > ranked[1][1], f"{algorithm} 第一名与第二名同分，排序无区分度"
    assert result.detail["both"]["in_vector"] is True
    assert result.detail["both"]["in_fulltext"] is True
    assert result.scores and all(not math.isnan(v) for v in result.scores.values())


# ---------------------------------------------------------------------------
# ③ weighted_average 权重口径
# ---------------------------------------------------------------------------
def test_weighted_average_vector_weight_only_uses_vector_channel() -> None:
    result = F.fuse("weighted_average", VECTOR_HITS, FULLTEXT_HITS,
                    vector_weight=1.0, fulltext_weight=0.0)
    order = [cid for cid, _ in result.ranked()]
    # 向量路内部的名次必须被完整保留：v1 > both > v3
    assert order.index("v1") < order.index("both") < order.index("v3")
    assert result.scores["v1"] == 1.0 and result.scores["both"] == 0.75
    # 只被全文路命中的块，权重为 0 ⇒ 得分必为 0
    assert result.scores["f1"] == 0.0
    assert result.scores["f3"] == 0.0
    assert (result.vector_weight, result.fulltext_weight) == (1.0, 0.0)


def test_weighted_average_fulltext_weight_only_uses_fulltext_channel() -> None:
    result = F.fuse("weighted_average", VECTOR_HITS, FULLTEXT_HITS,
                    vector_weight=0.0, fulltext_weight=1.0)
    order = [cid for cid, _ in result.ranked()]
    # 全文路内部的名次必须被完整保留：f1 > both > f3
    assert order.index("f1") < order.index("both") < order.index("f3")
    assert result.scores["f1"] == 1.0 and result.scores["both"] == 0.6
    assert result.scores["v1"] == 0.0
    assert result.scores["v3"] == 0.0
    assert (result.vector_weight, result.fulltext_weight) == (0.0, 1.0)


@pytest.mark.parametrize("vector_weight,fulltext_weight", [
    (0.3, 0.7), (2.5, 3.5), (0.0, 1.0), (1.0, 0.0), (100.0, 0.001), (0.5, 0.5),
], ids=["0.3/0.7", "2.5/3.5", "0/1", "1/0", "100/0.001", "0.5/0.5"])
def test_weighted_average_scores_never_nan_for_any_weights(vector_weight: float,
                                                           fulltext_weight: float) -> None:
    result = F.fuse("weighted_average", VECTOR_HITS, FULLTEXT_HITS,
                    vector_weight=vector_weight, fulltext_weight=fulltext_weight)
    assert result.scores, "两路都有候选时不应返回空结果"
    for cid, score in result.scores.items():
        assert not math.isnan(score), f"{cid} 的融合分为 NaN（权重 {vector_weight}/{fulltext_weight}）"
        assert not math.isinf(score), f"{cid} 的融合分为 inf"


@pytest.mark.parametrize("algorithm", list(F.ALGORITHMS))
def test_negative_weights_are_cleaned_and_never_produce_negative_scores(algorithm: str) -> None:
    """负权重必须被截断为 0（权重清洗），且清洗过程不得产生负分或 NaN。"""
    result = F.fuse(algorithm, VECTOR_HITS, FULLTEXT_HITS,
                    vector_weight=-2.0, fulltext_weight=0.5)
    assert result.vector_weight == 0.0
    assert result.fulltext_weight == 0.5
    for cid, score in result.scores.items():
        assert score >= 0.0, f"{algorithm}: {cid} 出现负分 {score}"
        assert not math.isnan(score)


def test_weighted_average_negative_weight_cleaning_is_recorded_in_notes() -> None:
    """（评审修正后的口径）**三个算法**都必须把权重清洗写进 notes。

    修正前的实现只有 ``weighted_average`` 留痕，``rrf`` / ``voting`` 清洗后不留痕 ——
    与 工单4「回退动作必须写日志、绝不静默」的纪律相悖。现已统一由 ``_weight_pair``
    返回说明并由三个算法各自写入 ``notes``。
    """
    for algorithm in F.ALGORITHMS:
        result = F.fuse(algorithm, VECTOR_HITS, FULLTEXT_HITS,
                        vector_weight=-2.0, fulltext_weight=0.5)
        assert any("负权重已截断" in note for note in result.notes), (algorithm, result.notes)


def test_both_weights_zero_fall_back_to_one_one_and_all_algorithms_record_it() -> None:
    """两路权重同时为 0 无意义 ⇒ 三个算法都回退 (1.0, 1.0) 并写明，绝不返回空结果。"""
    for algorithm in F.ALGORITHMS:
        result = F.fuse(algorithm, VECTOR_HITS, FULLTEXT_HITS,
                        vector_weight=0.0, fulltext_weight=0.0)
        assert (result.vector_weight, result.fulltext_weight) == (1.0, 1.0), algorithm
        assert any("同时为 0" in note for note in result.notes), (algorithm, result.notes)


def test_both_weights_zero_fall_back_to_one_one() -> None:
    """两路权重同时为 0 无意义 ⇒ 回退 (1.0, 1.0) 并写明，绝不返回空结果。"""
    result = F.fuse("weighted_average", VECTOR_HITS, FULLTEXT_HITS,
                    vector_weight=0.0, fulltext_weight=0.0)
    assert (result.vector_weight, result.fulltext_weight) == (1.0, 1.0)
    assert result.scores, "权重回退后仍应有候选"
    assert all(score >= 0.0 for score in result.scores.values())


# ---------------------------------------------------------------------------
# ④ rrf
# ---------------------------------------------------------------------------
def test_rrf_higher_rank_gets_higher_score() -> None:
    result = F.rrf_fusion(VECTOR_HITS, FULLTEXT_HITS, vector_weight=1.0, fulltext_weight=0.0)
    assert result.scores["v1"] > result.scores["both"] > result.scores["v3"]
    assert "rrf_k=60" in " ".join(result.notes)


def test_rrf_larger_k_flattens_first_last_gap() -> None:
    """rrf_k 越大，首尾分差越小（越强调「两路都出现」而非「某路第一名」）。"""

    def spread(k: int) -> float:
        ranked = F.fuse("rrf", VECTOR_HITS, FULLTEXT_HITS, rrf_k=k).ranked()
        return ranked[0][1] - ranked[-1][1]

    gaps = [spread(1), spread(10), spread(60)]
    assert gaps[0] > gaps[1] > gaps[2], f"rrf_k 未压平名次差异：{gaps}"
    assert gaps[2] > 0.0, "首尾仍应有可区分的分差"


def test_rrf_k_is_clamped_to_at_least_one() -> None:
    result = F.rrf_fusion(VECTOR_HITS, FULLTEXT_HITS, rrf_k=0)
    assert result.scores == F.rrf_fusion(VECTOR_HITS, FULLTEXT_HITS, rrf_k=1).scores
    assert "rrf_k=1" in " ".join(result.notes)


def test_rrf_ignores_raw_score_magnitude() -> None:
    """RRF 只用名次：把某路的原始分整体放大 1000 倍，结果必须逐位一致。"""
    scaled = [(cid, score * 1000.0) for cid, score in FULLTEXT_HITS]
    assert F.rrf_fusion(VECTOR_HITS, FULLTEXT_HITS).scores == F.rrf_fusion(VECTOR_HITS, scaled).scores


def test_rrf_empty_both_channels_returns_no_candidates() -> None:
    result = F.rrf_fusion([], [])
    assert result.scores == {}
    assert "两路均无候选" in " ".join(result.notes)


# ---------------------------------------------------------------------------
# ⑤ voting：双路命中加成
# ---------------------------------------------------------------------------
def test_voting_both_channel_bonus_beats_equal_borda_single_channel() -> None:
    """等票对比：``both``（两路各出 2 票）与 ``split``（向量路出 4 票）加成前票数相同，
    只有 ``both`` 拿到 1.5 倍加成 —— 这就是投票机制相对加权平均的核心优势。"""
    vector_hits = [("split", 0.95), ("v2", 0.80), ("both", 0.70), ("v4", 0.10)]
    fulltext_hits = [("f1", 9.0), ("f2", 8.0), ("both", 7.0), ("f4", 1.0)]
    result = F.voting_fusion(vector_hits, fulltext_hits, vector_weight=1.0, fulltext_weight=1.0)

    split, both = result.detail["split"], result.detail["both"]
    assert split["votes"] == both["votes"] == 4.0, "构造前提（加成前票数相同）不成立"
    assert split["both_channels"] is False and split["bonus"] == 1.0
    assert both["both_channels"] is True and both["bonus"] == F.VOTING_BOTH_CHANNEL_BONUS
    assert result.scores["both"] > result.scores["split"]
    assert both["borda_vector"] == 2.0 and both["borda_fulltext"] == 2.0


def test_voting_single_channel_only_never_gets_bonus() -> None:
    result = F.voting_fusion(VECTOR_HITS, [], vector_weight=0.5, fulltext_weight=0.5)
    assert result.scores
    for cid in ("v1", "both", "v3"):
        assert result.detail[cid]["both_channels"] is False
        assert result.detail[cid]["bonus"] == 1.0
        assert result.detail[cid]["in_fulltext"] is False


def test_voting_empty_both_channels_returns_no_candidates() -> None:
    result = F.voting_fusion([], [])
    assert result.scores == {}
    assert "两路均无候选" in " ".join(result.notes)


# ---------------------------------------------------------------------------
# ⑥ fuse() 统一入口
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["bogus", "", "   ", "rrf2", None, "weighted-average"],
                         ids=["bogus", "empty", "blank", "rrf2", "None", "连字符"])
def test_fuse_unknown_algorithm_raises_value_error(bad: object) -> None:
    """未知算法**不得**静默回退到某个默认算法。"""
    with pytest.raises(ValueError):
        F.fuse(bad, VECTOR_HITS, FULLTEXT_HITS)  # type: ignore[arg-type]


def test_fuse_dispatches_case_insensitively_and_matches_direct_call() -> None:
    upper = F.fuse("RRF", VECTOR_HITS, FULLTEXT_HITS)
    direct = F.rrf_fusion(VECTOR_HITS, FULLTEXT_HITS)
    assert upper.algorithm == F.ALGORITHM_RRF
    assert upper.scores == direct.scores


@pytest.mark.parametrize("algorithm", list(F.ALGORITHMS))
def test_fuse_passes_weights_and_rrf_k_through(algorithm: str) -> None:
    result = F.fuse(algorithm, VECTOR_HITS, FULLTEXT_HITS,
                    vector_weight=0.25, fulltext_weight=0.75, rrf_k=7)
    assert (result.vector_weight, result.fulltext_weight) == (0.25, 0.75)
    assert result.algorithm == algorithm
    if algorithm == F.ALGORITHM_RRF:
        assert "rrf_k=7" in " ".join(result.notes)


# ---------------------------------------------------------------------------
# ⑦ FusionResult 契约与算法清单
# ---------------------------------------------------------------------------
def test_fusion_result_ranked_is_descending_and_tie_breaks_by_id() -> None:
    result = F.FusionResult(algorithm="rrf", scores={"b": 0.5, "a": 0.5, "c": 1.0})
    assert result.ranked() == [("c", 1.0), ("a", 0.5), ("b", 0.5)]


def test_fusion_result_to_dict_contract() -> None:
    result = F.fuse("voting", VECTOR_HITS, FULLTEXT_HITS)
    payload = result.to_dict()
    assert payload["algorithm"] == "voting"
    assert payload["algorithm_label"] == F.ALGORITHM_LABELS["voting"]
    assert payload["candidates"] == len(result.scores) == 5
    assert payload["notes"] == result.notes
    assert "algorithm_label" in payload and "vector_weight" in payload


def test_describe_algorithms_covers_every_algorithm() -> None:
    described = F.describe_algorithms()
    assert [item["id"] for item in described] == list(F.ALGORITHMS)
    assert all(item["label"] and item["formula"] for item in described)
