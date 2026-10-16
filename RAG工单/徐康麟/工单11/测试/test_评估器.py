# 工单编号：人工智能NLP-RAG 项目-Embedding 模型微调任务
# -*- coding: utf-8 -*-
"""测试三：评估器自检（指标定义与极端情形）。"""
from __future__ import annotations

import math

import numpy as np
import pytest

from evaluator import (bootstrap_ci, mrr, ndcg_at_k, ranks_from_gold_sets, ranks_from_scores,
                       recall_at_k, retrieval_metrics, similarity_stats, summarize_binary_diff)


# --------------------------------------------------------------- 排名解算
def test_排名解算_完全正确():
    docs = np.eye(3, dtype="float32")
    q = np.eye(3, dtype="float32")
    ranks = ranks_from_scores(q, docs, [0, 1, 2])
    assert ranks == [1.0, 1.0, 1.0]


def test_排名解算_逆序():
    docs = np.eye(3, dtype="float32")
    q = np.eye(3, dtype="float32")
    ranks = ranks_from_scores(q, docs, [2, 1, 0])
    assert ranks == [3.0, 1.0, 2.0]


def test_排名解算_多金标准取最优():
    docs = np.eye(4, dtype="float32")
    q = np.eye(4, dtype="float32")
    ranks, counts = ranks_from_gold_sets(q, docs, [[3, 1], [0, 2], [2, 3], [0, 1]])
    # q1 的相似度顺序为 [1,0,2,3]，其金标准 {0,2} 的最优排名为 2
    assert ranks == [2.0, 2.0, 1.0, 2.0]
    assert counts == [2, 2, 2, 2]


def test_排名解算_无金标准():
    docs = np.eye(2, dtype="float32")
    q = np.eye(2, dtype="float32")
    ranks, counts = ranks_from_gold_sets(q, docs, [[], [1]])
    assert ranks[0] == float("inf") and counts[0] == 0


# ------------------------------------------------------------------- 指标
def test_recall手算():
    ranks = [1, 2, 3, 11]
    assert recall_at_k(ranks, 1) == pytest.approx(0.25)
    assert recall_at_k(ranks, 5) == pytest.approx(0.75)
    assert recall_at_k(ranks, 10) == pytest.approx(0.75)
    assert recall_at_k(ranks, 20) == pytest.approx(1.0)


def test_mrr手算():
    ranks = [1, 2, 4]
    assert mrr(ranks) == pytest.approx((1 + 0.5 + 0.25) / 3, abs=1e-6)


def test_mrr未命中计零():
    assert mrr([float("inf")]) == 0.0


def test_ndcg手算():
    ranks = [1, 2, 3, 11]
    expect = (1.0 / math.log2(2) + 1.0 / math.log2(3) + 1.0 / math.log2(4) + 0.0) / 4
    assert ndcg_at_k(ranks, 10) == pytest.approx(expect, abs=1e-6)


def test_ndcg完全命中为1():
    assert ndcg_at_k([1, 1, 1], 10) == pytest.approx(1.0)


def test_完美检索指标全1():
    m = retrieval_metrics([1, 1, 1, 1], ks=(1, 5, 10), ndcg_k=10)
    assert m["Recall@1"] == 1.0 and m["Recall@5"] == 1.0 and m["MRR"] == 1.0 and m["nDCG@10"] == 1.0


def test_全部未命中指标全0():
    m = retrieval_metrics([float("inf")] * 4, ks=(1, 5, 10), ndcg_k=10)
    assert m["Recall@1"] == 0.0 and m["MRR"] == 0.0 and m["nDCG@10"] == 0.0
    assert m["命中数"] == 0


def test_指标单调性():
    """排名越小指标越高：Recall@1 <= Recall@5 <= Recall@10。"""
    ranks = [1, 3, 7, 12, 30]
    r1, r5, r10 = (recall_at_k(ranks, k) for k in (1, 5, 10))
    assert r1 <= r5 <= r10


# ------------------------------------------------------------- 相似度指标
def test_相似度间隔手算():
    s = similarity_stats([0.8, 0.9], [0.2, 0.3])
    assert s["正例对平均相似度"] == pytest.approx(0.85)
    assert s["负例对平均相似度"] == pytest.approx(0.25)
    assert s["相似度间隔"] == pytest.approx(0.6)
    assert s["正负例判别准确率"] == 1.0


def test_相似度判别准确率_完全不可分():
    s = similarity_stats([0.5, 0.5], [0.5, 0.5])
    assert s["相似度间隔"] == pytest.approx(0.0)
    assert s["正负例判别准确率"] == pytest.approx(0.0)  # 严格大于，全部并列 → 0


def test_相似度指标空输入不崩():
    s = similarity_stats([], [])
    assert s["正例对平均相似度"] is None and s["相似度间隔"] is None


# -------------------------------------------------------------- 统计工具
def test_自助法置信区间包含均值():
    vals = [1.0] * 50 + [0.0] * 50
    lo, hi = bootstrap_ci(vals)
    assert lo <= 0.5 <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_配对差分计数():
    d = summarize_binary_diff([1, 0, 1, 0], [1, 1, 0, 0])
    assert d == {"提升条数": 1, "下降条数": 1, "不变条数": 2, "可比条数": 4}


# ------------------------------------------------------ 真实检索结果自检
def test_评估结果文件自洽(仓库根):
    """若已完成评估，校验 JSON 内部自洽（Recall@1 <= Recall@5 <= Recall@10 <= 1）。"""
    import json
    p = 仓库根 / "研发" / "评估" / "评估结果_微调后.json"
    if not p.exists():
        pytest.skip("尚未生成评估结果，跳过")
    d = json.loads(p.read_text(encoding="utf-8"))
    for key in ("留出测试集", "真实问题"):
        m = d.get(key)
        if not m:
            continue
        r1, r5, r10 = m.get("Recall@1"), m.get("Recall@5"), m.get("Recall@10")
        vals = [v for v in (r1, r5, r10) if isinstance(v, (int, float))]
        assert vals == sorted(vals), f"{key} 的 Recall 不满足单调性：{vals}"
        assert all(0.0 <= v <= 1.0 for v in vals), f"{key} 的 Recall 越界：{vals}"
        assert 0.0 <= m.get("MRR", 0) <= 1.0
