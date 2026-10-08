# -*- coding: utf-8 -*-
"""功能3：文档长度分布（分位数 + 区间分布）。"""
from __future__ import annotations

import math
from typing import List


def percentile(sorted_values: List[int], p: float) -> int:
    """线性插值分位数。"""
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * p / 100
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_values[int(k)]
    return int(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo))


def length_distribution(char_counts: List[int], config: dict) -> dict:
    """分位数 + 区间分布。char_counts 与文件顺序对齐。"""
    values = sorted(c for c in char_counts if c > 0)
    pcts = {f"P{p}": percentile(values, p) for p in config["percentiles"]}

    # 区间分布（bins 为边界，bin_labels 比 bins 多 1）
    bins = config["bins"]
    labels = config["bin_labels"]
    bucket_counts = [0] * len(labels)
    for c in char_counts:
        idx = 0
        while idx < len(bins) and c > bins[idx]:
            idx += 1
        bucket_counts[min(idx, len(labels) - 1)] += 1

    total = len(char_counts)
    bins_dist = [
        {"range": labels[i], "count": bucket_counts[i],
         "ratio": round(bucket_counts[i] / total, 4) if total else 0}
        for i in range(len(labels))
    ]
    empty_count = sum(1 for c in char_counts if c <= config["empty_threshold"])
    return {
        "total_docs": total,
        "empty_docs": empty_count,
        "mean_chars": int(sum(char_counts) / total) if total else 0,
        "percentiles": pcts,
        "bins_distribution": bins_dist,
    }
