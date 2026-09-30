# -*- coding: utf-8 -*-
"""Jmeter JTL（CSV 格式）结果解析：总样本、错误率、延迟分布、QPS。"""
import csv
# 解析：CSV 解析
import io
# 解析：字符串流（DictReader 需要文件对象）
import statistics
# 解析：统计（均值/中位数）


def parse_jtl(content: str) -> dict:
    """content: jtl 文件内容（CSV 头 + 行）。返回统计字典。"""
    reader = csv.DictReader(io.StringIO(content))
    # 解析：按 CSV 头解析每行
    rows = list(reader)
    # 解析：全部样本行
    if not rows:
        # 解析：无数据
        return {"total": 0, "errors": 0, "error_rate": 0.0, "qps": 0.0,
                "avg_ms": 0.0, "median_ms": 0, "p95_ms": 0, "min_ms": 0, "max_ms": 0}
        # 解析：全零返回

    elapsed = sorted(float(r["elapsed"]) for r in rows)
    # 解析：所有样本耗时排序（算分位数用）
    total = len(rows)
    # 解析：样本总数
    errors = sum(1 for r in rows if r.get("success") == "false")
    # 解析：失败样本数

    timestamps = sorted(float(r["timeStamp"]) for r in rows)
    # 解析：时间戳排序
    span_ms = max(timestamps[-1] - timestamps[0], 1.0)
    # 解析：时间跨度（毫秒，保底 1 防除零）

    return {
        # 解析：返回统计字典
        "total": total,
        # 解析：总请求数
        "errors": errors,
        # 解析：失败数
        "error_rate": errors / total,
        # 解析：错误率
        "qps": total / (span_ms / 1000.0),
        # 解析：QPS = 请求数 / 时间跨度（秒）
        "avg_ms": statistics.mean(elapsed),
        # 解析：平均延迟
        "median_ms": statistics.median(elapsed),
        # 解析：中位延迟
        "p95_ms": _percentile(elapsed, 0.95),
        # 解析：P95 延迟
        "min_ms": elapsed[0],
        # 解析：最小延迟
        "max_ms": elapsed[-1],
        # 解析：最大延迟
    }


def _percentile(sorted_values: list[float], p: float) -> float:
    # 解析：计算分位数
    if not sorted_values:
        # 解析：空列表
        return 0.0
        # 解析：返回 0
    index = min(int(p * len(sorted_values)), len(sorted_values) - 1)
    # 解析：分位索引（防越界）
    return sorted_values[index]
    # 解析：返回该位置的值
