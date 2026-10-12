# 工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流
"""格式分布统计 + 文档长度分布

这两个是「统计类」功能，都基于同一遍遍历的结果，所以放一起。

长度分布按工单要求输出 **P25/P50/P75/P90/P99 分位数 + 长度区间分布**。
分位数用**就近取整下标**而不是插值 —— 对「这批文档多长」这种问题，
拿一个真实存在的文档长度比拿插值出来的虚拟值更好解释。
"""


def format_stats(paths):
    """按扩展名统计数量与占比。"""
    from collections import Counter
    c = Counter(p.suffix.lower() or "(无扩展名)" for p in paths)
    total = sum(c.values()) or 1
    return {
        "total": sum(c.values()),
        "by_ext": dict(c.most_common()),
        "ratio": {k: round(v / total, 4) for k, v in c.most_common()},
    }


def _percentile(sorted_vals, p):
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * p / 100))
    return sorted_vals[idx]


def length_stats(records, cfg):
    """records: [{path, chars, error}, ...]（chars 为 None 表示抽取失败）"""
    c = cfg["length"]
    ok = [r for r in records if r.get("chars") is not None]
    failed = [r for r in records if r.get("chars") is None]
    vals = sorted(r["chars"] for r in ok)

    pct = {f"P{p}": _percentile(vals, p) for p in c["percentiles"]}

    # 区间分布
    bins = c["bins"]
    dist = []
    for i in range(len(bins)):
        lo = bins[i]
        hi = bins[i + 1] if i + 1 < len(bins) else None
        n = sum(1 for v in vals if v >= lo and (hi is None or v < hi))
        dist.append({"range": f"{lo}~{hi}" if hi else f">={lo}", "count": n,
                     "ratio": round(n / (len(vals) or 1), 4)})

    too_long = [r["path"] for r in ok if r["chars"] > c["too_long"]]
    too_short = [r["path"] for r in ok if r["chars"] < c["too_short"]]
    empty = [r["path"] for r in ok if r["chars"] == 0]
    return {
        "count": len(vals),
        "failed_count": len(failed),
        "total_chars": sum(vals),
        "mean": round(sum(vals) / len(vals), 1) if vals else 0,
        "percentiles": pct,
        "distribution": dist,
        "too_long": len(too_long),
        "too_short": len(too_short),
        "empty": len(empty),
        "_too_long": too_long,
        "_too_short": too_short,
        "_empty": empty,
        "_failed": [{"path": r["path"], "error": r.get("error", "")} for r in failed],
    }
