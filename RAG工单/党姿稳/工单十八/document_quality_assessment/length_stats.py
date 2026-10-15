# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-18-实现文档质量评估Skill并集成至智能体工作流
# 模块职责：文档长度分布统计——输出 P25/P50/P75/P90/P99 分位数与长度区间分布，
#           并按格式分别给出统计，便于判断哪类文档是“短文档”的主要来源。

import numpy as np                                                 # 分位数与分箱


def percentiles(values, points):
    """按给定分位点计算分位数；空输入返回全 0。"""
    if not values:
        return dict((("P%d" % p), 0) for p in points)              # 空集兜底
    arr = np.asarray(values, dtype=np.float64)                     # 转数组
    return dict((("P%d" % p), float(np.percentile(arr, p))) for p in points)


def histogram(values, bins, labels):
    """按区间边界统计落入各区间的文档数与占比。"""
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)                     # 转数组
    rows = []                                                      # 结果行
    total = len(arr)                                               # 文档总数
    for i in range(len(bins) - 1):                                 # 逐区间
        lo, hi = bins[i], bins[i + 1]                              # 区间上下界
        cnt = int(((arr >= lo) & (arr < hi)).sum())                # 左闭右开计数
        rows.append({"range": labels[i] if i < len(labels) else "%s-%s" % (lo, hi),
                     "lower": lo, "upper": hi, "count": cnt,       # 区间与数量
                     "ratio": round(cnt / float(total), 4)})
    return rows


def describe(values):
    """给出均值/中位/极值等概要。"""
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0, "max": 0}
    arr = np.asarray(values, dtype=np.float64)                     # 转数组
    return {"count": int(len(arr)),                                # 样本数
            "mean": round(float(arr.mean()), 2),                   # 均值
            "median": round(float(np.median(arr)), 2),             # 中位数
            "min": int(arr.min()), "max": int(arr.max()),          # 极值
            "std": round(float(arr.std()), 2)}                     # 标准差


def compute(records, cfg):
    """主入口：对全部记录算总体与分格式的长度分布。"""
    points = cfg.get("length.percentiles", [25, 50, 75, 90, 99])   # 分位点
    bins = cfg.get("length.bins", [])                              # 区间边界
    labels = cfg.get("length.bin_labels", [])                      # 区间标签

    all_chars = [r.get("text_chars", 0) for r in records]          # 全部文档字符数
    overall = {"summary": describe(all_chars),                     # 概要
               "percentiles": percentiles(all_chars, points),       # 分位数
               "histogram": histogram(all_chars, bins, labels)}    # 区间分布

    by_ext = {}                                                    # 分格式统计
    exts = sorted(set(r.get("ext", "") for r in records))          # 所有后缀
    for ext in exts:                                               # 逐格式
        vals = [r.get("text_chars", 0) for r in records if r.get("ext") == ext]
        by_ext[ext or "(无后缀)"] = {
            "summary": describe(vals),                             # 概要
            "percentiles": percentiles(vals, points),              # 分位数
            "histogram": histogram(vals, bins, labels)}            # 区间分布
    return {"overall": overall, "by_format": by_ext}


def outliers(records, cfg, low=True, limit=30):
    """给出过短/过长文档清单，供人工快速定位异常文件。"""
    th = cfg.get("classify.too_short_chars", 200)                  # 过短阈值
    items = [r for r in records                                     # 过短文档
             if r.get("text_chars", 0) < th and not r.get("error")]
    items.sort(key=lambda r: r.get("text_chars", 0))               # 由短到长
    return [{"path": r["path"], "ext": r["ext"],       # 路径 / 后缀
             "text_chars": r.get("text_chars", 0),     # 字符数
             "pdf_kind": r.get("pdf_kind")}            # PDF 类型（非 PDF 为 None）
            for r in items[:limit]]                    # 截断
