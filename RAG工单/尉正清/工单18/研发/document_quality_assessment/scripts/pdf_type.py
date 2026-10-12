# 工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流
"""PDF 文字型 / 扫描型 / 混合型 分流

## 判定逻辑（工单原文）

「按页判断 —— 每页字符数低于阈值就算扫描页，扫描页占比超过 70% 就标记整个文件
为扫描型。**这是一条启发式，用于分流与估算，不保证严格正确**；
因此所有阈值可配置，且输出『待确认』列表。」

最后那句是关键：既然承认不保证正确，就不能只给个结论，
得把**判定分数落在中间地带**的那些文件单独列出来给人看。

## 只看字符数不够：附图页会冒充扫描页（实测）

按工单原话「字符数低于阈值就算扫描页」实现后，拿 IMDR 1700 份实测，发现
**298 份被判成「混合型」**。逐份打开看低字符页是什么，答案是**附图页**：

    CN106926565B.pdf
      第 1~5 页  字符 925~1477   图片覆盖 0.01    ← 正文
      第 7~10 页 字符 32         图片覆盖 0.32~0.44  ← 附图页（整页只有「图1」「图2」）

附图页字符数当然少，但它是**数字排版的原生页**，不需要 OCR。
拿它当扫描页，等于凭空多出 298 份「要部分 OCR」的文档，白排队。

物理上两者差别很清楚，所以每页多加一个判据 —— **图片覆盖率**
（图片面积 ÷ 页面面积，在 `extract.pdf_parse` 里算好存进缓存）：

    扫描页：整张位图铺满  覆盖率 ≈ 1.00   ← 实测 622 份，几乎全是 1.00
    附图页：图占中间一块  覆盖率 ≈ 0.10   ← 实测 Text_PDF 中位 0.10

于是「扫描页」= **字符数低 且 图片铺满**。两个条件缺一不可：
只看字符数会把附图页算进来，只看图片覆盖率会把带整页底图的文字页算出去。

## 三种类型的边界

    扫描页占比 ≥ scanned_ratio(0.7)        → scanned   （必须走 OCR）
    扫描页占比 ≤ text_ratio(0.3)           → text      （可以直接抽文字）
    两者之间                                → mixed     （部分页要 OCR）
    再叠加「总页数字符数极少」→ 也归 scanned（整篇都是图）
"""
import json


def _is_scan_page(chars, cover, cfg):
    """这一页算不算「扫描页」：字符少 **且** 图片铺满。

    cover 缺失（老缓存、或读不到图片信息）时退回「只看字符数」，
    并仍按扫描页算 —— 宁可多送一次 OCR，也不要漏。
    """
    c = cfg["pdf_type"]
    if chars >= c["min_chars_per_page"]:
        return False
    if cover is None:
        return True
    return cover >= c["scan_cover_ratio"]


def classify(path, cfg, page_counts, page_covers=None, total_pages=0):
    """判一个 PDF 的类型。

    page_counts: 每页字符数（可能只抽了前 N 页）
    page_covers: 每页图片覆盖率，和 page_counts 一一对应；可为 None
    total_pages: 文档总页数
    返回 dict；打不开的文件 kind = "unreadable"
    """
    c = cfg["pdf_type"]
    if page_counts is None:
        return {"kind": "unreadable", "scanned_ratio": None, "sampled_pages": 0,
                "total_pages": total_pages, "uncertain": False,
                "reason": "PDF 打不开（损坏或加密）"}

    if not page_counts:
        return {"kind": "scanned", "scanned_ratio": 1.0, "sampled_pages": 0,
                "total_pages": total_pages, "uncertain": False,
                "reason": "没有可读页"}

    covers = page_covers or [None] * len(page_counts)
    flags = [_is_scan_page(n, cv, cfg) for n, cv in zip(page_counts, covers)]
    scanned = sum(flags)
    ratio = scanned / len(page_counts)

    if ratio >= c["scanned_ratio"]:
        kind, reason = "scanned", f"扫描页占比 {ratio:.0%} ≥ {c['scanned_ratio']:.0%}"
    elif ratio <= c["text_ratio"]:
        kind, reason = "text", f"扫描页占比 {ratio:.0%} ≤ {c['text_ratio']:.0%}"
    else:
        kind = "mixed"
        reason = (f"扫描页占比 {ratio:.0%} 落在 "
                  f"{c['text_ratio']:.0%}~{c['scanned_ratio']:.0%} 之间")

    # 抽样不全时判定可能不准 → 标进待确认
    sampled = len(page_counts)
    partial = c["sample_pages"] and total_pages > c["sample_pages"]
    uncertain = bool(c["emit_uncertain"] and (kind == "mixed"
                                              or (partial and 0.4 < ratio < 0.9)))

    return {"kind": kind, "scanned_ratio": round(ratio, 4), "sampled_pages": sampled,
            "total_pages": total_pages, "uncertain": uncertain,
            "reason": reason + (f"（只抽了前 {sampled}/{total_pages} 页）" if partial else "")}


def summarize(results):
    """汇总成报告里的一段。results 是 [(路径, 判定dict)]"""
    from collections import Counter
    kinds = Counter(r["kind"] for _, r in results)
    total = sum(kinds.values()) or 1
    uncertain = [p for p, r in results if r.get("uncertain")]
    unreadable = [p for p, r in results if r["kind"] == "unreadable"]
    return {
        "counts": dict(kinds),
        "ratios": {k: round(v / total, 4) for k, v in kinds.items()},
        "uncertain_count": len(uncertain),
        "unreadable_count": len(unreadable),
        "_uncertain": uncertain,
        "_unreadable": unreadable,
    }


def to_json(results, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for p, r in results:
            f.write(json.dumps({"path": str(p), **r}, ensure_ascii=False) + "\n")
