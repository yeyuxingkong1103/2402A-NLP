# -*- coding: utf-8 -*-
"""
DQA · PDF 页面类型识别（文字型 / 扫描型 / 混合型）

启发式规则（工单要求）：
  - 按页判断：每页字符数低于阈值 → 该页记作"扫描页"
  - 扫描页占比 > 阈值（默认 70%）→ 整个文件判为"扫描型"
  - 否则若存在少量扫描页 → "混合型"；否则 "文字型"
所有阈值可配置，且输出"待确认"列表（启发式不保证严格正确）。
"""
from __future__ import annotations

from pathlib import Path
from typing import List

try:
    import pymupdf as fitz          # PyMuPDF >= 1.24 的新包名
except ImportError:                 # 兼容旧版
    import fitz


def _page_char_count(page: fitz.Page) -> int:
    try:
        return len((page.get_text("text") or "").strip())
    except Exception:                                        # noqa: BLE001
        return 0


def classify_pdf_pages(path: str | Path, min_chars_per_page: int = 50,
                       scan_ratio_threshold: float = 0.7,
                       mixed_ratio_threshold: float = 0.15,
                       max_pages: int | None = None,
                       sample_step: int = 1) -> dict:
    """
    返回：
      type            : text | scanned | mixed | unknown
      scanned_pages   : 扫描页页码列表
      scanned_ratio   : 扫描页占比
      avg_chars       : 平均每页字符数
      needs_review    : 是否进入"待确认"列表
    """
    path = Path(path)
    out = {"file": str(path), "type": "unknown", "pages": 0,
           "scanned_pages": [], "scanned_ratio": 0.0, "avg_chars": 0,
           "needs_review": False, "error": ""}
    try:
        doc = fitz.open(str(path))
    except Exception as e:                                    # noqa: BLE001
        out["error"] = f"无法打开PDF：{e}"
        out["needs_review"] = True
        return out

    total = doc.page_count
    idxs = list(range(0, total, max(1, sample_step)))
    if max_pages:
        idxs = idxs[:max_pages]
    counts = []
    scanned = []
    for i in idxs:
        c = _page_char_count(doc[i])
        counts.append(c)
        if c < min_chars_per_page:
            scanned.append(i + 1)
    doc.close()

    if not counts:
        out["pages"] = total
        out["error"] = "空PDF"
        out["needs_review"] = True
        return out

    ratio = len(scanned) / len(counts)
    avg = sum(counts) / len(counts)
    if ratio >= scan_ratio_threshold:
        t = "scanned"
    elif ratio >= mixed_ratio_threshold:
        t = "mixed"
    else:
        t = "text"

    out.update({"type": t, "pages": total, "scanned_pages": scanned,
                "scanned_ratio": round(ratio, 4), "avg_chars": round(avg, 1)})
    # 启发式边界情况列入待确认
    if mixed_ratio_threshold <= ratio <= scan_ratio_threshold or avg < 120 or total == 0:
        out["needs_review"] = True
    return out


def scan_pdf_types(root: str | Path, exts=(".pdf",), min_chars_per_page: int = 50,
                   scan_ratio_threshold: float = 0.7, max_files: int | None = None,
                   progress=None) -> dict:
    """批量扫描目录下的 PDF，返回分类汇总 + 待确认列表"""
    root = Path(root)
    pdfs: List[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            pdfs.append(p)
    if max_files:
        pdfs = pdfs[:max_files]

    results = []
    for i, p in enumerate(pdfs, 1):
        r = classify_pdf_pages(p, min_chars_per_page, scan_ratio_threshold)
        results.append(r)
        if progress and i % 20 == 0:
            progress(i, len(pdfs))

    by_type = {}
    for r in results:
        by_type[r["type"]] = by_type.get(r["type"], 0) + 1
    n = max(len(results), 1)
    return {
        "PDF总数": len(results),
        "类型分布": {k: {"数量": v, "占比": round(v / n, 4)} for k, v in by_type.items()},
        "待确认列表": [{"file": r["file"], "type": r["type"],
                        "scanned_ratio": r["scanned_ratio"], "avg_chars": r["avg_chars"],
                        "reason": r["error"] or "启发式边界（扫描占比靠近阈值或平均字符数偏低）"}
                       for r in results if r["needs_review"]],
        "扫描型样例": [r["file"] for r in results if r["type"] == "scanned"][:20],
        "_detail": results,
    }
