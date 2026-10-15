# -*- coding: utf-8 -*-
"""
DQA · 文档长度分布
统计所有文档的字符数，输出分位数（P25/P50/P75/P90/P99）和长度区间分布。
"""
from __future__ import annotations

from pathlib import Path
from typing import List

try:
    import pymupdf as fitz
except ImportError:
    import fitz


def _char_count(path: Path) -> int:
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            doc = fitz.open(str(path))
            n = 0
            for page in doc:
                try:
                    n += len((page.get_text("text") or "").strip())
                except Exception:                                # noqa: BLE001
                    pass
            doc.close()
            return n
        if ext in (".txt", ".md", ".markdown", ".csv", ".json", ".jsonl", ".yaml", ".yml"):
            return len(path.read_text(encoding="utf-8", errors="ignore"))
        if ext in (".docx", ".doc"):
            try:
                from docx import Document                # python-docx
                d = Document(str(path))
                return sum(len(p.text) for p in d.paragraphs)
            except Exception:                                # noqa: BLE001
                return -1
        return -1
    except Exception:                                        # noqa: BLE001
        return -1


def _percentile(vals: List[int], p: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    k = (len(vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return round(vals[lo] + (vals[hi] - vals[lo]) * (k - lo), 1)


BINS = [(0, 200, "极短(<200)"), (200, 1000, "短(200-1k)"),
        (1000, 5000, "中(1k-5k)"), (5000, 20000, "长(5k-2w)"),
        (20000, 100000, "很长(2w-10w)"), (100000, 10 ** 12, "超长(>10w)")]


def analyze_length_distribution(root: str | Path, exts=None,
                                max_files: int | None = None,
                                progress=None) -> dict:
    root = Path(root)
    exts = exts or {".pdf", ".docx", ".doc", ".md", ".markdown", ".txt"}
    files = [p for p in sorted(root.rglob("*"))
             if p.is_file() and p.suffix.lower() in exts]
    if max_files:
        files = files[:max_files]

    lens, records, skipped = [], [], []
    for i, p in enumerate(files, 1):
        n = _char_count(p)
        if n < 0:
            skipped.append({"file": str(p), "reason": "无法解析或不支持的类型"})
            continue
        lens.append(n)
        records.append({"file": str(p), "chars": n})
        if progress and i % 20 == 0:
            progress(i, len(files))

    bins = []
    total = max(len(lens), 1)
    for lo, hi, label in BINS:
        cnt = sum(1 for n in lens if lo <= n < hi)
        bins.append({"区间": label, "数量": cnt, "占比": round(cnt / total, 4)})

    return {
        "统计文件数": len(lens),
        "跳过文件数": len(skipped),
        "跳过原因样例": skipped[:10],
        "总字符数": sum(lens),
        "平均字符数": round(sum(lens) / total, 1) if lens else 0,
        "P25": _percentile(lens, 0.25),
        "P50": _percentile(lens, 0.50),
        "P75": _percentile(lens, 0.75),
        "P90": _percentile(lens, 0.90),
        "P99": _percentile(lens, 0.99),
        "最小值": min(lens) if lens else 0,
        "最大值": max(lens) if lens else 0,
        "长度区间分布": bins,
        "_detail": records,
    }
