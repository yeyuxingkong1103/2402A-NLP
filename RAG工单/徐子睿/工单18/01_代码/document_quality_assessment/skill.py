# -*- coding: utf-8 -*-
"""
DocumentQualityAssessmentSkill · 主入口
assess(root, cfg) -> 结构化报告 dict
支持进度回调与中断恢复（已处理文件记入 progress_file，重跑跳过）。
"""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime
from typing import Callable, List, Optional

from . import detectors as D
from .config import Config


def assess(root: str,
           cfg: Optional[Config] = None,
           limit: int = 0,
           progress_cb: Optional[Callable[[int, int, str], None]] = None,
           progress_file: Optional[str] = None) -> dict:
    cfg = cfg or Config()
    files = list(D.iter_files(root, cfg.extensions))
    if limit:
        files = files[:limit]
    total = len(files)

    # —— 中断恢复：已完成的文本/类型结果缓存 ——
    cache = {}
    if progress_file and os.path.exists(progress_file):
        try:
            cache = json.load(open(progress_file, encoding="utf-8"))
        except Exception:
            cache = {}

    texts = {}          # path -> text
    pdf_types = {}      # path -> dict
    lengths = []        # 文档字符数分布
    labels = Counter()

    for i, f in enumerate(files, 1):
        if f in cache and "text" in cache[f]:
            txt = cache[f]["text"]
            pt = cache[f].get("pdf_type")
        else:
            ext = os.path.splitext(f)[1].lower()
            pt = D.pdf_page_type(f, cfg) if ext == ".pdf" else None
            txt = D.read_text(f)
            cache[f] = {"text": txt, "pdf_type": pt}
            if progress_file and i % 100 == 0:
                json.dump(cache, open(progress_file, "w", encoding="utf-8"), ensure_ascii=False)
        texts[f] = txt
        if pt is not None:
            pdf_types[f] = pt
        lengths.append(len(txt))
        labels[D.classify_doc(f, (pt or {}).get("type"), len(txt))] += 1
        if progress_cb and (i % 25 == 0 or i == total):
            progress_cb(i, total, os.path.basename(f))

    if progress_file:
        json.dump(cache, open(progress_file, "w", encoding="utf-8"), ensure_ascii=False)

    # —— 汇总 ——
    pdf_vals = list(pdf_types.values())
    openable = [v for v in pdf_vals if "error" not in v]
    pdf_summary = {
        "pdf_total": len(pdf_vals),
        "scan": sum(1 for v in openable if v["type"] == "扫描型"),
        "text": sum(1 for v in openable if v["type"] == "文字型"),
        "mixed": sum(1 for v in openable if v["type"] == "混合型"),
        "待确认": [{"file": k, **v} for k, v in pdf_types.items()
                   if "error" in v or v.get("type", "").startswith("待确认")],
    }

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "meta": {"root": root, "files": total, "config": cfg.to_dict()},
        "format_stats": D.format_stats(files),
        "pdf_types": pdf_summary,
        "length_distribution": D.length_distribution(lengths, cfg.length_quantiles),
        "duplicates": {
            "md5": D.md5_duplicates(files),
            "near": D.near_duplicates(texts, cfg),
        },
        "sensitive": D.sensitive_scan(texts, cfg),
        "classification": {
            "label_counts": dict(labels.most_common()),
            "待确认": [],
        },
    }
    return report
