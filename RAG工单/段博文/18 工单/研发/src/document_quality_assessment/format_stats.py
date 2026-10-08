# -*- coding: utf-8 -*-
"""功能1：格式分布统计。"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List


def collect_files(root: Path, extensions: List[str], recursive: bool) -> tuple[List[Path], List[Path]]:
    """按后缀收集文件，返回 (纳入评估的文件, 不支持的文件)。"""
    exts = {e.lower() for e in extensions}
    pattern = "**/*" if recursive else "*"
    matched, unsupported = [], []
    for p in root.glob(pattern):
        if not p.is_file():
            continue
        (matched if p.suffix.lower() in exts else unsupported).append(p)
    return sorted(matched), sorted(unsupported)


def format_distribution(files: List[Path]) -> dict:
    """统计各后缀的数量与占比。"""
    counter = Counter(f.suffix.lower() or "(无后缀)" for f in files)
    total = len(files)
    items = []
    for ext, cnt in counter.most_common():
        items.append({"format": ext, "count": cnt, "ratio": round(cnt / total, 4) if total else 0})
    return {"total_files": total, "by_format": items}
