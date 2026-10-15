# -*- coding: utf-8 -*-
"""
DQA · 格式分布统计
遍历文件夹，统计 .pdf/.docx/.md 等各类文件的数量与占比。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

DEFAULT_EXT_GROUPS = {
    "文档": {".docx", ".doc", ".wps"},
    "PDF": {".pdf"},
    "表格": {".xlsx", ".xls", ".csv", ".tsv"},
    "演示": {".pptx", ".ppt"},
    "文本/标记": {".md", ".markdown", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".html"},
    "图片": {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"},
    "其他": set(),
}


def _group_of(ext: str, groups: dict) -> str:
    for name, exts in groups.items():
        if ext in exts:
            return name
    return "其他"


def analyze_format_distribution(root: str | Path,
                                groups: dict | None = None,
                                recursive: bool = True) -> dict:
    root = Path(root)
    groups = groups or DEFAULT_EXT_GROUPS
    if not root.exists():
        raise FileNotFoundError(f"目录不存在：{root}")

    ext_count: dict = {}
    group_count: dict = {}
    total = 0
    total_bytes = 0
    files = []

    walker: Iterable[Path] = root.rglob("*") if recursive else root.glob("*")
    for p in walker:
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        total += 1
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        total_bytes += size
        ext_count[ext or "(无扩展名)"] = ext_count.get(ext or "(无扩展名)", 0) + 1
        g = _group_of(ext, groups)
        group_count[g] = group_count.get(g, 0) + 1
        files.append({"path": str(p), "ext": ext, "size": size})

    def pct(n):
        return round(n / total, 4) if total else 0.0

    return {
        "根目录": str(root),
        "文件总数": total,
        "总大小MB": round(total_bytes / 1024 / 1024, 2),
        "按扩展名": {k: {"数量": v, "占比": pct(v)}
                     for k, v in sorted(ext_count.items(), key=lambda x: -x[1])},
        "按类别": {k: {"数量": v, "占比": pct(v)}
                   for k, v in sorted(group_count.items(), key=lambda x: -x[1])},
        "_files": files,
    }
