# -*- coding: utf-8 -*-
"""
DQA · 敏感信息检测

支持：手机号、邮箱、身份证；银行卡检测默认关闭（误报率太高）。
关键设计：不返回"统计数字"，而是返回**待审核列表**，每条附带上下文片段，
方便操作员判断是真实敏感信息还是误报（例如专利号、说明书里的示例号码）。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

try:
    import pymupdf as fitz
except ImportError:
    import fitz

PATTERNS = {
    "手机号": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "邮箱": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "身份证": re.compile(r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])"
                        r"(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"),
    # 默认关闭：专利/标准文档里 16-19 位数字太多，误报率高
    "银行卡": re.compile(r"(?<!\d)(?:62|4|5[1-5])\d{14,17}(?!\d)"),
}
ENABLED_BY_DEFAULT = {"手机号": True, "邮箱": True, "身份证": True, "银行卡": False}


def _read_text(path: Path, cap: int = 300_000) -> str:
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            doc = fitz.open(str(path))
            buf = []
            for page in doc:
                try:
                    buf.append(page.get_text("text") or "")
                except Exception:                                # noqa: BLE001
                    pass
                if sum(len(b) for b in buf) > cap:
                    break
            doc.close()
            return "".join(buf)[:cap]
        if ext in (".txt", ".md", ".markdown", ".csv", ".json", ".jsonl"):
            return path.read_text(encoding="utf-8", errors="ignore")[:cap]
        if ext == ".docx":
            from docx import Document
            d = Document(str(path))
            return "\n".join(p.text for p in d.paragraphs)[:cap]
    except Exception:                                        # noqa: BLE001
        return ""
    return ""


def _context(text: str, start: int, end: int, pad: int = 40) -> str:
    s = max(0, start - pad)
    e = min(len(text), end + pad)
    return re.sub(r"\s+", " ", text[s:e])


def detect_sensitive_info(root: str | Path, exts=None,
                          enable: dict | None = None,
                          max_files: int | None = None,
                          progress=None) -> dict:
    root = Path(root)
    exts = exts or {".pdf", ".docx", ".doc", ".md", ".txt"}
    enable = enable or dict(ENABLED_BY_DEFAULT)
    files = [p for p in sorted(root.rglob("*"))
             if p.is_file() and p.suffix.lower() in exts]
    if max_files:
        files = files[:max_files]

    review: List[dict] = []          # 待审核列表
    counts = {k: 0 for k in PATTERNS}
    files_with_hit = set()

    for i, p in enumerate(files, 1):
        text = _read_text(p)
        if not text:
            continue
        for kind, pat in PATTERNS.items():
            if not enable.get(kind, ENABLED_BY_DEFAULT.get(kind, False)):
                continue
            for m in pat.finditer(text):
                counts[kind] += 1
                files_with_hit.add(str(p))
                review.append({
                    "类型": kind,
                    "命中内容": m.group(0),
                    "文件": str(p),
                    "上下文": _context(text, m.start(), m.end()),
                    "建议": "确认是否为真实敏感信息；示例/编号可忽略",
                })
        if progress and i % 20 == 0:
            progress(i, len(files))

    enabled_desc = [k for k, v in enable.items() if v]
    return {
        "检测文件数": len(files),
        "启用类型": enabled_desc,
        "各类型命中数": counts,
        "命中文件数": len(files_with_hit),
        "待审核敏感信息列表": review[:500],
        "待审核总数": len(review),
        "银行卡检测": enable.get("银行卡", False),
        "说明": "银行卡检测默认关闭（误报率高）；如需开启请改配置 enable.银行卡=true",
    }
