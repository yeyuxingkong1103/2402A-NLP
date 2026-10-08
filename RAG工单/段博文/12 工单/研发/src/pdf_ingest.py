# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-LightRAG优化
"""
PDF 解析模块：提取两份招股说明书全文并清洗。

处理逻辑：
    1. pymupdf 逐页提取文本；
    2. 清洗页眉页脚、多余空白；
    3. 按文档拼接为完整文本，保存为 txt 供 LightRAG 与传统 RAG 共用。
"""

import re
import time
from pathlib import Path

import pymupdf

from config import PDF_FILES, TXT_DIR
from logger import get_logger

logger = get_logger(__name__)

# 常见页眉页脚噪声模式
_NOISE_PATTERNS = [
    re.compile(r"^武汉力源信息科技股份有限公司\s*招股意向书\s*\d*$"),
    re.compile(r"^武汉兴图新科电子股份有限公司\s*招股意向书\s*\d*$"),
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"^第\s*\d+\s*页"),
]


def clean_text(text: str) -> str:
    """清洗单页文本：去页眉页脚、合并空白。"""
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if any(p.match(s) for p in _NOISE_PATTERNS):
            continue
        lines.append(s)
    return "\n".join(lines)


def extract_pdf(pdf_path: Path) -> str:
    """提取单个 PDF 的清洗后全文。"""
    doc = pymupdf.open(pdf_path)
    pages = []
    for page in doc:
        t = clean_text(page.get_text())
        if t:
            pages.append(t)
    doc.close()
    return "\n\n".join(pages)


def ingest_all() -> dict:
    """提取全部 PDF，保存 txt，返回统计信息。"""
    stats = {}
    for name, path in PDF_FILES.items():
        t0 = time.time()
        text = extract_pdf(path)
        out = TXT_DIR / (Path(name).stem + ".txt")
        out.write_text(text, encoding="utf-8")
        stats[name] = {
            "chars": len(text),
            "output": str(out),
            "time": round(time.time() - t0, 2),
        }
        logger.info(f"PDF 解析完成：{name} -> {len(text)} 字符（{stats[name]['time']}s）")
    return stats


if __name__ == "__main__":
    for name, s in ingest_all().items():
        print(f"{name}: {s['chars']} 字符 -> {s['output']}")
