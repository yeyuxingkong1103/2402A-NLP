# -*- coding: utf-8 -*-
"""
table_parser.py — PDF 表格解析模块
工单编号：人工智能NLP-RAG-PDF 文档的表格解析及检索优化

使用 pymupdf 的表格识别能力，把 PDF 中的表格解析为结构化、可检索的文本。
对中文财报类 PDF 的复杂版面做了误报过滤，只保留真实数据表格。
"""
from __future__ import annotations

from typing import List, Dict, Any

import rag_core


def _clean(rows: List[List[str]]) -> List[List[str]]:
    """清理行：去空、去空行、统一为字符串。"""
    rows = [[str(c).strip() for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    return rows


def _is_real_table(rows: List[List[str]]) -> bool:
    """过滤 pymupdf 的误报（单字符碎片 / 版面噪声），保留真实表格。

    判定条件：
      - 至少 2 行、2 列
      - 平均单元格长度 >= 2（过滤单字碎片）
      - 总字符数 >= 20
    """
    if len(rows) < 2:
        return False
    if any(len(r) < 2 for r in rows):
        return False
    cells = [c for r in rows for c in r if c]
    if not cells:
        return False
    avg_len = sum(len(c) for c in cells) / len(cells)
    if avg_len < 2.0:
        return False
    if sum(len(c) for c in cells) < 20:
        return False
    return True


def parse_tables(pdf_path: str) -> List[Dict[str, Any]]:
    """解析 PDF 全部页面，返回真实表格。

    返回：[{page_no, raw_rows, text}]，text 为格式化后的可检索文本。
    """
    pages = rag_core.load_pdf_pages(pdf_path)
    results: List[Dict[str, Any]] = []
    for p in pages:
        for raw in p["tables"]:
            rows = _clean(raw)
            if not _is_real_table(rows):
                continue
            results.append({
                "page_no": p["page_no"],
                "raw_rows": rows,
                "text": table_to_text(rows, p["page_no"]),
            })
    return results


def table_to_text(rows: List[List[str]], page_no: int = 0) -> str:
    """把表格行格式化为可检索文本。

    约定：第一行为表头，其余行按「表头: 单元格值」拼接，
    便于向量/全文检索定位到具体字段。
    """
    rows = _clean(rows)
    if not rows:
        return ""

    header = rows[0]
    if len(rows) == 1:
        return f"[表格/第{page_no}页] {'；'.join(header)}"

    lines = []
    for row in rows[1:]:
        while len(row) < len(header):
            row.append("")
        pairs = []
        for h, v in zip(header, row):
            if v and h:
                pairs.append(f"{h}: {v}")
            elif v:
                pairs.append(v)
        if pairs:
            lines.append("；".join(pairs))
    body = "\n".join(lines)
    return f"[表格/第{page_no}页] 表头：{' | '.join(header)}\n{body}"


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "../附件/招股说明书2.pdf"
    tables = parse_tables(path)
    print(f"共解析出 {len(tables)} 个真实表格")
    for t in tables[:5]:
        print("-" * 60)
        print(f"第 {t['page_no']} 页：\n{t['text'][:260]}")
