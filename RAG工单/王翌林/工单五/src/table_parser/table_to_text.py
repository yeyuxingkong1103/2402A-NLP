# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
src/table_parser/table_to_text.py —— 工单三表格转自然语言描述

职责（见 docs/02_表格检索优化方案.md §四）：
  把结构化表格转为自然语言文本，供 bge-m3 向量化使用。

输出示例（输入表格：发行股数与募集资金）：
  表标题: 发行股数与募集资金
  列: 项目、金额、比例
  行1: 发行股数 为 12 345 678，占比 100%
  行2: 募集资金总额 为 98 765 432 元
"""
import re
from typing import Any, Dict, List, Optional

# 工单三：列名归一化（"金额（元）" → "金额"），用于自然语言连接词
COL_NORMALIZE = {
    "金额": "金额", "金额（元）": "金额", "金额(元)": "金额", "金额（万元）": "金额（万元）",
    "比例": "比例", "占比": "比例", "百分比": "比例",
    "数量": "数量", "股数": "数量",
    "金额（股）": "数量", "金额(股)": "数量",
}


def _norm_col(col: str) -> str:
    """工单三：列名归一化（取多级表头末段）"""
    if not col:
        return ""
    last = col.split("/")[-1]
    return COL_NORMALIZE.get(last, last)


def _row_to_sentence(row: List[str], headers: List[str]) -> str:
    """工单三：单行转自然语言

    规则：
      - 行首列（通常是项目名/类别名）作为主语
      - 其余每列拼成 "列为 V" 形式
      - 含"比例/占比"列 → "占比 V"
    """
    row_pad = list(row) + [""] * (len(headers) - len(row))
    if not row_pad or all(c == "" for c in row_pad):
        return ""

    subject = row_pad[0].strip()
    parts: List[str] = []
    for i, (val, col) in enumerate(zip(row_pad[1:], headers[1:])):
        if val == "":
            continue
        col_n = _norm_col(col)
        if "比例" in col_n or col_n == "比例":
            parts.append(f"占比 {val}")
        elif "金额" in col_n:
            parts.append(f"金额为 {val}")
        elif "数量" in col_n or col_n == "数量":
            parts.append(f"数量为 {val}")
        else:
            parts.append(f"{col_n}为 {val}" if col_n else val)
    if not parts:
        # 全行只有首列（如合计/小计）
        return subject if subject else ""
    return f"{subject}，{'; '.join(parts)}" if subject else "；".join(parts)


def table_to_text(
    table: Dict[str, Any],
    doc_id: Optional[str] = None,
    company: Optional[str] = None,
    section_hint: Optional[str] = None,
) -> str:
    """工单三：单张结构化表格 → 自然语言描述

    Args:
        table: structure_tables 输出的结构化表
        doc_id: 文档 ID（回填到文本头部，便于引用）
        company: 公司名
        section_hint: 所属章节（可选）
    Returns:
        table_text: 多行自然语言文本
    """
    headers: List[str] = table.get("headers", []) or []
    rows: List[List[str]] = table.get("rows", []) or []
    caption: str = table.get("caption", "") or ""

    if not rows:
        return ""

    lines: List[str] = []
    # 1) 表头信息
    head = "表标题: " + (caption or "（无标题）")
    if section_hint:
        head += f" | 所属章节: {section_hint}"
    if company:
        head += f" | 公司: {company}"
    if doc_id:
        head += f" | 来源: {doc_id}"
    if table.get("page_range"):
        pr = table["page_range"]
        head += f" | 页码: {pr[0]}" + (f"-{pr[1]}" if pr[1] != pr[0] else "")
    lines.append(head)

    if headers:
        lines.append("列: " + "、".join(_norm_col(h) for h in headers))

    # 2) 行摘要
    for idx, row in enumerate(rows, 1):
        sent = _row_to_sentence(row, headers if headers else [f"col_{i+1}" for i in range(len(row))])
        if sent:
            lines.append(f"行{idx}: {sent}")

    return "\n".join(lines)


def tables_to_texts(
    tables: List[Dict[str, Any]],
    doc_id: Optional[str] = None,
    company: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """工单三：批量转 table_text，返回带 table_chunk_id 的列表"""
    out: List[Dict[str, Any]] = []
    for i, t in enumerate(tables, 1):
        text = table_to_text(t, doc_id=doc_id, company=company)
        out.append({
            "table_chunk_id": f"tc_{t.get('table_id', f'{i:03d}')}",
            "table_id": t.get("table_id"),
            "doc_id": doc_id or t.get("doc_id"),
            "company": company or t.get("company"),
            "page_range": t.get("page_range"),
            "caption": t.get("caption"),
            "table_text": text,
            "headers": t.get("headers"),
            "row_count": t.get("row_count"),
        })
    return out


# ================= CLI =================
def _main() -> int:  # pragma: no cover - CLI 仅供手动调试
    import argparse, json, sys
    from pathlib import Path

    from .table_extractor import extract_tables_from_pdf
    from .table_structurer import structure_tables

    parser = argparse.ArgumentParser(description="工单三：表格 → 文本 CLI")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--company", default=None)
    args = parser.parse_args()

    raw = extract_tables_from_pdf(args.pdf, doc_id=args.doc_id)
    doc_id = args.doc_id or Path(args.pdf).stem
    structured = structure_tables(raw, doc_id=doc_id, company=args.company)
    texts = tables_to_texts(structured, doc_id=doc_id, company=args.company)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"doc_id": doc_id, "table_count": len(structured), "table_texts": texts}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"doc_id": doc_id, "table_count": len(structured), "out": str(out)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(_main())
