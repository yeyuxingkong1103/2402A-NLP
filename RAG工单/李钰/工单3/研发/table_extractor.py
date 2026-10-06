# -*- coding: utf-8 -*-
"""
结构化表格提取器
工单编号: 人工智能 NLP-RAG-PDF 文档的表格解析及检索优化

核心改进 (对比 V1/V2 的 " | " 文本拼接):
    1. 表头行自动识别 (包含项目/名称/金额等关键词)
    2. 表格转为 JSON 结构化格式 (columns + rows + summary)
    3. 每张表格独立存储, 不碎片化
    4. 生成可读的 summary 供检索使用
"""
import os
import json
import re
import logging
from typing import List, Dict

import config_v3 as config

logger = logging.getLogger(__name__)


def _is_header_row(cells: List[str]) -> bool:
    """判断一行是否为表头行"""
    headers = [c for c in cells if c and c.strip()]
    if not headers:
        return False
    # 关键词匹配
    hits = sum(1 for c in headers
               if any(kw in c for kw in config.TABLE_HEADER_KEYWORDS))
    # 至少 30% 的单元格含关键词
    return hits / max(len(headers), 1) >= 0.3


def _clean_cell(val) -> str:
    """清理单元格值"""
    if val is None:
        return ""
    s = str(val).strip()
    # 去除多余空白
    s = re.sub(r"\s+", " ", s)
    return s


def _parse_table(raw_table: List[List[str]], page: int,
                 table_id: str, doc_id: str, company: str) -> Dict:
    """
    将 pdfplumber 提取的原始表格转为结构化 JSON

    Returns:
        {
            "table_id": str,
            "doc_id": str,
            "company": str,
            "page": int,
            "table_name": str,
            "columns": List[str],
            "rows": List[List[str]],
            "summary": str,       # 可读摘要
            "text_blocks": List[str],  # 供 TF-IDF 检索的文本片段
        }
    """
    # 清洗
    cells = [[_clean_cell(c) for c in row] for row in raw_table]
    # 过滤全空行
    cells = [r for r in cells if any(c for c in r)]

    if len(cells) < config.TABLE_MIN_ROWS or \
       any(len(r) < config.TABLE_MIN_COLS for r in cells):
        return {}

    # 识别表头
    header_row_idx = 0
    if _is_header_row(cells[0]):
        columns = cells[0]
        rows = cells[1:]
    else:
        columns = [f"列{i+1}" for i in range(len(cells[0]))]
        rows = cells

    # 对齐行数 (补空)
    max_cols = max(len(r) for r in [columns] + rows)
    columns = columns + [""] * (max_cols - len(columns))
    rows = [r + [""] * (max_cols - len(r)) for r in rows]

    # 生成表名 (从表头推断)
    table_name = " ".join([c for c in columns if c][:3]) or f"表格{table_id}"

    # 生成可读摘要
    summary_parts = []
    header_str = " / ".join([c for c in columns if c])
    summary_parts.append(f"表格: {table_name} (页{page}, 表头: {header_str})")
    for r in rows[:5]:  # 前5行
        row_items = [f"{c}" for c in r if c]
        if row_items:
            summary_parts.append(" → ".join(row_items))
    summary = "；".join(summary_parts)

    # 生成供检索的文本块 (每列的 key-value 格式)
    text_blocks = []
    for r in rows:
        kv_pairs = []
        for i, col in enumerate(columns):
            if col and i < len(r) and r[i]:
                kv_pairs.append(f"{col}: {r[i]}")
        if kv_pairs:
            text_blocks.append(" | ".join(kv_pairs))
    # 也加入表头+第一行的描述块
    text_blocks.insert(0, f"【表格 {table_name}】 表头: {header_str}; 共{len(rows)}行数据")

    return {
        "table_id": table_id,
        "doc_id": doc_id,
        "company": company,
        "page": page,
        "table_name": table_name,
        "columns": columns,
        "rows": rows,
        "summary": summary,
        "text_blocks": text_blocks,
    }


def extract_tables_from_pdf(pdf_path: str, doc_id: str,
                            company: str) -> List[Dict]:
    """
    从 PDF 提取所有结构化表格

    Returns:
        List[Dict] - 结构化表格列表
    """
    if not os.path.exists(pdf_path):
        logger.warning(f"表格提取: PDF 不存在 {pdf_path}")
        return []

    try:
        import pdfplumber
    except ImportError:
        raise ImportError("请安装 pdfplumber")

    tables_result = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw_tables = page.extract_tables()
            for ti, raw in enumerate(raw_tables):
                if not raw:
                    continue
                table = _parse_table(raw, page=i,
                                     table_id=f"{doc_id}_p{i}_t{ti+1}",
                                     doc_id=doc_id, company=company)
                if table:
                    tables_result.append(table)

    logger.info(f"[表格提取] {company}: {len(tables_result)} 张结构化表格")
    return tables_result


def extract_text_from_pdf(pdf_path: str, doc_id: str,
                          company: str) -> List[Dict]:
    """提取非表格文本块 (语义分块, 复用 V2 逻辑)"""
    try:
        import pdfplumber
    except ImportError:
        return []

    try:
        import sys
        v2_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "工单2", "研发")
        v2_dir = os.path.abspath(v2_dir)
        if v2_dir not in sys.path:
            sys.path.insert(0, v2_dir)
        from pdf_parser_v2 import extract_text_from_pdf as v2_extract
        from pdf_parser_v2 import build_semantic_chunks
        # 需要用 V2 的 config, 简化: 直接用 pdfplumber + 简单切块
    except Exception as e:
        logger.warning(f"V2 分块模块不可用 ({e}), 使用简单切块")

    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages.append({"page": i, "text": text,
                              "doc_id": doc_id, "company": company})
    except Exception as e:
        logger.error(f"文本提取失败: {e}")
        return []

    # 简单切块 (500字符, 80重叠)
    chunks = []
    chunk_id = 0
    for p in pages:
        t = p["text"]
        if not t.strip():
            continue
        start = 0
        while start < len(t):
            end = start + 500
            chunk = {
                "id": chunk_id,
                "doc_id": doc_id,
                "company": company,
                "page": p["page"],
                "text": t[start:end].strip(),
                "type": "text",
            }
            chunks.append(chunk)
            chunk_id += 1
            start = end - 80
    logger.info(f"[文本提取] {company}: {len(chunks)} 个文本块")
    return chunks


def save_tables(tables: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tables, f, ensure_ascii=False, indent=2)


def load_tables(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # 测试预设数据的表格解析
    with open(config.PRESET_TABLES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    for t in data["tables"]:
        print(f"\n=== {t['table_name']} ===")
        print(f"  列: {t['columns']}")
        for r in t["rows"]:
            print(f"  行: {r}")
        print(f"  摘要: {t['summary'][:80]}")
