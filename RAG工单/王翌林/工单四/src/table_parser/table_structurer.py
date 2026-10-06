# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
src/table_parser/table_structurer.py —— 工单三表格结构化

职责（见 docs/01_表格解析方案.md §四）：
  1. 解析合并单元格（rowspan/colspan）：基于列宽对齐推断
  2. 提取表头：首行指纹匹配 + 关键词规则
  3. 处理多级表头：连续表头行合并为单层（用 / 拼接）
  4. 跨页表格合并：表头继承 + 数据行拼接
  5. 输出标准化 JSON（见 docs/01 §六 输出结构）
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# 工单三：表头关键词（出现即视为表头行）
HEADER_KEYWORDS = {
    "项目", "类别", "名称", "金额", "比例", "占比", "金额（元）", "数量", "期间", "年份",
    "公司", "姓名", "职务", "关系", "关联方", "客户", "供应商", "股东", "持股", "股份",
    "项目名", "募集资金", "用途", "投入", "合计", "总计", "小计", "排名",
}
# 工单三：表头常见后缀（用于多级表头合并）
HEADER_SUFFIX_RE = re.compile(r"(类别|名称|金额|比例|占比|数量|期间|年份)$")


# ================= 工具 =================
def _cell_tokens(cell: str) -> set:
    """工单三：单元格分词，用于相似度比较"""
    return {t for t in re.split(r"[\s/、，,;；]+", cell) if t}


def _row_similarity(a: List[str], b: List[str]) -> float:
    """工单三：两行 Jaccard 相似度（按单元格 token 集合）"""
    sa = set()
    for c in a:
        sa |= _cell_tokens(c)
    sb = set()
    for c in b:
        sb |= _cell_tokens(c)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _looks_like_header(row: List[str]) -> bool:
    """工单三：判定一行是否为表头

    判定规则（避免子串误匹配，如"客户A"误命中"客户"）：
      - 至少一个单元格完全等于表头关键词；或
      - 整行无数字且非空（典型表头特征）
    """
    if not row:
        return False
    text = "".join(row)
    if not text.strip():
        return False
    # 工单三：精确匹配关键词（避免"客户A"误命中"客户"）
    has_exact_kw = any(c.strip() in HEADER_KEYWORDS for c in row)
    has_digit = any(ch.isdigit() for ch in text)
    return has_exact_kw or (not has_digit and len(text) >= 4)


# ================= 合并单元格推断 =================
def _infer_merged_cells(raw_rows: List[List[str]]) -> List[Dict[str, Any]]:
    """工单三：基于列对齐推断合并单元格

    pdfplumber/camelot 返回二维数组丢失了 rowspan/colspan，这里用规则推断：
      - 某格为空且上方同列非空 → 推断 rowspan（沿用上值）
      - 相邻列文本相同且看起来是横向合并 → 推断 colspan
    输出 cells 列表（r/c/text/rowspan/colspan）
    """
    if not raw_rows:
        return []
    max_cols = max(len(r) for r in raw_rows)
    # 补齐每行到 max_cols
    rows_pad = [r + [""] * (max_cols - len(r)) for r in raw_rows]
    cells: List[Dict[str, Any]] = []
    merged: List[Dict[str, Any]] = []

    for r, row in enumerate(rows_pad):
        for c in range(max_cols):
            val = row[c]
            # rowspan 推断：当前为空，且上方非空 → 复用上值，标记 rowspan
            if val == "" and r > 0:
                up_idx = None
                for k in range(r - 1, -1, -1):
                    if rows_pad[k][c] != "":
                        up_idx = k
                        break
                if up_idx is not None:
                    cells.append({"r": r, "c": c, "text": rows_pad[up_idx][c],
                                  "rowspan": r - up_idx, "colspan": 1})
                    merged.append({"r": r, "c": c, "type": "rowspan", "span": r - up_idx,
                                   "from": up_idx})
                    continue
            cells.append({"r": r, "c": c, "text": val, "rowspan": 1, "colspan": 1})
    # colspan 推断：相邻列同值（横向合并）
    for r, row in enumerate(rows_pad):
        c = 0
        while c < max_cols:
            val = row[c]
            if val == "":
                c += 1
                continue
            span = 1
            while c + span < max_cols and row[c + span] == val and val != "":
                span += 1
            if span > 1:
                for k in range(span):
                    cidx = next(i for i, cell in enumerate(cells)
                                if cell["r"] == r and cell["c"] == c + k)
                    cells[cidx]["colspan"] = span if k == 0 else 0
                merged.append({"r": r, "c": c, "type": "colspan", "span": span})
            c += span
    return cells


# ================= 多级表头处理 =================
def _merge_multi_level_headers(rows: List[List[str]], header_end: int) -> Tuple[List[str], int]:
    """工单三：多级表头合并为单层

    连续表头行用 / 拼接到首表头行，保留原列名层级。
    例如 [['项目','金额'], ['项目','金额（元）']] → ['项目/项目','金额/金额（元）']
    """
    if header_end <= 0:
        return [], 0
    header_rows = rows[:header_end]
    max_cols = max(len(r) for r in header_rows)
    pad = [r + [""] * (max_cols - len(r)) for r in header_rows]
    merged_header: List[str] = []
    for c in range(max_cols):
        parts = [pad[r][c] for r in range(len(pad)) if pad[r][c] != ""]
        merged_header.append("/".join(parts) if parts else f"col_{c+1}")
    return merged_header, header_end


def _detect_header_end(rows: List[List[str]]) -> int:
    """工单三：判定连续表头行的结束位置（1-based 不含）

    连续从首行开始判定，一旦某行不像表头就停止。
    最多前 3 行视为表头（多级表头通常 ≤3 行）。
    """
    end = 0
    for i, r in enumerate(rows[:3]):
        if _looks_like_header(r):
            end = i + 1
        else:
            break
    return end


# ================= 跨页合并 =================
def _merge_crosspage(tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """工单三：跨页表格合并

    触发条件：相邻页、列数相同、首行（次页）vs 末尾表头/数据行相似度 ≥ 0.5。
    合并策略：表头继承自前表，数据行追加；page_range 合并。
    """
    if len(tables) <= 1:
        for t in tables:
            t["page_range"] = [t["page"], t["page"]]
        return tables

    # 先各自结构化好（headers + rows 数据行）
    merged: List[Dict[str, Any]] = []
    cur = dict(tables[0])
    cur["page_range"] = [cur["page"], cur["page"]]

    for nxt in tables[1:]:
        is_continuation = (
            nxt["page"] == cur["page_range"][1] + 1
            and nxt.get("col_count") == cur.get("col_count")
        )
        if is_continuation and cur.get("headers") and nxt.get("rows"):
            # 工单三：相似度判定
            # 1) 次页表头 vs 当前表头（两页各自检测出的表头）
            sim_head = 0.0
            if nxt.get("headers") and not nxt.get("header_inferred"):
                sim_head = _row_similarity(cur["headers"], nxt["headers"])
            # 2) 次页首行（原始首行）vs 当前表头
            #    nxt["rows"][0] 是数据行（表头已被剥离），若次页有显式表头则用表头比较
            # 3) 次页首行 vs 当前末行（同表延续）
            sim_last = 0.0
            if cur.get("rows") and nxt.get("rows"):
                sim_last = _row_similarity(cur["rows"][-1], nxt["rows"][0])
            if sim_head >= 0.5 or sim_last >= 0.5:
                # 合并：次页表头已被剥离，数据行直接追加
                cur["rows"] = (cur.get("rows") or []) + nxt["rows"]
                cur["row_count"] = len(cur["rows"])
                cur["page_range"][1] = nxt["page"]
                cur["crosspage_merged"] = True
                logger.debug(
                    f"[table_structurer] 跨页合并 page {cur['page_range'][0]}→{nxt['page']}"
                )
                continue
        # 不连续则收尾当前表
        merged.append(cur)
        cur = dict(nxt)
        cur["page_range"] = [cur["page"], cur["page"]]
    merged.append(cur)
    return merged


# ================= 单表结构化 =================
def _structure_one(raw: Dict[str, Any], table_id: str) -> Dict[str, Any]:
    """工单三：单张表结构化"""
    rows = raw.get("raw_rows") or []
    if not rows:
        return {
            "table_id": table_id,
            "page": raw["page"],
            "page_range": [raw["page"], raw["page"]],
            "headers": [],
            "rows": [],
            "cells": [],
            "merged_cells": [],
            "header_inferred": False,
            "ocr_source": raw.get("source", "unknown"),
            "empty": True,
        }

    header_end = _detect_header_end(rows)
    header_inferred = False
    if header_end > 0:
        headers, _ = _merge_multi_level_headers(rows, header_end)
        data_rows = rows[header_end:]
    else:
        # 无显式表头：自动命名 col_1... 并标记
        max_cols = max(len(r) for r in rows)
        headers = [f"col_{i+1}" for i in range(max_cols)]
        data_rows = rows
        header_inferred = True

    cells = _infer_merged_cells(rows)
    merged_cells = [c for c in cells if c.get("rowspan", 1) > 1 or c.get("colspan", 1) > 1]

    # 表名（caption）：用首格内容或表头拼一个粗略名
    caption = ""
    if headers:
        caption = "、".join(h.split("/")[-1] for h in headers[:3])
    elif rows and rows[0]:
        caption = "、".join(c for c in rows[0] if c)[:40]

    return {
        "table_id": table_id,
        "page": raw["page"],
        "page_range": [raw["page"], raw["page"]],
        "caption": caption,
        "headers": headers,
        "rows": data_rows,
        "cells": cells,
        "merged_cells": merged_cells,
        "header_inferred": header_inferred,
        "ocr_source": raw.get("source", "unknown"),
        "low_confidence": False,
        "raw_bbox": raw.get("bbox"),
        "col_count": raw.get("col_count"),
        "row_count": len(data_rows),
    }


# ================= 主入口 =================
def structure_tables(
    raw_tables: List[Dict[str, Any]],
    doc_id: str,
    company: Optional[str] = None,
    doc_type: str = "招股说明书",
) -> List[Dict[str, Any]]:
    """工单三：把 table_extractor 输出的 raw_tables 结构化

    Args:
        raw_tables: extract_tables_from_pdf 的返回值
        doc_id: 文档 ID
        company: 公司名（可选，回填进每张表）
        doc_type: 文档类型
    Returns:
        List[structured_table]，见 docs/01_表格解析方案.md §六
    """
    if not raw_tables:
        logger.warning(f"[table_structurer] doc_id={doc_id} raw_tables 为空")
        return []

    # 1) 单表结构化（先各做一次，拿 headers/data_rows 用于跨页判断）
    partial: List[Dict[str, Any]] = []
    for i, raw in enumerate(raw_tables):
        # 截短 doc_id 防止 table_id 过长；跨页合并后会重新分配
        tid = f"tbl_{i+1:03d}_{raw['page']:03d}"
        s = _structure_one(raw, tid)
        s["doc_id"] = doc_id
        s["company"] = company
        s["doc_type"] = doc_type
        partial.append(s)

    # 2) 跨页合并
    merged = _merge_crosspage(partial)

    # 3) 重新分配 table_id（保证跨页合并后唯一）
    for i, t in enumerate(merged, 1):
        t["table_id"] = f"tbl_{i:03d}"
        # 重新计算 cells / merged_cells（基于合并后的 rows）
        if t.get("rows"):
            t["cells"] = _infer_merged_cells(t["rows"])
            t["merged_cells"] = [c for c in t["cells"]
                                 if c.get("rowspan", 1) > 1 or c.get("colspan", 1) > 1]
            t["row_count"] = len(t["rows"])

    logger.info(
        f"[table_structurer] doc_id={doc_id} 结构化完成：raw={len(raw_tables)} → "
        f"structured={len(merged)}（跨页合并 {len(raw_tables) - len(merged)} 处）"
    )
    return merged
