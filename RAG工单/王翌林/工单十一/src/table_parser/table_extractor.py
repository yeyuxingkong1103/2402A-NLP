# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
src/table_parser/table_extractor.py —— 工单三表格抽取器

职责（见 docs/01_表格解析方案.md §三）：
  1. 用 pdfplumber 定位并抽取 PDF 表格区域（主路径）
  2. camelot 兜底（lattice + stream，依赖 ghostscript，未安装则跳过）
  3. 输出原始表格列表（含 page / bbox / raw_rows）
  4. 跨页表格候选检测（结构化阶段在 table_structurer 中合并）

输出单表结构（raw_table）：
  {
    "doc_id": "招股说明书1",
    "page": 3,
    "bbox": [x0, top, x1, bottom],
    "raw_rows": [["项目","金额"], ["发行股数","12 345 678"]],
    "col_count": 2,
    "row_count": 2,
    "source": "pdfplumber" | "camelot-lattice" | "camelot-stream",
    "page_text_chars": 1234
  }

用法：
  python -m src.table_parser.table_extractor --pdf "../附件/招股说明书1.pdf" \
      --out "data/tables/招股说明书1_tables.json"
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

# 严格按依赖顺序尝试导入，缺失则降级（人工智能NLP-RAG-PDF文档的表格解析及检索优化）
try:
    import pdfplumber  # 主路径
except ImportError:  # pragma: no cover - 环境异常时跳过
    pdfplumber = None

try:
    import camelot  # 兜底路径
except ImportError:  # camelot 依赖 ghostscript，多数环境未装
    camelot = None


# ================= 工具函数 =================
def _safe_doc_id(pdf_path: str) -> str:
    """工单三：从 PDF 文件名生成 doc_id（招股说明书1.pdf → 招股说明书1）"""
    name = Path(pdf_path).stem
    return name or "unknown_doc"


def _bbox_to_list(bbox) -> Optional[List[float]]:
    """工单三：将 pdfplumber boundingbox 转为普通 list（便于 JSON 序列化）"""
    if bbox is None:
        return None
    try:
        return [float(round(x, 2)) for x in bbox]
    except Exception:
        return None


def _clean_cell(text: Any) -> str:
    """工单三：单元格文本清洗（保留换行、去首尾空白）"""
    if text is None:
        return ""
    return str(text).replace("\r", "").strip()


def _normalize_row(row: List[Any]) -> List[str]:
    """工单三：单行清洗，None → 空串"""
    return [_clean_cell(c) for c in row]


def _row_is_empty(row: List[str]) -> bool:
    """工单三：判定一行是否全空（用于过滤空行）"""
    return all(c == "" for c in row)


# ================= 主路径：pdfplumber =================
def _extract_with_pdfplumber(page) -> List[Dict[str, Any]]:
    """工单三：用 pdfplumber.extract_tables 抽取单页所有表格"""
    tables_out: List[Dict[str, Any]] = []
    try:
        tables = page.find_tables() or []
    except Exception as e:  # pragma: no cover - 极端 PDF 兜底
        logger.warning(f"[table_extractor] pdfplumber.find_tables 失败 page={page.page_number}: {e}")
        return tables_out

    for t_idx, tb in enumerate(tables):
        try:
            raw = tb.extract() or []
        except Exception as e:  # pragma: no cover
            logger.warning(f"[table_extractor] extract 失败 table#{t_idx} page={page.page_number}: {e}")
            continue

        rows: List[List[str]] = []
        for r in raw:
            r_norm = _normalize_row(r)
            if _row_is_empty(r_norm):
                continue  # 跳过空行
            rows.append(r_norm)
        if not rows:
            continue

        col_count = max(len(r) for r in rows)
        tables_out.append({
            "doc_id": "",  # 上层 extract_tables_from_pdf 回填
            "page": int(page.page_number),
            "bbox": _bbox_to_list(tb.bbox),
            "raw_rows": rows,
            "col_count": col_count,
            "row_count": len(rows),
            "source": "pdfplumber",
        })
    return tables_out


# ================= 兜底路径：camelot =================
def _extract_with_camelot(pdf_path: str, page_no: int) -> List[Dict[str, Any]]:
    """工单三：camelot 兜底抽取指定页（仅在 pdfplumber 无结果时调用）"""
    if camelot is None:
        return []
    tables_out: List[Dict[str, Any]] = []
    try:
        # lattice 适合有线框表，stream 适合无线框表，两次尝试
        for flavor in ("lattice", "stream"):
            try:
                tbls = camelot.read_pdf(pdf_path, pages=str(page_no), flavor=flavor)
            except Exception as e:  # pragma: no cover - ghostscript 缺失等
                logger.debug(f"[table_extractor] camelot {flavor} 失败 page={page_no}: {e}")
                continue
            for tb in tbls:
                df = tb.df
                if df.empty:
                    continue
                rows = [[_clean_cell(v) for v in row] for row in df.values.tolist()]
                rows = [r for r in rows if not _row_is_empty(r)]
                if not rows:
                    continue
                tables_out.append({
                    "doc_id": "",
                    "page": page_no,
                    "bbox": _bbox_to_list(tb._bbox) if hasattr(tb, "_bbox") else None,
                    "raw_rows": rows,
                    "col_count": max(len(r) for r in rows),
                    "row_count": len(rows),
                    "source": f"camelot-{flavor}",
                })
            if tables_out:
                break  # lattice 命中就不再 stream
    except Exception as e:  # pragma: no cover
        logger.warning(f"[table_extractor] camelot 整体异常 page={page_no}: {e}")
    return tables_out


# ================= 跨页候选检测 =================
def _mark_crosspage_candidates(tables: List[Dict[str, Any]]) -> None:
    """工单三：标记可能是跨页延续的表格（结构化阶段真正合并）

    判定：相邻页、列数相同、首页表尾行 vs 次页表首行 有共现列名。
    仅打 mark，不真合并，留给 table_structurer 处理。
    """
    for i in range(len(tables) - 1):
        cur, nxt = tables[i], tables[i + 1]
        if nxt["page"] != cur["page"] + 1:
            continue
        if cur["col_count"] != nxt["col_count"]:
            continue
        # 用首页末行与次页首行的字符串集合做相似度
        if not cur["raw_rows"] or not nxt["raw_rows"]:
            continue
        last = set(cur["raw_rows"][-1])
        first = set(nxt["raw_rows"][0])
        if len(last) == 0 or len(first) == 0:
            continue
        inter = last & first
        sim = len(inter) / max(len(last | first), 1)
        if sim >= 0.5:
            cur["crosspage_followed_by"] = nxt.get("local_idx", i + 1)
            nxt["crosspage_continues"] = cur.get("local_idx", i)


# ================= 主入口 =================
def extract_tables_from_pdf(
    pdf_path: str,
    doc_id: Optional[str] = None,
    use_camelot_fallback: bool = True,
) -> List[Dict[str, Any]]:
    """工单三：抽取单 PDF 全部表格，返回原始表列表

    Args:
        pdf_path: PDF 文件路径
        doc_id: 文档 ID（不传则按文件名生成）
        use_camelot_fallback: pdfplumber 无结果时是否启用 camelot 兜底
    Returns:
        List[raw_table]，每个含 doc_id/page/bbox/raw_rows/source
    """
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")
    if pdfplumber is None:
        raise RuntimeError("pdfplumber 未安装，无法抽取表格（工单三主路径不可用）")

    doc_id = doc_id or _safe_doc_id(pdf_path)
    logger.info(f"[table_extractor] 开始抽取 doc_id={doc_id} path={pdf_path}")

    all_tables: List[Dict[str, Any]] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_no = int(page.page_number)
            page_text = page.extract_text() or ""
            page_text_chars = len(page_text)

            # 1) 主路径 pdfplumber
            try:
                page_tables = _extract_with_pdfplumber(page)
            except Exception as e:  # pragma: no cover
                logger.warning(f"[table_extractor] pdfplumber 页异常 page={page_no}: {e}")
                page_tables = []

            # 2) 兜底 camelot（pdfplumber 无结果且页面有较多文字时尝试）
            if not page_tables and use_camelot_fallback and camelot is not None:
                logger.debug(f"[table_extractor] pdfplumber 无结果，camelot 兜底 page={page_no}")
                page_tables = _extract_with_camelot(pdf_path, page_no)

            for t in page_tables:
                t["doc_id"] = doc_id
                t["page_text_chars"] = page_text_chars
                all_tables.append(t)

    # 给每张表加 local_idx（用于跨页引用）
    for i, t in enumerate(all_tables):
        t["local_idx"] = i

    # 跨页候选标记
    _mark_crosspage_candidates(all_tables)

    logger.info(f"[table_extractor] doc_id={doc_id} 共抽取 {len(all_tables)} 张表")
    return all_tables


# ================= 全流程（抽取 → 结构化 → table-text） =================
def run_full_pipeline(
    pdf_path: str,
    out_path: str,
    doc_id: Optional[str] = None,
    company: Optional[str] = None,
    doc_type: str = "招股说明书",
    use_camelot_fallback: bool = True,
) -> Dict[str, Any]:
    """工单三：跑全流程并把结构化表格 + table-text 一起写出

    输出 JSON 结构：
      {
        "doc_id": ..., "company": ..., "pdf_path": ...,
        "raw_table_count": N, "structured_table_count": M,
        "sources": {pdfplumber: N1, ...},
        "tables": [structured_table...],
        "table_texts": [table_chunk...]
      }
    """
    # 延迟导入避免循环依赖
    from .table_structurer import structure_tables
    from .table_to_text import tables_to_texts

    doc_id = doc_id or _safe_doc_id(pdf_path)
    raw = extract_tables_from_pdf(
        pdf_path, doc_id=doc_id, use_camelot_fallback=use_camelot_fallback
    )
    structured = structure_tables(raw, doc_id=doc_id, company=company, doc_type=doc_type)
    texts = tables_to_texts(structured, doc_id=doc_id, company=company)

    sources = {s: sum(1 for t in raw if t["source"] == s) for s in {t["source"] for t in raw}}
    payload = {
        "doc_id": doc_id,
        "company": company,
        "doc_type": doc_type,
        "pdf_path": str(Path(pdf_path).resolve()),
        "raw_table_count": len(raw),
        "structured_table_count": len(structured),
        "sources": sources,
        "tables": structured,
        "table_texts": texts,
    }
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        f"[table_extractor] 全流程完成 doc_id={doc_id} raw={len(raw)} "
        f"structured={len(structured)} → {p}"
    )
    return payload


# ================= CLI =================
def _main() -> int:
    parser = argparse.ArgumentParser(description="工单三：PDF 表格抽取 CLI（默认跑全流程）")
    parser.add_argument("--pdf", required=True, help="PDF 文件路径")
    parser.add_argument("--out", required=True, help="输出 JSON 路径")
    parser.add_argument("--doc-id", default=None, help="文档 ID（不传则按文件名）")
    parser.add_argument("--company", default=None, help="公司名")
    parser.add_argument("--doc-type", default="招股说明书", help="文档类型")
    parser.add_argument("--no-camelot", action="store_true", help="禁用 camelot 兜底")
    parser.add_argument(
        "--raw-only", action="store_true",
        help="只跑抽取不结构化（调试用，默认 False 即跑全流程）",
    )
    args = parser.parse_args()

    if args.raw_only:
        tables = extract_tables_from_pdf(
            args.pdf, doc_id=args.doc_id,
            use_camelot_fallback=not args.no_camelot,
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "doc_id": tables[0]["doc_id"] if tables else _safe_doc_id(args.pdf),
            "pdf_path": str(Path(args.pdf).resolve()),
            "table_count": len(tables),
            "sources": {s: sum(1 for t in tables if t["source"] == s)
                        for s in {t["source"] for t in tables}},
            "tables": tables,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"[table_extractor] raw-only 写出 {len(tables)} 张表 → {out_path}")
        print(json.dumps({"doc_id": payload["doc_id"], "table_count": payload["table_count"],
                          "sources": payload["sources"], "out": str(out_path)},
                         ensure_ascii=False, indent=2))
        return 0

    payload = run_full_pipeline(
        args.pdf, args.out, doc_id=args.doc_id, company=args.company,
        doc_type=args.doc_type, use_camelot_fallback=not args.no_camelot,
    )
    print(json.dumps({
        "doc_id": payload["doc_id"],
        "company": payload["company"],
        "raw_table_count": payload["raw_table_count"],
        "structured_table_count": payload["structured_table_count"],
        "sources": payload["sources"],
        "out": args.out,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
