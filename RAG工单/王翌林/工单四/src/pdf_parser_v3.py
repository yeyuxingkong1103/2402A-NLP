# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
src/pdf_parser_v3.py —— 工单三多文档 PDF 解析器

职责（见 docs/03_多文档知识库方案.md §四）：
  1. 文本抽取：PyMuPDF（fitz）按页提取，工单一逻辑复用
  2. 表格抽取：调用 Step 2 的 src.table_parser 全流程（结构化 + table-text）
  3. doc_id 生成：hash(文件路径 + 文件名) —— 与工单二 doc_id 规则不同，工单三统一
  4. 文本 chunk / 表格 chunk 分开记录，便于后续分别入库 Milvus

输出 JSON（data/parsed_v3/<doc_name>_text.json）：
  {
    "doc_id": "...", "doc_name": "招股说明书1", "company": "...", "doc_type": "招股说明书",
    "filename": "...", "file_path": "...", "file_hash": "...", "file_size_mb": ...,
    "total_pages": N, "total_chars": M,
    "pages": [{"page":1,"text":"...","char_count":N}, ...],
    "text": "全量文本",
    "text_chunks": [{"chunk_id","doc_id","page","text","char_count"}, ...],
    "table_chunks": [{"table_chunk_id","doc_id","page_range","table_text",...}, ...],
    "table_count": K,
    "errors": []
  }
"""
import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None

from src.table_parser import extract_tables_from_pdf, structure_tables, tables_to_texts

# ================= 常量（人工智能NLP-RAG-PDF文档的表格解析及检索优化） =================
TEXT_CHUNK_SIZE = 800        # 文本 chunk 字符数（与工单二保持一致）
TEXT_CHUNK_OVERLAP = 120     # 文本 chunk 重叠字符数
DEFAULT_DOC_TYPE = "招股说明书"


def make_doc_id(pdf_path: str) -> str:
    """工单三：doc_id = sha1(file_path + file_name)[:16]

    与工单二的 (path+size+mtime) 规则不同：工单三强调路径+文件名稳定哈希，
    便于跨重建保持同一 doc_id（同一文件重复入库仍生成相同 doc_id）。
    """
    abs_path = str(Path(pdf_path).resolve())
    filename = Path(pdf_path).name
    raw = f"{abs_path}|{filename}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def file_hash(pdf_path: str) -> str:
    """工单三：文件内容 SHA256，用于去重"""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ================= 文本抽取（复用工单一逻辑） =================
def _extract_text(pdf_path: str) -> Dict[str, Any]:
    """工单三：PyMuPDF 按页提取文本"""
    if fitz is None:
        raise RuntimeError("PyMuPDF 未安装，无法提取文本")
    pages: List[Dict[str, Any]] = []
    all_text_parts: List[str] = []

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    for page_num in range(total_pages):
        try:
            page = doc[page_num]
            text = page.get_text("text") or ""
            text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
            all_text_parts.append(text)
            pages.append({"page": page_num + 1, "text": text, "char_count": len(text)})
        except Exception as e:  # pragma: no cover
            logger.warning(f"页 {page_num+1} 文字提取失败: {e}")
            pages.append({"page": page_num + 1, "text": "", "char_count": 0, "error": str(e)})
    doc.close()
    return {
        "total_pages": total_pages,
        "pages": pages,
        "text": "\n\n".join(all_text_parts),
        "total_chars": sum(len(t) for t in all_text_parts),
    }


# ================= 文本分块（按页 + 滑窗） =================
def _split_text_to_chunks(
    pages: List[Dict[str, Any]],
    doc_id: str,
    chunk_size: int = TEXT_CHUNK_SIZE,
    overlap: int = TEXT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """工单三：把每页文本按 chunk_size + overlap 滑窗切块

    每块记录所属页号，便于检索时回填引用。
    """
    chunks: List[Dict[str, Any]] = []
    cid = 0
    for pg in pages:
        text = pg["text"]
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                cid += 1
                chunks.append({
                    "chunk_id": f"tc_{doc_id}_{cid:05d}",
                    "doc_id": doc_id,
                    "page": pg["page"],
                    "text": piece,
                    "char_count": len(piece),
                })
            if end >= len(text):
                break
            start = end - overlap
    return chunks


# ================= 主入口：单 PDF 解析 =================
def parse_pdf_v3(
    pdf_path: str,
    doc_name: Optional[str] = None,
    company: Optional[str] = None,
    doc_type: str = DEFAULT_DOC_TYPE,
    use_camelot_fallback: bool = False,
) -> Dict[str, Any]:
    """工单三：解析单 PDF，返回文本 + 表格 + chunk 结构

    Args:
        pdf_path: PDF 路径
        doc_name: 文档名（不传则按文件名 stem）
        company: 公司名（可选，建议传入以提升检索）
        doc_type: 文档类型，默认"招股说明书"
        use_camelot_fallback: 是否启用 camelot 兜底
    """
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

    doc_id = make_doc_id(pdf_path)
    doc_name = doc_name or Path(pdf_path).stem
    t0 = datetime.now()

    logger.info(f"[pdf_parser_v3] doc_id={doc_id} doc_name={doc_name} path={pdf_path}")
    result: Dict[str, Any] = {
        "doc_id": doc_id,
        "doc_name": doc_name,
        "company": company,
        "doc_type": doc_type,
        "filename": Path(pdf_path).name,
        "file_path": str(Path(pdf_path).resolve()),
        "file_hash": file_hash(pdf_path),
        "file_size_mb": round(os.path.getsize(pdf_path) / 1024 / 1024, 2),
        "parse_started_at": t0.isoformat(),
        "errors": [],
    }

    # 1) 文本抽取
    try:
        text_data = _extract_text(pdf_path)
        result.update({
            "total_pages": text_data["total_pages"],
            "pages": text_data["pages"],
            "text": text_data["text"],
            "total_chars": text_data["total_chars"],
        })
    except Exception as e:
        result["errors"].append(f"文本抽取失败: {e}")
        logger.exception(f"[pdf_parser_v3] 文本抽取失败: {e}")
        result["total_pages"] = 0
        result["pages"] = []
        result["text"] = ""
        result["total_chars"] = 0

    # 2) 文本 chunk
    try:
        text_chunks = _split_text_to_chunks(result["pages"], doc_id=doc_id)
        result["text_chunks"] = text_chunks
        result["text_chunk_count"] = len(text_chunks)
    except Exception as e:
        result["errors"].append(f"文本分块失败: {e}")
        result["text_chunks"] = []
        result["text_chunk_count"] = 0

    # 3) 表格抽取（Step 2 全流程）
    try:
        raw_tables = extract_tables_from_pdf(
            pdf_path, doc_id=doc_name, use_camelot_fallback=use_camelot_fallback,
        )
        structured = structure_tables(raw_tables, doc_id=doc_name, company=company, doc_type=doc_type)
        table_chunks = tables_to_texts(structured, doc_id=doc_name, company=company)
        # 给 table_chunk 加上工单三 doc_id（与 text chunk 共享 doc_id 命名空间）
        for tc in table_chunks:
            tc["doc_id"] = doc_name
        result["tables"] = structured
        result["table_chunks"] = table_chunks
        result["table_count"] = len(structured)
        result["raw_table_count"] = len(raw_tables)
    except Exception as e:
        result["errors"].append(f"表格抽取失败: {e}")
        logger.exception(f"[pdf_parser_v3] 表格抽取失败: {e}")
        result["tables"] = []
        result["table_chunks"] = []
        result["table_count"] = 0
        result["raw_table_count"] = 0

    result["parse_finished_at"] = datetime.now().isoformat()
    result["parse_seconds"] = round((datetime.now() - t0).total_seconds(), 2)
    logger.info(
        f"[pdf_parser_v3] doc_id={doc_id} 完成：pages={result['total_pages']} "
        f"text_chunks={result['text_chunk_count']} tables={result['table_count']} "
        f"耗时={result['parse_seconds']}s"
    )
    return result


# ================= 写出 =================
def write_text_json(parsed: Dict[str, Any], out_path: str) -> str:
    """工单三：写出文本+chunk JSON 到 data/parsed_v3/<doc_name>_text.json"""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # 不重复写表格全量（已在 data/tables/<doc_name>_tables.json）；
    # 这里保留 table_chunks 摘要供下游统一入库
    payload = {k: v for k, v in parsed.items() if k != "tables"}
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[pdf_parser_v3] 写出文本 JSON → {p}")
    return str(p)


def write_tables_json(parsed: Dict[str, Any], out_path: str) -> str:
    """工单三：写出结构化表格 + table-text JSON 到 data/tables/<doc_name>_tables.json"""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "doc_id": parsed["doc_id"],
        "doc_name": parsed["doc_name"],
        "company": parsed["company"],
        "doc_type": parsed["doc_type"],
        "pdf_path": parsed["file_path"],
        "raw_table_count": parsed.get("raw_table_count", 0),
        "structured_table_count": parsed.get("table_count", 0),
        "sources": {"pdfplumber": parsed.get("raw_table_count", 0)} if parsed.get("raw_table_count") else {},
        "tables": parsed.get("tables", []),
        "table_texts": parsed.get("table_chunks", []),
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[pdf_parser_v3] 写出表格 JSON → {p}")
    return str(p)


# ================= CLI =================
def _main() -> int:
    parser = argparse.ArgumentParser(description="工单三：单 PDF 解析 CLI")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--out-dir", default="data/parsed_v3",
                        help="文本 JSON 输出目录")
    parser.add_argument("--tables-dir", default="data/tables",
                        help="表格 JSON 输出目录")
    parser.add_argument("--doc-name", default=None)
    parser.add_argument("--company", default=None)
    parser.add_argument("--doc-type", default=DEFAULT_DOC_TYPE)
    parser.add_argument("--no-camelot", action="store_true", default=True)
    args = parser.parse_args()

    parsed = parse_pdf_v3(
        args.pdf, doc_name=args.doc_name, company=args.company,
        doc_type=args.doc_type, use_camelot_fallback=not args.no_camelot,
    )
    doc_name = parsed["doc_name"]
    write_text_json(parsed, f"{args.out_dir}/{doc_name}_text.json")
    write_tables_json(parsed, f"{args.tables_dir}/{doc_name}_tables.json")
    print(json.dumps({
        "doc_id": parsed["doc_id"],
        "doc_name": doc_name,
        "company": parsed["company"],
        "total_pages": parsed["total_pages"],
        "total_chars": parsed["total_chars"],
        "text_chunk_count": parsed["text_chunk_count"],
        "table_count": parsed["table_count"],
        "raw_table_count": parsed["raw_table_count"],
        "parse_seconds": parsed["parse_seconds"],
        "errors": parsed["errors"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
