# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-基于PDF文档的问答系统
src/pdf_parser.py
    —— PyMuPDF 提取文字 + pdfplumber 提取表格
"""
import argparse, hashlib, json, os, sys, traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger

try:
    import fitz
except ImportError:
    fitz = None
    logger.error("PyMuPDF 未安装")
try:
    import pdfplumber
except ImportError:
    pdfplumber = None
    logger.warning("pdfplumber 未安装（表格提取跳过）")

def _make_doc_id(pdf_path: str) -> str:
    p = Path(pdf_path); stat = p.stat()
    raw = f"{p.resolve()}|{stat.st_size}|{stat.st_mtime}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

def parse_pdf(pdf_path: str, extract_tables: bool = True) -> Dict[str, Any]:
    result = {"doc_id":"","filename":os.path.basename(pdf_path),"file_path":os.path.abspath(pdf_path),
              "file_size":0,"parse_time":None,"total_pages":0,"pages":[],"text":"","tables":[],"metadata":{},"errors":[]}
    if not os.path.isfile(pdf_path):
        msg = f"文件不存在: {pdf_path}"; logger.error(msg); result["errors"].append(msg); return result
    if fitz is None:
        msg = "PyMuPDF 未安装"; logger.error(msg); result["errors"].append(msg); return result
    result["file_size"] = os.path.getsize(pdf_path)
    result["doc_id"] = _make_doc_id(pdf_path)
    t0 = datetime.now()
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        if "password" in str(e).lower():
            result["errors"].append(f"PDF 已加密: {e}")
        else:
            result["errors"].append(f"打开 PDF 失败: {e}")
        return result
    if doc.is_encrypted and doc.needs_pass:
        doc.close(); result["errors"].append("PDF 需要密码"); return result
    result["metadata"] = dict(doc.metadata) if doc.metadata else {}
    result["total_pages"] = len(doc)
    logger.info(f"打开 PDF: {pdf_path}, 共 {result['total_pages']} 页, {result['file_size']/1024/1024:.1f} MB")
    all_text_parts, pages_out = [], []
    for page_num in range(result["total_pages"]):
        try:
            page = doc[page_num]
            text = page.get_text("text") or ""
            text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
            all_text_parts.append(text)
            pages_out.append({"page": page_num+1, "text": text, "char_count": len(text)})
        except Exception as e:
            logger.warning(f"第 {page_num+1} 页文字提取失败: {e}")
            result["errors"].append(f"第 {page_num+1} 页: {e}")
            pages_out.append({"page": page_num+1, "text": "", "char_count": 0, "error": str(e)})
    doc.close()
    result["pages"] = pages_out
    result["text"] = "\n\n".join(all_text_parts)
    logger.info(f"文字提取完成: {len(result['text']):,} chars")
    if extract_tables and pdfplumber is not None:
        try:
            tables_out = []
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        tbls = page.extract_tables()
                        if not tbls: continue
                        for ti, tbl in enumerate(tbls):
                            cleaned = []
                            for row in tbl:
                                if row is None: continue
                                row_clean = [(c or "").strip() for c in row]
                                if any(row_clean): cleaned.append(row_clean)
                            if cleaned:
                                tables_out.append({"page": page_num+1, "table_index": ti+1,
                                                   "rows": len(cleaned), "cols": max(len(r) for r in cleaned), "data": cleaned})
                    except Exception as e:
                        logger.debug(f"第 {page_num+1} 页表格跳过: {e}")
            result["tables"] = tables_out
            logger.info(f"表格提取完成: {len(tables_out)} 张")
        except Exception as e:
            result["errors"].append(f"表格提取整体失败: {e}")
    elif extract_tables:
        logger.warning("pdfplumber 不可用，跳过表格提取")
    result["parse_time"] = (datetime.now() - t0).total_seconds()
    return result

def parse_pdf_to_file(pdf_path: str, out_path: str, extract_tables: bool = True) -> str:
    data = parse_pdf(pdf_path, extract_tables=extract_tables)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON 已写入: {out_path}"); return out_path

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PDF 解析器（工单：人工智能NLP-RAG-基于PDF文档的问答系统）")
    p.add_argument("--pdf", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--no-tables", action="store_true")
    args = p.parse_args()
    out_path = args.out or f"data/parsed/{Path(args.pdf).stem}.json"
    parse_pdf_to_file(args.pdf, out_path, extract_tables=not args.no_tables)
