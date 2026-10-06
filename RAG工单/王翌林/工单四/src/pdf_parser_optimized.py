# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
src/pdf_parser_optimized.py —— 工单二优化版 PDF 解析器

在工单一 src/pdf_parser.py（PyMuPDF 文字 + pdfplumber 表格）基础上增强：
  1. 版面分析：区分标题（heading）/正文（body）/表格（table）/页眉页脚（header/footer）
  2. 表格结构化提取：pdfplumber 主提取 + 表头识别 + Markdown 表示（camelot 可选补充）
  3. OCR 兜底：页面文字过少时调用 pytesseract（chi_sim+eng）或 PaddleOCR（可选），不可用时优雅降级
  4. 输出 JSON 新增字段：layout_blocks / tables_structured / headings

用法（项目根目录下）：
  python -m src.pdf_parser_optimized --pdf "../附件/招股说明书1.pdf" --out "data/optimized/招股说明书1_optimized.json"
"""
import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    logger.error("PyMuPDF 未安装")

try:
    import pdfplumber
except ImportError:
    pdfplumber = None
    logger.warning("pdfplumber 未安装（表格提取跳过）")

try:
    import pytesseract  # 工单二：OCR 兜底首选引擎（人工智能NLP-RAG-基于PDF文档的问答系统优化）
    _TESSERACT_BIN = shutil.which("tesseract")
except ImportError:
    pytesseract = None
    _TESSERACT_BIN = None

try:
    from paddleocr import PaddleOCR  # 工单二：OCR 备选引擎（可选，体积大默认不装）
except ImportError:
    PaddleOCR = None

try:
    import camelot  # 工单二：可选表格引擎（依赖 ghostscript，未安装则跳过）
except ImportError:
    camelot = None

# ---------- 常量（人工智能NLP-RAG-基于PDF文档的问答系统优化） ----------
EMPTY_PAGE_CHARS = 30          # 页面字符数低于该值触发 OCR 兜底
HEADER_FOOTER_RATIO = 0.5      # 顶部/底部重复行出现在 >= 50% 页面才判定为页眉页脚
HF_BAND_RATIO = 0.06           # 页面顶部/底部 6% 区域视为页眉页脚带
HEADING_L1 = 1.30              # 字号 >= 正文众数 × 1.30 判定一级标题
HEADING_L2 = 1.18              # 字号 >= 正文众数 × 1.18 判定二级标题
HEADING_L3 = 1.08              # 字号 >= 正文众数 × 1.08 且加粗判定三级标题


def _make_doc_id(pdf_path: str) -> str:
    """工单二：沿用工单一 doc_id 生成规则（路径+大小+mtime 哈希）"""
    p = Path(pdf_path)
    stat = p.stat()
    raw = f"{p.resolve()}|{stat.st_size}|{stat.st_mtime}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _normalize_hf_text(text: str) -> str:
    """页眉页脚归一化：去数字/空白后比较（页码每页不同，归一后才能聚成同一行）"""
    return re.sub(r"[\d\s]+", "", text).strip()


# ================= OCR 兜底（人工智能NLP-RAG-基于PDF文档的问答系统优化） =================
class OCREngine:
    """OCR 兜底引擎：优先 pytesseract（chi_sim+eng），可选 PaddleOCR；均不可用时优雅降级"""

    def __init__(self, prefer: str = "auto"):
        self.engine = None
        self._paddle = None
        if prefer in ("auto", "tesseract") and pytesseract is not None and _TESSERACT_BIN:
            self.engine = "tesseract"
            logger.info(f"✅ OCR 引擎: pytesseract ({_TESSERACT_BIN})")
        elif prefer in ("auto", "paddle") and PaddleOCR is not None:
            try:
                self._paddle = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
                self.engine = "paddle"
                logger.info("✅ OCR 引擎: PaddleOCR")
            except Exception as e:
                logger.warning(f"PaddleOCR 初始化失败: {e}")
        if self.engine is None:
            reason = []
            if pytesseract is None:
                reason.append("pytesseract 未安装")
            elif not _TESSERACT_BIN:
                reason.append("tesseract 系统二进制未安装（sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim）")
            logger.warning(f"⚠️ OCR 兜底不可用（{'; '.join(reason) or '引擎未配置'}），空页将被标记 ocr_skipped")

    @property
    def available(self) -> bool:
        return self.engine is not None

    def recognize(self, pix) -> str:
        """对 PyMuPDF pixmap（300 DPI 渲染）执行 OCR，返回文本"""
        if self.engine == "tesseract":
            img_bytes = pix.tobytes("png")
            return pytesseract.image_to_string(img_bytes, lang="chi_sim+eng")
        if self.engine == "paddle":
            import numpy as np
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            result = self._paddle.ocr(img, cls=True)
            lines = []
            for page_res in result or []:
                for item in page_res or []:
                    lines.append(item[1][0])
            return "\n".join(lines)
        return ""


# ================= 版面分析（人工智能NLP-RAG-基于PDF文档的问答系统优化） =================
def _collect_font_stats(doc) -> tuple:
    """第一遍扫描：统计全文档 span 字号分布与页面尺寸，得到正文字号众数"""
    size_counter: Counter = Counter()
    for page in doc:
        try:
            d = page.get_text("dict")
            for block in d.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        txt = span.get("text", "").strip()
                        if txt:
                            size_counter[round(span.get("size", 0), 1)] += len(txt)
        except Exception:
            continue
    body_size = size_counter.most_common(1)[0][0] if size_counter else 10.5
    return body_size, size_counter


def _detect_headers_footers(doc, total_pages: int) -> tuple:
    """第二遍扫描：统计页面顶部/底部 HF_BAND_RATIO 区域内重复出现的文本行 → 页眉/页脚集合"""
    top_counter: Counter = Counter()
    bottom_counter: Counter = Counter()
    for page in doc:
        try:
            h = page.rect.height
            d = page.get_text("dict")
            for block in d.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    txt = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
                    if not txt or len(txt) > 80:
                        continue
                    norm = _normalize_hf_text(txt)
                    if not norm:
                        continue
                    y0 = line.get("bbox", (0, 0, 0, 0))[1]
                    y1 = line.get("bbox", (0, 0, 0, 0))[3]
                    if y1 <= h * HF_BAND_RATIO:
                        top_counter[norm] += 1
                    elif y0 >= h * (1 - HF_BAND_RATIO):
                        bottom_counter[norm] += 1
        except Exception:
            continue
    threshold = max(3, int(total_pages * HEADER_FOOTER_RATIO))
    headers = {k for k, v in top_counter.items() if v >= threshold}
    footers = {k for k, v in bottom_counter.items() if v >= threshold}
    return headers, footers


def _classify_heading(span_size: float, is_bold: bool, body_size: float) -> Optional[int]:
    """按字号/加粗判定标题层级：1/2/3 或 None（非标题）"""
    if span_size >= body_size * HEADING_L1:
        return 1
    if span_size >= body_size * HEADING_L2:
        return 2
    if is_bold and span_size >= body_size * HEADING_L3:
        return 3
    return None


def _page_layout_blocks(page, body_size: float, headers: set, footers: set,
                        table_bboxes: List[tuple]) -> List[Dict[str, Any]]:
    """单页版面分析：输出 heading/body/table/header/footer 五类布局块"""
    h = page.rect.height
    blocks_out: List[Dict[str, Any]] = []
    try:
        d = page.get_text("dict")
    except Exception:
        return blocks_out

    def _in_table_band(bbox) -> bool:
        x0, y0, x1, y1 = bbox
        for tx0, ty0, tx1, ty1 in table_bboxes:
            if y0 >= ty0 - 5 and y1 <= ty1 + 5 and x0 >= tx0 - 10 and x1 <= tx1 + 10:
                return True
        return False

    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        bbox = block.get("bbox", (0, 0, 0, 0))
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(s.get("text", "") for s in spans).strip()
            if not text:
                continue
            line_bbox = line.get("bbox", bbox)
            norm = _normalize_hf_text(text)
            line_size = max(s.get("size", 0) for s in spans)
            is_bold = any(s.get("flags", 0) & 16 for s in spans)  # PyMuPDF flags bit4 = bold
            block_type = "body"
            level = None
            if norm and norm in headers:
                block_type = "header"
            elif norm and norm in footers:
                block_type = "footer"
            elif _in_table_band(line_bbox):
                block_type = "table"
            else:
                level = _classify_heading(line_size, is_bold, body_size)
                if level and len(text) <= 80:
                    block_type = "heading"
            item = {"page": page.number + 1, "type": block_type, "text": text,
                    "y0": round(line_bbox[1], 1), "y1": round(line_bbox[3], 1)}
            if block_type == "heading":
                item["level"] = level
                item["font_size"] = round(line_size, 1)
            blocks_out.append(item)
    return blocks_out


# ================= 表格结构化（人工智能NLP-RAG-基于PDF文档的问答系统优化） =================
def _extract_tables_structured(pdf_path: str, enable_camelot: bool = False) -> tuple:
    """pdfplumber 提取 + 表头识别 + Markdown 表示；camelot 可选补充 lattice 表格"""
    tables_structured: List[Dict[str, Any]] = []
    table_bboxes_by_page: Dict[int, List[tuple]] = {}
    if pdfplumber is None:
        return tables_structured, table_bboxes_by_page
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            try:
                raw_tables = page.extract_tables()
                found = page.find_tables()
                bboxes = [t.bbox for t in found]
                if bboxes:
                    table_bboxes_by_page[page_idx] = bboxes
                for ti, tbl in enumerate(raw_tables or []):
                    rows = []
                    for row in tbl:
                        if row is None:
                            continue
                        rc = [(c or "").replace("\n", " ").strip() for c in row]
                        if any(rc):
                            rows.append(rc)
                    if not rows:
                        continue
                    headers = rows[0]
                    body_rows = rows[1:] if len(rows) > 1 else []
                    md = _table_to_markdown(headers, body_rows)
                    tables_structured.append({
                        "page": page_idx, "table_index": ti + 1,
                        "n_rows": len(rows), "n_cols": max(len(r) for r in rows),
                        "headers": headers, "rows": body_rows, "markdown": md,
                        "engine": "pdfplumber",
                    })
            except Exception as e:
                logger.debug(f"第 {page_idx} 页表格提取跳过: {e}")
    # camelot 可选补充（lattice 模式对有边框表格更准）
    if enable_camelot and camelot is not None:
        try:
            cams = camelot.read_pdf(pdf_path, pages="all", flavor="lattice", suppress_stdout=True)
            for ci, cf in enumerate(cams):
                df = cf.df
                rows = [[str(c).replace("\n", " ").strip() for c in r] for r in df.values.tolist()]
                if not rows:
                    continue
                tables_structured.append({
                    "page": cf.page, "table_index": ci + 1,
                    "n_rows": len(rows), "n_cols": len(rows[0]) if rows else 0,
                    "headers": rows[0], "rows": rows[1:], "markdown": _table_to_markdown(rows[0], rows[1:]),
                    "engine": "camelot",
                })
        except Exception as e:
            logger.warning(f"camelot 补充提取失败（忽略）: {e}")
    return tables_structured, table_bboxes_by_page


def _table_to_markdown(headers: List[str], rows: List[List[str]]) -> str:
    """表格转 Markdown（供下游分块/检索直接引用）"""
    if not headers:
        return ""
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join((r + [""] * len(headers))[:len(headers)]) + " |" for r in rows]
    return "\n".join([header, sep] + body)


# ================= 主入口（人工智能NLP-RAG-基于PDF文档的问答系统优化） =================
def parse_pdf_optimized(pdf_path: str, extract_tables: bool = True, enable_ocr: bool = True,
                        enable_camelot: bool = False, max_pages: Optional[int] = None) -> Dict[str, Any]:
    """工单二优化版解析：文字 + 版面分析 + 结构化表格 + OCR 兜底"""
    result: Dict[str, Any] = {
        "doc_id": "", "filename": os.path.basename(pdf_path), "file_path": os.path.abspath(pdf_path),
        "file_size": 0, "parse_time": None, "total_pages": 0, "pages": [], "text": "",
        "tables": [], "layout_blocks": [], "tables_structured": [], "headings": [], "metadata": {}, "errors": [],
    }
    if not os.path.isfile(pdf_path):
        msg = f"文件不存在: {pdf_path}"
        logger.error(msg)
        result["errors"].append(msg)
        return result
    if fitz is None:
        result["errors"].append("PyMuPDF 未安装")
        return result

    result["file_size"] = os.path.getsize(pdf_path)
    result["doc_id"] = _make_doc_id(pdf_path)
    t0 = datetime.now()

    doc = fitz.open(pdf_path)
    if doc.is_encrypted and doc.needs_pass:
        doc.close()
        result["errors"].append("PDF 需要密码")
        return result
    result["metadata"] = dict(doc.metadata) if doc.metadata else {}
    total_pages = len(doc) if max_pages is None else min(len(doc), max_pages)
    result["total_pages"] = len(doc)
    logger.info(f"打开 PDF: {pdf_path}, 共 {result['total_pages']} 页, {result['file_size'] / 1024 / 1024:.1f} MB")

    # 1) 全文档字体统计 → 正文字号（版面分析基准）
    body_size, _sizes = _collect_font_stats(doc)
    logger.info(f"版面分析基准: 正文字号众数 = {body_size}")

    # 2) 页眉页脚检测（跨页重复行）
    headers, footers = _detect_headers_footers(doc, result["total_pages"])
    logger.info(f"页眉候选 {len(headers)} 条 / 页脚候选 {len(footers)} 条")

    # 3) 表格结构化提取 + bbox（供版面分析标记 table 块）
    tables_structured, table_bboxes_by_page = _extract_tables_structured(pdf_path, enable_camelot)
    result["tables_structured"] = tables_structured
    logger.info(f"结构化表格提取完成: {len(tables_structured)} 张")

    # 4) OCR 引擎初始化（懒加载）
    ocr = OCREngine() if enable_ocr else None

    # 5) 逐页：文字提取 + OCR 兜底 + 版面分析
    all_text_parts: List[str] = []
    pages_out: List[Dict[str, Any]] = []
    for page_num in range(total_pages):
        try:
            page = doc[page_num]
            text = page.get_text("text") or ""
            text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
            page_item: Dict[str, Any] = {"page": page_num + 1, "text": text, "char_count": len(text)}
            # —— OCR 兜底：文字过少的页渲染 300DPI 后识别（人工智能NLP-RAG-基于PDF文档的问答系统优化）
            if enable_ocr and ocr is not None and len(text) < EMPTY_PAGE_CHARS:
                if ocr.available:
                    try:
                        pix = page.get_pixmap(dpi=300)
                        ocr_text = ocr.recognize(pix).strip()
                        if ocr_text:
                            page_item["text"] = ocr_text
                            page_item["char_count"] = len(ocr_text)
                            page_item["ocr"] = True
                            logger.info(f"第 {page_num + 1} 页 OCR 兜底成功: {len(ocr_text)} chars")
                    except Exception as e:
                        page_item["ocr_error"] = str(e)
                        logger.warning(f"第 {page_num + 1} 页 OCR 失败: {e}")
                else:
                    page_item["ocr_skipped"] = True
            all_text_parts.append(page_item["text"])
            # —— 版面分析（人工智能NLP-RAG-基于PDF文档的问答系统优化）
            blocks = _page_layout_blocks(page, body_size, headers, footers,
                                         table_bboxes_by_page.get(page_num + 1, []))
            result["layout_blocks"].extend(blocks)
            if len(page_item["text"]) >= EMPTY_PAGE_CHARS:
                pages_out.append(page_item)
            else:
                # 文字与 OCR 均为空的页仅记录元信息，正文不拼接
                pages_out.append(page_item)
        except Exception as e:
            logger.warning(f"第 {page_num + 1} 页处理失败: {e}")
            result["errors"].append(f"第 {page_num + 1} 页: {e}")
            pages_out.append({"page": page_num + 1, "text": "", "char_count": 0, "error": str(e)})
    doc.close()

    result["pages"] = pages_out
    result["text"] = "\n\n".join(p["text"] for p in pages_out if p.get("text"))
    result["headings"] = [{"page": b["page"], "level": b["level"], "text": b["text"],
                           "font_size": b.get("font_size")}
                          for b in result["layout_blocks"] if b["type"] == "heading"]
    # 兼容工单一格式：tables 保留原始行列，下游 ingest 不受影响
    result["tables"] = [{"page": t["page"], "table_index": t["table_index"], "rows": t["n_rows"],
                         "cols": t["n_cols"], "data": [t["headers"]] + t["rows"]}
                        for t in tables_structured]
    result["parse_time"] = (datetime.now() - t0).total_seconds()
    return result


def parse_pdf_to_file(pdf_path: str, out_path: str, **kwargs) -> str:
    """解析并写 JSON 文件（人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
    data = parse_pdf_optimized(pdf_path, **kwargs)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON 已写入: {out_path}")
    return out_path


def print_summary(data: Dict[str, Any]) -> None:
    """输出解析统计：页数 / 表格数 / 标题数 / 前3个表格内容（工单二验证要求）"""
    n_pages = data["total_pages"]
    n_tables = len(data["tables_structured"])
    n_headings = len(data["headings"])
    ocr_pages = [p["page"] for p in data["pages"] if p.get("ocr")]
    print("=" * 60)
    print("工单二 PDF 解析统计（人工智能NLP-RAG-基于PDF文档的问答系统优化）")
    print(f"  解析页数   : {n_pages}")
    print(f"  表格数量   : {n_tables}")
    print(f"  标题数量   : {n_headings}")
    print(f"  布局块数量 : {len(data['layout_blocks'])}")
    print(f"  OCR 页面   : {len(ocr_pages)} 页 {ocr_pages[:10]}")
    print(f"  解析耗时   : {data['parse_time']:.1f}s")
    print("-" * 60)
    for t in data["tables_structured"][:3]:
        print(f"[表格] 第{t['page']}页 #{t['table_index']} ({t['n_rows']}行×{t['n_cols']}列, engine={t['engine']})")
        for line in t["markdown"].splitlines()[:6]:
            print("  " + line)
        if len(t["markdown"].splitlines()) > 6:
            print("  ...")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="工单二优化版 PDF 解析器（人工智能NLP-RAG-基于PDF文档的问答系统优化）")
    parser.add_argument("--pdf", required=True, help="PDF 文件路径")
    parser.add_argument("--out", default=None, help="输出 JSON 路径")
    parser.add_argument("--no-tables", action="store_true", help="跳过表格提取")
    parser.add_argument("--no-ocr", action="store_true", help="关闭 OCR 兜底")
    parser.add_argument("--camelot", action="store_true", help="启用 camelot lattice 补充提取（可选）")
    parser.add_argument("--max-pages", type=int, default=None, help="仅解析前 N 页（调试用）")
    args = parser.parse_args()
    out_path = args.out or f"data/optimized/{Path(args.pdf).stem}_optimized.json"
    data = parse_pdf_optimized(args.pdf, extract_tables=not args.no_tables,
                               enable_ocr=not args.no_ocr, enable_camelot=args.camelot,
                               max_pages=args.max_pages)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON 已写入: {out_path}")
    print_summary(data)
