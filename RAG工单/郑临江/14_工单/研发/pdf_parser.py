# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失工单
DeepDoc 风格 PDF 解析器（修复版）：
  - 用 PyMuPDF 提取文本、布局、表格、图片；
  - 对“低质量图片型 PDF”按页判定扫描页，扫描页走 OCR 兜底，
    避免图文信息丢失；
  - 依据 parser_id（paper/table/one/knowledge_graph）采用不同分块策略。
"""
import os
import zipfile
import tempfile

import pymupdf

import config


def load_single_pdf(pdf_name, doc_dir=None):
    """从已解压目录读取 PDF；若目录不存在则仅从 zip 中解压该文件到临时目录。"""
    doc_dir = doc_dir or config.DOC_DIR
    path = os.path.join(doc_dir, pdf_name)
    if os.path.exists(path):
        return path
    if os.path.exists(config.DATA_ZIP):
        z = zipfile.ZipFile(config.DATA_ZIP)
        target = f"original_problems/documents/{pdf_name}"
        tmp = os.path.join(tempfile.gettempdir(), pdf_name)
        with z.open(target) as src, open(tmp, "wb") as dst:
            dst.write(src.read())
        z.close()
        return tmp
    raise FileNotFoundError(pdf_name)


def _page_char_count(page):
    return len(page.get_text().strip())


def classify_page(page, threshold=None):
    """单页分类：文字页 / 扫描页。"""
    threshold = threshold or config.PAGE_CHAR_THRESHOLD
    return "text" if _page_char_count(page) >= threshold else "scanned"


def _ocr_page(page):
    """扫描页 OCR 兜底（优先 paddleocr，其次 pytesseract）。"""
    pix = page.get_pixmap(dpi=200)
    img = pix.tobytes("png")
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        result = ocr.ocr(img, cls=True)
        return "\n".join(line[1][0] for line in result[0]) if result and result[0] else ""
    except Exception:
        pass
    try:
        import pytesseract
        from PIL import Image
        import io
        return pytesseract.image_to_string(Image.open(io.BytesIO(img)), lang="chi_sim")
    except Exception:
        return ""


def extract_pdf(pdf_name, parser_id="one", doc_dir=None):
    """解析 PDF 并返回结构化内容（修复图文信息丢失）。"""
    path = load_single_pdf(pdf_name, doc_dir)
    doc = pymupdf.open(path)
    pages = []
    for pno, page in enumerate(doc, 1):
        ctype = classify_page(page)
        text = page.get_text()
        if ctype == "scanned" or not text.strip():
            text = _ocr_page(page)  # 低质量/扫描页 OCR 兜底
        tables = []
        try:
            for t in page.find_tables():
                tables.append(t.extract())
        except Exception:
            pass
        images = [{"page": pno, "xref": x[0]} for x in page.get_images(full=True)]
        pages.append({
            "page": pno, "type": ctype, "text": text,
            "tables": tables, "images": images,
        })
    doc.close()
    return chunk_pages(pages, parser_id)


def chunk_pages(pages, parser_id="one"):
    """按解析方法分块。"""
    chunks = []
    if parser_id == "one":
        # 整篇作为一个 chunk
        full = "\n".join(p["text"] for p in pages)
        chunks.append({"page": None, "text": full, "kind": "one"})
    elif parser_id == "table":
        # 每个表格一个 chunk
        for p in pages:
            for t in p["tables"]:
                txt = "\n".join(" | ".join(map(str, row)) for row in t)
                chunks.append({"page": p["page"], "text": txt, "kind": "table"})
    elif parser_id == "paper":
        # 论文式：按页切块
        for p in pages:
            chunks.append({"page": p["page"], "text": p["text"], "kind": "paper"})
    elif parser_id == "knowledge_graph":
        # 图谱式：按段落切块
        for p in pages:
            for para in p["text"].split("\n\n"):
                if para.strip():
                    chunks.append({"page": p["page"], "text": para.strip(), "kind": "kg"})
    else:
        # 默认按页
        for p in pages:
            chunks.append({"page": p["page"], "text": p["text"], "kind": "page"})
    return [c for c in chunks if c["text"].strip()]
