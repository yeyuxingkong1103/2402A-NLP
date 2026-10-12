# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
PDF 解析模块：提取正文文本、表格，并提取页面内嵌图片。
"""
import os
import pymupdf


def table_to_markdown(rows):
    if not rows:
        return ""
    rows = [[(c or "").strip().replace("\n", " ") for c in r] for r in rows]
    ncol = max(len(r) for r in rows)
    lines = []
    for i, r in enumerate(rows):
        r = r + [""] * (ncol - len(r))
        lines.append("| " + " | ".join(r) + " |")
        if i == 0:
            lines.append("|" + "---|" * ncol)
    return "\n".join(lines)


def extract_images(pdf_path: str, out_dir: str):
    """
    提取 PDF 内嵌图片，返回图片信息列表：
    [{page, xref, path, rect}]，其中 rect 用于定位图片在页面中的位置。
    """
    os.makedirs(out_dir, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    infos = []
    for pno, page in enumerate(doc):
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                pix = pymupdf.Pixmap(doc, xref)
                if pix.n - pix.alpha >= 4:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                fname = f"p{pno + 1}_x{xref}.png"
                fpath = os.path.join(out_dir, fname)
                pix.save(fpath)
                rects = page.get_image_rects(xref)
                infos.append({"page": pno + 1, "xref": xref,
                              "path": fpath, "rects": [list(r) for r in rects]})
                pix = None
            except Exception:
                continue
    doc.close()
    return infos


def extract_pdf_content(pdf_path: str):
    """返回 (text_blocks, table_blocks)。"""
    doc = pymupdf.open(pdf_path)
    text_blocks, table_blocks = [], []
    for pno, page in enumerate(doc):
        txt = page.get_text().strip()
        if txt:
            text_blocks.append(f"【第{pno + 1}页】\n{txt}")
        try:
            tabs = page.find_tables()
            if tabs.tables:
                for tb in tabs.tables:
                    md = table_to_markdown(tb.extract())
                    if md:
                        table_blocks.append(f"【第{pno + 1}页·表格】\n{md}")
        except Exception:
            pass
    doc.close()
    return text_blocks, table_blocks


def chunk_text(text, chunk_size, overlap):
    text = text.replace("\x00", " ").strip()
    chunks = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = end - overlap
    return [c for c in chunks if c]
