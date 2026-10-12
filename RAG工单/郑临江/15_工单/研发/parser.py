# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-优化技术图纸与文本的跨模态检索流程工单
解析器：提取文本 + 定位技术图纸（图3 等），生成图文块（图像描述 + 关联文本）。
"""
import os
import re
import tempfile
import zipfile

import pymupdf

import config


def load_single_pdf(pdf_name, doc_dir=None):
    doc_dir = doc_dir or config.DOC_DIR
    path = os.path.join(doc_dir, pdf_name)
    if os.path.exists(path):
        return path
    if os.path.exists(config.DATA_ZIP):
        z = zipfile.ZipFile(config.DATA_ZIP)
        tmp = os.path.join(tempfile.gettempdir(), pdf_name)
        with z.open(f"original_problems/documents/{pdf_name}") as src, open(tmp, "wb") as dst:
            dst.write(src.read())
        z.close()
        return tmp
    raise FileNotFoundError(pdf_name)


FIGURE_RE = re.compile(r"图\s*(\d+)")


def extract_pdf(pdf_name, doc_dir=None):
    """返回页面列表，每页含 text + figures（图纸编号） + images。"""
    path = load_single_pdf(pdf_name, doc_dir)
    doc = pymupdf.open(path)
    pages = []
    for pno, page in enumerate(doc, 1):
        text = page.get_text()
        figures = [int(m) for m in FIGURE_RE.findall(text)]
        images = [{"page": pno, "xref": x[0]} for x in page.get_images(full=True)]
        pages.append({"page": pno, "text": text, "figures": figures, "images": images})
    doc.close()
    return pages


def build_blocks(pages):
    """构造图文块：每个块含文本、图纸编号、图像描述。"""
    blocks = []
    for p in pages:
        # 文本块
        blocks.append({"page": p["page"], "text": p["text"], "figures": p["figures"],
                       "image_desc": "", "is_image": False})
        # 图文块：为每张图纸生成描述（离线：用页内文本上下文概括）
        for fno in p["figures"]:
            desc = f"第{p['page']}页 图{fno} 技术图纸，包含编号部件及其位置关系"
            blocks.append({"page": p["page"], "text": p["text"], "figures": [fno],
                           "image_desc": desc, "is_image": True})
    return blocks


def find_figure_block(blocks, figure_no):
    """按图纸编号定位图文块（供查询理解阶段强化检索）。"""
    for b in blocks:
        if b["is_image"] and figure_no in b["figures"]:
            return b
    return None
