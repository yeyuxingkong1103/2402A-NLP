# -*- coding: utf-8 -*-
# 工单14：生成模拟低质量工业扫描 PDF
"""
用专利 CN100342976C 的文本内容渲染为图片型 PDF，模拟 IMDR 数据集中
低分辨率、格式复杂的图片型工业文档（文字嵌入图片中，不可直接复制）。
"""
import textwrap
from pathlib import Path

import fitz  # pymupdf

OUT = Path(__file__).resolve().parent
TXT = OUT / "patent_text.txt"
PATENT_TEXT = TXT.read_text(encoding="utf-8")


def make_image_pdf():
    """将文本渲染为图片型 PDF（新建空白页只插入渲染图，确保零可提取文本）。"""
    doc = fitz.open()
    paragraphs = PATENT_TEXT.strip().split("\n\n")
    page_w, page_h = 595, 842
    font_size = 11
    chars_per_line = 42
    max_lines = int((page_h - 80) / (font_size * 1.8))

    tmp = fitz.open()  # 临时文档：渲染文本用
    for para in paragraphs:
        lines = textwrap.wrap(para, width=chars_per_line)
        for i in range(0, len(lines), max_lines):
            chunk = lines[i:i + max_lines]
            tpage = tmp.new_page(width=page_w, height=page_h)
            y = 50
            for line in chunk:
                tpage.insert_text((40, y), line, fontsize=font_size, fontname="china-s")
                y += font_size * 1.8
            # 渲染为 JPEG 图片
            pix = tpage.get_pixmap(dpi=90)
            img_bytes = pix.tobytes("jpeg", jpg_quality=60)
            # 在正式文档中新建空白页，只插入图片（无文本层）
            page = doc.new_page(width=page_w, height=page_h)
            page.insert_image(fitz.Rect(0, 0, page_w, page_h), stream=img_bytes)
    tmp.close()

    out = OUT / "CN100342976C.pdf"
    doc.save(out, garbage=4, deflate=True)
    doc.close()
    print(f"已生成模拟图片型 PDF: {out.name} ({out.stat().st_size // 1024} KB)")
    return out


if __name__ == "__main__":
    make_image_pdf()
