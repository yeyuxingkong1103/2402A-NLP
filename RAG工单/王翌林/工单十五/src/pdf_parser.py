# -*- coding: utf-8 -*-
"""
工单十五：PDF 图片提取（pypdfium2 高清渲染）
"""
import os
from typing import Dict, List
import pypdfium2 as pdfium


def extract_images_from_pdf(pdf_path: str, out_dir: str, zoom: float = 4.0) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    pdf = pdfium.PdfDocument(pdf_path)
    pages = []
    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(scale=zoom)
        img = bitmap.to_pil()
        out_path = os.path.join(out_dir, f"page_{i+1:02d}.png")
        img.save(out_path)
        pages.append({"page": i + 1, "images": [out_path], "width": img.size[0], "height": img.size[1]})
    pdf.close()
    return {"pages": pages, "page_count": len(pages)}


def get_best_image(page_info: Dict) -> str:
    return page_info["images"][0]
