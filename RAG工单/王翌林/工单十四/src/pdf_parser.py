# -*- coding: utf-8 -*-
"""
工单十四：PDF 图片提取模块
针对图片型PDF（无文本层），提取每页内嵌图片并高分辨率渲染，供 OCR/VLM 使用。
"""
import os
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List
from loguru import logger

try:
    import pymupdf
except ImportError:
    pymupdf = None
    logger.error("PyMuPDF 未安装")


def _make_doc_id(pdf_path: str) -> str:
    p = Path(pdf_path)
    stat = p.stat()
    raw = f"{p.resolve()}|{stat.st_size}|{stat.st_mtime}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def extract_images_from_pdf(pdf_path: str, out_dir: str, zoom: float = 4.0) -> Dict[str, Any]:
    """
    提取 PDF 每页图片：
    1. 优先提取内嵌原始图片（最高分辨率）
    2. 同时高分辨率渲染页面（zoom=4x）作为兜底
    返回页面清单
    """
    result = {"doc_id": "", "filename": os.path.basename(pdf_path), "file_path": os.path.abspath(pdf_path),
              "total_pages": 0, "pages": [], "errors": []}
    if not os.path.isfile(pdf_path):
        result["errors"].append(f"文件不存在: {pdf_path}")
        return result
    if pymupdf is None:
        result["errors"].append("PyMuPDF 未安装")
        return result

    result["doc_id"] = _make_doc_id(pdf_path)
    raw_dir = os.path.join(out_dir, "raw")
    render_dir = os.path.join(out_dir, "rendered")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    result["total_pages"] = len(doc)
    logger.info(f"打开 PDF: {pdf_path}, 共 {result['total_pages']} 页")

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_info = {"page": page_num + 1, "raw_image": None, "rendered_image": None}
        try:
            # 提取内嵌原始图片
            images = page.get_images(full=True)
            if images:
                xref = images[0][0]
                base_image = doc.extract_image(xref)
                ext = base_image.get("ext", "png")
                raw_path = os.path.join(raw_dir, f"page_{page_num+1:02d}.{ext}")
                with open(raw_path, "wb") as f:
                    f.write(base_image["image"])
                page_info["raw_image"] = raw_path
                page_info["raw_size"] = [base_image["width"], base_image["height"]]
            # 高分辨率渲染（兜底 + 统一尺寸）
            mat = pymupdf.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            render_path = os.path.join(render_dir, f"page_{page_num+1:02d}.png")
            pix.save(render_path)
            page_info["rendered_image"] = render_path
            page_info["rendered_size"] = [pix.width, pix.height]
        except Exception as e:
            logger.warning(f"第 {page_num+1} 页图片提取失败: {e}")
            result["errors"].append(f"第 {page_num+1} 页: {e}")
        result["pages"].append(page_info)

    doc.close()
    logger.info(f"图片提取完成: {len(result['pages'])} 页")
    return result


def get_best_image(page_info: Dict) -> str:
    """获取某页最佳图片（优先原始内嵌，其次渲染图）"""
    return page_info.get("raw_image") or page_info.get("rendered_image")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="工单十四 PDF 图片提取")
    p.add_argument("--pdf", required=True)
    p.add_argument("--out", default="./data/pages")
    p.add_argument("--zoom", type=float, default=4.0)
    args = p.parse_args()
    res = extract_images_from_pdf(args.pdf, args.out, zoom=args.zoom)
    print(json.dumps({"doc_id": res["doc_id"], "total_pages": res["total_pages"]}, ensure_ascii=False, indent=2))
