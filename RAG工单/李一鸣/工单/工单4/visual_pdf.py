"""人工智能NLP-RAG-图像内容解析及检索优化。

Render figure-heavy PDF pages, collect OCR/captions, and attach image records to
the same page metadata used by text retrieval.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Protocol


class VisionClient(Protocol):
    def describe(self, image_data_url: str, prompt: str) -> str: ...


def extract_visual_pages(pdf_path: str, output_dir: str = "visual_pages", dpi: int = 180) -> list[dict]:
    import pymupdf

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, 1):
            has_image = bool(page.get_images(full=True)) or bool(page.get_drawings())
            if not has_image:
                continue
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72), alpha=False)
            image_path = output / f"page-{page_number:04d}.png"
            pixmap.save(image_path)
            rows.append({"source": pdf_path, "page": page_number, "image_path": str(image_path), "text": page.get_text("text", sort=True)})
    return rows


def analyze_visual_pages(rows: list[dict], vision: VisionClient) -> list[dict]:
    for row in rows:
        payload = base64.b64encode(Path(row["image_path"]).read_bytes()).decode("ascii")
        data_url = f"data:image/png;base64,{payload}"
        prompt = "逐项识别图中的文字、图表关系、组织结构层级和可读数值；不确定处标注不确定。"
        row["visual_description"] = vision.describe(data_url, prompt)
        row["indexed_text"] = f"[Page {row['page']}]\n{row['text']}\n[Visual description]\n{row['visual_description']}"
    return rows


def save_records(rows: list[dict], path: str) -> None:
    Path(path).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf")
    parser.add_argument("--output", default="visual_pages")
    args = parser.parse_args()
    records = extract_visual_pages(args.pdf, args.output)
    save_records(records, str(Path(args.output) / "visual_index.json"))
