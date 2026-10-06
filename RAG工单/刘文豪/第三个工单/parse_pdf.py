# -*- coding: utf-8 -*-
"""
工单编号：人工智能 NLP-RAG-PDF 文档的表格解析及检索优化
步骤1：文档解析 —— 解析《招股说明书1》（武汉兴图新科）与《招股说明书2》（武汉力源信息）
运行：python parse_pdf.py   （PDF 路径见 pdf_paths.txt，每行一个）
"""
import json
import re
from pathlib import Path

import fitz  # PyMuPDF

DOC_NAMES = ["zhaogu1", "zhaogu2"]  # 与 pdf_paths.txt 行顺序对应
OUT_FILE = Path("data") / "pages_all.json"

HEADER_RES = [
    re.compile(r"武汉兴图新科电子股份有限公司\s*招股说明书（申报稿）"),
    re.compile(r"武汉力源信息技术股份有限公司\s*招股意向书"),
]


def clean(text: str) -> str:
    """去页眉、独立页码行、多余空白"""
    for h in HEADER_RES:
        text = h.sub("", text)
    text = re.sub(r"\n\s*\d{1,4}\s*(?=\n|$)", "\n", text)   # 独立成行的页码
    text = re.sub(r"[ \t\u3000]+", " ", text)
    return text.strip()


def main():
    paths = [p.strip() for p in Path("pdf_paths.txt").read_text(encoding="utf-8").splitlines() if p.strip()]
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_pages = []
    for doc_name, pdf in zip(DOC_NAMES, paths):
        doc = fitz.open(pdf)
        for i, page in enumerate(doc):
            text = clean(page.get_text("text"))
            if text:
                all_pages.append({"doc": doc_name, "page": i + 1, "text": text})
        n = len(doc)
        doc.close()
        print(f"{doc_name}: {n} 页解析完成")
    OUT_FILE.write_text(json.dumps(all_pages, ensure_ascii=False), encoding="utf-8")
    print(f"共 {len(all_pages)} 页 -> {OUT_FILE}")


if __name__ == "__main__":
    main()
