# -*- coding: utf-8 -*-
"""
工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
步骤1：文档解析（与工单01一致，PyMuPDF 按页提取+清洗）
运行：python parse_pdf.py
"""
import json
import re
from pathlib import Path

import fitz  # PyMuPDF

PDF_PATH_FILE = Path("pdf_path.txt")
OUT_FILE = Path("data") / "pages.json"


def clean(text: str) -> str:
    """页面文本清洗：去页眉页脚、页码、多余空白"""
    text = re.sub(r"武汉兴图新科电子股份有限公司\s*\n?\s*招股说明书（申报稿）", "", text)
    text = re.sub(r"\n\s*1-1-\d+\s*(?=\n|$)", "", text)
    text = re.sub(r"[ \t\u3000]+", " ", text)
    return text.strip()


def main():
    pdf_path = PDF_PATH_FILE.read_text(encoding="utf-8").strip()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = clean(page.get_text("text"))
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()

    OUT_FILE.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    total = sum(len(p["text"]) for p in pages)
    print(f"解析完成：{len(pages)} 页，共 {total} 字符 -> {OUT_FILE}")


if __name__ == "__main__":
    main()
