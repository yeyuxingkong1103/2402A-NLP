# -*- coding: utf-8 -*-
"""
工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统
步骤1：文档解析 —— 用 PyMuPDF 提取《招股说明书1》全文，按页存储为 JSON
运行：python parse_pdf.py   （在本工单目录下执行）
PDF 路径从 pdf_path.txt 读取，输出固定写入 data/pages.json
"""
import json
import re
from pathlib import Path

import fitz  # PyMuPDF

PDF_PATH_FILE = Path("pdf_path.txt")
OUT_FILE = Path("data") / "pages.json"


def clean(text: str) -> str:
    """页面文本清洗：去页眉页脚、页码、多余空白"""
    # 去掉 "武汉兴图新科电子股份有限公司 招股说明书（申报稿）" 页眉和 "1-1-50" 页码
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
