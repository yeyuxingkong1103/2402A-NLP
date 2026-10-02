# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
import pymupdf
import json
import os
import re
from collections import Counter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config_v2 import PDF_PATH, CHUNK_FILE, CHUNK_SIZE, CHUNK_OVERLAP

def parse_pdf(path):
    doc = pymupdf.open(path)
    pages = []
    for i, page in enumerate(doc):
        blocks = page.get_text("blocks")
        text = "\n".join(b[4] for b in blocks if b[4].strip())
        pages.append({"page": i + 1, "text": text})
    return pages

def remove_headers_footers(pages):
    lines = []
    for p in pages:
        for line in p["text"].split("\n"):
            lines.append(line.strip())
    counter = Counter(lines)
    common = {l for l, c in counter.items() if c > len(pages) * 0.3 and len(l) < 30}
    cleaned = []
    for p in pages:
        kept = [l for l in p["text"].split("\n") if l.strip() not in common]
        cleaned.append({"page": p["page"], "text": "\n".join(kept)})
    return cleaned

def split_pages(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n第", "\n\n", "\n", "。", "；", "，", " ", ""]
    )
    chunks = []
    for p in pages:
        for c in splitter.split_text(p["text"]):
            chunks.append({"page": p["page"], "text": c, "type": "text"})
    return chunks

if __name__ == "__main__":
    pages = parse_pdf(PDF_PATH)
    print("总页数:", len(pages))
    pages = remove_headers_footers(pages)
    chunks = split_pages(pages)
    print("优化后片段数:", len(chunks))
    os.makedirs(os.path.dirname(CHUNK_FILE), exist_ok=True)
    with open(CHUNK_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print("已保存:", CHUNK_FILE)