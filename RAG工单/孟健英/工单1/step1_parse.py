# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统
import pymupdf as fitz
import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import PDF_PATH, CHUNK_FILE, CHUNK_SIZE, CHUNK_OVERLAP

def parse_pdf(path):
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append({"page": i + 1, "text": text})
    return pages

def split_pages(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""]
    )
    chunks = []
    for p in pages:
        for c in splitter.split_text(p["text"]):
            chunks.append({"page": p["page"], "text": c})
    return chunks

if __name__ == "__main__":
    pages = parse_pdf(PDF_PATH)
    print("总页数:", len(pages))
    chunks = split_pages(pages)
    print("总片段数:", len(chunks))
    os.makedirs(os.path.dirname(CHUNK_FILE), exist_ok=True)
    with open(CHUNK_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print("已保存:", CHUNK_FILE)
    print("示例:", chunks[0]["text"][:200])