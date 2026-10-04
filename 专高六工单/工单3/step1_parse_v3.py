# 工单编号：人工智能 NLP-RAG-PDF 文档的表格解析及检索优化
import pymupdf
import json
import os
from collections import Counter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config_v3 import PDF_LIST, CHUNK_FILE, CHUNK_SIZE, CHUNK_OVERLAP


def table_to_text(table):
    """把 find_tables 的结果转成 Markdown 文本"""
    rows = table.extract()
    if not rows:
        return ""
    md = []
    for i, row in enumerate(rows):
        cells = [str(c).replace("\n", " ").strip() if c else "" for c in row]
        md.append("| " + " | ".join(cells) + " |")
        if i == 0:
            md.append("|" + "---|" * len(cells))
    return "\n".join(md)


def parse_pdf(path, source):
    doc = pymupdf.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = "\n".join(b[4] for b in page.get_text("blocks") if b[4].strip())
        tables = []
        try:
            for t in page.find_tables():
                md = table_to_text(t)
                if md:
                    tables.append(md)
        except Exception:
            pass
        pages.append({"page": i + 1, "text": text, "tables": tables, "source": source})
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
        cleaned.append({
            "page": p["page"],
            "text": "\n".join(kept),
            "tables": p["tables"],
            "source": p["source"]
        })
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
            chunks.append({
                "page": p["page"],
                "text": c,
                "type": "text",
                "source": p["source"]
            })
        for t in p["tables"]:
            chunks.append({
                "page": p["page"],
                "text": t,
                "type": "table",
                "source": p["source"]
            })
    return chunks


if __name__ == "__main__":
    all_chunks = []
    for source, path in PDF_LIST.items():
        print(f"解析 {source}: {path}")
        pages = parse_pdf(path, source)
        print("  页数:", len(pages))
        pages = remove_headers_footers(pages)
        chunks = split_pages(pages)
        text_n = sum(1 for c in chunks if c["type"] == "text")
        table_n = sum(1 for c in chunks if c["type"] == "table")
        print(f"  正文片段: {text_n}, 表格片段: {table_n}")
        all_chunks.extend(chunks)

    print("总片段数:", len(all_chunks))
    os.makedirs(os.path.dirname(CHUNK_FILE), exist_ok=True)
    with open(CHUNK_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print("已保存:", CHUNK_FILE)