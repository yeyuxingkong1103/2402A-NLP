# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化
import pymupdf
import json
import os
from collections import Counter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config_v4 import (
    PDF_LIST, CHUNK_FILE, IMAGE_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, MIN_IMG_W, MIN_IMG_H
)


def table_to_text(table):
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


def extract_images(doc, source):
    """提取每页大图，保存到本地，返回 [{page, path}]"""
    os.makedirs(IMAGE_DIR, exist_ok=True)
    imgs = []
    for i, page in enumerate(doc):
        for idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                pix = pymupdf.Pixmap(doc, xref)
                if pix.width < MIN_IMG_W or pix.height < MIN_IMG_H:
                    continue
                if pix.n > 4:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                path = os.path.join(IMAGE_DIR, f"{source}_p{i+1}_{idx}.png")
                pix.save(path)
                imgs.append({"page": i + 1, "path": path, "source": source})
            except Exception:
                continue
    return imgs


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
    imgs = extract_images(doc, source)
    return pages, imgs


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
            chunks.append({"page": p["page"], "text": c, "type": "text", "source": p["source"]})
        for t in p["tables"]:
            chunks.append({"page": p["page"], "text": t, "type": "table", "source": p["source"]})
    return chunks


if __name__ == "__main__":
    all_chunks = []
    all_images = []
    for source, path in PDF_LIST.items():
        print(f"解析 {source}: {path}")
        pages, imgs = parse_pdf(path, source)
        print(f"  页数: {len(pages)}, 提取大图: {len(imgs)}")
        pages = remove_headers_footers(pages)
        chunks = split_pages(pages)
        text_n = sum(1 for c in chunks if c["type"] == "text")
        table_n = sum(1 for c in chunks if c["type"] == "table")
        print(f"  正文: {text_n}, 表格: {table_n}")
        all_chunks.extend(chunks)
        all_images.extend(imgs)

    print("总片段数:", len(all_chunks))
    print("总图像数:", len(all_images))

    os.makedirs(os.path.dirname(CHUNK_FILE), exist_ok=True)
    with open(CHUNK_FILE, "w", encoding="utf-8") as f:
        json.dump({"chunks": all_chunks, "images": all_images}, f, ensure_ascii=False, indent=2)
    print("已保存:", CHUNK_FILE)