# -*- coding: utf-8 -*-
"""
工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统
步骤2：分块 + 向量化 + 入库 ChromaDB
运行：python build_index.py
"""
import json
from pathlib import Path

import chromadb
import ollama

CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
BATCH = 32
CHROMA_DIR = "chroma_db"
COLLECTION = "zhaogu1_v1"


def make_chunks(pages):
    """按固定窗口+重叠分块，保留页码元数据"""
    chunks = []
    for p in pages:
        text, page_no = p["text"], p["page"]
        step = CHUNK_SIZE - CHUNK_OVERLAP
        for start in range(0, len(text), step):
            part = text[start:start + CHUNK_SIZE]
            if len(part) < 50:  # 末尾过短的碎片并入上一块
                if chunks and chunks[-1]["page"] == page_no:
                    chunks[-1]["text"] += part
                break
            chunks.append({"page": page_no, "text": part})
            if start + CHUNK_SIZE >= len(text):
                break
    return chunks


def embed_texts(texts):
    """通过本机 Ollama 官方 SDK 生成 BGE-M3 向量（固定端点 127.0.0.1:11434）"""
    resp = ollama.embed(model="bge-m3", input=texts)
    return resp["embeddings"]


def main():
    pages = json.loads(Path("data/pages.json").read_text(encoding="utf-8"))
    chunks = make_chunks(pages)
    print(f"分块完成：{len(chunks)} 块")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    if col.count() > 0:
        print(f"集合 {COLLECTION} 已有 {col.count()} 条，跳过重建")
        return

    texts = [c["text"] for c in chunks]
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        vecs = embed_texts(batch)
        col.add(
            ids=[f"c{i+j}" for j in range(len(batch))],
            embeddings=vecs,
            documents=batch,
            metadatas=[{"page": chunks[i+j]["page"]} for j in range(len(batch))],
        )
        print(f"  已入库 {min(i+BATCH, len(texts))}/{len(texts)}")
    print(f"索引构建完成：{col.count()} 条向量 -> {CHROMA_DIR}")


if __name__ == "__main__":
    main()
