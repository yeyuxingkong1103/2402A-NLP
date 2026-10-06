# -*- coding: utf-8 -*-
"""
工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
步骤2：分块优化 —— 段落/表格感知分块（工单01固定窗口会把表格行和列举句切碎）
规则：
  1) 短行（<30字符，多为表格行/表头/列举项）连续合并为一个语义块，不再切断；
  2) 长行按段落成块；
  3) 块按顺序打包为 <=800 字符的 chunk，相邻 chunk 保留上一块末尾 1 个块作重叠；
  4) 保留页码元数据。
运行：python build_index.py
"""
import json
import re
from pathlib import Path

import chromadb
import ollama

# 噪声页过滤：声明页/备查文件页只有套话，会以高相似度挤占召回名额
NOISE_RE = re.compile(r"不存在虚假记载、误导性陈述|备查文件|发行保荐书|法律意见书")


def is_noise(text: str) -> bool:
    return bool(NOISE_RE.search(text))

CHUNK_SIZE = 800
SHORT_LINE = 30          # 短于该值视为表格行/列举项
CHROMA_DIR = "chroma_db"
COLLECTION = "zhaogu1_v2"
BATCH = 32


def page_blocks(text: str):
    """页面文本 -> 语义块列表（表格行合并，段落成块）"""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    blocks, buf = [], ""
    for line in lines:
        if len(line) < SHORT_LINE:  # 短行并入当前块（表格行/表头）
            buf = (buf + " " + line).strip()
            continue
        if buf:
            blocks.append(buf)
        buf = line
    if buf:
        blocks.append(buf)
    return blocks


def make_chunks(pages):
    chunks = []
    for p in pages:
        if is_noise(p["text"]):
            continue
        blocks = page_blocks(p["text"])
        cur = ""
        for b in blocks:
            if not cur:
                cur = b
            elif len(cur) + 1 + len(b) <= CHUNK_SIZE:
                cur = cur + " " + b
            else:
                chunks.append({"page": p["page"], "text": cur})
                cur = b  # 重叠：下一块从被截断的块重新开始（块本身不切断）
        if cur:
            chunks.append({"page": p["page"], "text": cur})
    return chunks


def embed_texts(texts):
    """本机 Ollama 官方 SDK 生成 BGE-M3 向量（固定端点 127.0.0.1:11434）"""
    return ollama.embed(model="bge-m3", input=texts)["embeddings"]


def main():
    pages = json.loads(Path("data/pages.json").read_text(encoding="utf-8"))
    chunks = make_chunks(pages)
    print(f"分块完成：{len(chunks)} 块（工单01固定窗口为 667 块）")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    if col.count() > 0:
        print(f"集合 {COLLECTION} 已有 {col.count()} 条，跳过重建")
        return

    texts = [c["text"] for c in chunks]
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        col.add(
            ids=[f"v2c{i+j}" for j in range(len(batch))],
            embeddings=embed_texts(batch),
            documents=batch,
            metadatas=[{"page": chunks[i + j]["page"]} for j in range(len(batch))],
        )
        print(f"  已入库 {min(i + BATCH, len(texts))}/{len(texts)}")
    print(f"索引构建完成：{col.count()} 条向量 -> {CHROMA_DIR}/{COLLECTION}")


if __name__ == "__main__":
    main()
