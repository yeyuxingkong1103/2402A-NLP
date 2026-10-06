# -*- coding: utf-8 -*-
"""
工单编号：人工智能 NLP-RAG-PDF 文档的表格解析及检索优化
步骤2：表格解析 + 文本分块，统一入库
  - 表格：PyMuPDF find_tables 结构化抽取，转"表头|行"文本块（表格类问题的答案多在表格里，
    纯文本流会把表格行列切碎，这是工单01/02 中表格问题检索失败的根因）
  - 文本：沿用工单02 的段落/表格行感知分块
  - 两份招股书入库同一集合，metadata 带 doc/page，检索可跨文档
运行：python build_index.py
"""
import json
from pathlib import Path

import chromadb
import fitz
import ollama

CHUNK_SIZE = 800
SHORT_LINE = 30
CHROMA_DIR = "chroma_db"
# 每份招股书一个集合（chromadb hnsw 段在大集合跨进程重载时有 bug，小集合已验证稳定）
COLLECTIONS = {"zhaogu1": "zhaogu1_v3", "zhaogu2": "zhaogu2_v3"}
BATCH = 32

NOISE_RE_TEXT = ("不存在虚假记载、误导性陈述", "备查文件", "发行保荐书", "法律意见书")


def is_noise(text: str) -> bool:
    return any(k in text for k in NOISE_RE_TEXT)


def page_blocks(text: str):
    """页面文本 -> 语义块（短行=表格行/列举项 合并；长行=段落成块）"""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    blocks, buf = [], ""
    for line in lines:
        if len(line) < SHORT_LINE:
            buf = (buf + " " + line).strip()
            continue
        if buf:
            blocks.append(buf)
        buf = line
    if buf:
        blocks.append(buf)
    return blocks


def make_text_chunks(pages):
    chunks = []
    for p in pages:
        if is_noise(p["text"]):
            continue
        cur = ""
        for b in page_blocks(p["text"]):
            if not cur:
                cur = b
            elif len(cur) + 1 + len(b) <= CHUNK_SIZE:
                cur = cur + " " + b
            else:
                chunks.append({"doc": p["doc"], "page": p["page"], "text": cur, "kind": "text"})
                cur = b
        if cur:
            chunks.append({"doc": p["doc"], "page": p["page"], "text": cur, "kind": "text"})
    return chunks


def extract_tables(pdf_path: str, doc_name: str):
    """结构化抽取每页表格 -> 文本块：表头一行 + 数据行"列值1 | 列值2 | ..." """
    doc = fitz.open(pdf_path)
    rows_out = []
    for i, page in enumerate(doc):
        try:
            tabs = page.find_tables()
        except Exception:
            continue
        for t_idx, tab in enumerate(tabs.tables):
            data = tab.extract()
            if not data or len(data) < 2:
                continue
            cells = [["" if c is None else str(c).replace("\n", "") for c in row] for row in data]
            header = " | ".join(c for c in cells[0] if c.strip())
            lines = [f"【表格】({doc_name} 第{i+1}页 表{t_idx+1}) 表头: {header}"]
            for row in cells[1:]:
                line = " | ".join(c for c in row if c.strip())
                if line.strip():
                    lines.append(line)
            text = "\n".join(lines)
            if len(text) > 30:
                rows_out.append({"doc": doc_name, "page": i + 1, "text": text, "kind": "table"})
    doc.close()
    return rows_out


def embed_texts(texts):
    """本机 Ollama 官方 SDK 生成 BGE-M3 向量（固定端点 127.0.0.1:11434）"""
    return ollama.embed(model="bge-m3", input=texts)["embeddings"]


def main():
    paths = [p.strip() for p in Path("pdf_paths.txt").read_text(encoding="utf-8").splitlines() if p.strip()]
    pages = json.loads(Path("data/pages_all.json").read_text(encoding="utf-8"))

    chunks = make_text_chunks(pages)
    for doc_name, pdf in zip(["zhaogu1", "zhaogu2"], paths):
        tb = extract_tables(pdf, doc_name)
        print(f"{doc_name}: {len(tb)} 个表格块")
        chunks.extend(tb)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    total = 0
    for doc_name, coll_name in COLLECTIONS.items():
        doc_chunks = [c for c in chunks if c["doc"] == doc_name]
        col = client.get_or_create_collection(coll_name, metadata={"hnsw:space": "cosine"})
        if col.count() > 0:
            print(f"集合 {coll_name} 已有 {col.count()} 条，跳过")
            total += col.count()
            continue
        for i in range(0, len(doc_chunks), BATCH):
            batch = doc_chunks[i:i + BATCH]
            col.add(
                ids=[f"{coll_name}c{i+j}" for j in range(len(batch))],
                embeddings=embed_texts([c["text"] for c in batch]),
                documents=[c["text"] for c in batch],
                metadatas=[{"doc": c["doc"], "page": c["page"], "kind": c["kind"]} for c in batch],
            )
            print(f"  [{coll_name}] 已入库 {min(i + BATCH, len(doc_chunks))}/{len(doc_chunks)}")
        # 构建后立即验证：强制加载 HNSW 并压实
        vec = ollama.embed(model="bge-m3", input=["验证"])["embeddings"][0]
        r = col.query(query_embeddings=[vec], n_results=3)
        print(f"[{coll_name}] 构建完成 {col.count()} 条，验证查询命中页码：{[m['page'] for m in r['metadatas'][0]]}")
        total += col.count()
    print(f"两集合合计 {total} 条")


if __name__ == "__main__":
    main()
