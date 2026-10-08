# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-LightRAG优化
"""
传统 RAG 基线模块（扁平向量检索）。

设计说明（公平对比原则）：
    1. 文本切块：直接复用 LightRAG 切好的 chunks（读取 LightRAG 工作目录下的
       kv_store_text_chunks.json），保证两种方案面对完全相同的文本块；
    2. 向量化：同样使用 bge-m3（1024 维）；
    3. 检索方式：传统的"扁平"向量库检索——查询向量与全部 chunk 向量做
       余弦相似度，取 Top-K。这正是 LightRAG 论文中所对比的 VanLLaMA /
       naively-RAG 的"chunk-vector"范式；
    4. 不做任何图扩展、关键词提取与重排，以凸显双层检索机制的增益。
"""

import json
import time
from pathlib import Path

import numpy as np

from config import WORK_DIR, EMBED_DIM, RESULT_DIR
from lightrag_build import _get_embed_model
from logger import get_logger

logger = get_logger(__name__)

# 模块级缓存：chunks 与向量矩阵
_cache = {"chunks": None, "matrix": None, "ids": None}


def _find_kv_file() -> Path:
    """定位 LightRAG 的 text_chunks KV 文件。"""
    for cand in (WORK_DIR / "kv_store_text_chunks.json",
                 WORK_DIR / "kv_store" / "kv_store_text_chunks.json"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"未找到 text_chunks KV 文件（工作目录 {WORK_DIR}）")


def load_chunks() -> list[dict]:
    """加载 LightRAG 切好的文本块。"""
    if _cache["chunks"] is not None:
        return _cache["chunks"]

    kv = json.loads(_find_kv_file().read_text(encoding="utf-8"))
    chunks = []
    for cid, item in kv.items():
        content = item.get("content", "") if isinstance(item, dict) else str(item)
        if content.strip():
            chunks.append({"id": cid, "content": content,
                           "source": item.get("file_path", "") if isinstance(item, dict) else ""})
    _cache["chunks"] = chunks
    logger.info(f"已加载 {len(chunks)} 个文本块（来自 LightRAG 切块）")
    return chunks


def build_index() -> tuple[list[dict], np.ndarray]:
    """对全部 chunks 向量化，构建扁平向量索引（npy 缓存）。"""
    if _cache["matrix"] is not None:
        return _cache["chunks"], _cache["matrix"]

    chunks = load_chunks()
    vec_file = RESULT_DIR / "baseline_chunk_vectors.npy"
    if vec_file.exists() and vec_file.stat().st_mtime > _find_kv_file().stat().st_mtime:
        matrix = np.load(vec_file)
        logger.info(f"向量索引缓存已加载：{matrix.shape}")
    else:
        model = _get_embed_model()
        texts = [c["content"] for c in chunks]
        t0 = time.time()
        matrix = np.asarray(model.encode(texts, batch_size=16,
                                         normalize_embeddings=True,
                                         show_progress_bar=True))
        np.save(vec_file, matrix)
        logger.info(f"向量索引构建完成：{matrix.shape}，耗时 {time.time() - t0:.1f}s")

    _cache["matrix"] = matrix
    _cache["ids"] = [c["id"] for c in chunks]
    return chunks, matrix


def search(query: str, top_k: int = 8) -> list[dict]:
    """传统 RAG 检索：余弦相似度 Top-K。"""
    chunks, matrix = build_index()
    model = _get_embed_model()
    t0 = time.time()
    qvec = np.asarray(model.encode([query], normalize_embeddings=True))[0]
    sims = matrix @ qvec
    top_idx = np.argsort(-sims)[:top_k]
    elapsed = time.time() - t0
    results = []
    for rank, i in enumerate(top_idx, 1):
        results.append({
            "rank": rank,
            "id": chunks[i]["id"],
            "content": chunks[i]["content"],
            "score": float(sims[i]),
            "source": chunks[i]["source"],
        })
    logger.info(f"传统RAG检索完成：query={query[:20]}... top_k={top_k} 耗时{elapsed:.2f}s")
    return results


if __name__ == "__main__":
    rs = search("武汉兴图新科电子股份有限公司注册资本是多少", top_k=3)
    for r in rs:
        print(f"[{r['rank']}] {r['score']:.4f} | {r['content'][:80]}...")
