# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
src/table_parser/table_embedding.py —— 工单三表格向量化

职责（见 docs/02_表格检索优化方案.md §七）：
  对 table_to_text 生成的自然语言描述做 bge-m3 嵌入，
  输出 (table_chunks, vectors) 供 table_store 入 Milvus rag_tables collection。

复用工单二 src.embedding.get_embedder（单例，避免重复加载模型）。
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from src.embedding import get_embedder, DEFAULT_BATCH_SIZE  # 工单二复用


def embed_table_chunks(
    table_chunks: List[Dict[str, Any]],
    batch_size: Optional[int] = None,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """工单三：对 table_chunks 列表做 bge-m3 嵌入

    Args:
        table_chunks: tables_to_texts 输出，每项含 table_text 字段
    Returns:
        np.ndarray shape=(N, 1024)，空表跳过（全零向量占位）
    """
    if not table_chunks:
        return np.zeros((0, 1024), dtype=np.float32)

    bs = batch_size if batch_size is not None else DEFAULT_BATCH_SIZE
    embedder = get_embedder()
    texts = [(tc.get("table_text") or "").strip() for tc in table_chunks]
    # 空文本用占位符避免编码错（后续标记 low_confidence）
    valid_texts = [t if t else "（空表）" for t in texts]
    vecs = embedder.encode(
        valid_texts, batch_size=bs, show_progress_bar=show_progress_bar,
    ).astype(np.float32)
    # 空表向量置零（检索时自然排后）
    for i, t in enumerate(texts):
        if not t:
            vecs[i] = np.zeros_like(vecs[i])
    logger.info(
        f"[table_embedding] 嵌入完成: {len(table_chunks)} 张表, "
        f"dim={vecs.shape[1] if len(vecs) else 'N/A'}"
    )
    return vecs


def embed_tables_json(
    tables_json_path: str,
    out_path: Optional[str] = None,
) -> str:
    """工单三：读 data/tables/<doc>_tables.json → 嵌入 → 写回（含 vectors）

    Args:
        tables_json_path: data/tables/招股说明书1_tables.json
        out_path: 输出路径（默认覆盖原文件）
    Returns:
        写出的路径
    """
    p = Path(tables_json_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    table_chunks = data.get("table_texts") or []
    if not table_chunks:
        logger.warning(f"[table_embedding] {p.name} 无 table_texts，跳过")
        return str(p)

    vecs = embed_table_chunks(table_chunks)
    # 把向量写回每个 table_chunk
    for i, tc in enumerate(table_chunks):
        tc["embedding"] = vecs[i].tolist() if i < len(vecs) else []
    data["embedding_model"] = "BAAI/bge-m3"
    data["embedding_dim"] = int(vecs.shape[1]) if len(vecs) else 1024

    out = Path(out_path) if out_path else p
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    logger.info(f"[table_embedding] 写回 {len(table_chunks)} 条向量 → {out}")
    return str(out)


def load_table_chunks_with_vectors(tables_json_path: str) -> Tuple[List[Dict], np.ndarray]:
    """工单三：读 JSON，若已有 embedding 直接用，否则现算

    Returns:
        (table_chunks, vectors[N, 1024])
    """
    p = Path(tables_json_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    chunks = data.get("table_texts") or []
    if chunks and "embedding" in chunks[0] and chunks[0]["embedding"]:
        vecs = np.array([c["embedding"] for c in chunks], dtype=np.float32)
        logger.info(f"[table_embedding] {p.name} 已有向量, shape={vecs.shape}")
        return chunks, vecs
    vecs = embed_table_chunks(chunks)
    return chunks, vecs


if __name__ == "__main__":  # pragma: no cover
    import argparse
    cli = argparse.ArgumentParser(description="工单三：表格向量化 CLI")
    cli.add_argument("--input", required=True, help="data/tables/<doc>_tables.json")
    cli.add_argument("--out", default=None)
    args = cli.parse_args()
    embed_tables_json(args.input, args.out)
