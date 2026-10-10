# -*- coding: utf-8 -*-
"""知识库：构建、持久化与加载（向量 + BM25 双索引）。

工单编号: 人工智能 NLP-RAG-Query 理解优化任务

知识来源（三通道）：
    - text   ：正文段落切片（结构感知 + 重叠）
    - table  ：表格键值对块（字段型问题的关键来源）
    - figure ：图形语义块（组织结构图 → 层级化可检索文本，沿用 04 工单能力）

持久化文件（vector_db/）：
    - chunks.jsonl   ：知识块文本与元数据（source / page / kind）
    - embeddings.npy ：知识块稠密向量（已 L2 归一化）
    - meta.json      ：向量库元信息（块数、维度、模型、构建时间）
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import List

import numpy as np

from src import config
from src.chunking import Chunk, build_chunks
from src.embedding import Embedder, normalize
from src.pdf_parser import parse_all


@dataclass
class KnowledgeBase:
    """知识库运行时对象。"""
    chunks: List[Chunk]
    vectors: np.ndarray            # (N, D) 已归一化
    bm25: object = None            # rank_bm25.BM25Okapi
    tokenized: List[List[str]] = None

    @property
    def size(self) -> int:
        return len(self.chunks)


def _tokenize(text: str) -> List[str]:
    """中文分词（jieba）+ 英文/数字词元。"""
    import jieba
    toks = [w.strip() for w in jieba.cut(text or "")]
    return [w for w in toks if w and not w.isspace()]


def append_figure_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """把图形语义文本作为知识块追加（kind="figure"）。

    组织结构图内的文字在 PDF 文本层被打散为竖排单字，直接入库不可检索；
    `figure_semantics` 用「节点框 + 连线」还原层级后得到结构化语句，
    这里将其作为独立知识块接入检索链路。
    """
    if not config.USE_FIGURE_SEMANTICS:
        return chunks
    try:
        from src.figure_semantics import extract_all_figure_semantics

        sems = extract_all_figure_semantics(config.PDF_PATHS)
    except Exception as exc:      # 图形解析失败不应阻断文本/表格知识库构建
        print(f"[KB] 图形语义解析跳过：{exc}", flush=True)
        return chunks

    cid = max((c.id for c in chunks), default=-1) + 1
    for s in sems:
        if not s.semantic_text:
            continue
        chunks.append(Chunk(id=cid, text=s.semantic_text, source=s.source,
                            page=s.page, kind="figure"))
        cid += 1
    print(f"[KB] 图形语义块 {len(sems)} 个（组织结构图）", flush=True)
    return chunks


def build_index(chunks: List[Chunk], verbose: bool = True) -> None:
    """构建并持久化知识库索引。"""
    from rank_bm25 import BM25Okapi

    config.ensure_dirs()
    embedder = Embedder()
    texts = [c.text for c in chunks]
    if verbose:
        print(f"[KB] 知识块 {len(chunks)} 个，开始向量化（{config.EMBEDDING_MODEL}）...", flush=True)
    t0 = time.time()
    vecs = normalize(embedder.encode(texts, verbose=verbose))
    if verbose:
        print(f"[KB] 向量化完成，用时 {time.time() - t0:.1f}s，维度 {vecs.shape}", flush=True)

    # 持久化切片
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    np.save(config.EMB_PATH, vecs)
    meta = {
        "chunks": len(chunks),
        "dim": int(vecs.shape[1]) if vecs.size else 0,
        "model": config.EMBEDDING_MODEL,
        "sources": sorted({c.source for c in chunks}),
        "kinds": {k: sum(1 for c in chunks if c.kind == k)
                  for k in sorted({c.kind for c in chunks})},
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config.META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # BM25 索引（构建时预热 jieba 词典）
    tokenized = [_tokenize(t) for t in texts]
    _ = BM25Okapi(tokenized)  # 触发校验
    if verbose:
        print(f"[KB] 索引已落盘 → {config.DB_DIR}", flush=True)


def load_kb(with_bm25: bool = True) -> KnowledgeBase:
    """加载已构建的知识库（若不存在则自动构建）。"""
    if not config.CHUNKS_PATH.exists() or not config.EMB_PATH.exists():
        raise FileNotFoundError(
            f"未找到知识库，请先运行 build_kb.py（缺失 {config.DB_DIR}）")
    chunks: List[Chunk] = []
    with open(config.CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            chunks.append(Chunk(id=d["id"], text=d["text"], source=d["source"],
                                page=d["page"], kind=d["kind"]))
    vectors = np.load(config.EMB_PATH)
    kb = KnowledgeBase(chunks=chunks, vectors=vectors)
    if with_bm25:
        from rank_bm25 import BM25Okapi
        kb.tokenized = [_tokenize(c.text) for c in chunks]
        kb.bm25 = BM25Okapi(kb.tokenized)
    return kb


def build_from_scratch(max_pages: int | None = None, verbose: bool = True) -> dict:
    """完整构建流程：解析 PDF → 切片 → 图形语义 → 向量化 → 落盘。"""
    pages = parse_all(max_pages=max_pages, with_tables=True)
    chunks = build_chunks(pages)
    if verbose:
        kinds = {k: sum(1 for c in chunks if c.kind == k) for k in {c.kind for c in chunks}}
        print(f"[KB] 解析 {len(pages)} 页 → 切片 {len(chunks)} 个 {kinds}", flush=True)
    chunks = append_figure_chunks(chunks)
    build_index(chunks, verbose=verbose)
    return json.loads(config.META_PATH.read_text(encoding="utf-8"))