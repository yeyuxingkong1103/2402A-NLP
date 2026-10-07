"""
朴素 RAG 基线（「优化前」对照实现）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

工单2 明确要求「给出优化解决方案，并对比优化前后检索精确度的变化」。
要做对比，就必须有一个**可运行、可复现**的「优化前」版本。本模块就是这个对照实现。

⚠️ 诚实声明：本项目在工单1 阶段是一步到位写成优化版的，没有留存早期代码。
因此这里的「优化前」**不是某个历史提交**，而是按 RAG 工程中最常见的
朴素做法（naive RAG）**重新构造的对照实现（ablation baseline）**。
它存在两个意义：
  1) 作为「优化前」的量化基准，让优化收益可度量、可复现；
  2) 作为反例，说明「把 PDF 切碎了直接塞向量库」这类最常见的入门写法到底差在哪。

逐层对照（这也是工单2「从 pdf 解析、分块、检索等层面优化」的三个层面）：

    层面         朴素基线（优化前）                     主线系统（优化后）
    ------------ --------------------------------------- -------------------------------------------
    PDF 解析     不做行内标题断行                        split_inline_headings 强制断行（Word 转 PDF 的坑）
                 不抽取表格                              疑似表格页抽 Markdown 表格并独立成块
    分块         定长 500 字滑窗，无结构感知             标题感知打包 + 一级标题硬断 + 长度兜底
                 表格混进普通文本                        表格按行拆分并重复表头
    检索         纯向量余弦 top-k                        余弦 + 0.30 × BM25 归一
                 无无区分度词过滤                        DF 过滤（自动剔除公司全称这类噪音词）
                 无阈值闸门                              依据分阈值闸门（拦掉没依据的问题）

**公平性约束**（差异必须只来自上面三层，其余全部一致）：
  * 同一个向量模型（bge-small-zh-v1.5，512 维）
  * 同一个生成模型、同一段 system prompt、同一个 top_k
  * 同一个页眉/页码去噪函数（这是 PDF 处理的入门功课，两边都做，不算优化点）
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np

from .chunker import Chunk
from .config import (
    INDEX_NAIVE_DIR,
    NAIVE_CHUNK_MIN_LENGTH,
    NAIVE_CHUNK_OVERLAP,
    NAIVE_CHUNK_SIZE,
    NAIVE_CHUNKS_JSONL,
    NAIVE_EMBEDDINGS_NPY,
    NAIVE_INDEX_META_JSON,
    RETRIEVAL_FINAL_TOP_K,
    WORK_ORDER_NO_OPT,  # noqa: F401
)
from .embedder import cosine_scores, embed_query, embed_texts
from .pdf_parser import ParsedDoc, parse_pdf
from .retriever import RetrievedChunk, build_context


# ------------------------------------------------------------------ 朴素解析


def parse_naive(pdf_path, with_tables: bool = False) -> ParsedDoc:
    """
    「优化前」的解析：只去页眉页码，**不做**行内标题断行、**不抽**表格。

    with_tables 默认 False —— 朴素做法通常只 `page.extract_text()` 拿到一段文本，
    不会再去识别表格结构。
    """
    return parse_pdf(pdf_path, with_tables=with_tables, split_inline=False)


# ------------------------------------------------------------------ 朴素分块


def chunk_naive(
    parsed: ParsedDoc,
    size: int = NAIVE_CHUNK_SIZE,
    overlap: int = NAIVE_CHUNK_OVERLAP,
    min_len: int = NAIVE_CHUNK_MIN_LENGTH,
) -> list[Chunk]:
    """
    「优化前」的分块：**定长滑窗**，完全无视章节结构。

    这才是 RAG 入门最常见的写法。它有两个内生的毛病，正是后面要优化的对象：
      1) 窗口边界是**字符位置**决定的，会从句子中间、从表格中间切开。
         一个财务数字表和它的表头被切到两个块里之后，两个块都不再自洽 ——
         检索命中其中任意一个，模型都读不出「这列是 2018 年度」。
      2) 一个块可能横跨两个毫不相干的章节，块向量被两个主题平均，谁都不像。

    页码映射：先按页拼接成一个大字符串并记录每页的字符区间，
    切完之后用区间反查每个块覆盖了哪些页，保证引用页码仍然准确（这点两边一致，不是优化点）。
    """
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []  # (start, end, page)
    pos = 0
    for p in parsed.pages:
        if not p.text:
            continue
        if parts:
            parts.append("\n")
            pos += 1
        start = pos
        parts.append(p.text)
        pos += len(p.text)
        spans.append((start, pos, p.page))

    full = "".join(parts)
    if not full:
        return []

    def pages_for(a: int, b: int) -> tuple[int, int]:
        """返回区间 [a, b) 覆盖的页码范围。"""
        hit = [pg for (s, e, pg) in spans if e > a and s < b]
        return (hit[0], hit[-1]) if hit else (0, 0)

    chunks: list[Chunk] = []
    seq = 0
    step = max(1, size - overlap)
    i = 0
    n = len(full)
    while i < n:
        piece = full[i : i + size].strip()
        if len(piece) >= min_len:
            seq += 1
            p0, p1 = pages_for(i, min(i + size, n))
            chunks.append(
                Chunk(
                    chunk_id=f"n{seq:05d}",
                    doc=parsed.source,
                    page=p0,
                    page_end=p1,
                    section="",          # 朴素分块没有章节概念
                    type="text",         # 表格也一律当普通文本
                    text=piece,
                )
            )
        i += step
    return chunks


# ------------------------------------------------------------------ 朴素检索


@dataclass
class NaiveResult:
    """朴素检索的结果（字段对齐 RetrievalResult，便于复用同样的下游消费逻辑）。"""

    query: str
    items: list[RetrievedChunk]
    gated: bool = False

    def contexts(self) -> list[str]:
        return [it.text for it in self.items]


class NaiveIndex:
    """
    朴素向量索引：只存 (分块, 向量)，检索 = 全量余弦取 top-k。

    **没有** BM25、**没有** DF 过滤、**没有**阈值闸门 —— 这三样都是主线系统的优化点。
    """

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray):
        self.chunks = chunks
        self.embeddings = embeddings

    # -------------------------------------------------- 构建
    @classmethod
    def build(cls, parsed: ParsedDoc, size: int = NAIVE_CHUNK_SIZE,
              overlap: int = NAIVE_CHUNK_OVERLAP) -> "NaiveIndex":
        chunks = chunk_naive(parsed, size=size, overlap=overlap)
        vecs = embed_texts([c.text for c in chunks])
        return cls(chunks, vecs)

    # -------------------------------------------------- 落盘
    def save(self, meta_extra: dict | None = None) -> dict:
        INDEX_NAIVE_DIR.mkdir(parents=True, exist_ok=True)
        with NAIVE_CHUNKS_JSONL.open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        np.save(NAIVE_EMBEDDINGS_NPY, self.embeddings.astype(np.float32))
        meta = {
            "work_order_no": WORK_ORDER_NO_OPT,
            "variant": "naive_dense_only",
            "chunk_size": NAIVE_CHUNK_SIZE,
            "chunk_overlap": NAIVE_CHUNK_OVERLAP,
            "num_chunks": len(self.chunks),
            "embedding_dim": int(self.embeddings.shape[1]) if self.embeddings.size else 0,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            **(meta_extra or {}),
        }
        NAIVE_INDEX_META_JSON.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return meta

    @classmethod
    def load(cls) -> "NaiveIndex":
        if not NAIVE_CHUNKS_JSONL.exists() or not NAIVE_EMBEDDINGS_NPY.exists():
            raise FileNotFoundError(
                "朴素基线索引不存在。请先执行：python scripts/build_baseline_index.py"
            )
        chunks = [
            Chunk(**json.loads(ln))
            for ln in NAIVE_CHUNKS_JSONL.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        return cls(chunks, np.load(NAIVE_EMBEDDINGS_NPY))

    # -------------------------------------------------- 检索
    def retrieve(self, query: str, top_k: int = RETRIEVAL_FINAL_TOP_K) -> NaiveResult:
        """纯向量余弦 top-k，不设阈值闸门（朴素做法就是「总能返回 k 条」）。"""
        qvec = embed_query(query)
        scores = cosine_scores(self.embeddings, qvec)
        order = np.argsort(-scores)[:top_k]
        items = [
            RetrievedChunk(
                chunk_id=self.chunks[i].chunk_id,
                text=self.chunks[i].text,
                page=self.chunks[i].page,
                page_end=self.chunks[i].page_end,
                section="",
                type=self.chunks[i].type,
                doc=self.chunks[i].doc,
                cosine=float(scores[i]),
                evidence=float(scores[i]),   # 朴素基线里「依据分」就等于余弦
                score=float(scores[i]),
            )
            for i in order
        ]
        return NaiveResult(query=query, items=items, gated=False)


# ------------------------------------------------------------------ 朴素问答链路


_naive_singleton: NaiveIndex | None = None


def get_naive_index() -> NaiveIndex:
    global _naive_singleton
    if _naive_singleton is None:
        _naive_singleton = NaiveIndex.load()
    return _naive_singleton


def naive_contexts(question: str, top_k: int = RETRIEVAL_FINAL_TOP_K) -> list[str]:
    """只取召回文本，不调 LLM。消融实验与指标计算用这个，快且零成本。"""
    return get_naive_index().retrieve(question, top_k=top_k).contexts()


def answer_naive(question: str, top_k: int = RETRIEVAL_FINAL_TOP_K) -> dict:
    """
    「优化前」的端到端问答：朴素检索 → 同一段 prompt → 同一个模型。

    返回 dict（而不是 rag.Answer），是为了让调用方一眼看出这是对照实现、
    不要拿它当主链路用。
    """
    from . import llm
    from .rag import _RAG_SYSTEM

    t0 = time.perf_counter()
    res = get_naive_index().retrieve(question, top_k=top_k)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    contexts = build_context(res.items, max_chars=4800)
    t = time.perf_counter()
    text = llm.chat(
        [
            {"role": "system", "content": _RAG_SYSTEM},
            {
                "role": "user",
                "content": f"【资料片段】\n{contexts}\n\n【问题】\n{question}\n\n请依据上述资料片段回答。",
            },
        ]
    )
    gen_ms = (time.perf_counter() - t) * 1000

    return {
        "question": question,
        "answer": text.strip(),
        "mode": "naive_rag",
        "citations": [
            {
                "index": i,
                "chunk_id": it.chunk_id,
                "doc": it.doc,
                "page": it.page,
                "page_end": it.page_end,
                "section": it.section,
                "type": it.type,
                "score": round(it.score, 4),
                "evidence": round(it.evidence, 4),
                "text": it.text,
            }
            for i, it in enumerate(res.items, 1)
        ],
        "contexts": res.contexts(),
        # 与主线一致：评估要用**真正喂给模型的那份上下文**（含页码表头），
        # 否则评审模型看不到页码，会把正确的页码引用误判成无依据。
        "context_text": contexts,
        "timing": {
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(gen_ms, 2),
            "total_ms": round((time.perf_counter() - t0) * 1000, 2),
        },
    }
