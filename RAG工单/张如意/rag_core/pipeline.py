# -*- coding: utf-8 -*-
"""
RAG 主流程编排
工单编号：人工智能NLP-RAG（供 01~13 全部工单共用）

把「解析 → 分块 → 建索引 → 检索 → 生成」串成一条可配置的流水线，
各工单通过 PipelineConfig 切换能力开关，形成递进式版本：

  工单01  基础版：fixed 分块 + 纯向量检索 + LLM 生成
  工单02  优化版：structure 分块 + 向量检索 + TF-IDF 重排
  工单03  表格版：+ 表格解析入库
  工单04  图像版：+ 图像多模态解析入库
  工单05  多轮版：+ Query 理解与指代消解
  工单06  混合版：+ BM25 全文检索 + 混合融合 + 级联重排
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from . import config, generator, query_understand
from .chunk import chunk_blocks
from .pdf_parse import ParsedDoc, parse_pdf
from .retriever import Retriever


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """流水线能力开关（各工单在此差异化的基础上递进）。"""
    name: str = "default"
    chunk_strategy: str = "structure"     # fixed | recursive | semantic | structure
    chunk_size: int = config.CHUNK_SIZE
    chunk_overlap: int = config.CHUNK_OVERLAP
    with_tables: bool = True              # 工单03 开启
    with_images: bool = False             # 工单04 开启
    strategy: str = "hybrid"              # vector | fulltext | hybrid
    fusion: str = "rrf"                   # weighted | rrf | vote
    reranker: str = "tfidf"               # none | llm | tfidf | adaptive | cascade
    top_k: int = config.TOP_K_RERANK
    recall_k: int = config.TOP_K_RECALL
    alpha: float = config.HYBRID_ALPHA
    use_query_understanding: bool = True  # 工单05 开启

    def to_dict(self) -> dict:
        return asdict(self)


# 各工单对应的预设配置
PRESETS: dict[str, PipelineConfig] = {
    "wo01_baseline": PipelineConfig(
        name="wo01_baseline", chunk_strategy="fixed", with_tables=False,
        strategy="vector", reranker="none", use_query_understanding=False),
    "wo02_optimized": PipelineConfig(
        name="wo02_optimized", chunk_strategy="structure", with_tables=False,
        strategy="vector", reranker="tfidf", use_query_understanding=False),
    "wo03_table": PipelineConfig(
        name="wo03_table", chunk_strategy="structure", with_tables=True,
        strategy="vector", reranker="tfidf", use_query_understanding=False),
    "wo04_image": PipelineConfig(
        name="wo04_image", chunk_strategy="structure", with_tables=True,
        with_images=True, strategy="hybrid", fusion="rrf", reranker="cascade"),
    "wo05_multiturn": PipelineConfig(
        name="wo05_multiturn", chunk_strategy="structure", with_tables=True,
        with_images=True, strategy="hybrid", fusion="rrf",
        reranker="cascade", use_query_understanding=True),
    "wo06_hybrid": PipelineConfig(
        name="wo06_hybrid", chunk_strategy="structure", with_tables=True,
        with_images=True, strategy="hybrid", fusion="rrf",
        reranker="cascade", use_query_understanding=True),
}


# ---------------------------------------------------------------------------
# 流水线
# ---------------------------------------------------------------------------
class Pipeline:
    """
    RAG 流水线。

    用法：
        p = Pipeline(PRESETS["wo06_hybrid"])
        p.build_index([PDF_PROSPECTUS_1, PDF_PROSPECTUS_2])
        print(p.ask("军用领域的收入分别是多少？"))
    """

    def __init__(self, cfg: PipelineConfig, collection: str = "prospectus"):
        self.cfg = cfg
        self.collection = collection
        self.retriever = Retriever(collection)
        self.docs: list[ParsedDoc] = []
        self._index_path = config.INDEX_DIR / f"{collection}.meta.json"

    # -- 建索引 -------------------------------------------------------------
    def build_index(self, pdf_paths: list[Path],
                    doc_names: list[str] | None = None,
                    rebuild: bool = True,
                    verbose: bool = True) -> dict:
        """
        解析 PDF 并建立向量库 + BM25 索引。

        Returns:
            索引统计信息（文档数、块数、耗时）
        """
        from .bm25 import BM25Retriever
        from .vectorstore import VectorStore

        doc_names = doc_names or [Path(p).stem for p in pdf_paths]
        t0 = time.perf_counter()

        all_chunks = []
        for path, name in zip(pdf_paths, doc_names):
            t = time.perf_counter()
            if verbose:
                print(f"[1/2] 解析《{name}》…")
            parsed = parse_pdf(path, name, with_tables=self.cfg.with_tables,
                               with_images=self.cfg.with_images)
            self.docs.append(parsed)
            chunks = chunk_blocks(
                parsed.blocks, strategy=self.cfg.chunk_strategy,
                size=self.cfg.chunk_size, overlap=self.cfg.chunk_overlap,
            )
            all_chunks.extend(chunks)
            if verbose:
                print(f"      {parsed.n_pages} 页 → {len(chunks)} 个块 "
                      f"({time.perf_counter() - t:.1f}s)")

        if rebuild:
            VectorStore(self.collection).reset()

        vs = VectorStore(self.collection)
        vs.add_chunks(all_chunks, show_progress=verbose)

        bm25 = BM25Retriever()
        bm25.build(all_chunks)
        bm25.save()

        stats = {
            "collection": self.collection,
            "config": self.cfg.to_dict(),
            "docs": [{"name": d.name, "pages": d.n_pages,
                      "blocks": len(d.blocks)} for d in self.docs],
            "n_chunks": len(all_chunks),
            "n_vectors": vs.count(),
            "n_bm25_docs": len(bm25.doc_ids),
            "build_seconds": round(time.perf_counter() - t0, 2),
            "chunk_types": _count_types(all_chunks),
        }
        self._index_path.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

        if verbose:
            print(f"[2/2] 索引完成：{len(all_chunks)} 块，"
                  f"向量库 {vs.count()} 条，耗时 {stats['build_seconds']}s")
        return stats

    def load_index(self) -> None:
        """已有索引时直接装载，跳过重建。"""
        self.retriever.load_bm25()

    # -- 问答 ---------------------------------------------------------------
    def ask(
        self,
        question: str,
        history: list[query_understand.Turn] | None = None,
        top_k: int | None = None,
        return_trace: bool = False,
    ) -> dict:
        """
        完整问答流程：Query理解 → 检索 → （重排）→ 生成。

        Returns:
            {question, rewritten, answer, citations, timings, ...}
        """
        history = history or []
        top_k = top_k or self.cfg.top_k
        trace: dict = {"question": question, "timings": {}}
        t_all = time.perf_counter()

        # ---- 1. Query 理解 ----
        t = time.perf_counter()
        if self.cfg.use_query_understanding:
            qu = query_understand.understand(question, history)
        else:
            qu = query_understand.QueryUnderstanding(raw=question, rewritten=question)
        trace["understanding"] = qu.to_dict()
        trace["timings"]["query_understanding"] = time.perf_counter() - t

        if not qu.needs_retrieval:
            trace["answer"] = "您好，我是招股说明书问答助手，请提出与文档内容相关的问题。"
            trace["timings"]["total"] = time.perf_counter() - t_all
            return trace

        query = qu.rewritten or question

        # ---- 2. 检索（复杂问题走多路子问题检索）----
        t = time.perf_counter()
        if len(qu.sub_questions) > 1:
            docs = self._multi_query_retrieve(qu.sub_questions, top_k)
            trace["timings"]["retrieve"] = time.perf_counter() - t
        else:
            res = self.retriever.retrieve(
                query, strategy=self.cfg.strategy, top_k=top_k,
                recall_k=self.cfg.recall_k, reranker=self.cfg.reranker,
                fusion=self.cfg.fusion, alpha=self.cfg.alpha,
            )
            docs = res.docs
            trace["timings"].update(res.timings)
            trace["retrieval"] = res.to_dict()

        trace["docs"] = docs

        # ---- 3. 生成 ----
        t = time.perf_counter()
        ctx = self.retriever.format_context(docs)
        hist = [(h.question, h.answer) for h in history]
        gen = generator.generate_rag(question, docs, ctx, history=hist)
        trace["timings"]["generate"] = time.perf_counter() - t
        trace["timings"]["total"] = time.perf_counter() - t_all

        trace["answer"] = gen.answer
        trace["citations"] = gen.citations
        trace["refused"] = gen.refused

        if return_trace:
            return trace
        return {"question": question, "answer": gen.answer,
                "citations": gen.citations, "docs": docs}

    def _multi_query_retrieve(self, sub_questions: list[str], top_k: int) -> list[dict]:
        """多子问题并行检索后按 chunk_id 去重合并。"""
        seen, merged = set(), []
        per_k = max(top_k // max(len(sub_questions), 1) + 1, 2)
        for sq in sub_questions:
            docs = self.retriever.search(
                sq, strategy=self.cfg.strategy, top_k=per_k,
                recall_k=self.cfg.recall_k, reranker=self.cfg.reranker,
                fusion=self.cfg.fusion, alpha=self.cfg.alpha,
            )
            for d in docs:
                cid = d.get("chunk_id")
                if cid and cid not in seen:
                    seen.add(cid)
                    merged.append(d)
        return merged[:top_k * 2]

    # -- 多轮对话 -----------------------------------------------------------
    def chat(self, questions: list[str], verbose: bool = True) -> list[query_understand.Turn]:
        """
        依次执行多轮对话，自动维护上下文（工单05 演示用）。

        Returns:
            Turn 列表，含每轮的理解结果、检索片段与答案
        """
        history: list[query_understand.Turn] = []
        for i, q in enumerate(questions, 1):
            result = self.ask(q, history=history)
            turn = query_understand.Turn(
                question=q, answer=result["answer"],
                docs=result.get("docs", []),
            )
            history.append(turn)
            if verbose:
                print(f"\n{'=' * 70}\n第 {i} 轮")
                print(f"Q：{q}")
                ru = result.get("understanding", {})
                if ru.get("改写后") and ru["改写后"] != q:
                    print(f"   ↳ 指代消解：{ru['改写后']}")
                print(f"A：{result['answer']}")
        return history


def _count_types(chunks) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in chunks:
        out[c.type] = out.get(c.type, 0) + 1
    return out
