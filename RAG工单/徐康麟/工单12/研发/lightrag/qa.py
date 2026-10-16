# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""统一问答后端：``backend='rag' | 'lightrag'``。

**公平性约定（对比的前提）**：两种后端
  * 共用同一个向量索引与 BM25（``vector_index.load_index``）；
  * 共用同一段回答提示词模板、同一生成模型、同一 temperature / max_tokens；
  * 共用同一套引用标注规则（每个数字后标 ``【p页码】``）；
  * 唯一变量 = 检索通道（RAG：向量+BM25；LightRAG：局部向量+全局图谱）。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from lightrag.config import Cfg
from lightrag.double_retrieval import (DEFAULT_DOC_FILTER, Hit, RetrievalResult,
                                       retrieve_lightrag, retrieve_rag)
from lightrag.logging_util import get_logger
from lightrag.ollama_client import OllamaClient
from lightrag.routing import build_entity_dict, norm_ws, route_question

ANSWER_PROMPT = """你是招股说明书问答助手。只依据下面的【资料】回答问题，不得编造。

要求：
1. 用简洁的中文回答，先给结论，再给依据。
2. 每个关键数字后面标注来源页码，格式为【p页码】，例如【p22】。
3. 资料中没有的信息，直接回答「资料中未找到相关信息」，不要猜测。

【资料】
{context}

【问题】{question}
【回答】"""


@dataclass
class AnswerResult:
    question: str
    backend: str
    answer: str
    hits: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    latency_s: float = 0.0
    retrieve_s: float = 0.0
    generate_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    trace: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend, "question": self.question, "answer": self.answer,
            "hits": self.hits, "sources": self.sources,
            "latency_s": round(self.latency_s, 3), "retrieve_s": round(self.retrieve_s, 3),
            "generate_s": round(self.generate_s, 3),
            "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
            "trace": self.trace, "error": self.error,
        }


class QAEngine:
    """加载索引 / 图谱 / 块表，提供统一 ``answer()`` 接口。"""

    def __init__(self, cfg: Cfg, *, client: OllamaClient | None = None, logger: Any = None) -> None:
        self.cfg = cfg
        self.log = logger or get_logger("qa", cfg)
        self.client = client or OllamaClient(cfg, logger=self.log)
        self.vindex: Any = None
        self.bm25: Any = None
        self.chunks_by_id: dict[str, dict[str, Any]] = {}
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []
        self.entity_dict: dict[str, str] = {}
        self.doc_filter: tuple[str, ...] = DEFAULT_DOC_FILTER
        self._loaded = False

    # ------------------------------------------------------------------ 载入
    def load(self, *, doc_filter: Sequence[str] | None = None) -> dict[str, Any]:
        """载入索引、图谱与块表（不调 LLM）。"""
        from lightrag.vector_index import load_index

        self.vindex, self.bm25 = load_index(self.cfg, logger=self.log)

        # 块表：全部 3013 块，检索侧再按 doc_filter 过滤（不重建索引）
        chunks_path = Path(self.cfg.path_str("processed_dir")) / "chunks.jsonl"
        self.chunks_by_id = {}
        with chunks_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.chunks_by_id[str(row.get("chunk_id"))] = row

        graph_dir = Path(self.cfg.path_str("graph_dir"))
        for name, sink in (("nodes.jsonl", "nodes"), ("edges.jsonl", "edges")):
            path = graph_dir / name
            rows: list[dict[str, Any]] = []
            if path.is_file():
                with path.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            try:
                                rows.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            setattr(self, sink, rows)

        self.entity_dict = build_entity_dict(self.nodes,
                                             min_len=int(self.cfg.routing.min_entity_len),
                                             max_len=int(self.cfg.routing.max_entity_len))
        self.doc_filter = tuple(doc_filter) if doc_filter is not None else DEFAULT_DOC_FILTER
        self._loaded = True
        info = {
            "index_chunks": len(self.vindex.ids), "bm25_docs": int(self.bm25.size),
            "chunks_loaded": len(self.chunks_by_id), "graph_nodes": len(self.nodes),
            "graph_edges": len(self.edges), "entity_dict": len(self.entity_dict),
            "doc_filter": list(self.doc_filter),
            "embed_model": str(self.cfg.llm.embed_model), "gen_model": str(self.cfg.llm.gen_model),
        }
        self.log.info("QAEngine 载入完成：%s", json.dumps(info, ensure_ascii=False))
        return info

    # ------------------------------------------------------------------ 检索
    def retrieve(self, question: str, *, backend: str) -> RetrievalResult:
        if not self._loaded:
            raise RuntimeError("请先调用 load()")
        if backend == "rag":
            return retrieve_rag(question, cfg=self.cfg, vindex=self.vindex, bm25=self.bm25,
                                embed_query=self.client.embed_query,
                                chunks_by_id=self.chunks_by_id, doc_filter=self.doc_filter)
        if backend in ("lightrag", "lightrag_strict"):
            edge_mode = ("constraint_ok" if backend == "lightrag_strict"
                         else str(self.cfg.retrieval.get("global_edge_mode") or "all"))
            route = route_question(question, self.entity_dict, self.cfg)
            res = retrieve_lightrag(question, route, cfg=self.cfg, vindex=self.vindex, bm25=self.bm25,
                                    embed_query=self.client.embed_query,
                                    chunks_by_id=self.chunks_by_id, nodes=self.nodes,
                                    edges=self.edges, edge_mode=edge_mode, doc_filter=self.doc_filter)
            res.backend = backend          # 用请求的标签，便于区分「全量边 / 仅相容边」两组口径
            return res
        raise ValueError(f"未知后端：{backend}（只支持 rag / lightrag / lightrag_strict）")

    # ------------------------------------------------------------------ 组上下文
    def build_context(self, hits: Sequence[Hit]) -> tuple[str, list[dict[str, Any]]]:
        """取前 ``top_k_in_prompt`` 块拼上下文，总长截断到 ``max_context_chars``。"""
        a = self.cfg.answer
        limit = int(a.top_k_in_prompt)
        cap = int(a.max_context_chars)
        parts: list[str] = []
        sources: list[dict[str, Any]] = []
        used = 0
        for idx, hit in enumerate(hits[:limit], start=1):
            meta = self.chunks_by_id.get(hit.chunk_id) or {}
            content = str(meta.get("content") or "")
            remain = cap - used
            if remain <= 0:
                break
            piece = content[:remain]
            used += len(piece)
            parts.append(f"[{idx}]（第{hit.page}页，{hit.doc_key}，来源通道：{hit.source}）\n{piece}")
            sources.append({"idx": idx, "chunk_id": hit.chunk_id, "page": hit.page,
                            "doc_key": hit.doc_key, "type": hit.type, "source": hit.source,
                            "score": round(float(hit.score), 6)})
        return "\n\n".join(parts), sources

    # ------------------------------------------------------------------ 问答
    def answer(self, question: str, *, backend: str, generate: bool = True) -> AnswerResult:
        t0 = time.perf_counter()
        result = self.retrieve(question, backend=backend)
        t_ret = time.perf_counter() - t0

        context, sources = self.build_context(result.hits)
        prompt = ANSWER_PROMPT.format(context=context or "（无检索结果）", question=question)

        answer_text = ""
        gen_s = 0.0
        ptok = ctok = 0
        error = ""
        if generate:
            t1 = time.perf_counter()
            try:
                answer_text, stats = self.client.generate_with_stats(
                    prompt, temperature=float(self.cfg.answer.temperature),
                    max_tokens=int(self.cfg.answer.max_tokens), kind="gen")
                ptok, ctok = int(stats.prompt_tokens), int(stats.completion_tokens)
            except Exception as exc:  # noqa: BLE001 - 单题失败不中断整批
                error = f"{type(exc).__name__}: {exc}"
                self.log.warning("问题生成失败：%s", error)
            gen_s = time.perf_counter() - t1
            answer_text = answer_text.strip()

        return AnswerResult(
            question=question, backend=backend, answer=answer_text,
            hits=[h.to_dict() for h in result.hits], sources=sources,
            latency_s=time.perf_counter() - t0, retrieve_s=t_ret, generate_s=gen_s,
            prompt_tokens=ptok, completion_tokens=ctok, trace=result.trace, error=error,
        )

    # ------------------------------------------------------------------ 收尾
    def unload(self) -> list[str]:
        """卸载 Ollama 模型（keep_alive=0），确保跑完不残留驻留模型。"""
        from lightrag.graph_extract import unload_models
        return unload_models(self.cfg, logger=self.log)


def extract_numbers(text: str) -> list[str]:
    """抽出文本中的数值/比例/金额（去空白归一后），供关键词命中判定复用。"""
    import re

    return re.findall(r"\d[\d,]*(?:\.\d+)?%?", norm_ws(text))
