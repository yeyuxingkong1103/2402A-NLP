# -*- coding: utf-8 -*-
"""工单8 Graph RAG 检索与问答引擎。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

链路（与 工单7 的「向量+BM25+RRF+重排」严格衔接，图谱作为**第三条召回通道**叠加）：

    问句
      → ① 基础召回：复用 工单7 的 ``retriever.retrieve``（bge-m3 向量 + BM25 → RRF → 轻量重排）
      → ② 图谱召回：问句实体识别 → 一跳/两跳邻域扩展 → 邻居实体的提及块（带跳数衰减分）
      → ③ 融合排序：基础命中按跳数获得**可审计的加成**，图谱独有候选按跳数分入池
      → ④ 证据补全：命中页同类块（枚举完整性）
      → ⑤ 生成：Ollama ``qwen2.5:3b`` 基于证据块作答 / 失败降级为抽取式
      → ⑥ 返回：答案 + **答案所在文本块**（chunk_id / 页码 / 文件名 / 文本）+ 图谱结构缩略

为什么图谱通道用「加成」而不是「加权求和」：
    基础分（重排 final_score ≈ 4.x）与图谱结构分（0.02~0.06）**语义空间不同**，
    直接相加会让图谱分被淹没、或让图谱块无理压过向量命中。
    因此：基础命中加**小量结构分**（只影响同分档的次序），图谱独有候选才用结构分定序，
    且每块都留 ``graph_bonus`` / ``retrieval_channel`` 字段供审计（报告里逐题可查）。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import BASE_INDEX_DIR, GraphRAGConfig, get_config
from .llm import GraphLLM, render_evidence
from .store import GraphStore, graph_digest, load_graph

WORK_ORDER = "人工智能NLP-RAG-基于Graph RAG 实现金融问答"

#: 基础检索/生成层沿用 工单7 的默认值（评估口径必须一致，才谈得上「比对 07 工单的测试结果」）
DEFAULT_ENV: dict[str, str] = {
    "RAG_DATA__INDEX_DIR": str(BASE_INDEX_DIR),
    # 日志目录钉在本工单：app/core 是从 工单7 复制的，其 REPO_ROOT 会算到 工单7
    # ⇒ 不覆盖就会越界写 工单7 的日志（实测踩坑，见 设计/环境事实.md §3.4）
    "RAG_LOG__DIR": str(Path(__file__).resolve().parents[3] / "部署" / "日志"),
    "RAG_RETRIEVAL__TOP_K": "10",
    "RAG_RETRIEVAL__PROMPT_TOP_K": "3",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
}


def apply_base_env(*, override: bool = False) -> dict[str, str]:
    """把基础检索所需的环境变量灌进 ``os.environ``（默认不覆盖调用方已设的值）。

    关键两条：
        * ``RAG_DATA__INDEX_DIR`` 指向 **工单7 的索引绝对路径** —— 本工单不重新解析、不重新嵌入；
        * ``RAG_LOG__DIR`` 指向 **本工单** 的日志目录 —— ``config.py`` 的 ``REPO_ROOT`` 按文件位置
          计算，而 ``app/core`` 是从 工单7 复制的，不覆盖就会把日志写进只读的 工单7 目录
          （实测踩坑，见 ``设计/环境事实.md`` §3.4）。
    """
    applied: dict[str, str] = {}
    for key, value in DEFAULT_ENV.items():
        if override or not os.environ.get(key):
            os.environ[key] = value
            applied[key] = value
    return applied


@dataclass(slots=True)
class GraphAnswer:
    """一次 Graph RAG 问答的完整结果。"""

    question: str
    answer: str
    language: str = "zh"
    refused: bool = False
    generation_method: str = "graph_llm"
    backend: str = "ollama"
    model: str = "qwen2.5:3b"
    #: **答案所在文本块**（工单硬性要求：chunk_id / 页码 / 文件名 / 文本）
    evidence_chunks: list[dict[str, Any]] = field(default_factory=list)
    all_chunks: list[dict[str, Any]] = field(default_factory=list)
    graph: dict[str, Any] = field(default_factory=dict)
    graph_used: bool = False
    latency_ms: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question, "answer": self.answer, "language": self.language,
            "refused": self.refused, "generation_method": self.generation_method,
            "backend": self.backend, "model": self.model,
            "evidence_chunks": self.evidence_chunks, "chunks": self.all_chunks,
            "graph": self.graph, "graph_used": self.graph_used,
            "latency_ms": self.latency_ms, "diagnostics": self.diagnostics,
            "trace_id": self.trace_id,
        }


def _payload(hit: Mapping[str, Any], *, text_limit: int | None = None) -> dict[str, Any]:
    """命中 → 对外 payload（含 chunk_id / 页码 / 文件名 / 文本，即「答案所在文本块」）。"""
    text = str(hit.get("content") or hit.get("text") or "")
    if text_limit is not None:
        text = text[:text_limit]
    return {
        "chunk_id": str(hit.get("chunk_id") or ""),
        "page": int(hit.get("page") or 0),
        "page_basis": "1-based-physical",
        "file_name": str(hit.get("file_name") or ""),
        "type": str(hit.get("type") or hit.get("modality") or "text"),
        "text": text,
        "score": float(hit.get("score") or hit.get("final_score") or 0.0),
        "base_score": float(hit.get("base_score") or 0.0),
        "graph_bonus": float(hit.get("graph_bonus") or 0.0),
        "graph_distance": hit.get("graph_distance"),
        "graph_entities": list(hit.get("graph_entities") or []),
        "retrieval_channel": str(hit.get("retrieval_channel") or "base"),
    }


class GraphRAGEngine:
    """Graph RAG 引擎：图谱增强检索 + 证据约束生成。"""

    def __init__(self, cfg: GraphRAGConfig | None = None, *, logger: Any = None,
                 graph: GraphStore | None = None, llm: GraphLLM | None = None,
                 warmup: bool = False) -> None:
        apply_base_env()
        self.cfg = cfg or get_config(logger=logger)
        self.log = logger
        self.graph = graph if graph is not None else load_graph(self.cfg, logger=logger)
        self.llm = llm if llm is not None else GraphLLM(self.cfg, logger=logger)
        self._retriever: Any = None
        self._analyzer: Any = None
        self._analysis_cache: dict[str, Any] = {}
        if warmup:
            self.warmup()

    # ------------------------------------------------------------------ 惰性依赖
    @property
    def retriever(self) -> Any:
        """工单7 的检索模块（只读复用）。"""
        if self._retriever is None:
            from app.core import retriever as _retriever

            self._retriever = _retriever
        return self._retriever

    def warmup(self, *, with_llm: bool = False) -> dict[str, Any]:
        """预热：加载 工单7 索引束 + 分词器 + 嵌入 + 图谱；可选探测 Ollama。"""
        started = time.perf_counter()
        out: dict[str, Any] = {}
        try:
            bundle = self.retriever.load_bundle(logger=self.log)
            out["index_chunks"] = bundle.size
            out["index_dir"] = bundle.index_dir
        except Exception as exc:  # noqa: BLE001 —— 索引不可用时如实上报，由调用方决定是否继续
            out["index_error"] = f"{type(exc).__name__}: {exc}"
        out["graph"] = {"nodes": self.graph.node_count, "edges": self.graph.edge_count,
                        "ready": self.graph.ready, "load_ms": self.graph.load_ms}
        if with_llm:
            out["llm_available"] = self.llm.available()
        out["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        if self.log is not None:
            self.log.log_event("graphrag.warmup", **out)
        return out

    # ------------------------------------------------------------------ 图谱扩展
    def match_entities(self, question: str) -> list[str]:
        """问句里命中的图谱实体（长名优先；限制数量避免「一句话命中几十个实体」）。"""
        if not self.graph.ready:
            return []
        limit = int(self.cfg.retrieval.get("max_expand_nodes", 40))
        return self.graph.match_names(question, min_len=int(self.cfg.retrieval.get("min_entity_name_len", 2)),
                                      limit=max(limit // 2, 8))

    def graph_expand(self, question: str, *, corpus_hint: str | None = None) -> dict[str, Any]:
        """图谱扩展主过程：实体命中 → 一跳/两跳邻域 → 邻居提及块（跳数衰减分）。

        ``corpus_hint``（问句点名的语料键）非空且 ``graph_corpus_filter`` 为真时，
        **只保留该语料自己的块**。依据（实测）：9 份年报混排时，母公司/子公司互相提及会让
        两跳走到别家公司年报（Q07「平安银行」的两跳带进中国平安/招商银行/国泰君安的年报块），
        这些块对「平安银行的营业收入」毫无用处，却会挤掉真正的答案块。
        """
        started = time.perf_counter()
        ret = self.cfg.retrieval
        hop = int(ret.get("hop", 2))
        max_nodes = int(ret.get("max_expand_nodes", 40))
        min_conf = float(ret.get("min_confidence", 0.55))
        seeds = self.match_entities(question)
        out: dict[str, Any] = {"seeds": [], "seed_names": [], "distances": {}, "edges_used": [],
                               "truncated": False, "hop": hop, "nodes": 0, "chunks": {},
                               "corpus_hint": corpus_hint, "filtered_out_of_corpus": 0,
                               "elapsed_ms": 0.0}
        if not seeds:
            out["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return out
        expanded = self.graph.expand(seeds, hop=hop, max_nodes=max_nodes, min_confidence=min_conf)
        distances: dict[str, int] = expanded["distances"]
        seed_score = float(ret.get("graph_seed_score", 0.055))
        hop1_score = float(ret.get("graph_hop1_score", 0.035))
        hop2_score = float(ret.get("graph_hop2_score", 0.022))
        base_by_distance = {0: seed_score, 1: hop1_score, 2: hop2_score}
        chunks: dict[str, dict[str, Any]] = {}
        limit_chunks = int(ret.get("max_graph_chunks", 15))
        apply_filter = bool(ret.get("graph_corpus_filter", True)) and bool(corpus_hint)
        for node_id, distance in sorted(distances.items(), key=lambda kv: (kv[1], kv[0])):
            node = self.graph.node(node_id)
            if node is None:
                continue
            weight = base_by_distance.get(int(distance), hop2_score * 0.8)
            for chunk in self.graph.chunks_of(node_id, limit=4):
                cid = str(chunk.get("chunk_id") or "")
                if not cid:
                    continue
                if apply_filter and self._corpus_of_file(str(chunk.get("file_name") or "")) != str(corpus_hint):
                    out["filtered_out_of_corpus"] += 1
                    continue
                entry = chunks.get(cid)
                if entry is None:
                    if len(chunks) >= limit_chunks * 2:
                        continue
                    entry = {"chunk_id": cid, "page": int(chunk.get("page") or 0),
                             "file_name": str(chunk.get("file_name") or ""),
                             "type": str(chunk.get("type") or "text"), "distance": int(distance),
                             "score": 0.0, "entities": []}
                    chunks[cid] = entry
                entry["distance"] = min(int(entry["distance"]), int(distance))
                entry["score"] = round(float(entry["score"]) + weight
                                       * (1.0 + 0.05 * min(int(node.get("count") or 0), 20)), 6)
                if node.get("name") not in entry["entities"]:
                    entry["entities"].append(str(node.get("name")))
        # 按结构分排序截断，保证图谱通道贡献可控
        ranked = sorted(chunks.values(), key=lambda c: (-float(c["score"]), c["chunk_id"]))[:limit_chunks]
        out.update({
            "seeds": seeds,
            "seed_names": [self.graph.name_of(s) for s in seeds],
            # distances 以 **node_id** 为键（便于用 id 取节点属性），另给 names 映射供 UI 显示
            "distances": {k: int(v) for k, v in distances.items()},
            "distances_named": {self.graph.name_of(k): int(v) for k, v in distances.items()},
            "edges_used": expanded["edges_used"][:60],
            "truncated": bool(expanded["truncated"]),
            "nodes": len(distances),
            "chunks": {c["chunk_id"]: c for c in ranked},
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        })
        return out

    # ------------------------------------------------------------------ 检索
    def retrieve(self, question: str, *, top_k: int | None = None,
                 use_graph: bool = True) -> dict[str, Any]:
        """Graph RAG 检索：基础召回 + 图谱通道融合，返回候选与诊断。"""
        started = time.perf_counter()
        ret = self.cfg.retrieval
        k = int(top_k if top_k is not None else ret.get("top_k", 10))
        base_hits: list[dict[str, Any]] = []
        analysis = None
        try:
            analyzer = self._analyzer
            if analyzer is None:
                from app.core.query_understanding import analyze

                analyzer = analyze
                self._analyzer = analyzer
            bundle = self.retriever.load_bundle(logger=self.log)
            analysis = analyzer(question, corpus_issuers=bundle.corpus_issuers, logger=self.log)
            hits = self.retriever.retrieve(question, top_k=k, analysis=analysis, logger=self.log,
                                           use_cache=bool(ret.get("use_cache", False)))
            base_hits = [dict(h) for h in hits]
        except Exception as exc:  # noqa: BLE001 —— 基础检索失败必须显式上报（不静默返回空）
            if self.log is not None:
                self.log.log_event("graphrag.base_retrieve_failed", level="ERROR",
                                   error=f"{type(exc).__name__}: {exc}")
            return {"question": question, "hits": [], "graph": {"error": f"{type(exc).__name__}: {exc}"},
                    "graph_used": False, "error": f"{type(exc).__name__}: {exc}",
                    "latency_ms": {"total": round((time.perf_counter() - started) * 1000, 2)}}
        base_ids = {str(h.get("chunk_id")) for h in base_hits}
        hint = str(getattr(analysis, "corpus_hint", "") or "") or None
        graph_info = self.graph_expand(question, corpus_hint=hint) if use_graph else {
            "seeds": [], "chunks": {}, "nodes": 0, "seed_names": [], "distances": {},
            "edges_used": [], "elapsed_ms": 0.0, "corpus_hint": None, "filtered_out_of_corpus": 0}
        graph_chunks: Mapping[str, dict[str, Any]] = graph_info.get("chunks") or {}
        merged: dict[str, dict[str, Any]] = {}
        for hit in base_hits:
            cid = str(hit.get("chunk_id"))
            info = graph_chunks.get(cid)
            base_score = float(hit.get("score") or hit.get("final_score") or 0.0)
            item = dict(hit)
            item["base_score"] = round(base_score, 6)
            item["graph_bonus"] = round(float(info["score"]), 6) if info else 0.0
            item["graph_distance"] = int(info["distance"]) if info else None
            item["graph_entities"] = list(info["entities"]) if info else []
            item["retrieval_channel"] = ("base+graph" if info else "base")
            item["score"] = round(base_score + float(item["graph_bonus"]), 6)
            item["final_score"] = item["score"]
            merged[cid] = item
        # 图谱独有候选：没有基础分，按结构分入池（系数 <1 ⇒ 不会无理压过基础命中）
        # **相关性闸门**：图谱独有块必须至少含一个问句实词/命中实体名，否则不许进榜。
        # 依据（实测踩坑）：纯结构分（0.02~0.06）与基础分不同量纲，不设闸门时
        # 「邮储银行不良率」会被同图里的「中国平安年报 p3」「国泰君安 p4」挤进 Top-10。
        keywords = _query_terms(question, list(graph_info.get("seed_names") or []))
        rejected: list[str] = []
        for cid, info in graph_chunks.items():
            if cid in merged:
                continue
            row = self._chunk_row(cid)
            if row is None:
                continue
            if keywords and not any(k in str(row.get("content") or "") for k in keywords):
                rejected.append(cid)
                continue
            item = dict(row)
            item["base_score"] = 0.0
            item["graph_bonus"] = round(float(info["score"]), 6)
            item["graph_distance"] = int(info["distance"])
            item["graph_entities"] = list(info["entities"])
            item["retrieval_channel"] = "graph_only"
            item["score"] = round(float(info["score"]), 6)
            item["final_score"] = item["score"]
            merged[cid] = item
        ranked = sorted(merged.values(), key=lambda c: (-float(c.get("score") or 0.0), str(c.get("chunk_id"))))
        for rank, item in enumerate(ranked, 1):
            item["rank"] = rank
        top = ranked[: max(k, 1)]
        # ---- 图谱独有候选的处置（两种模式，配置化）----
        # 模式 A（**默认**，`graph_slots=""`）：**追加**在 Top-K 之后，基础排名完全不动。
        #   依据（实测）：9 份年报混排时图谱两跳会走到别家公司年报，占名额会把真正的答案块挤出去
        #   （Q07：中国平安 p7 / 招商银行 p7 / 国泰君安 p4 顶掉 p19/p44 表格，答案质量反降）。
        # 模式 B（`graph_slots="2,5,9"`）：按名额交错，**需通过净增益闸门**（见下）。
        ret_cfg = self.cfg.retrieval
        slots = [int(x) for x in str(ret_cfg.get("graph_slots", "")).split(",") if str(x).strip()]
        graph_only_ranked = sorted([h for h in ranked if h.get("retrieval_channel") == "graph_only"],
                                   key=lambda c: (-float(c.get("graph_bonus") or 0.0), str(c.get("chunk_id"))))
        displaced: list[dict[str, Any]] = []
        blocked: list[dict[str, Any]] = []
        trade_terms = _trade_terms(question, list(graph_info.get("seed_names") or []))
        if graph_only_ranked and slots:
            merged_top: list[dict[str, Any]] = []
            graph_iter = iter(graph_only_ranked)
            used_ids: set[str] = set()
            for position in range(1, max(k, 1) + 1):
                base_item = top[position - 1] if position <= len(top) else None
                if position in slots:
                    candidate = None
                    for item in graph_iter:                      # 找到第一个能通过净增益闸门的候选
                        if str(item.get("chunk_id")) in used_ids:
                            continue
                        gain = _term_hits(str(item.get("content") or ""), trade_terms)
                        have = _term_hits(str((base_item or {}).get("content") or ""), trade_terms)
                        if base_item is None or gain > have:
                            candidate = item
                            if base_item is not None:
                                displaced.append({"position": position,
                                                  "chunk_id": base_item.get("chunk_id"),
                                                  "page": base_item.get("page"),
                                                  "score": base_item.get("score"),
                                                  "term_hits": have})
                                candidate["displaced_term_hits"] = have
                            candidate["term_hits"] = gain
                            break
                        blocked.append({"position": position, "chunk_id": item.get("chunk_id"),
                                        "graph_term_hits": gain, "base_term_hits": have,
                                        "reason": "图谱候选的问句实词命中数不多于被替换块 ⇒ 不替换"})
                    if candidate is not None:
                        used_ids.add(str(candidate.get("chunk_id")))
                        merged_top.append(candidate)
                        continue
                if base_item is not None:
                    merged_top.append(base_item)
            for item in ranked:
                if len(merged_top) >= max(k, 1):
                    break
                if str(item.get("chunk_id")) in used_ids or item in merged_top:
                    continue
                merged_top.append(item)
            top = merged_top[: max(k, 1)]
            for rank, item in enumerate(top, 1):
                item["rank"] = rank
        elif graph_only_ranked:
            # 模式 A：图谱块追加在 Top-K 之后（不改变前 K 名，命中判定口径不受影响）
            limit = int(ret_cfg.get("graph_augment_max", 5))
            seen = {str(h.get("chunk_id")) for h in top}
            for item in graph_only_ranked:
                if len(top) >= max(k, 1) + limit:
                    break
                if str(item.get("chunk_id")) in seen:
                    continue
                seen.add(str(item.get("chunk_id")))
                top.append(item)
        out_hits = top
        graph_only = [h for h in out_hits if h.get("retrieval_channel") == "graph_only"]
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        diag = {
            "base_hits": len(base_hits),
            "graph_hits": len(graph_chunks),
            "graph_only_in_topk": len(graph_only),
            "graph_only_ids": [h.get("chunk_id") for h in graph_only],
            "graph_slots": slots,
            "graph_displaced": displaced,
            "graph_blocked_by_net_gain": blocked[:10],
            "graph_rejected_irrelevant": rejected[:10],
            "boosted": sum(1 for h in out_hits if float(h.get("graph_bonus") or 0) > 0),
            "graph_nodes": int(graph_info.get("nodes") or 0),
            "graph_seed_names": list(graph_info.get("seed_names") or []),
            "graph_truncated": bool(graph_info.get("truncated")),
            "distances": dict(list((graph_info.get("distances_named") or {}).items())[:30]),
            "hop": int(graph_info.get("hop") or 0),
            "graph_ms": float(graph_info.get("elapsed_ms") or 0.0),
            "total_ms": elapsed,
        }
        if self.log is not None:
            self.log.log_event("graphrag.retrieve", question_digest=str(question)[:60], **diag)
        return {"question": question, "hits": [_payload(h) for h in out_hits],
                "hits_raw": out_hits, "graph": self._local_graph(graph_info),
                "graph_used": bool(graph_info.get("seeds")), "analysis": _analysis_digest(analysis),
                "latency_ms": {"total": elapsed, "graph": diag["graph_ms"],
                               "base": round(max(elapsed - diag["graph_ms"], 0.0), 2)},
                "diagnostics": diag}

    def _chunk_row(self, chunk_id: str) -> dict[str, Any] | None:
        """从 工单7 索引束取块正文（图谱独有候选需要它的 content）。"""
        try:
            bundle = self.retriever.load_bundle(logger=self.log)
            return bundle.chunk(str(chunk_id))
        except Exception as exc:  # noqa: BLE001
            if self.log is not None:
                self.log.log_event("graphrag.chunk_lookup_failed", level="WARNING",
                                   chunk_id=chunk_id, error=f"{type(exc).__name__}: {exc}")
            return None

    def _corpus_of_file(self, file_name: str) -> str:
        """文件名 → 语料键（用 工单7 索引清单里的权威映射，缺失则回落到文件名启发式）。"""
        if not file_name:
            return ""
        try:
            bundle = self.retriever.load_bundle(logger=self.log)
            return self.retriever.corpus_key_of_file(file_name, getattr(bundle, "corpus_of_file", None))
        except Exception:  # noqa: BLE001 —— 映射不可用时退回启发式（不阻断检索）
            return self.retriever.corpus_key_of(file_name)

    def _local_graph(self, graph_info: Mapping[str, Any], *, max_nodes: int = 40,
                     max_edges: int = 60) -> dict[str, Any]:
        """问句命中的**局部子图**（UI 上展示「问题 → 图谱结构」的那部分）。

        ``graph_info["distances"]`` 以 node_id 为键（见 ``graph_expand``），因此这里直接取节点属性。
        """
        seeds = list(graph_info.get("seeds") or [])
        distances: Mapping[str, int] = graph_info.get("distances") or {}
        if not seeds:
            return {"seed_names": [], "distances": {}, "distances_named": {}, "nodes": [], "edges": []}
        keep: dict[str, int] = {}
        for sid in seeds:
            keep[str(sid)] = 0
        for nid, distance in distances.items():
            if nid not in keep:
                keep[str(nid)] = int(distance)
            if len(keep) >= max_nodes:
                break
        nodes = []
        for nid, distance in sorted(keep.items(), key=lambda kv: (kv[1], kv[0])):
            node = self.graph.node(nid)
            if node is None:
                continue
            nodes.append({"id": nid, "name": node.get("name"), "type": node.get("type"),
                          "type_zh": node.get("type_zh"), "distance": distance,
                          "degree": len(self.graph.adjacency.get(nid, []))})
        edges = []
        for edge in (graph_info.get("edges_used") or [])[:max_edges]:
            edges.append({"head": edge.get("head"), "relation": edge.get("relation"),
                          "tail": edge.get("tail"), "confidence": edge.get("confidence")})
        return {"seed_names": list(graph_info.get("seed_names") or []),
                "distances": dict(distances),
                "distances_named": dict(graph_info.get("distances_named") or {}),
                "nodes": nodes, "edges": edges,
                "edge_total_used": len(graph_info.get("edges_used") or [])}

    # ------------------------------------------------------------------ 问答
    def answer(self, question: str, *, top_k: int | None = None, use_graph: bool = True,
               language: str = "auto", use_llm: bool = True,
               return_graph: bool = True) -> GraphAnswer:
        """Graph RAG 端到端问答：检索 → 图谱扩展 → 生成 → 返回答案与证据块。"""
        trace_id = f"g{int(time.time() * 1000) % 10 ** 10:010d}"
        started = time.perf_counter()
        lang = _resolve_language(question, language)
        retrieval = self.retrieve(question, top_k=top_k, use_graph=use_graph)
        hits_raw: list[dict[str, Any]] = list(retrieval.get("hits_raw") or [])
        if retrieval.get("error"):
            lat = float((retrieval.get("latency_ms") or {}).get("total") or 0.0)
            return GraphAnswer(question=question, answer=f"检索失败：{retrieval['error']}",
                               language=lang, refused=True, generation_method="error",
                               graph=retrieval.get("graph") or {}, trace_id=trace_id,
                               latency_ms={"retrieve": lat, "generate": 0.0, "total": lat},
                               diagnostics=dict(retrieval.get("diagnostics") or {}))
        # 证据补全（同页同类块）——枚举完整性与 工单7 口径一致
        prompt_top_k = int(self.cfg.retrieval.get("prompt_top_k", 3))
        generation_hits = select_evidence(
            hits_raw, top_k=prompt_top_k,
            max_per_page=int(self.cfg.retrieval.get("evidence_max_per_page", 2)),
            prefer_text=bool(self.cfg.retrieval.get("evidence_prefer_text", True)),
        )
        compose_hits = list(generation_hits)
        try:
            compose_hits = self.retriever.expand_with_page_context(
                self.retriever.load_bundle(logger=self.log), generation_hits,
                max_extra=int(self.cfg.retrieval.get("page_context_extra", 6)), logger=self.log)
        except Exception as exc:  # noqa: BLE001 —— 补全失败不影响已有证据
            if self.log is not None:
                self.log.log_event("graphrag.page_context_failed", level="WARNING",
                                   error=f"{type(exc).__name__}: {exc}")
        generate_ms = 0.0
        method = "graph_grounded"
        backend = "ollama"
        model = self.llm.model
        text = ""
        error: str | None = None
        diag_extra: dict[str, Any] = {}
        if use_llm and self.llm.available():
            outcome = self._grounded_generate(question, lang, compose_hits)
            if outcome is not None:
                # 兼容两种结果对象：本工单 GroundedResult（elapsed_ms）与 工单7 GenerationOutcome（total_ms）
                generate_ms = float(getattr(outcome, "total_ms", None)
                                    or getattr(outcome, "elapsed_ms", 0.0) or 0.0)
                text = str(getattr(outcome, "text", "") or "")
                method = str(getattr(outcome, "method", "graph_grounded") or "graph_grounded")
                backend = str(getattr(outcome, "backend", "ollama") or "ollama")
                model = str(getattr(outcome, "model", self.llm.model) or self.llm.model)
                error = getattr(outcome, "error", None)
                guard = None
                if hasattr(outcome, "to_dict"):
                    try:
                        guard = outcome.to_dict()
                    except Exception:  # noqa: BLE001 —— 诊断序列化失败不影响主流程
                        guard = None
                if guard:
                    diag_extra["generation_guard"] = guard
            if not text:
                result = self.llm.answer(question, compose_hits, language=lang)
                generate_ms += float(result.elapsed_ms)
                if result.ok:
                    text = result.text
                    method = "graph_llm"
                    backend = result.backend
                    model = result.model
                else:
                    error = result.error
        else:
            method = "extractive_fallback"
            backend = "rule"
            model = "rule-based"
            error = error or ("未启用 LLM" if not use_llm else "Ollama 不可用")
        if not text:
            method = "extractive_fallback"
            backend = "rule"
            model = "rule-based"
            text = _extractive_answer(question, compose_hits)
        total_ms = round((time.perf_counter() - started) * 1000, 2)
        evidence = [_payload(h) for h in compose_hits]
        diag = dict(retrieval.get("diagnostics") or {})
        diag.update({"generation_method": method, "generation_error": error,
                     "prompt_top_k": prompt_top_k, "compose_chunks": len(compose_hits),
                     "prompt_chars": len(render_evidence(compose_hits)) if compose_hits else 0,
                     "config_fingerprint": self.cfg.fingerprint()})
        diag.update(diag_extra)
        if self.log is not None:
            self.log.log_event("graphrag.answer", trace_id=trace_id, method=method, chars=len(text),
                               refused=False, total_ms=total_ms, language=lang,
                               graph_used=bool(retrieval.get("graph_used")))
        return GraphAnswer(
            question=question, answer=text, language=lang, refused=False, generation_method=method,
            backend=backend, model=model, evidence_chunks=evidence,
            all_chunks=[_payload(h) for h in hits_raw], graph=(retrieval.get("graph") or {}) if return_graph else {},
            graph_used=bool(retrieval.get("graph_used")),
            latency_ms={"retrieve": float((retrieval.get("latency_ms") or {}).get("total") or 0.0),
                        "graph": float((retrieval.get("latency_ms") or {}).get("graph") or 0.0),
                        "generate": generate_ms, "total": total_ms},
            diagnostics=diag, trace_id=trace_id)

    # ------------------------------------------------------------------ 生成
    def _grounded_generate(self, question: str, lang: str,
                           chunks: Sequence[Mapping[str, Any]]) -> Any:
        """落地式生成：确定性草稿 → LLM 润色 → **数值原子保全护栏**（见 ``grounded.py``）。

        为什么必须用（实测教训，见 ``优化/图谱抽取优化说明.md`` §3）：
            直接「证据块 + 自由生成」时 3B 模型会改写数字 —— 实测 Q07 把证据里的
            「营业收入 137,958（百万元）· 同比 18.2%」写成「1,323.46 亿元」、
            把「拨备覆盖率 183.12%」写成 188.75%，关键词覆盖率归零。
        """
        try:
            from .grounded import generate_grounded

            return generate_grounded(question, chunks, llm=self.llm, language=lang, logger=self.log,
                                     analysis=self._analyze(question))
        except Exception as exc:  # noqa: BLE001 —— 落地链路不可用退回自由生成，绝不裸崩
            if self.log is not None:
                self.log.log_event("graphrag.grounded_generate_failed", level="WARNING",
                                   error=f"{type(exc).__name__}: {exc}",
                                   action="退回自由生成（app/graphrag/llm.py）")
            return None

    def _analyze(self, question: str) -> Any:
        """问句理解（带进程内缓存；失败返回 ``None``，调用方按「无字段词」处理）。"""
        cached = self._analysis_cache.get(question)
        if cached is not None:
            return cached
        try:
            analyzer = self._analyzer
            if analyzer is None:
                from app.core.query_understanding import analyze

                analyzer = analyze
                self._analyzer = analyzer
            bundle = self.retriever.load_bundle(logger=self.log)
            analysis = analyzer(question, corpus_issuers=bundle.corpus_issuers, logger=self.log)
            self._analysis_cache[question] = analysis
            return analysis
        except Exception as exc:  # noqa: BLE001 —— 问句理解失败不得阻断问答
            if self.log is not None:
                self.log.log_event("graphrag.analyze_failed", level="WARNING",
                                   error=f"{type(exc).__name__}: {exc}")
            return None

    # ------------------------------------------------------------------ 诊断
    def health(self) -> dict[str, Any]:
        """健康检查（不伪装未就绪能力）。"""
        import sys

        checks: dict[str, Any] = {"interpreter": {"ok": True, "python": sys.version.split()[0],
                                                  "executable": sys.executable}}
        try:
            bundle = self.retriever.load_bundle(logger=self.log)
            checks["index"] = {"ok": True, "chunks": bundle.size, "dim": bundle.vector_index.dim,
                               "path": bundle.index_dir}
        except Exception as exc:  # noqa: BLE001
            checks["index"] = {"ok": False, "reason": f"{type(exc).__name__}: {exc}",
                               "expected": str(BASE_INDEX_DIR)}
        checks["graph"] = {"ok": bool(self.graph.ready), "nodes": self.graph.node_count,
                           "edges": self.graph.edge_count, "chunks": len(self.graph.chunk_meta),
                           "dir": self.graph.source_dir, "load_ms": self.graph.load_ms}
        checks["llm"] = {"ok": bool(self.llm.available()), "model": self.llm.model,
                         "base_url": self.llm.base_url,
                         "note": "不可用时生成层自动降级为抽取式（答案仍带证据块）"}
        return {"status": "ok" if all(c.get("ok") for c in checks.values()) else "degraded",
                "checks": checks, "config": {"hop": self.cfg.retrieval.get("hop"),
                                             "top_k": self.cfg.retrieval.get("top_k"),
                                             "prompt_top_k": self.cfg.retrieval.get("prompt_top_k"),
                                             "graph_dir": str(self.cfg.graph_dir),
                                             "index_dir": str(self.cfg.index_dir)}}


def _resolve_language(question: str, requested: str) -> str:
    """语言判定（复用 工单7 的 language 模块，失败时按字符占比兜底）。"""
    try:
        from app.core.language import resolve_language

        return str(resolve_language(question, requested).resolved)
    except Exception:  # noqa: BLE001 —— 语言模块不可用不得阻断问答
        import re

        cjk = len(re.findall(r"[\u4e00-\u9fff]", question or ""))
        latin = len(re.findall(r"[A-Za-z]", question or ""))
        if str(requested).lower() in {"zh", "en"}:
            return str(requested).lower()
        return "zh" if cjk >= latin else "en"


def _extractive_answer(question: str, chunks: Sequence[Mapping[str, Any]], *, limit: int = 260) -> str:
    """抽取式兜底：选与问句词重叠度最高、且含数字的句子（**不编造**，直接给原文证据）。"""
    import re

    tokens = [t for t in re.split(r"[\s，,。、；;：:！!？?（）()【】\[\]]+", str(question or "")) if len(t) >= 2]
    best = ""
    best_score = -1.0
    for idx, chunk in enumerate(chunks[:3], 1):
        content = re.sub(r"\s+", " ", str(chunk.get("content") or ""))
        for sentence in re.split(r"[。；;]", content):
            sentence = sentence.strip()
            if len(sentence) < 8:
                continue
            score = sum(1.0 for token in tokens if token in sentence)
            score += 0.4 if re.search(r"\d", sentence) else 0.0
            score += 0.3 / idx
            if score > best_score:
                best_score = score
                best = sentence
    if not best:
        first = str((chunks[0].get("content") if chunks else "") or "")
        best = re.sub(r"\s+", " ", first)[:limit]
    return f"{best[:limit]}（抽取式兜底：Ollama 不可用，直接返回证据原文）"


class _EngineLLMAdapter:
    """把本工单的 :class:`GraphLLM` 适配成 工单7 ``generator`` 期望的 LLM 客户端接口。

    说明：当前落地式生成（``grounded.py``）已改为**自实现**，不再依赖 工单7 的私有生成链路；
    本适配器保留为**可选的互操作入口**（需要复用 工单7 生成层时可直接传入），
    因此保留但不在主链路里调用。
    """

    def __init__(self, llm: GraphLLM) -> None:
        self._llm = llm
        self.backend = _EngineBackend(llm)

    def generate_full(self, prompt: str, *, max_tokens: int = 512,
                      temperature: float = 0.2) -> Any:
        result = self._llm._chat(prompt, temperature=temperature, max_tokens=max_tokens)
        return _EngineLLMReply(text=result.text, backend=result.model, elapsed_ms=result.elapsed_ms,
                               ok=result.ok, error=result.error)


@dataclass(slots=True)
class _EngineBackend:
    """给 工单7 的生成层看的后端描述（只读属性）。"""

    llm: GraphLLM
    name: str = "ollama"
    model: str = ""
    base_url: str = ""

    def __post_init__(self) -> None:
        self.model = self.llm.model
        self.base_url = self.llm.base_url


@dataclass(slots=True)
class _EngineLLMReply:
    """``generate_full`` 的返回（字段名与 工单7 ``LLMResult`` 对齐）。"""

    text: str
    backend: str
    elapsed_ms: float
    ok: bool = True
    error: str | None = None


def _term_hits(text: str, terms: Sequence[str]) -> int:
    """文本命中查询实词的个数（**净增益闸门**的判据；只数不权重，口径简单可解释）。"""
    if not text or not terms:
        return 0
    return sum(1 for term in terms if term and term in text)


def select_evidence(hits: Sequence[Mapping[str, Any]], *, top_k: int = 3,
                    max_per_page: int = 2, prefer_text: bool = True) -> list[dict[str, Any]]:
    """挑出**进生成阶段的证据块**：**尊重检索名次** + 同页限流（表格让位给同页正文）。

    ⚠️ 三次实测迭代（都必须记住，见 ``优化/图谱抽取优化说明.md`` §3）：
        1. 最初直接取 ``hits[:3]``：Q01 的 Top-3 里有两块是 p5 的**表格**（标题「资产重组时所作承诺」），
           确定性草稿据此写成一份无关的「重大资产重组承诺」清单；
        2. 改成「文本块绝对优先」：更糟 —— 把 Q07 的第 1 名表格（含 `137,958`/`18.2%`）
           挤出去，证据里一个关键数字都没有（实测草稿变成「净额1,586.67亿元」这种无关内容）；
        3. **最终口径（本实现）**：按名次顺序取，只做两件事 ——
           ① 同一页最多 ``max_per_page`` 块；② 当第 ① 条挡住的块是**表格**、
              而后续有**同页正文块**时，让正文块顶替（正文能给出指标的完整表述）。
           非表格的名次冲突直接跳过冲突块、不改变已有次序。
    """
    picked: list[dict[str, Any]] = []
    per_page: dict[tuple[str, int], int] = {}
    need = max(int(top_k), 1)
    for hit in hits:
        if len(picked) >= need:
            break
        key = (str(hit.get("file_name") or ""), int(hit.get("page") or 0))
        if per_page.get(key, 0) < int(max_per_page):
            per_page[key] = per_page.get(key, 0) + 1
            picked.append(dict(hit))
            continue
        if not prefer_text or str(hit.get("type") or "text") != "table":
            continue
        replacement = _same_page_text(hits, key, exclude={str(h.get("chunk_id")) for h in picked})
        if replacement is not None:
            picked.append(replacement)
    if len(picked) < need:                      # 同页限流导致条数不足 ⇒ 按原名次补齐
        for hit in hits:
            if len(picked) >= need:
                break
            if any(str(h.get("chunk_id")) == str(hit.get("chunk_id")) for h in picked):
                continue
            picked.append(dict(hit))
    return picked


def _same_page_text(hits: Sequence[Mapping[str, Any]], key: tuple[str, int],
                    *, exclude: set[str]) -> dict[str, Any] | None:
    """在同一页里找第一个**正文块**（用于顶替被同页限流挡住的表格块）。"""
    for hit in hits:
        if str(hit.get("chunk_id")) in exclude:
            continue
        if str(hit.get("type") or "text") != "text":
            continue
        if (str(hit.get("file_name") or ""), int(hit.get("page") or 0)) == key:
            return dict(hit)
    return None


#: 中文停用词/疑问词（不进「净增益闸门」的实词表）
_QUERY_STOPWORDS: frozenset[str] = frozenset({
    "哪些", "哪个", "什么", "多少", "怎么", "如何", "是否", "以及", "分别", "包括", "请问",
    "说明", "分析", "结合", "下列", "以下", "关于", "对于", "他们", "它们", "这个", "那个",
    "年度报告", "年报", "报告", "公司", "集团", "银行", "请问", "根据", "以上", "如下",
    "what", "were", "which", "with", "that", "this", "from", "according", "annual", "report",
    "ratio", "bank", "group", "chinese", "china",
})


def _trade_terms(question: str, seed_names: Sequence[str], *, max_terms: int = 18) -> list[str]:
    """「净增益闸门」用的**实词表**：命中实体名 + 问句实词（jieba 分词，去疑问词/纯数字）。

    ⚠️ 实测踩坑：中文没有词边界，若用标点切分，得到的片段长 16~20 字，
    长度过滤（2~6）会把全部片段滤掉 ⇒ 实词表只剩实体名，
    甚至**整表为空**（此时闸门恒真，退化为「无条件替换」）。故必须分词后再筛。
    """
    import re as _re

    terms: list[str] = []

    def _push(token: str) -> None:
        token = str(token or "").strip()
        if not (2 <= len(token) <= 8) or token in terms or token.lower() in _QUERY_STOPWORDS:
            return
        if _re.fullmatch(r"[\d.,%]+", token):
            return
        terms.append(token)

    for name in seed_names:
        _push(name)
    try:
        import jieba

        for word in jieba.lcut(str(question or "")):
            _push(word)
    except Exception:  # noqa: BLE001 —— 分词不可用时退化为标点切分（不阻断检索）
        for piece in _re.split(r"[\s，,。、；;：:！!？?（）()【】\[\]]+", str(question or "")):
            _push(piece)
    terms.sort(key=len, reverse=True)
    return terms[:max_terms]


def _query_terms(question: str, seed_names: Sequence[str], *, max_terms: int = 24) -> list[str]:
    """问句实词 + 命中实体名（图谱独有候选的相关性闸门用；**长词优先**）。"""
    import re as _re

    terms: list[str] = []
    for name in seed_names:
        clean = str(name or "").strip()
        if len(clean) >= 3 and clean not in terms:
            terms.append(clean)
    for token in _re.split(r"[\s，,。、；;：:！!？?（）()【】\[\]]+", str(question or "")):
        token = token.strip()
        if len(token) >= 3 and token not in terms:
            terms.append(token)
    terms.sort(key=len, reverse=True)
    return terms[:max_terms]


def _analysis_digest(analysis: Any) -> dict[str, Any]:
    if analysis is None:
        return {}
    try:
        data = analysis.to_dict() if hasattr(analysis, "to_dict") else dict(analysis)
    except Exception:  # noqa: BLE001
        return {}
    return {key: data.get(key) for key in ("language", "corpus_hint", "issuer_hint", "answer_shape",
                                           "intents", "keywords") if key in data}


def load_question_set(path: str | Path) -> list[dict[str, Any]]:
    """读测试问题集（jsonl）。"""
    out: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def graph_stats_brief(store: GraphStore) -> dict[str, Any]:
    """图谱概览（供 UI 与报告引用；避免各处重复实现统计）。"""
    digest = graph_digest(store, max_nodes=1, max_edges=1)
    return {"node_total": digest["node_total"], "edge_total": digest["edge_total"],
            "chunk_total": digest["chunk_total"], "by_entity_type": digest["by_entity_type"],
            "by_relation_group": digest["by_relation_group"],
            "schema_version": digest["schema_version"]}
