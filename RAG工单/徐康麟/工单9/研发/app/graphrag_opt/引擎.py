# -*- coding: utf-8 -*-
"""工单9 优化后 Graph RAG 引擎：三层面优化的端到端编排。

工单编号：人工智能NLP-RAG-Graph RAG 优化任务

两套 profile（**同一套评估口径**，保证「优化前 vs 优化后」可比）：

* ``before`` —— 完全走 **工单8 的配置与实现**（``app.graphrag.engine.GraphRAGEngine``），
  图谱用 工单8 原始产物（9817 节点 / 4861 边），检索层与生成层都是 工单8 口径；
* ``after``  —— 本工单的三层面优化（图谱优化后产物 + 检索层融合/裁剪/扩展 + 生成层模板/护栏）。

链路（``after``）::

    问句
      → ① 查询改写（实体别名 + 字段锚点，确定性、零额外耗时）
      → ② 实体链接（图谱词典最长匹配，返回种子实体）
      → ③ 基础召回（复用 工单7：bge-m3 向量 + BM25 → RRF → 轻量重排），取 base_k=20
      → ④ 图谱扩展（自适应跳数 + 邻居按「置信度×度数倒数」剪枝 + 跨语料闸门）
      → ⑤ 融合（min-max 归一化后加权：base 0.62 / graph 0.23 / feature 0.15）
      → ⑥ 重排裁剪（裁到 14 条 → 按问句实词命中率重排 → top_k=10）
      → ⑦ 同页上下文扩展（只扩展 Top-N 命中页，且限定在已锁定语料内）
      → ⑧ 证据选取（evidence_k=8，名次×词命中的加权排序 + 同页限流）
      → ⑨ 拒答护栏 → ⑩ 模板化落地生成（快路径）/ LLM 润色（慢路径）
      → 返回：答案 + 证据块（主口径上下文）+ Top-K（附口径上下文）+ 图谱三元组 + 计时

⚠️ **公平性**：``after`` 与 ``before`` 都不使用评估集的 ``expected_pages_hint`` /
``answer_keywords``；两者的差异只在**系统自身**的三层优化上。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..graphrag.engine import apply_base_env
from ..graphrag.llm import GraphLLM
from ..graphrag.store import build_store, load_graph
from . import 检索层 as R
from . import 生成层 as G
from .配置 import OptConfig, get_config

WORK_ORDER = "人工智能NLP-RAG-Graph RAG 优化任务"


@dataclass(slots=True)
class OptAnswer:
    """一次优化后问答的完整结果（字段与 ``GraphRAGEngine`` 的 ``GraphAnswer`` 对齐，便于同口径评估）。"""

    question: str
    answer: str
    language: str = "zh"
    refused: bool = False
    generation_method: str = "grounded_template"
    backend: str = "rule"
    model: str = "template"
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
            "latency_ms": self.latency_ms, "diagnostics": self.diagnostics, "trace_id": self.trace_id,
        }


def _payload(hit: Mapping[str, Any]) -> dict[str, Any]:
    """命中 → 对外 payload（与 工单8 ``engine._payload`` 同字段，保证评估脚本可共用）。"""
    return {
        "chunk_id": str(hit.get("chunk_id") or ""),
        "page": int(hit.get("page") or 0),
        "page_basis": "1-based-physical",
        "file_name": str(hit.get("file_name") or ""),
        "type": str(hit.get("type") or hit.get("modality") or "text"),
        "text": str(hit.get("content") or hit.get("text") or ""),
        "score": float(hit.get("score") or 0.0),
        "base_score": float(hit.get("base_score") or 0.0),
        "graph_score": float(hit.get("graph_score") or 0.0),
        "graph_bonus": float(hit.get("graph_score") or 0.0),
        "graph_distance": hit.get("graph_distance"),
        "graph_entities": list(hit.get("graph_entities") or []),
        "retrieval_channel": str(hit.get("retrieval_channel") or "base"),
        "rank": int(hit.get("rank") or 0),
        "lexical_hits": int(hit.get("lexical_hits") or 0),
    }


class OptimizedEngine:
    """工单9 优化后引擎（``profile="after"``）/ 工单8 基线包装（``profile="before"``）。"""

    def __init__(self, cfg: OptConfig | None = None, *, profile: str = "after", logger: Any = None,
                 warmup: bool = False) -> None:
        self.profile = str(profile).lower()
        self.cfg = cfg or get_config(logger=logger)
        self.log = logger
        apply_base_env()
        # 日志目录钉在本工单（app/core 是从 工单7 复制的，REPO_ROOT 会算到别处）
        os.environ["RAG_LOG__DIR"] = str(self.cfg.log_dir)
        self._retriever: Any = None
        self._analyzer: Any = None
        self._analysis_cache: dict[str, Any] = {}
        self._base_engine: Any = None
        self.graph: Any = None
        self.llm: GraphLLM | None = None
        if self.profile == "before":
            from ..graphrag.engine import GraphRAGEngine

            self._base_engine = GraphRAGEngine(logger=logger)
            self.graph = self._base_engine.graph
            self.llm = self._base_engine.llm
        else:
            self.graph = self._load_opt_graph()
            self.llm = GraphLLM(self._llm_cfg(), logger=logger)
        if warmup:
            self.warmup()

    # ------------------------------------------------------------------ 装配
    def _llm_cfg(self) -> Any:
        """把优化后的 LLM 预算包装成 ``GraphLLM`` 期望的 cfg 形状。"""
        gen = dict(self.cfg.generation)

        class _Cfg:
            llm = {"base_url": self.cfg.generation.get("base_url", "http://127.0.0.1:11434"),
                   "model": self.cfg.generation.get("model", "qwen2.5:3b"),
                   "timeout_s": gen.get("llm_timeout_s", 20.0),
                   "num_ctx": gen.get("llm_num_ctx", 1536),
                   "keep_alive": gen.get("keep_alive", "0s")}

        return _Cfg()

    def _load_opt_graph(self) -> Any:
        """加载**优化后图谱**；缺失时显式降级为「纯向量+BM25」（不崩、不静默）。"""
        opt_dir = Path(self.cfg.opt_graph_dir)
        ent = opt_dir / "entities.jsonl"
        rel = opt_dir / "relations.jsonl"
        started = time.perf_counter()
        try:
            if ent.is_file() and rel.is_file():
                store = build_store(_read_jsonl(ent), _read_jsonl(rel))
                store.source_dir = str(opt_dir)
                store.load_ms = round((time.perf_counter() - started) * 1000, 2)
                return store
            if self.log is not None:
                self.log.log_event("optengine.opt_graph_missing", level="WARNING", dir=str(opt_dir),
                                   action="回退到 工单8 原始图谱（先跑 研发/scripts/构建优化图谱.py）")
        except Exception as exc:  # noqa: BLE001 —— 优化图谱损坏不得让问答崩掉
            if self.log is not None:
                self.log.log_event("optengine.opt_graph_failed", level="ERROR",
                                   error=f"{type(exc).__name__}: {exc}")
        # 回退：读 工单8 原始图谱（只读）
        try:
            from ..graphrag.config import GraphRAGConfig

            return load_graph(GraphRAGConfig.from_env(logger=self.log), logger=self.log)
        except Exception as exc:  # noqa: BLE001
            if self.log is not None:
                self.log.log_event("optengine.fallback_graph_failed", level="ERROR",
                                   error=f"{type(exc).__name__}: {exc}")
            from ..graphrag.store import GraphStore

            return GraphStore()

    @property
    def retriever(self) -> Any:
        if self._retriever is None:
            from app.core import retriever as _retriever

            self._retriever = _retriever
        return self._retriever

    def warmup(self, *, with_llm: bool = False) -> dict[str, Any]:
        """预热：加载索引束 + 图谱（可选探测 Ollama）。"""
        started = time.perf_counter()
        out: dict[str, Any] = {"profile": self.profile}
        if self._base_engine is not None:
            out.update(self._base_engine.warmup(with_llm=with_llm))
            out["profile"] = self.profile
            return out
        try:
            bundle = self.retriever.load_bundle(logger=self.log)
            out["index_chunks"] = bundle.size
            out["index_dir"] = bundle.index_dir
        except Exception as exc:  # noqa: BLE001
            out["index_error"] = f"{type(exc).__name__}: {exc}"
        out["graph"] = {"nodes": self.graph.node_count, "edges": self.graph.edge_count,
                        "ready": self.graph.ready, "load_ms": self.graph.load_ms,
                        "source_dir": self.graph.source_dir}
        if with_llm and self.llm is not None:
            out["llm_available"] = self.llm.available()
        out["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        if self.log is not None:
            self.log.log_event("optengine.warmup", **out)
        return out

    # ------------------------------------------------------------------ 辅助
    def _analyze(self, question: str) -> Any:
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
            result = analyzer(question, corpus_issuers=bundle.corpus_issuers, logger=self.log)
            self._analysis_cache[question] = result
            return result
        except Exception as exc:  # noqa: BLE001 —— 问句理解失败不得阻断问答
            if self.log is not None:
                self.log.log_event("optengine.analyze_failed", level="WARNING",
                                   error=f"{type(exc).__name__}: {exc}")
            return None

    def _corpus_key(self, file_name: str, bundle: Any) -> str:
        try:
            return str(self.retriever.corpus_key_of_file(file_name, getattr(bundle, "corpus_of_file", None)))
        except Exception:  # noqa: BLE001
            try:
                return str(self.retriever.corpus_key_of(file_name))
            except Exception:  # noqa: BLE001
                return ""

    def _resolve_language(self, question: str, requested: str) -> str:
        try:
            from app.core.language import resolve_language

            return str(resolve_language(question, requested).resolved)
        except Exception:  # noqa: BLE001
            import re

            cjk = len(re.findall(r"[\u4e00-\u9fff]", question or ""))
            latin = len(re.findall(r"[A-Za-z]", question or ""))
            if str(requested).lower() in {"zh", "en"}:
                return str(requested).lower()
            return "zh" if cjk >= latin else "en"

    # ------------------------------------------------------------------ 检索
    def retrieve(self, question: str, *, top_k: int | None = None) -> dict[str, Any]:
        """优化后检索链路（详见模块 docstring 的 ①~⑦）。"""
        started = time.perf_counter()
        ret = self.cfg.retrieval
        k = int(top_k if top_k is not None else ret.get("top_k", 10))
        base_k = int(ret.get("base_k") or max(k, 10))
        timing: dict[str, float] = {}

        # ---- ① 问句理解 + 实体链接 + 查询改写 ----
        t0 = time.perf_counter()
        bundle = None
        try:
            bundle = self.retriever.load_bundle(logger=self.log)
        except Exception as exc:  # noqa: BLE001 —— 索引不可用时如实上报
            if self.log is not None:
                self.log.log_event("optengine.index_unavailable", level="ERROR",
                                   error=f"{type(exc).__name__}: {exc}")
            return {"question": question, "hits": [], "graph": {}, "graph_used": False,
                    "error": f"索引不可用：{type(exc).__name__}: {exc}",
                    "latency_ms": {"total": round((time.perf_counter() - started) * 1000, 2)}}
        analysis = self._analyze(question)
        corpus_hint = str(getattr(analysis, "corpus_hint", "") or "") or None
        # 对比题判定：问句同时点名多家公司/含对比词 ⇒ 放宽**焦点页轮转**（不放弃语料闸门）
        compare_mode = R.is_comparison_question(question, source_files=[])
        linked = R.link_entities(question, self.graph, cfg=ret, corpus_hint=corpus_hint)
        lang = self._resolve_language(question, "auto")
        rewritten = R.rewrite_query(question, linked_names=linked.get("names") or [], cfg=ret,
                                    language=lang)
        # 评分用实词：中文提问直接用问句实词；**英文提问必须并入中文术语**。
        # ⚠️ 实测踩坑（Q13 的根因）：英文问句的实词表是 "According/Postal/net/profit…"，
        #    在中文年报块上一个都匹配不到 ⇒ `lexical_score` 全是 0 ⇒ 所有候选同分，
        #    排序退化为按 chunk_id 字典序 ⇒ 字段锚点捞回来的 p18（含 610.36亿元 / 16.52%）
        #    被字典序更小的 p104/p105 顶掉，该题上下文召回只有 0.667。
        #    并入英中映射术语（净利润/不良贷款率/拨备覆盖率…）后，中文块才能被正确打分。
        terms = list(dict.fromkeys(list(rewritten.get("terms") or [])
                                   + list(rewritten.get("added_terms") or [])))
        # **语料路由兜底（多语言）**：英文问句里没有中文公司名 ⇒ `analyze` 识别不出
        # corpus_hint ⇒ 字段锚点会在 9 份年报里乱捞。这里若原问句路由为空、
        # 而查询改写补出了中文公司名，就用**增强查询**再分析一次拿到 corpus_hint。
        if not corpus_hint and rewritten.get("added_companies"):
            extra_analysis = self._analyze(str(rewritten.get("query") or question))
            hint2 = str(getattr(extra_analysis, "corpus_hint", "") or "")
            if hint2:
                corpus_hint = hint2
                linked = R.link_entities(question, self.graph, cfg=ret, corpus_hint=corpus_hint)
        timing["link_rewrite"] = round((time.perf_counter() - t0) * 1000, 2)

        # ---- ③ 基础召回（复用 工单7）----
        # 中文题用**原问句**（与 工单8 基线同一入口，保证对比公平）；
        # 英文题用**增强查询**（附加中文术语），否则中文语料上 BM25 通道整个失效。
        t0 = time.perf_counter()
        base_hits: list[dict[str, Any]] = []
        try:
            hits = self.retriever.retrieve(str(rewritten.get("retrieval_query") or question),
                                           top_k=base_k, analysis=analysis, logger=self.log,
                                           use_cache=False)
            base_hits = [dict(h) for h in hits]
        except Exception as exc:  # noqa: BLE001
            if self.log is not None:
                self.log.log_event("optengine.base_retrieve_failed", level="ERROR",
                                   error=f"{type(exc).__name__}: {exc}")
            return {"question": question, "hits": [], "graph": {}, "graph_used": False,
                    "error": f"基础检索失败：{type(exc).__name__}: {exc}",
                    "latency_ms": {"total": round((time.perf_counter() - started) * 1000, 2)}}
        timing["base"] = round((time.perf_counter() - t0) * 1000, 2)

        # ---- ④ 图谱扩展（自适应跳数 + 邻居剪枝 + 跨语料闸门）----
        t0 = time.perf_counter()
        hop_info = R.adaptive_hop(corpus_hint=corpus_hint, linked=linked, cfg=ret)
        expansion = R.expand_pruned(self.graph, linked.get("seeds") or [], hop=int(hop_info["hop"]),
                                    cfg=ret,
                                    corpus_of_file=lambda fn: self._corpus_key(fn, bundle),
                                    corpus_hint=corpus_hint)
        graph_chunks: dict[str, dict[str, Any]] = {}
        if expansion["distances"]:
            seed_score = float(ret.get("graph_seed_score", 0.055) or 0.055)
            hop1 = float(ret.get("graph_hop1_score", 0.035) or 0.035)
            hop2 = float(ret.get("graph_hop2_score", 0.022) or 0.022)
            decay = {0: seed_score, 1: hop1, 2: hop2}
            limit_chunks = int(ret.get("max_graph_chunks") or 20)
            apply_filter = bool(ret.get("graph_corpus_filter", True)) and bool(corpus_hint)
            for node_id, distance in sorted(expansion["distances"].items(), key=lambda kv: (kv[1], kv[0])):
                node = self.graph.node(node_id)
                if node is None:
                    continue
                weight = decay.get(int(distance), hop2 * 0.8)
                for chunk in self.graph.chunks_of(node_id, limit=int(ret.get("mentions_per_entity") or 4)):
                    cid = str(chunk.get("chunk_id") or "")
                    if not cid:
                        continue
                    if apply_filter and self._corpus_key(str(chunk.get("file_name") or ""), bundle) != corpus_hint:
                        continue
                    entry = graph_chunks.get(cid)
                    if entry is None:
                        if len(graph_chunks) >= limit_chunks * 2:
                            continue
                        entry = {"chunk_id": cid, "page": int(chunk.get("page") or 0),
                                 "file_name": str(chunk.get("file_name") or ""),
                                 "type": str(chunk.get("type") or "text"),
                                 "graph_distance": int(distance), "graph_score": 0.0,
                                 "graph_entities": [], "content": str(chunk.get("snippet") or ""),
                                 "best_node_degree": 0}
                        graph_chunks[cid] = entry
                    entry["graph_distance"] = min(int(entry["graph_distance"]), int(distance))
                    entry["graph_score"] = round(float(entry["graph_score"]) + weight, 6)
                    entry["best_node_degree"] = max(int(entry["best_node_degree"]),
                                                    len(self.graph.adjacency.get(node_id, [])))
                    if node.get("name") not in entry["graph_entities"]:
                        entry["graph_entities"].append(str(node.get("name")))
        timing["graph"] = round((time.perf_counter() - t0) * 1000, 2)

        # ---- ⑤ 组装候选 + 融合 ----
        t0 = time.perf_counter()
        rows: list[dict[str, Any]] = []
        for hit in base_hits:
            cid = str(hit.get("chunk_id"))
            info = graph_chunks.get(cid)
            row = dict(hit)
            row["base_score"] = float(hit.get("score") or hit.get("final_score") or 0.0)
            row["graph_score"] = float(info["graph_score"]) if info else 0.0
            row["graph_distance"] = int(info["graph_distance"]) if info else None
            row["graph_entities"] = list(info["graph_entities"]) if info else []
            row["retrieval_channel"] = "base+graph" if info else "base"
            row["best_node_degree"] = int(info["best_node_degree"]) if info else 0
            feats = R.graph_features(distance=row["graph_distance"], entity_count=len(row["graph_entities"]),
                                     node_degree=row["best_node_degree"], cfg=ret)
            row.update(feats)
            rows.append(row)
        # 图谱独有候选（带相关性闸门：必须命中至少一个问句实词/实体名）
        linked_names = [str(n) for n in (linked.get("names") or [])]
        gate_terms = list(dict.fromkeys(terms + linked_names))
        for cid, info in graph_chunks.items():
            if any(str(r.get("chunk_id")) == cid for r in rows):
                continue
            text = str(info.get("content") or "")
            row = self._chunk_row(cid, bundle)
            if row is None:
                continue
            text = str(row.get("content") or "") or text
            if gate_terms and not any(t in text for t in gate_terms):
                continue
            row["base_score"] = 0.0
            row["graph_score"] = float(info["graph_score"])
            row["graph_distance"] = int(info["graph_distance"])
            row["graph_entities"] = list(info["graph_entities"])
            row["retrieval_channel"] = "graph_only"
            row["best_node_degree"] = int(info["best_node_degree"])
            row.update(R.graph_features(distance=row["graph_distance"],
                                        entity_count=len(row["graph_entities"]),
                                        node_degree=row["best_node_degree"], cfg=ret))
            rows.append(row)
        fused = R.fuse_scores(rows, cfg=ret)
        timing["fusion"] = round((time.perf_counter() - t0) * 1000, 2)

        # ---- ⑥ 重排裁剪 ----
        t0 = time.perf_counter()
        reranked = R.rerank_trim(fused, terms=terms, top_k=k, cfg=ret)
        top = reranked["hits"]
        timing["rerank"] = round((time.perf_counter() - t0) * 1000, 2)

        # ---- ⑦ 同页上下文扩展 ----
        t0 = time.perf_counter()
        expanded = R.page_expand(top, bundle=bundle, cfg=ret,
                                 corpus_of_file=lambda fn: self._corpus_key(fn, bundle),
                                 corpus_hint=corpus_hint)
        timing["page_expand"] = round((time.perf_counter() - t0) * 1000, 2)

        # ---- ⑦' 字段锚点召回（按问句里的财务字段词定点补块）----
        # ⚠️ 实测记录：曾对「对比题」放开字段锚点的语料闸门（`probe_hint=None`），
        #   想让 Q09 取到招商银行的 2,697.03 —— 结果锚点从 9 份年报里各捞一块，
        #   全局上下文精度从 0.8025 掉到 0.7395（净收益为负），故**保留语料闸门**。
        t0 = time.perf_counter()
        probe_hint = corpus_hint
        probed = R.field_probe(question, bundle, self._field_index(bundle), cfg=ret, terms=terms,
                               corpus_of_file=lambda fn: self._corpus_key(fn, bundle),
                               corpus_hint=probe_hint, anchor_hits=top)
        timing["field_probe"] = round((time.perf_counter() - t0) * 1000, 2)

        # ---- ⑦'' 锚点页兄弟块扩展 ----
        # 字段锚点定位到某一页（如邮储银行 p18 的财务摘要）后，把**同页其余块**一并取回。
        # 依据（实测 Q13）：p18 共 2 块，锚点只带回 p18-x0（含净利润 610.36亿元 / 16.52%），
        # 而 `81.80亿元` / `2.96%` 在 **p18-x1**，缺了它该题召回只有 0.667。
        t0 = time.perf_counter()
        probe_siblings = (R.page_expand(probed.get("extra") or [], bundle=bundle, cfg=ret,
                                        corpus_of_file=lambda fn: self._corpus_key(fn, bundle),
                                        corpus_hint=probe_hint, channel="probe_expand")
                          if bool(ret.get("probe_expand", False))
                          else {"extra": [], "pages": [], "skipped": "probe_expand 关闭（实测净收益为负）"})
        timing["probe_expand"] = round((time.perf_counter() - t0) * 1000, 2)

        total_ms = round((time.perf_counter() - started) * 1000, 2)
        diagnostics = {
            "profile": self.profile,
            "corpus_hint": corpus_hint,
            "compare_mode": bool(compare_mode),
            "linked_entities": linked.get("matched") or [],
            "linked_names": linked_names[:12],
            "rewrite_added_names": rewritten.get("added_names"),
            "rewrite_added_fields": rewritten.get("added_fields"),
            "rewrite_added_terms": rewritten.get("added_terms"),
            "rewrite_added_companies": rewritten.get("added_companies"),
            "base_query_rewritten": bool(rewritten.get("use_rewrite_for_base")),
            "hop": int(hop_info["hop"]), "hop_reason": hop_info["reason"],
            "graph_nodes": len(expansion["distances"]),
            "graph_chunks": len(graph_chunks),
            "graph_filtered_cross_corpus": int(expansion.get("filtered_cross_corpus") or 0),
            "base_candidates": len(base_hits),
            "fused_candidates": len(fused),
            "trim_count": reranked["trim_count"],
            "page_expand_pages": expanded.get("pages"),
            "page_expand_extra": len(expanded.get("extra") or []),
            "field_probe_fields": probed.get("fields"),
            "field_probe_extra": len(probed.get("extra") or []),
            "field_probe_note": probed.get("skipped"),
            "probe_expand_pages": probe_siblings.get("pages"),
            "probe_expand_extra": len(probe_siblings.get("extra") or []),
            "fusion_weights": dict(ret.get("fusion_weights") or {}),
            "fusion_summary": R.summarize_fusion(top),
            "timing": timing,
            "total_ms": total_ms,
        }
        if self.log is not None:
            self.log.log_event("optengine.retrieve", engine_profile=self.profile,
                               question_digest=str(question)[:60],
                               **{k: v for k, v in diagnostics.items()
                                  if k not in ("linked_entities", "fusion_summary", "profile")})
        all_raw = (top + list(expanded.get("extra") or []) + list(probed.get("extra") or [])
                   + list(probe_siblings.get("extra") or []))
        return {"question": question, "hits": [_payload(h) for h in all_raw],
                "hits_top": [_payload(h) for h in top],
                "hits_raw": all_raw,
                "top_raw": top,
                "page_expand_raw": list(expanded.get("extra") or []),
                "field_probe_raw": list(probed.get("extra") or []),
                "probe_expand_raw": list(probe_siblings.get("extra") or []),
                # **评分用实词**（含英文题并入的中文术语）要一路带到生成层做证据排序。
                # ⚠️ 实测踩坑：只在 `retrieve()` 内部合并、不往上传时，`answer()` 里
                # 重新用 `query_terms(question)` 得到的是**纯英文**实词表 ——
                # 中文证据句的 `lex` 全为 0 ⇒ 打分并列 ⇒ 按证据名次决胜 ⇒
                # 焦点页（邮储 p8/p15 贷款集中度）压过真正含「净利润 610.36亿元」的锚点块，
                # 英文答案因此答非所问。
                "terms": terms,
                "graph": self._local_graph(linked, expansion),
                "graph_used": bool(linked.get("seeds")),
                "analysis": _analysis_digest(analysis),
                "latency_ms": {**timing, "total": total_ms},
                "diagnostics": diagnostics}

    def _field_index(self, bundle: Any) -> dict[str, list[str]]:
        """字段倒排索引（**引擎级缓存**：只在第一次检索时构建一次）。"""
        cached = getattr(self, "_field_index_cache", None)
        if cached is None:
            started = time.perf_counter()
            cached = R.build_field_index(bundle)
            self._field_index_cache = cached
            if self.log is not None:
                self.log.log_event("optengine.field_index_built", terms=len(cached),
                                   chunks=sum(len(v) for v in cached.values()),
                                   elapsed_ms=round((time.perf_counter() - started) * 1000, 2))
        return cached

    def _chunk_row(self, chunk_id: str, bundle: Any) -> dict[str, Any] | None:
        try:
            return bundle.chunk(str(chunk_id))
        except Exception as exc:  # noqa: BLE001
            if self.log is not None:
                self.log.log_event("optengine.chunk_lookup_failed", level="WARNING",
                                   chunk_id=chunk_id, error=f"{type(exc).__name__}: {exc}")
            return None

    def _local_graph(self, linked: Mapping[str, Any], expansion: Mapping[str, Any]) -> dict[str, Any]:
        distances: Mapping[str, int] = expansion.get("distances") or {}
        seed_names = list(linked.get("names") or [])
        nodes = []
        for node_id, distance in list(distances.items())[:40]:
            node = self.graph.node(node_id)
            if node is None:
                continue
            nodes.append({"id": node_id, "name": node.get("name"), "type": node.get("type"),
                          "type_zh": node.get("type_zh"), "distance": int(distance),
                          "degree": len(self.graph.adjacency.get(node_id, []))})
        edges = [{"head": e.get("head"), "relation": e.get("relation"), "tail": e.get("tail"),
                  "confidence": e.get("confidence")} for e in (expansion.get("edges_used") or [])[:60]]
        return {"seed_names": seed_names, "distances": dict(distances),
                "distances_named": {self.graph.name_of(k): int(v) for k, v in list(distances.items())[:40]},
                "nodes": nodes, "edges": edges, "edge_total_used": len(expansion.get("edges_used") or [])}

    # ------------------------------------------------------------------ 问答
    def answer(self, question: str, *, top_k: int | None = None, language: str = "auto",
               use_llm: bool = True) -> Any:
        """端到端问答。``before`` profile 直接委派给 工单8 引擎。"""
        if self._base_engine is not None:
            return self._base_engine.answer(question, top_k=top_k, use_graph=True,
                                            language=language, use_llm=use_llm)
        gen = self.cfg.generation
        trace_id = f"o{int(time.time() * 1000) % 10 ** 10:010d}"
        started = time.perf_counter()
        lang = self._resolve_language(question, language)
        retrieval = self.retrieve(question, top_k=top_k)
        if retrieval.get("error"):
            lat = float((retrieval.get("latency_ms") or {}).get("total") or 0.0)
            return OptAnswer(question=question, answer=f"检索失败：{retrieval['error']}", language=lang,
                             refused=True, generation_method="error", backend="rule", model="rule-based",
                             graph=retrieval.get("graph") or {}, trace_id=trace_id,
                             latency_ms={"retrieve": lat, "generate": 0.0, "total": lat},
                             diagnostics=dict(retrieval.get("diagnostics") or {}))

        # 评分用实词：**必须用检索层算好的那份**（英文题已并入中文术语映射）。
        terms = list(retrieval.get("terms") or R.query_terms(question))
        top_raw = list(retrieval.get("top_raw") or [])
        extra_raw = (list(retrieval.get("page_expand_raw") or [])
                     + list(retrieval.get("field_probe_raw") or [])
                     + list(retrieval.get("probe_expand_raw") or []))
        # ---- ⑧ 证据选取：Top-K / 同页扩展块 / 字段锚点块一起参与 ----
        evidence = R.select_evidence(top_raw + extra_raw, terms=terms, cfg=self.cfg.retrieval,
                                     corpus_of_file=lambda fn: self._corpus_key(
                                         fn, self.retriever.load_bundle(logger=None)),
                                     corpus_hint=str((retrieval.get("diagnostics") or {}).get("corpus_hint") or ""),
                                     compare_mode=bool((retrieval.get("diagnostics") or {}).get("compare_mode")),
                                     language=lang)
        triples = G.build_triples_block(retrieval.get("graph") or {}, cfg=gen)

        # ---- ⑨ 拒答护栏 ----
        guard = G.refusal_guard(question, evidence, terms=terms, cfg=gen)
        generate_ms = 0.0
        method = "grounded_template"
        backend, model = "rule", "template"
        detail: dict[str, Any] = {"refusal_guard": guard, "triples": triples.get("count", 0)}
        if guard["refuse"]:
            text = G.refusal_text(lang)
            method, backend, model = "refusal_guard", "rule", "rule-based"
        else:
            # ---- ⑩ 模板化落地生成（快路径）----
            try:
                from ..graphrag.grounded import build_draft

                draft = build_draft(question, evidence, analysis=self._analyze(question))
            except Exception as exc:  # noqa: BLE001 —— 草稿失败退回证据原文，绝不裸崩
                draft = ""
                detail["draft_error"] = f"{type(exc).__name__}: {exc}"
            conf = G.template_confidence(question, evidence, draft=draft, cfg=gen)
            detail["template_confidence"] = conf
            composed = G.compose_answer(question, evidence, draft=draft, triples=triples,
                                        language=lang, cfg=gen, terms=terms)
            if conf["ok"] and composed["text"]:
                text = composed["text"]
                method, backend, model = "grounded_template", "rule", "template"
                detail["template"] = {"data_sentences": composed.get("data_sentences"),
                                      "triples_used": composed.get("triples_used")}
            else:
                # ---- 慢路径：LLM 润色（预算已压缩）----
                llm_out = G.llm_polish(question, evidence, llm=self.llm, triples=triples,
                                       draft=draft, language=lang, cfg=gen) if use_llm else {
                    "ok": False, "text": "", "elapsed_ms": 0.0, "error": "use_llm=False"}
                generate_ms += float(llm_out.get("elapsed_ms") or 0.0)
                if llm_out.get("ok"):
                    text = str(llm_out["text"])
                    method, backend, model = "llm_polished", "ollama", str(self.llm.model)
                else:
                    text = composed["text"] or draft or "现有证据不足以形成答案。"
                    method, backend, model = "draft_fallback", "rule", "rule-based"
                    detail["llm_error"] = llm_out.get("error")
                detail["llm"] = {k: v for k, v in llm_out.items() if k != "text"}

        total_ms = round((time.perf_counter() - started) * 1000, 2)
        evidence_payload = [_payload(h) for h in evidence]
        diag = dict(retrieval.get("diagnostics") or {})
        diag.update({"generation_method": method, "config_fingerprint": self.cfg.fingerprint(),
                     "evidence_count": len(evidence), "prompt_chars": len(G.render_evidence(
                         evidence, max_chars=int(gen.get("llm_evidence_chars") or 220))),
                     **detail})
        if self.log is not None:
            self.log.log_event("optengine.answer", trace_id=trace_id, method=method,
                               total_ms=total_ms, language=lang, refused=bool(guard["refuse"]),
                               evidence=len(evidence))
        return OptAnswer(
            question=question, answer=text, language=lang, refused=bool(guard["refuse"]),
            generation_method=method, backend=backend, model=model,
            evidence_chunks=evidence_payload,
            all_chunks=list(retrieval.get("hits") or []),
            graph=retrieval.get("graph") or {}, graph_used=bool(retrieval.get("graph_used")),
            latency_ms={"retrieve": float((retrieval.get("latency_ms") or {}).get("total") or 0.0),
                        "graph": float((retrieval.get("latency_ms") or {}).get("graph") or 0.0),
                        "generate": generate_ms, "total": total_ms},
            diagnostics=diag, trace_id=trace_id)

    # ------------------------------------------------------------------ 健康检查
    def health(self) -> dict[str, Any]:
        """健康检查（不伪装未就绪能力）。"""
        import sys

        checks: dict[str, Any] = {"profile": self.profile,
                                  "interpreter": {"ok": True, "python": sys.version.split()[0],
                                                  "executable": sys.executable}}
        if self._base_engine is not None:
            return self._base_engine.health()
        try:
            bundle = self.retriever.load_bundle(logger=self.log)
            checks["index"] = {"ok": True, "chunks": bundle.size, "dim": bundle.vector_index.dim,
                               "path": bundle.index_dir}
        except Exception as exc:  # noqa: BLE001
            checks["index"] = {"ok": False, "reason": f"{type(exc).__name__}: {exc}",
                               "expected": str(self.cfg.index_dir)}
        checks["graph"] = {"ok": bool(self.graph.ready), "nodes": self.graph.node_count,
                           "edges": self.graph.edge_count, "dir": self.graph.source_dir,
                           "load_ms": self.graph.load_ms}
        checks["llm"] = {"ok": bool(self.llm is not None and self.llm.available()),
                         "model": getattr(self.llm, "model", ""), "note": "不可用时生成层走模板快路径"}
        checks["config"] = {"fingerprint": self.cfg.fingerprint(), "source": self.cfg.config_source}
        ok_keys = [k for k in ("index", "graph") if isinstance(checks.get(k), Mapping)]
        return {"status": "ok" if all(checks[k].get("ok") for k in ok_keys) else "degraded",
                "checks": checks}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _analysis_digest(analysis: Any) -> dict[str, Any]:
    if analysis is None:
        return {}
    try:
        data = analysis.to_dict() if hasattr(analysis, "to_dict") else dict(analysis)
    except Exception:  # noqa: BLE001
        return {}
    return {key: data.get(key) for key in ("language", "corpus_hint", "issuer_hint", "answer_shape",
                                           "intents", "keywords") if key in data}


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    """读评估题集（jsonl）。"""
    return _read_jsonl(Path(path))
