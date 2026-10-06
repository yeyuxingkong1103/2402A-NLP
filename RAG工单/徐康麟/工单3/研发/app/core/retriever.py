# -*- coding: utf-8 -*-
"""工单3 混合检索（设计/接口设计.md §2.3、§3.13 冻结）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

链路：``向量召回 + 自实现 BM25`` → **按 file_name 硬过滤** → **RRF 融合** → 加权（表格/数字/关键词）
→ 确定性重排 → ``top_k``（默认 5）；数字类问题且结果无表块时追加**表块兜底检索**。

热路径纪律（实测依据）：
    * ``jieba`` 冷启动 ≈ 1.06 s、``embedder`` 冷启动 ≈ 0.1~0.19 s → **``load()`` 内显式预热**
      （``text_utils.warmup_tokenizer()`` + ``embedder.warmup()``），
      否则第一个问题的首字延迟会白吃约 1 s，直接威胁「首字 ≤ 3 s」验收项；
    * 重排默认走确定性启发式（不走 LLM），保证可复现与预算。
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import bm25_index, embedder, reranker, retrieval_utils, text_utils, vector_store
from .chunker import Chunk
from .config import AppConfig, get_config, model_slug
from .errors import IndexMissingError, RetrievalError, wrap

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"


# ---------------------------------------------------------------------------
# 数据结构（设计 §2.3 冻结）
# ---------------------------------------------------------------------------
# t21（§24）：招股书「发行人基本情况」类**多字段结构化块**的字段名（通用清单，非某题特判）。
# 用途：字段型问题补「结构化支持块」时，优先挑「同时含多个基本情况字段」的块（发行人自己的基本情况），
# 而不是只提到一次该字段的简历/声明块（实测：轮 2 被 p255 简历块带偏成「法定代表人为程勇波」）。
STRUCTURED_BASIC_FIELDS: tuple[str, ...] = (
    "公司名称", "中文名称", "英文名称", "注册资本", "法定代表人", "注册地址", "成立日期", "实收资本",
    "股本总额", "股份总数", "统一社会信用代码", "办公地址", "联系电话", "经营范围",
)


@dataclass(slots=True)
class RetrievedChunk:
    """一条召回结果（含各阶段分数与加权系数，供 UI 展示与测试断言）。"""

    chunk_id: str
    file_name: str
    page: int
    type: str
    content: str
    section: str
    table_id: str | None
    score: float
    vector_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    boosts: dict[str, float] = field(default_factory=dict)
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalResult:
    """一次检索的完整结果与阶段耗时。"""

    query: str
    normalized_query: str
    rewritten_query: str | None
    file_names: list[str] | None
    top_k: int
    chunks: list[RetrievedChunk]
    stages: dict[str, float]
    candidates_count: int
    table_fallback_used: bool
    trace_id: str
    # T7（t12）新增：数值锚定支持块 —— 只进**生成上下文**（排在最前），不参与 top-k 排名，
    # 因此 T5 的召回判据（证据必须在返回 top-5）语义不变。默认空列表，向后兼容。
    support_chunks: list[RetrievedChunk] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "normalized_query": self.normalized_query,
            "rewritten_query": self.rewritten_query,
            "file_names": self.file_names,
            "top_k": self.top_k,
            "chunks": [c.to_dict() for c in self.chunks],
            "support_chunks": [c.to_dict() for c in self.support_chunks],
            "stages": self.stages,
            "candidates_count": self.candidates_count,
            "table_fallback_used": self.table_fallback_used,
            "trace_id": self.trace_id,
        }


@dataclass(slots=True)
class EvidenceHitReport:
    """命中判定报告（判据 = 证据原文是否落在返回块中）。"""

    question_id: int
    top_k: int
    hit: bool
    matched_chunk_ids: list[str]
    missing_pages: list[int]
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _lazy_logger(logger: Any, module: str = "retriever") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


def _new_trace_id() -> str:
    """短 trace_id（时间戳 + 进程内计数）。"""
    return f"t{int(time.perf_counter() * 1000) % 100000000:08d}"


class HybridRetriever:
    """向量 + BM25 混合检索器（UI/评估的唯一检索入口）。"""

    def __init__(self, *, index_dir: Path | str | None = None, cfg: AppConfig | None = None,
                 logger: Any = None) -> None:
        self.cfg = cfg or get_config()
        self.log = _lazy_logger(logger)
        self.index_root = Path(index_dir) if index_dir is not None else self.cfg.paths.index_dir
        self.model_dir: Path | None = None
        self.vector_index: vector_store.LoadedVectorIndex | None = None
        self.bm25: bm25_index.BM25Index | None = None
        self.chunk_map: dict[str, Chunk] = {}
        self.warmup_info: dict[str, Any] = {}
        self.loaded = False

    # -- 加载 ------------------------------------------------------------
    def _resolve_model_dir(self) -> Path:
        """定位索引目录：给的是模型目录就直接用；给索引根则自动找含 ids.json 的子目录。"""
        if (self.index_root / vector_store.IDS_FILE).is_file():
            return self.index_root
        candidates = sorted(p for p in self.index_root.glob("*") if (p / vector_store.IDS_FILE).is_file())
        if not candidates:
            raise IndexMissingError(
                f"索引目录下找不到任何含 {vector_store.IDS_FILE} 的子目录：{self.index_root}",
                detail={"index_root": str(self.index_root)},
            )
        preferred = [p for p in candidates if p.name == model_slug(self.cfg.llm.ollama_embed_model, 1024)]
        return preferred[0] if preferred else candidates[0]

    def _load_chunks(self) -> dict[str, Chunk]:
        """加载块元数据：优先 SQLite ``chunks`` 表，缺失时回退 chunks.jsonl（显式留痕）。"""
        db_path = self.cfg.paths.index_dir / "rag.sqlite3"
        if db_path.is_file():
            try:
                conn = sqlite3.connect(str(db_path))
                try:
                    rows = conn.execute(
                        "SELECT chunk_id,file_name,page,page_start,page_end,type,section,content,keywords,"
                        "table_id,char_count,ord,source_block_ids FROM chunks"
                    ).fetchall()
                finally:
                    conn.close()
                chunk_map: dict[str, Chunk] = {}
                for row in rows:
                    chunk_map[row[0]] = Chunk(
                        chunk_id=row[0], file_name=row[1], page=int(row[2]), page_start=int(row[3]),
                        page_end=int(row[4]), type=row[5], section=row[6] or "", content=row[7] or "",
                        keywords=list(json.loads(row[8]) if row[8] else []), table_id=row[9],
                        char_count=int(row[10]), order=int(row[11]),
                        source_block_ids=list(json.loads(row[12]) if row[12] else []),
                    )
                self.log.log_event("retriever.chunks_loaded", source="sqlite", count=len(chunk_map),
                                   db=str(db_path))
                return chunk_map
            except (sqlite3.Error, ValueError, TypeError) as exc:
                self.log.log_event("retriever.chunks_sqlite_failed", level="WARNING",
                                   error_type=type(exc).__name__, message=str(exc),
                                   degrade="回退 chunks.jsonl")
        jsonl = self.cfg.paths.processed_dir / "chunks.jsonl"
        if not jsonl.is_file():
            raise IndexMissingError(f"块元数据缺失：既无 {db_path} 也无 {jsonl}",
                                    detail={"db": str(db_path), "jsonl": str(jsonl)})
        chunk_map = {c.chunk_id: c for c in
                     (Chunk.from_dict(row) for row in text_utils.read_jsonl(jsonl))}
        self.log.log_event("retriever.chunks_loaded", source="jsonl", count=len(chunk_map), path=str(jsonl))
        return chunk_map

    def load(self, *, warmup: bool = True) -> "HybridRetriever":
        """加载向量索引 + BM25 + 块元数据，并**预热分词器与嵌入后端**（在线热路径必需）。"""
        with self.log.enter("HybridRetriever.load", {"index_root": str(self.index_root),
                                                     "warmup": warmup}) as span:
            started = time.perf_counter()
            self.model_dir = self._resolve_model_dir()
            self.vector_index = vector_store.load_vector_index(
                self.model_dir, logger=self.log, use_faiss=self.cfg.vector_store.use_faiss)
            self.bm25 = bm25_index.BM25Index.load(self.model_dir, logger=self.log)
            self.chunk_map = self._load_chunks()
            if warmup:
                # ① 分词器：jieba 冷启动 ≈1.06 s；② 嵌入：冷启动 ≈0.1~0.19 s
                tokenizer_info = text_utils.warmup_tokenizer(logger=self.log)
                embedding_info = embedder.warmup(self.cfg, logger=self.log)
                self.warmup_info = {"tokenizer": tokenizer_info, "embedding": embedding_info}
                self.log.log_event("retriever.warmed_up", **self.warmup_info)
            self.loaded = True
            span.set_output({"model_dir": str(self.model_dir), "chunks": len(self.chunk_map),
                             "vectors": self.vector_index.count, "bm25": self.bm25.size(),
                             "warmup": self.warmup_info,
                             "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)})
            return self

    def _ensure_loaded(self) -> None:
        if not self.loaded:
            self.load()

    # -- 检索 ------------------------------------------------------------
    def _search_one_route(self, query: str, allowed_ids: set[str] | None, stages: dict[str, float],
                          *, vector_k: int, bm25_k: int, query_vector: Any | None) -> tuple[
                              list[tuple[str, float]], list[tuple[str, float]]]:
        """跑向量与 BM25 两路（向量失败时显式降级为 BM25-only）。"""
        vector_hits: list[tuple[str, float]] = []
        bm25_hits: list[tuple[str, float]] = []
        if query_vector is not None and self.vector_index is not None:
            t0 = time.perf_counter()
            try:
                vector_hits = vector_store.search_vectors(self.vector_index, query_vector, vector_k,
                                                          allowed_ids=allowed_ids, logger=self.log)
            except Exception as exc:  # noqa: BLE001 —— 向量路失败不得阻断 BM25 路，必须留痕
                self.log.log_event("retriever.vector_failed", level="ERROR",
                                   error_type=type(exc).__name__, message=str(exc),
                                   degrade="本次检索仅用 BM25")
                vector_hits = []
            stages["vector_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        t0 = time.perf_counter()
        if self.bm25 is not None:
            bm25_hits = self.bm25.search(query, bm25_k, allowed_ids=allowed_ids, logger=self.log)
        stages["bm25_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return vector_hits, bm25_hits

    def retrieve(self, query: str, *, top_k: int | None = None,
                 file_names: Sequence[str] | None = None,
                 rewritten_query: str | None = None,
                 trace_id: str | None = None, logger: Any = None) -> RetrievalResult:
        """混合检索主入口（语义见设计 §3.13）。"""
        log = logger or self.log
        self._ensure_loaded()
        cfg = self.cfg
        effective_top_k = int(top_k or cfg.retrieval.top_k)
        effective_query = (rewritten_query or query).strip() or query
        trace = trace_id or _new_trace_id()
        stages: dict[str, float] = {}
        started = time.perf_counter()
        with log.enter("HybridRetriever.retrieve",
                       {"query": query, "rewritten_query": rewritten_query, "top_k": effective_top_k,
                        "file_names": list(file_names) if file_names else None, "trace_id": trace}) as span:
            log.log_event("retrieval.start", trace_id=trace, query_digest=text_utils.text_digest(query, limit=80),
                          normalized_query=effective_query[:80], file_names=list(file_names) if file_names else None,
                          top_k=effective_top_k)
            try:
                allowed_ids = retrieval_utils.filter_ids_by_file(self.chunk_map, file_names)
                if allowed_ids is not None and not allowed_ids:
                    log.log_event("retrieval.empty_file_filter", level="WARNING", file_names=list(file_names),
                                  reason="所选文件在索引中没有任何块 → 返回空结果（不做退化为全库检索）")
                    result = RetrievalResult(query=query, normalized_query=effective_query,
                                             rewritten_query=rewritten_query, file_names=list(file_names),
                                             top_k=effective_top_k, chunks=[], stages=stages,
                                             candidates_count=0, table_fallback_used=False, trace_id=trace)
                    span.set_output({"chunks": 0, "reason": "file filter empty"})
                    return result

                # ① 查询向量（失败→BM25-only，不静默）
                query_vector = None
                t0 = time.perf_counter()
                try:
                    query_vector = embedder.embed_query(effective_query, cfg=cfg, logger=log)
                except Exception as exc:  # noqa: BLE001 —— 嵌入不可用时显式降级
                    log.log_event("retrieval.embed_failed", level="ERROR",
                                  error_type=type(exc).__name__, message=str(exc),
                                  degrade="本次检索退化为 BM25-only")
                stages["embed_ms"] = round((time.perf_counter() - t0) * 1000, 2)

                # ② 两路召回
                vector_hits, bm25_hits = self._search_one_route(
                    effective_query, allowed_ids, stages,
                    vector_k=cfg.retrieval.vector_k, bm25_k=cfg.retrieval.bm25_k,
                    query_vector=query_vector,
                )
                log.log_event("retrieval.stage", stage="recall", vector=len(vector_hits),
                              bm25=len(bm25_hits), filtered=allowed_ids is not None,
                              vector_ms=stages.get("vector_ms"), bm25_ms=stages.get("bm25_ms"))

                # ③ RRF 融合
                t0 = time.perf_counter()
                fused = retrieval_utils.rrf_fuse(
                    vector_hits, bm25_hits, k=cfg.retrieval.rrf_k,
                    weights=(cfg.retrieval.vector_weight, cfg.retrieval.bm25_weight),
                    rescue_weight=cfg.retrieval.rank_rescue_weight,
                )
                stages["fusion_ms"] = round((time.perf_counter() - t0) * 1000, 2)

                # ④ 加权（表格/数字/关键词）
                query_tokens = text_utils.tokenize(effective_query)
                # 数字加权仅在「问题本身在问数字」时生效（否则会把无数字的正文证据挤出 top-5，T5 实测）
                expects_numeric = (text_utils.has_numeric_signal(effective_query)
                                   if cfg.retrieval.numeric_boost_require_numeric_query else None)
                boosted = retrieval_utils.apply_boosts(
                    fused, self.chunk_map,
                    table_boost=cfg.retrieval.table_boost, numeric_boost=cfg.retrieval.numeric_boost,
                    keyword_boost=cfg.retrieval.keyword_boost, query_tokens=query_tokens,
                    query_expects_numeric=expects_numeric,
                )
                source_scores = {
                    cid: {"rrf_score": rrf, "vector_score": dict(vector_hits).get(cid),
                          "bm25_score": dict(bm25_hits).get(cid)}
                    for cid, rrf in fused
                }

                # ⑤ 重排
                t0 = time.perf_counter()
                chunks = reranker.rerank(effective_query, boosted, self.chunk_map,
                                         top_k=effective_top_k, cfg=cfg, source_scores=source_scores,
                                         expects_numeric=bool(expects_numeric), logger=log)
                stages["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 2)

                # ⑥ 表块兜底（数字类问题）
                fallback_used = False
                if not any(c.type == "table" for c in chunks) and reranker.needs_table_retrieval(
                        effective_query):
                    table_ids = {cid for cid, c in self.chunk_map.items() if c.type == "table"}
                    if allowed_ids is not None:
                        table_ids &= allowed_ids
                    if table_ids:
                        t0 = time.perf_counter()
                        t_vector, t_bm25 = self._search_one_route(
                            effective_query, table_ids, {},
                            vector_k=cfg.retrieval.vector_k, bm25_k=cfg.retrieval.bm25_k,
                            query_vector=query_vector,
                        )
                        t_fused = retrieval_utils.rrf_fuse(t_vector, t_bm25, k=cfg.retrieval.rrf_k,
                                                           weights=(cfg.retrieval.vector_weight,
                                                                    cfg.retrieval.bm25_weight),
                                                           rescue_weight=cfg.retrieval.rank_rescue_weight)
                        merged = retrieval_utils.merge_unique(
                            [(c.chunk_id, c.score) for c in chunks], t_fused, top_k=max(effective_top_k * 3, 10))
                        boosted_all = retrieval_utils.apply_boosts(
                            merged, self.chunk_map, table_boost=cfg.retrieval.table_boost,
                            numeric_boost=cfg.retrieval.numeric_boost,
                            keyword_boost=cfg.retrieval.keyword_boost, query_tokens=query_tokens,
                            query_expects_numeric=expects_numeric)
                        chunks = reranker.rerank(effective_query, boosted_all, self.chunk_map,
                                                 top_k=effective_top_k, cfg=cfg,
                                                 source_scores=source_scores,
                                                 expects_numeric=bool(expects_numeric), logger=log)
                        fallback_used = True
                        stages["table_fallback_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                        log.log_event("retrieval.table_fallback", reason="结果无表块且问题似数字类",
                                      added=len(t_fused), table_candidates=len(table_ids))

                stages["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
                support = self.numeric_support_chunks(effective_query, chunks, allowed_ids=allowed_ids,
                                                      query_vector=query_vector, stages=stages, logger=log)
                # t21（§24）：字段型问题再补「字段结构化支持块」（同一 support_chunks 机制；去重后总数上限 3）
                field_support = self.field_support_chunks(effective_query, chunks, allowed_ids=allowed_ids,
                                                          logger=log)
                seen_ids = {str(getattr(c, "chunk_id", "")) for c in support}
                for chunk in field_support:
                    chunk_id = str(getattr(chunk, "chunk_id", ""))
                    if chunk_id and chunk_id not in seen_ids:
                        support.append(chunk)
                        seen_ids.add(chunk_id)
                support = support[:3]
                result = RetrievalResult(
                    query=query, normalized_query=effective_query, rewritten_query=rewritten_query,
                    file_names=list(file_names) if file_names else None, top_k=effective_top_k,
                    chunks=chunks, stages=stages, candidates_count=len(fused),
                    table_fallback_used=fallback_used, trace_id=trace, support_chunks=support,
                )
                log.log_event("retrieval.done", trace_id=trace, chunk_ids=[c.chunk_id for c in chunks],
                              pages=[c.page for c in chunks], scores=[c.score for c in chunks],
                              stages=stages, candidates=len(fused), table_fallback=fallback_used,
                              support_chunks=[c.chunk_id for c in support],
                              hits=retrieval_utils.summarize_hits(chunks, limit=5))
                span.set_output({"chunks": len(chunks), "stages": stages,
                                 "top": [c.chunk_id for c in chunks[:5]]})
                return result
            except Exception as exc:  # noqa: BLE001 —— 检索异常统一转 RetrievalError（上层转「不清楚」）
                if isinstance(exc, RetrievalError):
                    raise
                log.log_event("retrieval.failed", level="ERROR", trace_id=trace,
                              error_type=type(exc).__name__, message=str(exc))
                raise wrap(exc, code="RAG-4000", stage="retrieve", trace_id=trace, query=query[:80]) from exc

    def numeric_support_chunks(self, query: str, ranked: Sequence[Any], *,
                               allowed_ids: set[str] | None = None, query_vector: Any | None = None,
                               stages: dict[str, float] | None = None, logger: Any = None) -> list[Any]:
        """取「数值锚定支持块」：数值类问题里 BM25 排在前面、**含带单位数值**但未进 top-k 的块。

        实测动机（§15.2，题 207）：含答案原文的 333 字小块 ``p0490_x1131`` 在 BM25 排第 7、RRF 排第 13，
        向量路完全没召回它 → top-5 里没有任何 ``15,000 万元``。本方法把它作为**生成上下文**补进来
        （排在最前），而 ``chunks``（top-k 排名）保持不变，所以召回判据与引用校验语义都不变。
        """
        log = logger or self.log
        config = self.cfg
        with log.enter("HybridRetriever.numeric_support_chunks",
                       {"query": query[:60], "ranked": len(ranked), "filtered": allowed_ids is not None}) as span:
            if self.bm25 is None:
                span.set_output({"support": 0, "reason": "无 BM25 索引"})
                return []
            # 触发条件：问题本身像数值题，**或**分类为表格型字段（募集资金/关联方/注册资本…）。
            # 后者修「问的是清单+金额、但问题里没有数字」的题（实测题 2 的募投项目清单）。
            table_field = False
            try:
                from .query_understanding import TABLE_FIELD_TYPES, classify_question

                table_field = str(classify_question(query)[0]) in TABLE_FIELD_TYPES
            except Exception as exc:  # noqa: BLE001 —— 分类失败显式降级为「只按数字信号判断」，不静默
                log.log_event("retrieval.support_classify_failed", level="WARNING",
                              error_type=type(exc).__name__, message=str(exc))
            if not (text_utils.has_numeric_signal(query) or table_field):
                span.set_output({"support": 0, "reason": "问题非数值型且非表格型字段"})
                return []
            try:
                hits = self.bm25.search(query, max(config.retrieval.bm25_k, 20), allowed_ids=allowed_ids,
                                        logger=log)
            except Exception as exc:  # noqa: BLE001 —— 支持块失败只降级，不影响主检索
                log.log_event("retrieval.support_failed", level="ERROR", error_type=type(exc).__name__,
                              message=str(exc), degrade="本次不带数值锚定支持块")
                span.set_output({"support": 0, "reason": "BM25 查询异常，已降级"})
                return []
            ranked_ids = {str(getattr(c, "chunk_id", "")) for c in ranked}
            # 只允许与 top-k 同文件的块做支持块：修「PDF2 的问题被 PDF1 的块污染」（实测题 2 曾引用 1 号 PDF）
            ranked_files = {str(getattr(c, "file_name", "")) for c in ranked}
            wanted_units = re.compile(r"\d[\d,，.]*\s*(?:万元|亿元|元|万股|%|％|次|倍|年|月|日)")
            # 清单型表（表头含「序号 + 项目名称」）也是高价值支持块：top-k 里常常只有同页的正文块，
            # 表格块本身没进来（t13 实测题 2：top-k 的 p22 是正文块，真正的募投表 p0022_t07 未召回）
            list_table = re.compile(r"序号[^\n]*项目名称|项目名称[^\n]*序号")

            def _as_support(chunk: Any, score: float, kind: str) -> Any:
                """把一个块包成支持块（``kind`` 用于 boosts 说明来源）。"""
                return RetrievedChunk(
                    chunk_id=chunk.chunk_id, file_name=chunk.file_name, page=chunk.page, type=chunk.type,
                    content=chunk.content, section=chunk.section, table_id=chunk.table_id,
                    score=round(float(score), 6), vector_score=None,
                    bm25_score=float(score) if score else None, rrf_score=None,
                    boosts={kind: 1.0}, rank=0)

            support: list[Any] = []
            candidates: list[tuple[Any, float]] = []
            for chunk_id, _score in hits:
                if chunk_id in ranked_ids:
                    continue
                chunk = self.chunk_map.get(chunk_id)
                if chunk is None or (ranked_files and chunk.file_name not in ranked_files):
                    continue
                candidates.append((chunk, float(_score)))
            # ① 清单表优先（题 2 的募投表：表头含「序号 + 项目名称」）
            for chunk, score in candidates:
                if str(chunk.type) == "table" and list_table.search(str(chunk.content)[:300]):
                    support.append(_as_support(chunk, score, "list_table"))
                    break
            # ② 表格型字段但在 BM25 前排找不到清单表 → 按「同文件 + 表结构」直接挑，
            #    优先挑页码已在 top-k 里的那块（引用落在候选集内，便于溯源核对）
            if not support and table_field:
                ranked_pages = {(str(getattr(c, "file_name", "")), int(getattr(c, "page", 0))) for c in ranked}
                scans = [(c, 100.0) for c in self.chunk_map.values()
                         if str(c.type) == "table" and not (ranked_files and c.file_name not in ranked_files)
                         and c.chunk_id not in ranked_ids and list_table.search(str(c.content)[:300])]
                scans.sort(key=lambda item: (0 if (item[0].file_name, item[0].page) in ranked_pages else 1,
                                             item[0].page))
                if scans:
                    chunk, score = scans[0]
                    support.append(_as_support(chunk, score, "list_table_scan"))
                    log.log_event("retrieval.support_scan_fallback", level="WARNING", chunk_id=chunk.chunk_id,
                                  page=chunk.page, candidates=len(scans),
                                  reason="BM25 前排无清单表，按表结构直接挑支持块（优先 top-k 同页）")
            # ③ 其余名额补「带单位数值」的块
            for chunk, score in candidates:
                if len(support) >= 2:
                    break
                if any(str(getattr(c, "chunk_id", "")) == chunk.chunk_id for c in support):
                    continue
                if wanted_units.search(str(chunk.content)):
                    support.append(_as_support(chunk, score, "numeric_support"))
            if stages is not None:
                stages["support_ms"] = stages.get("support_ms", 0.0)
            log.log_event("retrieval.numeric_support", support=[c.chunk_id for c in support],
                          pages=[c.page for c in support])
            span.set_output({"support": len(support)})
            return support

    def field_support_chunks(self, query: str, ranked: Sequence[Any], *,
                             allowed_ids: set[str] | None = None, logger: Any = None) -> list[Any]:
        """取「字段结构化支持块」（t21，§24）：字段型问题优先拿到**发行人自己的**字段结构化块。

        实测动机（多轮字段型指代，A4）：轮 2「那法定代表人呢？」改写后检索命中的是中介机构/简历类表格
        （p32/p33/p72/p255 —— 块里的「法定代表人」指向**别的主体**：保荐机构、评估机构、子公司），
        LLM 据此答成「法定代表人为程勇波」并引 p255；而**同一问法的单轮题 531 能答对**，只因为它的
        top-k 里恰好有**发行人基本情况块**（p22 表格 / p52 正文）。
        本方法把「发行人自己的基本情况结构化块」补进**生成上下文**（``support_chunks``：排在最前、
        可被引用、**不参与 top-k 排名**，与数值支持块同一机制），字段型问题不再依赖这种偶然性。

        选块优先级（全部**通用**判据，无题目特判）：
            ① 块内「公司名称/中文名称」字段取值 ∈ 动态发行人名单（`discover_issuer_names()`，禁硬编码）；
            ② 块内基本情况字段数（`STRUCTURED_BASIC_FIELDS`）更多者优先；
            ③ 页码已在 top-k 优先；④ 页码顺序（招股书结构顺序：基本情况在前）。
        **跳过条件**：top-k 里已存在「优先级不低于最佳候选」的结构化块 → 不补块
        （避免把已经答对的场景改坏；实测：无此闸门时轮 1 的注册资本被子公司块带偏成「兴图投资 1,204.00 万元」）。
        """
        log = logger or self.log
        with log.enter("HybridRetriever.field_support_chunks",
                       {"query": query[:60], "ranked": len(ranked), "filtered": allowed_ids is not None}) as span:
            try:
                from .query_understanding import FIELD_KEYWORDS, classify_question

                field_type = str(classify_question(query)[0])
                keywords = [k for k in FIELD_KEYWORDS.get(field_type, ()) if k in query]
                issuer_names = [str(name) for name in discover_issuer_names(logger=log)]
            except Exception as exc:  # noqa: BLE001 —— 分类失败必须留痕并显式降级，不影响主检索
                log.log_event("retrieval.field_support_degrade", level="WARNING",
                              error_type=type(exc).__name__, message=str(exc),
                              degrade="本次不带字段结构化支持块")
                span.set_output({"support": 0, "reason": "字段分类异常，已降级"})
                return []
            if not keywords:
                span.set_output({"support": 0, "reason": "问题不含字段关键词"})
                return []
            ranked_ids = {str(getattr(c, "chunk_id", "")) for c in ranked}
            ranked_files = {str(getattr(c, "file_name", "")) for c in ranked}
            ranked_pages = {(str(getattr(c, "file_name", "")), int(getattr(c, "page", 0))) for c in ranked}
            # 结构化赋值形态：①「字段：取值」② Markdown 表行「| 字段 | 取值 |」③「字段 取值」（同行空白分隔）
            patterns: list[tuple[str, Any]] = []
            for keyword in keywords:
                escaped = re.escape(keyword)
                patterns.append((keyword, re.compile(rf"{escaped}\s*[:：]\s*[|｜]?\s*([^\s|｜]{{1,20}})")))
                patterns.append((keyword, re.compile(rf"\|\s*{escaped}\s*\|\s*([^|｜\n]{{1,20}})\|")))
                patterns.append((keyword, re.compile(rf"{escaped}\s+([\u4e00-\u9fff]{{2,6}})(?:\s|$)")))
            name_field = re.compile(r"(?:公司名称|中文名称)\s*[:：]?\s*[|｜]?\s*([^\s|｜，。；;]{2,30})")

            def _structured(chunk: Any) -> tuple[str, str, int, bool]:
                """(取值, 命中的字段词, 基本情况字段数, 是否「公司名称＝发行人」)。非结构化形态返回空串。"""
                content = str(getattr(chunk, "content", "") or "")
                if not content:
                    return "", "", 0, False
                value, keyword_hit = "", ""
                for keyword, pattern in patterns:
                    match = pattern.search(content)
                    if match:
                        value = match.group(1).strip(" \t:：|｜")
                        keyword_hit = keyword
                        break
                name_hit = False
                name_match = name_field.search(content)
                if name_match:
                    name_value = name_match.group(1).strip(" \t:：|｜")
                    name_hit = any(name_value and (name_value in name or name in name_value)
                                   for name in issuer_names)
                basic_fields = sum(1 for name in STRUCTURED_BASIC_FIELDS if name in content)
                return value, keyword_hit, basic_fields, name_hit

            def _rank(value: str, keyword_hit: str, basic_fields: int, name_hit: bool, chunk: Any) -> int:
                """结构化块的优先级打分：发行人自身基本情况块 = 3；多字段 = 2；单字段 = 1；非结构化 = 0。"""
                if not value or not keyword_hit:
                    return 0
                if name_hit:
                    return 3
                return 2 if basic_fields >= 2 else 1

            ranked_best = 0
            for ranked_chunk in ranked:
                value, keyword_hit, basic_fields, name_hit = _structured(ranked_chunk)
                ranked_best = max(ranked_best, _rank(value, keyword_hit, basic_fields, name_hit, ranked_chunk))
            candidates: list[tuple[int, int, int, Any, str, str]] = []
            for chunk in self.chunk_map.values():
                chunk_id = str(getattr(chunk, "chunk_id", ""))
                if chunk_id in ranked_ids:
                    continue
                if allowed_ids is not None and chunk_id not in allowed_ids:
                    continue
                if ranked_files and str(getattr(chunk, "file_name", "")) not in ranked_files:
                    continue
                content = str(getattr(chunk, "content", "") or "")
                if not content or not any(keyword in content for keyword in keywords):
                    continue
                value, keyword_hit, basic_fields, name_hit = _structured(chunk)
                score = _rank(value, keyword_hit, basic_fields, name_hit, chunk)
                if score <= 0 or len(value) > 20:
                    continue
                priority = 0 if (str(getattr(chunk, "file_name", "")), int(getattr(chunk, "page", 0))) in ranked_pages \
                    else 1
                candidates.append((score, priority, int(getattr(chunk, "page", 0)), chunk, value, keyword_hit))
            if not candidates:
                span.set_output({"support": 0, "reason": "无可用结构化字段块"})
                return []
            candidates.sort(key=lambda item: (-item[0], item[1], item[2],
                                              str(getattr(item[3], "chunk_id", ""))))
            best_score = candidates[0][0]
            if ranked_best >= best_score:
                log.log_event("retrieval.field_support_skip", level="WARNING",
                              ranked_best=ranked_best, best_candidate=best_score,
                              top= str(getattr(candidates[0][3], "chunk_id", "")),
                              reason="top-k 已含不低于最佳候选的结构化字段块，不再补块")
                span.set_output({"support": 0, "reason": "top-k 已含同级或更强的结构化块"})
                return []
            support: list[Any] = []
            for score, priority, _page, chunk, value, keyword_hit in candidates[:2]:
                support.append(RetrievedChunk(
                    chunk_id=chunk.chunk_id, file_name=chunk.file_name, page=chunk.page, type=chunk.type,
                    content=chunk.content, section=chunk.section, table_id=chunk.table_id,
                    score=round(score / 3.0, 6), vector_score=None, bm25_score=None, rrf_score=None,
                    boosts={"field_support": 1.0}, rank=0))
                log.log_event("retrieval.field_support", level="WARNING", chunk_id=chunk.chunk_id,
                              page=chunk.page, keyword=keyword_hit, value=value, score=score,
                              file_name=chunk.file_name)
            log.log_event("retrieval.field_support_done", support=[c.chunk_id for c in support],
                          pages=[c.page for c in support])
            span.set_output({"support": len(support), "best_score": best_score})
            return support

    # -- 命中判定 --------------------------------------------------------
    def retrieve_evidence(self, chunks: Sequence[RetrievedChunk], evidence: str, *,
                          threshold: float = 0.90, question_id: int = 0,
                          top_k: int | None = None) -> EvidenceHitReport:
        """判定「证据原文是否落在返回的 chunk 中」（**不得**用页码相等来判定）。"""
        matched = [c.chunk_id for c in chunks
                   if c.content and text_utils.evidence_contains(c.content, evidence, threshold=threshold)]
        report = EvidenceHitReport(
            question_id=int(question_id), top_k=int(top_k if top_k is not None else len(chunks)),
            hit=bool(matched), matched_chunk_ids=matched, missing_pages=[],
            note=("证据原文命中返回块" if matched else "证据原文未落在返回的任何 chunk 中"),
        )
        self.log.log_event("retrieval.evidence_check", question_id=report.question_id,
                           top_k=report.top_k, hit=report.hit, matched=matched,
                           threshold=threshold)
        return report

    # -- 健康检查 --------------------------------------------------------
    def health(self) -> dict[str, Any]:
        """返回索引与预热状态（供 UI/服务探活与测试断言）。"""
        info = {
            "loaded": self.loaded,
            "index_root": str(self.index_root),
            "model_dir": str(self.model_dir) if self.model_dir else None,
            "vectors": int(self.vector_index.count) if self.vector_index else 0,
            "dim": int(self.vector_index.dim) if self.vector_index else 0,
            "bm25_chunks": self.bm25.size() if self.bm25 else 0,
            "bm25_vocab": self.bm25.vocab_size() if self.bm25 else 0,
            "chunks": len(self.chunk_map),
            "files": sorted({c.file_name for c in self.chunk_map.values()}),
            "warmup": self.warmup_info,
        }
        self.log.log_event("retriever.health", **{k: v for k, v in info.items() if k != "warmup"})
        return info


def build_retriever(*, index_dir: Path | str | None = None, cfg: AppConfig | None = None,
                    warmup: bool = True) -> HybridRetriever:
    """构造并加载检索器（**在线服务启动时调用它即完成预热**）。"""
    return HybridRetriever(index_dir=index_dir, cfg=cfg).load(warmup=warmup)
