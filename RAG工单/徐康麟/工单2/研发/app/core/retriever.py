"""混合检索器：向量（块级 + 子块级）+ 自实现 BM25，相对化加权 + 重排，支持页码过滤。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 检索优化（核心，对应 设计/接口设计.md §2.8、优化方案设计.md §3）

算法（与设计文档 §3.1~§3.3 一一对应）::

    base(d)        = w_v · v_norm(d) + w_b · b_norm(d)      # 两路各自 min-max 归一
    boost_total(d) = min(cap, Π boost_j(d))                 # cap = 1.6
    score(d)       = base(d) · boost_total(d)

**本工单新增（实测驱动的排序优化）**：向量侧改为**块级 + 子块级双粒度**。
实测发现证据句被长块"语义稀释"（同一证据：整块余弦 0.50，子块余弦 0.69），
因此额外建立子块级向量索引并映射回父块；分块规模仍保持工单 6.2 要求的 400~600。
``vector_score`` 仍为**块级原始余弦**（保证拒答阈值口径与基线可比）。

其中 boost 项：表格块 / 目标数值覆盖 / 多值百分比 / 数字密度 / 领域关键词（按命中归一）；
降权项：释义页（三段取最小，下限 0.30）/ 碎片。
检索结果字段（工单 6.3 要求）：``chunk_id / page / score / content / type``，
另附 ``vector_score / bm25_score / rerank_score / boosts`` 供审计。
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.core.bm25_index import BM25Index
from app.core.config import get_settings
from app.core.embedder import Embedder, get_embedder
from app.core.errors import RetrievalError
from app.core.logging_conf import current_trace_id, log_event, logger, trace
from app.core.retrieval_utils import (
    boilerplate_penalty,
    digit_density,
    fragment_penalty,
    keyword_boost_factor,
    numeric_coverage,
    percent_count,
    required_numeric_types,
    table_hint,
    value_coverage_score,
)
from app.core.text_utils import STOPWORDS, tokenize
from app.core.vector_store import VectorStore, get_vector_store
from app.models.schemas import Chunk, QueryAnalysis, RetrievedChunk


@dataclass
class RetrievalDebug:
    """最近一次检索的调试快照（供日志、UI 折叠区与测试取证）。"""

    vector_hits: int = 0
    bm25_hits: int = 0
    merged_hits: int = 0
    reranked: bool = False
    rerank_mode: str = "off"
    page_filter: list[int] = field(default_factory=list)
    boosted_table: int = 0
    boosted_numeric: int = 0
    boosted_keyword: int = 0
    penalized_boilerplate: int = 0
    penalized_fragment: int = 0
    top_cosine: float = 0.0
    top_score: float = 0.0
    variants: list[str] = field(default_factory=list)
    trace_id: str = ""
    elapsed_ms: float = 0.0
    pages: list[int] = field(default_factory=list)
    top_chunk_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """序列化（写入日志/追踪表）。"""
        return {
            "vector_hits": self.vector_hits,
            "bm25_hits": self.bm25_hits,
            "merged_hits": self.merged_hits,
            "reranked": self.reranked,
            "rerank_mode": self.rerank_mode,
            "page_filter": list(self.page_filter),
            "boosted_table": self.boosted_table,
            "boosted_numeric": self.boosted_numeric,
            "boosted_keyword": self.boosted_keyword,
            "penalized_boilerplate": self.penalized_boilerplate,
            "penalized_fragment": self.penalized_fragment,
            "top_cosine": self.top_cosine,
            "top_score": self.top_score,
            "variants": list(self.variants),
            "trace_id": self.trace_id,
            "elapsed_ms": self.elapsed_ms,
            "pages": list(self.pages),
            "top_chunk_ids": list(self.top_chunk_ids),
        }


class Retriever:
    """混合检索器（向量 + BM25 + 加权 + 重排）。"""

    def __init__(
        self,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        bm25_index: BM25Index | None = None,
        reranker: Any | None = None,
    ) -> None:
        self._settings = get_settings()
        self._embedder = embedder or get_embedder()
        self._store = vector_store or get_vector_store()
        self._bm25 = bm25_index or BM25Index()
        self._reranker = reranker
        self._chunks: dict[str, Chunk] = {}
        self._debug = RetrievalDebug()
        self._lock = threading.RLock()
        self._ready = False
        # 子块级向量索引：[(chunk_id, segment)] 与对应向量矩阵
        self._segment_owner: list[str] = []
        self._segment_text: list[str] = []
        self._segment_vectors: np.ndarray | None = None

    # ------------------------------------------------------------------
    # 索引管理
    # ------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        """索引是否就绪（向量与分块都在位）。"""
        return self._ready and bool(self._chunks)

    def set_chunks(self, chunks: list[Chunk]) -> None:
        """设置分块表（检索结果回填正文用）。"""
        with self._lock:
            self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        logger.info("app.core.retriever", "分块表已设置", count=len(self._chunks))

    @trace
    def build_index(self, chunks: list[Chunk], reset: bool = True) -> dict[str, Any]:
        """构建块级 + 子块级向量索引与 BM25 索引（嵌入用当前嵌入后端）。"""
        try:
            if not chunks:
                raise RetrievalError("分块为空，无法构建索引")
            if reset:
                self._store.reset()
                self._bm25.build([])
                self._segment_owner, self._segment_text, self._segment_vectors = [], [], None
            self.set_chunks(chunks)
            texts = [chunk.content for chunk in chunks]
            started = time.perf_counter()
            vectors = self._embedder.encode(texts)
            embed_ms = (time.perf_counter() - started) * 1000
            count = self._store.add(chunks, vectors)
            self._bm25.build(chunks)
            segment_stats: dict[str, Any] = {"segments": 0, "segment_embed_ms": 0.0}
            if self._settings.retrieval.segment_recall_enabled:
                segment_stats = self._build_segments(chunks)
            self._ready = True
            stats = {
                "chunks": len(chunks),
                "vectors": count,
                "dimension": int(vectors.shape[1]),
                "bm25_docs": self._bm25.size,
                "embedder": self._embedder.name,
                "index_dir": str(self._store.index_dir),
                "embed_ms": round(embed_ms, 2),
                **segment_stats,
            }
            logger.info("app.core.retriever", "索引构建完成", **stats)
            return stats
        except RetrievalError:
            logger.exception("app.core.retriever", "索引构建失败（空语料）")
            raise
        except Exception as exc:
            logger.exception("app.core.retriever", "索引构建异常")
            raise RetrievalError(f"索引构建失败: {exc}") from exc

    def _build_segments(self, chunks: list[Chunk]) -> dict[str, Any]:
        """构建子块级向量索引（解决长块语义稀释；命中后映射回父块）。"""
        settings = self._settings.retrieval
        owner: list[str] = []
        texts: list[str] = []
        for chunk in chunks:
            for segment in split_segments(
                chunk.content,
                max_chars=settings.segment_max_chars,
                min_chars=settings.segment_min_chars,
                max_segments=settings.segment_max_per_chunk,
            ):
                owner.append(chunk.chunk_id)
                texts.append(segment)
        if not texts:
            logger.warning("app.core.retriever", "子块切分为空，跳过子块级索引")
            return {"segments": 0, "segment_embed_ms": 0.0}
        started = time.perf_counter()
        matrix = self._embedder.encode(texts)
        elapsed = (time.perf_counter() - started) * 1000
        self._segment_owner = owner
        self._segment_text = texts
        self._segment_vectors = np.asarray(matrix, dtype=np.float32)
        logger.info(
            "app.core.retriever",
            "子块级向量索引构建完成",
            segments=len(texts),
            elapsed_ms=round(elapsed, 2),
        )
        return {"segments": len(texts), "segment_embed_ms": round(elapsed, 2)}

    @trace
    def save_index(self) -> dict[str, str]:
        """持久化块级向量索引、子块级向量索引与 BM25 索引。"""
        result: dict[str, str] = {}
        try:
            vector_dir = self._store.save()
            if vector_dir is not None:
                result["vectors"] = str(vector_dir)
            if self._segment_vectors is not None and self._segment_owner:
                segment_dir = self._store.index_dir
                np.save(segment_dir / "segments.npy", self._segment_vectors)
                with open(segment_dir / "segments_meta.jsonl", "w", encoding="utf-8", newline="\n") as handle:
                    for chunk_id, text in zip(self._segment_owner, self._segment_text):
                        handle.write(json.dumps({"chunk_id": chunk_id, "text": text}, ensure_ascii=False) + "\n")
                result["segments"] = str(segment_dir / "segments.npy")
            bm25_path = self._store.index_dir / "bm25_index.pkl"
            saved = self._bm25.save(bm25_path)
            result["bm25"] = str(saved)
            logger.info("app.core.retriever", "索引已保存", **result)
        except Exception as exc:
            logger.exception("app.core.retriever", "索引保存失败")
            raise RetrievalError(f"索引保存失败: {exc}") from exc
        return result

    @trace
    def load_index(self, chunks: list[Chunk]) -> bool:
        """加载块级/子块级向量索引与 BM25 索引；维度不一致时返回 False（不静默混用）。"""
        try:
            self.set_chunks(chunks)
            ok = self._store.load()
            if not ok:
                logger.error(
                    "app.core.retriever",
                    "向量索引加载失败（缺失或维度不一致）",
                    index_dir=str(self._store.index_dir),
                    dimension=self._store.dimension,
                )
                self._ready = False
                return False
            bm25_path = self._store.index_dir / "bm25_index.pkl"
            if bm25_path.exists():
                self._bm25 = BM25Index.load(bm25_path)
            else:
                logger.warning(
                    "app.core.retriever",
                    "BM25 索引缺失，用现有分块就地重建（仅内存，不写盘）",
                    path=str(bm25_path),
                )
                self._bm25.build(chunks)
            self._load_segments()
            self._ready = True
            logger.info(
                "app.core.retriever",
                "索引加载完成",
                chunks=len(chunks),
                vectors=self._store.count(),
                segments=len(self._segment_owner),
                bm25_docs=self._bm25.size,
                dimension=self._store.dimension,
                embedder=self._embedder.name,
            )
            return True
        except Exception:
            logger.exception("app.core.retriever", "索引加载异常")
            self._ready = False
            return False

    def _load_segments(self) -> None:
        """加载子块级向量索引；缺失时按当前分块就地重建（仅内存）。"""
        segment_path = self._store.index_dir / "segments.npy"
        meta_path = self._store.index_dir / "segments_meta.jsonl"
        try:
            if segment_path.exists() and meta_path.exists():
                matrix = np.load(segment_path)
                owner: list[str] = []
                texts: list[str] = []
                with open(meta_path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        owner.append(str(row.get("chunk_id", "")))
                        texts.append(str(row.get("text", "")))
                if matrix.shape[0] == len(owner) and matrix.shape[0] > 0:
                    self._segment_vectors = np.asarray(matrix, dtype=np.float32)
                    self._segment_owner = owner
                    self._segment_text = texts
                    return
                logger.error(
                    "app.core.retriever",
                    "子块索引条数与元数据不一致，按当前分块重建",
                    vectors=int(matrix.shape[0]),
                    meta=len(owner),
                )
            logger.warning("app.core.retriever", "子块级索引缺失，按当前分块就地重建（仅内存，不写盘）")
            self._build_segments(list(self._chunks.values()))
        except Exception:
            logger.exception("app.core.retriever", "子块级索引加载失败，禁用子块路径")
            self._segment_owner, self._segment_text, self._segment_vectors = [], [], None

    def _search_segments(self, query_vector: np.ndarray) -> dict[str, float]:
        """子块级召回：返回 ``{父块 chunk_id: 最高子块余弦}``。"""
        try:
            if self._segment_vectors is None or not self._segment_owner:
                return {}
            query = np.asarray(query_vector, dtype=np.float32).ravel()
            if query.shape[0] != self._segment_vectors.shape[1]:
                logger.error(
                    "app.core.retriever",
                    "子块索引维度与查询向量不一致，跳过子块路径",
                    index_dim=int(self._segment_vectors.shape[1]),
                    query_dim=int(query.shape[0]),
                )
                return {}
            norm = float(np.linalg.norm(query))
            if norm > 0:
                query = query / norm
            scores = self._segment_vectors @ query
            top_k = min(self._settings.retrieval.segment_top_k, scores.shape[0])
            if top_k <= 0:
                return {}
            if top_k < scores.shape[0]:
                candidate_index = np.argpartition(-scores, top_k - 1)[:top_k]
            else:
                candidate_index = np.arange(scores.shape[0])
            best: dict[str, float] = {}
            for index in candidate_index.tolist():
                chunk_id = self._segment_owner[index]
                value = float(scores[index])
                if value > best.get(chunk_id, -1.0):
                    best[chunk_id] = value
            return best
        except Exception:
            logger.exception("app.core.retriever", "子块级检索异常，跳过子块路径")
            return {}

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    @trace
    def retrieve(
        self,
        query: str,
        analysis: QueryAnalysis | None = None,
        top_k: int | None = None,
        page_filter: list[int] | None = None,
    ) -> list[RetrievedChunk]:
        """单路检索：向量 + BM25 → 归一融合 → 相对化加权 → 重排取 top_k。"""
        started = time.perf_counter()
        settings = self._settings.retrieval
        try:
            if not self._chunks:
                raise RetrievalError("分块表为空（索引未加载）")
            text = (query or "").strip()
            if not text:
                logger.warning("app.core.retriever", "空查询，返回空结果")
                return []
            # 1) 块级向量召回
            vector_scores: dict[str, float] = {}
            segment_scores: dict[str, float] = {}
            query_vector: np.ndarray | None = None
            try:
                query_vector = self._embedder.encode_one(text)
                for chunk_id, score in self._store.search(query_vector, top_k=settings.vector_top_k):
                    vector_scores[chunk_id] = float(score)
            except Exception:
                logger.exception("app.core.retriever", "块级向量召回失败，尝试子块路径", query=text[:80])
            # 1b) 子块级向量召回（解决长块语义稀释；命中映射回父块）
            if query_vector is not None and settings.segment_recall_enabled:
                try:
                    segment_scores = self._search_segments(query_vector)
                except Exception:
                    logger.exception("app.core.retriever", "子块级召回失败，仅用块级路径", query=text[:80])
            # 1c) 子块命中的父块若不在块级 top-k，补算其块级余弦（置信度口径需要）
            missing = [cid for cid in segment_scores if cid not in vector_scores]
            if missing and query_vector is not None:
                try:
                    vector_scores.update(self._store.score_ids(query_vector, missing))
                except Exception:
                    logger.exception("app.core.retriever", "子块父块余弦回填失败", count=len(missing))
            # 2) BM25 召回
            bm25_scores: dict[str, float] = {}
            try:
                for chunk_id, score in self._bm25.search(text, top_k=settings.bm25_top_k):
                    bm25_scores[chunk_id] = float(score)
            except Exception:
                logger.exception("app.core.retriever", "BM25 召回失败，仅用向量路径", query=text[:80])

            if not vector_scores and not bm25_scores:
                logger.warning("app.core.retriever", "两路召回均为空，返回空结果", query=text[:80])
                self._debug = RetrievalDebug(trace_id=current_trace_id(), variants=[text])
                return []

            # 3) 融合（各自 min-max 归一后线性加权）
            #    向量侧 = max(块级归一, 子块级归一 × segment_path_weight)：子块路径解决长块稀释
            vector_norm = _min_max(vector_scores)
            bm25_norm = _min_max(bm25_scores)
            segment_norm = _min_max(segment_scores)
            candidates: list[RetrievedChunk] = []
            # 加权/重排所依据的"问题"必须是**实际检索用的查询串**：
            # 英文提问经语言桥接后检索串是中文，若拿英文原句算领域关键词与数值类型，
            # 会全部识别不到 → 数值补充召回与关键词加权失效（t11 跨语言回归根因之二）。
            question = feature_question(analysis, text)
            boosted = {"table": 0, "numeric": 0, "keyword": 0, "boilerplate": 0, "fragment": 0}
            for chunk_id in set(vector_scores) | set(bm25_scores) | set(segment_scores):
                chunk = self._chunks.get(chunk_id)
                if chunk is None:
                    continue
                raw_cosine = vector_scores.get(chunk_id, 0.0)
                vector_side = max(
                    vector_norm.get(chunk_id, 0.0),
                    segment_norm.get(chunk_id, 0.0) * settings.segment_path_weight,
                )
                base = settings.vector_weight * vector_side + settings.bm25_weight * bm25_norm.get(chunk_id, 0.0)
                boosts, hit_flags = self._compute_boosts(chunk, question)
                total_boost = min(settings.total_boost_cap, _product(boosts.values()))
                score = base * total_boost
                for key in boosted:
                    if hit_flags.get(key):
                        boosted[key] += 1
                source = "hybrid"
                if chunk.type == "table":
                    source = "table"
                elif chunk_id in vector_scores and chunk_id not in bm25_scores:
                    source = "vector"
                elif chunk_id in bm25_scores and chunk_id not in vector_scores:
                    source = "bm25"
                candidates.append(
                    RetrievedChunk(
                        chunk=chunk,
                        score=round(score, 6),
                        vector_score=round(raw_cosine, 6),
                        bm25_score=round(bm25_scores.get(chunk_id, 0.0), 6),
                        rerank_score=None,
                        source=source,  # type: ignore[arg-type]
                        boosts={
                            **{k: round(v, 4) for k, v in boosts.items()},
                            "vector_norm": round(vector_norm.get(chunk_id, 0.0), 4),
                            "segment_norm": round(segment_norm.get(chunk_id, 0.0), 4),
                            "segment_cos": round(segment_scores.get(chunk_id, 0.0), 4),
                            "vector_side": round(vector_side, 4),
                            "bm25_norm": round(bm25_norm.get(chunk_id, 0.0), 4),
                            "total_boost": round(total_boost, 4),
                        },
                    )
                )
            # 4) 页码过滤（过滤后无候选要显式返回空，**不再静默忽略**）
            pages = list(page_filter or (analysis.page_filter if analysis is not None else []) or [])
            if pages:
                allowed = set(pages)
                before = len(candidates)
                candidates = [item for item in candidates if item.chunk.page in allowed]
                if not candidates:
                    logger.warning(
                        "app.core.retriever",
                        "指定页码过滤后无候选，返回空结果（PAGE_FILTER_EMPTY）",
                        page_filter=pages,
                        before=before,
                    )
                    self._debug = RetrievalDebug(
                        vector_hits=len(vector_scores),
                        bm25_hits=len(bm25_scores),
                        merged_hits=before,
                        page_filter=pages,
                        trace_id=current_trace_id(),
                        variants=[text],
                        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                    )
                    return []
            candidates.sort(key=lambda item: item.score, reverse=True)
            # 4b) 数值型问题：额外召回"多值齐备度"最高的块（实测：真实证据在向量/BM25 上不占优）
            pool: list[RetrievedChunk] = list(candidates[: settings.fusion_top_k])
            if settings.value_augment_enabled:
                pool = self._augment_value_candidates(question, pool, candidates)
            # 5) 重排（融合池 + 数值补充池 → top5）
            want = int(top_k or settings.rerank_top_n)
            if self._reranker is not None and getattr(self._reranker, "mode", "off") != "off":
                final = self._reranker.rerank(question, pool[: settings.rerank_pool_size], top_n=want)
                rerank_mode = str(getattr(self._reranker, "mode", "rule"))
            else:
                final = pool[:want]
                rerank_mode = "off"
            top_cosine = max((item.vector_score for item in candidates), default=0.0)
            elapsed_ms = (time.perf_counter() - started) * 1000
            self._debug = RetrievalDebug(
                vector_hits=len(vector_scores),
                bm25_hits=len(bm25_scores),
                merged_hits=len(candidates),
                reranked=rerank_mode != "off",
                rerank_mode=rerank_mode,
                page_filter=pages,
                boosted_table=boosted["table"],
                boosted_numeric=boosted["numeric"],
                boosted_keyword=boosted["keyword"],
                penalized_boilerplate=boosted["boilerplate"],
                penalized_fragment=boosted["fragment"],
                top_cosine=round(top_cosine, 6),
                top_score=round(final[0].score, 6) if final else 0.0,
                variants=[text],
                trace_id=current_trace_id(),
                elapsed_ms=round(elapsed_ms, 2),
                pages=[item.chunk.page for item in final],
                top_chunk_ids=[item.chunk.chunk_id for item in final],
            )
            log_event(
                "retrieval",
                "app.core.retriever",
                "Retriever.retrieve",
                trace_id=self._debug.trace_id,
                query=question,
                rewritten=analysis.rewritten if analysis is not None else "",
                variants=[text],
                vector_hits=len(vector_scores),
                bm25_hits=len(bm25_scores),
                merged=len(candidates),
                boosts={
                    "table": boosted["table"],
                    "numeric": boosted["numeric"],
                    "keyword": boosted["keyword"],
                    "boilerplate": boosted["boilerplate"],
                    "fragment": boosted["fragment"],
                },
                pages=self._debug.pages,
                top_cosine=self._debug.top_cosine,
                top_score=self._debug.top_score,
                rerank_mode=rerank_mode,
                elapsed_ms=self._debug.elapsed_ms,
            )
            logger.info(
                "app.core.retriever",
                "检索完成",
                query=text[:60],
                vector_hits=len(vector_scores),
                bm25_hits=len(bm25_scores),
                merged=len(candidates),
                top_pages=self._debug.pages,
                top_cosine=self._debug.top_cosine,
                rerank_mode=rerank_mode,
                elapsed_ms=self._debug.elapsed_ms,
            )
            return final
        except RetrievalError:
            logger.exception("app.core.retriever", "检索失败（索引未就绪）")
            raise
        except Exception as exc:
            logger.exception("app.core.retriever", "检索异常")
            raise RetrievalError(f"检索失败: {exc}") from exc

    def _compute_boosts(self, chunk: Chunk, question: str) -> tuple[dict[str, float], dict[str, bool]]:
        """计算单块的加权/降权系数（相对化，含硬上限的乘性项）。"""
        settings = self._settings.retrieval
        boosts: dict[str, float] = {}
        flags: dict[str, bool] = {}

        # 表格块
        if chunk.type == "table":
            factor = settings.table_boost_hint if table_hint(question) else settings.table_boost
            boosts["table"] = factor
            flags["table"] = factor != 1.0
        # 目标数值覆盖
        coverage = numeric_coverage(question, chunk.content)
        if coverage > 0:
            factor = 1.0 + settings.numeric_boost_max * coverage
            boosts["numeric"] = factor
            flags["numeric"] = factor > 1.0
        # 多值百分比块（"比重分别是多少"）
        if percent_count(chunk.content) >= settings.multi_percent_min_count:
            boosts["multi_percent"] = settings.multi_percent_boost
        # 数字密度微弱修正
        density = digit_density(chunk.content)
        if density > 0:
            boosts["density"] = 1.0 + settings.density_boost_max * min(
                density / settings.density_reference, 1.0
            )
        # 领域关键词（按命中归一）
        keyword_factor, keyword_hits = keyword_boost_factor(question, chunk.content)
        if keyword_factor > 1.0:
            boosts["keyword"] = keyword_factor
            flags["keyword"] = True
        # 降权：释义页（取最小值，不叠乘）
        penalty, reason = boilerplate_penalty(chunk)
        if penalty < 1.0:
            boosts["boilerplate_penalty"] = penalty
            flags["boilerplate"] = True
        # 降权：碎片
        fragment = fragment_penalty(chunk.content)
        if fragment < 1.0:
            boosts["fragment_penalty"] = fragment
            flags["fragment"] = True
        if keyword_hits:
            boosts["keyword_hits"] = float(len(keyword_hits))  # 审计用（不计入乘积）
        return boosts, flags

    def _augment_value_candidates(
        self,
        question: str,
        pool: list[RetrievedChunk],
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """数值型问题补充"多值齐备度"最高的块进入精排池。

        为什么需要：Q260/Q33 这类"分别是多少/比重分别是多少"的问题，真实证据块需要在同一块内
        并列多个金额与百分比；实测该块在向量（rank 44~57）与 BM25（rank 22）上都不占优，
        单靠融合排序会把它挡在精排池之外。此路径按"去重金额数 + 去重百分比数"独立召回一次。
        """
        try:
            settings = self._settings.retrieval
            needed = required_numeric_types(question)
            if not ({"amount", "percent"} & needed):
                return pool
            existing = {item.chunk.chunk_id for item in pool}
            scored: list[tuple[float, Chunk]] = []
            for chunk in self._chunks.values():
                if chunk.chunk_id in existing:
                    continue
                score = value_coverage_score(question, chunk.content)
                if score > 0:
                    scored.append((score, chunk))
            if not scored:
                return pool
            scored.sort(key=lambda pair: pair[0], reverse=True)
            by_id = {item.chunk.chunk_id: item for item in candidates}
            added = 0
            for score, chunk in scored[: settings.value_augment_top_n]:
                if chunk.chunk_id in existing:
                    continue
                base = by_id.get(chunk.chunk_id)
                if base is not None:
                    base.boosts["value_augment"] = round(score, 4)
                    pool.append(base)
                else:
                    pool.append(
                        RetrievedChunk(
                            chunk=chunk,
                            score=0.0,
                            vector_score=0.0,
                            bm25_score=0.0,
                            source="table" if chunk.type == "table" else "vector",
                            boosts={"value_augment": round(score, 4)},
                        )
                    )
                existing.add(chunk.chunk_id)
                added += 1
            if added:
                logger.info(
                    "app.core.retriever",
                    "数值型问题已补充多值齐备块进入精排池",
                    added=added,
                    pool=len(pool),
                    question=question[:60],
                )
            return pool
        except Exception:
            logger.exception("app.core.retriever", "数值补充召回失败（不影响主链路）")
            return pool

    # ------------------------------------------------------------------
    @trace
    def retrieve_multi(
        self,
        queries: list[str] | list[tuple[str, str]],
        analysis: QueryAnalysis | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """多路变体检索：按变体相对置信度平方衰减合并（防弱变体拉偏）。"""
        try:
            pairs: list[tuple[str, str]] = []
            for item in queries or []:
                if isinstance(item, tuple):
                    pairs.append((item[0], item[1]))
                else:
                    pairs.append((item, "variant"))
            if not pairs:
                return []
            if len(pairs) == 1:
                return self.retrieve(pairs[0][0], analysis=analysis, top_k=top_k)

            per_variant: list[tuple[str, str, list[RetrievedChunk]]] = []
            for text, kind in pairs:
                hits = self.retrieve(text, analysis=analysis, top_k=self._settings.retrieval.fusion_top_k)
                per_variant.append((text, kind, hits))
            # 变体置信度：用该变体最高原始余弦
            confidences = [
                max((item.vector_score for item in hits), default=0.0) for _, _, hits in per_variant
            ]
            best = max(confidences) or 1.0
            merged: dict[str, RetrievedChunk] = {}
            hits_count: dict[str, int] = {}
            weights: dict[str, float] = {}
            for (text, kind, hits), confidence in zip(per_variant, confidences):
                weight = max(0.2, min(1.0, (confidence / best) ** 2))
                weights[text] = round(weight, 4)
                for item in hits:
                    scored = item.score * weight
                    existing = merged.get(item.chunk.chunk_id)
                    if existing is None:
                        item.score = round(scored, 6)
                        item.boosts["variant_weight"] = round(weight, 4)
                        item.boosts["variant_kind"] = 0.0
                        merged[item.chunk.chunk_id] = item
                        hits_count[item.chunk.chunk_id] = 1
                    else:
                        hits_count[item.chunk.chunk_id] += 1
                        if scored > existing.score:
                            existing.score = round(scored, 6)
                            existing.boosts["variant_weight"] = round(weight, 4)
            # 多路命中奖励（上限 0.1）
            for chunk_id, item in merged.items():
                bonus = min(0.1, 0.03 * (hits_count[chunk_id] - 1))
                item.score = round(item.score + bonus, 6)
                item.boosts["multi_variant_bonus"] = round(bonus, 4)
            ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
            fused = ranked[: self._settings.retrieval.fusion_top_k]
            want = int(top_k or self._settings.retrieval.rerank_top_n)
            if self._reranker is not None and getattr(self._reranker, "mode", "off") != "off":
                question = feature_question(analysis, pairs[0][0])
                final = self._reranker.rerank(question, fused, top_n=want)
            else:
                final = fused[:want]
            self._debug.variants = [text for text, _ in pairs]
            self._debug.merged_hits = len(merged)
            self._debug.top_score = round(final[0].score, 6) if final else 0.0
            logger.info(
                "app.core.retriever",
                "多变体检索完成",
                variants=self._debug.variants,
                variant_weights=weights,
                merged=len(merged),
                top_pages=[item.chunk.page for item in final],
            )
            return final
        except RetrievalError:
            raise
        except Exception as exc:
            logger.exception("app.core.retriever", "多变体检索异常")
            raise RetrievalError(f"多变体检索失败: {exc}") from exc

    # ------------------------------------------------------------------
    # 置信度与可答性
    # ------------------------------------------------------------------
    def is_confident(self, results: list[RetrievedChunk]) -> bool:
        """置信度判定：用**原始余弦**（绝对尺度，与拒答口径一致）。"""
        try:
            settings = self._settings.retrieval
            if not results:
                return False
            top_cosine = max((item.vector_score for item in results), default=0.0)
            top_score = max((item.score for item in results), default=0.0)
            ok = top_cosine >= settings.min_confidence_cosine and top_score >= settings.min_relevance_score
            if not ok:
                logger.info(
                    "app.core.retriever",
                    "置信度不足（三条件之一不满足）",
                    top_cosine=round(top_cosine, 4),
                    min_cosine=settings.min_confidence_cosine,
                    top_score=round(top_score, 4),
                    min_score=settings.min_relevance_score,
                )
            return ok
        except Exception:
            logger.exception("app.core.retriever", "置信度判定异常，按不可信处理")
            return False

    def is_answerable(self, question: str, results: list[RetrievedChunk]) -> bool:
        """可答性校验：问题实义词至少有 ``min_overlap_terms`` 个出现在片段中。"""
        try:
            settings = self._settings.retrieval
            if settings.min_overlap_terms <= 0:
                return True
            tokens = [token for token in tokenize(question) if token not in STOPWORDS and len(token) > 1]
            if not tokens:
                return True
            joined = " ".join(item.chunk.content for item in results[:3])
            hits = sum(1 for token in set(tokens) if token in joined)
            ok = hits >= settings.min_overlap_terms
            if not ok:
                logger.info(
                    "app.core.retriever",
                    "可答性校验不通过（问题实义词未出现在片段中）",
                    question=question[:60],
                    terms=len(set(tokens)),
                    hits=hits,
                )
            return ok
        except Exception:
            logger.exception("app.core.retriever", "可答性校验异常，按可答处理")
            return True

    def is_topic_covered(self, question: str, contexts: list[RetrievedChunk], intent: str = "") -> tuple[bool, str]:
        """主题/答案类型一致性闸门（t13）：含主体名的无关问题不得靠"高频泛词覆盖"放行。

        与 ``is_answerable``（实义词覆盖）互补：这里要求**语料中最罕见的主题词**出现在证据里，
        并按意图校验**取值类型**（金额/百分比/人名/标准名/奖项/企业名）。详见 ``answerability`` 模块。
        """
        try:
            from app.core.answerability import check

            return check(question, intent, contexts, self._idf_lookup)
        except Exception:
            logger.exception("app.core.retriever", "主题一致性闸门异常，按可答处理（不误拒正常问题）")
            return True, "gate_error"

    def _idf_lookup(self, term: str) -> float:
        """返回词的 IDF（语料内越罕见值越大）；未登录词给**高于语料最大 IDF** 的值。

        为什么未登录词要最大：它代表"证据里根本没有、而问题偏偏强调"的词——
        正是 t13 需要拦下的情形（如「食堂」「高铁」「奖学金」）。
        """
        try:
            backend = getattr(self._bm25, "_backend", None)  # 自实现 PureBM25
            idf = getattr(backend, "idf", None) if backend is not None else None
            if idf:
                return float(idf.get(term, max(idf.values()) + 1.0))
            rank_backend = getattr(self._bm25, "_rank_backend", None)  # rank_bm25（可选依赖）
            rank_idf = getattr(rank_backend, "idf", None)
            if rank_idf is not None:
                try:
                    values = [float(value) for value in rank_idf.values()]
                except AttributeError:
                    import numpy as np  # rank_bm25 的 idf 为 ndarray

                    values = [float(value) for value in np.asarray(rank_idf).tolist()]
                if values:
                    # rank_bm25 为数组 IDB：保守地用"平均 IDF + 1"（不做词表映射，避免错配）
                    return (sum(values) / len(values)) + 1.0
            return 10.0
        except Exception:
            logger.exception("app.core.retriever", "IDF 查询失败，按未登录词处理", term=term)
            return 10.0

    # ------------------------------------------------------------------
    def debug_snapshot(self) -> RetrievalDebug:
        """返回最近一次检索的调试快照。"""
        return self._debug

    def health(self) -> dict[str, Any]:
        """健康信息。"""
        return {
            "embedder": self._embedder.health(),
            "vector_store": self._store.health(),
            "bm25_docs": self._bm25.size,
            "chunks": len(self._chunks),
            "segments": len(self._segment_owner),
            "reranker": self._reranker.health() if self._reranker is not None else {"mode": "off"},
            "ready": self.ready,
        }


def feature_question(analysis: QueryAnalysis | None, fallback: str) -> str:
    """选取"加权/重排所依据的问题串"。

    规则：优先用 ``analysis.search_query``（**实际检索串**）；跨语言场景下它是桥接后的中文问句，
    而 ``analysis.original`` 是英文原句——后者会让领域关键词/数值类型识别全空，
    导致数值补充召回与关键词加权失效（t11 跨语言召回回归的根因之二）。
    """
    if analysis is None:
        return fallback
    return (analysis.search_query or analysis.original or fallback).strip() or fallback


def _min_max(scores: dict[str, float]) -> dict[str, float]:
    """min-max 归一化到 [0,1]；全部相等时统一给 1.0（说明无从区分）。"""
    if not scores:
        return {}
    low = min(scores.values())
    high = max(scores.values())
    if high - low <= 1e-12:
        return {key: 1.0 for key in scores}
    span = high - low
    return {key: (value - low) / span for key, value in scores.items()}


#: 子块切分的句末标点
SEGMENT_SPLIT = re.compile(r"[。！？；!?;\n]")


def split_segments(
    content: str, max_chars: int = 180, min_chars: int = 20, max_segments: int = 12
) -> list[str]:
    """把块内容切成"子块"（检索用细粒度单元，映射回父块）。

    为什么这样切：实测证据句在 400~600 字符的整块内向量余弦仅 0.50，
    单独编码同一证据句为 0.69（长块把答案句稀释掉）。子块按句末标点聚合到
    ``max_chars`` 以内，既保证语义完整，又避免过碎（碎片会降低可读性）。
    """
    raw = (content or "").strip()
    if not raw:
        return []
    sentences = [seg.strip() for seg in SEGMENT_SPLIT.split(raw) if seg and seg.strip()]
    if not sentences:
        return [raw[:max_chars]] if len(raw) >= min_chars else []
    segments: list[str] = []
    buffer = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if buffer:
                segments.append(buffer)
                buffer = ""
            for start in range(0, len(sentence), max_chars):
                piece = sentence[start : start + max_chars]
                if len(piece) >= min_chars:
                    segments.append(piece)
            continue
        if len(buffer) + len(sentence) + 1 <= max_chars:
            buffer = f"{buffer}{sentence}。" if buffer else f"{sentence}。"
        else:
            if buffer:
                segments.append(buffer)
            buffer = f"{sentence}。"
    if buffer:
        segments.append(buffer)
    cleaned = [seg for seg in segments if len(seg) >= min_chars]
    return cleaned[:max_segments]


def _product(values: Any) -> float:
    """乘性汇总（空集合返回 1.0）。"""
    result = 1.0
    for value in values:
        result *= float(value)
    return result


_retriever: Retriever | None = None
_lock = threading.Lock()


def get_retriever() -> Retriever:
    """获取进程级检索器单例（含重排器）。"""
    global _retriever
    with _lock:
        if _retriever is None:
            from app.core.reranker import get_reranker  # 局部导入避免循环依赖

            _retriever = Retriever(reranker=get_reranker())
    return _retriever


def reset_retriever() -> None:
    """重置单例（测试用）。"""
    global _retriever
    with _lock:
        _retriever = None
