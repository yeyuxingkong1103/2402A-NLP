"""检索器：混合检索（向量 + BM25）+ 领域加权 + 可选重排。

工单要求（5.3）：
- 向量 top_k=10 + BM25 top_k=10，合并去重；
- 可选重排（bge-reranker-base），取 top_n=5；
- 针对《招股说明书1.pdf》的特攻优化：
  * 表格内容单独建索引（表格 chunk 在融合时获得额外权重）；
  * 对“收入/占比/注册资本/法定代表人/募集资金/上下游/供应商/客户/技术标准/
    科技进步奖”等领域关键词加权；
  * 支持按页码过滤。

融合策略：加权线性融合 + 领域词加成 + 表格加成，全部归一化到 0~1，
便于用 ``min_relevance_score`` 判定“无相关内容”。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.core.bm25_index import BM25Index
from app.core.config import get_settings
from app.core.embedder import Embedder, get_embedder
from app.core.logging_conf import logger, trace
from app.core.text_utils import STOPWORDS, tokenize
from app.core.vector_store import VectorStore, get_vector_store
from app.models.schemas import Chunk, QueryAnalysis, RetrievedChunk

try:  # pragma: no cover
    from sentence_transformers import CrossEncoder

    HAS_CROSS_ENCODER = True
except Exception:  # pragma: no cover
    CrossEncoder = None  # type: ignore
    HAS_CROSS_ENCODER = False


@dataclass
class RetrievalDebug:
    """检索过程的可观测信息（只写日志，不返回给用户）。"""

    vector_hits: int = 0
    bm25_hits: int = 0
    merged_hits: int = 0
    reranked: bool = False
    page_filter: list[int] = field(default_factory=list)
    boosted_chunks: int = 0
    penalized_chunks: int = 0
    fragment_chunks: int = 0
    top_score: float = 0.0


# 招股书中的“术语定义 / 释义”类样板内容：几乎每页之外的前置章节都在重复公司全称，
# 任何查询都会把它们顶到前面，从而挤掉真正的证据段落。这里显式降权。
BOILERPLATE_MARKERS: tuple[str, ...] = (
    "基本用语",
    "释义",
    "本招股意向书、招",
    "指",
)
COMPANY_NAME = "武汉兴图新科电子股份有限公司"


def boilerplate_penalty(content: str) -> tuple[float, bool]:
    """计算样板内容惩罚系数。

    判定依据（同时满足才算样板）：
    1. 公司全称在片段中**反复出现**（≥3 次），或片段带有明显的释义表结构；
    2. 片段本身很短（释义条目通常很短）。

    Returns:
        ``(惩罚系数, 是否判定为样板)``；惩罚系数 ≤ 1.0。
    """
    if not content:
        return 1.0, False
    hits = content.count(COMPANY_NAME)
    looks_like_glossary = any(marker in content for marker in BOILERPLATE_MARKERS) and content.count("|") >= 4
    if hits >= 3 or (looks_like_glossary and hits >= 1):
        # 重复度越高惩罚越重，但最多降到 0.35，避免把真正的定义类证据彻底排除
        factor = max(0.35, 1.0 - 0.12 * hits)
        return factor, True
    return 1.0, False


# 片段以这些字符/词开头，说明它是从句子中间被切出来的（滑窗切分或 PDF 断页所致）
_FRAGMENT_START_MARKERS: tuple[str, ...] = ("（即", "(即", "即相当于", "一体化工程", "系统）", "体系）", "等）")
_CLOSERS = "）)】」》”"


def fragment_penalty(content: str) -> tuple[float, bool]:
    """惩罚“从半句开始”的片段，让完整句子优先进入提示词。

    背景：招股书同一段结论会在多个章节重复出现，其中某一次恰好被滑窗切成
    ``美军的C4ISR系统）荣获…`` 这种从半句开始的碎片。碎片本身没有错，
    但作为答案依据会让回答读起来不完整，因此适度降权。

    Returns:
        ``(惩罚系数, 是否判定为碎片)``。
    """
    text = (content or "").lstrip()
    if not text:
        return 1.0, False
    first = text[0]
    # 情况一：以右括号/书名号收尾符开头，且没有对应的左括号
    if first in _CLOSERS:
        return 0.55, True
    # 情况二：以续写性词语开头
    if any(text.startswith(marker) for marker in _FRAGMENT_START_MARKERS):
        return 0.55, True
    # 情况三：首行是表格标题行之外的纯续写（以“、”或“，”开头）
    if first in "、，,":
        return 0.75, True
    return 1.0, False


class Retriever:
    """混合检索器。"""

    def __init__(
        self,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        bm25_index: BM25Index | None = None,
    ) -> None:
        self.settings = get_settings()
        self.embedder = embedder or get_embedder()
        self.vector_store = vector_store or get_vector_store()
        self.bm25 = bm25_index or BM25Index()
        self._chunks: dict[str, Chunk] = {}
        self._reranker: object | None = None
        self._reranker_failed = False
        # 最近一次检索的原始余弦相似度最高值（未归一化、未加权）。
        # ``is_confident`` 依赖它做“能否回答”的判定——归一化后的融合分数
        # 恒有最大值 1.0，无法用来判断问题是否真的与文档相关。
        self.last_top_cosine: float = 0.0

    # ------------------------------------------------------------------
    # 索引装载
    # ------------------------------------------------------------------
    def set_chunks(self, chunks: list[Chunk]) -> None:
        """注册全量 chunk（供结果回填与关键词加成判断）。"""
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}

    def chunk_of(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    @trace
    def build_index(self, chunks: list[Chunk], reset: bool = True) -> dict[str, object]:
        """建立向量索引与 BM25 索引。

        Args:
            chunks: 全部待索引分块。
            reset: 是否先清空向量库。重建同一份内容时传 ``False`` 更快，
                因为 chunk_id 稳定，``upsert`` 本身就是幂等的。
        """
        self.set_chunks(chunks)
        if not chunks:
            logger.warning("app.core.retriever", "空 chunk 列表，索引未建立")
            return {"vector": 0, "bm25": 0}

        vectors = self.embedder.encode([chunk.content for chunk in chunks])
        if reset:
            self.vector_store.reset()
        added = self.vector_store.add(chunks, vectors)
        self.bm25.build(chunks)
        info = {
            "vector": added,
            "bm25": self.bm25.size,
            "dimension": int(vectors.shape[1]) if vectors.size else 0,
            "embedder": self.embedder.name,
            "reset": reset,
        }
        logger.info("app.core.retriever", "索引构建完成", **info)
        return info

    def save_index(self) -> dict[str, str]:
        """持久化 BM25 与向量库。"""
        bm25_path = self.bm25.save()
        vector_path = self.vector_store.save()
        return {"bm25": str(bm25_path), "vector": str(vector_path) if vector_path else ""}

    def load_index(self, chunks: list[Chunk]) -> bool:
        """从磁盘加载索引；chunk 内容来自 SQLite（避免重复嵌入）。

        Returns:
            ``True`` 表示向量库与 BM25 都已从磁盘恢复；
            ``False`` 表示至少有一项是**就地重建**的（磁盘索引缺失）。

        注意：重建时 ``reset=False``，因为 chunk_id 稳定，``upsert`` 幂等；
        而且向量库（Chroma）是持久化的，清空会造成不必要的重算。
        """
        self.set_chunks(chunks)

        # ---- BM25：优先从磁盘加载 ----
        if self.bm25.size == 0:
            loaded_bm25 = BM25Index.load()
            if loaded_bm25.size:
                self.bm25 = loaded_bm25
            else:
                logger.warning("app.core.retriever", "BM25 索引文件缺失，按 SQLite 内容重建")
                self.bm25.build(chunks)

        # ---- 向量库：Chroma 自动持久化；numpy 后端需要显式 load ----
        vector_ok = self.vector_store.load()
        if not vector_ok:
            if chunks:
                logger.warning("app.core.retriever", "向量库为空，按 SQLite 内容重建向量索引")
                self.build_index(chunks, reset=False)

        ready = vector_ok and self.bm25.size > 0
        logger.info(
            "app.core.retriever",
            "索引装载结果",
            vector_backend=self.vector_store.name,
            vector_count=self.vector_store.count(),
            bm25_documents=self.bm25.size,
            restored_from_disk=ready,
        )
        return ready

    # ------------------------------------------------------------------
    # 重排
    # ------------------------------------------------------------------
    def _get_reranker(self) -> object | None:
        """惰性加载重排模型；失败只记录一次，不反复重试。"""
        if self._reranker is not None or self._reranker_failed:
            return self._reranker
        if not self.settings.retrieval.use_reranker or not HAS_CROSS_ENCODER:
            self._reranker_failed = True
            if self.settings.retrieval.use_reranker and not HAS_CROSS_ENCODER:
                logger.warning("app.core.retriever", "未安装 sentence-transformers，跳过重排")
            return None
        try:
            self._reranker = CrossEncoder(self.settings.retrieval.reranker_model, device=self.settings.embedding.device)
            logger.info("app.core.retriever", "重排模型加载完成", model=self.settings.retrieval.reranker_model)
        except Exception as exc:
            self._reranker_failed = True
            logger.warning(
                "app.core.retriever",
                "重排模型加载失败，本次运行不再重排",
                model=self.settings.retrieval.reranker_model,
                error=f"{type(exc).__name__}: {exc}",
            )
        return self._reranker

    # ------------------------------------------------------------------
    # 领域加权
    # ------------------------------------------------------------------
    def _keyword_boost(self, chunk: Chunk, query: str) -> tuple[float, float]:
        """计算领域关键词加成系数。

        只有当关键词**同时**出现在问题和 chunk 中时才加权，
        避免无差别地抬高所有含该词的片段。

        Returns:
            ``(综合加成系数, 命中的最大关键词权重)``；后者用于诊断日志。
        """
        boost = 1.0
        max_weight = 1.0
        for keyword, weight in self.settings.retrieval.keyword_boost.items():
            if keyword in query and keyword in chunk.content:
                boost *= weight
                max_weight = max(max_weight, weight)
        # 表格片段加成：招股书的关键数据大多在表格里
        if chunk.type == "table":
            boost *= self.settings.retrieval.table_boost
        return boost, max_weight

    # ------------------------------------------------------------------
    # 主检索
    # ------------------------------------------------------------------
    @trace
    def retrieve(
        self,
        query: str,
        analysis: QueryAnalysis | None = None,
        top_k: int | None = None,
        page_filter: list[int] | None = None,
    ) -> list[RetrievedChunk]:
        """执行混合检索，返回按分数降序的候选片段。"""
        if not query.strip():
            logger.warning("app.core.retriever", "空查询，返回空结果")
            return []

        settings = self.settings.retrieval
        debug = RetrievalDebug()
        # 必须使用调用方传入的查询串。
        #
        # 这里曾错误地写成 ``analysis.rewritten or query``，导致多查询变体检索
        # 全部退化成“用同一个 original 问题检索”——每个变体算出的余弦完全一样，
        # 多路召回形同虚设，英文问题的中文模板也被英文原文覆盖掉。
        # 历史改写已经由 QueryUnderstanding 写入 rewritten，并作为首选变体传入，
        # 因此检索层不再做任何替换。
        effective_query = query
        pages = page_filter or (analysis.page_filter if analysis else []) or []
        debug.page_filter = list(pages)

        # ---- 1. 向量召回 ----
        query_vector = self.embedder.encode_one(effective_query)
        vector_hits = self.vector_store.search(query_vector, top_k=settings.vector_top_k)
        debug.vector_hits = len(vector_hits)
        # 记录原始余弦最高值：这是“问题与文档是否相关”的最可靠信号
        self.last_top_cosine = max((float(score) for _, score in vector_hits), default=0.0)

        # ---- 2. BM25 召回 ----
        bm25_hits = self.bm25.search(effective_query, top_k=settings.bm25_top_k)
        debug.bm25_hits = len(bm25_hits)

        if not vector_hits and not bm25_hits:
            logger.warning(
                "app.core.retriever",
                "向量与 BM25 均无命中",
                query=query,
                index_size=self.vector_store.count(),
            )
            return []

        # ---- 3. 融合去重 ----
        fused: dict[str, RetrievedChunk] = {}
        for chunk_id, score in vector_hits:
            chunk = self._chunks.get(chunk_id)
            if chunk is None:
                continue
            fused[chunk_id] = RetrievedChunk(
                chunk=chunk,
                score=settings.vector_weight * float(score),
                vector_score=float(score),
                source="vector",
            )
        for chunk_id, score in bm25_hits:
            chunk = self._chunks.get(chunk_id)
            if chunk is None:
                continue
            existing = fused.get(chunk_id)
            if existing:
                existing.bm25_score = float(score)
                existing.score += settings.bm25_weight * float(score)
                existing.source = "hybrid"
            else:
                fused[chunk_id] = RetrievedChunk(
                    chunk=chunk,
                    score=settings.bm25_weight * float(score),
                    bm25_score=float(score),
                    source="bm25",
                )
        debug.merged_hits = len(fused)

        # ---- 4. 领域加权 ----
        for item in fused.values():
            boost, _max_weight = self._keyword_boost(item.chunk, effective_query)
            if boost > 1.0:
                item.score *= boost
                debug.boosted_chunks += 1

        # ---- 4.1 样板内容与半句碎片降权 ----
        for item in fused.values():
            factor, is_boilerplate = boilerplate_penalty(item.chunk.content)
            fragment_factor, is_fragment = fragment_penalty(item.chunk.content)
            combined = factor * fragment_factor
            if combined < 1.0:
                item.score *= combined
                if is_boilerplate:
                    debug.penalized_chunks += 1
                if is_fragment:
                    debug.fragment_chunks += 1

        # ---- 5. 页码过滤 ----
        candidates = list(fused.values())
        if pages:
            allowed = set(pages)
            filtered = [item for item in candidates if item.chunk.page in allowed]
            if filtered:
                candidates = filtered
            else:
                logger.warning(
                    "app.core.retriever",
                    "页码过滤后无候选，忽略过滤条件",
                    page_filter=pages,
                    before=len(candidates),
                )

        candidates.sort(key=lambda item: item.score, reverse=True)
        candidates = candidates[: settings.fusion_top_k]

        # ---- 6. 归一化分数（方便阈值判断） ----
        if candidates:
            top = candidates[0].score or 1.0
            for item in candidates:
                item.score = item.score / top
            debug.top_score = float(candidates[0].score)

        # ---- 7. 可选重排 ----
        reranker = self._get_reranker()
        if reranker is not None and candidates:
            try:
                pairs = [(effective_query, item.chunk.content) for item in candidates]
                scores = reranker.predict(pairs)  # type: ignore[attr-defined]
                for item, score in zip(candidates, np.asarray(scores, dtype=np.float32)):
                    item.rerank_score = float(score)
                candidates.sort(key=lambda item: item.rerank_score or 0.0, reverse=True)
                if candidates:
                    top = candidates[0].rerank_score or 1.0
                    for item in candidates:
                        item.rerank_score = (item.rerank_score or 0.0) / top
                        item.score = item.rerank_score
                debug.reranked = True
            except Exception as exc:
                logger.warning("app.core.retriever", "重排执行失败，沿用融合分数", error=f"{type(exc).__name__}: {exc}")

        limit = top_k or settings.rerank_top_n
        result = candidates[:limit]

        logger.info(
            "app.core.retriever",
            "检索完成",
            query=query,
            rewritten=effective_query if effective_query != query else None,
            vector_hits=debug.vector_hits,
            bm25_hits=debug.bm25_hits,
            merged=debug.merged_hits,
            boosted=debug.boosted_chunks,
            penalized=debug.penalized_chunks,
            fragments=debug.fragment_chunks,
            reranked=debug.reranked,
            returned=len(result),
            top_score=round(debug.top_score, 4),
            pages=[item.chunk.page for item in result],
        )
        return result

    # ------------------------------------------------------------------
    @trace
    def retrieve_multi(
        self,
        queries: list[str] | list[tuple[str, str]],
        analysis: QueryAnalysis | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """多查询变体检索：逐个召回后按 chunk 取最高分合并。

        为什么需要：单条查询串的措辞会显著影响召回。实测同一份索引下，

        - ``武汉兴图新科电子股份有限公司的注册资本是多少 注册资本`` -> 最高余弦 0.539
          （公司全称与“注册资本”都在“释义”章节高频出现，把语义方向拉偏）
        - ``注册资本是多少`` -> 最高余弦 0.694（命中正确页）

        因此对同一问题给出多个变体（中文问句模板 / 关键词 / 去主体名版本），
        各自检索后合并，以**最优结果**为准，鲁棒性明显好于赌单个措辞。
        单条查询时等价于 ``retrieve``。

        Args:
            queries: 查询串列表，或 ``[(查询串, 变体类型)]``。
                     变体类型为 ``"clean"``（干净问句，直接检索、不加关键词）
                     或 ``"keyword"``（关键词串）。
        """
        variants: list[str] = []
        clean_queries: set[str] = set()
        for item in queries:
            query, kind = item if isinstance(item, tuple) else (item, "clean")
            cleaned = (query or "").strip()
            if not cleaned or cleaned in variants:
                continue
            variants.append(cleaned)
            if kind == "clean":
                clean_queries.add(cleaned)

        if not variants:
            return []

        merged: dict[str, RetrievedChunk] = {}
        best_cosine = 0.0

        # 先跑一遍拿到每个变体的“语义置信度”（最高原始余弦），
        # 再用**相对置信度**加权后再合并。
        #
        # 为什么不能直接按归一化分数取最大：每个变体的 top1 都会被归一化成 1.0，
        # 于是一个“词袋式关键词变体”（例如 “供应商 客户 武汉兴图新科电子股份有限公司”
        # 的余弦 0.829）会与真正贴题的变体（余弦 0.762）等权，
        # 把命中公司名的“释义/基本情况”页顶到最前面（实测 p22 被顶到第一）。
        # 按相对置信度加权后，语义方向更准的变体自然占优。
        variant_scores: list[tuple[str, float, list[RetrievedChunk]]] = []
        for query in variants:
            results = self.retrieve(query, analysis=analysis, top_k=top_k)
            cosine = self.last_top_cosine
            best_cosine = max(best_cosine, cosine)
            variant_scores.append((query, cosine, results))

        def weight_of(cosine: float) -> float:
            """相对于最强变体的权重（平方衰减，弱变体贡献迅速变小）。"""
            if best_cosine <= 0:
                return 1.0
            ratio = max(cosine, 0.0) / best_cosine
            return ratio * ratio

        for query, cosine, results in variant_scores:
            weight = weight_of(cosine)
            for item in results:
                weighted = item.score * weight
                current = merged.get(item.chunk.chunk_id)
                if current is None or weighted > current.score:
                    merged[item.chunk.chunk_id] = item.model_copy(update={"score": weighted})

        # 置信度取所有变体的最优原始余弦，避免某个措辞把整体判死
        self.last_top_cosine = best_cosine
        ordered = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        limit = top_k or self.settings.retrieval.rerank_top_n
        logger.info(
            "app.core.retriever",
            "多查询变体检索完成",
            variants=variants,
            variant_cosines=[round(cosine, 4) for _, cosine, _ in variant_scores],
            merged=len(ordered),
            best_cosine=round(best_cosine, 4),
            returned=min(limit, len(ordered)),
        )
        return ordered[:limit]

    def is_answerable(self, question: str, results: list[RetrievedChunk]) -> bool:
        """检查证据片段是否真的覆盖了问题里的实义词。

        与 ``is_confident``（只看整体语义相似度）互补：相似度会被“同一份文档、
        同一家公司”这种背景相似抬高，而实义词覆盖能识别出“问的东西语料里没有”。

        例：多轮追问「那知识产权呢？」改写后仍带上一轮的“募集资金”主题，
        检索余弦很高，但“知识产权”一词在所有候选片段里都不存在，
        此时应当回复“不清楚”而不是把上一轮的答案再答一遍。
        """
        required = self.settings.retrieval.min_overlap_terms
        if required <= 0 or not results or not question:
            return True

        terms = [
            term
            for term in tokenize(question)
            if len(term) >= 2 and term not in STOPWORDS
        ]
        if not terms:
            return True

        haystack = "".join(item.chunk.content for item in results)
        hit = sum(1 for term in terms if term in haystack)
        answerable = hit >= required
        if not answerable:
            logger.warning(
                "app.core.retriever",
                "证据片段未覆盖问题的实义词，判定为无法回答",
                question_terms=terms[:8],
                hit=hit,
                required=required,
            )
        return answerable

    def is_confident(self, results: list[RetrievedChunk]) -> bool:
        """判断检索结果是否足以回答（否则触发“不清楚”兜底）。

        判据：**原始余弦相似度**是否达到 ``min_confidence_cosine``。

        这里刻意不使用融合分数：融合分数在 ``retrieve`` 里被归一化到 1.0，
        任何问题（包括“今天天气怎么样”）都会得到 1.0，从而永远判为“有把握”。
        原始余弦不会被归一化，实测本语料相关问题 ≥0.737、无关问题 ≤0.391，
        区分度充足。
        """
        if not results:
            logger.warning("app.core.retriever", "无检索结果，判定为无法回答")
            return False

        threshold = self.settings.retrieval.min_confidence_cosine
        cosine = self.last_top_cosine
        confident = cosine >= threshold
        if not confident:
            logger.warning(
                "app.core.retriever",
                "最高原始余弦低于阈值，判定为无法回答（将回复“不清楚”）",
                top_cosine=round(cosine, 4),
                threshold=threshold,
                query_pages=[item.chunk.page for item in results[:3]],
            )
        return confident

    def health(self) -> dict[str, object]:
        """检索链路健康状态。"""
        return {
            "embedder": self.embedder.health(),
            "vector_store": self.vector_store.health(),
            "bm25_documents": self.bm25.size,
            "chunks_registered": len(self._chunks),
            "rerank_enabled": self.settings.retrieval.use_reranker,
            "rerank_available": HAS_CROSS_ENCODER,
        }
