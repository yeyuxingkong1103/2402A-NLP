# -*- coding: utf-8 -*-
"""工单4 混合检索（FAISS 向量 + 自实现 BM25 + RRF 融合 + 轻量重排 + 证据页补全）。

工单编号：人工智能NLP-RAG-图像内容解析及检索优化
工单6 增补：人工智能NLP-RAG-混合检索任务
    —— 新增 ``strategy`` / ``strategy_diag`` / ``use_cache`` 三个形参（检索策略层接入、
    诊断透传、缓存作用域修正），以及 ``_retrieve_by_strategy`` 与 ``_strategy_fingerprint``。
    **不传 ``strategy`` 时仍走 工单4 原生 ``hybrid_search``，该路径未改动。**

设计依据：
    * 设计/接口设计.md §4.2：``retrieve(question, top_k=5, modality=None, use_image_channel=True)``；
      融合用 **RRF**，**禁止把不同语义空间的相似度直接相加**；
    * 任务书性能策略：问题级缓存、预热、Top-K 检索、FAISS + BM25 混合 + 轻量重排；
    * 任务书 ★A：枚举/多点数值题必须逐项完整 ⇒ 检索负责「把证据页捞全」，
      由 ``expand_with_page_context``（证据页补全）把 Top 命中所在页的同类块补进上下文。

索引来源（T4 产出，只读使用，不修改）：
    ``研发/data/index/bge-m3_1024/``：``vectors.npy`` / ``ids.json`` / ``faiss.index`` /
    ``bm25.pkl`` / ``chunks_indexed.jsonl``（行序 == vectors.npy 行序）。

显式纪律：
    * 不重写 T4 的索引文件；本模块只读；
    * 索引缺失抛 ``IndexMissingError``（UI/CLI 据此给退出码 3）；
    * 图像块（``type == "image"``）是**一等检索对象**，题 5/6 的证据就在图像块里。
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .bm25_index import BM25Index
from .cache import as_rows_major, get_cache
from .citation import PAGE_BASIS
from .config import AppConfig, get_config, model_slug
from .embedder import embed_query
from .errors import IndexMissingError, RetrievalError
from .reranker import rerank
from .text_utils import normalize_text
from .vector_store import LoadedVectorIndex, load_vector_index, search_vectors

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"

CHUNKS_FILE = "chunks_indexed.jsonl"
_ISSUER_RE = re.compile(r"[\u4e00-\u9fff]{2,12}(?:股份)?有限公司")

# ---------------------------------------------------------------------------
# T7 检索侧调参（**只在本模块内生效**，不动 config.py 的既有默认值）
# ---------------------------------------------------------------------------
#: 通道权重因子。T7 实测（16 题全量、extractive 口径扫描 vf×bf = 0.70×1.20 / 0.85×1.10 /
#: 1.00×1.00 / 0.60×1.40 / 0.50×1.60）：**重加权并不能提升权威层命中**（0.70×1.20 反而把
#: 题 33 的「军种销售收入占比」块顶进 top-3，导致口径答错），因此保留 1.0×1.0，
#: 向量侧的增益改由「查询富化」（见 hybrid_search 的 vector_query）获得 —— 这是实测结论，不是默认值。
VECTOR_FACTOR: float = float(os.environ.get("RAG_T7__VECTOR_FACTOR", "1.00"))
BM25_FACTOR: float = float(os.environ.get("RAG_T7__BM25_FACTOR", "1.00"))
#: 重排预算：只对 RRF 融合分最高的前 N 条做特征重排（候选裁剪），并按相对分阈值提前截断
RERANK_BUDGET: int = int(os.environ.get("RAG_T7__RERANK_BUDGET", "60"))
RERANK_MIN_RELATIVE: float = float(os.environ.get("RAG_T7__RERANK_MIN_RELATIVE", "0.02"))
#: 图像块证据权重（T4 定义：diagram/chart=1.0 强证据、picture=0.4 弱证据）
DEFAULT_EVIDENCE_WEIGHT: float = float(os.environ.get("RAG_T7__EVIDENCE_WEIGHT_DEFAULT", "1.0"))

#: 强指向性字段（问这些字段时必须能确定「问的是哪一家」）
STRONG_FIELD_TERMS: tuple[str, ...] = (
    "注册资本", "实收资本", "发行股数", "发行后总股本", "法定代表人", "控股股东",
    "实际控制人", "成立日期", "每股面值", "保荐人", "主承销商",
)
#: 公司实体样式的短语（用于「问句点名的主体是否在语料内」判定）
_COMPANY_LIKE = re.compile(r"[\u4e00-\u9fffA-Za-z]{2,24}(?:股份有限公司|有限公司|公司|集团)")
#: **泛化名 / 占位名**：不是真实主体，任何答案都不可能有依据（T6 N1 实测踩坑）
_GENERIC_NAME = re.compile(
    r"(某某|某(?:公司|集团|科技|股份|有限|企业|单位|机构|银行|医院|学校)"
    r"|[xｘXＸ]{2,}\s*(?:公司|集团|科技|corp|ltd)?"
    r"|(?:甲|乙|丙|丁|戊)公司|A\s*公司|B\s*公司"
    r"|company\s*[xyx]|xyz\s*(?:corp|ltd|company)?|abc\s*(?:corp|ltd|company))",
    re.IGNORECASE)


def _lazy_logger(logger: Any, module: str = "retriever") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


class Hit(dict):
    """一条检索命中（``dict`` 子类：既能直接 JSON 序列化，也能属性访问）。"""

    __slots__ = ()

    @property
    def chunk_id(self) -> str:
        return str(self.get("chunk_id") or "")

    @property
    def page(self) -> int:
        return int(self.get("page") or 0)

    @property
    def file_name(self) -> str:
        return str(self.get("file_name") or "")

    @property
    def modality(self) -> str:
        return str(self.get("type") or self.get("modality") or "text")

    @property
    def score(self) -> float:
        return float(self.get("score") or self.get("final_score") or 0.0)

    def to_dict(self) -> dict[str, Any]:
        return dict(self)


@dataclass
class IndexBundle:
    """内存态索引束（块表 + 向量 + BM25 + 页索引 + 主体表）。"""

    index_dir: str
    chunks: dict[str, dict[str, Any]]
    order: list[str]
    vectors: Any
    vector_index: LoadedVectorIndex
    bm25: BM25Index
    manifest: dict[str, Any]
    corpus_issuers: dict[str, list[str]] = field(default_factory=dict)
    pages: dict[tuple[str, int], list[str]] = field(default_factory=dict)
    corpus_of_file: dict[str, str] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.chunks)

    def to_dict(self) -> dict[str, Any]:
        return {"index_dir": self.index_dir, "chunks": self.size,
                "vector": self.vector_index.to_dict(), "corpora": self.corpus_of_file,
                "corpus_issuers": self.corpus_issuers, "pages": len(self.pages)}

    def chunk(self, chunk_id: str) -> dict[str, Any] | None:
        return self.chunks.get(chunk_id)

    def page_chunks(self, file_name: str, page: int) -> list[dict[str, Any]]:
        """某文件某页的全部块（按 ``order`` 升序）。"""
        ids = self.pages.get((str(file_name), int(page)), [])
        return [self.chunks[cid] for cid in ids if cid in self.chunks]


_BUNDLE_CACHE: dict[str, IndexBundle] = {}
_BUNDLE_LOCK = threading.Lock()


def corpus_key_of(file_name: str) -> str:
    """文件名 → 语料键（``招股说明书1.pdf`` → ``pdf1``；与 golden ``corpus`` 口径一致）。"""
    match = re.search(r"(\d+)", str(file_name))
    return f"pdf{match.group(1)}" if match else str(file_name)


def build_corpus_issuers(chunks: Iterable[Mapping[str, Any]], *, limit: int = 3) -> dict[str, list[str]]:
    """从块内容自动统计每份语料的发行人名字（**禁止硬编码公司名**）。"""
    counter: dict[str, dict[str, int]] = {}
    for chunk in chunks:
        if str(chunk.get("type")) != "text":
            continue
        corpus = corpus_key_of(str(chunk.get("file_name") or ""))
        text = str(chunk.get("content") or "")
        for name in _ISSUER_RE.findall(text[:400]):
            counter.setdefault(corpus, {})
            counter[corpus][name] = counter[corpus].get(name, 0) + 1
    out: dict[str, list[str]] = {}
    for corpus, names in counter.items():
        out[corpus] = [n for n, _ in sorted(names.items(), key=lambda kv: -kv[1])[:limit]]
    return out


def load_bundle(cfg: AppConfig | None = None, *, force: bool = False, logger: Any = None) -> IndexBundle:
    """加载（并进程内缓存）索引束；索引缺失抛 ``IndexMissingError``。"""
    config = cfg or get_config()
    log = _lazy_logger(logger)
    slug = model_slug(config.llm.ollama_embed_model, 1024)
    index_dir = Path(config.paths.index_dir) / slug
    key = str(index_dir)
    with _BUNDLE_LOCK:
        if not force and key in _BUNDLE_CACHE:
            return _BUNDLE_CACHE[key]
    with log.enter("load_bundle", {"index_dir": str(index_dir), "force": force}) as span:
        started = time.perf_counter()
        chunks_path = index_dir / CHUNKS_FILE
        if not chunks_path.is_file():
            raise IndexMissingError(f"索引块文件缺失：{chunks_path}（先跑 研发/scripts/build_index.py）",
                                    detail={"index_dir": str(index_dir), "file": CHUNKS_FILE})
        chunks: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        pages: dict[tuple[str, int], list[str]] = {}
        with chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                cid = str(row.get("chunk_id"))
                chunks[cid] = row
                order.append(cid)
                page = int(row.get("page") or 0)
                pages.setdefault((str(row.get("file_name") or ""), page), []).append(cid)
        for ids in pages.values():
            ids.sort(key=lambda cid: int(chunks[cid].get("order") or 0))
        vector_index = load_vector_index(index_dir, expected_dim=int(
            json.loads((index_dir / "ids.json").read_text(encoding="utf-8")).get("dim") or 1024),
            logger=log, use_faiss=bool(config.vector_store.use_faiss))
        bm25 = BM25Index.load(index_dir, logger=log)
        manifest: dict[str, Any] = {}
        manifest_path = Path(config.paths.index_dir) / "index_manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:  # 显式降级：清单坏不影响检索
                log.log_event("index.manifest_unreadable", level="WARNING", path=str(manifest_path),
                              error=f"{type(exc).__name__}: {exc}")
        corpus_of_file = {str(name): corpus_key_of(str(name))
                          for name in {str(c.get("file_name") or "") for c in chunks.values()}}
        # T7 内存布局优化：向量矩阵整理为 C 连续 float32 —— 点积/FAISS add 都按行主序访问，
        # 非连续切片会在每次检索多做一次拷贝（2676×1024 规模下是「免费」的确定性收益）。
        vector_index.vectors = as_rows_major(vector_index.vectors)
        bundle = IndexBundle(
            index_dir=str(index_dir), chunks=chunks, order=order, vectors=vector_index.vectors,
            vector_index=vector_index, bm25=bm25, manifest=manifest,
            corpus_issuers=build_corpus_issuers(chunks.values()), pages=pages,
            corpus_of_file=corpus_of_file,
        )
        with _BUNDLE_LOCK:
            _BUNDLE_CACHE[key] = bundle
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        # T7 启动预热：加载即做一次「向量 + BM25」空转检索，把 FAISS 反序列化后的首访、
        # jieba 倒排表与 numpy 缓存都热起来（首问延迟不再含这部分一次性成本）。
        warm = warmup_retrieval(bundle, logger=log)
        log.log_event("index.bundle_loaded", chunks=bundle.size, dim=vector_index.dim,
                      faiss=vector_index.faiss_index is not None, elapsed_ms=elapsed_ms,
                      contiguous=bool(getattr(vector_index.vectors, "flags", None)
                                      and vector_index.vectors.flags["C_CONTIGUOUS"]),
                      warmup_ms=warm.get("elapsed_ms"),
                      corpora=corpus_of_file, issuers=bundle.corpus_issuers,
                      chunk_by_modality=manifest.get("chunk_by_modality"))
        span.set_output({"chunks": bundle.size, "dim": vector_index.dim,
                         "elapsed_ms": elapsed_ms, "pages": len(pages)})
        return bundle


def clear_bundle_cache() -> None:
    """清空索引束缓存（测试与索引重建后使用）。"""
    with _BUNDLE_LOCK:
        _BUNDLE_CACHE.clear()
    _CORPUS_ID_CACHE.clear()


def warmup_retrieval(bundle: IndexBundle, *, logger: Any = None) -> dict[str, Any]:
    """预热检索通道：一次向量空转 + 一次 BM25 空转 + 语料 id 集合预建。

    依据（T7 目标「索引加载与预热优化」）：FAISS 反序列化后的首次 ``search``、BM25 倒排表的首访、
    以及「语料路由」的 id 集合构建都属于**一次性成本**；把它们挪到加载期，首问延迟不再承担。
    """
    import numpy as np

    log = _lazy_logger(logger)
    started = time.perf_counter()
    out: dict[str, Any] = {}
    try:
        dummy = np.zeros(int(bundle.vector_index.dim), dtype=np.float32)
        dummy[0] = 1.0
        hits = search_vectors(bundle.vector_index, dummy, 3, logger=log)
        out["vector_warm_hits"] = len(hits)
    except Exception as exc:  # noqa: BLE001 —— 预热失败不得阻断加载（显式留痕）
        out["vector_error"] = f"{type(exc).__name__}: {exc}"
        log.log_event("retrieval.warmup_vector_failed", level="WARNING", error=out["vector_error"])
    try:
        out["bm25_warm_hits"] = len(bundle.bm25.search("注册资本", top_k=3, logger=log))
    except Exception as exc:  # noqa: BLE001
        out["bm25_error"] = f"{type(exc).__name__}: {exc}"
        log.log_event("retrieval.warmup_bm25_failed", level="WARNING", error=out["bm25_error"])
    for corpus in set(bundle.corpus_of_file.values()):
        _corpus_ids(bundle, corpus)
    out["corpus_id_sets"] = {c: len(ids) for c, ids in _CORPUS_ID_CACHE.items() if c != "*"}
    out["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    log.log_event("retrieval.warmed", **out)
    return out


_CORPUS_ID_CACHE: dict[str, set[str]] = {}


def _corpus_ids(bundle: IndexBundle, corpus: str) -> set[str]:
    """语料 → chunk_id 集合（**带缓存**）：避免每次问句都全量扫 2676 条块。"""
    key = f"{bundle.index_dir}|{corpus}"
    cached = _CORPUS_ID_CACHE.get(key)
    if cached is None:
        cached = {cid for cid, row in bundle.chunks.items()
                  if corpus_key_of(str(row.get("file_name") or "")) == str(corpus)}
        _CORPUS_ID_CACHE[key] = cached
    return cached


def _rrf_channel(hits: Sequence[tuple[str, float]], weight: float, rrf_k: int) -> dict[str, float]:
    """单通道 RRF 贡献：``weight / (rrf_k + rank)``（rank 从 1 开始）。"""
    return {cid: weight / (rrf_k + rank) for rank, (cid, _score) in enumerate(hits, 1)}


def hybrid_search(bundle: IndexBundle, question: str, *, cfg: AppConfig,
                  analysis: Any = None, top_k: int = 5, modality: str | None = None,
                  use_image_channel: bool = True, allowed_ids: set[str] | None = None,
                  logger: Any = None) -> list[dict[str, Any]]:
    """混合检索主流程，返回重排后的候选列表（dict，含各通道名次与特征）。"""
    config = cfg or get_config()
    log = _lazy_logger(logger)
    ret = config.retrieval
    with log.enter("hybrid_search", {"question": question[:80], "top_k": top_k,
                                     "modality": modality, "vector_k": ret.vector_k,
                                     "bm25_k": ret.bm25_k, "rrf_k": ret.rrf_k,
                                     "allowed": len(allowed_ids) if allowed_ids else None}) as span:
        started = time.perf_counter()
        # ① 向量通道（本地 Ollama bge-m3，1024 维；英文提问同样可用——bge-m3 多语）
        #    T7 优化「向量查询富化」：把问句的字段词一并嵌入。实测依据（T6/T4）：
        #    题 260/33/34 的向量通道权威层命中为 0，而它们的问句都很「口语化」
        #    （「…来自军用领域的收入分别是多少」），语料侧的措辞则围绕字段名；
        #    富化后同一问句的查询向量带上字段词，无需改动 RRF/相似度语义。
        field_terms = list(getattr(analysis, "field_terms", []) or [])
        vector_query = question if not field_terms else f"{question} {' '.join(field_terms)}"
        query_vector = embed_query(vector_query, cfg=config, logger=log)
        vector_hits = search_vectors(bundle.vector_index, query_vector, int(ret.vector_k),
                                     allowed_ids=allowed_ids, logger=log)
        # ② BM25 通道：问句 + 字段词（提升字段名权重，不改 BM25 公式）
        bm25_query = question if not field_terms else f"{question} {' '.join(field_terms)}"
        bm25_hits = bundle.bm25.search(bm25_query, top_k=int(ret.bm25_k),
                                       allowed_ids=allowed_ids, logger=log)
        # ③ 通道内名次 → RRF 融合（禁止直接相加不同语义空间的相似度）
        #    T7：通道权重按实测调（向量 0.70 × 配置权重，BM25 1.20 × 配置权重）
        v_weight = float(ret.vector_weight) * VECTOR_FACTOR
        b_weight = float(ret.bm25_weight) * BM25_FACTOR
        contributions: dict[str, float] = {}
        for cid, value in _rrf_channel(vector_hits, v_weight, int(ret.rrf_k)).items():
            contributions[cid] = contributions.get(cid, 0.0) + value
        for cid, value in _rrf_channel(bm25_hits, b_weight, int(ret.rrf_k)).items():
            contributions[cid] = contributions.get(cid, 0.0) + value
        # ④ 单路高分救援：某通道 rank1 而另一通道完全没召回时，避免被 RRF 稀释
        if vector_hits and float(ret.rank_rescue_weight) > 0:
            contributions[vector_hits[0][0]] = contributions.get(vector_hits[0][0], 0.0) \
                + float(ret.rank_rescue_weight) / (int(ret.rrf_k) + 1)
        if bm25_hits and float(ret.rank_rescue_weight) > 0:
            contributions[bm25_hits[0][0]] = contributions.get(bm25_hits[0][0], 0.0) \
                + float(ret.rank_rescue_weight) / (int(ret.rrf_k) + 1)
        vector_rank = {cid: rank for rank, (cid, _s) in enumerate(vector_hits, 1)}
        bm25_rank = {cid: rank for rank, (cid, _s) in enumerate(bm25_hits, 1)}
        vector_score = {cid: score for cid, score in vector_hits}
        bm25_score = {cid: score for cid, score in bm25_hits}
        candidates: list[dict[str, Any]] = []
        for cid, rrf in contributions.items():
            row = bundle.chunks.get(cid)
            if row is None:
                continue
            if modality and str(row.get("type") or row.get("modality")) != str(modality):
                continue
            item = dict(row)
            item.update({
                "chunk_id": cid, "rrf_score": rrf,
                "vector_rank": vector_rank.get(cid), "vector_score": vector_score.get(cid),
                "bm25_rank": bm25_rank.get(cid), "bm25_score": bm25_score.get(cid),
            })
            candidates.append(item)
        # ⑤ 重排预算裁剪：先按 RRF 分排序，只对前 RERANK_BUDGET 条做特征重排；
        #    并丢弃相对分低于 RERANK_MIN_RELATIVE × 最高分的尾部候选（它们不可能进 Top-K）。
        ordered = sorted(candidates, key=lambda c: (-float(c.get("rrf_score") or 0.0), str(c.get("chunk_id"))))
        max_rrf = float(ordered[0].get("rrf_score") or 0.0) if ordered else 0.0
        cut = 0 if max_rrf <= 0 else max_rrf * max(0.0, RERANK_MIN_RELATIVE)
        budgeted = [c for c in ordered if float(c.get("rrf_score") or 0.0) >= cut][:max(1, RERANK_BUDGET)]
        pruned = len(candidates) - len(budgeted)
        # ⑥ 轻量重排（词法/字段/答案形态/模态/主体路由/极性/数值邻近/身份簇）
        ranked = rerank(analysis, budgeted, idf=bundle.bm25.bm25.idf if bundle.bm25.bm25 else None,
                        rrf_weight=1.0, feature_weight=1.0, logger=log)
        # ⑦ 图像证据权重校正（T4 定义：diagram/chart=1.0 强证据、picture=0.4 弱证据）：
        #    picture 块只有 CLIP 零样本标签，不得与正文同权 —— 这里对 final_score 做证据权重缩放。
        for item in ranked:
            weight = float(item.get("evidence_weight") or DEFAULT_EVIDENCE_WEIGHT)
            if weight != 1.0:
                item["evidence_weight_applied"] = weight
                item["final_score"] = round(float(item.get("final_score") or 0.0) * weight, 6)
        ranked.sort(key=lambda c: (-float(c.get("final_score") or 0.0), str(c.get("chunk_id"))))
        for rank, item in enumerate(ranked, 1):
            item["rank"] = rank
            item["score"] = item.get("final_score")
            item["page_basis"] = PAGE_BASIS
        result = ranked[: max(int(top_k), 1)]
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        log.log_event("retrieval.hybrid", question_digest=question[:60], candidates=len(candidates),
                      budgeted=len(budgeted), pruned=pruned, returned=len(result),
                      vector_hits=len(vector_hits), bm25_hits=len(bm25_hits),
                      weights={"vector": round(v_weight, 3), "bm25": round(b_weight, 3)},
                      top_chunk=(result[0]["chunk_id"] if result else None), elapsed_ms=elapsed_ms,
                      image_hits=sum(1 for r in result if r.get("type") == "image"))
        span.set_output({"returned": len(result), "elapsed_ms": elapsed_ms, "pruned": pruned,
                         "top_k_ids": [r["chunk_id"] for r in result]})
        return result


def _strategy_fingerprint(strategy: Any) -> str:
    """检索策略指纹（进缓存键，避免「同问句换策略」命中旧策略的缓存）。

    返回 ``""`` 表示未启用策略层（工单4 默认链路），与既有缓存键兼容。
    """
    if strategy is None:
        return ""
    if isinstance(strategy, Mapping):
        data = dict(strategy)
    elif hasattr(strategy, "to_dict"):
        data = dict(strategy.to_dict())
    else:  # pragma: no cover - 兜底：任何对象都用 repr 的摘要
        data = {"repr": repr(strategy)}
    keep = ("mode", "embedding_model", "vector_k", "fulltext_k", "top_k", "fusion",
            "vector_weight", "fulltext_weight", "rrf_k", "rerank_model", "score_threshold",
            "vector_min_similarity", "fulltext_fields", "fulltext_op", "enable_fuzzy",
            "use_image_channel")
    return json.dumps({k: data.get(k) for k in keep}, ensure_ascii=False, sort_keys=True,
                      default=str)


def _retrieve_by_strategy(bundle: IndexBundle, question: str, strategy: Any, *,
                          analysis: Any, cfg: AppConfig, allowed_ids: set[str] | None,
                          top_k: int, strategy_diag: dict[str, Any] | None,
                          logger: Any) -> list[dict[str, Any]]:
    """工单6：走显式检索策略（向量 / 全文 / 混合），返回**未包装**的候选 dict 列表。

    与 ``hybrid_search`` 的两点差异：
        * ``top_k`` 以调用方传入的为准（策略自身的 ``top_k`` 仅在未显式指定时生效），
          保证与 工单4「判定 K=5」的评测口径一致；
        * 语料路由（``allowed_ids``）在此生效，0 命中时**回退全量**并记录，绝不静默。
    """
    from .retrieval_strategy import StrategyConfig, search_by_strategy

    strategy_obj = strategy if isinstance(strategy, StrategyConfig) else StrategyConfig.from_dict(strategy)
    strategy_obj = strategy_obj.replace(top_k=int(top_k))
    result = search_by_strategy(bundle, question, strategy_obj, analysis=analysis, cfg=cfg,
                                allowed_ids=allowed_ids, logger=logger)
    hits = list(result.get("hits") or [])
    if not hits and allowed_ids is not None:
        logger.log_event("retrieval.route_fallback", level="WARNING",
                         reason="策略检索在限定语料内 0 命中", action="回退全量检索")
        result = search_by_strategy(bundle, question, strategy_obj, analysis=analysis, cfg=cfg,
                                    allowed_ids=None, logger=logger)
        hits = list(result.get("hits") or [])
        if strategy_diag is not None:
            strategy_diag["route_fallback"] = True
    if strategy_diag is not None:
        strategy_diag.clear()
        strategy_diag.update({k: v for k, v in result.items() if k != "hits"})
    return hits


def retrieve(question: str, top_k: int | None = None, *, modality: str | None = None,
             use_image_channel: bool = True, analysis: Any = None,
             cfg: AppConfig | None = None, logger: Any = None,
             strategy: Any = None, strategy_diag: dict[str, Any] | None = None,
             use_cache: bool = True) -> list[Hit]:
    """对外契约：``retrieve(question, top_k=5, modality=None, use_image_channel=True) -> list[Hit]``。

    * 问句点名发行人时启用**语料路由**（硬过滤到该文件；命中不足则自动回退全量，
      回退动作写 WARNING，绝不静默——见 golden 的跨文件串答陷阱）；
    * 返回的 ``Hit`` 是 dict 子类，含 ``chunk_id/page/file_name/type/content`` 与各通道名次。

    **工单6 新增**：``strategy`` —— 传入 ``retrieval_strategy.StrategyConfig``（或等价 dict）时，
    改走「向量 / 全文 / 混合」显式检索策略（``search_by_strategy``）；
    不传时保持工单4 既有 RRF 混合链路，**行为与既有基线逐位一致**。
    可选的 ``strategy_diag`` 是出参：策略层会把各阶段耗时、融合明细、说明写进去供上层记录。

    ``use_cache=False`` 会**同时**关掉检索结果缓存（工单4 遗留：内层缓存不受外层开关管辖，
    导致「响应时间测试」量到缓存命中时间）。指定 ``strategy`` 时也会跳过检索缓存 ——
    因为缓存里没有策略诊断，命中缓存会让诊断字段为空（详见函数内注释）。
    """
    config = cfg or get_config()
    log = _lazy_logger(logger)
    k = int(top_k if top_k is not None else config.retrieval.top_k)
    cache = get_cache(config, logger=log)
    with log.enter("retrieve", {"question": question[:80], "top_k": k, "modality": modality,
                                "use_image_channel": use_image_channel,
                                "strategy": (getattr(strategy, "mode", None) or
                                             (strategy or {}).get("mode") if strategy else None)}) as span:
        bundle = load_bundle(config, logger=log)
        allowed: set[str] | None = None
        hint = getattr(analysis, "corpus_hint", None) if analysis is not None else None
        if hint:
            wanted = _corpus_ids(bundle, str(hint))
            if len(wanted) >= 50:
                allowed = wanted
            else:
                log.log_event("retrieval.route_skipped", level="WARNING", hint=hint,
                              candidates=len(wanted), action="回退全量检索")
        # T7 检索结果缓存（第二级）：键含配置指纹 + 归一化问句 + Top-K + 模态 + 语料范围
        # 工单6：策略指纹一并进键，否则「同问句换策略」会命中旧策略的缓存。
        scope = "all" if allowed is None else f"corpus:{hint}"
        strategy_fp = _strategy_fingerprint(strategy)
        rkey = cache.retrieval_key(question, top_k=k, modality=modality, scope=scope,
                                  extra={"shape": getattr(analysis, "answer_shape", None),
                                         "strategy": strategy_fp})
        #
        # ⚠️ **两条读缓存纪律（工单6 评审修正，实测踩坑）**：
        #   1) ``strategy is not None`` 时**不读**检索缓存。原因：缓存里只存了命中行，
        #      没有策略诊断（模式/权重/融合/重排/各阶段耗时），命中缓存时 ``strategy_diag``
        #      会是空 ⇒ 界面「三模式对比」出现整行为空的假象、报告缺耗时。
        #      实测症状：混合检索命中缓存时 ``retrieve=0.5 ms``、``diagnostics.strategy=None``，
        #      而同一次会话里未命中缓存的 vector/fulltext 却有完整诊断 —— 数据自相矛盾。
        #      检索层耗时仅 8~120 ms（生成约 1000 ms），这点缓存收益不值得用「不可解释」去换。
        #   2) 调用方显式传 ``use_cache=False`` 时也不读 —— 否则「响应时间测试」量到的是
        #      缓存命中时间，而不是真实检索时间（工单4 遗留：内层缓存未受外层开关管辖）。
        read_cache = bool(use_cache) and strategy is None
        if cache.enabled and read_cache:
            hit, cached = cache.retrieval.get(rkey)
            if hit and cached:
                out = [Hit({**row, "cache_hit": True}) for row in cached]
                log.log_event("retrieval.cache_hit", key=rkey, returned=len(out), scope=scope)
                span.set_output({"returned": len(out), "cache": "hit"})
                return out
        if cache.enabled and not read_cache:
            log.log_event("retrieval.cache_skipped",
                          reason=("显式指定检索策略（需完整策略诊断，跳过检索缓存）"
                                  if strategy is not None else "调用方 use_cache=False"),
                          scope=scope)
        retrieve_started = time.perf_counter()
        if strategy is not None:
            hits = _retrieve_by_strategy(bundle, question, strategy, analysis=analysis,
                                         cfg=config, allowed_ids=allowed, top_k=k,
                                         strategy_diag=strategy_diag, logger=log)
        else:
            hits = hybrid_search(bundle, question, cfg=config, analysis=analysis, top_k=k,
                                 modality=modality, use_image_channel=use_image_channel,
                                 allowed_ids=allowed, logger=log)
            if not hits and allowed is not None:
                log.log_event("retrieval.route_fallback", level="WARNING", hint=hint,
                              reason="限定语料内 0 命中", action="回退全量检索")
                hits = hybrid_search(bundle, question, cfg=config, analysis=analysis, top_k=k,
                                     modality=modality, use_image_channel=use_image_channel,
                                     allowed_ids=None, logger=log)
        elapsed_ms = round((time.perf_counter() - retrieve_started) * 1000, 2)
        if cache.enabled and hits:
            cache.retrieval.put(rkey, [{kk: vv for kk, vv in row.items()} for row in hits],
                                saved_ms=elapsed_ms)
        out = [Hit(h) for h in hits]
        # T7「同页证据块补位」：把 Top-2 命中所在页的**表格块**追加到返回列表末尾（**不改变前 K 名**）。
        # 依据（T7 实测）：题 207 的单值来自同页表格块 pdf1-p30-t0-0（由证据页补全进入合成上下文），
        # 但它不在检索 Top-5 里 ⇒ 引用构造只能在 Top-5 内兜底，最终引用了 p32（该页其实不含答案证据，
        # 评测口径 citation_ok=False）。把同页表格块附在 Top-K 之后，引用与合成都能看到它，
        # 而**前 5 条的次序与内容完全不变**（命中判定口径 K=5 不受影响）。
        promoted = _promote_same_page_tables(bundle, out, max_extra=2, logger=log)
        out.extend(promoted)
        log.log_event("retrieval.done", returned=len(out), top_k=len(hits),
                      promoted=[h.chunk_id for h in promoted],
                      pages=[(h.file_name, h.page) for h in out[:5]],
                      modalities=[h.modality for h in out[:5]])
        span.set_output({"returned": len(out), "top": [h.chunk_id for h in out[:5]],
                         "promoted": [h.chunk_id for h in promoted]})
        return out


def _promote_same_page_tables(bundle: IndexBundle, hits: Sequence[Mapping[str, Any]], *,
                              max_extra: int = 3, logger: Any = None) -> list[Hit]:
    """同页表格块补位（追加，不替换）：**每个 Top-3 页面最多补 1 个表格块**。

    为什么每页只补 1 个：T7 实测题 207 时，Top-1 页面（p32）本身有两个表格块，
    若按「先到先得」会把名额用光，真正承载答案的 p30 表格进不来（引用仍指向 p32）。
    """
    log = _lazy_logger(logger)
    seen = {str(h.get("chunk_id")) for h in hits}
    promoted: list[Hit] = []
    for hit in list(hits)[:3]:
        if len(promoted) >= max_extra:
            break
        file_name = str(hit.get("file_name") or "")
        page = int(hit.get("page") or 0)
        for row in bundle.page_chunks(file_name, page):
            cid = str(row.get("chunk_id"))
            if cid in seen:
                continue
            modality = str(row.get("type") or row.get("modality") or "text")
            if modality != "table" and float(row.get("evidence_weight") or 0.0) < 1.0:
                continue
            seen.add(cid)
            item = Hit({**row, "sibling_of": str(hit.get("chunk_id")), "sibling_promoted": True})
            item["score"] = float(hit.get("score") or 0.0) * 0.8
            item["final_score"] = item["score"]
            promoted.append(item)
            log.log_event("retrieval.sibling_promoted", chunk_id=cid, page=page,
                          modality=modality, sibling_of=hit.get("chunk_id"))
            break                     # 每页最多补 1 个
    return promoted


def _noun_tokens(text: str) -> list[str]:
    """名词性 token（``n*`` 词性）与英文词——只有这些才可能是「实体」。

    为什么不用全部 token（T5 实测教训）：把整个问句实词拿去查词表，会把
    「武汉力源信息技术股份有限公司」这类**多词合成专名**判成词表外（它不是词表里的单个 token），
    也会把「看出 / 可以 / 今天」这类动词/时间词判成实体 → 16 题里 9 题被误拒答。
    因此：先按词性筛出名词/英文词，再逐个子词查词表；子词都在词表里即视为「语料内实体」。
    """
    try:
        import jieba.posseg as pseg

        return [str(word) for word, flag in pseg.cut(text) if flag.startswith("n") or flag == "eng"]
    except Exception:  # noqa: BLE001 —— 词性标注不可用时退化为「整词查询」，仍不编造
        return [str(text)]


#: 疑问/泛指短语（不是实体，不参与「词表外实体」判定；T5 实测：「哪些项目」会被词性标注当成名词）
_INTERROGATIVE = ("哪些", "哪种", "哪个", "哪家", "哪一", "什么", "多少", "几个", "几家", "几项",
                  "如何", "怎么", "是否", "为什么", "请问", "介绍", "告诉", "包括", "涉及")


def _is_entity_candidate(token: str) -> bool:
    """是否值得做「词表外实体」判定：长度 ≥2、含中文或英文、且不是疑问/泛指短语。"""
    key = str(token).strip()
    if len(key) < 2:
        return False
    if not re.search(r"[\u4e00-\u9fffA-Za-z]", key):
        return False
    return not any(word in key for word in _INTERROGATIVE)


def warmup_entity_tagger(*, logger: Any = None) -> dict[str, Any]:
    """预热词性标注（jieba.posseg 首次调用要加载词典，实测 ~1 s，会污染首题延迟）。

    依据（T5 实测）：抽取式后端下首题总耗时 1539 ms，其中生成仅 1.87 ms、检索 83.5 ms，
    余下 ≈1.4 s 全在首题触发的 posseg 词典加载 —— 预热后首题回到 400 ms 量级。
    """
    log = _lazy_logger(logger)
    started = time.perf_counter()
    sample = _noun_tokens("武汉兴图新科电子股份有限公司注册资本是多少")
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    out = {"ok": bool(sample), "sample_nouns": sample[:6], "elapsed_ms": elapsed_ms}
    log.log_event("retrieval.entity_tagger_warmed", **out)
    return out


def _company_like_terms(text: str) -> list[str]:
    """问句里出现的「公司样式」短语（用于判定点名主体是否在语料内）。"""
    return [m.group(0) for m in _COMPANY_LIKE.finditer(text or "")]


def _corpus_subject_known(bundle: IndexBundle, query_text: str) -> bool:
    """问句点名的主体是否在语料实体表内（按语料自动统计的发行人名 + 全称/核心名匹配）。"""
    flat = re.sub(r"\s+", "", query_text or "")
    for names in (bundle.corpus_issuers or {}).values():
        for name in names or []:
            clean = re.sub(r"\s+", "", str(name))
            if not clean:
                continue
            if clean in flat:
                return True
            core = re.sub(r"(武汉|股份有限公司|有限公司|电子)", "", clean)
            if len(core) >= 4 and core in flat:
                return True
    return False


def out_of_vocabulary(bundle: IndexBundle, keywords: Sequence[str],
                      evidence_texts: Sequence[str]) -> list[str]:
    """问句实词中「语料无法支撑其主体」的词 —— 拒答闸门的输入（T5 建、T7 扩）。

    三类判据（**只看存在性，不看覆盖率**）：

    1. **词表外实体**（T5）：实体词在 23k 词表与证据里都不存在（如「特斯拉」「苹果公司」）；
    2. **泛化名 / 占位名**（T7 / T6 的 N1）：「某某科技有限公司」分词后全是常用词，
       旧闸门因此放行，检索拿「注册资本」在 PDF1 第 64 页拼出无关的「100 万元」答案。
       占位名**不可能**有依据 ⇒ 直接判为不可答；
    3. **强指向性字段却点不出语料内主体**（T7）：问「注册资本 / 法定代表人 / 发行股数…」这类
       唯一指向某主体的字段时，问句里出现的公司名若**不在**语料实体表内 ⇒ 主体无法确定，拒答。
       （不点名任何公司时不触发——多轮追问「本次发行股数是多少？」仍可作答。）

    纯数字/百分比、长度 <2 的 token、非名词性 token、疑问短语都不参与判定。
    """
    vocab = bundle.bm25.bm25.idf if bundle.bm25.bm25 is not None else {}
    joined = "".join(str(t) for t in evidence_texts).lower().replace(" ", "")
    out: list[str] = []
    seen: set[str] = set()
    for kw in keywords:
        for token in _noun_tokens(str(kw)):
            key = token.strip()
            if key in seen or not _is_entity_candidate(key):
                continue
            seen.add(key)
            if key.lower() in joined:
                continue
            if key in vocab:
                continue
            if any(key in term for term in vocab):     # 分词差异兜底（子串命中视为存在）
                continue
            out.append(key)
    # ② 泛化名 / 占位名（对整句判定，避免分词把「某某科技有限公司」拆散后漏掉）
    query_text = "".join(str(k) for k in keywords)
    generic = _GENERIC_NAME.search(query_text)
    if generic:
        token = generic.group(0).strip()
        if token and token not in out:
            out.append(token)
    # ③ 强指向性字段 + 点名的公司不在语料内
    if any(term in query_text for term in STRONG_FIELD_TERMS):
        companies = _company_like_terms(query_text)
        if companies and not _corpus_subject_known(bundle, query_text):
            marked = f"<主体不在语料内:{companies[0][:14]}>"
            out.append(marked)
    # ④ 时间越界：问句提到的年份在语料里**完全不存在**（如「2025年营业收入」）
    years = {y for y in re.findall(r"(?:19|20)\d{2}", query_text)}
    if years:
        known = _corpus_years(bundle)
        if not (years & known):
            out.append(f"<时间越界:{sorted(years)[0]}>")
    return out


def _corpus_years(bundle: IndexBundle) -> set[str]:
    """语料中出现的年份集合（缓存）。依据：T6 负例 N2「兴图新科 2025 年营业收入」应拒答。"""
    key = f"{bundle.index_dir}|years"
    cached = _CORPUS_ID_CACHE.get(key)
    if cached is not None:
        return {str(y) for y in cached}  # type: ignore[return-value]
    years: set[str] = set()
    for row in bundle.chunks.values():
        for year in re.findall(r"(?:19|20)\d{2}", str(row.get("content") or "")):
            years.add(year)
    _CORPUS_ID_CACHE[key] = years  # type: ignore[assignment]
    return years


def expand_with_page_context(bundle: IndexBundle, hits: Sequence[Mapping[str, Any]], *,
                             max_extra: int = 6, logger: Any = None) -> list[dict[str, Any]]:
    """**证据页补全**：把 Top 命中所在页的同类块补进上下文（枚举完整性保障）。

    依据（任务书 ★A）：枚举/多点数值题的答案可能分散在同一页的多个块里
    （题 2 的第 5 项金额、题 5 的 4 部门与 6 销售处、题 33/260 的同句两段），
    只给 3 条上下文会漏项 ⇒ 命中块所在**页**的其余块一并取出（按页内 order 升序）。

    返回补全后的块列表（原命中在前，保持相对顺序；``page_context=True`` 标记新增块）。
    """
    log = _lazy_logger(logger)
    with log.enter("expand_with_page_context", {"hits": len(hits), "max_extra": max_extra}) as span:
        seen = {str(h.get("chunk_id")) for h in hits}
        out: list[dict[str, Any]] = [dict(h) for h in hits]
        extra = 0
        for hit in hits:
            if extra >= max_extra:
                break
            file_name = str(hit.get("file_name") or "")
            page = int(hit.get("page") or 0)
            for row in bundle.page_chunks(file_name, page):
                cid = str(row.get("chunk_id"))
                if cid in seen:
                    continue
                seen.add(cid)
                item = dict(row)
                item["page_context"] = True
                item["score"] = float(hit.get("score") or 0.0) * 0.5
                out.append(item)
                extra += 1
                if extra >= max_extra:
                    break
        log.log_event("retrieval.page_context", base=len(hits), added=extra, total=len(out))
        span.set_output({"base": len(hits), "added": extra, "total": len(out)})
        return out
