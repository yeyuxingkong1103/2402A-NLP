# -*- coding: utf-8 -*-
"""工单3 自实现 BM25 索引（设计/接口设计.md §3.10 冻结；**禁 `rank_bm25`**）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

为什么自实现：本机**缺失 `rank_bm25` 且断网无法安装**（实测 find_spec 未命中），
按硬性要求自实现 BM25（jieba 分词）。

实现要点：
    * 每个块的 token = ``tokenize(content)``；**表格块额外注入关键数字 token**（提升数字题召回）；
    * 打分公式：``idf(t) * tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl))``，
      ``idf = ln(1 + (N - df + 0.5)/(df + 0.5))``（BM25+ 平滑，避免 df=N 时 idf 为负）；
    * 支持 ``field_weight``：按 token 给查询词加权（如工单的关键字段词典命中的词）；
    * 落盘 ``bm25.pkl``（内部数据）+ ``bm25.meta.json``（可读元信息），加载时校验条数。
"""

from __future__ import annotations

import math
import pickle
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .chunker import Chunk
from .errors import IndexDimensionError, IndexMissingError, RagError
from .text_utils import extract_numbers, text_digest, tokenize

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"

BM25_PICKLE = "bm25.pkl"
BM25_META = "bm25.meta.json"
# 词表规模上限说明：不设硬上限，但把 top-N 高频词中「纯数字且长度 <= 2」的噪声计入过滤


def _lazy_logger(logger: Any, module: str = "bm25_index") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class PureBM25:
    """纯 Python BM25（无第三方依赖），接口与冻结设计一致。"""

    def __init__(self, corpus: list[list[str]], *, k1: float = 1.5, b: float = 0.75,
                 field_weight: Mapping[str, float] | None = None) -> None:
        if not corpus:
            raise RagError("BM25 语料为空", code="RAG-4000", stage="index")
        self.corpus = corpus
        self.k1 = float(k1)
        self.b = float(b)
        self.field_weight: dict[str, float] = dict(field_weight or {})
        self.doc_count = len(corpus)
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / self.doc_count if self.doc_count else 0.0
        # 词频表与文档频率
        self.term_freqs: list[Counter[str]] = [Counter(doc) for doc in corpus]
        df: Counter[str] = Counter()
        for tf in self.term_freqs:
            df.update(tf.keys())
        self.df = df
        self.idf: dict[str, float] = {
            term: math.log(1.0 + (self.doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        self.vocab_size = len(df)
        # 倒排表：term → [(doc_row, tf)]。**只用含该词的文档参与打分**，
        # 避免每次查询都遍历全量 2806 块（实测全量扫描单查 949 ms，会吃掉「首字 ≤3 s」预算）。
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for row, tf_map in enumerate(self.term_freqs):
            for term, freq in tf_map.items():
                self.postings.setdefault(term, []).append((row, freq))

    def get_scores(self, query_tokens: Iterable[str]) -> list[float]:
        """对全部文档打分（查询词按 ``field_weight`` 加权；仅遍历含查询词的文档）。"""
        scores = [0.0] * self.doc_count
        if not self.doc_count or self.avgdl <= 0:
            return scores
        weighted: Counter[str] = Counter()
        for token in query_tokens:
            if token:
                weighted[token] += 1
        for term, qtf in weighted.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            boost = float(self.field_weight.get(term, 1.0)) * float(qtf)
            if boost == 0.0:
                continue
            for row, tf in self.postings.get(term, ()):  # 倒排表：只碰候选文档
                denom = tf + self.k1 * (1.0 - self.b + self.b * self.doc_lengths[row] / self.avgdl)
                scores[row] += idf * (tf * (self.k1 + 1.0) / denom) * boost
        return scores

    def get_top_n(self, query_tokens: Iterable[str], n: int = 10) -> list[tuple[int, float]]:
        """返回 ``[(row_index, score)]`` 降序（分数为 0 的不返回）。"""
        scores = self.get_scores(query_tokens)
        ranked = [(row, score) for row, score in enumerate(scores) if score > 0.0]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[: max(int(n), 1)]


class BM25Index:
    """块级 BM25 索引（自实现），支持过滤、落盘与加载。"""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75, use_jieba: bool = True) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.use_jieba = bool(use_jieba)
        self.bm25: PureBM25 | None = None
        self.ids: list[str] = []
        self.doc_tokens: list[list[str]] = []
        self.meta: dict[str, dict[str, Any]] = {}      # chunk_id → {file_name,page,type,table_id}
        self.updated_at = _now_iso()

    # -- 构建 ------------------------------------------------------------
    def tokens_for(self, chunk: Chunk) -> list[str]:
        """块的 token：正文分词；表格块**额外注入关键数字**（content 的 [关键数字] 行已含数值）。"""
        tokens = tokenize(chunk.content, use_jieba=self.use_jieba)
        if chunk.type == "table":
            tokens.extend(extract_numbers(chunk.content))
        return tokens

    def build(self, chunks: Sequence[Chunk], *, logger: Any = None) -> None:
        """用块序列构建索引（含 token 统计与元数据）。"""
        log = _lazy_logger(logger)
        with log.enter("BM25Index.build", {"chunks": len(chunks), "k1": self.k1, "b": self.b,
                                           "use_jieba": self.use_jieba}) as span:
            started = time.perf_counter()
            if not chunks:
                raise RagError("没有可索引的块（BM25 构建输入为空）", code="RAG-4000", stage="index")
            self.ids = [c.chunk_id for c in chunks]
            if len(set(self.ids)) != len(self.ids):
                raise RagError("chunk_id 重复，拒绝构建 BM25 索引", code="RAG-2201", stage="index")
            self.doc_tokens = [self.tokens_for(c) for c in chunks]
            self.meta = {
                c.chunk_id: {"file_name": c.file_name, "page": c.page, "type": c.type,
                             "table_id": c.table_id, "page_start": c.page_start, "page_end": c.page_end}
                for c in chunks
            }
            self.bm25 = PureBM25(self.doc_tokens, k1=self.k1, b=self.b)
            self.updated_at = _now_iso()
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            log.log_event("bm25.build", count=self.size(), vocab_size=self.vocab_size(),
                          k1=self.k1, b=self.b, elapsed_ms=elapsed_ms,
                          avg_doc_len=round(self.bm25.avgdl, 2),
                          table_chunks=sum(1 for c in chunks if c.type == "table"))
            span.set_output({"count": self.size(), "vocab_size": self.vocab_size(),
                             "elapsed_ms": elapsed_ms})
        return None

    # -- 检索 ------------------------------------------------------------
    def search(self, query: str, top_k: int = 20, *,
               allowed_ids: set[str] | None = None, logger: Any = None) -> list[tuple[str, float]]:
        """检索：返回 ``[(chunk_id, score)]`` 降序；``allowed_ids`` 在打分前裁剪候选。"""
        log = _lazy_logger(logger)
        with log.enter("BM25Index.search", {"query_digest": text_digest(query), "top_k": top_k,
                                            "allowed": len(allowed_ids) if allowed_ids is not None else None}) as span:
            started = time.perf_counter()
            if self.bm25 is None:
                raise IndexMissingError("BM25 索引尚未构建或加载", detail={"index": "bm25"})
            query_tokens = tokenize(query, use_jieba=self.use_jieba)
            if not query_tokens:
                log.log_event("bm25.empty_query", level="WARNING", query_digest=text_digest(query))
                span.set_output({"hits": 0, "note": "查询分词后为空"})
                return []
            rows = range(self.bm25.doc_count)
            if allowed_ids is not None:
                rows = [i for i, cid in enumerate(self.ids) if cid in allowed_ids]
                if not rows:
                    log.log_event("bm25.search_empty_filter", level="WARNING",
                                  reason="allowed_ids 与索引无交集", allowed=len(allowed_ids))
                    span.set_output({"hits": 0, "filtered": True})
                    return []
            scores = self.bm25.get_scores(query_tokens)
            ranked = [(i, scores[i]) for i in rows if scores[i] > 0.0]
            ranked.sort(key=lambda item: (-item[1], item[0]))
            hits = [(self.ids[i], float(score)) for i, score in ranked[: max(int(top_k), 1)]]
            log.log_event("bm25.search", query_digest=text_digest(query, limit=60), top_k=top_k,
                          candidates=len(rows), hits=len(hits), filtered=allowed_ids is not None,
                          elapsed_ms=round((time.perf_counter() - started) * 1000, 2))
            span.set_output({"hits": len(hits), "top_score": hits[0][1] if hits else None,
                             "query_tokens": len(query_tokens)})
            return hits

    # -- 持久化 ----------------------------------------------------------
    def save(self, path: Path | str, *, dim_model: str | None = None, logger: Any = None) -> Path:
        """保存到 ``{path}/bm25.pkl`` + ``{path}/bm25.meta.json``（原子写）。

        ``dim_model``：写入 meta 的「嵌入模型#维度」标识（如 ``bge-m3:latest#1024``），
        供加载方核对「BM25 与向量索引是否同源」，缺省为 None（不编造）。
        """
        log = _lazy_logger(logger)
        target = Path(path)
        with log.enter("BM25Index.save", {"index_dir": str(target), "count": self.size()}) as span:
            started = time.perf_counter()
            if self.bm25 is None:
                raise IndexMissingError("BM25 索引尚未构建，无法保存", detail={"index": "bm25"})
            target.mkdir(parents=True, exist_ok=True)
            payload = {
                "ids": self.ids, "doc_tokens": self.doc_tokens, "meta": self.meta,
                "k1": self.k1, "b": self.b, "use_jieba": self.use_jieba, "updated_at": self.updated_at,
            }
            pickle_path = target / BM25_PICKLE
            tmp_pickle = target / (BM25_PICKLE + ".tmp")
            with tmp_pickle.open("wb") as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_pickle.replace(pickle_path)
            meta = {
                "count": self.size(), "vocab_size": self.vocab_size(), "k1": self.k1, "b": self.b,
                "updated_at": self.updated_at, "dim_model": dim_model,
                "avg_doc_len": round(self.bm25.avgdl, 2), "use_jieba": self.use_jieba,
                "table_chunks": sum(1 for m in self.meta.values() if m.get("type") == "table"),
            }
            meta_path = target / BM25_META
            tmp_meta = target / (BM25_META + ".tmp")
            import json as _json

            tmp_meta.write_text(_json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            tmp_meta.replace(meta_path)
            log.log_event("bm25.save", count=meta["count"], vocab_size=meta["vocab_size"],
                          path=str(pickle_path), bytes=pickle_path.stat().st_size,
                          elapsed_ms=round((time.perf_counter() - started) * 1000, 2))
            span.set_output({"path": str(pickle_path), **meta})
            return pickle_path

    @classmethod
    def load(cls, path: Path | str, *, logger: Any = None) -> "BM25Index":
        """从 ``{path}/bm25.pkl`` 加载（缺失抛 ``IndexMissingError``；条数自洽校验）。"""
        log = _lazy_logger(logger)
        target = Path(path)
        with log.enter("BM25Index.load", {"index_dir": str(target)}) as span:
            started = time.perf_counter()
            pickle_path = target / BM25_PICKLE
            if not pickle_path.is_file():
                raise IndexMissingError(f"BM25 索引缺失：{pickle_path}", detail={"index_dir": str(target)})
            try:
                with pickle_path.open("rb") as fh:
                    payload = pickle.load(fh)
            except (OSError, pickle.UnpicklingError, EOFError) as exc:
                raise RagError(f"BM25 索引损坏：{pickle_path}（{exc}）", code="RAG-3100",
                               stage="index", detail={"path": str(pickle_path)}) from exc
            index = cls(k1=float(payload.get("k1", 1.5)), b=float(payload.get("b", 0.75)),
                        use_jieba=bool(payload.get("use_jieba", True)))
            index.ids = list(payload.get("ids") or [])
            index.doc_tokens = list(payload.get("doc_tokens") or [])
            index.meta = dict(payload.get("meta") or {})
            index.updated_at = str(payload.get("updated_at") or _now_iso())
            if len(index.ids) != len(index.doc_tokens) or not index.ids:
                raise IndexDimensionError(
                    f"BM25 索引自洽性校验失败：ids={len(index.ids)} tokens={len(index.doc_tokens)}",
                    detail={"ids": len(index.ids), "tokens": len(index.doc_tokens)},
                )
            index.bm25 = PureBM25(index.doc_tokens, k1=index.k1, b=index.b)
            log.log_event("bm25.load", count=index.size(), vocab_size=index.vocab_size(),
                          elapsed_ms=round((time.perf_counter() - started) * 1000, 2))
            span.set_output({"count": index.size(), "vocab_size": index.vocab_size()})
            return index

    # -- 信息 ------------------------------------------------------------
    def size(self) -> int:
        """块数。"""
        return len(self.ids)

    def vocab_size(self) -> int:
        """词表规模。"""
        return int(self.bm25.vocab_size) if self.bm25 is not None else 0

    def idf_of(self, term: str) -> float:
        """单词语 idf（供测试与解释性输出）。"""
        return float(self.bm25.idf.get(term, 0.0)) if self.bm25 is not None else 0.0
