# app/core/retrieve_service.py
"""检索服务：多查询 × 多路召回 → RRF 融合 → BGE-reranker 精排。

三段式设计：

    召回  ① 稠密向量路  语义相近
          ② 稀疏权重路  BM25 风格，命中关键词与编号
          ③ 元数据路    按法条号精确过滤（MySQL 索引 + Milvus 取原文）
          —— 扩写出的每个查询都跑一遍 ①②，多角度各撒一次网

    融合  各路结果用 RRF（倒数排名融合）合并成单一序列

    精排  交叉编码器逐对打分，取 top_k

RRF 的好处是不依赖各路分数量纲：向量相似度和 BM25 分数不可比，
但「排第几」是可比的，用排名倒数求和天然规避了归一化问题。
"""
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

from sqlalchemy.orm import Session

from app.config import settings
from app.config.components import get_embeddings, get_reranker
from app.db.milvus_conn import expr_eq, get_milvus
import logging

logger = logging.getLogger(__name__)

RRF_K = 60              # RRF 平滑常数，经验值
METADATA_WEIGHT = 2.5   # 元数据路（精确匹配）在融合中的权重
# 扩写出的各查询等权参与融合。
# 曾试过给「改写后的原问题」1.5 倍权重，指望它压住扩写变体带来的噪声，
# 但 50 题 RAGAS 实测四项平均从 +0.0073 掉到 -0.0075，反而更差，故回退等权。
# 注：0.0073 是当时那一轮判分下的对照值。判分文件后来重跑过，按当前
# evals/scores_*.json 复算是 +0.0048（见 docs/10-测试报告.md §2.2），
# 两组数字来自不同轮次，不要混用；此处保留原值是为了不改写实验记录。
PRIMARY_QUERY_WEIGHT = 1.0


class RerankService:
    """BGE-reranker 交叉编码器精排。"""

    def __init__(self, model_path: str, use_fp16: bool = True):
        from FlagEmbedding import FlagReranker
        logger.info("加载 Rerank 模型: %s", model_path)
        self.model = FlagReranker(model_path, use_fp16=use_fp16)
        logger.info("Rerank 模型加载完成")

    def rerank(self, query: str, documents: List[str],
               top_k: int = 5) -> List[Dict[str, Any]]:
        """按与 query 的相关性降序返回 [{content, score, index}]。"""
        if not documents:
            return []
        scores = self.model.compute_score([[query, d] for d in documents])
        if not isinstance(scores, list):
            scores = [scores]
        ranked = sorted(zip(documents, scores, range(len(documents))),
                        key=lambda x: x[1], reverse=True)
        return [{"content": doc, "score": float(score), "index": idx}
                for doc, score, idx in ranked[:top_k]]


class RetrieveService:

    def __init__(self):
        self.milvus = get_milvus()
        self.collection = settings.MILVUS_COLLECTION
        self.limit = settings.RETRIEVE_LIMIT
        self.top_k = settings.RERANK_TOP_K

    # ---------------- 对外入口 ----------------
    def search(self, queries: Union[str, List[str]], role_id: str,
               db: Optional[Session] = None,
               top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """返回精排后的文档：[{text, title, source, doc_type, score, route}]"""
        top_k = top_k or self.top_k
        if isinstance(queries, str):
            queries = [queries]
        queries = [q.strip() for q in queries if q and q.strip()]
        if not queries:
            return []

        expr = expr_eq(role_id=role_id)
        route_lists: List[List[Dict]] = []
        weights: List[float] = []

        # ① ② 稠密 + 稀疏：每个查询各跑一遍
        for q in queries:
            hits = self._vector_route(q, expr)
            if hits:
                route_lists.append(hits)
                weights.append(PRIMARY_QUERY_WEIGHT)

        # ③ 元数据路：只在识别到法条号时才有结果
        exact_hits: List[Dict] = []
        if db is not None and getattr(settings, "ENABLE_METADATA_ROUTE", True):
            exact_hits = self._metadata_route(queries[0], role_id, db)
            if exact_hits:
                route_lists.append(exact_hits)
                # 精确匹配是强信号，给更高权重；否则单路召回的精确命中
                # 会被「多路都提到」的宽泛结果挤下去
                weights.append(METADATA_WEIGHT)

        if not route_lists:
            logger.debug("所有召回路径均无结果")
            return []

        fused = self._rrf(route_lists, weights=weights)
        logger.debug("多路召回融合后 %s 条候选（来自 %s 路）",
                     len(fused), len(route_lists))

        by_text = {c["text"]: c for c in fused if c.get("text")}
        if not by_text:
            return []

        # 法条号精确命中直接置顶，不参与重排排序。
        # 重排打的是「语义相似度」，而条号是「精确事实」：把「第一百八十八条」
        # 丢给语义去排，它会被一堆主题相近的时效条文挤下去——实测排到末位。
        # 识别到明确条号时，用户要的就是那一条，不该再猜。
        pinned = [h for h in exact_hits if h.get("text") in by_text][:top_k]
        pinned_texts = {h["text"] for h in pinned}
        rest = [t for t in by_text if t not in pinned_texts]

        ranked: List[Dict[str, Any]] = []
        if rest:
            try:
                ranked = get_reranker().rerank(
                    queries[0], rest, top_k=max(top_k - len(pinned), 1))
            except Exception as e:
                logger.warning("重排序失败，退化为融合顺序: %s", e)
                ranked = [{"content": t, "score": None} for t in rest[:top_k]]

        out = [self._format(by_text[h["text"]]) for h in pinned]
        out += [self._format(by_text.get(r["content"], {"text": r["content"]}),
                             r.get("score")) for r in ranked]
        return out[:top_k]

    @staticmethod
    def _format(meta: Dict, score=None) -> Dict[str, Any]:
        return {
            "text": meta.get("text", ""),
            "title": meta.get("title") or meta.get("source") or "",
            "source": meta.get("source", ""),
            "doc_type": meta.get("doc_type", ""),
            "law": meta.get("law", ""),
            "article": meta.get("article", ""),
            "summary": meta.get("summary", ""),
            "route": meta.get("route", "vector"),
            "score": score if score is not None else meta.get("score"),
        }

    # ---------------- 各路召回 ----------------
    def _vector_route(self, query: str, expr: str) -> List[Dict]:
        """稠密 + 稀疏双路，Milvus 内部用 RRF 融合成一路。"""
        emb = get_embeddings()
        try:
            dense = emb.embed_query(query)
            sparse = emb.compute_sparse([query])[0]
        except Exception as e:                              # pragma: no cover
            logger.warning("查询向量化失败: %s", e)
            return []

        fields = ["text", "title", "source", "doc_type", "law",
                  "article", "summary"]
        try:
            hits = self.milvus.hybrid_search(
                self.collection, dense, sparse, expr=expr,
                limit=self.limit, output_fields=fields)
        except Exception as e:
            logger.warning("混合检索失败，降级纯向量: %s", e)
            try:
                hits = self.milvus.dense_search(
                    self.collection, dense, expr=expr,
                    limit=self.limit, output_fields=fields)
            except Exception as e2:
                logger.error("向量检索也失败: %s", e2)
                return []
        for h in hits:
            h["route"] = "vector"
        return hits

    def _metadata_route(self, question: str, role_id: str,
                        db: Session) -> List[Dict]:
        """按法条号精确召回，识别不到编号时静默返回空。"""
        try:
            from app.core.metadata_service import get_metadata_service
            return get_metadata_service().search(db, question, role_id)
        except Exception as e:
            logger.warning("元数据路召回失败: %s", e)
            return []

    # ---------------- 融合 ----------------
    @staticmethod
    def _rrf(route_lists: List[List[Dict]],
             weights: Optional[List[float]] = None,
             k: int = RRF_K) -> List[Dict]:
        """加权倒数排名融合：score(d) = Σ w_i / (k + rank_i(d))。

        只用排名不用分数，天然规避了「向量相似度」与「BM25 分数」
        量纲不可比的问题；权重用来体现不同路的可信度差异。
        """
        weights = weights or [1.0] * len(route_lists)
        scores: Dict[str, float] = defaultdict(float)
        store: Dict[str, Dict] = {}
        for w, hits in zip(weights, route_lists):
            for rank, item in enumerate(hits):
                text = item.get("text")
                if not text:
                    continue
                scores[text] += w / (k + rank + 1)
                # 同一段文本可能被多路命中：保留来源标记更"精确"的那份，
                # 便于结果里看出它是被哪一路召回的
                prev = store.get(text)
                if prev is None or (prev.get("route") == "vector"
                                    and item.get("route") != "vector"):
                    store[text] = item
        ordered = sorted(scores.items(), key=lambda x: -x[1])
        return [store[t] for t, _ in ordered]

_retriever: Optional[RetrieveService] = None


def get_retriever() -> RetrieveService:
    global _retriever
    if _retriever is None:
        _retriever = RetrieveService()
    return _retriever
