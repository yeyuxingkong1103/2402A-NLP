# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
src/table_parser/table_retriever.py —— 工单三表格感知检索器

职责（见 docs/02_表格检索优化方案.md §五、§六、§八）：
  1. 在 rag_tables 中向量检索表格 top_k
  2. 按 doc_id / company 过滤（多文档隔离）
  3. 与工单二文本检索融合（hybrid 模式）：
     - 文本 top_k（工单二 Retriever）+ 表格 top_k（本模块）
     - RRF 融合去重
     - bge-reranker 重排序
  4. 返回结构化表格 + 自然语言描述（table_text）
  5. 路由信号驱动：text_only / table_only / hybrid
"""
import time
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from src.embedding import get_embedder
from src.reranker import Reranker
from src.table_parser.table_store import TableStore
from src.table_parser.query_router import route_query, RouteResult


class TableRetriever:
    """工单三：表格感知检索器

    依赖：
      - TableStore（rag_tables collection）
      - Embedder（bge-m3，复用工单二）
      - Reranker（bge-reranker-v2-m3，复用工单二，可选）
      - 工单二 Retriever（文本检索，hybrid 模式时调用，可选）
    """

    def __init__(
        self,
        table_store: Optional[TableStore] = None,
        embedder=None,
        reranker: Optional[Reranker] = None,
        text_retriever=None,  # 工单二 src.retriever.Retriever
        use_rerank: bool = True,
        table_top_k: int = 10,
        text_top_k: int = 10,
        final_top_k: int = 5,
        rrf_k: int = 60,
    ):
        self.table_store = table_store or TableStore()
        self.embedder = embedder or get_embedder()
        self.text_retriever = text_retriever
        self.use_rerank = use_rerank
        self.table_top_k = table_top_k
        self.text_top_k = text_top_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k
        self._reranker = reranker
        if use_rerank and self._reranker is None:
            try:
                self._reranker = Reranker()
            except Exception as e:
                logger.warning(f"[table_retriever] Reranker 初始化失败（降级）: {e}")
                self._reranker = None
                self.use_rerank = False

    # ================= 表格向量检索 =================
    def search_tables(
        self,
        query: str,
        top_k: Optional[int] = None,
        doc_id: Optional[str] = None,
        company: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """工单三：在 rag_tables 中向量检索 + 关键词加权

        向量召回 top_k*3 → 关键词加权（query 关键词在 table_text 中命中则加分）
        → 重排取 top_k。解决短表（如"发行股数 1,670万股"）被长表挤掉的问题。

        Returns:
            [{doc_id, table_id, table_text, page, score, metadata, source:"table"}, ...]
        """
        k = top_k or self.table_top_k
        # 工单三：提取关键词（用于向量拼接 + BM25-like 精确匹配）
        query_kws = self._extract_query_keywords(query)

        # ---- 路径 1：向量检索（关键词拼接版 query） ----
        table_query = " ".join(query_kws) if query_kws else query
        qvec = self.embedder.encode([table_query], show_progress_bar=False)[0]
        raw_k = max(k * 3, 30)
        vec_hits = self.table_store.search_tables(
            qvec, top_k=raw_k, doc_id=doc_id, company=company,
        )
        for h in vec_hits:
            h["source"] = "table"
            h["content"] = h.get("table_text") or h.get("content") or ""
            h["search_path"] = "vector"

        # ---- 路径 2：BM25-like 关键词精确匹配（解决短表被漏召回） ----
        kw_hits = []
        if query_kws:
            kw_hits = self.table_store.search_tables_by_keyword(
                query_kws, doc_id=doc_id, top_k=raw_k,
            )
            for h in kw_hits:
                h["search_path"] = "keyword"

        # ---- 合并向量 + 关键词结果（保向量分数 + 关键词保底） ----
        fused = self._merge_table_results(vec_hits, kw_hits, query_kws)
        return fused[:k]

    @staticmethod
    def _merge_table_results(
        vec_hits: List[Dict], kw_hits: List[Dict], query_kws: List[str],
    ) -> List[Dict]:
        """工单三：合并向量 + 关键词表格召回

        策略（非 RRF，保留原始分数 + 关键词保底）：
          - 向量结果：保留 vec_score + kw_boost
          - 关键词结果（未在向量中）：给 0.6 + kw_hits*0.1 保底分
          - 保证关键词命中 ≥2 的短表不被长表挤出
        """
        seen: Dict[str, Dict] = {}
        # 向量结果：保留原始向量分数
        for h in vec_hits:
            key = f"{h.get('doc_id','')}|{h.get('table_id','')}"
            content = h.get("content") or ""
            kw_match = sum(1 for kw in query_kws if kw in content)
            kw_boost = min(kw_match * 0.05, 0.25)
            h["score"] = h.get("score", 0) + kw_boost
            h["kw_hits"] = kw_match
            h["search_path"] = "vector"
            seen[key] = h
        # 关键词结果：未在向量结果中的给保底分
        for h in kw_hits:
            key = f"{h.get('doc_id','')}|{h.get('table_id','')}"
            if key in seen:
                # 已在向量结果中，仅累加关键词命中
                seen[key]["kw_hits"] = max(seen[key]["kw_hits"], h.get("kw_hits", 0))
                seen[key]["search_path"] = "vector+keyword"
            else:
                # 关键词新召回：保底分 0.6 + kw_hits*0.1
                content = h.get("content") or ""
                kw_match = h.get("kw_hits", sum(1 for kw in query_kws if kw in content))
                h["score"] = 0.6 + min(kw_match * 0.1, 0.4)
                h["kw_hits"] = kw_match
                h["search_path"] = "keyword"
                seen[key] = h
        fused = sorted(seen.values(), key=lambda x: x.get("score", 0), reverse=True)
        return fused

    @staticmethod
    def _extract_query_keywords(query: str) -> List[str]:
        """工单三：提取 query 中的关键词（2+ 字，仅去功能词）

        注意：保留"发行""股数""股本""收入""持股"等业务关键词，
        只过滤"的/了/是/在/有/和/就/都/也/不/一/个/上/下/中/到/去/你/好/为"
        等纯功能词。避免把"发行""股本"误删导致 BM25 失效。
        """
        import jieba
        # 工单三：极小停用词表（只去功能词，保留业务术语）
        STOPWORDS = set(
            "的 了 和 是 在 有 我 他 她 它 这 那 就 不 也 都 一 个 上 下 中 到 说 去 "
            "你 好 为 什么 怎么 哪 谁 几 多 少 吗 呢 吧 啊 哦 嗯 会 能 可以 应该 "
            "需要 希望 想 看 听 问 答 知道 了解 多少 是".split()
        )
        tokens = [t for t in jieba.lcut(query)
                   if len(t.strip()) >= 2 and t not in STOPWORDS]
        return list(dict.fromkeys(tokens))  # 去重保序

    # ================= 文本检索（工单三 TextRetrieverV3） =================
    def _search_text(
        self,
        query: str,
        top_k: Optional[int] = None,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """工单三：调用 TextRetrieverV3（多文档文本检索 + doc_id 过滤）"""
        if self.text_retriever is None:
            return []
        try:
            # 工单三：TextRetrieverV3.retrieve 返回 list[dict]
            # 兼容工单二 HybridRetriever（返回 dict["results"]）
            result = self.text_retriever.retrieve(
                query, top_k=top_k or self.text_top_k, doc_id=doc_id,
            )
            if isinstance(result, dict) and "results" in result:
                text_hits = result["results"]
            elif isinstance(result, list):
                text_hits = result
            else:
                text_hits = []
            for h in text_hits:
                h.setdefault("source", "text")
                h.setdefault("content",
                            h.get("content") or h.get("text") or "")
            return text_hits
        except Exception as e:
            logger.warning(f"[table_retriever] 文本检索失败: {e}")
            return []

    # ================= RRF 融合 =================
    @staticmethod
    def _rrf_fuse(
        table_hits: List[Dict], text_hits: List[Dict], rrf_k: int = 60,
    ) -> List[Dict]:
        """工单三：RRF 融合表格 + 文本候选

        用 (source, doc_id, table_id_or_chunk_id) 作主键去重。
        """
        seen: Dict[str, Dict] = {}
        for rank, h in enumerate(table_hits):
            key = f"table|{h.get('doc_id','')}|{h.get('table_id','')}"
            score = 1.0 / (rrf_k + rank + 1)
            if key in seen:
                seen[key]["rrf_score"] += score
            else:
                seen[key] = dict(h, rrf_score=score, source="table")
        for rank, h in enumerate(text_hits):
            cid = h.get("chunk_id") or h.get("table_id") or ""
            key = f"text|{h.get('doc_id','')}|{cid}"
            score = 1.0 / (rrf_k + rank + 1)
            if key in seen:
                seen[key]["rrf_score"] += score
            else:
                seen[key] = dict(h, rrf_score=score, source=h.get("source", "text"))
        fused = sorted(seen.values(), key=lambda x: x["rrf_score"], reverse=True)
        return fused

    # ================= 重排序 =================
    def _rerank(
        self, query: str, candidates: List[Dict], top_k: int,
    ) -> List[Dict]:
        """工单三：bge-reranker 重排序（失败降级为 RRF 保序）"""
        if not candidates:
            return []
        if not self.use_rerank or self._reranker is None:
            return sorted(candidates,
                           key=lambda x: x.get("rrf_score", 0),
                           reverse=True)[:top_k]
        try:
            return self._reranker.rerank(
                query, candidates, top_k=top_k, content_key="content",
            )
        except Exception as e:
            logger.warning(f"[table_retriever] Rerank 失败（降级 RRF）: {e}")
            return sorted(candidates,
                           key=lambda x: x.get("rrf_score", 0),
                           reverse=True)[:top_k]

    # ================= 主入口：路由驱动检索 =================
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        doc_id: Optional[str] = None,
        company: Optional[str] = None,
        route: Optional[RouteResult] = None,
    ) -> Dict[str, Any]:
        """工单三：表格感知检索主入口

        Returns:
            {
              "query": str, "route": RouteResult,
              "table_hits": [...], "text_hits": [...],
              "fused": [...],   # 融合重排后 top_k
              "elapsed_ms": float,
            }
        """
        t0 = time.time()
        k = top_k or self.final_top_k
        if route is None:
            route = route_query(query)

        table_hits, text_hits = [], []
        if route.route in ("table_only", "hybrid"):
            table_hits = self.search_tables(
                query, top_k=self.table_top_k,
                doc_id=doc_id, company=company,
            )
        if route.route in ("text_only", "hybrid"):
            text_hits = self._search_text(
                query, top_k=self.text_top_k, doc_id=doc_id,
            )

        # 融合
        if route.route == "table_only":
            fused = table_hits
            for h in fused:
                h.setdefault("rrf_score", h.get("score", 0.0))
        elif route.route == "text_only":
            fused = text_hits
            for h in fused:
                h.setdefault("rrf_score", h.get("score", 0.0))
        else:
            fused = self._rrf_fuse(table_hits, text_hits, self.rrf_k)

        # 重排
        ranked = self._rerank(query, fused, k)
        elapsed = (time.time() - t0) * 1000
        return {
            "query": query,
            "route": route,
            "table_hits": table_hits,
            "text_hits": text_hits,
            "fused": ranked,
            "elapsed_ms": round(elapsed, 1),
        }


if __name__ == "__main__":  # pragma: no cover
    import argparse, json
    p = argparse.ArgumentParser(description="工单三：表格感知检索 CLI")
    p.add_argument("--query", required=True)
    p.add_argument("--doc-id", default=None)
    p.add_argument("--company", default=None)
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()
    tr = TableRetriever()
    res = tr.retrieve(args.query, top_k=args.top_k,
                      doc_id=args.doc_id, company=args.company)
    print(json.dumps({
        "query": res["query"],
        "route": res["route"].route,
        "confidence": res["route"].confidence,
        "table_hits": len(res["table_hits"]),
        "text_hits": len(res["text_hits"]),
        "fused_top": [
            {
                "source": h.get("source"),
                "doc_id": h.get("doc_id"),
                "table_id": h.get("table_id"),
                "page": h.get("page"),
                "score": round(h.get("rerank_score",
                                     h.get("rrf_score", 0)), 4),
                "preview": (h.get("content") or "")[:120],
            } for h in res["fused"]
        ],
        "elapsed_ms": res["elapsed_ms"],
    }, ensure_ascii=False, indent=2))
