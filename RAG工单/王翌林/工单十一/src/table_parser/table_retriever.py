# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
（工单四 人工智能NLP-RAG-图像内容解析及检索优化：table_only 双弱门控文本兜底，见 _need_text_fallback）
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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from src.embedding import get_embedder
from src.reranker import Reranker, get_reranker
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
                # 工单四：与 TextRetrieverV3 共用进程级单例
                self._reranker = get_reranker()
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

    # ================= 工单四：跨源融合（reranker 可用/降级两路径） =================
    def _fuse_cross_source(
        self, query: str, table_hits: List[Dict], text_hits: List[Dict],
        top_k: int, ranked_tables: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """工单四（人工智能NLP-RAG-图像内容解析及检索优化）：
        reranker 可用时两源各自精排、按归一化 rerank_score 合并（免融合后二次
        精排）；reranker 不可用（use_rerank=False / 加载失败）时回退工单三
        RRF 融合 + 保序，保证降级路径排序不退化。
        ranked_tables 可传入已精排表格（table_only 兜底复用，避免重复计算）。
        """
        if self.use_rerank and self._reranker is not None:
            if ranked_tables is None:
                ranked_tables = self._rerank(
                    query, table_hits, len(table_hits) or top_k)
            return self._merge_by_rerank_score(
                ranked_tables, text_hits, top_k)
        fused = self._rrf_fuse(table_hits, text_hits, self.rrf_k)
        return self._rerank(query, fused, top_k)

    # ================= 工单四：跨源分数合并（免二次精排） =================
    @staticmethod
    def _merge_by_rerank_score(
        table_hits: List[Dict], text_hits: List[Dict], top_k: int,
    ) -> List[Dict]:
        """工单四（人工智能NLP-RAG-图像内容解析及检索优化）：
        表格/文本两源各自精排后，直接按同一 bge-reranker 的归一化分数合并。

        bge-reranker-v2-m3 compute_score(normalize=True) 对同一 query 的全部
        候选输出跨候选可比的相关性概率，因此无需对融合集整体再精排一遍
        （热路径省 0.4-0.9s，16 题评估准确率不变）。TextRetrieverV3 输出的
        text_hits 已带 rerank_score；表格候选在调用前经 self._rerank 精排。
        缺 rerank_score（reranker 不可用降级）时退回 rrf_score/score。
        """
        pool = list(table_hits) + list(text_hits)
        if not any("rerank_score" in h for h in pool):
            pool.sort(
                key=lambda x: x.get("rrf_score", x.get("score", 0.0)),
                reverse=True)
        else:
            for h in pool:
                h.setdefault("rerank_score",
                             h.get("rrf_score", h.get("score", 0.0)))
            pool.sort(key=lambda x: x["rerank_score"], reverse=True)
        seen, out = set(), []
        for h in pool:
            key = (f"{h.get('source', '')}|{h.get('doc_id', '')}|"
                   f"{h.get('table_id') or h.get('chunk_id') or ''}")
            if key in seen:
                continue
            seen.add(key)
            out.append(h)
        return out[:top_k]

    # ================= 工单四：table_only 文本兜底门控 =================
    # 工单四（人工智能NLP-RAG-图像内容解析及检索优化）：通用财务/疑问词不算核心业务词。
    # 仅当这些词之外的业务实体（客户/募集/股东/军用…）在 top3 表格中零命中时才判弱。
    _FALLBACK_GENERIC_CN = {
        "比例", "收入", "营业", "金额", "总额", "数量", "股份", "股数", "公司",
        "本次", "多少", "分别", "报告", "期内", "构成", "情况", "主要", "五大",
        "五名", "十大", "是什么", "是多少", "占比", "总计", "合计", "请问",
    }
    _FALLBACK_GENERIC_EN = {
        "what", "is", "are", "the", "of", "how", "many", "much", "total",
        "amount", "please", "tell", "ratio",
    }

    @classmethod
    def _need_text_fallback(cls, query_kws: List[str],
                            ranked_tables: List[Dict[str, Any]]) -> bool:
        """工单四：判定 table_only 结果是否双弱（语义+业务词均未真正命中）

        判据（同时满足才触发文本兜底，尽量避免误伤正常表格题、控制延迟）：
          1. rerank 后 top3 中没有任何向量通道（vector / vector+keyword）候选；
          2. query 核心业务词（剔除通用财务/疑问词）在 top3 表格文本中零命中。
        典型场景："前五大客户占营业收入的比例"——答案在正文"风险因素"段落。
        """
        if not ranked_tables:
            return True
        top = ranked_tables[:3]
        # 1) 向量语义通道已命中相关表 → 不兜底
        if any("vector" in (h.get("search_path") or "") for h in top):
            return False
        # 2) 核心业务词命中检查
        blob = " ".join(h.get("content") or "" for h in top)
        core_terms = [
            k for k in query_kws
            if len(k) >= 2
            and k not in cls._FALLBACK_GENERIC_CN
            and k.lower() not in cls._FALLBACK_GENERIC_EN
        ]
        return not any(k in blob for k in core_terms)

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
        # 工单四：perf_counter 单调时钟，规避 WSL2 墙钟跳变（人工智能NLP-RAG-图像内容解析及检索优化）
        t0 = time.perf_counter()
        k = top_k or self.final_top_k
        if route is None:
            route = route_query(query)

        table_hits, text_hits = [], []
        if route.route == "hybrid":
            # 工单四（人工智能NLP-RAG-图像内容解析及检索优化）：hybrid 两路
            # 召回互不依赖，线程并行（bge encode / Milvus gRPC 各自独立；
            # 共用 reranker 的 GPU 前向由 CUDA stream 自然序列化，线程安全），
            # 热路径省 ~0.2s。table_only 因双弱门控依赖表格精排结果，保持串行。
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_t = pool.submit(self.search_tables, query,
                                    self.table_top_k, doc_id, company)
                fut_x = pool.submit(self._search_text, query,
                                    self.text_top_k, doc_id)
                table_hits = fut_t.result()
                text_hits = fut_x.result()
        elif route.route == "table_only":
            table_hits = self.search_tables(
                query, top_k=self.table_top_k,
                doc_id=doc_id, company=company,
            )
        else:
            text_hits = self._search_text(
                query, top_k=self.text_top_k, doc_id=doc_id,
            )

        # 融合
        # 工单四（人工智能NLP-RAG-图像内容解析及检索优化）：跨源复用同一
        # bge-reranker 的归一化分数合并，消除"文本先精排、融合后再精排"的
        # 重复 cross-encoder 计算（热路径省 0.4-0.9s）。
        for h in table_hits:   # 降级（无 reranker）排序与兜底门控需要 rrf_score
            h.setdefault("rrf_score", h.get("score", 0.0))
        if route.route == "table_only":
            # 先对表格重排（全量精排，保留分数供跨源合并），双弱门控触发时
            # 补文本检索并跨源融合，解决"前五大客户占比"类数据在正文段落
            # 导致 table_only 漏召回的问题。
            ranked_tables = self._rerank(
                query, table_hits, len(table_hits) or k)
            query_kws = self._extract_query_keywords(query)
            if (self.text_retriever is not None
                    and self._need_text_fallback(query_kws, ranked_tables)):
                logger.info("[table_retriever] table_only 双弱命中，触发文本兜底")
                text_hits = self._search_text(
                    query, top_k=self.text_top_k, doc_id=doc_id)
                if text_hits:
                    ranked = self._fuse_cross_source(
                        query, table_hits, text_hits, k,
                        ranked_tables=ranked_tables)
                else:
                    ranked = ranked_tables[:k]
            else:
                ranked = ranked_tables[:k]
        elif route.route == "text_only":
            # text_hits 已在 TextRetrieverV3 内完成精排（带 rerank_score），
            # 直接按分排序取 k，不再二次精排；无 reranker 时其内部已按 score
            # 保序返回，_merge 降级按 rrf_score/score 排序，行为与工单三一致。
            for h in text_hits:
                h.setdefault("rrf_score", h.get("score", 0.0))
            ranked = self._merge_by_rerank_score([], text_hits, k)
        else:
            # hybrid：两源各自精排后按归一化分合并（reranker 不可用则 RRF）。
            ranked = self._fuse_cross_source(
                query, table_hits, text_hits, k)
        elapsed = (time.perf_counter() - t0) * 1000
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
