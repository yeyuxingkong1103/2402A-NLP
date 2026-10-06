# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
模块：混合检索主类
功能：支持向量检索 / 全文检索 / 混合检索 + 3 种融合算法
"""

from typing import List, Dict, Any
from retrieval_config import RetrievalConfig
from vector_retriever import VectorRetriever
from fulltext_search import FulltextSearcher
from rerankers import build_reranker


class HybridRetriever:
    """混合检索器（可配置）"""

    def __init__(self, config: RetrievalConfig = None):
        self.cfg = config or RetrievalConfig()
        self.vector_retriever = None
        self.fulltext_searcher = None
        self.reranker = None
        self.chunks = []

    def build_index(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        print("=" * 60)
        print("构建混合检索索引...")
        print("=" * 60)

        # 1. 向量索引
        self.vector_retriever = VectorRetriever(model_name=self.cfg.embedding_model)
        self.vector_retriever.build_index(chunks)

        # 2. 全文索引
        self.fulltext_searcher = FulltextSearcher()
        self.fulltext_searcher.build_index(chunks)

        # 3. 重排器（按需加载）
        if self.cfg.use_rerank:
            self.reranker = build_reranker(self.cfg.rerank_method)

        print("✅ 混合检索索引构建完成")

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """主检索入口"""
        top_k = top_k or self.cfg.rerank_top_k
        mode = self.cfg.mode

        # 动态加载 reranker（如果打开重排但未加载）
        if self.cfg.use_rerank and self.reranker is None:
            self.reranker = build_reranker(self.cfg.rerank_method)

        if mode == "vector":
            return self._vector_only(query, top_k)
        elif mode == "fulltext":
            return self._fulltext_only(query, top_k)
        else:
            return self._hybrid(query, top_k)

    def _vector_only(self, query, top_k):
        results = self.vector_retriever.retrieve(query, top_k=self.cfg.vector_top_k)
        if self.cfg.use_rerank and self.reranker:
            results = self.reranker.rerank(query, results, top_k=top_k)
        return results[:top_k]

    def _fulltext_only(self, query, top_k):
        results = self.fulltext_searcher.search(query, top_k=self.cfg.fulltext_top_k)
        if self.cfg.use_rerank and self.reranker:
            results = self.reranker.rerank(query, results, top_k=top_k)
        return results[:top_k]

    def _hybrid(self, query, top_k):
        """混合检索"""
        # 1. 向量召回
        vec_results = self.vector_retriever.retrieve(query, top_k=self.cfg.vector_top_k)
        for r in vec_results:
            r["_vec_idx"] = True
            r["_vec_score"] = r.get("score", 0.0)

        # 2. 全文召回
        ft_results = self.fulltext_searcher.search(query, top_k=self.cfg.fulltext_top_k)
        for r in ft_results:
            r["_ft_idx"] = True

        # 3. 融合
        if self.cfg.fusion_method == "rrf":
            merged = self._rrf_fusion(vec_results, ft_results)
        elif self.cfg.fusion_method == "weighted":
            merged = self._weighted_fusion(vec_results, ft_results)
        elif self.cfg.fusion_method == "voting":
            merged = self._voting_fusion(vec_results, ft_results)
        else:
            merged = vec_results

        # 4. 重排
        if self.cfg.use_rerank and self.reranker:
            merged = self.reranker.rerank(query, merged[:20], top_k=top_k)

        return merged[:top_k]

    def _rrf_fusion(self, vec_results, ft_results):
        """RRF（倒数排名融合）"""
        k = self.cfg.rrf_k
        rrf_scores = {}
        chunk_map = {}

        for rank, r in enumerate(vec_results):
            key = r.get("chunk_id", r["content"][:50])
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank + 1)
            chunk_map[key] = r

        for rank, r in enumerate(ft_results):
            key = r.get("chunk_id", r["content"][:50])
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank + 1)
            if key not in chunk_map:
                chunk_map[key] = r

        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])
        results = []
        for k_ in sorted_keys:
            item = dict(chunk_map[k_])
            item["fusion_score"] = rrf_scores[k_]
            results.append(item)
        return results

    def _weighted_fusion(self, vec_results, ft_results):
        """加权平均融合"""
        wv = self.cfg.weight_vector
        wf = self.cfg.weight_fulltext
        merged = {}

        # 归一化
        if vec_results:
            max_v = max(r.get("_vec_score", 0) for r in vec_results) or 1
        else:
            max_v = 1
        if ft_results:
            max_f = max(r.get("fulltext_score", 0) for r in ft_results) or 1
        else:
            max_f = 1

        for r in vec_results:
            key = r.get("chunk_id", r["content"][:50])
            merged[key] = dict(r)
            merged[key]["fusion_score"] = wv * (r.get("_vec_score", 0) / max_v)

        for r in ft_results:
            key = r.get("chunk_id", r["content"][:50])
            score = wf * (r.get("fulltext_score", 0) / max_f)
            if key in merged:
                merged[key]["fusion_score"] += score
            else:
                merged[key] = dict(r)
                merged[key]["fusion_score"] = score

        return sorted(merged.values(), key=lambda x: -x["fusion_score"])

    def _voting_fusion(self, vec_results, ft_results):
        """投票融合"""
        votes = {}
        chunk_map = {}
        for r in vec_results:
            key = r.get("chunk_id", r["content"][:50])
            votes[key] = votes.get(key, 0) + 1
            chunk_map[key] = r
        for r in ft_results:
            key = r.get("chunk_id", r["content"][:50])
            votes[key] = votes.get(key, 0) + 1
            if key not in chunk_map:
                chunk_map[key] = r
        sorted_keys = sorted(votes.keys(), key=lambda x: -votes[x])
        results = []
        for k_ in sorted_keys:
            item = dict(chunk_map[k_])
            item["votes"] = votes[k_]
            results.append(item)
        return results


if __name__ == "__main__":
    from pdf_parser import PDFParser
    parser = PDFParser("./data/招股说明书1.pdf")
    pages = parser.extract_text()
    chunks = parser.chunk_text(pages, chunk_size=300, overlap=80)

    print("\n" + "=" * 60)
    print("测试 1：向量检索模式")
    print("=" * 60)
    cfg = RetrievalConfig(mode="vector", use_rerank=False)
    hr = HybridRetriever(cfg)
    hr.build_index(chunks)
    for r in hr.retrieve("注册资本是多少？", top_k=3):
        print(f"  第{r['page']}页 score={r.get('score', 0):.4f}")

    print("\n" + "=" * 60)
    print("测试 2：全文检索模式")
    print("=" * 60)
    hr.cfg.switch_mode("fulltext")
    for r in hr.retrieve("注册资本是多少？", top_k=3):
        print(f"  第{r['page']}页 score={r.get('fulltext_score', 0):.2f}")

    print("\n" + "=" * 60)
    print("测试 3：混合检索模式（RRF + Rerank）")
    print("=" * 60)
    hr.cfg.switch_mode("hybrid")
    hr.cfg.use_rerank = True
    for r in hr.retrieve("注册资本是多少？", top_k=3):
        print(f"  第{r['page']}页 fusion={r.get('fusion_score', 0):.4f} rerank={r.get('rerank_score', 0):.4f}")
