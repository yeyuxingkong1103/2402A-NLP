# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
src/hybrid_retriever.py —— 工单二混合检索器（BM25 + 向量多路召回 + RRF + 查询改写 + 可选 HyDE）

与工单一 src/retriever.py 的区别：
  1. 索引基于工单二"父子块"分块产物（chunker_optimized 输出，子块检索 → 父块生成）
  2. 多路召回参数：向量 top_k=10 + BM25 top_k=10，chunk_id 合并去重后 RRF 融合
  3. 查询改写：领域同义词扩展 + 意图识别（statistics/summary/table/fact）
  4. 可选 HyDE：LLM 生成假设答案后以其向量再召回（默认关闭，--hyde 开启）
  5. 向量后端双通道：Milvus（use_milvus=True）或本地矩阵余弦（默认，索引自动缓存）

输出统一格式：{chunk_id, content, score, source, page, chunk_type, heading, parent_id}
用法（项目根目录）：
  python -m src.hybrid_retriever --query "军用领域收入"
  python -m pytest tests/test_hybrid_retriever.py -v
"""
import argparse
import hashlib
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

# ---------- 工单二多路召回参数（人工智能NLP-RAG-基于PDF文档的问答系统优化） ----------
TOP_K_VECTOR = 10   # 向量召回路
TOP_K_BM25 = 10     # BM25 召回路
RRF_K = 60          # RRF 融合常数
VEC_CACHE = "data/optimized/chunk_vectors.npy"  # 本地向量索引缓存

# 工单二：招股书领域同义词扩展词典（检索词 → 扩展词）
SYNONYMS: Dict[str, List[str]] = {
    "军用": ["军品", "军事", "国防", "军方"],
    "军品": ["军用", "军事", "国防"],
    "收入": ["营业收入", "营收", "销售额"],
    "营收": ["营业收入", "收入"],
    "客户": ["主要客户", "客户集中"],
    "风险": ["风险因素", "不确定性"],
    "利润": ["净利润", "盈利", "营业利润"],
    "研发": ["研发投入", "研发费用", "技术开发"],
    "募资": ["募集资金", "募投项目", "发行"],
    "股东": ["股本", "控股股东", "发行人"],
    "revenue": ["营业收入", "收入"],
    "risk": ["风险"],
}

# 工单二：意图识别正则（复用工单一 query_understanding 思路，精简内聚）
_INTENT_RULES = [
    ("statistics", r"多少|比例|占比|金额|几|数量|百分比"),
    ("table", r"表格|排名|前五|前十大|构成"),
    ("summary", r"概述|介绍|总结|哪些|什么|如何"),
    ("fact", r"是谁|是谁的|哪一|哪里|何时"),
]


def detect_intent(query: str) -> List[str]:
    """意图识别（人工智能NLP-RAG-基于PDF文档的问答系统优化）：命中多类则全部返回"""
    intents = [name for name, pat in _INTENT_RULES if re.search(pat, query)]
    return intents or ["qa"]


def expand_query(query: str, max_extra: int = 6) -> Dict[str, Any]:
    """查询改写（人工智能NLP-RAG-基于PDF文档的问答系统优化）：
    同义词扩展 + 意图识别；返回 {expanded_terms, intents, expanded_query}"""
    terms: List[str] = []
    for word, syns in SYNONYMS.items():
        if word in query:
            terms.extend(s for s in syns if s not in query)
    terms = terms[:max_extra]
    return {"original": query, "expanded_terms": terms, "intents": detect_intent(query),
            "expanded_query": query + " " + " ".join(terms) if terms else query}


# 工单二：表格键值专项召回（O7，人工智能NLP-RAG-基于PDF文档的问答系统优化）
# 招股书中公司基本信息为"键: 值"表格排版，常规向量/BM25 易被长文稀释；
# 查询命中键词时，强制召回含该键值的紧凑表格块参与融合与重排
TABLE_KEY_PATTERNS = ["法定代表人", "注册资本", "实收资本", "成立日期", "注册地址",
                      "办公地址", "经营范围", "实际控制人", "控股股东", "董事会秘书",
                      "总经理", "董事长", "监事会主席", "英文名称", "证券简称"]


def _tokenize(text: str) -> List[str]:
    """jieba 分词（BM25 语料/查询统一处理）"""
    import jieba
    return [t.strip() for t in jieba.cut(text) if t.strip() and not re.match(r"^[\s|\-—]*$", t)]


class LocalVectorIndex:
    """本地向量索引（人工智能NLP-RAG-基于PDF文档的问答系统优化）：
    对子块批量编码并缓存 npy；查询时矩阵余弦相似度 top-k。测试可注入伪 embedder/向量"""

    def __init__(self, chunks: List[Dict[str, Any]], cache_path: Optional[str] = VEC_CACHE,
                 embedder=None, vectors: Optional[np.ndarray] = None):
        self.chunks = chunks
        self.cache_path = cache_path
        self._embedder = embedder
        self._matrix = vectors
        if self._matrix is None and cache_path and os.path.isfile(cache_path):
            self._matrix = np.load(cache_path)
            logger.info(f"✅ 向量索引缓存载入: {cache_path} {self._matrix.shape}")
        if self._matrix is not None and len(self._matrix) != len(chunks):
            logger.warning("向量缓存与分块数不一致，重建索引")
            self._matrix = None

    def _get_embedder(self):
        if self._embedder is None:
            from src.embedding import get_embedder  # 复用工单一 bge-m3 封装
            self._embedder = get_embedder()
        return self._embedder

    def _build(self):
        emb = self._get_embedder()
        texts = [c.get("text", "") for c in self.chunks]
        logger.info(f"正在编码 {len(texts)} 个子块构建本地向量索引 ...")
        self._matrix = np.asarray(emb.encode(texts, batch_size=64, show_progress_bar=False),
                                  dtype="float32")
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            np.save(self.cache_path, self._matrix)
            logger.info(f"✅ 向量索引已缓存: {self.cache_path}")

    def ensure(self):
        if self._matrix is None:
            self._build()

    def search(self, query_vec: np.ndarray, top_k: int = TOP_K_VECTOR) -> List[Dict[str, Any]]:
        """余弦相似度检索（向量已归一化时点积即余弦）（人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
        self.ensure()
        q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        sims = (self._matrix @ q.T).ravel()
        idx = np.argsort(-sims)[:top_k]
        return [{"chunk": self.chunks[int(i)], "vec_score": float(sims[int(i)])} for i in idx]


class HybridRetriever:
    """工单二混合检索器（人工智能NLP-RAG-基于PDF文档的问答系统优化）：
    查询改写 → 多路召回（向量10 + BM25 10）→ chunk_id 去重 RRF → bge-reranker 精排"""

    def __init__(self, chunks_path: str = "data/optimized/招股说明书1_chunks_optimized.json",
                 use_milvus: bool = False, use_rerank: bool = True, use_hyde: bool = False,
                 top_k_vector: int = TOP_K_VECTOR, top_k_bm25: int = TOP_K_BM25,
                 vector_store=None, embedder=None, reranker=None, chunks: Optional[List[Dict]] = None,
                 vec_cache_path: Optional[str] = VEC_CACHE):
        # vec_cache_path：本地向量缓存路径，测试必须传 None 防止污染生产缓存
        # （人工智能NLP-RAG-基于PDF文档的问答系统优化）
        # 加载工单二子块（chunker_optimized 产物的 chunks 字段 = 检索用子块）
        if chunks is None:
            chunks = json_load(chunks_path).get("chunks", [])
        self.chunks = chunks
        self._by_id = {c["chunk_id"]: c for c in chunks}
        self.top_k_vector = top_k_vector
        self.top_k_bm25 = top_k_bm25
        self.use_hyde = use_hyde
        # BM25 索引（jieba 分词）
        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi([_tokenize(c.get("text", "")) for c in chunks]) if chunks else None
        # 工单二：表格键值倒排索引（key → 含键值块的索引列表，人工智能NLP-RAG-基于PDF文档的问答系统优化）
        self._kv_index: Dict[str, List[int]] = {}
        for i, c in enumerate(chunks):
            t = c.get("text", "")
            for key in TABLE_KEY_PATTERNS:
                if key in t and (key + "：" in t or key + ":" in t):
                    self._kv_index.setdefault(key, []).append(i)
        # 向量后端：Milvus 或本地索引
        self.use_milvus = use_milvus
        self._vs = vector_store
        self._local = None if use_milvus else LocalVectorIndex(chunks, vec_cache_path, embedder=embedder)
        # 重排序器
        self._reranker = reranker if reranker is not None else (None if not use_rerank else __import__("src.reranker", fromlist=["Reranker"]).Reranker())
        self.use_rerank = use_rerank and self._reranker is not None

    # ---------- 召回路 ----------
    def _vec_recall(self, query_vec: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        if self.use_milvus and self._vs is not None:
            hits = self._vs.search(query_vec, top_k=top_k)
            return [{"chunk": self._by_id.get(h["chunk_id"], h), "vec_score": h.get("score", 0.0)}
                    for h in hits if h.get("chunk_id") in self._by_id]
        return self._local.search(query_vec, top_k)

    def _bm25_recall(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        idx = np.argsort(-np.asarray(scores))[:top_k]
        return [{"chunk": self.chunks[int(i)], "bm25_score": float(scores[int(i)])}
                for i in idx if scores[i] > 0]

    def _kv_table_recall(self, query: str, max_k: int = 3) -> List[Dict[str, Any]]:
        """工单二表格键值专项召回（人工智能NLP-RAG-基于PDF文档的问答系统优化）：
        查询含表格键词 → 召回含"键：值"的块；排序按块内键值密度降序（公司主信息表
        同时含 注册资本/法定代表人/注册地址 等多键，密度最高优先），同密度取更短块"""
        def density(i: int):
            t = self.chunks[i].get("text", "")
            n_keys = sum(1 for k in TABLE_KEY_PATTERNS if k + "：" in t or k + ":" in t)
            return (-n_keys, len(t))
        hits, seen = [], set()
        for key in TABLE_KEY_PATTERNS:
            if key not in query:
                continue
            idxs = sorted(self._kv_index.get(key, []), key=density)
            for i in idxs[:max_k]:
                cid = self.chunks[i]["chunk_id"]
                if cid not in seen:
                    seen.add(cid)
                    hits.append({"chunk": self.chunks[i], "kv_key": key})
        return hits

    # ---------- HyDE（可选，人工智能NLP-RAG-基于PDF文档的问答系统优化） ----------
    def _hyde_vector(self, query: str) -> Optional[np.ndarray]:
        try:
            from src.llm_client import simple_generate
            hypo = simple_generate(
                f"假设你正在阅读一份招股说明书，请用两三句话直接回答（不需要引用）：{query}",
                system="你是专业的投资分析助手。", temperature=0.2, max_tokens=200)
            emb = self._local._get_embedder() if self._local is not None else self._get_vec_embedder()
            vec = np.asarray(emb.encode([hypo], show_progress_bar=False)[0], dtype="float32")
            logger.info("✅ HyDE 假设答案检索向量已生成")
            return vec
        except Exception as e:
            logger.warning(f"HyDE 生成失败（回退原始查询向量）: {e}")
            return None

    def _get_vec_embedder(self):
        from src.embedding import get_embedder
        return get_embedder()

    # ---------- 主入口 ----------
    def retrieve(self, query: str, top_k: int = 5, use_hyde: Optional[bool] = None) -> Dict[str, Any]:
        """混合检索（人工智能NLP-RAG-基于PDF文档的问答系统优化）
        返回 {rewritten, results:[{chunk_id, content, score, source, page, ...}]}"""
        rewrite = expand_query(query)
        intents = rewrite["intents"]
        eff_query = rewrite["expanded_query"]
        # 1) 查询向量（HyDE 可选）
        emb = self._get_vec_embedder() if self.use_milvus else self._local._get_embedder()
        q_vec = np.asarray(emb.encode([query], show_progress_bar=False)[0], dtype="float32")
        use_hyde = self.use_hyde if use_hyde is None else use_hyde
        hyde_used = False
        if use_hyde:
            h_vec = self._hyde_vector(query)
            if h_vec is not None:
                hyde_used = True
        # 2) 多路召回：向量 top10 + BM25 top10（改写词并入 BM25 查询）+ 表格键值专项
        vec_hits = self._vec_recall(q_vec, self.top_k_vector)
        bm25_hits = self._bm25_recall(eff_query, self.top_k_bm25)
        kv_hits = self._kv_table_recall(query)  # 工单二 O7 专项通道（用原始查询，避免扩展词误触）
        # 3) chunk_id 去重 + RRF 融合
        fused: Dict[str, Dict[str, Any]] = {}
        for rank, h in enumerate(vec_hits):
            cid = h["chunk"]["chunk_id"]
            fused.setdefault(cid, {"chunk": h["chunk"], "vec_rank": None, "bm25_rank": None, "kv": False})
            fused[cid]["vec_rank"] = rank
        for rank, h in enumerate(bm25_hits):
            cid = h["chunk"]["chunk_id"]
            fused.setdefault(cid, {"chunk": h["chunk"], "vec_rank": None, "bm25_rank": None, "kv": False})
            if fused[cid]["bm25_rank"] is None:
                fused[cid]["bm25_rank"] = rank
        kv_rank = self.top_k_vector + self.top_k_bm25  # 键值块以补位 rank 参与融合（rerank 决定去留）
        for h in kv_hits:
            cid = h["chunk"]["chunk_id"]
            item = fused.setdefault(cid, {"chunk": h["chunk"], "vec_rank": None,
                                          "bm25_rank": None, "kv": True})
            item["kv"] = True
            if item["vec_rank"] is None and item["bm25_rank"] is None:
                item["kv_rank"] = kv_rank
                kv_rank += 1
        for item in fused.values():
            vr, br, kvr = item["vec_rank"], item["bm25_rank"], item.get("kv_rank")
            item["rrf"] = (1.0 / (RRF_K + vr + 1) if vr is not None else 0.0) + \
                          (1.0 / (RRF_K + br + 1) if br is not None else 0.0) + \
                          (1.0 / (RRF_K + kvr + 1) if (kvr is not None and vr is None and br is None) else 0.0)
        candidates = sorted(fused.values(), key=lambda x: -x["rrf"])
        # 4) 重排序（bge-reranker）——工单二：kv 键值块强制占位（最多 2 个），防止被挤出
        if self.use_rerank:
            cands = [dict(c["chunk"], content=c["chunk"].get("text", ""), score=c["rrf"],
                          kv=c.get("kv", False))
                     for c in candidates[: max(20, top_k)]]
            ranked = self._reranker.rerank(eff_query, cands, top_k=len(cands))
        else:
            ranked = [dict(c["chunk"], content=c["chunk"].get("text", ""), score=c["rrf"],
                           kv=c.get("kv", False)) for c in candidates[:top_k]]
        kv_items = [c for c in ranked if c.get("kv")][:2]           # 强制保留键值块
        non_kv = [c for c in ranked if not c.get("kv")]
        keep = non_kv[:max(top_k - len(kv_items), 0)] + kv_items    # 键值块占位
        ranked = sorted(keep, key=lambda x: -float(x.get("rerank_score", x.get("score", 0))))[:top_k]
        # 5) 统一输出格式（kv=True 表示来自表格键值专项召回，人工智能NLP-RAG-基于PDF文档的问答系统优化）
        results = [{
            "chunk_id": r["chunk_id"],
            "content": r.get("content", r.get("text", "")),
            "score": round(float(r.get("rerank_score", r.get("score", 0.0))), 6),
            "source": f"{self._filename}#page={r.get('page')}",
            "page": r.get("page"),
            "chunk_type": r.get("chunk_type"),
            "heading": r.get("heading"),
            "parent_id": r.get("parent_id"),
            "kv": bool(r.get("kv", False)),
        } for r in ranked]
        return {"rewritten": rewrite, "hyde_used": hyde_used, "results": results}

    @property
    def _filename(self):
        return "招股说明书1.pdf"


def json_load(path: str) -> Dict[str, Any]:
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="工单二混合检索器演示（人工智能NLP-RAG-基于PDF文档的问答系统优化）")
    ap.add_argument("--query", default="军用领域收入", help="查询语句")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--hyde", action="store_true", help="启用 HyDE 假设答案检索")
    ap.add_argument("--no-rerank", action="store_true", help="关闭重排序")
    ap.add_argument("--milvus", action="store_true", help="使用 Milvus 向量后端")
    args = ap.parse_args()

    retriever = HybridRetriever(use_milvus=args.milvus, use_rerank=not args.no_rerank,
                                use_hyde=args.hyde)
    out = retriever.retrieve(args.query, top_k=args.top_k)
    print("=" * 70)
    print(f"查询: {args.query}")
    print(f"意图: {out['rewritten']['intents']} | 同义扩展: {out['rewritten']['expanded_terms']} | HyDE: {out['hyde_used']}")
    print("=" * 70)
    for i, r in enumerate(out["results"], 1):
        print(f"\n#{i} chunk_id={r['chunk_id']}  score={r['score']:.4f}  page={r['page']}  type={r['chunk_type']}")
        print(f"   heading: {r['heading']}")
        print(f"   content: {r['content'][:120]}...")
