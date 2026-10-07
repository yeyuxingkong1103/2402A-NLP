# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-混合检索任务
"""
检索模块（混合检索策略版）：向量检索 / 全文检索 / 混合检索，策略可配置。

核心功能：
    1. 三种检索策略：
       - vector   向量检索（bge-m3 嵌入 + Milvus 余弦召回 + 重排）；
       - fulltext 全文检索（倒排索引 BM25 + 多字段加权 + 查询语法）；
       - hybrid   混合检索（向量+全文并行召回 → 融合 → 重排）。
    2. 全文检索查询语法：
       - 布尔查询：AND / OR / NOT（也支持 + | - 简写）；
       - 短语匹配："双引号包裹" 做精确子串匹配；
       - 模糊匹配：词后加 ~，允许编辑距离 ≤1 的拼写误差；
       - 多字段检索：title（文档名）/ content（正文）/ type（块类型）加权打分。
    3. 三种重排算法（可配置）：
       - cross_encoder  bge-reranker-large 交叉编码器重排（默认）；
       - tfidf          TF-IDF 余弦相似度重排器（轻量、零成本）；
       - llm            DeepSeek 大模型批量打分重排器（语义最强）。
    4. 三种融合算法（可配置）：
       - rrf       倒数排名融合（Reciprocal Rank Fusion）；
       - weighted  归一化加权平均（vector_weight / fulltext_weight 可调）；
       - voting    Borda 计数投票机制。
    5. LRU 缓存：检索结果缓存，重复查询毫秒级返回。
"""

import hashlib
import math
import re
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import (
    TOP_K_RECALL, TOP_K_RERANK, SCORE_THRESHOLD,
    RERANK_MODEL, RERANK_DEVICE,
    CACHE_ENABLED, CACHE_MAX_SIZE,
    SEARCH_STRATEGY, FUSION_ALGORITHM, RERANK_ALGORITHM,
    VECTOR_WEIGHT, FULLTEXT_WEIGHT, RRF_K,
    FIELD_WEIGHT_TITLE, FIELD_WEIGHT_CONTENT, FIELD_WEIGHT_TYPE,
    FULLTEXT_QUERY_SYNTAX, FUZZY_MAX_DISTANCE, LLM_RERANK_TOP_N,
)
from logger import get_logger
from db_milvus import get_vectorstore, get_all_texts

logger = get_logger(__name__)

_bm25_cache: Optional[BM25Okapi] = None
_bm25_docs: Optional[List[Document]] = None
_reranker: Optional[CrossEncoder] = None

# ==================== 运行时检索配置（可通过 /api/retrieve_config 动态修改） ====================
_runtime_config: Dict = {
    "strategy": SEARCH_STRATEGY,        # vector / fulltext / hybrid
    "fusion": FUSION_ALGORITHM,         # rrf / weighted / voting
    "reranker": RERANK_ALGORITHM,       # cross_encoder / tfidf / llm
    "vector_weight": VECTOR_WEIGHT,
    "fulltext_weight": FULLTEXT_WEIGHT,
    "rrf_k": RRF_K,
}


def get_retrieve_config() -> Dict:
    """获取当前检索策略配置。"""
    return dict(_runtime_config)


def update_retrieve_config(**kwargs) -> Dict:
    """动态更新检索策略配置（校验合法性）。"""
    valid = {
        "strategy": {"vector", "fulltext", "hybrid"},
        "fusion": {"rrf", "weighted", "voting"},
        "reranker": {"cross_encoder", "tfidf", "llm"},
    }
    for key, value in kwargs.items():
        if value is None:
            continue
        if key in valid:
            if value not in valid[key]:
                raise ValueError(f"{key} 只能是 {sorted(valid[key])} 之一，收到：{value}")
            _runtime_config[key] = value
        elif key in ("vector_weight", "fulltext_weight"):
            _runtime_config[key] = max(0.0, float(value))
        elif key == "rrf_k":
            _runtime_config[key] = max(1, int(value))
    clear_cache()  # 配置变更后旧缓存失效
    logger.info(f"检索配置已更新：{_runtime_config}")
    return dict(_runtime_config)


# ==================== LRU 缓存 ====================
class SearchCache:
    """LRU 缓存实现。"""

    def __init__(self, max_size: int = 256):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[List[Document]]:
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None

    def set(self, key: str, value: List[Document]):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_info(self) -> dict:
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / (self.hits + self.misses) * 100, 2) if (self.hits + self.misses) > 0 else 0,
        }


_search_cache = SearchCache(CACHE_MAX_SIZE)


def _get_cache_key(query: str, top_k: int) -> str:
    """缓存键：查询 + top_k + 当前策略配置。"""
    cfg = _runtime_config
    content = (f"{query}_{top_k}_{cfg['strategy']}_{cfg['fusion']}_{cfg['reranker']}"
               f"_{cfg['vector_weight']}_{cfg['fulltext_weight']}")
    return hashlib.md5(content.encode()).hexdigest()


def get_reranker() -> CrossEncoder:
    """获取 bge-reranker 交叉编码器重排模型单例。"""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL, device=RERANK_DEVICE)
        logger.info(f"重排模型已加载：{RERANK_MODEL}")
    return _reranker


# ==================== 全文检索：倒排索引 + 查询语法 ====================
def _tokenize(text: str) -> List[str]:
    """中文分词（jieba）+ 英文小写化。"""
    return [t.lower() for t in jieba.cut(text) if t.strip()]


def refresh_bm25():
    """重建全文检索倒排索引（BM25 + 多字段）。

    每个文档拆成三个字段：
        title   文档名（source），权重 FIELD_WEIGHT_TITLE；
        content 正文（text），权重 FIELD_WEIGHT_CONTENT；
        type    块类型（text/table/image），权重 FIELD_WEIGHT_TYPE。
    """
    global _bm25_cache, _bm25_docs

    logger.info("开始重建全文检索倒排索引（BM25 + 多字段）...")
    all_texts = get_all_texts()

    if not all_texts:
        logger.warning("没有可用于索引的文本")
        _bm25_cache = None
        _bm25_docs = None
        return

    documents = []
    corpus = []

    for item in all_texts:
        text = item.get("text", "")
        if not text:
            continue

        doc = Document(
            page_content=text,
            metadata={
                "pk": item.get("pk"),
                "source": item.get("source", ""),
                "page": item.get("page", 0),
                "block_type": item.get("block_type", "text"),
            },
        )
        documents.append(doc)

        # 多字段拼装：标题字段重复加权次（简单实现字段权重）
        title_tokens = _tokenize(str(doc.metadata["source"])) * int(FIELD_WEIGHT_TITLE)
        type_tokens = _tokenize(str(doc.metadata["block_type"])) * int(FIELD_WEIGHT_TYPE)
        content_tokens = _tokenize(text)
        corpus.append(title_tokens + type_tokens + content_tokens)

    _bm25_docs = documents
    _bm25_cache = BM25Okapi(corpus)

    logger.info(f"全文倒排索引重建完成：{len(_bm25_docs)} 个文档（多字段加权）")


def _parse_query_syntax(query: str) -> Dict:
    """解析全文检索查询语法。

    支持：
        布尔：武汉 AND 兴图 / 军用 OR 民用 / NOT 风险（或 + | - 简写）
        短语："法定代表人" 精确子串匹配
        模糊：兴图~ 允许编辑距离 ≤ FUZZY_MAX_DISTANCE

    Returns:
        dict: {must:[], should:[], must_not:[], phrases:[], fuzzy:[]}
    """
    result = {"must": [], "should": [], "must_not": [], "phrases": [], "fuzzy": []}

    if not FULLTEXT_QUERY_SYNTAX:
        result["should"] = _tokenize(query)
        return result

    # 1. 提取短语（双引号，中英文引号都支持）
    def _take_phrase(m):
        result["phrases"].append(m.group(1))
        return " "
    rest = re.sub(r'["“]([^"”]+)["”]', _take_phrase, query)

    # 2. 按布尔操作符切分（保留操作符）
    tokens = re.split(r"\s+", rest)
    mode = "should"  # 默认 OR
    for tok in tokens:
        if not tok:
            continue
        upper = tok.upper()
        if upper in ("AND", "+"):
            mode = "must"
            continue
        if upper in ("OR", "|"):
            mode = "should"
            continue
        if upper in ("NOT", "-"):
            mode = "must_not"
            continue
        # 模糊匹配后缀 ~
        is_fuzzy = tok.endswith("~")
        word = tok[:-1] if is_fuzzy else tok
        # 整词保留（不分词），布尔过滤用整词子串匹配，避免"力源"被切成"力"误剔除
        words = [word.lower()]
        for w in words:
            if not w:
                continue
            if is_fuzzy:
                result["fuzzy"].append(w)
            else:
                result[mode].append(w)
        if mode != "should":
            mode = "should"  # 单个词后复位为默认
    return result


def _edit_distance_leq(a: str, b: str, max_d: int) -> bool:
    """判断两词编辑距离是否 ≤ max_d（早停 DP）。"""
    if abs(len(a) - len(b)) > max_d:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            val = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            cur.append(val)
            row_min = min(row_min, val)
        if row_min > max_d:
            return False
        prev = cur
    return prev[-1] <= max_d


def _fulltext_search(query: str, top_k: int) -> List[Document]:
    """全文检索：倒排索引 BM25 + 查询语法（布尔/短语/模糊）+ 多字段加权。

    打分流程：
        1. BM25 基础分（多字段加权语料）；
        2. 布尔过滤：must 词必须命中、must_not 词命中则剔除；
        3. 短语加分：正文包含短语子串则加权；
        4. 模糊匹配：查询词与文档词编辑距离 ≤1 则按 0.8 系数计分。
    """
    global _bm25_cache, _bm25_docs

    if _bm25_cache is None or _bm25_docs is None:
        logger.warning("全文倒排索引未初始化")
        return []

    parsed = _parse_query_syntax(query)
    logger.info(f"全文检索语法解析：{parsed}")

    # 布尔过滤用整词（parsed），BM25 打分用分词（score_terms）
    base_terms = parsed["must"] + parsed["should"]
    score_terms = []
    for t in base_terms:
        score_terms.extend(_tokenize(t) if re.search(r"[一-鿿]", t) else [t])
    # 短语也参与打分（避免纯短语查询基础分为 0 被过滤）
    for ph in parsed["phrases"]:
        score_terms.extend(_tokenize(ph))

    if not score_terms and not parsed["fuzzy"] and not parsed["phrases"]:
        score_terms = _tokenize(query)
        base_terms = score_terms

    # BM25 基础分
    scores = _bm25_cache.get_scores(score_terms) if score_terms else [0.0] * len(_bm25_docs)

    # 模糊词补充打分：对语料词表做编辑距离匹配（只扫词表，不扫全文）
    if parsed["fuzzy"]:
        vocab = list(_bm25_cache.idf.keys())
        for fz in parsed["fuzzy"]:
            matched = [v for v in vocab if _edit_distance_leq(fz, v, FUZZY_MAX_DISTANCE)]
            if matched:
                fuzzy_scores = _bm25_cache.get_scores(matched)
                scores = [s + 0.8 * fs for s, fs in zip(scores, fuzzy_scores)]
                logger.info(f"模糊匹配：'{fz}~' 命中词表 {matched[:5]}")

    # 布尔过滤 + 短语加权
    candidates = []
    for idx, doc in enumerate(_bm25_docs):
        text_lower = doc.page_content.lower()
        doc_tokens = set(_tokenize(doc.page_content))

        # must：必须全部命中（子串匹配，避免分词切碎问题）
        if parsed["must"] and not all(t in text_lower for t in parsed["must"]):
            continue
        # must_not：命中即剔除
        if parsed["must_not"] and any(t in text_lower for t in parsed["must_not"]):
            continue
        # should：至少命中一个（当有 should 词时）
        if parsed["should"] and not parsed["must"] and not any(
            t in text_lower for t in parsed["should"]
        ):
            continue

        score = float(scores[idx])
        # 短语精确匹配加分（每个短语 +50%，子串匹配）
        for ph in parsed["phrases"]:
            if ph.lower() in text_lower:
                score *= 1.5
        # 标题字段命中加分
        src = str(doc.metadata.get("source", "")).lower()
        if any(t in src for t in base_terms):
            score *= 1.2

        if score > 0:
            doc.metadata["score"] = score
            doc.metadata["search_type"] = "fulltext"
            candidates.append(doc)

    candidates.sort(key=lambda d: d.metadata["score"], reverse=True)
    return candidates[:top_k]


# ==================== 向量检索 ====================
def _vector_search(query: str, top_k: int) -> List[Document]:
    """向量召回：bge-m3 嵌入 + Milvus 余弦相似度。"""
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search_with_score(query, k=top_k)

    results = []
    for doc, score in docs:
        doc.metadata["score"] = score
        doc.metadata["search_type"] = "vector"
        results.append(doc)

    return results


# ==================== 融合算法（三种） ====================
def _normalize_scores(docs: List[Document]) -> Dict[str, float]:
    """Min-Max 归一化文档分数到 [0,1]，返回 {pk: norm_score}。"""
    if not docs:
        return {}
    scores = [d.metadata.get("score", 0.0) for d in docs]
    lo, hi = min(scores), max(scores)
    span = hi - lo if hi > lo else 1.0
    return {
        d.metadata.get("pk", ""): (d.metadata.get("score", 0.0) - lo) / span
        for d in docs
    }


def _collect_docs(*result_lists: List[Document]) -> Dict[str, Document]:
    """按 pk 合并多个结果列表中的文档。"""
    pk_to_doc = {}
    for docs in result_lists:
        for doc in docs:
            pk = doc.metadata.get("pk", "")
            if pk and pk not in pk_to_doc:
                pk_to_doc[pk] = doc
    return pk_to_doc


def _rrf_fusion(vector_results: List[Document], fulltext_results: List[Document],
                k: int = None) -> List[Document]:
    """RRF 倒数排名融合：score = Σ 1/(rank + k)。"""
    k = k or _runtime_config["rrf_k"]
    scores: Dict[str, float] = {}

    for rank, doc in enumerate(vector_results, start=1):
        pk = doc.metadata.get("pk", "")
        scores[pk] = scores.get(pk, 0) + 1 / (rank + k)

    for rank, doc in enumerate(fulltext_results, start=1):
        pk = doc.metadata.get("pk", "")
        scores[pk] = scores.get(pk, 0) + 1 / (rank + k)

    pk_to_doc = _collect_docs(vector_results, fulltext_results)
    sorted_pks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for pk, fused in sorted_pks:
        if pk in pk_to_doc:
            doc = pk_to_doc[pk]
            doc.metadata["fused_score"] = fused
            results.append(doc)
    return results


def _weighted_fusion(vector_results: List[Document], fulltext_results: List[Document]) -> List[Document]:
    """加权平均融合：score = w_v * norm(vector) + w_f * norm(fulltext)，权重可调。"""
    w_v = _runtime_config["vector_weight"]
    w_f = _runtime_config["fulltext_weight"]
    total = w_v + w_f
    w_v, w_f = (w_v / total, w_f / total) if total > 0 else (0.5, 0.5)

    norm_v = _normalize_scores(vector_results)
    norm_f = _normalize_scores(fulltext_results)

    all_pks = set(norm_v) | set(norm_f)
    scores = {pk: w_v * norm_v.get(pk, 0.0) + w_f * norm_f.get(pk, 0.0) for pk in all_pks}

    pk_to_doc = _collect_docs(vector_results, fulltext_results)
    sorted_pks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for pk, fused in sorted_pks:
        if pk in pk_to_doc:
            doc = pk_to_doc[pk]
            doc.metadata["fused_score"] = fused
            results.append(doc)
    logger.info(f"加权融合：w_vector={w_v:.2f}, w_fulltext={w_f:.2f}")
    return results


def _voting_fusion(vector_results: List[Document], fulltext_results: List[Document]) -> List[Document]:
    """Borda 投票融合：每路结果按名次给票（第1名得 N 票），总票数排序。"""
    scores: Dict[str, float] = {}
    n_v, n_f = len(vector_results), len(fulltext_results)

    for rank, doc in enumerate(vector_results):
        pk = doc.metadata.get("pk", "")
        scores[pk] = scores.get(pk, 0) + (n_v - rank)

    for rank, doc in enumerate(fulltext_results):
        pk = doc.metadata.get("pk", "")
        scores[pk] = scores.get(pk, 0) + (n_f - rank)

    pk_to_doc = _collect_docs(vector_results, fulltext_results)
    sorted_pks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for pk, fused in sorted_pks:
        if pk in pk_to_doc:
            doc = pk_to_doc[pk]
            doc.metadata["fused_score"] = fused
            results.append(doc)
    return results


def _fuse(vector_results: List[Document], fulltext_results: List[Document]) -> List[Document]:
    """按配置选择融合算法。"""
    algo = _runtime_config["fusion"]
    if algo == "weighted":
        return _weighted_fusion(vector_results, fulltext_results)
    if algo == "voting":
        return _voting_fusion(vector_results, fulltext_results)
    return _rrf_fusion(vector_results, fulltext_results)


# ==================== 重排算法（三种） ====================
def _rerank_cross_encoder(query: str, documents: List[Document], top_k: int) -> List[Document]:
    """重排器一：bge-reranker-large 交叉编码器（默认，精度最高）。"""
    try:
        reranker = get_reranker()
        pairs = [(query, doc.page_content[:512]) for doc in documents]
        scores = reranker.predict(pairs)

        scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        results = []
        for doc, score in scored_docs[:top_k]:
            if score >= SCORE_THRESHOLD:
                doc.metadata["rerank_score"] = float(score)
                doc.metadata["reranker"] = "cross_encoder"
                results.append(doc)
        return results
    except Exception as e:
        logger.error(f"cross_encoder 重排失败：{e}")
        return documents[:top_k]


def _rerank_tfidf(query: str, documents: List[Document], top_k: int) -> List[Document]:
    """重排器二：TF-IDF 余弦相似度重排器（轻量、零推理成本）。

    对候选文档与查询做 TF-IDF 向量化，计算余弦相似度排序。
    """
    if not documents:
        return []

    # 构建词表与 IDF
    docs_tokens = [_tokenize(d.page_content) for d in documents]
    query_tokens = _tokenize(query)
    n_docs = len(docs_tokens)

    df: Dict[str, int] = {}
    for tokens in docs_tokens:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1
    # 查询词即使不在候选中也要入词表（df=0 → idf 平滑）
    idf = {t: math.log((n_docs + 1) / (df.get(t, 0) + 1)) + 1 for t in set(query_tokens) | set(df)}

    def tfidf_vec(tokens: List[str]) -> Dict[str, float]:
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        return {t: (c / max(1, len(tokens))) * idf.get(t, 0.0) for t, c in tf.items()}

    q_vec = tfidf_vec(query_tokens)
    q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

    results = []
    for doc, tokens in zip(documents, docs_tokens):
        d_vec = tfidf_vec(tokens)
        d_norm = math.sqrt(sum(v * v for v in d_vec.values())) or 1.0
        dot = sum(v * d_vec.get(t, 0.0) for t, v in q_vec.items())
        sim = dot / (q_norm * d_norm)
        doc.metadata["rerank_score"] = sim
        doc.metadata["reranker"] = "tfidf"
        results.append((doc, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in results[:top_k]]


def _rerank_llm(query: str, documents: List[Document], top_k: int) -> List[Document]:
    """重排器三：DeepSeek LLM 批量打分重排器（语义理解最强）。

    一次调用让 LLM 给全部候选打 0-10 相关性分，控制成本（仅前 N 条候选）。
    """
    from llm_client import _call_deepseek_api  # 延迟导入避免循环依赖

    candidates = documents[:LLM_RERANK_TOP_N]
    rest = documents[LLM_RERANK_TOP_N:]

    snippets = []
    for i, doc in enumerate(candidates):
        snippet = doc.page_content[:200].replace("\n", " ")
        snippets.append(f"[{i}] {snippet}")

    prompt = (
        "你是检索相关性评估器。给定用户查询和若干候选文档片段，"
        "请为每个片段评估与查询的相关性，输出 0-10 的分数（10 最相关）。\n"
        f"查询：{query}\n\n"
        "候选片段：\n" + "\n".join(snippets) + "\n\n"
        '只输出 JSON 数组，如 [8, 3, 9]，不要输出其他内容。'
    )

    try:
        raw = _call_deepseek_api(prompt)
        match = re.search(r"\[[\d.,\s]+\]", raw)
        scores = [float(x) for x in re.findall(r"[\d.]+", match.group())] if match else []
        if len(scores) != len(candidates):
            raise ValueError(f"LLM 返回分数数 {len(scores)} 与候选数 {len(candidates)} 不符")

        scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        results = []
        for doc, s in scored:
            doc.metadata["rerank_score"] = s / 10.0
            doc.metadata["reranker"] = "llm"
            results.append(doc)
        results.extend(rest)
        return results[:top_k]
    except Exception as e:
        logger.error(f"llm 重排失败，回退 tfidf：{e}")
        return _rerank_tfidf(query, documents, top_k)


def _rerank(query: str, documents: List[Document], top_k: int) -> List[Document]:
    """按配置选择重排算法。"""
    if not documents:
        return []
    algo = _runtime_config["reranker"]
    if algo == "tfidf":
        return _rerank_tfidf(query, documents, top_k)
    if algo == "llm":
        return _rerank_llm(query, documents, top_k)
    return _rerank_cross_encoder(query, documents, top_k)


# ==================== 统一检索入口 ====================
def search(query: str, top_k: int = TOP_K_RERANK,
           strategy: str = None, fusion: str = None, reranker: str = None) -> List[Document]:
    """统一检索入口：按策略执行向量检索 / 全文检索 / 混合检索。

    Args:
        query: 查询文本
        top_k: 返回结果数量
        strategy: 检索策略（vector/fulltext/hybrid），不传用全局配置
        fusion: 融合算法（rrf/weighted/voting），仅 hybrid 生效
        reranker: 重排算法（cross_encoder/tfidf/llm）

    Returns:
        List[Document]: 检索结果（含 rerank_score）
    """
    if not query or not query.strip():
        raise ValueError("查询不能为空")

    # 请求级参数临时覆盖全局配置
    saved = dict(_runtime_config)
    if strategy or fusion or reranker:
        if strategy:
            _runtime_config["strategy"] = strategy
        if fusion:
            _runtime_config["fusion"] = fusion
        if reranker:
            _runtime_config["reranker"] = reranker
    try:
        return _do_search(query, top_k)
    finally:
        _runtime_config.update(saved)


def _do_search(query: str, top_k: int) -> List[Document]:
    """检索主流程。"""
    cfg = _runtime_config
    strategy = cfg["strategy"]

    if CACHE_ENABLED:
        cache_key = _get_cache_key(query, top_k)
        cached_result = _search_cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"检索缓存命中：{query[:30]}...")
            return cached_result

    logger.info(f"开始检索（策略={strategy}，融合={cfg['fusion']}，重排={cfg['reranker']}）：{query[:50]}...")

    vector_results: List[Document] = []
    fulltext_results: List[Document] = []

    # 1. 按策略召回
    if strategy in ("vector", "hybrid"):
        vector_results = _vector_search(query, TOP_K_RECALL)
        logger.info(f"向量召回：{len(vector_results)} 条")
    if strategy in ("fulltext", "hybrid"):
        fulltext_results = _fulltext_search(query, TOP_K_RECALL)
        logger.info(f"全文召回：{len(fulltext_results)} 条")

    # 2. 融合
    if strategy == "hybrid" and vector_results and fulltext_results:
        fused = _fuse(vector_results, fulltext_results)
    elif vector_results:
        fused = vector_results
    else:
        fused = fulltext_results
    logger.info(f"融合后：{len(fused)} 条")

    # 3. 重排
    reranked = _rerank(query, fused, top_k)
    logger.info(f"重排后（{cfg['reranker']}）：{len(reranked)} 条")

    # 4. 缓存
    if CACHE_ENABLED and reranked:
        _search_cache.set(_get_cache_key(query, top_k), reranked)

    return reranked


def hybrid_search(query: str, top_k: int = TOP_K_RERANK) -> List[Document]:
    """兼容旧接口：走统一检索入口（默认全局配置策略）。"""
    return search(query, top_k=top_k)


def get_cache_info() -> dict:
    """获取缓存统计信息。"""
    return _search_cache.get_info()


def clear_cache():
    """清空缓存。"""
    _search_cache.clear()
    logger.info("检索缓存已清空")
