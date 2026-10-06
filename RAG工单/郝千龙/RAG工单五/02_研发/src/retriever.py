# -*- coding: utf-8 -*-
# 【检索模块 · retriever.py】双路召回→RRF融合→离线精排→Top-K
# 工单编号：人工智能NLP-RAG-Query理解优化任务

"""检索层主链路：

TF-IDF 向量召回 Top-N ∪ BM25 关键词召回 Top-N
  → RRF 倒数排名融合（数字/专名由 BM25 兜住语义漂移）
  → 事实型信号精排：IDF 加权 Query 覆盖度 + 长短语精确命中 + 答案类型吻合
  → 返回 final_top_k 个证据块。
"""
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config import CONFIG
from vector_store import IndexStore, tokenize

QUERY_STOPWORDS = {
    "根据", "按照", "招股意向书", "招股说明书", "招股", "意向书", "意向",
    "说明书", "股份有限公司", "有限责任公司", "有限公司",
    "公司", "企业", "本次", "报告期", "报告期内", "分别", "多少", "哪些",
    "哪个", "什么样", "请问", "一下", "以及", "对于", "相关", "主要",
    "是", "的", "了", "在", "和", "与", "或", "为", "有", "不", "对",
    "中", "该", "其", "用", "于", "哪", "么", "怎", "何", "谁", "呢",
    "吗", "这", "那", "个", "项", "次", "将", "已", "均", "各类", "各种",
    "时", "后", "前", "本",
}
_COMPANY_RE = re.compile(r"[\u4e00-\u9fa5]{2,20}(?:股份有限公司|有限责任公司|有限公司)")


@dataclass
class Evidence:
    """单条检索证据。"""

    chunk_id: int
    score: float
    dense_rank: int
    sparse_rank: int
    page_no: int
    heading_path: str
    text: str
    parent_text: str


def reciprocal_rank_fusion(dense_hits: List[Tuple[int, float]],
                           sparse_hits: List[Tuple[int, float]]) -> dict:
    """RRF 融合两路排名。"""
    fused: dict = {}
    for rank, (idx, _) in enumerate(dense_hits, start=1):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (CONFIG.rrf_k + rank)
    for rank, (idx, _) in enumerate(sparse_hits, start=1):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (CONFIG.rrf_k + rank)
    return fused


def normalize_query(query: str) -> str:
    """查询归一化：剔除公司全称。"""
    return _COMPANY_RE.sub("", query)


def query_terms(query: str, idf: Dict[str, float]) -> List[Tuple[str, float]]:
    """抽取带 IDF 权重的查询词项（去停用词）。"""
    normalized = normalize_query(query)
    tokens = tokenize(normalized)
    weighted = []
    for t in tokens:
        if t in QUERY_STOPWORDS:
            continue
        w = idf.get(t, 8.0)
        if f"的{t}" in normalized:
            w *= 1.3
        weighted.append((t, w))
    dedup = {t: w for t, w in weighted}
    return sorted(dedup.items(), key=lambda x: x[1], reverse=True)


def exact_phrases(query: str) -> List[str]:
    """抽取问题中的关键长短语，用于精确命中加分。"""
    normalized = normalize_query(query)
    phrases = {t for t in tokenize(normalized) if len(t) >= 3
               and t not in QUERY_STOPWORDS}
    source = normalized
    for word in sorted(QUERY_STOPWORDS, key=len, reverse=True):
        source = source.replace(word, "")
    for run in re.findall(r"[\u4e00-\u9fa5]{4,14}", source):
        for size in (4, 5, 6):
            for start in range(0, len(run) - size + 1, 2):
                phrases.add(run[start:start + size])
    return list(phrases)


def answer_type_bonus(query: str, text: str) -> float:
    """按问题期望的答案类型给予证据块类型吻合加分。"""
    bonus = 0.0
    if re.search(r"多少|比重|比例|金额|注册资本|收入|数量|几个", query):
        if re.search(r"\d[\d,\.]*\s*(?:%|万元|亿元|个|名)?", text):
            bonus += 0.12
    if re.search(r"谁|代表人|发明人", query) and re.search(r"[是为：:]\s*[\u4e00-\u9fa5]{2,4}", text):
        bonus += 0.12
    if re.search(r"哪些|包括|涉及", query) and ("、" in text or "；" in text):
        bonus += 0.08
    return bonus


class Reranker:
    """离线精排器：RRF 底座 + IDF覆盖度 + 长短语 + 类型吻合。"""

    def __init__(self, idf: Optional[Dict[str, float]] = None) -> None:
        self.idf = idf or {}

    def _offline_score(self, query: str, text: str, base: float) -> float:
        terms = query_terms(query, self.idf)
        total_w = sum(w for _, w in terms) or 1.0
        coverage_w = sum(w for t, w in terms if t in text)
        score = base + 1.5 * coverage_w / total_w
        phrase_bonus = min(0.3,
                           sum(0.15 for p in exact_phrases(query) if p in text))
        score += phrase_bonus
        score += answer_type_bonus(query, text)
        return score

    def rerank(self, query: str, pairs: List[Tuple[int, str]],
               base_scores: dict) -> List[Tuple[int, float]]:
        if not pairs:
            return []
        return sorted(
            ((idx, self._offline_score(query, text, base_scores.get(idx, 0.0)))
             for idx, text in pairs),
            key=lambda x: x[1], reverse=True)


class Retriever:
    """组装双路召回、RRF 融合与重排的检索器。"""

    def __init__(self, store: IndexStore) -> None:
        self.store = store
        self.reranker = Reranker(idf=store.sparse.idf)

    def retrieve(self, query: str,
                 top_k: Optional[int] = None,
                 candidate_n: int = 30) -> List[Evidence]:
        top_k = top_k or CONFIG.final_top_k
        sparse_query = normalize_query(query)
        dense_hits = self.store.dense_search(query, CONFIG.dense_top_k)
        sparse_hits = self.store.sparse_search(sparse_query, CONFIG.bm25_top_k)
        fused = reciprocal_rank_fusion(dense_hits, sparse_hits)
        dense_rank = {idx: r for r, (idx, _) in enumerate(dense_hits, 1)}
        sparse_rank = {idx: r for r, (idx, _) in enumerate(sparse_hits, 1)}

        # 多文档场景：若 query 含明确公司名，给主体为该公司的块加分，
        # 避免中介机构等同名槽位（法定代表人/注册资本）压制目标公司块。
        target_company = ""
        clean_q = re.sub(r"^(那|那么|那末)\s*", "", query)
        m = _COMPANY_RE.search(clean_q)
        if m:
            target_company = m.group(0)
        if target_company:
            for idx in list(fused.keys()):
                chunk = self.store.chunks[idx]
                issuer_match = re.search(r"主体：([^\n]+)", chunk.text)
                issuer = issuer_match.group(1) if issuer_match else ""
                if target_company in issuer or issuer in target_company:
                    fused[idx] = fused.get(idx, 0.0) + 0.5

        candidate_idx = sorted(fused, key=fused.get, reverse=True)[:candidate_n]
        pairs = [(idx, self.store.chunks[idx].text) for idx in candidate_idx]
        ranked = self.reranker.rerank(query, pairs, fused)

        evidences: List[Evidence] = []
        for idx, score in ranked[:top_k]:
            chunk = self.store.chunks[idx]
            evidences.append(Evidence(
                chunk_id=chunk.chunk_id, score=score,
                dense_rank=dense_rank.get(idx, -1),
                sparse_rank=sparse_rank.get(idx, -1),
                page_no=chunk.page_no, heading_path=chunk.heading_path,
                text=chunk.text, parent_text=chunk.parent_text,
            ))
        return evidences
