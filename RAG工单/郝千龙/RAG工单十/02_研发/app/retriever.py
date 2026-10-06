# -*- coding: utf-8 -*-
# 【混合检索模块 · retriever.py】TF-IDF与BM25双路召回 -> RRF融合 -> 事实信号离线精排
# 工单编号：人工智能NLP-RAG-金融问答系统部署
# 说明：容器为纯 CPU 离线路径，不加载任何外部模型；精排全部基于词项权重与规则信号。

"""检索层主链路：

TF-IDF 字符向量召回 Top-N ∪ BM25 关键词召回 Top-N
  → RRF 倒数排名融合（数字/专名由 BM25 兜住语义漂移）
  → 事实型信号精排：IDF 加权 Query 覆盖度 + 长短语精确命中 + 答案类型吻合
  → 返回 final_top_k 个证据块。
"""
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from config import CONFIG
from index_store import IndexStore, tokenize

# 查询中的泛化停用词（对答案定位无区分度）
QUERY_STOPWORDS = {
    "根据", "按照", "招股意向书", "招股说明书", "招股", "意向书", "意向",
    "说明书", "股份有限公司", "有限公司",
    "公司", "企业", "本次", "报告期", "报告期内", "分别", "多少", "哪些",
    "哪个", "什么样", "请问", "一下", "以及", "对于", "相关", "主要",
    "是", "的", "了", "在", "和", "与", "或", "为", "有", "不", "对",
    "中", "该", "其", "用", "于", "哪", "么", "怎", "何", "谁", "呢",
    "吗", "这", "那", "个", "项", "次", "将", "已", "均", "各类", "各种",
    "时", "后", "前", "本",
}
# 公司名模式（同一文档中高频出现，对检索无区分度，查询侧剔除）
_COMPANY_RE = re.compile(r"[一-龥]{2,20}(?:股份有限公司|有限责任公司|有限公司)")


@dataclass
class Evidence:
    """单条检索证据。"""

    chunk_id: int
    score: float
    dense_rank: int
    sparse_rank: int
    page_no: int
    heading_path: str
    text: str          # 子块文本
    parent_text: str   # Small-to-Big 父段落


def reciprocal_rank_fusion(dense_hits: List[Tuple[int, float]],
                           sparse_hits: List[Tuple[int, float]]) -> dict:
    """RRF 融合两路排名（只依赖名次，天然解决分数量纲不一致）。

    :param dense_hits: TF-IDF 召回 [(块下标, 分数)]
    :param sparse_hits: BM25 召回 [(块下标, 分数)]
    :return: {块下标: rrf 分数}
    """
    fused: dict = {}
    for rank, (idx, _) in enumerate(dense_hits, start=1):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (CONFIG.rrf_k + rank)
    for rank, (idx, _) in enumerate(sparse_hits, start=1):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (CONFIG.rrf_k + rank)
    return fused


def normalize_query(query: str) -> str:
    """查询归一化：剔除公司全称，降低高频实体对 BM25 的干扰。

    :param query: 原始用户问题
    :return: 归一化后的问题
    """
    return _COMPANY_RE.sub("", query)


def query_terms(query: str, idf: dict) -> List[Tuple[str, float]]:
    """抽取带 IDF 权重的查询词项（去停用词）。

    :param query: 原始问题
    :param idf: BM25 倒排中的词项 IDF 表
    :return: [(词项, idf 权重)] 按权重降序
    """
    normalized = normalize_query(query)
    weighted = []
    for t in tokenize(normalized):
        if t in QUERY_STOPWORDS:
            continue
        w = idf.get(t, 8.0)
        # “电子信息行业的下游”中“下游”是中心词（位于“的”之后），提升权重
        if f"的{t}" in normalized:
            w *= 1.3
        weighted.append((t, w))
    dedup = {t: w for t, w in weighted}
    return sorted(dedup.items(), key=lambda x: x[1], reverse=True)


def exact_phrases(query: str) -> List[str]:
    """抽取问题中的关键长短语，用于精确命中加分。

    组成：① jieba 切出的 3 字以上词项（如“注册资本”“法定代表人”）；
    ②连续中文片段上 4~6 字滑窗子串（覆盖“军用领域收入”等组合）。

    :param query: 原始问题
    :return: 去重后的长短语列表
    """
    normalized = normalize_query(query)
    phrases = {t for t in tokenize(normalized) if len(t) >= 3
               and t not in QUERY_STOPWORDS}
    # 滑窗原文先剔除停用词，避免噪声短语反复命中
    source = normalized
    for word in sorted(QUERY_STOPWORDS, key=len, reverse=True):
        source = source.replace(word, "")
    for run in re.findall(r"[一-龥]{4,14}", source):
        for size in (4, 5, 6):
            for start in range(0, len(run) - size + 1, 2):
                phrases.add(run[start:start + size])
    return list(phrases)


def answer_type_bonus(query: str, text: str) -> float:
    """按问题期望的答案类型给予证据块类型吻合加分。

    :param query: 原始问题
    :param text: 候选块文本
    :return: 类型吻合加分值
    """
    bonus = 0.0
    if re.search(r"多少|比重|比例|金额|注册资本|收入|数量|几个", query):
        if re.search(r"\d[\d,\.]*\s*(?:%|万元|亿元|个|名)?", text):
            bonus += 0.12
    if re.search(r"谁|代表人|发明人", query) and re.search(
            r"[是为：:]\s*[一-龥]{2,4}", text):
        bonus += 0.12
    if re.search(r"哪些|包括|涉及", query) and ("、" in text or "；" in text):
        bonus += 0.08
    return bonus


def offline_score(query: str, text: str, base: float, idf: dict) -> float:
    """离线精排打分：RRF 底座 + IDF 覆盖度 + 长短语 + 类型吻合。

    :param query: 原始问题
    :param text: 候选块文本
    :param base: RRF 融合基础分
    :param idf: BM25 IDF 权重表
    :return: 精排得分
    """
    terms = query_terms(query, idf)
    total_w = sum(w for _, w in terms) or 1.0
    coverage_w = sum(w for t, w in terms if t in text)
    score = base + 1.5 * coverage_w / total_w
    # 长短语命中加分封顶，避免超长问题中“广撒网”短语垄断排序
    score += min(0.3, sum(0.15 for p in exact_phrases(query) if p in text))
    score += answer_type_bonus(query, text)
    return score


class Retriever:
    """组装双路召回、RRF 融合与离线精排的检索器（全局单例使用）。"""

    def __init__(self, store: IndexStore) -> None:
        """保存已加载的索引。

        :param store: 已构建/加载的索引
        """
        self.store = store

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 candidate_n: Optional[int] = None) -> List[Evidence]:
        """执行完整检索链路。

        :param query: 用户问题（中文/英文）
        :param top_k: 最终返回证据数
        :param candidate_n: 进入精排的融合候选数
        :return: Evidence 列表，按相关性降序
        """
        top_k = top_k or CONFIG.final_top_k
        candidate_n = candidate_n or CONFIG.candidate_n
        sparse_query = normalize_query(query)
        dense_hits = self.store.dense_search(query, CONFIG.dense_top_k)
        sparse_hits = self.store.sparse_search(sparse_query, CONFIG.bm25_top_k)
        fused = reciprocal_rank_fusion(dense_hits, sparse_hits)
        dense_rank = {idx: r for r, (idx, _) in enumerate(dense_hits, 1)}
        sparse_rank = {idx: r for r, (idx, _) in enumerate(sparse_hits, 1)}

        candidate_idx = sorted(fused, key=fused.get, reverse=True)[:candidate_n]
        idf = self.store.sparse.idf
        ranked = sorted(
            ((idx, offline_score(query, self.store.chunks[idx].text,
                                 fused.get(idx, 0.0), idf))
             for idx in candidate_idx),
            key=lambda x: x[1], reverse=True)

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
