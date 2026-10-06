# -*- coding: utf-8 -*-
# 【检索模块 · retriever.py】公司路由 + TF-IDF/BM25双路召回 + RRF融合 + 表格感知精排
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

"""检索层主链路（对应设计文档第 5 章）：

公司路由（力源/兴图硬过滤，消除两本招股书同名章节串扰）
  → TF-IDF Top-N ∪ BM25 Top-N → RRF 倒数排名融合
  → 事实型离线精排：IDF 加权覆盖度 + 长短语精确命中
    + 答案类型吻合 + 表格行/整表加权
  → 返回 Top-K 证据块。
"""
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from config import CONFIG
from vector_store import IndexStore, tokenize

# 查询泛化停用词（对答案定位无区分度）
QUERY_STOPWORDS = {
    "根据", "按照", "招股意向书", "招股说明书", "招股", "意向书", "意向",
    "说明书", "股份有限公司", "有限公司", "公司", "企业", "本次", "报告期",
    "报告期内", "分别", "多少", "哪些", "哪个", "什么样", "请问", "一下",
    "以及", "对于", "相关", "主要", "是", "的", "了", "在", "和", "与",
    "或", "为", "有", "对", "中", "该", "其", "用", "于", "哪", "么",
    "怎", "何", "谁", "呢", "吗", "这", "那", "个", "项", "次", "将", "已",
    "均", "各类", "各种", "时", "后", "前", "本",
    # 注意：“存在/关系”必须保留，否则长短语“不存在控制关系”被拆碎，
    # 无法区分“存在/不存在控制关系的关联方”两张表（id3 与 id4）
}
_COMPANY_SUFFIX_RE = re.compile(
    r"[\u4e00-\u9fa5A-Za-z]{2,24}(?:股份有限公司|有限责任公司|有限公司|"
    r"Co\.,?\s*Ltd\.?|Electronics)", re.IGNORECASE)

# 召回期同义词扩展：招股书表格用“国防领域/金额/占比”措辞，
# 而问题常用“军用/收入/比重”，仅在召回查询补词，精排仍按原问题
SYNONYMS = {
    "军用": "国防 军品",
    "军品": "军用 国防",
    "收入": "销售额 金额",
    "比重": "占比",
    "占比": "比重 比例",
}


def recall_query(query: str) -> str:
    """生成召回增强查询（原问题 + 命中同义词）。

    :param query: 原始问题
    :return: 用于双路召回的扩展查询串
    """
    extra = [exp for word, exp in SYNONYMS.items() if word in query]
    return query + (" " + " ".join(extra) if extra else "")


@dataclass
class Evidence:
    """单条检索证据。"""

    chunk_id: int
    score: float
    dense_rank: int
    sparse_rank: int
    page_no: int
    heading_path: str
    company: str
    chunk_type: str
    table_title: str
    text: str
    parent_text: str
    row_claim: str = ""


def detect_company(query: str, chunks: List) -> Optional[str]:
    """公司路由：根据问题中的公司名/别名判定目标发行人。

    :param query: 原始问题（中/英文）
    :param chunks: 全部检索块（用于取实际公司全称集合）
    :return: 命中的公司全称；未命中返回 None（全库检索）
    """
    q = query.lower()
    companies = []
    for c in chunks:
        if c.company and c.company not in companies:
            companies.append(c.company)
    # 显式别名优先
    for doc in CONFIG.pdf_docs:
        for alias in doc["alias"]:
            if alias.lower() in q:
                return doc["company"]
    for name in companies:
        if name in query:
            return name
    # 英文名兜底
    if "xingtu" in q or "xinke" in q:
        for doc in CONFIG.pdf_docs:
            if "兴图" in doc["short"]:
                return doc["company"]
    if "liyuan" in q:
        for doc in CONFIG.pdf_docs:
            if "力源" in doc["short"]:
                return doc["company"]
    return None


def normalize_query(query: str) -> str:
    """查询归一化：剔除公司全称，避免高频实体干扰 BM25 打分。

    :param query: 原始问题
    :return: 归一化问题
    """
    return _COMPANY_SUFFIX_RE.sub("", query)


def reciprocal_rank_fusion(dense_hits, sparse_hits) -> dict:
    """RRF 融合两路排名。

    :param dense_hits: TF-IDF 召回
    :param sparse_hits: BM25 召回
    :return: {块下标: 融合分}
    """
    fused: dict = {}
    for rank, (idx, _) in enumerate(dense_hits, start=1):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (CONFIG.rrf_k + rank)
    for rank, (idx, _) in enumerate(sparse_hits, start=1):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (CONFIG.rrf_k + rank)
    return fused


def query_terms(query: str, idf: dict) -> List[Tuple[str, float]]:
    """抽取带 IDF 权重的查询词项（去停用词）。

    :param query: 原始问题
    :param idf: BM25 IDF 表
    :return: [(词项, 权重)] 降序
    """
    normalized = normalize_query(query)
    weighted = {}
    for t in tokenize(normalized):
        if t in QUERY_STOPWORDS:
            continue
        w = idf.get(t, 8.0)
        if f"的{t}" in normalized:  # “的+中心词”提升（如“下游”）
            w *= 1.3
        weighted[t] = max(weighted.get(t, 0.0), w)
    return sorted(weighted.items(), key=lambda x: x[1], reverse=True)


def exact_phrases(query: str) -> List[str]:
    """抽取关键长短语（jieba 长词 + 中文片段 4~6 字滑窗）。

    :param query: 原始问题
    :return: 长短语列表
    """
    normalized = normalize_query(query)
    phrases = {t for t in tokenize(normalized) if len(t) >= 3
               and t not in QUERY_STOPWORDS}
    source = normalized
    for word in sorted(QUERY_STOPWORDS, key=len, reverse=True):
        source = source.replace(word, "")
    for run in re.findall(r"[\u4e00-\u9fa5]{4,16}", source):
        for size in (4, 5, 6):
            for start in range(0, len(run) - size + 1, 2):
                phrases.add(run[start:start + size])
    return list(phrases)


def answer_type_bonus(query: str, text: str) -> float:
    """答案类型吻合加分。

    :param query: 原始问题
    :param text: 候选块文本
    :return: 加分值
    """
    bonus = 0.0
    if re.search(r"多少|比重|比例|金额|注册资本|收入|数量|几|占", query):
        if re.search(r"\d[\d,\.]*\s*(?:%|万元|亿元|万股|个|名)?", text):
            bonus += 0.12
    if re.search(r"谁|代表人", query) and \
            re.search(r"[是为：:]\s*[\u4e00-\u9fa5]{2,4}", text):
        bonus += 0.12
    if re.search(r"哪些|包括|涉及|投资", query) and \
            ("、" in text or "；" in text or "丨" in text):
        bonus += 0.08
    return bonus


class Retriever:
    """组装公司路由、双路召回、RRF 融合与表格感知精排的检索器。"""

    def __init__(self, store: IndexStore) -> None:
        """保存索引。

        :param store: 已构建/加载的索引
        """
        self.store = store
        self.idf = store.sparse.idf
        # 公司 → 块下标（公司路由用）
        self.company_indices: dict = {}
        for idx, ch in enumerate(store.chunks):
            self.company_indices.setdefault(ch.company, []).append(idx)

    def _rerank_score(self, query: str, text: str, chunk_type: str,
                      base: float, heading_path: str = "") -> float:
        """离线精排打分。

        :param query: 原始问题
        :param text: 候选块文本
        :param chunk_type: 块类型（table_row/table/text）
        :param base: RRF 底座分
        :return: 精排得分
        """
        terms = query_terms(query, self.idf)
        total_w = sum(w for _, w in terms) or 1.0
        coverage_w = sum(w for t, w in terms if t in text)
        score = base + 1.5 * coverage_w / total_w
        score += min(0.3, sum(0.15 for p in exact_phrases(query) if p in text))
        score += answer_type_bonus(query, text)
        # 表格行陈述句信号密度最高，整表次之；枚举/数字题进一步倾斜
        if chunk_type == "table_row":
            boost = CONFIG.table_row_boost
            if re.search(r"哪些|多少|分别|比例|关联方|项目|投资", query):
                boost += 0.05
            score += boost
        elif chunk_type == "table":
            score += CONFIG.table_whole_boost
        # 主体事实题（注册资本/法定代表人）排噪：子公司出资表、
        # 中介机构声明、发行当事人页不是答案来源；资料卡共现强加权
        if re.search(r"注册资本|法定代表人", query):
            if re.search(
                    r"子公司|孙公司|实缴|出资额|出资比例|会计师事务所|"
                    r"律师事务所|证券股份|保荐机构|主承销商|资产评估", text):
                score -= 0.6
            if re.search(r"有关声明|当事人|中介机构", heading_path):
                score -= 0.8
            if re.search(
                    r"注册资本[\s\S]{0,24}法定代表人|"
                    r"法定代表人[\s\S]{0,24}注册资本", text):
                score += 0.3
        return score

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 candidate_n: Optional[int] = None,
                 use_routing: bool = True) -> List[Evidence]:
        """执行完整检索链路。

        :param query: 用户问题（中/英文）
        :param top_k: 返回证据数
        :param candidate_n: 进入精排的候选数
        :param use_routing: 是否启用公司路由（基线对比可关闭）
        :return: Evidence 列表（相关性降序）
        """
        top_k = top_k or CONFIG.final_top_k
        candidate_n = candidate_n or CONFIG.candidate_n

        subset_arr = None
        subset_set = None
        target_company = detect_company(query, self.store.chunks) \
            if use_routing else None
        if target_company and target_company in self.company_indices:
            subset_arr = np_array(self.company_indices[target_company])
            subset_set = set(self.company_indices[target_company])

        sparse_query = normalize_query(recall_query(query))
        dense_hits = self.store.dense_search(
            recall_query(query), CONFIG.dense_top_k, subset_arr)
        sparse_hits = self.store.sparse_search(
            sparse_query, CONFIG.bm25_top_k, subset_set)
        fused = reciprocal_rank_fusion(dense_hits, sparse_hits)
        dense_rank = {idx: r for r, (idx, _) in enumerate(dense_hits, 1)}
        sparse_rank = {idx: r for r, (idx, _) in enumerate(sparse_hits, 1)}

        candidate_idx = sorted(fused, key=fused.get,
                               reverse=True)[:candidate_n]
        ranked = sorted(
            ((idx, self._rerank_score(
                query, self.store.chunks[idx].text,
                self.store.chunks[idx].chunk_type, fused.get(idx, 0.0),
                self.store.chunks[idx].heading_path))
             for idx in candidate_idx),
            key=lambda x: x[1], reverse=True)

        evidences: List[Evidence] = []
        for idx, score in ranked[:top_k]:
            ch = self.store.chunks[idx]
            evidences.append(Evidence(
                chunk_id=ch.chunk_id, score=score,
                dense_rank=dense_rank.get(idx, -1),
                sparse_rank=sparse_rank.get(idx, -1),
                page_no=ch.page_no, heading_path=ch.heading_path,
                company=ch.company, chunk_type=ch.chunk_type,
                table_title=ch.table_title, text=ch.text,
                parent_text=ch.parent_text, row_claim=ch.row_claim))
        return evidences


def np_array(items: List[int]):
    """延迟导入 numpy 并转数组（避免模块顶部循环依赖风险）。

    :param items: 下标列表
    :return: numpy 数组
    """
    import numpy as np
    return np.array(items, dtype=np.int64)
