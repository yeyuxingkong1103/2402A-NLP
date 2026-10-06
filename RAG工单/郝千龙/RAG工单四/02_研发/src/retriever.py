# -*- coding: utf-8 -*-
# 【图文检索模块 · retriever.py】双路召回→RRF融合→重排（图像证据类型加权）→Top-K
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化

"""检索层主链路：

BGE向量召回Top20 ∪ BM25关键词召回Top20
  → RRF倒数排名融合
  → 离线事实信号精排（IDF覆盖度/长短语/答案类型），图像类问题对图像块加权
  → 本地有bge-reranker-large时Cross-Encoder精排，否则离线精排
  → 返回Top-K证据（文字/表格/图像，图像证据带图片路径）。
"""
import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from config import CONFIG
from multimodal_index import IndexStore, tokenize

logger = logging.getLogger(__name__)

# 查询中的泛化停用词（对答案定位无区分度）
QUERY_STOPWORDS = {
    "根据", "按照", "招股意向书", "招股说明书", "招股", "意向书", "意向",
    "说明书", "股份有限公司", "有限公司",
    "公司", "企业", "本次", "报告期", "报告期内", "分别", "多少", "哪些",
    "哪个", "请问", "一下", "以及", "对于", "相关",
    "是", "的", "了", "在", "和", "与", "或", "为", "有", "不", "对",
    "中", "该", "其", "用", "于", "哪", "么", "怎", "何", "谁", "呢",
    "吗", "这", "那", "个", "项", "次", "将", "已", "均", "各类", "各种",
    "时", "后", "前", "本", "可以", "看出", "中可以",
}
_COMPANY_RE = re.compile(r"[\u4e00-\u9fa5]{2,20}(?:股份有限公司|有限责任公司|有限公司)")
_IMAGE_Q_RE = re.compile(CONFIG.image_query_pattern)


@dataclass
class Evidence:
    """单条检索证据（文字/表格/图像）。"""

    chunk_id: int
    score: float
    dense_rank: int
    sparse_rank: int
    page_no: int
    doc: str
    heading_path: str
    text: str
    parent_text: str
    chunk_type: str = "text"          # text / table / image
    image_path: str = ""              # 图像证据原图路径
    thumb_path: str = ""
    image_caption: str = ""
    degraded: bool = False


def reciprocal_rank_fusion(dense_hits: List[Tuple[int, float]],
                           sparse_hits: List[Tuple[int, float]]) -> dict:
    """RRF融合两路排名（只依赖名次，天然解决分数量纲不一致）。

    :param dense_hits: 稠密召回[(块下标,分数)]
    :param sparse_hits: 稀疏召回[(块下标,分数)]
    :return: {块下标: 融合分}
    """
    fused: dict = {}
    for rank, (idx, _) in enumerate(dense_hits, start=1):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (CONFIG.rrf_k + rank)
    for rank, (idx, _) in enumerate(sparse_hits, start=1):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (CONFIG.rrf_k + rank)
    return fused


def normalize_query(query: str) -> str:
    """查询归一化：剔除公司全称，降低高频实体对BM25的干扰。

    :param query: 原始问题
    :return: 归一化问题
    """
    return _COMPANY_RE.sub("", query)


def query_terms(query: str, idf: dict) -> List[Tuple[str, float]]:
    """抽取带IDF权重的查询词项（去停用词，“的”后中心词加权）。

    :param query: 原始问题
    :param idf: BM25的IDF表
    :return: [(词项,权重)] 降序
    """
    normalized = normalize_query(query)
    weighted = []
    for t in tokenize(normalized):
        if t in QUERY_STOPWORDS:
            continue
        w = idf.get(t, 8.0)
        if f"的{t}" in normalized:
            w *= 1.3
        weighted.append((t, w))
    dedup = {t: w for t, w in weighted}
    return sorted(dedup.items(), key=lambda x: x[1], reverse=True)


def exact_phrases(query: str) -> List[str]:
    """抽取关键长短语（3字以上词项+中文片段滑窗子串）用于精确命中加分。

    :param query: 原始问题
    :return: 去重长短语列表
    """
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
    """按问题期望的答案类型给予吻合加分。

    :param query: 原始问题
    :param text: 候选块文本
    :return: 加分值
    """
    bonus = 0.0
    if re.search(r"多少|比重|比例|金额|注册资本|收入|数量|几个", query):
        if re.search(r"\d[\d,\.]*\s*(?:%|万元|亿元|个|名)?", text):
            bonus += 0.12
    if re.search(r"谁|代表人|发明人", query) and \
            re.search(r"[是为：:]\s*[\u4e00-\u9fa5]{2,4}", text):
        bonus += 0.12
    if re.search(r"哪些|包括|涉及", query) and ("、" in text or "；" in text):
        bonus += 0.08
    return bonus


def model_available_locally(model_name: str) -> bool:
    """检测HF模型本地缓存是否存在。

    :param model_name: HF模型ID
    :return: 本地可用返回True
    """
    import glob
    if os.path.exists(model_name):
        return True
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    repo = "models--" + model_name.replace("/", "--")
    return bool(glob.glob(os.path.join(hub, repo, "snapshots", "*")))


class Reranker:
    """Cross-Encoder精排器；无本地模型时使用事实信号离线精排。"""

    def __init__(self, idf: Optional[dict] = None) -> None:
        """尝试加载本地bge-reranker，不可用则离线精排。

        :param idf: BM25 IDF表（离线精排权重）
        """
        self.idf = idf or {}
        self.model = None
        if CONFIG.use_reranker and model_available_locally(CONFIG.reranker_model):
            try:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(CONFIG.reranker_model)
                logger.info("重排模型已加载（本地缓存）：%s", CONFIG.reranker_model)
            except Exception as exc:
                logger.warning("Cross-Encoder加载失败，降级离线精排：%s", exc)
        elif CONFIG.use_reranker:
            logger.info("未检测到%s本地缓存，启用离线精排（不联网下载）",
                        CONFIG.reranker_model)

    def _offline_score(self, query: str, text: str, base: float,
                       chunk_type: str) -> float:
        """离线精排：RRF底座+IDF覆盖度+长短语+类型吻合+图像块加权。

        :param query: 原始问题
        :param text: 候选块文本
        :param base: RRF基础分
        :param chunk_type: 块类型
        :return: 精排得分
        """
        terms = query_terms(query, self.idf)
        total_w = sum(w for _, w in terms) or 1.0
        coverage_w = sum(w for t, w in terms if t in text)
        score = base + 1.5 * coverage_w / total_w
        score += min(0.3, sum(0.15 for p in exact_phrases(query) if p in text))
        score += answer_type_bonus(query, text)
        if chunk_type == "image" and _IMAGE_Q_RE.search(query):
            score += CONFIG.image_chunk_boost
        return score

    def rerank(self, query: str, pairs: List[Tuple[int, str, str]],
               base_scores: dict) -> List[Tuple[int, float]]:
        """对候选块精排。

        :param query: 用户查询
        :param pairs: [(块下标, 块文本, 块类型)]
        :param base_scores: RRF融合基础分
        :return: [(块下标,最终分)] 降序
        """
        if not pairs:
            return []
        if self.model is not None:
            try:
                tr = [t[: CONFIG.rerank_truncate] for _, t, _ in pairs]
                raw = self.model.predict([(query, t) for t in tr])
                scored = [(pairs[i][0], float(raw[i])) for i in range(len(pairs))]
                # 图像问题在Cross-Encoder分数上叠加小幅类型加权
                if _IMAGE_Q_RE.search(query):
                    scored = [(idx, sc + (CONFIG.image_chunk_boost
                                          if pairs[i][2] == "image" else 0.0))
                              for i, (idx, sc) in enumerate(scored)]
                return sorted(scored, key=lambda x: x[1], reverse=True)
            except Exception as exc:
                logger.warning("Cross-Encoder推理失败，本次降级：%s", exc)
        return sorted(
            ((idx, self._offline_score(query, text,
                                       base_scores.get(idx, 0.0), ctype))
             for idx, text, ctype in pairs),
            key=lambda x: x[1], reverse=True)


class Retriever:
    """组装双路召回、RRF融合与重排的图文检索器。"""

    def __init__(self, store: IndexStore) -> None:
        """保存索引并初始化重排器（建议全局单例）。

        :param store: 已构建/加载的索引
        """
        self.store = store
        self.reranker = Reranker(idf=store.sparse.idf)

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 candidate_n: int = 30) -> List[Evidence]:
        """执行完整检索链路。

        :param query: 用户问题（中文/英文）
        :param top_k: 返回证据数
        :param candidate_n: 进入精排的融合候选数
        :return: Evidence列表（相关性降序）
        """
        top_k = top_k or CONFIG.final_top_k
        sparse_query = normalize_query(query)
        dense_hits = self.store.dense_search(query, CONFIG.dense_top_k)
        sparse_hits = self.store.sparse_search(sparse_query, CONFIG.bm25_top_k)
        fused = reciprocal_rank_fusion(dense_hits, sparse_hits)
        dense_rank = {idx: r for r, (idx, _) in enumerate(dense_hits, 1)}
        sparse_rank = {idx: r for r, (idx, _) in enumerate(sparse_hits, 1)}

        candidate_idx = sorted(fused, key=fused.get, reverse=True)[:candidate_n]
        pairs = [(idx, self.store.chunks[idx].text,
                  self.store.chunks[idx].chunk_type) for idx in candidate_idx]
        ranked = self.reranker.rerank(query, pairs, fused)

        evidences: List[Evidence] = []
        for idx, score in ranked[:top_k]:
            chunk = self.store.chunks[idx]
            ev = Evidence(
                chunk_id=chunk.chunk_id, score=score,
                dense_rank=dense_rank.get(idx, -1),
                sparse_rank=sparse_rank.get(idx, -1),
                page_no=chunk.page_no, doc=chunk.doc,
                heading_path=chunk.heading_path,
                text=chunk.text, parent_text=chunk.parent_text,
                chunk_type=chunk.chunk_type)
            if chunk.figure is not None:
                ev.image_path = chunk.figure.image_path
                ev.thumb_path = chunk.figure.thumb_path
                ev.image_caption = chunk.figure.caption
                ev.degraded = chunk.figure.degraded
            evidences.append(ev)
        return evidences
