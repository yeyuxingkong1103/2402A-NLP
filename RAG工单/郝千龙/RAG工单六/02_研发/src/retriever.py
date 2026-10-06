# -*- coding: utf-8 -*-
# 【统一检索编排 · retriever.py】向量召回/全文召回/混合召回三路可选，融合与重排策略可热配切换
# 工单编号：人工智能NLP-RAG-混合检索任务

"""检索层主链路（对应设计文档“总体检索链路”）：

1. 召回阶段（三选一，可热配）：
   - vector   向量检索：多模型嵌入（bge/m3e，离线 TF-IDF 降级）+ 余弦 Top-N；
   - fulltext 全文检索：五字段加权 BM25 倒排（布尔/短语/模糊匹配）；
   - hybrid   混合检索：两路同时召回，再经融合算法合并。
2. 融合阶段（混合模式下三选一，可热配）：
   weighted_avg 加权平均 / vote 投票融合(Borda) / rrf 倒数排名融合；
   双路权重 w_vec、w_full 支持运行期滑块热配。
3. 重排阶段（三选一，可热配）：
   llm LLM(Cross-Encoder)重排 / tfidf TF-IDF重排 / adaptive 用户反馈自适应重排。
"""
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import CONFIG, RT_CONFIG
from feedback_store import FeedbackStore
from fusion import fuse
from rerankers import (
    BaseReranker, FeedbackAdaptiveReranker, LLMReranker,
    RERANKER_REGISTRY, TfidfReranker, expand_cross_lingual,
    expand_zh_synonyms,
)
from vector_store import IndexStore

# 查询中的公司实体 -> 唯一所属招股书（多文档库实体消歧，防止跨文档串库）。
# 命中规则按特异性排序：力源/赵马克只在招股书2，兴图/新科/程家明只在招股书1。
_ENTITY_DOC_RULES = [
    (re.compile(r"力源|赵马克|P&S", re.IGNORECASE), "招股说明书2"),
    (re.compile(r"兴图|新科|程家明|Xingtu", re.IGNORECASE), "招股说明书1"),
]


def infer_doc(query: str) -> Optional[str]:
    """从查询文本推断唯一目标文档（公司实体硬消歧）。

    问题显式提及发行人简称/高管/英文名时，答案只可能出自该文档，
    直接作为检索硬过滤；未提及实体时返回 None（全库召回）。

    :param query: 原始或跨语言扩展后的查询
    :return: 文档名；无法判定时为 None
    """
    for pattern, doc in _ENTITY_DOC_RULES:
        if pattern.search(query):
            return doc
    return None


@dataclass
class Evidence:
    """单条检索证据（携带双路名次、融合分与多字段定位信息）。"""

    chunk_id: int
    score: float
    dense_rank: int       # 向量路名次（未召回为 -1）
    sparse_rank: int      # 全文路名次（未召回为 -1）
    page_no: int
    doc_name: str         # 来源招股书
    heading_path: str
    chunk_type: str       # text / table
    text: str             # 命中块文本
    parent_text: str      # Small-to-Big 父段落
    retrieve_ms: float = 0.0  # 本次检索+重排耗时（毫秒）


class HybridRetriever:
    """组装双路召回、三种融合与三种重排的统一检索器（建议全局单例）。"""

    def __init__(self, store: IndexStore,
                 feedback_store: Optional[FeedbackStore] = None) -> None:
        """注入索引与反馈存储，并预构建三种重排器。

        :param store: 已构建/加载的多字段索引
        :param feedback_store: 用户反馈存储（自适应重排使用）
        """
        self.store = store
        self.feedback_store = feedback_store or FeedbackStore(CONFIG.feedback_file)
        chunk_texts = [c.text for c in store.chunks]
        idf = store.sparse.idf
        # 重排器缓存：LLM 模型只加载一次；TF-IDF/自适应全库矩阵只拟合一次
        self._rerankers: Dict[str, BaseReranker] = {
            "llm": LLMReranker(idf=idf),
            "tfidf": TfidfReranker(chunk_texts, idf=idf),
            "adaptive": FeedbackAdaptiveReranker(
                chunk_texts, self.feedback_store, idf=idf),
        }

    def get_reranker(self, name: str) -> BaseReranker:
        """按短名取重排器（自适应重排器在反馈变化后自动刷新画像）。

        :param name: llm / tfidf / adaptive
        :return: 重排器实例
        """
        reranker = self._rerankers[name]
        if name == "adaptive":
            # 反馈 JSONL 可能被界面或评测脚本追加，重建画像保证读到最新反馈
            reranker.refresh()
        return reranker

    def _recall(self, query: str, mode: str,
                doc_name: Optional[str] = None
                ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """按模式执行召回（支持按文档硬过滤，仅在目标招股书块内召回）。

        召回查询保留公司全称（多字段 BM25 的 company 字段靠它做跨文档
        实体消歧，高频词由 BM25-IDF 自动降权）；英文问题先翻译/追加中文检索词。

        :param query: 原始问题（或跨语言扩展后的问题）
        :param mode: vector / fulltext / hybrid
        :param doc_name: 目标文档名（招股说明书1/招股说明书2），None 为全库
        :return: (向量路命中, 全文路命中)；未执行的通道返回空表
        """
        dense_hits: List[Tuple[int, float]] = []
        sparse_hits: List[Tuple[int, float]] = []
        if mode in ("vector", "hybrid"):
            dense_hits = self.store.dense_search(
                query, CONFIG.dense_top_k, doc_name=doc_name)
        if mode in ("fulltext", "hybrid"):
            sparse_hits = self.store.fulltext_search(
                query, CONFIG.bm25_top_k, doc_name=doc_name)
        return dense_hits, sparse_hits

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 mode: Optional[str] = None, fusion_method: Optional[str] = None,
                 reranker_name: Optional[str] = None,
                 vector_weight: Optional[float] = None,
                 fulltext_weight: Optional[float] = None,
                 doc_name: Optional[str] = None) -> List[Evidence]:
        """执行一次完整检索（参数缺省时读取 RT_CONFIG 热配置）。

        :param query: 用户问题（中/英）
        :param top_k: 最终返回证据数
        :param mode: vector / fulltext / hybrid
        :param fusion_method: weighted_avg / vote / rrf（仅混合模式）
        :param reranker_name: llm / tfidf / adaptive
        :param vector_weight: 向量路热配权重
        :param fulltext_weight: 全文路热配权重
        :param doc_name: 文档过滤名（仅在该文档块内召回与融合）；
            缺省时按查询中的公司实体（力源/兴图新科等）自动硬消歧，
            实体也无法判定才全库召回，杜绝跨文档串库
        :return: Evidence 列表，相关性降序
        """
        t0 = time.perf_counter()
        mode = mode or RT_CONFIG.search_mode
        fusion_method = fusion_method or RT_CONFIG.fusion_method
        reranker_name = reranker_name or RT_CONFIG.reranker_name
        w_vec = (vector_weight if vector_weight is not None
                 else RT_CONFIG.vector_weight)
        w_full = (fulltext_weight if fulltext_weight is not None
                  else RT_CONFIG.fulltext_weight)
        top_k = top_k or CONFIG.final_top_k

        # ① 查询扩展：英文先离线翻译为中文并保留英文原句；中文追加同义表述
        #    （行文用词对齐，如“前五大客户”↔“前五名客户”），双路召回共用
        search_query = expand_zh_synonyms(expand_cross_lingual(query))
        # ② 文档过滤：显式指定优先；否则按公司实体硬消歧（含英文公司名）
        target_doc = doc_name or infer_doc(search_query)
        dense_hits, sparse_hits = self._recall(
            search_query, mode, doc_name=target_doc)

        # ③ 融合（单路模式直接使用该路分数；双路榜单已在同一文档内）
        if mode == "vector":
            base_scores = dict(dense_hits)
        elif mode == "fulltext":
            base_scores = dict(sparse_hits)
        else:
            base_scores = fuse(fusion_method, dense_hits, sparse_hits,
                               w_vec=w_vec, w_full=w_full)

        # ④ 候选池截取（LLM 重排为 CPU 3 秒预算只进十余个候选）；
        #    混合模式下对双路各自 Top-N 保底，防止“一路强相关但融合分靠后”
        #    的精确数字块被散文块挤出 LLM 候选池
        candidate_n = (CONFIG.llm_rerank_candidates if reranker_name == "llm"
                       else CONFIG.other_rerank_candidates)
        fused_order = sorted(base_scores, key=base_scores.get, reverse=True)
        candidate_idx = fused_order[:candidate_n]
        if mode == "hybrid" and reranker_name == "llm":
            pool_set = set(candidate_idx)
            guaranteed = ([i for i, _ in dense_hits[:CONFIG.channel_guarantee]]
                          + [i for i, _ in sparse_hits[:CONFIG.channel_guarantee]])
            extras = [i for i in guaranteed if i not in pool_set]
            keep_n = max(0, candidate_n - len(extras))
            candidate_idx = fused_order[:keep_n] + extras

        pairs = [(idx, self.store.chunks[idx].text) for idx in candidate_idx]

        # ⑤ 重排（重排查询同样携带中文翻译词，重排候选已全部来自目标文档）
        ranked = self.get_reranker(reranker_name).rerank(
            search_query, pairs, base_scores)

        dense_rank = {idx: r for r, (idx, _) in enumerate(dense_hits, 1)}
        sparse_rank = {idx: r for r, (idx, _) in enumerate(sparse_hits, 1)}
        elapsed_ms = (time.perf_counter() - t0) * 1000

        evidences: List[Evidence] = []
        for idx, score in ranked[:top_k]:
            chunk = self.store.chunks[idx]
            evidences.append(Evidence(
                chunk_id=chunk.chunk_id, score=score,
                dense_rank=dense_rank.get(idx, -1),
                sparse_rank=sparse_rank.get(idx, -1),
                page_no=chunk.page_no, doc_name=chunk.doc_name,
                heading_path=chunk.heading_path, chunk_type=chunk.chunk_type,
                text=chunk.text, parent_text=chunk.parent_text,
                retrieve_ms=elapsed_ms,
            ))
        return evidences
