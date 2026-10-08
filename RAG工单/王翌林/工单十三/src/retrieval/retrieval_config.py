# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
src/retrieval/retrieval_config.py —— 工单六 检索策略配置中心

集中定义三类检索策略的可配置项：
  1. 检索模式 mode：vector（向量召回+重排）/ fulltext（全文检索）/ hybrid（混合）
  2. 融合算法 fusion：rrf（投票机制）/ weighted（加权平均），权重可调
  3. 重排器 reranker：llm（交叉编码重排器）/ tfidf（TF-IDF 重排器）/
     adaptive（基于用户反馈的自适应重排器）
  4. 全文检索匹配方式 match：and / or / phrase / fuzzy
  5. 多字段 fields：title(=doc_id) / content(正文) / summary(摘要)
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

WORK_ORDER = "人工智能NLP-RAG-混合检索任务"

# 工单六：可选值常量（供 API/UI 下拉与参数校验使用）
MODE_VECTOR = "vector"
MODE_FULLTEXT = "fulltext"
MODE_HYBRID = "hybrid"
VALID_MODES = (MODE_VECTOR, MODE_FULLTEXT, MODE_HYBRID)

FUSION_RRF = "rrf"              # 工单六：投票机制（Reciprocal Rank Fusion）
FUSION_WEIGHTED = "weighted"    # 工单六：加权平均
VALID_FUSIONS = (FUSION_RRF, FUSION_WEIGHTED)

RERANKER_LLM = "llm"            # 工单六：基于 LLM/交叉编码器的重排器（bge-reranker-v2-m3）
RERANKER_TFIDF = "tfidf"        # 工单六：基于 TF-IDF 的重排器
RERANKER_ADAPTIVE = "adaptive"  # 工单六：基于用户反馈的自适应重排器
VALID_RERANKERS = (RERANKER_LLM, RERANKER_TFIDF, RERANKER_ADAPTIVE)

MATCH_AND = "and"               # 工单六：布尔 AND（所有关键词命中）
MATCH_OR = "or"                 # 工单六：布尔 OR（任一关键词命中）
MATCH_PHRASE = "phrase"         # 工单六：短语匹配（连续子串）
MATCH_FUZZY = "fuzzy"           # 工单六：模糊匹配（字符 bigram Jaccard）
VALID_MATCHES = (MATCH_AND, MATCH_OR, MATCH_PHRASE, MATCH_FUZZY)

FIELD_TITLE = "title"           # 工单六：标题字段（doc_id 文档名）
FIELD_CONTENT = "content"       # 工单六：正文字段（chunk 全文）
FIELD_SUMMARY = "summary"       # 工单六：摘要字段（chunk 前 80 字）
VALID_FIELDS = (FIELD_TITLE, FIELD_CONTENT, FIELD_SUMMARY)

# 工单六：字段默认权重（标题命中信号最强但短，正文为主，摘要辅助）
DEFAULT_FIELD_WEIGHTS = {FIELD_TITLE: 0.5, FIELD_CONTENT: 1.0, FIELD_SUMMARY: 0.6}


@dataclass
class RetrievalConfig:
    """工单六：检索策略配置（API/UI 可覆盖任意字段）"""

    mode: str = MODE_HYBRID                 # 检索模式
    fusion: str = FUSION_RRF                # 混合融合算法（仅 hybrid 生效）
    reranker: str = RERANKER_LLM            # 重排器
    vector_weight: float = 0.6              # 混合检索：向量通道权重（weighted 融合）
    fulltext_weight: float = 0.4            # 混合检索：全文通道权重（weighted 融合）
    match: str = MATCH_AND                  # 全文检索匹配方式
    fields: List[str] = field(
        default_factory=lambda: [FIELD_CONTENT, FIELD_TITLE, FIELD_SUMMARY])
    field_weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_FIELD_WEIGHTS))
    top_k: int = 8                          # 最终返回条数
    vector_recall_k: int = 24               # 向量召回候选数
    fulltext_recall_k: int = 24             # 全文召回候选数
    rrf_k: int = 60                         # RRF 融合常数
    use_table: bool = True                  # 工单六：保留工单四表格通道
    use_image: bool = True                  # 工单六：保留工单四图像通道

    def __post_init__(self) -> None:
        """工单六：参数校验（非法值回退默认，保证服务容错）"""
        if self.mode not in VALID_MODES:
            self.mode = MODE_HYBRID
        if self.fusion not in VALID_FUSIONS:
            self.fusion = FUSION_RRF
        if self.reranker not in VALID_RERANKERS:
            self.reranker = RERANKER_LLM
        if self.match not in VALID_MATCHES:
            self.match = MATCH_AND
        # 工单六：权重归一化（weighted 融合要求二者和为 1）
        s = self.vector_weight + self.fulltext_weight
        if s <= 0:
            self.vector_weight, self.fulltext_weight = 0.6, 0.4
        else:
            self.vector_weight = round(self.vector_weight / s, 3)
            self.fulltext_weight = round(self.fulltext_weight / s, 3)
        if not self.fields:
            self.fields = [FIELD_CONTENT]
        self.fields = [f for f in self.fields if f in VALID_FIELDS]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "RetrievalConfig":
        """工单六：从 API 请求体构造（忽略 None 与未知字段）"""
        if not data:
            return cls()
        known = {k: v for k, v in data.items()
                 if v is not None and k in cls.__dataclass_fields__}
        return cls(**known)


# 工单六：三套预设策略（演示"检索策略的配置及应用"一键切换）
PRESETS: Dict[str, Dict[str, Any]] = {
    "vector_only": {
        "mode": MODE_VECTOR, "reranker": RERANKER_LLM,
        "description": "向量检索：bge-m3 嵌入 + 余弦相似度召回 + LLM 交叉编码重排"},
    "fulltext_only": {
        "mode": MODE_FULLTEXT, "reranker": RERANKER_TFIDF, "match": MATCH_AND,
        "description": "全文检索：倒排索引 + 布尔/短语/模糊匹配 + TF-IDF 评分"},
    "hybrid_rrf_llm": {
        "mode": MODE_HYBRID, "fusion": FUSION_RRF, "reranker": RERANKER_LLM,
        "description": "混合检索：向量+全文 RRF 投票融合 + LLM 重排（默认，精度最优）"},
    "hybrid_weighted_adaptive": {
        "mode": MODE_HYBRID, "fusion": FUSION_WEIGHTED,
        "vector_weight": 0.6, "fulltext_weight": 0.4,
        "reranker": RERANKER_ADAPTIVE,
        "description": "混合检索：加权平均（0.6/0.4）+ 用户反馈自适应重排"},
}
