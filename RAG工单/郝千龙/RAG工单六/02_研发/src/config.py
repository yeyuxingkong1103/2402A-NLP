# -*- coding: utf-8 -*-
# 【全局配置 · config.py】集中管理混合检索系统的解析/分块/嵌入/多字段/融合/重排/反馈参数，支持融合方式与权重热配
# 工单编号：人工智能NLP-RAG-混合检索任务

"""全局配置模块。

本模块集中管理工单六全部可调参数，业务代码不允许硬编码阈值。其中：
- ``CONFIG`` 为建库期静态配置（PDF 路径、分块参数、字段权重、模型名等）；
- ``RT_CONFIG`` 为运行期热配置（检索模式、融合方式、双路权重、重排器），
  Streamlit 侧边栏或评测脚本可在不重新建库的情况下即时调整，
  对应工单“支持向量检索和全文检索的权重调整”“提供混合检索的配置选项”要求。
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List


# 本地可用的多嵌入模型注册表：键为业务短名，值为 HuggingFace 模型ID/本地路径
# bge（BAAI 中英双语）、m3e（Moka 中文）均要求“本地有缓存才加载，禁止联网下载”，
# 本地无缓存时由 embeddings 工厂自动降级为 TF-IDF 离线嵌入
EMBEDDER_MODELS: Dict[str, str] = {
    "bge": "BAAI/bge-base-zh-v1.5",
    "bge-large": "BAAI/bge-large-zh-v1.5",
    "bge-m3": "BAAI/bge-m3",
    "m3e": "moka-ai/m3e-base",
    "m3e-small": "moka-ai/m3e-small",
    "tfidf": "tfidf",  # 离线降级嵌入，不依赖任何外部模型
}

# bge 系列中文模型检索时官方建议追加的查询指令
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


@dataclass
class Config:
    """系统静态配置（解析、分块、索引、性能预算等）。"""

    # ---------- PDF 解析 ----------
    # 页眉页脚正则：招股书页眉形如“招股意向书 1-1-52”“招股意向书 304”
    header_footer_patterns: List[str] = field(default_factory=lambda: [
        r"招股意向书\s*\d{1,2}-\d{1,2}-\d{1,3}",
        r"招股意向书\s*\d{1,4}\s*$",
        r"招股说明书\s*\d{1,2}-\d{1,2}-\d{1,3}",
        r"^\s*\d{1,3}\s*$",
        r"^\s*\d{1,2}-\d{1,2}-\d{1,3}\s*$",
    ])
    # 单页字符数低于该阈值判定为“表格/图片页”，回退 pdfplumber 表格解析
    table_fallback_chars: int = 60
    # 噪声词（跨两份招股书通用的页眉署名不在此硬编码，解析层用公司名正则清洗）
    noise_words: tuple = ()

    # ---------- 分块 ----------
    chunk_size: int = 420          # 目标块长度（字符）
    chunk_overlap: int = 70        # 相邻块重叠字符数
    # 标题模式：第X节、一、（一）、1. 等
    heading_pattern: str = r"^(第[一二三四五六七八九十百]+[章节]|[一二三四五六七八九十]+、|（[一二三四五六七八九十]+）|\d+[\.、])"
    summary_max_chars: int = 80    # 摘要字段最大长度（标题+首句）

    # ---------- 多嵌入模型 ----------
    embedding_backend: str = "bge"  # 可选键见 EMBEDDER_MODELS：bge / m3e / tfidf
    query_instruction: str = BGE_QUERY_INSTRUCTION

    # ---------- 召回 ----------
    # 双路各取60：招股书事实卡/长表格在单路名次常在20~60（词法/语义各有盲区），
    # 30条截断会把注册资本卡、客户表等gold块截在融合池外；60条仍为毫秒级开销
    dense_top_k: int = 60          # 向量（语义）召回数
    bm25_top_k: int = 60           # 全文（关键词）召回数
    recall_pool_k: int = 80        # 融合候选池上限
    final_top_k: int = 5           # 重排后默认返回块数
    eval_top_k: int = 10           # 评测召回率口径 Recall@10 的 k

    # ---------- 多字段加权（全文检索） ----------
    # 标题/公司实体/表格/摘要/正文 五个字段的字段权重（工单要求多字段加权）
    field_weights: Dict[str, float] = field(default_factory=lambda: {
        "title": 3.0,      # 标题字段：章节名直接命中，权重最高
        "company": 2.5,    # 公司实体字段：发行人/子公司实体消歧
        "table": 2.0,      # 表格字段：数字类事实密度高
        "summary": 1.5,    # 摘要字段：标题+首句
        "body": 1.0,       # 正文字段：基础权重
        "parent": 0.8,     # 父段落字段：Small-to-Big 上下文（引导句在父段落时召回子块）
    })
    phrase_boost: float = 0.35     # 短语（相邻词项有序命中）加分系数
    fuzzy_threshold: float = 0.75  # 模糊匹配字符 bigram 相似度阈值
    # 公司实体字段不作为普通 BM25 字段线性叠加（实体名在同文档每块近似常量，
    # 会压垮内容相关性），改为“文档级实体消歧先验”：Query 命中该文档独有
    # 实体词（如“兴图新科/力源”）时，仅对该文档的块加固定小分
    entity_prior_score: float = 6.0

    # ---------- 融合（可热配） ----------
    rrf_k: int = 60                # RRF 倒数排名融合常数
    default_vector_weight: float = 0.5   # 混合检索中向量路默认权重
    default_fulltext_weight: float = 0.5  # 混合检索中全文路默认权重

    # ---------- 重排 ----------
    default_reranker: str = "llm"  # llm / tfidf / adaptive
    reranker_model: str = "BAAI/bge-reranker-large"  # 本地 Cross-Encoder（LLM重排器）
    llm_rerank_candidates: int = 12   # 进入 LLM 重排的候选数（CPU 3秒预算）
    other_rerank_candidates: int = 60  # 进入 TF-IDF/自适应重排的候选数（随召回池加深）
    rerank_truncate: int = 96      # LLM 重排单候选正文精华窗口字符数（去标题前缀后）
    channel_guarantee: int = 6     # LLM 候选池对召回双路各保底的条数（防单通道漏召回）
    llm_lexical_blend: float = 0.28  # LLM 重排中词法/答案类型特征的混合权重
    llm_rerank_num_penalty: float = 0.25  # 数字型问题候选块无数字时的惩罚
    adaptive_positive: float = 0.6   # 自适应重排：采纳信号加分系数
    adaptive_negative: float = 0.8   # 自适应重排：不采纳信号惩罚系数
    adaptive_query_sim: float = 0.45  # 历史相似问题触发阈值（TF-IDF 余弦）

    # ---------- 验收性能预算 ----------
    response_timeout_s: float = 3.0  # 端到端响应预算（验收线 3 秒）

    # ---------- 路径 ----------
    base_dir: str = field(default_factory=lambda: os.path.dirname(
        os.path.abspath(__file__)))                # 02_研发/src
    rnd_dir: str = ""                              # 02_研发
    work_order_dir: str = ""                       # RAG工单六
    index_dir: str = ""                            # 索引持久化目录
    pdf_files: List[str] = field(default_factory=list)  # 两份招股书
    feedback_file: str = ""                        # 用户反馈持久化文件

    def __post_init__(self) -> None:
        """补齐研发目录、工单目录、语料 PDF 与索引目录路径。"""
        self.rnd_dir = os.path.dirname(self.base_dir)
        self.work_order_dir = os.path.dirname(self.rnd_dir)
        if not self.pdf_files:
            cands = [os.path.join(self.work_order_dir, "招股说明书1.pdf"),
                     os.path.join(self.work_order_dir, "招股说明书2.pdf")]
            self.pdf_files = [p for p in cands if os.path.exists(p)]
        if not self.index_dir:
            self.index_dir = os.path.join(self.rnd_dir, "index_store")
        if not self.feedback_file:
            self.feedback_file = os.path.join(self.index_dir, "feedback.jsonl")


@dataclass
class RuntimeConfig:
    """运行期热配置：不重建索引即可切换检索模式/融合方式/双路权重/重排器。"""

    search_mode: str = "hybrid"        # vector（向量）/ fulltext（全文）/ hybrid（混合）
    fusion_method: str = "rrf"         # weighted_avg（加权平均）/ vote（投票）/ rrf
    vector_weight: float = 0.5         # 向量路权重
    fulltext_weight: float = 0.5       # 全文路权重
    reranker_name: str = "llm"         # llm / tfidf / adaptive
    embedding_backend: str = "bge"     # 仅在建库/展示时使用

    def set_weights(self, vector_weight: float, fulltext_weight: float) -> None:
        """热更新双路权重（界面滑块直接调用）。

        :param vector_weight: 向量路权重（自动归一）
        :param fulltext_weight: 全文路权重（自动归一）
        """
        total = float(vector_weight) + float(fulltext_weight)
        if total <= 0:
            total = 1.0
        self.vector_weight = float(vector_weight) / total
        self.fulltext_weight = float(fulltext_weight) / total


# 全局单例
CONFIG = Config()
RT_CONFIG = RuntimeConfig(
    vector_weight=CONFIG.default_vector_weight,
    fulltext_weight=CONFIG.default_fulltext_weight,
    reranker_name=CONFIG.default_reranker,
    embedding_backend=CONFIG.embedding_backend,
)
