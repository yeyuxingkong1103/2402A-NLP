# -*- coding: utf-8 -*-
# 【全局配置 · config.py】集中管理双招股书表格解析与检索问答的全部可调参数
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

"""全局配置：PDF 路径与公司主体、表格清洗门控、分块、检索、延迟阈值均在此统一调整。"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """系统配置数据类（解析、表格、分块、检索、性能阈值集中配置）。"""

    # ---------- 双招股书与公司主体 ----------
    # 招股说明书1=兴图新科；招股说明书2=力源信息（本工单新增）
    pdf_docs: List[dict] = field(default_factory=list)

    # ---------- 页眉页脚清洗 ----------
    header_footer_patterns: List[str] = field(default_factory=lambda: [
        r"招股意向书\s*\d{0,2}-\d{1,3}-\d{1,3}",
        r"招股意向书\s*$",
        r"^\s*\d{1,3}\s*$",
        r"^\s*\d{0,2}-\d{1,3}-\d{1,3}\s*$",
    ])
    noise_words: tuple = ("北京八维信息集团",)

    # ---------- pdfplumber 表格识别 ----------
    parse_tables_every_page: bool = True   # 本工单核心：每页都尝试识别表格
    table_side_margin_pt: float = 46.0     # 表格左右边框外侧多远内仍回收“越界词”
    table_caption_above_pt: float = 95.0   # 表格上方多远内的文本作为表标题
    table_min_rows: int = 2                # 质量门控：最少行数
    table_min_cols: int = 2                # 质量门控：最少列数
    table_min_fill_rate: float = 0.45      # 质量门控：非空格占比
    table_min_row_ratio: float = 0.70      # 质量门控：至少两格非空的行占比
    table_max_cell_len: int = 60           # 质量门控：单元格平均字符上限（防段落误判）
    text_cover_ratio: float = 0.75         # 正文行被表格单元格语料覆盖比例阈值
    text_cover_min_len: int = 8            # 参与覆盖判定的正文行最短长度

    # ---------- 分块 ----------
    chunk_size: int = 450
    chunk_overlap: int = 80
    heading_pattern: str = (
        r"^(第[一二三四五六七八九十百]+[章节]|[一二三四五六七八九十]+、"
        r"|（[一二三四五六七八九十]+）|\d+[\.、]|\d+、)"
    )

    # ---------- 检索 ----------
    embedding_model: str = "BAAI/bge-base-zh-v1.5"  # 可选语义模型（本地无缓存自动降级）
    dense_top_k: int = 50
    bm25_top_k: int = 50
    rrf_k: int = 60
    final_top_k: int = 3
    candidate_n: int = 40
    answer_top_n: int = 10                 # 答案抽取阶段使用的证据块数
    table_row_boost: float = 0.15          # 表格行陈述句精排加权
    table_whole_boost: float = 0.08        # 整表块精排加权
    response_timeout_s: float = 3.0        # 端到端响应预算（验收线 3 秒）

    # ---------- 离线降级 ----------
    fallback_embedding: str = "tfidf"      # 本环境强制离线 TF-IDF
    tfidf_max_features: int = 30000

    # ---------- 路径 ----------
    base_dir: str = field(default_factory=lambda: os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))  # 02_研发
    index_dir: str = ""

    def __post_init__(self) -> None:
        """补齐双 PDF 路径、公司主体映射与索引目录。"""
        root = os.path.dirname(self.base_dir)  # RAG工单三
        self.pdf_docs = [
            {"file": os.path.join(root, "招股说明书1.pdf"),
             "company": "武汉兴图新科电子股份有限公司",
             "short": "兴图新科",
             "alias": ("兴图", "新科", "xingtuxinke", "xinke",
                       "wuhanxingtu")},
            {"file": os.path.join(root, "招股说明书2.pdf"),
             "company": "武汉力源信息技术股份有限公司",
             "short": "力源信息",
             "alias": ("力源", "liyuan", "wuhanliyuan")},
        ]
        if not self.index_dir:
            self.index_dir = os.path.join(self.base_dir, "index_store")


# 全局单例配置
CONFIG = Config()
