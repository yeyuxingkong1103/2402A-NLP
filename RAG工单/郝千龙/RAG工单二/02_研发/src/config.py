# -*- coding: utf-8 -*-
# 【全局配置 · config.py】集中管理工单二问答系统的可调参数
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化

"""全局配置：所有阈值集中配置，避免在业务代码中硬编码。"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """系统配置数据类（解析、分块、检索、性能阈值均可在此调整）。"""

    # ---------- PDF 解析 ----------
    # 页眉页脚正则：招股书页眉“招股意向书 1-1-52”可能与正文同行或分两行出现
    header_footer_patterns: List[str] = field(default_factory=lambda: [
        r"招股意向书\s*\d{1,2}-\d{1,2}-\d{1,3}",
        r"招股意向书\s*$",
        r"^\s*\d{1,3}\s*$",
        r"^\s*\d{1,2}-\d{1,2}-\d{1,3}\s*$",
    ])
    # 单页字符数低于该阈值时，判定该页为“表格/图片页”，回退 pdfplumber 表格解析
    table_fallback_chars: int = 60
    # 自定义停用词/噪声词
    noise_words: tuple = ("北京八维信息集团",)

    # ---------- 分块 ----------
    chunk_size: int = 450          # 目标块长度（字符）
    chunk_overlap: int = 80        # 相邻块重叠字符数
    # 标题模式：第X节、一、（一）、1. 等
    heading_pattern: str = r"^(第[一二三四五六七八九十百]+[章节]|[一二三四五六七八九十]+、|（[一二三四五六七八九十]+）|\d+[\.、])"

    # ---------- 检索 ----------
    embedding_model: str = "BAAI/bge-base-zh-v1.5"  # 中英双语 Embedding
    query_instruction: str = "为这个句子生成表示以用于检索相关文章："
    dense_top_k: int = 20         # 向量召回数
    bm25_top_k: int = 20         # 关键词召回数
    rrf_k: int = 60              # RRF 融合常数
    final_top_k: int = 3         # 重排后返回块数
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_truncate: int = 256   # 重排输入截断 token，控制耗时
    title_boost: float = 0.05    # 标题命中 Query 实体的加权
    response_timeout_s: float = 3.0  # 端到端响应预算（验收线 3 秒）

    # ---------- 降级方案 ----------
    # 无 sentence-transformers / 断网时使用的离线嵌入
    fallback_embedding: str = "tfidf"

    # ---------- 路径 ----------
    base_dir: str = field(default_factory=lambda: os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    index_dir: str = ""           # 向量索引持久化目录
    pdf_path: str = ""            # 默认问答文档

    def __post_init__(self) -> None:
        """补齐默认文档与索引目录路径。"""
        root = os.path.dirname(self.base_dir)  # 工单二文件夹
        if not self.pdf_path:
            cand = os.path.join(root, "招股说明书1.pdf")
            self.pdf_path = cand if os.path.exists(cand) else ""
        if not self.index_dir:
            self.index_dir = os.path.join(self.base_dir, "index_store")


# 全局单例配置
CONFIG = Config()
