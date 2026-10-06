# -*- coding: utf-8 -*-
# 【全局配置 · config.py】集中管理工单五多轮问答系统的可调参数
# 工单编号：人工智能NLP-RAG-Query理解优化任务

"""全局配置：解析、分块、检索、多轮对话状态、性能阈值集中配置。"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """系统配置数据类（所有阈值均可在此处调整，避免业务代码硬编码）。"""

    # ---------- PDF 解析 ----------
    header_footer_patterns: List[str] = field(default_factory=lambda: [
        r"招股意向书\s*\d{1,2}-\d{1,2}-\d{1,3}",
        r"招股意向书\s*$",
        r"^\s*\d{1,3}\s*$",
        r"^\s*\d{1,2}-\d{1,2}-\d{1,3}\s*$",
    ])
    table_fallback_chars: int = 60
    noise_words: tuple = ("北京八维信息集团",)

    # ---------- 分块 ----------
    chunk_size: int = 450
    chunk_overlap: int = 80
    heading_pattern: str = r"^(第[一二三四五六七八九十百]+[章节]|[一二三四五六七八九十]+、|（[一二三四五六七八九十]+）|\d+[\.、])"

    # ---------- 检索 ----------
    embedding_mode: str = "tfidf"        # 无 GPU 无外网，强制 TF-IDF 离线路径
    dense_top_k: int = 25
    bm25_top_k: int = 25
    rrf_k: int = 60
    final_top_k: int = 3
    response_timeout_s: float = 3.0       # 端到端响应预算（验收线 3 秒）

    # ---------- 多轮对话 ----------
    max_history_turns: int = 8            # 对话历史滑窗保留的最大轮数
    max_history_chars: int = 800          # 单条历史摘要最大字符数
    entity_window_turns: int = 3          # 实体槽位回溯轮数（指代消解用）

    # ---------- 路径 ----------
    base_dir: str = field(default_factory=lambda: os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    index_dir: str = ""
    pdf_paths: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """补齐默认文档与索引目录路径。"""
        root = os.path.dirname(self.base_dir)  # 工单五文件夹
        if not self.pdf_paths:
            for name in ("招股说明书1.pdf", "招股说明书2.pdf"):
                cand = os.path.join(root, name)
                if os.path.exists(cand):
                    self.pdf_paths.append(cand)
        if not self.index_dir:
            self.index_dir = os.path.join(self.base_dir, "index_store")


# 全局单例配置
CONFIG = Config()
