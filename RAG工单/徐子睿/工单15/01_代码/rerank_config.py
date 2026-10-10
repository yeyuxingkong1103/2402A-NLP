# -*- coding: utf-8 -*-
"""
工单 15 · 跨模态重排与检索参数配置（外部化，不硬编码）

说明：RAGFlow 的检索参数通过知识库/对话的配置下发：
  - vector_similarity_weight : 向量分权重（默认 0.3，tkweight = 1 - 该值）
  - similarity_threshold     : 最低相似度（默认 0.2）
  - rerank_mdl               : 绑定的重排模型（LLMType.RERANK）
本模块把这些参数 + 新增的「双路召回融合」参数集中管理，供 patched 检索层读取。
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass
class CrossModalRetrievalConfig:
    # —— 融合策略 ——
    fusion: str = os.environ.get("RAG_FUSION", "rrf")      # rrf | weighted
    rrf_k: int = _env_int("RAG_RRF_K", 60)
    route_weights: tuple = (1.0, 1.0)                       # [文本路, 图像增强路]

    # —— 加权混合（RAGFlow 原生）——
    vector_similarity_weight: float = _env_float("RAG_VSW", 0.3)
    similarity_threshold: float = _env_float("RAG_SIM_TH", 0.2)

    # —— 重排 ——
    rerank_switch: bool = os.environ.get("RAG_RERANK", "1") == "1"
    rerank_limit: int = _env_int("RAG_RERANK_LIMIT", 30)

    # —— 视觉引用识别 ——
    enable_visual_ref: bool = os.environ.get("RAG_VISUAL_REF", "1") == "1"

    @property
    def tk_weight(self) -> float:
        return 1.0 - self.vector_similarity_weight


if __name__ == "__main__":
    c = CrossModalRetrievalConfig()
    print(c)
