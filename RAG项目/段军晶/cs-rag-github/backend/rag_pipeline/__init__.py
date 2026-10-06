# -*- coding: utf-8 -*-
"""
RAG 检索链路包

各版本链路说明（迭代为「逐轮增强」，V1 代码始终保留可运行）：
    v1_pipeline.py —— V1 基线：朴素稠密检索
    v2_pipeline.py —— V2 增强：视觉解析 + 稠密/稀疏混合检索 + RRF 融合
    v3_pipeline.py —— V3 进阶：查询改写 + BGE-reranker 重排

保留多版本并存的原因（见技术决策记录 ADR-010）：
    M5 门禁要求给出「对比基线的指标提升数据」。若 V2 直接覆盖 V1，
    就无法在同一环境下重新运行 V1 验证基线，也无法在指标波动时回退定位。
"""

from typing import Any


def get_pipeline_by_name(name: str) -> Any:
    """
    按版本名获取链路实例。

    用于评测脚本在同一套评测集上横向对比各版本（V1/V2/V3 共用同一接口：
    answer / retrieve），也供 API 层按配置切换链路。
    """
    key = (name or "v1").strip().lower()

    if key == "v1":
        from backend.rag_pipeline.v1_pipeline import get_pipeline
        return get_pipeline()

    if key == "v2":
        from backend.rag_pipeline.v2_pipeline import get_pipeline
        return get_pipeline()

    if key == "v3":
        from backend.rag_pipeline.v3_pipeline import get_pipeline
        return get_pipeline()

    raise ValueError(f"未知的链路版本：{name}（可选 v1 / v2 / v3）")


__all__ = ["get_pipeline_by_name"]
