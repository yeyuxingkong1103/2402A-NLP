# -*- coding: utf-8 -*-
"""工单9 Graph RAG 三层面优化包。

工单编号：人工智能NLP-RAG-Graph RAG 优化任务

模块划分（与 `设计/优化方案设计.md` 的三层面一一对应）：

    配置.py    —— 全部优化开关与阈值（唯一真源，可被 部署/配置/优化配置.json 与环境变量覆盖）
    抽取层.py  —— **实体/关系构建层**：别名归并、关系方向与类型规范化、噪声剪枝、置信度过滤
    检索层.py  —— **检索层**：查询改写与实体链接、图谱跳数与剪枝、图特征×向量/BM25 融合、重排裁剪、同页扩展
    生成层.py  —— **生成层**：图谱三元组结构化注入、引用约束、拒答护栏、模板化落地生成
    引擎.py    —— 端到端编排（优化前/优化后两套 profile 共用同一套评估口径）

用法::

    from app.graphrag_opt.配置 import get_config
    from app.graphrag_opt.引擎 import OptimizedEngine
    eng = OptimizedEngine(profile="after")
    ans = eng.answer("平安银行2019年营业收入是多少？")
"""

from __future__ import annotations

WORK_ORDER = "人工智能NLP-RAG-Graph RAG 优化任务"

__all__ = ["WORK_ORDER"]
