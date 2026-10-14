# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
# 工单编号：人工智能NLP-RAG-功能测试及评估
# （上游来源：工单06 混合检索；本工单复用于 ccf_competition 语料的功能测试与评估）
"""
工单编号：人工智能NLP-RAG-混合检索任务
（工单 PDF 原文：人工智能NLP-RAG-混合检索任务）

模块：运行时检索设置（界面/服务可实时切换策略的单一事实源）

为什么要一个全局设置对象：
  工单要求「三种检索策略可配置、可应用」——用户在界面上下拉切换策略、
  拖动权重、换 Reranker 模型后，**下一次提问立刻生效**，不需要重启进程。
  若把参数一路透传到 retriever/rerankers/rag，调用点会四处漏改；
  用一个线程安全的设置单例，各模块在"未显式传参"时读取它作为默认值，
  显式传参（评估脚本、测试）仍然优先。

线程安全：读写都加锁；读路径每次返回快照（dataclass 拷贝），
避免并发请求读到"改了一半"的配置。
"""
from __future__ import annotations

import threading
from dataclasses import replace

from src import config
from src.strategies import RetrievalConfig

_lock = threading.RLock()
_settings: RetrievalConfig = RetrievalConfig()
_evidence: dict = {"updated_by": "default", "revision": 0}


def get() -> RetrievalConfig:
    """取当前设置快照。"""
    with _lock:
        return replace(_settings)


def update(**kwargs) -> RetrievalConfig:
    """局部更新设置（界面调用）。返回更新后的快照。

    仅接受 RetrievalConfig 的字段；策略/融合/算法的非法取值会被拒绝
    （抛 ValueError），由调用方（界面）提示用户。
    """
    global _settings
    with _lock:
        valid = {k: v for k, v in kwargs.items() if v is not None}
        unknown = set(valid) - set(RetrievalConfig.__dataclass_fields__)
        if unknown:
            raise ValueError(f"未知设置项：{sorted(unknown)}")
        if "strategy" in valid and valid["strategy"] not in config.RETRIEVAL_STRATEGIES:
            raise ValueError(f"未知检索策略 {valid['strategy']!r}")
        if "fusion" in valid and valid["fusion"] not in config.FUSION_METHODS:
            raise ValueError(f"未知融合算法 {valid['fusion']!r}")
        if ("rerank_algorithm" in valid
                and valid["rerank_algorithm"] not in config.RERANK_ALGORITHMS):
            raise ValueError(f"未知精排算法 {valid['rerank_algorithm']!r}")
        if ("reranker_model" in valid
                and valid["reranker_model"] not in config.RERANKER_MODELS):
            raise ValueError(f"未知 Reranker 模型 {valid['reranker_model']!r}")
        if ("embedding_model" in valid
                and valid["embedding_model"] not in config.EMBEDDING_MODELS):
            raise ValueError(f"未知嵌入模型 {valid['embedding_model']!r}")
        _settings = replace(_settings, **valid)
        _evidence["revision"] += 1
        _evidence["updated_by"] = kwargs.pop("_who", "api")
        return replace(_settings)


def reset() -> RetrievalConfig:
    """恢复默认（测试/收尾用）。"""
    global _settings
    with _lock:
        _settings = RetrievalConfig()
        _evidence["revision"] += 1
        _evidence["updated_by"] = "reset"
        return replace(_settings)


def describe() -> dict:
    """当前设置的可展示快照。"""
    cfg = get()
    return {
        "strategy": cfg.strategy,
        "strategy_label": config.RETRIEVAL_STRATEGIES.get(cfg.strategy, cfg.strategy),
        "top_k_recall": cfg.top_k_recall,
        "top_k_rerank": cfg.top_k_rerank,
        "score_threshold": cfg.score_threshold,
        "vector_weight": cfg.vector_weight,
        "fulltext_weight": cfg.fulltext_weight,
        "fusion": cfg.fusion,
        "rerank_enabled": cfg.rerank_enabled,
        "rerank_algorithm": cfg.rerank_algorithm,
        "reranker_model": cfg.reranker_model,
        "fulltext_rerank": cfg.fulltext_rerank,
        "embedding_model": cfg.embedding_model,
        "revision": _evidence["revision"],
        "updated_by": _evidence["updated_by"],
    }
