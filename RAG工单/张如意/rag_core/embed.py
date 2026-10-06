# -*- coding: utf-8 -*-
"""
向量嵌入模块
工单编号：人工智能NLP-RAG（供 01~13 全部工单共用）

默认使用 BAAI/bge-large-zh-v1.5（中文金融/招股书场景表现优秀）。
工单06 要求「支持多种嵌入模型（bge、m3e 及其他）」——本模块通过
`set_model()` 支持运行时切换，并内置模型注册表。
工单11 微调后的模型通过 config.FINETUNED_EMBED_DIR 注册为 "bge-ft-finance"。
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

from . import config

# 关闭 tokenizers 并行告警
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_model = None
_model_name: str | None = None
_lock = threading.Lock()

# bge 系列中文检索官方推荐的 query 前缀
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# ---------------------------------------------------------------------------
# 模型注册表（工单06：支持多种嵌入模型）
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, dict] = {
    "bge-large-zh-v1.5": {
        "path": "BAAI/bge-large-zh-v1.5",
        "dim": 1024,
        "query_prefix": BGE_QUERY_PREFIX,
        "desc": "BGE 中文大模型，1024 维，金融/招股书场景主用",
    },
    "bge-m3": {
        "path": "BAAI/bge-m3",
        "dim": 1024,
        "query_prefix": "",
        "desc": "BGE M3，多语言长文本，支持稠密+稀疏+多向量",
    },
    "m3e-base": {
        "path": "moka-ai/m3e-base",
        "dim": 768,
        "query_prefix": "",
        "desc": "M3E 中文基础模型，768 维，轻量",
    },
    "bge-ft-finance": {                      # 工单11 微调产物
        "path": str(config.FINETUNED_EMBED_DIR),
        "dim": 1024,
        "query_prefix": BGE_QUERY_PREFIX,
        "desc": "基于 bge-large-zh-v1.5 在招股书问答对上微调（工单11产出）",
    },
}


def set_model(name: str):
    """切换当前嵌入模型（工单06 多嵌入模型对比用）。"""
    global _model, _model_name
    with _lock:
        if name == _model_name and _model is not None:
            return _model
        path = MODEL_REGISTRY.get(name, {}).get("path", name)
        if name == "bge-ft-finance" and not Path(path).exists():
            raise FileNotFoundError(
                f"微调模型尚未训练：{path}。请先运行工单11 的微调脚本。"
            )
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(path, device=_device())
        _model_name = name
    return _model


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def current_model_name() -> str:
    return _model_name or "bge-large-zh-v1.5"


def get_model():
    """获取默认嵌入模型（惰性加载）。"""
    global _model
    if _model is None:
        set_model("bge-large-zh-v1.5")
    return _model


def encode(
    texts: list[str] | str,
    is_query: bool = False,
    batch_size: int | None = None,
    normalize: bool = True,
    show_progress: bool = False,
) -> np.ndarray:
    """
    将文本编码为向量。

    Args:
        is_query: True 时对 BGE 系列自动加检索指令前缀（官方要求，
                  不加会明显掉点）
    Returns:
        shape = (n, dim) 的 float32 数组，已 L2 归一化（可直接点积=余弦）
    """
    single = isinstance(texts, str)
    if single:
        texts = [texts]

    model = get_model()
    prefix = MODEL_REGISTRY.get(current_model_name(), {}).get("query_prefix", "")
    if is_query and prefix:
        texts = [prefix + t for t in texts]

    emb = model.encode(
        texts,
        batch_size=batch_size or config.EMBED_BATCH_SIZE,
        normalize_embeddings=normalize,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    emb = np.asarray(emb, dtype=np.float32)
    return emb[0] if single else emb


def similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """余弦相似度（向量已归一化时等价于点积）。"""
    return a @ b.T
