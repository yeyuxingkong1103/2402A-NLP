# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块：向量化（Embedding）

选型：BAAI/bge-m3 —— 本地路径 D:\\model\\bge-m3（1024 维）
理由：
  - 工单优先级为 "优先 BGE 系列（bge-large-zh、bge-m3）"，bge-m3 是其首选；
  - 1024 维、最长 8194 token，中英双语（工单要求支持中英文问答）；
  - 本地是完整的 sentence-transformers 目录（含 modules.json / 1_Pooling），可直接离线加载。

硬约束遵守：
  - 构造时显式传 `local_files_only=True`；
  - 进程启动即由 bootstrap 置 HF_HUB_OFFLINE=1，双保险；
  - 设备优先 cuda，不可用时自动回退 cpu，绝不触发下载。
"""

from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

import os
import threading
import time
from typing import Sequence

from src import config

_model = None
_model_lock = threading.Lock()
_device_used = None


def _pick_device() -> str:
    """选择可用设备：cuda -> cpu。"""
    want = config.EMBEDDING_DEVICE
    if want.startswith("cuda"):
        try:
            import torch
            if torch.cuda.is_available():
                return want
        except Exception:
            pass
        return "cpu"
    return want


def get_model():
    """懒加载并缓存 embedding 模型（线程安全）。"""
    global _model, _device_used
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model

        path = config.EMBEDDING_MODEL_PATH
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"本地 embedding 模型不存在: {path}\n"
                f"请修改 src/config.py 的 EMBEDDING_MODEL_PATH 指向本机已下载的模型目录。")

        from sentence_transformers import SentenceTransformer

        device = _pick_device()
        t0 = time.time()
        # 显存/内存吃紧的机器上按 fp16 加载（bge-m3 约 1.1G 而非 2.2G）。
        # 本机 8G 显存常被桌面应用占去大半，fp32 加载会偶发
        # "OSError 1455 页面文件太小"；fp16 后稳定。CPU 上仍用 fp32（fp16 无加速反而不稳）。
        kwargs = {}
        if device.startswith("cuda"):
            kwargs["model_kwargs"] = {"torch_dtype": "float16"}
        model = SentenceTransformer(path, device=device,
                                    local_files_only=True, **kwargs)
        model.max_seq_length = config.EMBEDDING_MAX_LEN
        _device_used = device
        print(f"[embedder] 已加载 bge-m3（{device}，{time.time() - t0:.1f}s）", flush=True)

        _model = model
        return _model


def embed_texts(texts: Sequence[str], batch_size: int | None = None,
                show_progress: bool = False) -> list[list[float]]:
    """把文本批量编码为归一化向量（COSINE 距离下归一化可提升数值稳定性）。"""
    if not texts:
        return []
    model = get_model()
    vecs = model.encode(
        list(texts),
        batch_size=batch_size or config.EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vecs]


def embed_query(query: str) -> list[float]:
    """编码单条查询。"""
    return embed_texts([query])[0]


def get_dim() -> int:
    """返回实际向量维度（优先用模型自报，回退配置值）。"""
    try:
        return int(get_model().get_sentence_embedding_dimension())
    except Exception:
        return config.EMBEDDING_DIM


def info() -> dict:
    """返回当前 embedding 配置信息，供界面/文档展示。"""
    return {
        "model_path": config.EMBEDDING_MODEL_PATH,
        "dim": config.EMBEDDING_DIM,
        "device": _device_used or _pick_device(),
        "loaded": _model is not None,
        "local_files_only": True,
    }
