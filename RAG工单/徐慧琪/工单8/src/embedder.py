# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
# 工单编号：人工智能NLP-RAG-功能测试及评估
# （上游来源：工单06 混合检索；本工单复用于 ccf_competition 语料的功能测试与评估）
"""
工单编号：人工智能NLP-RAG-混合检索任务
（工单 PDF 原文：人工智能NLP-RAG-混合检索任务）

模块：向量化（Embedding）—— **多嵌入模型支持**

工单要求「支持多种嵌入模型（如 bge、m3e 以及其他嵌入模型）」：
  · 本机扫描（docs/01）结果：无 m3e 权重，已有 BGE 系列 3 个（bge-m3 / bge-m3-ft-v1 /
    all-MiniLM-L6-v2）与 Ollama 侧 bge 系列；
  · 故注册表落 **3 个可选嵌入模型**，可按需切换（config.EMBEDDING_MODELS）。
  · 缺 m3e 不阻塞：BGE 与 M3E 属同类中文嵌入模型，功能等价（详见扫描报告"降级说明"）。

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

_models: dict[str, object] = {}          # name -> SentenceTransformer
_devices: dict[str, str] = {}            # name -> 实际设备
_model_lock = threading.RLock()
# 当前活动嵌入模型：建库/检索的默认取值（界面或建库脚本可切换）。
# 用"活动模型"而不是散落的 config 常量：建库时一次切换，
# embedder / vector_store / build_orgchart 等全部调用点自动对齐到同一模型与同一 collection。
_active_model: str = config.EMBEDDING_MODEL_NAME


def set_active_model(name: str) -> None:
    """切换活动嵌入模型（校验存在性）。"""
    global _active_model
    if name not in config.EMBEDDING_MODELS:
        raise KeyError(f"未知嵌入模型 {name!r}；可选：{list(config.EMBEDDING_MODELS)}")
    _active_model = name


def active_embedding_model() -> str:
    """当前活动嵌入模型名。"""
    return _active_model

# 兼容旧调用点（上游代码里的 _model / _device_used）
_model = None
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


def _spec(name: str) -> dict:
    if name not in config.EMBEDDING_MODELS:
        raise KeyError(
            f"未知嵌入模型 {name!r}；可选：{list(config.EMBEDDING_MODELS)}")
    return config.EMBEDDING_MODELS[name]


def get_model(name: str | None = None):
    """懒加载并缓存指定嵌入模型（线程安全）。name 缺省用默认模型。"""
    global _model, _device_used
    name = name or _active_model
    if name in _models:
        return _models[name]
    with _model_lock:
        if name in _models:
            return _models[name]

        path = _spec(name)["path"]
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"本地 embedding 模型不存在: {path}\n"
                f"请运行 python scan_local_models.py 查看本机可用模型，"
                f"或修改 src/config.py 的 EMBEDDING_MODELS[{name!r}]['path']。")

        # 【本机环境规避】sentence_transformers 冷导入时，其深层依赖链
        # （datasets→pandas.compat→pyarrow.lib）在本机会触发原生 access
        # violation（崩点随运行在 pyarrow/sklearn 间漂移，非 Python 异常）。
        # 实测：裸导入 5/5 崩溃；先导入 pyarrow 后 3/3 正常。两包均为本工程
        # 既有依赖，预导入零成本。详见 docs/11 容错说明。
        import pyarrow  # noqa: F401
        from sentence_transformers import SentenceTransformer

        device = _pick_device()
        t0 = time.time()
        # 显存/内存吃紧的机器上按 fp16 加载（bge-m3 约 1.1G 而非 2.2G）。
        # 本机 8G 显存常被桌面应用占去大半，fp32 加载会偶发
        # "OSError 1455 页面文件太小"；fp16 后稳定。CPU 上仍用 fp32。
        kwargs = {}
        if device.startswith("cuda"):
            kwargs["model_kwargs"] = {"torch_dtype": "float16"}
        model = SentenceTransformer(path, device=device,
                                    local_files_only=True, **kwargs)
        model.max_seq_length = config.EMBEDDING_MAX_LEN
        _models[name] = model
        _devices[name] = device
        if name == _active_model:
            _model, _device_used = model, device
        print(f"[embedder] 已加载 {name}（{device}，{time.time() - t0:.1f}s）",
              flush=True)
        return model


def embed_texts(texts: Sequence[str], batch_size: int | None = None,
                show_progress: bool = False,
                model: str | None = None) -> list[list[float]]:
    """把文本批量编码为归一化向量（COSINE 距离下归一化可提升数值稳定性）。"""
    if not texts:
        return []
    m = get_model(model)
    vecs = m.encode(
        list(texts),
        batch_size=batch_size or config.EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vecs]


def embed_query(query: str, model: str | None = None) -> list[float]:
    """编码单条查询。"""
    return embed_texts([query], model=model)[0]


def get_dim(name: str | None = None) -> int:
    """返回实际向量维度（优先用模型自报，回退配置值）。"""
    name = name or _active_model
    try:
        return int(get_model(name).get_sentence_embedding_dimension())
    except Exception:
        return _spec(name)["dim"]


def is_loaded(name: str | None = None) -> bool:
    return (name or _active_model) in _models


def unload(name: str | None = None) -> None:
    """卸载模型释放显存（长跑/切换模型时用）。"""
    global _model, _device_used
    target = name or _active_model
    with _model_lock:
        _models.pop(target, None)
        _devices.pop(target, None)
        if target == _active_model:
            _model = _device_used = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def info(name: str | None = None) -> dict:
    """返回嵌入模型配置信息，供界面/文档展示。"""
    target = name or _active_model
    spec = _spec(target)
    return {
        "name": target,
        "model_path": spec["path"],
        "dim": spec["dim"],
        "desc": spec["desc"],
        "device": _devices.get(target) or _pick_device(),
        "loaded": target in _models,
        "local_files_only": True,
        "available_models": list(config.EMBEDDING_MODELS),
    }
