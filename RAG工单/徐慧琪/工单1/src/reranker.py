# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块：精排（Rerank）

选型：BAAI/bge-reranker-v2-m3 —— 本地路径 D:\\model\\reranker
  理由：工单把 "bge-reranker 系列" 列为可选优化项，本机已有该权重；
        属 Cross-Encoder，把 (问题, 候选块) 拼在一起做相关性打分，
        比双塔向量的余弦相似度判别力强得多，能显著把"含答案的那一块"
        顶到前面，从而减少送入 LLM 的上下文长度（同时利于 3 秒响应目标）。

流程：向量+BM25 混合召回 RERANKER_TOP_K_IN 条 → Cross-Encoder 打分
      → 取 RERANKER_TOP_K_OUT 条送 LLM。

硬约束遵守：from_pretrained(..., local_files_only=True)，设备 cuda→cpu 自动回退；
            任何异常都不会让主链路崩溃（退化为不精排，直接截断）。
"""

from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

import os
import threading
import time
from typing import Sequence

from src import config

_tokenizer = None
_model = None
_device = None
_lock = threading.Lock()
# 加载失败后的冷却截止时间（不是永久熔断）。
# 【踩坑记录】本机 8G 显存/16G 内存，且桌面应用常占满显存，模型加载会偶发
# OSError 1455（页面文件/提交内存不足）或直接段错误。若把一次失败记成
# "永久不可用"，那么一次瞬时内存不足就会让整个进程此后都不再精排 ——
# 可用性优先：失败后冷却一段时间再试。
_load_failed_until = 0.0
_LOAD_COOLDOWN_S = 120


def is_available() -> bool:
    """本地权重目录存在，且不在失败冷却期内。"""
    if not os.path.isdir(config.RERANKER_MODEL_PATH):
        return False
    return time.time() >= _load_failed_until


def _pick_device() -> str:
    try:
        import torch
        if config.EMBEDDING_DEVICE.startswith("cuda") and torch.cuda.is_available():
            return config.EMBEDDING_DEVICE
    except Exception:
        pass
    return "cpu"


def _load_once():
    """真正执行一次加载（供 get_model 调用，便于做失败重试）。"""
    path = config.RERANKER_MODEL_PATH
    if not os.path.isdir(path):
        raise FileNotFoundError(f"本地精排模型不存在: {path}")

    import torch
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)

    device = _pick_device()
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

    kwargs = {"local_files_only": True}
    if device.startswith("cuda"):
        # 关键：按 fp16 直接加载，并用 device_map 直送显存。
        # 本机实测（16G 内存 / 8G 显存）：
        #   fp32 先落内存再 .to(cuda) —— 峰值 = 内存副本 + 显存副本，
        #   在桌面应用占满显存的机器上会触发 "OSError 1455 页面文件太小"，
        #   甚至段错误（5 次里失败 3 次）。
        #   fp16 + device_map 直载后峰值减半，连续多次加载均稳定通过。
        kwargs["torch_dtype"] = torch.float16
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                path, device_map=device, **kwargs)
        except Exception:
            # device_map 需要 accelerate，缺失时退回"先加载后搬运"
            model = AutoModelForSequenceClassification.from_pretrained(path, **kwargs)
            model.to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(path, **kwargs)
        model.to(device)

    model.eval()
    print(f"[reranker] 已加载 bge-reranker-v2-m3（{device}，fp16="
          f"{device.startswith('cuda')}，{time.time() - t0:.1f}s）", flush=True)
    return tokenizer, model, device


def get_model():
    """懒加载精排模型（线程安全，失败后冷却重试）。返回 (tokenizer, model, device)。"""
    global _tokenizer, _model, _device, _load_failed_until
    if _model is not None:
        return _tokenizer, _model, _device
    with _lock:
        if _model is not None:
            return _tokenizer, _model, _device
        if time.time() < _load_failed_until:
            raise RuntimeError("精排模型处于失败冷却期，暂不重试")

        try:
            tokenizer, model, device = _load_once()
        except Exception:
            _load_failed_until = time.time() + _LOAD_COOLDOWN_S
            raise
        _tokenizer, _model, _device = tokenizer, model, device
        _load_failed_until = 0.0
        return _tokenizer, _model, _device


def _sort_with_tiebreak(candidates: Sequence[dict]) -> list[dict]:
    """按精排分降序排；分数"打平"时回退到融合名次（RRF）裁决。

    【为什么需要】实测第 260 题（问"军用领域收入分别是多少"）：
      · RRF 融合把"含金额的片段"(p129) 排在"只含占比的片段"(p343) 之前 —— 正确；
      · Cross-Encoder 给的分数是 0.9956 vs 0.9972，**差 0.0016**，属于噪声量级，
        却把顺序翻了过来 —— 结果模型读了排第一的"占比"片段，
        答成了"占比分别是 82.10%…"，答非所问。
    精排分几乎相同时，谁排前面本就没有统计意义；而 RRF 名次聚合了稠密+稀疏
    两路独立信号，此时更可信。故以 epsilon 分桶，桶内用 RRF 分裁决。
    """
    eps = getattr(config, "RERANKER_TIE_EPSILON", 0.0) or 0.0
    if eps <= 0:
        return sorted(candidates, key=lambda c: c.get("rerank_score", 0.0), reverse=True)

    def key(c: dict):
        bucket = round(c.get("rerank_score", 0.0) / eps)
        return (-bucket, -c.get("rrf_score", 0.0), -c.get("vector_score", 0.0))

    return sorted(candidates, key=key)


def rerank(query: str, candidates: Sequence[dict],
           top_k: int | None = None) -> list[dict]:
    """对候选块精排，返回 top_k 条（附 rerank_score）。

    任何失败都不抛出：直接把候选按原顺序截断返回，保证问答不中断。
    """
    top_k = top_k or config.RERANKER_TOP_K_OUT
    cands = list(candidates)
    if not cands:
        return []
    if not config.RERANKER_ENABLED or not is_available():
        return cands[:top_k]

    try:
        import torch

        tokenizer, model, device = get_model()
        # 精排用"干净正文"：去掉【标题】前缀，避免前缀词干扰相关性判断
        pairs = [(query, c.get("raw_text") or c.get("text", "")) for c in cands]

        scores: list[float] = []
        bs = config.RERANKER_BATCH_SIZE
        with torch.no_grad():
            for i in range(0, len(pairs), bs):
                batch = pairs[i:i + bs]
                enc = tokenizer([p[0] for p in batch], [p[1] for p in batch],
                                padding=True, truncation=True,
                                max_length=config.RERANKER_MAX_LEN,
                                return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(**enc).logits.view(-1).float()
                scores.extend(torch.sigmoid(logits).cpu().tolist())

        for c, s in zip(cands, scores):
            c["rerank_score"] = float(s)
        ranked = _sort_with_tiebreak(cands)
        return ranked[:top_k]
    except Exception as exc:  # pragma: no cover - 兜底路径
        # 容错优先：精排挂了也必须能出答案，退化为"按召回顺序截断"。
        # 冷却期由 get_model 统一维护，这里不重复设置。
        print(f"[reranker] 精排不可用，本次退化为按召回顺序截断：{exc}", flush=True)
        return cands[:top_k]


def info() -> dict:
    cooling = max(0.0, _load_failed_until - time.time())
    return {
        "enabled": config.RERANKER_ENABLED and is_available(),
        "path": config.RERANKER_MODEL_PATH,
        "loaded": _model is not None,
        "device": _device or _pick_device(),
        "top_k_in": config.RERANKER_TOP_K_IN,
        "top_k_out": config.RERANKER_TOP_K_OUT,
        "max_len": config.RERANKER_MAX_LEN,
        "cooldown_s": round(cooling, 1),
    }
