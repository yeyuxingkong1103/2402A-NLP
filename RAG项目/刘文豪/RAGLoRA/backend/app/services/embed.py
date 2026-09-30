# -*- coding: utf-8 -*-
"""BGE-M3 编码器：一次前向同时产出 dense 向量与 learned sparse 词权重。

设计取舍（M0 验证结论）：
  不使用 FlagEmbedding —— 它要求 transformers>=4.44.2，会破坏 rag_env 的版本链。
  这里用 rag_env 已有的 transformers + torch 直接实现，数学上完全等价：

      dense  = mean_pooling(last_hidden_state, attention_mask) → L2 normalize
      sparse = relu(sparse_linear(last_hidden_state)) * attention_mask
               → 按 token_id 取 max 聚合（过滤特殊符）

设备策略（实测吞吐）：
  离线入库 → GPU fp16，约 234 chunks/s（1.4 万条约 1 分钟）
  在线查询 → CPU fp32，约 8 chunks/s；单条 query 仅几十毫秒，把显存全留给 Ollama 生成
"""
import threading
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from ..core import config
from ..core.logging import get_logger

log = get_logger("embed")

_load_lock = threading.Lock()
_encode_lock = threading.Lock()     # 设备切换期间串行化，避免 forward 跑到一半模型被换掉

_tok = None
_model = None
_sparse_linear = None
_special_ids: set[int] = set()
_current_device: str | None = None
_current_dtype = None


def _resolve_dtype(device: str):
    return torch.float16 if device == "cuda" else torch.float32


def _ensure(device: str):
    """确保模型已按指定设备加载；设备变化时重新加载。"""
    global _tok, _model, _sparse_linear, _special_ids, _current_device, _current_dtype

    dtype = _resolve_dtype(device)
    if _model is not None and _current_device == device and _current_dtype == dtype:
        return _tok, _model, _sparse_linear, _special_ids

    with _load_lock:
        if _model is not None and _current_device == device and _current_dtype == dtype:
            return _tok, _model, _sparse_linear, _special_ids

        if _model is not None:
            log.info("编码器切换设备 %s -> %s，卸载原模型", _current_device, device)
            _model = _sparse_linear = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        path = Path(config.EMBED_MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(f"嵌入模型路径不存在: {path}")

        log.info("加载 bge-m3 -> %s (%s)", device, str(dtype).replace("torch.", ""))
        _tok = AutoTokenizer.from_pretrained(str(path))
        _model = AutoModel.from_pretrained(str(path), torch_dtype=dtype).eval().to(device)

        lin = nn.Linear(_model.config.hidden_size, 1)
        lin.load_state_dict(torch.load(path / "sparse_linear.pt",
                                       map_location="cpu", weights_only=False))
        _sparse_linear = lin.to(device).to(dtype).eval()

        # 特殊符（<s>/</s>/<pad>/<unk>）会带上可观权重，必须从稀疏向量里排除
        _special_ids = set(_tok.all_special_ids)
        _current_device, _current_dtype = device, dtype
        log.info("bge-m3 就绪 | device=%s | hidden=%d | 特殊符 %d 个",
                 device, _model.config.hidden_size, len(_special_ids))
        return _tok, _model, _sparse_linear, _special_ids


def encode(texts: list[str], batch_size: int | None = None, device: str = "cpu",
           progress: bool = False) -> tuple[list[list[float]], list[dict]]:
    """批量编码。

    返回 (dense_vectors, sparse_vectors)：
        dense_vectors  -> [[float]*1024, ...]  已 L2 归一化
        sparse_vectors -> [{"indices": [int], "values": [float]}, ...]
    """
    if not texts:
        return [], []

    if batch_size is None:
        batch_size = 16 if device == "cuda" else 8

    all_dense: list[list[float]] = []
    all_sparse: list[dict] = []

    with _encode_lock:
        tok, model, sparse_linear, special_ids = _ensure(device)

        rng = range(0, len(texts), batch_size)
        if progress:
            from tqdm import tqdm
            rng = tqdm(rng, desc=f"编码({device})", unit="batch")

        for start in rng:
            batch = texts[start:start + batch_size]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=config.EMBED_MAX_LEN, return_tensors="pt").to(device)

            with torch.inference_mode():
                hidden = model(**enc).last_hidden_state        # [B, L, H]
                mask = enc["attention_mask"].unsqueeze(-1)     # [B, L, 1]

                # ---- dense: 掩码平均池化 + L2 归一化 ----
                dense = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                dense = torch.nn.functional.normalize(dense, p=2, dim=1)

                # ---- sparse: relu(Linear(h)) * mask ----
                weights = torch.relu(sparse_linear(hidden)).squeeze(-1) * enc["attention_mask"]

            all_dense.extend(dense.float().tolist())

            for row_ids, row_w in zip(enc["input_ids"].tolist(), weights.float().tolist()):
                agg: dict[int, float] = {}
                for tid, val in zip(row_ids, row_w):
                    if val >= config.SPARSE_MIN_WEIGHT and tid not in special_ids:
                        if val > agg.get(tid, 0.0):
                            agg[tid] = val
                all_sparse.append({
                    "indices": list(agg.keys()),
                    "values": [round(v, 6) for v in agg.values()],
                })

    return all_dense, all_sparse


def encode_one(text: str, device: str = "cpu") -> tuple[list[float], dict]:
    """单条编码（在线检索用，默认 CPU）。"""
    dense, sparse = encode([text], device=device)
    return dense[0], sparse[0]


def release_gpu() -> None:
    """释放 GPU 占用（批量入库结束后调用），把显存还给 Ollama。"""
    global _model, _sparse_linear, _current_device, _current_dtype
    with _load_lock:
        if _current_device == "cuda":
            _model = _sparse_linear = None
            _current_device = _current_dtype = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.info("编码器已释放 GPU")


def warmup(device: str = "cpu") -> None:
    encode(["预热"], device=device)


def current_device() -> str | None:
    return _current_device
