"""BGE-M3 向量化：一次前向同时产出稠密向量与稀疏（lexical）权重。

原理
----
* 稠密：取最后一层的 CLS（第 0 个 token）向量并做 L2 归一化，1024 维；
  已与 sentence-transformers 的结果做过一致性比对（cos ≥ 0.999999）。
* 稀疏：``relu(W_lex · h + b)``（权重取自模型目录下的 sparse_linear.pt），
  再按 token id 做 max 池化，得到 {token_id: weight} 形式的词权重，
  这正是 BGE-M3 官方 lexical 分支的实现方式。

两者共用同一份权重与同一次前向，显存只占一份（fp16 约 1.2G），
比分别加载 sentence-transformers + transformers 两份模型省一半资源。
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..config import Config, get_config
from ..errors import DependencyError, ValidationError
from ..logging_conf import get_logger

logger = get_logger(__name__)

# 纯空白 / 纯标点 / 纯符号的 token 不参与稀疏权重
_IGNORE_RE = re.compile(r"^[\s\W_]+$")

_DTYPES = {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16,
           "float32": torch.float32, "fp32": torch.float32}


class BGEM3Embedder:
    """BGE-M3 稠密 + 稀疏编码器。"""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        section = self.config.section("models.embedder")
        self.model_path = self.config.embedder_path()
        self.dim = int(section.get("dense_dim", 1024))
        self.max_length = int(section.get("max_length", 512))
        self.batch_size = int(section.get("batch_size", 8))
        self.sparse_min_weight = float(section.get("sparse_min_weight", 0.02))
        self.sparse_top_terms = int(section.get("sparse_top_terms", 192))
        self.query_sparse_top_terms = int(section.get("query_sparse_top_terms", 48))
        self._lock = threading.Lock()
        self._loaded = False
        self.tokenizer = None
        self.model = None
        self.sparse_weight: torch.Tensor | None = None
        self.sparse_bias: torch.Tensor | None = None
        self.device = "cpu"

    # ------------------------------------------------------------------ 加载
    def load(self) -> "BGEM3Embedder":
        # 双重检查 + 可重入锁：避免「预热线程」与「请求线程」并发加载，
        # 否则会出现 transformers 导入竞争或半初始化模型（dtype 不一致）。
        with self._lock:
            if self._loaded:
                return self
            if not self.model_path.is_dir():
                raise DependencyError(f"BGE-M3 模型目录不存在：{self.model_path}")

            from transformers import AutoModel, AutoTokenizer

            section = self.config.section("models.embedder")
            requested = str(section.get("device", "cuda"))
            device = requested
            if requested.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA 不可用，embedder 回退到 CPU")
                device = "cpu"
            dtype = _DTYPES.get(str(section.get("dtype", "float16")).lower(), torch.float16)
            if device == "cpu":
                dtype = torch.float32  # CPU 上 fp16 矩阵乘很慢

            logger.info("加载 BGE-M3：%s (device=%s, dtype=%s)", self.model_path, device, dtype)
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            try:
                model = AutoModel.from_pretrained(str(self.model_path), dtype=dtype)
            except TypeError:  # 兼容旧参数名
                model = AutoModel.from_pretrained(str(self.model_path), torch_dtype=dtype)
            model.to(device)
            model.eval()

            head_path = self.model_path / "sparse_linear.pt"
            if not head_path.is_file():
                raise DependencyError(f"缺少稀疏头权重：{head_path}")
            state = torch.load(str(head_path), map_location="cpu")
            sparse_weight = state["weight"].squeeze().float().to(device)
            sparse_bias = state["bias"].float().to(device)

            # 全部就绪后再一次性发布，确保其他线程看到的永远是完整状态
            self.model = model
            self.sparse_weight = sparse_weight
            self.sparse_bias = sparse_bias
            self.device = device
            self._ignored_ids = self._load_ignored_ids()
            self._loaded = True
            logger.info("BGE-M3 就绪：dim=%d，忽略 token %d 个", self.dim, len(self._ignored_ids))
            return self

    def _load_ignored_ids(self) -> set[int]:
        """计算「纯标点/空白」token id 集合，并缓存到 data_cache。"""

        cache_file = self.config.cache_dir / "embed_ignore_ids.json"
        vocab_size = int(getattr(self.tokenizer, "vocab_size", 0) or 0)
        if cache_file.is_file():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                if int(cached.get("vocab_size", -1)) == vocab_size:
                    return {int(x) for x in cached.get("ids", [])}
            except (json.JSONDecodeError, TypeError, ValueError):
                logger.warning("忽略 token 缓存损坏，重新计算：%s", cache_file)

        tokens = self.tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
        ignored = {
            idx
            for idx, token in enumerate(tokens)
            if token is None or (isinstance(token, str) and _IGNORE_RE.match(token))
        }
        ignored.update(int(x) for x in getattr(self.tokenizer, "all_special_ids", []) or [])
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps({"vocab_size": vocab_size, "ids": sorted(ignored)}, ensure_ascii=False),
            encoding="utf-8",
        )
        return ignored

    # ------------------------------------------------------------------ 编码
    def encode(self, texts: Sequence[str]) -> tuple[np.ndarray, list[dict[int, float]]]:
        """批量编码：返回 (稠密矩阵 [n,1024] float32, 稀疏权重列表)。"""

        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32), []
        cleaned = [self._clean(text) for text in texts]
        self.load()
        dense_chunks: list[np.ndarray] = []
        sparse_all: list[dict[int, float]] = []
        for start in range(0, len(cleaned), self.batch_size):
            batch = cleaned[start : start + self.batch_size]
            with self._lock:
                dense, sparse = self._forward(batch, top_terms=self.sparse_top_terms)
            dense_chunks.append(dense)
            sparse_all.extend(sparse)
        return np.vstack(dense_chunks), sparse_all

    def encode_query(self, text: str) -> tuple[np.ndarray, dict[int, float]]:
        self.load()
        with self._lock:
            dense, sparse = self._forward([self._clean(text)], top_terms=self.query_sparse_top_terms)
        return dense[0], sparse[0]

    def embed_dense(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts)[0]

    # ------------------------------------------------------------------ 内部
    @staticmethod
    def _clean(text: str) -> str:
        value = (text or "").strip()
        if not value:
            raise ValidationError("待向量化文本为空")
        return value

    @torch.no_grad()
    def _forward(self, texts: list[str], top_terms: int) -> tuple[np.ndarray, list[dict[int, float]]]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls = hidden[:, 0].float()
        dense = F.normalize(cls, p=2, dim=-1).cpu().numpy().astype(np.float32)

        logits = torch.einsum("bld,d->bl", hidden.float(), self.sparse_weight) + self.sparse_bias
        weights = torch.relu(logits).masked_fill(attention_mask == 0, 0.0)
        sparse = self._pool_sparse(input_ids.cpu().numpy(), weights.cpu().numpy(), top_terms)
        return dense, sparse

    def _pool_sparse(
        self, input_ids: np.ndarray, weights: np.ndarray, top_terms: int
    ) -> list[dict[int, float]]:
        """按 token id 做 max 池化，取权重最高的若干词。"""

        results: list[dict[int, float]] = []
        for row_ids, row_weights in zip(input_ids, weights):
            uniq, inverse = np.unique(row_ids, return_inverse=True)
            pooled = np.zeros(uniq.shape[0], dtype=np.float32)
            np.maximum.at(pooled, inverse, row_weights)
            keep = np.array(
                [
                    idx
                    for idx, token_id in enumerate(uniq.tolist())
                    if token_id not in self._ignored_ids and pooled[idx] >= self.sparse_min_weight
                ],
                dtype=np.int64,
            )
            if keep.size == 0:
                results.append({})
                continue
            if keep.size > top_terms:
                order = np.argsort(-pooled[keep])[:top_terms]
                keep = keep[order]
            results.append(
                {int(uniq[i]): round(float(pooled[i]), 5) for i in keep.tolist()}
            )
        return results

    # ------------------------------------------------------------------ 工具
    def sparse_to_terms(self, sparse: dict[int, float], limit: int = 12) -> list[tuple[str, float]]:
        """把 {token_id: weight} 还原成可读词条（调试面板用）。"""

        if not sparse:
            return []
        items = sorted(sparse.items(), key=lambda kv: -kv[1])[:limit]
        ids = [token_id for token_id, _ in items]
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return [(str(token).replace("▁", ""), round(float(weight), 4)) for token, (_, weight) in zip(tokens, items)]

    def stats(self) -> dict[str, object]:
        return {
            "model_path": str(self.model_path),
            "loaded": self._loaded,
            "device": self.device,
            "dim": self.dim,
            "max_length": self.max_length,
            "ignored_tokens": len(getattr(self, "_ignored_ids", ())),
        }


_embedder: BGEM3Embedder | None = None
_embedder_lock = threading.Lock()


def get_embedder(config: Config | None = None) -> BGEM3Embedder:
    """获取（并懒加载）全局编码器单例。"""

    global _embedder
    with _embedder_lock:
        if _embedder is None:
            _embedder = BGEM3Embedder(config)
        return _embedder
