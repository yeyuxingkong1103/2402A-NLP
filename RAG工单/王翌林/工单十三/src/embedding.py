# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化
src/embedding.py — BAAI/bge-m3 嵌入模型封装
优先本地 /home/dabaie/models/bge-m3

显存优化（工单三）：
  - RAG_EMBED_DEVICE=cpu  → 强制 CPU（避免 GPU OOM，默认 auto）
  - RAG_EMBED_BATCH_SIZE=8 → 默认 batch_size（原 64 易爆显存，降为 8）
  - RAG_EMBED_MAX_SEQ=512  → 限制最大序列长度（原 8192 极耗显存，表格文本短，512 够用）
"""
import os
import threading
from collections import OrderedDict
from typing import List, Optional
import numpy as np
from loguru import logger

# 工单十三：查询嵌入 LRU 缓存开关（优化项：同一查询文本跨通道/跨请求复用向量，
# 省去 bge-m3 重复编码；默认关闭以保留原始行为，优化后基准测试置 1 开启）
EMBED_CACHE_ON = os.getenv("RAG_V13_EMBED_CACHE", "0") == "1"
_EMBED_CACHE_MAX = int(os.getenv("RAG_V13_EMBED_CACHE_SIZE", "512"))
_embed_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
_embed_cache_lock = threading.Lock()

_LOCAL_BGE_M3 = "/home/dabaie/models/bge-m3"
if os.path.isdir(_LOCAL_BGE_M3) and os.path.isfile(os.path.join(_LOCAL_BGE_M3, "config.json")):
    DEFAULT_MODEL_NAME = _LOCAL_BGE_M3
    logger.info(f"使用本地嵌入模型: {DEFAULT_MODEL_NAME}")
else:
    DEFAULT_MODEL_NAME = "BAAI/bge-m3"
    logger.info(f"回退 HuggingFace ID: {DEFAULT_MODEL_NAME}")

DEFAULT_RERANKER_PATH = "/home/dabaie/models/bge-reranker-v2-m3"
DEFAULT_EMBEDDING_DIM = 1024

# 工单三：显存优化环境变量
DEFAULT_BATCH_SIZE = int(os.getenv("RAG_EMBED_BATCH_SIZE", "8"))
DEFAULT_MAX_SEQ_LENGTH = int(os.getenv("RAG_EMBED_MAX_SEQ", "512"))


class Embedder:
    _instance = None
    # 工单四（人工智能NLP-RAG-图像内容解析及检索优化）：并行检索（表格/文本
    # 线程池 + 图像通道）首次并发调用时，无锁单例会各建一个 Embedder 并重复
    # 把 bge-m3 载入 GPU（实测一次请求加载 3 份，8GB 显存超卖致推理卡数分钟）。
    _instance_lock = threading.Lock()
    _load_lock = threading.Lock()
    def __init__(self, model_name=None, device=None, normalize_embeddings=True):
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.normalize_embeddings = normalize_embeddings
        self._model = None; self._dim = 0; self._device = device
    def _load(self):
        if self._model is not None: return
        with Embedder._load_lock:
            if self._model is not None: return
            # 工单四：竞态兜底——若别的实例已先加载完，直接共享其权重
            inst = Embedder._instance
            if inst is not None and inst is not self and inst._model is not None:
                self._model, self._dim, self._device = inst._model, inst._dim, inst._device
                return
            import torch
            from sentence_transformers import SentenceTransformer
            # 工单三：device 优先级：显式传入 > env RAG_EMBED_DEVICE > auto
            if self._device is None:
                env_device = os.getenv("RAG_EMBED_DEVICE", "").strip().lower()
                if env_device in ("cpu", "cuda", "cuda:0"):
                    self._device = env_device
                else:
                    self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"加载嵌入模型: {self.model_name}  device={self._device}")
            self._model = SentenceTransformer(self.model_name, device=self._device)
            # 工单三：限制 max_seq_length，避免长序列爆显存（表格文本通常 < 512 tokens）
            try:
                self._model.max_seq_length = DEFAULT_MAX_SEQ_LENGTH
                logger.info(f"max_seq_length={self._model.max_seq_length}")
            except Exception as e:
                logger.warning(f"设置 max_seq_length 失败: {e}")
            dummy = self._model.encode(["测试"], normalize_embeddings=self.normalize_embeddings)
            self._dim = int(dummy.shape[1])
            logger.info(f"嵌入模型就绪: dim={self._dim}")
    @property
    def dim(self):
        if self._model is None: self._load()
        return self._dim
    @property
    def device(self):
        if self._model is None: self._load()
        return self._device
    def encode(self, texts, batch_size=None, show_progress_bar=True):
        if self._model is None: self._load()
        bs = batch_size if batch_size is not None else DEFAULT_BATCH_SIZE
        if not texts: return np.zeros((0, self.dim), dtype=np.float32)
        # 工单十三：单条查询编码走 LRU 缓存（文本通道/表格通道同一查询只编码一次）
        if EMBED_CACHE_ON and len(texts) == 1:
            key = str(texts[0])
            with _embed_cache_lock:
                if key in _embed_cache:
                    _embed_cache.move_to_end(key)
                    return _embed_cache[key].reshape(1, -1).astype(np.float32).copy()
        valid_idx = [i for i, t in enumerate(texts) if t and t.strip()]
        if len(valid_idx) != len(texts): logger.warning(f"跳过 {len(texts)-len(valid_idx)} 条空文本")
        valid_texts = [texts[i] for i in valid_idx]
        vecs = self._model.encode(valid_texts, batch_size=bs,
                                 normalize_embeddings=self.normalize_embeddings,
                                 show_progress_bar=show_progress_bar, convert_to_numpy=True).astype(np.float32)
        if len(valid_idx) != len(texts):
            full = np.zeros((len(texts), self.dim), dtype=np.float32)
            full[valid_idx] = vecs; vecs = full
        # 工单十三：写入查询缓存
        if EMBED_CACHE_ON and len(texts) == 1:
            key = str(texts[0])
            with _embed_cache_lock:
                _embed_cache[key] = vecs[0]
                _embed_cache.move_to_end(key)
                while len(_embed_cache) > _EMBED_CACHE_MAX:
                    _embed_cache.popitem(last=False)
        return vecs

def get_embedder(model_name=None, device=None):
    # 工单四：加锁双重检查，防止并行检索线程重复加载 bge-m3（显存超卖）
    if Embedder._instance is None or (model_name and Embedder._instance.model_name != model_name):
        with Embedder._instance_lock:
            if Embedder._instance is None or (model_name and Embedder._instance.model_name != model_name):
                Embedder._instance = Embedder(model_name=model_name, device=device)
    return Embedder._instance

def encode_chunks(chunks_json_path: str, out_path=None, model_name=None):
    import json
    with open(chunks_json_path, "r", encoding="utf-8") as f: data = json.load(f)
    texts = [c["text"] for c in data["chunks"]]
    logger.info(f"开始向量化: {len(texts)} chunks  batch_size={DEFAULT_BATCH_SIZE}")
    embedder = get_embedder(model_name=model_name)
    vectors = embedder.encode(texts, batch_size=DEFAULT_BATCH_SIZE, show_progress_bar=True)
    for i, chunk in enumerate(data["chunks"]): chunk["vector"] = vectors[i].tolist()
    data["embedding_model"] = embedder.model_name
    data["embedding_dim"] = embedder.dim
    data["embedding_normalized"] = embedder.normalize_embeddings
    out = out_path or chunks_json_path
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False)
    logger.info(f"向量化完成: dim={embedder.dim}, 已写入 {out}")
    return out

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="嵌入模型 CLI（工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化）")
    p.add_argument("--chunks", default="data/chunks/招股说明书1_chunks.json")
    p.add_argument("--out", default=None); p.add_argument("--model", default=None)
    args = p.parse_args()
    encode_chunks(args.chunks, args.out, args.model)
