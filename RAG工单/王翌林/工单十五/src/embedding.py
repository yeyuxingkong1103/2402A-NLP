# -*- coding: utf-8 -*-
"""
工单十五：bge-m3 嵌入模型封装
显存优化：强制 CPU、batch_size=8、max_seq_length=512
"""
import os
import threading
from typing import List, Optional
import numpy as np
from loguru import logger

_LOCAL_BGE_M3 = "/home/dabaie/models/bge-m3"
if os.path.isdir(_LOCAL_BGE_M3) and os.path.isfile(os.path.join(_LOCAL_BGE_M3, "config.json")):
    DEFAULT_MODEL_NAME = _LOCAL_BGE_M3
else:
    DEFAULT_MODEL_NAME = "BAAI/bge-m3"

DEFAULT_EMBEDDING_DIM = 1024
DEFAULT_BATCH_SIZE = int(os.getenv("RAG_EMBED_BATCH_SIZE", "8"))
DEFAULT_MAX_SEQ_LENGTH = int(os.getenv("RAG_EMBED_MAX_SEQ", "512"))


class Embedder:
    _instance = None
    _instance_lock = threading.Lock()
    _load_lock = threading.Lock()

    def __init__(self, model_name=None, device=None, normalize_embeddings=True):
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.normalize_embeddings = normalize_embeddings
        self._model = None
        self._dim = 0
        self._device = device

    def _load(self):
        if self._model is not None:
            return
        with Embedder._load_lock:
            if self._model is not None:
                return
            inst = Embedder._instance
            if inst is not None and inst is not self and inst._model is not None:
                self._model, self._dim, self._device = inst._model, inst._dim, inst._device
                return
            import torch
            from sentence_transformers import SentenceTransformer
            if self._device is None:
                env_device = os.getenv("RAG_EMBED_DEVICE", "").strip().lower()
                if env_device in ("cpu", "cuda", "cuda:0"):
                    self._device = env_device
                else:
                    self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"加载嵌入模型: {self.model_name}  device={self._device}")
            self._model = SentenceTransformer(self.model_name, device=self._device)
            try:
                self._model.max_seq_length = DEFAULT_MAX_SEQ_LENGTH
            except Exception:
                pass
            dummy = self._model.encode(["测试"], normalize_embeddings=self.normalize_embeddings)
            self._dim = int(dummy.shape[1])
            logger.info(f"嵌入模型就绪: dim={self._dim}")

    @property
    def dim(self):
        if self._model is None:
            self._load()
        return self._dim

    @property
    def device(self):
        if self._model is None:
            self._load()
        return self._device

    def encode(self, texts, batch_size=None, show_progress_bar=False):
        if self._model is None:
            self._load()
        bs = batch_size if batch_size is not None else DEFAULT_BATCH_SIZE
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        valid_idx = [i for i, t in enumerate(texts) if t and t.strip()]
        valid_texts = [texts[i] for i in valid_idx]
        vecs = self._model.encode(valid_texts, batch_size=bs,
                                  normalize_embeddings=self.normalize_embeddings,
                                  show_progress_bar=show_progress_bar, convert_to_numpy=True).astype(np.float32)
        if len(valid_idx) != len(texts):
            full = np.zeros((len(texts), self.dim), dtype=np.float32)
            full[valid_idx] = vecs
            vecs = full
        return vecs


def get_embedder(model_name=None, device=None):
    if Embedder._instance is None or (model_name and Embedder._instance.model_name != model_name):
        with Embedder._instance_lock:
            if Embedder._instance is None or (model_name and Embedder._instance.model_name != model_name):
                Embedder._instance = Embedder(model_name=model_name, device=device)
    return Embedder._instance
