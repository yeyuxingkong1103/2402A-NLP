# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-基于PDF文档的问答系统
src/embedding.py — BAAI/bge-m3 嵌入模型封装
优先本地 /home/dabaie/models/bge-m3
"""
import os
from typing import List, Optional
import numpy as np
from loguru import logger

_LOCAL_BGE_M3 = "/home/dabaie/models/bge-m3"
if os.path.isdir(_LOCAL_BGE_M3) and os.path.isfile(os.path.join(_LOCAL_BGE_M3, "config.json")):
    DEFAULT_MODEL_NAME = _LOCAL_BGE_M3
    logger.info(f"使用本地嵌入模型: {DEFAULT_MODEL_NAME}")
else:
    DEFAULT_MODEL_NAME = "BAAI/bge-m3"
    logger.info(f"回退 HuggingFace ID: {DEFAULT_MODEL_NAME}")

DEFAULT_RERANKER_PATH = "/home/dabaie/models/bge-reranker-v2-m3"
DEFAULT_EMBEDDING_DIM = 1024

class Embedder:
    _instance = None
    def __init__(self, model_name=None, device=None, normalize_embeddings=True):
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.normalize_embeddings = normalize_embeddings
        self._model = None; self._dim = 0; self._device = device
    def _load(self):
        if self._model is not None: return
        import torch
        from sentence_transformers import SentenceTransformer
        if self._device is None: self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"加载嵌入模型: {self.model_name}  device={self._device}")
        self._model = SentenceTransformer(self.model_name, device=self._device)
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
    def encode(self, texts, batch_size=64, show_progress_bar=True):
        if self._model is None: self._load()
        if not texts: return np.zeros((0, self.dim), dtype=np.float32)
        valid_idx = [i for i, t in enumerate(texts) if t and t.strip()]
        if len(valid_idx) != len(texts): logger.warning(f"跳过 {len(texts)-len(valid_idx)} 条空文本")
        valid_texts = [texts[i] for i in valid_idx]
        vecs = self._model.encode(valid_texts, batch_size=batch_size,
                                 normalize_embeddings=self.normalize_embeddings,
                                 show_progress_bar=show_progress_bar, convert_to_numpy=True).astype(np.float32)
        if len(valid_idx) != len(texts):
            full = np.zeros((len(texts), self.dim), dtype=np.float32)
            full[valid_idx] = vecs; vecs = full
        return vecs

def get_embedder(model_name=None, device=None):
    if Embedder._instance is None or (model_name and Embedder._instance.model_name != model_name):
        Embedder._instance = Embedder(model_name=model_name, device=device)
    return Embedder._instance

def encode_chunks(chunks_json_path: str, out_path=None, model_name=None):
    import json
    with open(chunks_json_path, "r", encoding="utf-8") as f: data = json.load(f)
    texts = [c["text"] for c in data["chunks"]]
    logger.info(f"开始向量化: {len(texts)} chunks")
    embedder = get_embedder(model_name=model_name)
    vectors = embedder.encode(texts, batch_size=64, show_progress_bar=True)
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
    p = argparse.ArgumentParser(description="嵌入模型 CLI（工单：人工智能NLP-RAG-基于PDF文档的问答系统）")
    p.add_argument("--chunks", default="data/chunks/招股说明书1_chunks.json")
    p.add_argument("--out", default=None); p.add_argument("--model", default=None)
    args = p.parse_args()
    encode_chunks(args.chunks, args.out, args.model)
