# -*- coding: utf-8 -*-
"""证据脚本：判定工单1 基线向量索引由哪个嵌入后端生成（semantic model / hashing fallback）。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：优化 / 基线采集（前置探查）
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

W1 = Path(r"E:\gao6gongdan\工单1")

chunks = [json.loads(l) for l in (W1 / r"data\processed\chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
metas = [json.loads(l) for l in (W1 / r"data\index\vectors_meta.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
vec = np.load(W1 / r"data\index\vectors.npy")
vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)

# 1) meta 与 chunks 是否同序对齐
same = sum(1 for m, c in zip(metas, chunks) if m["chunk_id"] == c["chunk_id"])
page_same = sum(1 for m, c in zip(metas, chunks) if int(m["page"]) == int(c["page"]))
print(f"meta 与 chunks 同序一致: chunk_id {same}/{len(chunks)}, page {page_same}/{len(chunks)}")

# 2) 语义模型对比
idx = [0, 100, 1500, 2500, 2966]
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(str(W1 / "models" / "bge-small-zh-v1.5"), device="cpu")
print("model dim:", model.get_sentence_embedding_dimension())
for tag, prefix in (("raw", ""), ("instruct", "为这个句子生成表示以用于检索相关文章：")):
    texts = [prefix + chunks[i]["content"] for i in idx]
    emb = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    cos = [float(np.dot(emb[k], vec[i])) for k, i in enumerate(idx)]
    print(f"  bge-small-zh-v1.5 [{tag}] cos(vs vectors.npy): {[round(c, 4) for c in cos]}")

# 3) 哈希兜底后端对比（零依赖复刻，仅用于排除）
import hashlib

try:
    import jieba

    def tokenize(text):
        return [t.strip() for t in jieba.lcut(text) if t.strip()]
except Exception as exc:  # pragma: no cover
    print("jieba 不可用:", exc)

    def tokenize(text):
        return re.findall(r"[\w\u4e00-\u9fff]+", text)


def hashing_embed(text, dim=512):
    v = np.zeros(dim, dtype=np.float32)
    for token in tokenize(text) or [text]:
        d = hashlib.md5(token.encode("utf-8")).digest()
        i = int.from_bytes(d[:4], "little") % dim
        v[i] += 1.0 if d[4] % 2 == 0 else -1.0
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


for i in idx[:3]:
    h = hashing_embed(chunks[i]["content"])
    print(f"  hashing-fallback cos(chunk{i}): {float(np.dot(h, vec[i])):.4f}")
