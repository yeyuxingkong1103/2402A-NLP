# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
向量缓存预构建：分别编码【文本/表格块】与【图像块】，再按 文本+图像 的顺序合并为
data/vectors.npy（与 kb.load_all_chunks() 的块顺序严格一致），
避免评估时重复编码，并可跨工单目录复用同一份向量缓存。
"""
import os
import sys
import json
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from config import CHUNKS_FILE, IMAGE_CHUNKS_FILE, DATA_DIR, VECTORS_FILE
from embedder import embed_texts

TEXT_VEC = os.path.join(DATA_DIR, "vectors_text.npy")
IMG_VEC = os.path.join(DATA_DIR, "vectors_image.npy")


def main():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        text_chunks = json.load(f)
    img_chunks = []
    if os.path.exists(IMAGE_CHUNKS_FILE):
        with open(IMAGE_CHUNKS_FILE, "r", encoding="utf-8") as f:
            img_chunks = json.load(f)

    if os.path.exists(TEXT_VEC) and np.load(TEXT_VEC).shape[0] == len(text_chunks):
        vt = np.load(TEXT_VEC)
        print(f"[向量缓存] 复用文本/表格向量 {vt.shape}")
    else:
        print(f"[向量缓存] 编码文本/表格块 {len(text_chunks)} 个...")
        vt = embed_texts([c["text"] for c in text_chunks])
        np.save(TEXT_VEC, vt)

    if img_chunks:
        if os.path.exists(IMG_VEC) and np.load(IMG_VEC).shape[0] == len(img_chunks):
            vi = np.load(IMG_VEC)
            print(f"[向量缓存] 复用图像向量 {vi.shape}")
        else:
            print(f"[向量缓存] 编码图像块 {len(img_chunks)} 个...")
            vi = embed_texts([c["text"] for c in img_chunks])
            np.save(IMG_VEC, vi)
        v = np.vstack([vt, vi])
    else:
        v = vt

    np.save(VECTORS_FILE, v)
    print(f"[向量缓存] 合并完成 {v.shape} -> {VECTORS_FILE}")


if __name__ == "__main__":
    main()
