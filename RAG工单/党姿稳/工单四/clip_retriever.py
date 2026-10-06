# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
CLIP 多模态图像检索模块：
  使用 sentence-transformers 的 CLIP 模型，把"页面图像"与"文本查询"映射到同一向量空间，
  通过余弦相似度实现"以文搜图"，用于定位与问题语义最相关的图表页。
"""
import os
import numpy as np
from config import CLIP_MODEL

_model = None
_img_cache = {}


def get_clip():
    """CLIP 模型单例"""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[CLIP] 正在加载多模态模型: {CLIP_MODEL}")
        _model = SentenceTransformer(CLIP_MODEL)
    return _model


def embed_images(paths):
    """图像向量化，返回归一化向量矩阵"""
    from PIL import Image
    model = get_clip()
    imgs = [Image.open(p).convert("RGB") for p in paths]
    return model.encode(imgs, normalize_embeddings=True, convert_to_numpy=True)


def embed_text(texts):
    """文本向量化（CLIP 文本塔），返回归一化向量矩阵"""
    model = get_clip()
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


class ClipImageIndex:
    """图像索引：给定一批(图像路径, 块)建立 CLIP 向量，支持以文搜图"""

    def __init__(self, image_chunks):
        self.chunks = [c for c in image_chunks if c.get("image") and os.path.exists(c["image"])]
        print(f"[CLIP] 对 {len(self.chunks)} 张图像建立向量索引...")
        self.vectors = embed_images([c["image"] for c in self.chunks]) if self.chunks else None

    def search(self, query, top_k=3):
        """以文本查询检索最相关的图像块，返回 [(chunk, score), ...]"""
        if not self.chunks:
            return []
        qv = embed_text([query])[0]
        scores = self.vectors @ qv
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in idx]


if __name__ == "__main__":
    from image_parser import load_image_chunks
    idx = ClipImageIndex(load_image_chunks())
    for c, s in idx.search("组织结构图 销售部", 3):
        print(round(s, 4), c["doc"], c["page"])
