# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""CLIP 跨模态检索：以文搜图。"""
from __future__ import annotations

import logging

from rag04.config import Settings
from rag04.schema import Hit

logger = logging.getLogger("rag04.image_index")


def clip_search(question: str, s: Settings, store, k: int = 5) -> list[Hit]:
    """问题文本经 CLIP 文本编码器 → 与 image_chunks 的图像向量做相似度检索。"""
    from rag04.ingest.store import COLL_IMAGE
    from rag04.ingest.vlparser import clip_encode_text

    tv = clip_encode_text(question, s)
    hits = store.search(COLL_IMAGE, tv, k=k)
    for h in hits:
        h.channel = "clip"
    return hits
