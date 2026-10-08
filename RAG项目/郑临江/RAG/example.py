# -*- coding: utf-8 -*-
"""rag2 统一离线包调用示例：在线混合检索 + 离线 SQLite 流水线。

运行前：
    1. 在线路：确保 Milvus 已启动（WSL: http://localhost:19530）；
    2. 需安装 sentence-transformers 与 bge-m3 本地权重（D:/modelscope/bge-m3）；
    3. 离线路：无需任何服务器，仅需一个 .sqlite 文件路径即可跑通。
"""

import rag2


def embed(texts):
    """向量化函数：把一段文本变成一串数字（向量），语义相近的文本其向量也相近。

    这是整个 RAG 的「语义引擎」——检索时把问题也变成向量，再在库里找最接近的向量。
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("D:/modelscope/bge-m3", device=rag2.resolve_device())
    # normalize_embeddings=True：向量归一化，使点积=余弦相似度（与库里存储方式保持一致）
    return [v.tolist() for v in model.encode(texts, normalize_embeddings=True)]


# ============ 一、在线：Milvus + BM25 混合检索 + bge 重排 ============
retriever = rag2.HybridRetriever(
    uri="http://localhost:19530", collection="my_kb",
    dim=1024, embed_fn=embed,
    rerank_model=rag2.DEFAULT_RERANK_MODEL, device="auto",
)
retriever.ingest(texts=["……", "……"], metadatas=[{"source": "a.pdf", "page": 1}, {}])
for h in retriever.search("问题", top_k=3, mode="hybrid", rerank=True):
    print("[在线]", h.score, h.text[:50])


# ============ 二、离线：SQLite 本地库，全链路无需服务器 ============
rag = rag2.OfflineRAG(db_path="kb.sqlite", embed_fn=embed)  # 或 embed_model="D:/modelscope/bge-m3"
rag.add_file("D:/桌面D/专高/专高六/项目/RAG_1/data/农业知识.81130062_47.pdf")
for h in rag.search("农业知识相关问题", top_k=3, mode="hybrid"):
    print("[离线]", h.score, h.text[:50])
