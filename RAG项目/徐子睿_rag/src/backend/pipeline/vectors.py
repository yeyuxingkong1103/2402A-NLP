# -*- coding: utf-8 -*-
"""pipeline/vectors.py —— 向量化与写入 Milvus。

在链路中的位置：
    backend/pipeline 构建管线的第四步：把 chunk 文本变成向量并连同出处一起入库。

三个函数：
    embed                    调 Ollama 批量向量化
    delete_document_vectors  入库前按来源删旧向量（实现同名文档"替换"而非"堆积"）
    store_chunks             写入 Milvus，带 page/section/source 等出处字段

与 vector_store 包的关系：
    本文件只负责"准备数据"，真正的 Milvus 读写由 backend/vector_store 包完成。
    下面的双导入兜底与拆分前一致：作为 backend.pipeline 包导入时用 ..vector_store，
    作为顶层包导入时（backend/ 在 sys.path 上）退回 vector_store。
"""
from __future__ import annotations

import time
import uuid

import requests

from .config import EMBED_MODEL, OLLAMA_BASE

try:
    from ..vector_store import COLLECTION, delete_vectors, equal_filter, upsert_vectors
except ImportError:
    from vector_store import COLLECTION, delete_vectors, equal_filter, upsert_vectors

# ------------------------------ 向量化和入库


def embed(texts: list[str]) -> list[list[float]]:
    """调 Ollama 把文本批量转成向量。

    参数：
        texts: 待向量化的文本列表（这里传的是一个文档的全部 chunk）
    返回：
        与输入等长的向量列表，每个向量 1024 维（bge-m3）。

    为什么必须向量化：
        文本无法直接做相似度计算。变成向量后，"回充设备"和"气体重新充入装置"
        这种字面不同但语义相同的表达才能在向量空间里靠近，被语义检索召回到。
    """
    response = requests.post(
        f"{OLLAMA_BASE}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=600,  # 整个文档一次性批量编码，首次加载模型较慢，给足 10 分钟
    )
    response.raise_for_status()
    return response.json()["embeddings"]

def delete_document_vectors(filename: str) -> None:
    """按来源文件名删除该文档已有的全部向量。

    参数：
        filename: 原始 PDF 文件名（与入库时的 source 字段一致）

    为什么入库前必须调用：
        重复上传同一份 PDF 时，如果只追加不删除，同一段文字会在库里存在多份，
        检索结果被自己的副本挤满（重复召回），答案质量反而下降。
        先删后写 = 同名文档的语义是"替换"而非"堆积"。
    """
    delete_vectors(COLLECTION, equal_filter("source", filename))

def store_chunks(filename: str, chunks: list[dict], vectors: list[list[float]]) -> int:
    """把 chunk 文本、向量与出处信息一起写入 Milvus。

    参数：
        filename: 原始 PDF 文件名，写入 source 字段用于溯源和按文档删除
        chunks:   chunk_pages 的输出
        vectors:  与 chunks 一一对应的向量（embed 的输出）
    返回：
        Milvus 确认写入的记录数。

    记录结构说明：
        - vector:  用于语义检索
        - source / doc_id: 来源文件名，用于按文档过滤和删除
        - page:    页码，答案引用里的"第 N 页"就靠它
        - section: 章节名，让引用可读（"来自 3.2 设备要求"）
        - text:    chunk 原文，最终拼进提示词给模型看
        - summary: 前 160 字摘要，供知识库列表页展示，避免列表接口拖回整段长文本
        - chunk_id: 分块编号，便于定位
        - created_at / updated_at: 构建时间，便于排查"这份 PDF 是什么时候入库的"

    主键用 uuid5 而不是随机 uuid4：
        同一个 (文件名, chunk_id) 永远算出同一个主键，重复构建时是覆盖同一行，
        天然幂等 —— 这是"重复上传即替换"在数据层面的第二道保险。
    """
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    records = [
        {
            "id": uuid.uuid5(uuid.NAMESPACE_DNS, f"{filename}:{chunk['chunk_id']}"),
            "vector": vector,
            "payload": {
                "source": filename,
                "doc_id": filename,
                "page": chunk["page"],
                "section": chunk["section"],
                "text": chunk["text"],
                "summary": chunk["text"][:160],
                "chunk_id": chunk["chunk_id"],
                "created_at": now,
                "updated_at": now,
            },
        }
        for chunk, vector in zip(chunks, vectors)
    ]
    return upsert_vectors(COLLECTION, records)
