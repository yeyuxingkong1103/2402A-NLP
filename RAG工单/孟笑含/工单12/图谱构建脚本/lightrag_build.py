# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-LightRAG优化任务
模块：LightRAG 知识图谱构建
"""

import os
import asyncio
import fitz
import re
import numpy as np
from typing import List
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from lightrag_config import LightRAGConfig

cfg = LightRAGConfig()


async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    """调用 DeepSeek LLM"""
    return await openai_complete_if_cache(
        cfg.LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=cfg.LLM_API_KEY,
        base_url=cfg.LLM_BASE_URL,
        **kwargs,
    )


async def embedding_func(texts: List[str]):
    """本地 bge-base 嵌入"""
    from sentence_transformers import SentenceTransformer

    if not hasattr(embedding_func, "_model"):
        print("  正在加载 bge-base-zh-v1.5...")
        embedding_func._model = SentenceTransformer(
            cfg.EMBEDDING_MODEL,
            model_kwargs={"use_safetensors": True},
        )

    embeddings = embedding_func._model.encode(
        texts, normalize_embeddings=True, show_progress_bar=False,
    )
    return np.array(embeddings, dtype=np.float32)


def extract_chunks(pdf_paths, chunk_size=500, max_per_pdf=250):
    """抽取 PDF chunks"""
    chunks = []
    for pdf_path in pdf_paths:
        doc_name = os.path.basename(pdf_path)
        doc = fitz.open(pdf_path)
        n = 0
        for page_num in range(len(doc)):
            if n >= max_per_pdf:
                break
            text = doc[page_num].get_text()
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) < 200:
                continue
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                if len(chunk) >= 200:
                    chunks.append({"content": chunk, "doc": doc_name, "page": page_num + 1})
                    n += 1
                    if n >= max_per_pdf:
                        break
        doc.close()
        print(f"  {doc_name}: {n} chunks")
    return chunks


async def main():
    print("=" * 60)
    print("LightRAG 图谱构建")
    print("=" * 60)

    print("\n[1/3] 抽取 PDF chunks...")
    pdf_paths = ["./data/招股说明书1.pdf", "./data/招股说明书2.pdf"]
    chunks = extract_chunks(pdf_paths, chunk_size=500, max_per_pdf=250)
    print(f"  ✅ 共 {len(chunks)} chunks")

    print("\n[2/3] 初始化 LightRAG...")
    rag = LightRAG(
        working_dir=cfg.WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=512,
            max_token_size=512,
            func=embedding_func,
        ),
        chunk_token_size=cfg.CHUNK_TOKEN_SIZE,
        chunk_overlap_token_size=cfg.CHUNK_OVERLAP_TOKEN_SIZE,
    )
    await rag.initialize_storages()
    print("  ✅ LightRAG 初始化完成")

    print(f"\n[3/3] 插入 {len(chunks)} chunks（15~30 分钟）...")
    for i, chunk in enumerate(chunks):
        try:
            await rag.ainsert(chunk["content"])
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(chunks)}")
        except Exception as e:
            print(f"  [ERROR] {i}: {e}")
            continue

    print("\n✅ 图谱构建完成")
    print(f"  工作目录：{cfg.WORKING_DIR}")


if __name__ == "__main__":
    cfg = LightRAGConfig()
    if not cfg.LLM_API_KEY:
        print("❌ 请配置 DEEPSEEK_API_KEY")
        exit(1)
    asyncio.run(main())
