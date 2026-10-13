# 工单编号：人工智能 NLP-RAG-LightRAG 优化
import os
import asyncio
import numpy as np
import pymupdf
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer
from config_v12 import (
    DOCS_DIR, WORKING_DIR, EMBED_MODEL_PATH, MAX_PAGES,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
)

os.makedirs(WORKING_DIR, exist_ok=True)
embed_model = SentenceTransformer(EMBED_MODEL_PATH, device="cuda")


async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        DEEPSEEK_MODEL, prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        **kwargs
    )


@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
async def embedding_func(texts: list[str]) -> np.ndarray:
    return embed_model.encode(texts, normalize_embeddings=True)


async def main():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        llm_model_name=DEEPSEEK_MODEL,
    )
    await rag.initialize_storages()

    for fn in os.listdir(DOCS_DIR):
        if not fn.lower().endswith(".pdf"):
            continue
        path = os.path.join(DOCS_DIR, fn)
        doc = pymupdf.open(path)
        # 只取前 MAX_PAGES 页
        pages = min(MAX_PAGES, len(doc))
        text = "\n".join(doc[i].get_text() for i in range(pages))
        print(f"插入 {fn}（前 {pages} 页，{len(text)} 字符）")
        await rag.ainsert(text)

    print("索引完成，存储目录:", WORKING_DIR)


if __name__ == "__main__":
    asyncio.run(main())