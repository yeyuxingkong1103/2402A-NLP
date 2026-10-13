# 工单编号：人工智能 NLP-RAG-LightRAG 优化
import json
import asyncio
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer
from config_v12 import (
    WORKING_DIR, EMBED_MODEL_PATH, QUESTION_FILE, COMPARE_RESULT_FILE,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
)

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
    with open(QUESTION_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        llm_model_name=DEEPSEEK_MODEL,
    )
    await rag.initialize_storages()

    results = []
    for q in questions:
        print(f"[{q['id']}] {q['question']}")
        try:
            context_light = await rag.aquery(
                q["question"],
                param=QueryParam(mode="hybrid", only_need_context=True)
            )
            context_str = str(context_light)
        except Exception as e:
            context_str = ""
            print(f"  失败: {e}")

        results.append({
            "id": q["id"],
            "question": q["question"],
            "lightrag_context": context_str
        })
        print(f"  LightRAG 上下文长度: {len(context_str)}")

    with open(COMPARE_RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到 {COMPARE_RESULT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())