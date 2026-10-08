# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-LightRAG优化
src/lightrag_v12/lightrag_wrapper.py —— LightRAG 封装

使用 DeepSeek（OpenAI 兼容）作为 LLM、bge-m3 作为 embedding，
NetworkX 作为图存储，构建金融招股说明书知识图谱。

关键设计：使用后台持久事件循环，避免 embedding 函数的 asyncio.Queue
在多次 asyncio.run() 之间绑定到不同 loop 导致失败。
"""
import asyncio
import os
import threading
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

WORK_ORDER = "人工智能NLP-RAG-LightRAG优化"

# ========== 全局单例 ==========
_rag = None
_embed_model = None
_loop = None
_loop_thread = None


def _ensure_loop():
    """工单十二：确保有一个持久运行的事件循环（后台线程）"""
    global _loop, _loop_thread
    if _loop is not None and _loop.is_running():
        return _loop

    def _run_loop():
        global _loop
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        _loop.run_forever()

    _loop_thread = threading.Thread(target=_run_loop, daemon=True)
    _loop_thread.start()
    # 等待 loop 启动
    while _loop is None or not _loop.is_running():
        import time
        time.sleep(0.01)
    return _loop


def _run_async(coro):
    """工单十二：在持久事件循环中同步运行协程"""
    loop = _ensure_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def _get_embed_model():
    """工单十二：懒加载 bge-m3 嵌入模型（CPU，避免显存占用）"""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        model_path = os.environ.get("EMBEDDING_MODEL_PATH", "/home/dabaie/models/bge-m3")
        _embed_model = SentenceTransformer(model_path, device="cpu")
    return _embed_model


async def _bge_embedding(texts):
    """工单十二：bge-m3 嵌入函数，供 LightRAG 调用"""
    model = _get_embed_model()
    # 工单十二：bge-m3 不接受 max_seq_length 参数，仅用 batch_size
    emb = model.encode(texts, batch_size=8,
                       normalize_embeddings=True, show_progress_bar=False)
    return np.array(emb, dtype=np.float32)


def _make_llm_func():
    """工单十二：构造 LightRAG 所需的 llm_model_func

    策略：
    - entity_extraction / keyword_extraction：使用规则抽取（离线，无需 API）
    - 普通查询：优先调用 DeepSeek API，失败时返回基于上下文的兜底回答
    """
    from lightrag.llm.openai import openai_complete_if_cache
    from .rule_based_llm import rule_based_entity_extraction, rule_based_keyword_extraction

    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    async def llm_func(prompt, system_prompt=None, history_messages=None,
                       keyword_extraction=False, entity_extraction=False, **kwargs):
        # 工单十二：从 prompt 内容判断任务类型（LightRAG 1.5+ 不传标志）
        is_entity = ("Extract entities" in prompt) or ("missed or incorrectly formatted" in prompt) or entity_extraction
        is_keyword = ("keywords" in prompt.lower() and len(prompt) < 1000) or keyword_extraction

        # 工单十二：实体/关键词抽取走规则，不消耗 API 额度
        if is_entity:
            return rule_based_entity_extraction(prompt)
        if is_keyword:
            return rule_based_keyword_extraction(prompt)

        # 工单十二：普通查询优先走 API，失败则返回基于上下文的兜底
        try:
            return await openai_complete_if_cache(
                model=model, prompt=prompt, system_prompt=system_prompt,
                history_messages=history_messages, base_url=base_url,
                api_key=api_key, keyword_extraction=keyword_extraction,
                entity_extraction=entity_extraction, timeout=60, **kwargs,
            )
        except Exception:
            # 工单十二：API 不可用时，返回检索到的上下文（system_prompt 含图谱检索结果）
            ctx = (system_prompt or "") + "\n" + (prompt or "")
            return ctx[:2000]

    return llm_func


def get_lightrag(working_dir=None):
    """工单十二：获取 LightRAG 单例"""
    global _rag
    if _rag is not None:
        return _rag

    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    if working_dir is None:
        working_dir = str(Path(__file__).resolve().parents[2] / "data" / "lightrag_v12")
    os.makedirs(working_dir, exist_ok=True)

    embedding_dim = int(os.environ.get("EMBEDDING_DIM", "1024"))
    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=512,
        func=_bge_embedding,
    )

    _rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=_make_llm_func(),
        embedding_func=embedding_func,
        entity_extract_max_gleaning=1,
        entity_extract_max_entities=40,
        entity_extract_max_records=100,
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        top_k=40,
        chunk_top_k=20,
        enable_llm_cache=True,
    )
    return _rag


def lightrag_insert_text(text, doc_id=None):
    """工单十二：向 LightRAG 插入文本（构建知识图谱）"""
    async def _do():
        rag = get_lightrag()
        await rag.initialize_storages()
        await rag.ainsert(text)

    _run_async(_do())


def lightrag_query(query, mode="hybrid", top_k=40):
    """工单十二：LightRAG 查询

    Args:
        query: 用户问题
        mode: local（局部/实体）/ global（全局/社区）/ hybrid（双层）

    工单十二：LLM 不可用时，全量查询返回的兜底文本含提示词模板（---Role---），
    此时改用 only_need_context=True 直接获取真实检索上下文
    （知识图谱实体/关系 + 相关文本块）作为答案。
    """
    from lightrag import QueryParam

    async def _do():
        rag = get_lightrag()
        await rag.initialize_storages()
        param = QueryParam(mode=mode, top_k=top_k)
        result = await rag.aquery(query, param=param)
        # 工单十二：检测到兜底模板时，改为返回纯检索上下文
        if not result or "---Role---" in result[:100]:
            param_ctx = QueryParam(mode=mode, top_k=top_k, only_need_context=True)
            ctx = await rag.aquery(query, param=param_ctx)
            if ctx:
                return ctx
        return result

    return _run_async(_do())


def get_graph_stats():
    """工单十二：获取知识图谱统计信息（实体数、关系数）"""
    import json
    rag = get_lightrag()
    wd = Path(rag.working_dir)
    stats = {"entities": 0, "relations": 0}
    try:
        with open(wd / "kv_store_full_entities.json") as f:
            stats["entities"] = len(json.load(f))
    except Exception:
        pass
    try:
        with open(wd / "kv_store_full_relations.json") as f:
            stats["relations"] = len(json.load(f))
    except Exception:
        pass
    return stats


if __name__ == "__main__":
    rag = get_lightrag()
    print(f"[v12] LightRAG 初始化成功，working_dir={rag.working_dir}")
