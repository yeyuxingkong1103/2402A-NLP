# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG 项目-LightRAG 优化任务
# 关联工单：人工智能NLP-RAG-基于PDF文档的问答系统 | 人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 模块：lightrag_rag/build_lightrag —— 用 LightRAG 对《招股说明书1/2》建「实体-关系」知识图谱
# 说明：语料取招股书 1/2 的分块结果（按 (文档,页) 聚合，跳过图像块），
#       交给 LightRAG 做实体/关系抽取并写入图存储（NetworkX）+ 向量存储（NanoVectorDB）。
#       Embedding = Ollama bge-m3（本地，与既有 RAG 同模型）；抽取实体的 LLM 支持两种后端：
#         · LR_LLM=ollama（默认）：本机 qwen2:7b，完全离线，但 CPU/GPU 慢（每页 10–30 分钟）
#         · LR_LLM=deepseek：DeepSeek 云端 API（deepseek-flash），建图快很多；检索问答仍用本地 qwen2:7b
# 用法（在 .venv_lightrag 下运行）：
#   set LIMIT=5 && python lightrag_rag\build_lightrag.py          # 冒烟
#   set LR_LLM=deepseek && python lightrag_rag\build_lightrag.py  # 云端建图（快）
import asyncio
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "app"))
sys.stdout.reconfigure(encoding="utf-8")

from lightrag import LightRAG                     # noqa: E402
from lightrag.llm.ollama import ollama_model_complete, ollama_embed  # noqa: E402
from lightrag.utils import EmbeddingFunc          # noqa: E402

from config import OLLAMA_BASE, EMBED_MODEL       # noqa: E402

STORE = os.path.join(ROOT, "data", "lightrag_store")
CHUNKS = os.path.join(ROOT, "data", "index", "chunks.jsonl")
GEN_MODEL = os.environ.get("LR_GEN_MODEL", "qwen2:7b")
MAX_ASYNC = int(os.environ.get("LR_MAX_ASYNC", "2"))
GLEANING = int(os.environ.get("LR_GLEANING", "0"))
CHUNK_TOK = int(os.environ.get("LR_CHUNK_TOKENS", "1200"))
LIMIT = int(os.environ.get("LIMIT", "0"))
SCOPE = os.environ.get("LR_SCOPE", "0") == "1"
HOST = OLLAMA_BASE.replace("/v1", "")
SCOPE_JSON = os.path.join(HERE, "scope_pages.json")

# ---- 抽取实体的 LLM 后端（工单 12：支持本地 Ollama / 云端 DeepSeek 两种）----
LR_LLM = os.environ.get("LR_LLM", "ollama")
DS_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-836293b19d3e43c19d8497f99cf2a4ae")
DS_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
DS_BASE = os.environ.get("DEEPSEEK_BASE", "https://api.deepseek.com/v1")


async def deepseek_complete(prompt, system_prompt=None, history_messages=None, **kwargs):
    """OpenAI 兼容的 DeepSeek 抽取函数（LightRAG llm_model_func 签名）。"""
    from lightrag.llm.openai import openai_complete_if_cache
    return await openai_complete_if_cache(
        DS_MODEL, prompt,
        system_prompt=system_prompt, history_messages=history_messages or [],
        base_url=DS_BASE, api_key=DS_KEY, timeout=180, **kwargs)


def load_scope():
    if not SCOPE:
        return None
    d = json.load(open(SCOPE_JSON, encoding="utf-8"))
    return {k: set(str(x) for x in v) for k, v in d.items()}


def load_docs(scope=None):
    """按 (文档,页) 聚合语料，生成「一页一文档」的插入单元。"""
    groups = {}
    with open(CHUNKS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c.get("type") == "image":
                continue                      # 图像块只有 CLIP 标签，无实体信息
            key = (c.get("doc"), str(c.get("page")))
            groups.setdefault(key, []).append(c.get("text") or "")
    docs, cur = [], None
    order = []
    for (doc, page), texts in groups.items():
        if doc not in order:
            order.append(doc)
    for doc in order:
        for (d, page), texts in groups.items():
            if d != doc:
                continue
            if scope is not None and page not in scope.get(doc, set()):
                continue
            txt = "\n".join(t for t in texts if t).strip()
            if len(txt) < 30:
                continue
            docs.append("【%s·第%s页】\n%s" % (doc, page, txt))
    return docs


def build_rag(llm=None):
    raw_embed = getattr(ollama_embed, "func", ollama_embed)
    llm = llm or LR_LLM
    if llm == "deepseek":
        llm_func, llm_name = deepseek_complete, DS_MODEL
        llm_kwargs = {}
    else:
        llm_func, llm_name = ollama_model_complete, GEN_MODEL
        llm_kwargs = {"host": HOST, "keep_alive": "30m",
                      "options": {"num_ctx": 8192, "temperature": 0.05}}
    return LightRAG(
        working_dir=STORE,
        llm_model_func=llm_func,
        llm_model_name=llm_name,
        llm_model_max_async=MAX_ASYNC,
        llm_model_kwargs=llm_kwargs,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024, max_token_size=8192,
            func=lambda texts: raw_embed(texts, embed_model=EMBED_MODEL, host=HOST)),
        entity_extract_max_gleaning=GLEANING,
        chunk_token_size=CHUNK_TOK,
        chunk_overlap_token_size=100,
        log_level="WARNING",
        log_file_path=os.path.join(ROOT, "data", "lightrag.log"),
    )


async def main():
    scope = load_scope()
    docs = load_docs(scope)
    if LIMIT:
        docs = docs[:LIMIT]
    print("[lightrag] 待插入文档（页）= %d，LLM=%s，gleaning=%d，chunk=%d，async=%d" %
          (len(docs), (DS_MODEL if LR_LLM == "deepseek" else GEN_MODEL), GLEANING, CHUNK_TOK, MAX_ASYNC))
    rag = build_rag()
    await rag.initialize_storages()
    t0 = time.time()
    try:
        await rag.ainsert(docs)
    except AttributeError:
        await rag.insert(docs)
    print("[lightrag] 插入完成，用时 %.1f 分钟" % ((time.time() - t0) / 60))
    try:
        await rag.finalize_storages()
    except Exception as e:  # noqa: BLE001
        print("finalize warn:", e)
    st = os.path.join(STORE, "kv_store_doc_status.json")
    if os.path.exists(st):
        d = json.load(open(st, encoding="utf-8"))
        print("[lightrag] doc_status 条数 =", len(d))
    print("LIGHTRAG_BUILD_DONE")


if __name__ == "__main__":
    asyncio.run(main())
