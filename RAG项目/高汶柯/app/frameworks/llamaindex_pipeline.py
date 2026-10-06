"""LlamaIndex 引擎：入库分块与检索的替代路径。

通过 ``RAG_ENGINE=llamaindex`` 启用：分块用 LlamaIndex SentenceSplitter，
检索用自定义 BaseRetriever 复用 native 混合检索，生成复用统一 LLM。
"""
from __future__ import annotations

from app.config import settings
from app.db.mysql_store import get_role
from app.db.redis_store import redis_store
from app.logging_conf import log
from app.rag.prompt import DEFAULT_RULES, build_context
from app.rag.rerank import rerank


def chunk_with_llamaindex(text: str, chunk_size: int | None = None,
                          overlap: int | None = None) -> list[str]:
    """用 LlamaIndex SentenceSplitter 分块（入库替代路径）。"""
    try:
        from llama_index.core.node_parser import SentenceSplitter

        splitter = SentenceSplitter(
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=settings.chunk_overlap if overlap is None else overlap,
        )
        return [n.text for n in splitter.get_nodes_from_documents([_doc(text)]) if n.text.strip()]
    except Exception as exc:  # noqa: BLE001
        log.warning("LlamaIndex 分块不可用: %s", exc)
        from app.ingest.chunker import chunk_sentence

        return chunk_sentence(text, chunk_size or settings.chunk_size, 0)


def _doc(text: str):
    from llama_index.core import Document

    return Document(text=text)


def _retrieve_nodes(query: str, domain: str, top_k: int) -> list[dict]:
    """自定义检索器：复用 native 多路召回 + 重排，返回并列节点。"""
    try:
        from llama_index.core.schema import TextNode
    except Exception as exc:  # noqa: BLE001
        log.warning("LlamaIndex 不可用: %s", exc)
        return []

    from app.rag.retriever import retrieve

    rows = retrieve(query, domain=domain, use_rag=True)["docs"]
    rows = rerank(query, rows, top_k)
    return [
        {"text": r.get("text"), "source": r.get("source"),
         "node": TextNode(text=r.get("text", ""), metadata={"source": r.get("source", "")})}
        for r in rows
    ]


def query(user_id: str, role_id: int, message: str, use_rag: bool = True) -> dict | None:
    """LlamaIndex 检索路径问答；依赖缺失时返回 None 以便回退。"""
    role = get_role(role_id)
    if not role:
        return None
    domain = role.get("domain", "general")
    if use_rag:
        try:
            nodes = _retrieve_nodes(message, domain, settings.rerank_top_k)
        except Exception as exc:  # noqa: BLE001
            log.warning("LlamaIndex 检索失败: %s", exc)
            return None
    else:
        nodes = []

    docs = [{"text": n["text"], "source": n["source"]} for n in nodes]
    context = build_context(docs)

    from app.core.registry import get_llm
    from app.rag.pipeline import _after_reply

    system = f"{role['system_prompt']}\n\n【参考资料】\n{context}\n\n【回答要求】\n{DEFAULT_RULES}"
    messages = [{"role": "system", "content": system}]
    messages += redis_store.get_messages(user_id, role_id)
    messages.append({"role": "user", "content": message})
    reply = get_llm().chat(messages)

    _after_reply(user_id, role_id, message, reply,
                 {"context": context, "history": redis_store.get_messages(user_id, role_id)})
    return {"reply": reply, "role_id": role_id,
            "sources": [{"source": d["source"]} for d in docs], "rewritten_query": None}
