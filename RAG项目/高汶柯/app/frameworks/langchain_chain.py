"""LangChain 引擎：六要素（Models / Prompts / Chains / Memory / Indexes / Agents）落地。

通过 ``RAG_ENGINE=langchain`` 启用；由 :mod:`app.rag.pipeline` 调用，失败时自动回退 native。
"""
from __future__ import annotations

from app.config import settings
from app.core.llm import PROVIDERS
from app.db.mysql_store import get_role
from app.db.redis_store import redis_store
from app.logging_conf import log
from app.rag.postprocess import postprocess
from app.rag.prompt import DEFAULT_RULES, build_context
from app.rag.rerank import rerank


def _build_llm():
    """Models 要素：ChatOpenAI（OpenAI 兼容，支持本地 vLLM/SGLang）。"""
    from langchain_openai import ChatOpenAI

    preset = PROVIDERS.get(settings.llm_provider.lower(), PROVIDERS["deepseek"])
    return ChatOpenAI(
        model=settings.llm_model or preset["model"],
        base_url=settings.llm_base_url or preset["base_url"],
        api_key=settings.llm_api_key or "sk-no-key",
        temperature=settings.llm_temperature,
    )


class MilvusRetriever:
    """Indexes 要素：复用 native 混合检索 + 重排作为检索器。"""

    def __init__(self, domain: str, top_k: int | None = None) -> None:
        self.domain = domain
        self.top_k = top_k or settings.rerank_top_k

    def invoke(self, query: str) -> list[dict]:
        from app.rag.retriever import retrieve

        result = retrieve(query, domain=self.domain, use_rag=True)
        return rerank(query, result["docs"], self.top_k)


def _history_messages(user_id: str, role_id: int) -> list:
    """Memory 要素：把 Redis 短期记忆转为 LangChain 消息。"""
    try:
        from langchain_core.messages import AIMessage, HumanMessage
    except Exception:  # noqa: BLE001
        return []
    messages = []
    for m in redis_store.get_messages(user_id, role_id):
        cls = HumanMessage if m.get("role") == "user" else AIMessage
        messages.append(cls(content=m.get("content", "")))
    return messages


def _build_chain():
    """Prompts + Chains 要素：system/history/human 三段式链。"""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{role_prompt}\n\n【参考资料】\n{context}\n\n【回答要求】\n{rules}"),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )
    return prompt | _build_llm() | StrOutputParser()


def route_intent(message: str) -> str:
    """Agents 要素（轻量路由）：判断是否需要检索知识库。"""
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        router = ChatPromptTemplate.from_messages(
            [("system", "判断用户问题是否需要查询知识库。只回答 RAG 或 CHAT。"),
             ("human", "{q}")]
        ) | _build_llm() | StrOutputParser()
        return "RAG" if "RAG" in router.invoke({"q": message}).upper() else "CHAT"
    except Exception:  # noqa: BLE001
        return "RAG"


def answer(user_id: str, role_id: int, message: str, use_rag: bool = True) -> dict | None:
    """LangChain 链式问答；依赖缺失时返回 None 以便上层回退。"""
    role = get_role(role_id)
    if not role:
        return None
    try:
        _build_chain()
    except Exception as exc:  # noqa: BLE001
        log.warning("LangChain 不可用: %s", exc)
        return None

    domain = role.get("domain", "general")
    docs = MilvusRetriever(domain).invoke(message) if use_rag else []
    context = build_context(docs)

    chain = _build_chain()
    raw = chain.invoke(
        {
            "role_prompt": role["system_prompt"],
            "context": context,
            "rules": DEFAULT_RULES,
            "history": _history_messages(user_id, role_id),
            "question": message,
        }
    )
    reply = postprocess(raw, context)

    from app.rag.pipeline import _after_reply

    _after_reply(
        user_id, role_id, message, reply,
        {"context": context, "history": redis_store.get_messages(user_id, role_id)},
    )
    return {
        "reply": reply,
        "role_id": role_id,
        "sources": [{"source": d.get("source")} for d in docs],
        "rewritten_query": None,
    }
