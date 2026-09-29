# -*- coding: utf-8 -*-
"""角色人格组装：把数据库里的三层人格渲染成最终 system prompt。

三层解耦：
    身份层 identity_block      角色是谁
    风格层 style_json          怎么说话
    约束层 domain_constraints  不能做什么
模板骨架存在 characters.prompt_template，所有角色共用。
"""
from ..core import config
from ..models import Character

# 知识库 → 人类可读的范围描述（用于告诉模型「你的知识范围」）
SCOPE_LABELS = {
    config.COLLECTION_MEDICAL: "中国高血压防治指南（2024年修订版）、国家基层高血压防治管理指南（2025版）",
    config.COLLECTION_LEGAL: "中国现行法律法规（民法典、消费者权益保护法等 176 部法律）",
}


def _render_style(style: dict | None) -> str:
    if not style:
        return "自然、清晰地表达。"
    parts = []
    if style.get("tone"):
        parts.append(f"语气：{style['tone']}")
    if style.get("address"):
        parts.append(f"称呼用户为「{style['address']}」")
    if style.get("length"):
        parts.append(f"篇幅：{style['length']}")
    if style.get("extra"):
        parts.append(style["extra"])
    return "；".join(parts) + "。" if parts else "自然、清晰地表达。"


def _render_context(hits: list[dict]) -> str:
    """把检索结果编号后拼成知识片段。"""
    if not hits:
        return "（本次没有检索到相关知识片段）"
    from .retrieval import format_source
    blocks = []
    for i, h in enumerate(hits, 1):
        text = (h.get("text") or "")[:config.CHUNK_MAX_CHARS]
        blocks.append(f"[{i}] 来源：{format_source(h)}\n{text}")
    return "\n\n".join(blocks)


def _render_memory(memory: list[dict]) -> str:
    """把短期记忆渲染成对话历史。"""
    if not memory:
        return "（这是本轮对话的开始）"
    role_name = {"user": "用户", "assistant": "你"}
    lines = []
    for m in memory:
        content = (m.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role_name.get(m.get('role'), m.get('role'))}：{content}")
    return "\n".join(lines) if lines else "（这是本轮对话的开始）"


def render_system(character: Character, question: str,
                  hits: list[dict], memory: list[dict],
                  long_memories: list[dict] | None = None) -> str:
    """渲染最终 system prompt。

    手写链路(build_messages)与 LangChain 链路(lc_chain)共用此函数，
    确保两条链路的提示词完全一致 —— 否则对比实验失去意义。

    `long_memories`：跨会话长期记忆（由 `long_memory.recall` 召回）。
    **刻意采用「格式化后拼接」而不是新增模板占位符** ——
    现有角色的 `prompt_template` 是存在数据库里的，加占位符得改数据；
    而拼接对模板零要求，也不会因某个角色模板漏写占位符而 KeyError。
    """
    template = character.prompt_template or (
        "{identity_block}\n\n## 【知识片段】\n{context}\n\n## 【用户问题】\n{question}"
    )

    system = template.format(
        identity_block=character.identity_block or f"你是{character.name}。",
        knowledge_scope=(character.description
                         or SCOPE_LABELS.get(character.kb_collection, "你的专业知识")),
        style_block=_render_style(character.style_json),
        domain_constraints=character.domain_constraints or "保持专业边界。",
        context=_render_context(hits),
        memory=_render_memory(memory),
        question=question,
    )

    if long_memories:
        from .long_memory import format_for_prompt
        block = format_for_prompt(long_memories)
        if block:
            # 放在最前：属于「背景信息」，应早于知识片段出现
            system = f"{block}\n\n{system}"
    return system


def build_messages(character: Character, question: str,
                   hits: list[dict], memory: list[dict],
                   long_memories: list[dict] | None = None) -> list[dict]:
    """构造送给大模型的 messages。"""
    return [
        {"role": "system",
         "content": render_system(character, question, hits, memory, long_memories)},
        {"role": "user", "content": question},
    ]


def build_sources(hits: list[dict]) -> list[dict]:
    """构造返回给前端的来源列表。"""
    from .retrieval import format_source
    sources = []
    for i, h in enumerate(hits, 1):
        sources.append({
            "idx": i,
            "source": format_source(h),
            "collection": h.get("collection"),
            "law_name": h.get("law_name"),
            "article_no": h.get("article_no"),
            "page": h.get("page"),
            "score": h.get("score"),
            "rerank_score": h.get("rerank_score"),
            "text": (h.get("text") or "")[:300],
        })
    return sources
