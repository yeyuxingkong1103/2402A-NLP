"""提示词模板与消息拼装。"""
from __future__ import annotations

DEFAULT_RULES = (
    "1. 严格依据【参考资料】回答，禁止编造资料中未出现的事实、数据或法条。\n"
    "2. 若资料不足以回答，请明确说明，并给出可进一步核实的方向。\n"
    "3. 始终保持角色设定、语气与专业边界。\n"
    "4. 回答结构清晰，必要时分点说明。"
)

SYSTEM_TEMPLATE = "{role_prompt}\n\n【参考资料】\n{context}\n\n【回答要求】\n{rules}"


def build_context(docs: list[dict], max_chars: int = 3000) -> str:
    """把检索结果拼装为带来源标注的上下文，优先使用父块。"""
    parts: list[str] = []
    total = 0
    for i, d in enumerate(docs, 1):
        content = d.get("parent") or d.get("text") or ""
        block = f"[{i}] 来源：{d.get('source', '未知')}\n{content}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts) or "（未检索到相关资料）"


def build_system(role_prompt: str, context: str, rules: str | None = None) -> str:
    return SYSTEM_TEMPLATE.format(
        role_prompt=role_prompt, context=context, rules=rules or DEFAULT_RULES
    )


def build_messages(
    role_prompt: str, context: str, history: list[dict], user_message: str,
    rules: str | None = None,
) -> list[dict]:
    """拼装 system + 历史对话 + 当前问题。"""
    messages = [{"role": "system", "content": build_system(role_prompt, context, rules)}]
    for m in history:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_message})
    return messages
