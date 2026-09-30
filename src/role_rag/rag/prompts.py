"""角色化提示词：人设 + 记忆 + 知识片段 + 引用协议。

提示词按「小模型友好」设计：指令短、结构固定、把知识片段放在问题之前，
并要求模型用 ``[编号]`` 标注依据，便于后续做引用校验。
"""

from __future__ import annotations

from typing import Any, Sequence

from ..memory.memory import MemoryBundle
from ..roles import Role

CITATION_RULES = """# 回答规则
1. 只依据「知识片段」作答；片段里没有的内容，直接回答「知识库中没有相关内容」，不要编造。
2. 每个关键结论后面标注来源编号，格式为 [1]、[2]；编号必须来自知识片段。
3. 不要编造编号、文献、条文、数据或链接。
4. 用简体中文作答；先给结论，再给依据与适用条件；条目化、可执行。
5. 与角色身份无关或超出知识范围的问题，礼貌说明并给出可执行的下一步。"""


def system_prompt(role: Role) -> str:
    """拼装系统提示词（人设 + 红线 + 回答规则）。"""

    return f"{role.persona_block()}\n\n{CITATION_RULES}"


def format_contexts(contexts: Sequence[Any]) -> str:
    """把召回片段渲染成带编号的知识片段块。"""

    if not contexts:
        return "（本轮没有检索到相关知识片段）"
    blocks: list[str] = []
    for index, chunk in enumerate(contexts, start=1):
        title = getattr(chunk, "doc_title", "") or "未命名文档"
        section = getattr(chunk, "section", "") or ""
        scope = getattr(chunk, "scope", "")
        text = (getattr(chunk, "text", "") or "").strip()
        header = f"[{index}] 《{title}》"
        if section and section != title:
            header += f" › {section}"
        if scope:
            header += f"（知识域：{scope}）"
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)


def format_memory(bundle: MemoryBundle | None) -> str:
    """把记忆渲染成提示词中的「已知信息」块。"""

    if bundle is None:
        return ""
    lines: list[str] = []
    if bundle.summary:
        lines.append(f"- 会话摘要：{bundle.summary}")
    for fact in bundle.facts[:6]:
        lines.append(f"- {fact.text}")
    if not lines:
        return ""
    return "# 关于用户的已知信息\n" + "\n".join(lines)


def build_messages(
    role: Role,
    question: str,
    contexts: Sequence[Any],
    memory: MemoryBundle | None = None,
    max_recent_turns: int = 5,
    safety_notice: str = "",
) -> list[dict[str, str]]:
    """组装最终送给模型的 messages。"""

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt(role)}]
    memory_block = format_memory(memory)
    if memory_block:
        messages.append({"role": "system", "content": memory_block})

    if memory is not None:
        recent = list(memory.recent)[-max_recent_turns * 2 :]
        for item in recent:
            role_name = "user" if item.get("role") == "user" else "assistant"
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            messages.append({"role": role_name, "content": content[:1500]})

    if safety_notice:
        messages.append({"role": "system", "content": safety_notice})

    user_content = (
        "# 知识片段\n"
        f"{format_contexts(contexts)}\n\n"
        "# 用户问题\n"
        f"{question.strip()}\n\n"
        "请依据上面的知识片段回答，并在句末标注来源编号。"
    )
    messages.append({"role": "user", "content": user_content})
    return messages


def rewrite_prompt(history_lines: Sequence[str], question: str) -> list[dict[str, str]]:
    """查询改写：把带指代的问题补全成可独立检索的问句。"""

    history = "\n".join(history_lines[-6:]) or "（无历史）"
    return [
        {
            "role": "user",
            "content": (
                "下面是用户与助手的最近对话，以及用户的最后一个问题。\n"
                "请把最后一个问题改写成一个不依赖上下文、可以直接用于检索的独立问句，"
                "只输出改写后的问句本身，不要解释，不要加引号。\n\n"
                f"对话：\n{history}\n\n最后一个问题：{question}\n改写结果："
            ),
        }
    ]
