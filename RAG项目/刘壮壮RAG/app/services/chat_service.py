from app.core.memory import ShortTermMemory


def build_messages(
    system_prompt: str,
    history: list[dict],
    knowledge: list[str],
    memory: list[str],
    user_message: str,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    context_parts = []
    if knowledge:
        context_parts.append("【知识库】\n" + "\n".join(f"- {k}" for k in knowledge))
    if memory:
        context_parts.append("【长期记忆】\n" + "\n".join(f"- {m}" for m in memory))
    if context_parts:
        messages.append(
            {
                "role": "system",
                "content": "以下是可供参考的上下文信息：\n\n" + "\n\n".join(context_parts),
            }
        )

    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages


async def maybe_compress(
    memory: ShortTermMemory, conversation_id: int, llm, max_messages: int = 40
) -> None:
    history = await memory.get(conversation_id)
    if len(history) <= max_messages:
        return
    to_compress = history[:-6]
    recent = history[-6:]
    prompt = (
        "请把以下对话压缩为一段简洁的摘要，保留关键信息（人物、话题、结论）：\n\n"
        + "\n".join(f"{m['role']}: {m['content']}" for m in to_compress)
    )
    summary = ""
    async for piece in llm.chat_stream([{"role": "user", "content": prompt}]):
        summary += piece
    await memory.clear(conversation_id)
    await memory.append(conversation_id, "system", f"[对话摘要] {summary}")
    for m in recent:
        await memory.append(conversation_id, m["role"], m["content"])
