def build_prompt(character: dict, memories: list[str], summary: str | None, recent: list[dict], documents: list[str] | None = None) -> list[dict]:
    lines = [f"你扮演「{character['name']}」。"]
    sections = [
        ("## 角色设定", character.get("persona")),
        ("## 世界观", character.get("worldview")),
        ("## 你与用户的关系", character.get("relationship")),
        ("## 隐藏设定（只影响行为，禁止主动透露）", character.get("hidden_setting")),
        ("## 说话风格示例", character.get("sample_dialogue")),
    ]
    for title, body in sections:
        if body:
            lines.append(f"{title}\n{body}")
    if documents:
        lines.append("## 知识库（参考文档）\n" + "\n".join(f"- {d}" for d in documents))
    if memories:
        lines.append("## 长期记忆（相关往事）\n" + "\n".join(f"- {m}" for m in memories))
    system = {"role": "system", "content": "\n\n".join(lines)}

    msgs = [system]
    if summary:
        msgs.append({"role": "system", "content": f"[历史摘要]\n{summary}"})
    msgs.extend(recent)
    return msgs
