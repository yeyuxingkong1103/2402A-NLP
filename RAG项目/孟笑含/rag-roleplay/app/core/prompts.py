# -*- coding: utf-8 -*-
"""角色扮演提示词模板：把人设 + 短期记忆历史 + 用户输入拼成发给大模型的消息。"""

DEFAULT_PROMPT_TEMPLATE = """你正在扮演「{role_name}」。
# 解析：默认提示词模板（存库）。{role_name} 占位符由角色名替换

【角色设定】
{persona}
# 解析：{persona} 占位符由角色人设（persona 字段）替换

【行为规则】
1. 始终保持角色身份，不要跳出角色，不要提及你是AI或语言模型
# 解析：行为规则第 1 条：约束模型不暴露自己是 AI
2. 用口语化的中文交流，回答简洁自然，符合角色说话习惯
# 解析：行为规则第 2 条：约束语言风格口语化
3. 与角色无关的问题，也尽量用角色的口吻回应
# 解析：行为规则第 3 条：任何问题都以角色身份回应
4. 回答中不要输出任何分析过程或括号备注
# 解析：行为规则第 4 条：禁止输出思考过程与备注

【历史对话】
{history}
# 解析：{history} 占位符由短期记忆（最近 10 轮）渲染的文本替换

{knowledge}
# 解析：{knowledge} 占位符由知识库检索块渲染的文本替换（旧模板无此占位符也不报错）
用户：{user_input}
# 解析：本轮用户输入写进模板
{role_name}：
# 解析：模板以角色名冒号结尾，引导模型接续输出（回答直接跟在冒号后）
"""

REQUIRED_PLACEHOLDERS = ("{role_name}", "{persona}", "{history}", "{user_input}")
# 解析：四个必需占位符——模板缺少任何一个都无法正确渲染


def format_history(history: list[dict], role_name: str) -> str:
    """把 [{role, content}, ...] 渲染成文本历史；空历史返回空串。"""
    lines = []
    # 解析：收集每轮渲染后的文本行
    for item in history:
        # 解析：遍历短期记忆中的每条消息
        sender = role_name if item["role"] == "assistant" else "用户"
        # 解析：发言者——assistant 显示角色名，user 显示"用户"
        lines.append(f"{sender}：{item['content']}")
        # 解析：按"发言者：内容"格式追加一行
    return "\n".join(lines)
    # 解析：所有行用换行拼接返回（空历史返回空串）


def format_memory_block(memories: list[str]) -> str:
    """把长期记忆渲染成提示词段落；无记忆返回空串（整段省略）。"""
    if not memories:
        # 解析：没有记忆时直接返回空串（提示词中省略整段）
        return ""
    items = "\n\n".join(m.strip() for m in memories if m and m.strip())
    # 解析：去掉每条记忆的首尾空白，过滤空条目，用空行拼接
    if not items:
        # 解析：记忆全是空白时同样返回空串
        return ""
    return (
        # 解析：返回【长期记忆】段——与知识块风格一致但提示语不同（"自然地参考"而非"严格依据"）
        "【长期记忆】\n"
        "以下是你们过去的对话片段，可在回答中自然地参考这些信息：\n"
        f"{items}"
        # 解析：记忆内容作为段落主体
    )


def format_knowledge_block(knowledge: str) -> str:
    """把检索到的知识渲染成提示词段落（准确性优先）；空知识返回空串（整段省略）。"""
    if not knowledge.strip():
        # 解析：无知识返回空串（提示词中省略整段）
        return ""
    return (
        # 解析：返回【知识库内容】段——5 条准确性指令 + 2 组 few-shot 示例（三轮 RAGAS 评测的成果）
        "【知识库内容】\n"
        f"{knowledge.strip()}\n"
        # 解析：知识块原文（去除首尾空白）
        "回答要求：\n"
        # 解析：以下 5 条指令强制回答严格基于知识库
        "1. 必须严格依据以上知识库内容回答，不得加入知识库之外的信息或自己的推测\n"
        # 解析：指令 1：杜绝幻觉（faithfulness 提升的关键）
        "2. 涉及数值、标准、指标时，必须使用知识库中的精确数值\n"
        # 解析：指令 2：数值必须精确（correctness 提升的关键）
        "3. 用户问题包含多个子问题时，逐个子问题回答，不要遗漏\n"
        # 解析：指令 3：多子问题逐一回答不遗漏
        "4. 知识库中没有的信息，明确回答\"知识库中没有相关信息\"，不要编造\n"
        # 解析：指令 4：不知道就明说，不编造
        "5. 知识库与问题完全无关时，才可正常回答\n"
        # 解析：指令 5：只有检索结果完全不相关时才自由回答
        "【回答示例】\n"
        # 解析：以下 2 组中性 few-shot 示例（枚举列全 + 归类照原文，用甲乙丙占位不泄漏领域内容）
        "问：这类物品有哪些？\n"
        # 解析：示例 1 的问题
        "答：知识库中共列出五种：甲、乙、丙、丁、戊。（枚举类问题应全部列出，不遗漏）\n"
        # 解析：示例 1 的回答——示范枚举要列全
        "\n"
        # 解析：两组示例之间的空行
        "问：X属于哪一类？\n"
        # 解析：示例 2 的问题
        "答：根据知识库，X属于甲类。（归类以知识库原文为准）"
        # 解析：示例 2 的回答——示范归类要照原文
    )


def build_messages(
    role_name: str,
    persona: str,
    prompt_template: str,
    history: list[dict],
    user_input: str,
    knowledge: str = "",
    memories: list[str] = None,
) -> list[dict]:
    """拼装发给大模型的完整消息列表：[{role: system, ...}, {role: user, ...}]。

    knowledge 为 RAG 检索到的知识；模板含 {knowledge} 占位符时才注入，
    旧模板（无该占位符）不受影响。
    memories 为长期记忆；追加在模板格式化之后（模板无关，所有角色通用）。
    """
    missing = [p for p in REQUIRED_PLACEHOLDERS if p not in prompt_template]
    # 解析：检查模板是否包含全部四个必需占位符
    if missing:
        # 解析：缺占位符则报错（配置错误要尽早暴露）
        raise ValueError(f"提示词模板缺少占位符: {', '.join(missing)}")
        # 解析：错误信息列出所有缺失的占位符

    system_content = prompt_template.format(
        # 解析：用实际值填充模板
        role_name=role_name,
        # 解析：角色名替换 {role_name}
        persona=persona,
        # 解析：人设替换 {persona}
        history=format_history(history, role_name),
        # 解析：短期记忆渲染成文本替换 {history}
        user_input=user_input,
        # 解析：本轮输入替换 {user_input}
        knowledge=format_knowledge_block(knowledge),
        # 解析：知识块渲染成文本替换 {knowledge}（模板没有该占位符时此参数被忽略）
    )
    memory_block = format_memory_block(memories or [])
    # 解析：长期记忆渲染成段落（memories 为 None 时按空列表处理）
    if memory_block:
        # 解析：有记忆时才追加
        system_content = f"{system_content}\n\n{memory_block}"
        # 解析：长期记忆段追加在模板格式化之后——模板无关，自定义模板也通用
    return [
        # 解析：返回标准消息列表（OpenAI 兼容格式）
        {"role": "system", "content": system_content},
        # 解析：系统消息——角色设定+规则+历史+知识+记忆
        {"role": "user", "content": user_input},
        # 解析：用户消息——本轮输入
    ]
