# -*- coding: utf-8 -*-
"""提示词模板：角色设定 + 检索知识 + 历史对话 + 用户问题。

四段式结构与设计文档一致；检索知识标注来源与**条号**，用于生成可追溯的引用。

规则分两层（**多角色共用同一套模板**，所以必须按角色域拆分，不能把法律专有规则
塞给医生/心理/理财/证券等角色）：
  * ``COMMON_RULES`` —— 所有角色通用（只依据资料、不虚构、**资料与问题不匹配要明说**、
    出处、分点、通用免责声明）；
  * ``LEGAL_RULES`` —— **仅法律域角色附加**（``domain == "法律"`` 或 ``role_id == "lawyer"``）：
    非法律术语先澄清并请用户确认、法律意见免责声明。

2026-09-16 修订依据（一次真实问答事故：用户问「签了黑合同能否毁约不担责」，模型自行
编造「黑合同」定义、并拿《民法典》第五百零六条（免责条款无效）硬答出一段循环论证）：
  1. 「资料与问题不匹配必须明说」——**通用**规则，任何领域都适用；
  2. 「非法律术语先澄清」——**仅法律域**；
  3. 免责声明——通用一条 + 法律域一条（语气与执业指向不同）。

另：检索片段**不再把融合分当相似度**写进上下文（融合分上限 2.0、跨条不可比），改为标注条号。
"""
from __future__ import annotations

import re

from ..schemas import Message, Role, SearchHit

CONTEXT_HEADER = "【检索知识】"
HISTORY_HEADER = "【历史对话】"
QUESTION_HEADER = "【用户问题】"
#: B-9 长期记忆召回：**独立分区**，与「检索知识」并列但性质不同 ——
#: 它讲的是**用户本人**（跨会话沉淀下来的旧对话），**不是法律依据**。
MEMORY_HEADER = "【用户长期记忆】"
MEMORY_NOTE = ("（上述内容是这位用户**此前对话**里留下的记录，只用来理解他本人的情况与指代"
               "——比如他自称的姓名、所在地、案情背景。**它不是法律依据**：不得据此给条号、"
               "不得写进「参考：」、不得当作检索到的资料；与本次问题无关时请忽略。"
               "若问题问的正是**用户本人**（如「我姓什么」「我住哪」），可以据此直接回答，"
               "但**不要提到这段记录本身**，也不要说「根据记忆/资料」。）")

_ARTICLE_NO_RE = re.compile(r"第[一二三四五六七八九十百千零〇两]+条")

#: 通用规则（所有角色）
COMMON_RULES = (
    "回答要求：\n"
    "1. 只依据上面提供的检索知识作答；知识中没有的内容，明确说明「现有资料未涵盖」，"
    "不得虚构出处、编号或案例。\n"
    "2. 引用时写明出处（文档来源；若资料是法律文本，写到条号，"
    "如「《中华人民共和国民法典》第五百零六条」）。\n"
    "3. 如果检索到的资料与用户的问题**对不上**，必须直接说明「检索到的资料与你的问题不匹配」，"
    "并说明还缺哪类资料；**不得把不相关的资料当作依据硬答**。\n"
    "4. 使用简体中文，条理清晰，必要时分点作答。\n"
    "5. 回答末尾用「参考：」列出用到的资料名称。\n"
    "6. 涉及重大权益时提醒用户：本回答不构成专业意见，请就具体情况咨询相关执业人员。"
)

#: 仅法律域角色附加
LEGAL_RULES = (
    "法律领域补充要求：\n"
    "7. 如果用户问题里的词**不是法律术语**（如「黑合同」「假合同」这类口语说法），"
    "先说明它不是法律术语、可能对应哪些法律概念（如阴阳合同、无效合同、可撤销合同、"
    "显失公平、受欺诈等），再请用户确认具体指哪一种；**不要在确认前自行给它下定义"
    "并据此得出结论**。\n"
    "8. 涉及重大权益时提醒：本回答不构成法律意见，请就具体案情咨询执业律师。"
)

#: 兼容别名：历史调用方只认 ``BASE_RULES``（现在等价于通用规则）
BASE_RULES = COMMON_RULES

COT_HINT = "先在心中梳理推理步骤，再给出最终答案；不要输出思考过程本身。"

#: 日常对话 / 通用问题专用系统提示（t109 三分类路由的「通用」分支）。
#:
#: 为什么单开一条提示词、而不是复用法律人设：实测（用户截图 + 改前报文）里，非法律问题
#: 走法律人设时会**硬套法条**（"今天天气怎么样" ⇒ 引用《气象法》；"你好" ⇒ 塞 3 条
#: 司法解释/条例当引用），还会写出系统视角的"检索到的资料与你的问题不匹配"。
#: 通用分支的硬约束：不引法规、不提资料/依据/检索、不复述上一轮话题，只用常识自然作答。
GENERAL_CHAT_PROMPT = (
    "你是「法律 RAG 助手」的**通用对话通道**：这一轮的问题不是法律咨询。\n"
    "回答要求：\n"
    "1. 用常识自然回答：打招呼就自然回应，闲聊就聊两句，常识问题就给常识答案；\n"
    "2. **不得**引用任何法律法规、司法解释、判决或资料出处，也不要给条号；\n"
    "3. **不得**提及「检索」「知识库」「资料」「依据」「引用」这类系统视角的词；\n"
    "4. **只回应本轮问题本身**，不要复述或延伸之前的对话话题；\n"
    "5. 不知道就直说不知道，不要编造；若用户其实想问法律问题，提醒他把法律问题说清楚；\n"
    "6. 使用简体中文，简洁自然。"
)


def build_general_messages(question: str, memory_block: str = "") -> list[dict]:
    """通用对话分支的消息：**不带检索上下文，也（刻意）不带历史对话**。

    不带历史是**有意的**（t109 验收项）：用户先问法律问题、紧接着问「你好」时，带历史
    会让模型把上一轮的法律话题锚定到本轮（改前报文就是这个错法）；日常对话也不需要它。

    但**长期记忆例外**：那是"关于用户本人的事实"（姓名/所在地/案情背景），不是"上一轮
    的法律话题"。问「我姓什么」这类关于自己的问题走的正是通用分支，若这里也把记忆丢掉，
    长期记忆就永远用不上（B-9 实机结论）。所以 ``memory_block`` 非空时照常注入，
    并明确要求**不得**把它当法律依据、**不得**因此引用法规。
    """
    messages = [{"role": "system", "content": GENERAL_CHAT_PROMPT}]
    body = f"{memory_block}\n\n{question}" if memory_block else question
    messages.append({"role": "user", "content": body})
    return messages


def first_article(text: str) -> str:
    """取片段里第一个条号（用于上下文标注与引用落条）；没有则返回空串。"""
    match = _ARTICLE_NO_RE.search(text or "")
    return match.group(0) if match else ""


#: 内部文件名后缀：``.md``/``.txt``… 以及去重脚本追加的 ``__<hex>``
_SOURCE_SUFFIX_RE = re.compile(r"(__[0-9a-f]{6,})?\.(md|txt|json|pdf)$", re.IGNORECASE)


def clean_source_label(source: str) -> str:
    """把**内部文件名**变成能给人看的来源标注。

    真机踩坑（评测 v2 L24）：prompt 里直接把 ``中华人民共和国著作权法__2c909fdd.md``
    塞进上下文标注，模型就照着念，答案里出现 ``《中华人民共和国著作权法__2c909fdd.md》``
    —— 用户看到的是我们的内部文件名。规则：取文件名 → 去掉 ``__<hash>`` → 去掉扩展名。
    """
    text = str(source or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    return _SOURCE_SUFFIX_RE.sub("", text) or str(source or "").strip()


def is_legal_domain(role: Role | None) -> bool:
    """是否法律域角色（决定要不要附加法律专有规则）。

    注意：``role=None`` 时用的是默认法律助手人设，因此按法律域处理。
    """
    if role is None:
        return True
    return (getattr(role, "domain", "") or "") == "法律" or getattr(role, "role_id", "") == "lawyer"


def build_system_prompt(role: Role | None = None, cot: bool = False) -> str:
    persona = role.persona if role is not None else (
        "你是一位专业的法律助手，请基于现行有效法律条文回答问题，不得虚构法条。"
    )
    parts = [persona, "", COMMON_RULES]
    if is_legal_domain(role):
        parts.extend(["", LEGAL_RULES])
    if cot:
        parts.extend(["", COT_HINT])
    return "\n".join(parts)


def build_context_block(hits: list[SearchHit]) -> str:
    if not hits:
        return f"{CONTEXT_HEADER}\n（未检索到相关资料）"

    lines = [CONTEXT_HEADER]
    for index, hit in enumerate(hits, start=1):
        source = clean_source_label(hit.chunk.source) or "未知来源"
        article = first_article(hit.chunk.text)
        label = f"[{index}] 来源：{source}"
        if article:
            label += f"（{article}）"
        lines.append(label)
        lines.append(hit.chunk.text.strip())
        lines.append("")
    return "\n".join(lines).strip()


def build_question_block(question: str) -> str:
    return f"{QUESTION_HEADER}\n{question.strip()}"


def build_history_block(history: list[Message] | None) -> str:
    if not history:
        return ""
    lines = [HISTORY_HEADER]
    for message in history:
        speaker = "用户" if message.role == "user" else "助手"
        lines.append(f"{speaker}：{message.content.strip()}")
    return "\n".join(lines)


def build_memory_block(texts: list[str] | None) -> str:
    """把**长期记忆召回**的片段渲染成提示词里的独立分区（B-9 召回接线）。

    要点（都来自实机教训）：

    * **独立分区**：绝不混进「【检索知识】」—— 否则模型会把用户自己的旧话当成法条依据，
      甚至写进「参考：」；分区 + 显式否证（``MEMORY_NOTE``）两件一起做才拦得住。
    * **不参与引用**：调用方**不得**把这些片段交给 ``build_citations()``。
    * **空就是空**：没有召回内容时返回空串，提示词与"没开召回"逐字一致（便于 A/B 对照）。
    """
    items = [str(text).strip() for text in (texts or []) if str(text).strip()]
    if not items:
        return ""
    lines = [MEMORY_HEADER]
    lines.extend(f"- {item}" for item in items)
    lines.append(MEMORY_NOTE)
    return "\n".join(lines)


def build_messages(role: Role | None,
                   hits: list[SearchHit],
                   history: list[Message] | None,
                   question: str,
                   cot: bool = False,
                   few_shot: list[dict] | None = None,
                   memory_block: str = "") -> list[dict]:
    """组装 OpenAI 兼容的 messages 列表。

    ``memory_block``（B-9 长期记忆召回）为空时，输出与改动前**逐字一致**。
    """
    messages: list[dict] = [{"role": "system", "content": build_system_prompt(role, cot)}]

    for example in few_shot or []:
        messages.append({"role": "user", "content": str(example.get("user", ""))})
        messages.append({"role": "assistant", "content": str(example.get("assistant", ""))})

    body_parts = [build_context_block(hits)]
    if memory_block:
        body_parts.append(memory_block)
    history_block = build_history_block(history)
    if history_block:
        body_parts.append(history_block)
    body_parts.append(build_question_block(question))
    messages.append({"role": "user", "content": "\n\n".join(body_parts)})
    return messages


def build_citations(hits: list[SearchHit], limit: int = 5) -> list[dict]:
    """引用落条：snippet 优先以条号开头，另有独立 ``article`` 字段供前端展示。

    ``score`` 仍保留在载荷里（兼容既有调用方），但它是**两路融合分**（上限 2.0、
    跨条不可比），**不得当作相似度展示**——界面改为展示来源与条号。
    """
    citations: list[dict] = []
    for hit in hits[:limit]:
        article = first_article(hit.chunk.text)
        snippet = hit.chunk.text.strip()
        if article and not snippet.startswith(article):
            snippet = f"{article} {snippet}"
        citations.append({
            "source": clean_source_label(hit.chunk.source),
            "chunk_id": hit.chunk.id,
            "article": article,
            "score": round(float(hit.score), 4),
            "snippet": snippet[:160],
        })
    return citations
