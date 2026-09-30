"""
intent.py — 用户意图识别

判断用户输入属于哪个领域，决定后续路由到哪个知识库：

    legal    法律问题   → 法律顾问
    medical  医疗问题   → 医疗咨询
    english  英语学习   → 英语学习助手
    chat     闲聊       → 通用助手，跳过知识库检索

主路径是让大模型做 few-shot 分类；模型不可用或输出异常时，
用领域关键词表兜底，保证任何时候都能给出一个可用结果。
"""

from __future__ import annotations

import re

import config

DOMAINS = ("legal", "medical", "english", "chat")


def _keywords() -> dict[str, tuple[str, ...]]:
    """从 domains 注册表取各领域的关键词表。

    关键词属于领域定义的一部分，收在 domains/*.py 里；这里只借用，
    不再各存一份，否则改了一处忘了另一处就会两边判断不一致。
    """
    from domains import all_domains

    return {dom.domain: dom.keywords for dom in all_domains()}


# 纯寒暄，命中即判定为闲聊
SMALL_TALK: tuple[str, ...] = (
    "你好", "您好", "hi", "hello", "在吗", "谢谢", "感谢", "再见", "拜拜",
    "你是谁", "介绍下你自己", "早上好", "晚上好", "下午好",
)

INTENT_SYSTEM_PROMPT = """你是一个意图分类器。请判断用户的问题属于以下哪个领域，只输出一个英文标签，不要输出任何其他内容。

可选标签：
- legal：法律相关问题，包括合同、劳动纠纷、婚姻继承、交通事故、刑事、诉讼维权等
- medical：医疗健康问题，包括症状描述、用药咨询、检查报告解读、疾病科普等
- english：英语学习问题，包括语法、词汇、发音、写作批改、翻译、口语练习等
- chat：闲聊、问候、与上述三个领域都无关的通用问题

示例：
用户：劳动合同到期不续签，公司要赔偿吗？      → legal
用户：我最近总是失眠，需要吃药吗？            → medical
用户：虚拟语气怎么用，能举个例子吗？          → english
用户：你好，今天过得怎么样？                  → chat
用户：帮我把这段话翻译成英文                → english
用户：邻居装修太吵，我能报警吗？              → legal

现在请判断下面这个问题："""

# 模型输出里出现这些词时，映射回标准标签
_LABEL_ALIASES: dict[str, str] = {
    "法律": "legal", "法规": "legal", "法务": "legal",
    "医疗": "medical", "医学": "medical", "健康": "medical", "医生": "medical",
    "英语": "english", "英文": "english", "语言": "english",
    "闲聊": "chat", "聊天": "chat", "问候": "chat", "其他": "chat", "通用": "chat",
}


def detect_by_keywords(user_input: str) -> str | None:
    """纯关键词判断，命中足够多的领域词时返回标签，否则返回 None。"""
    text = (user_input or "").strip().lower()
    if not text:
        return None

    if len(text) <= 10 and any(word in text for word in SMALL_TALK):
        return "chat"

    table = _keywords()
    scores = {
        domain: sum(1 for word in words if word.lower() in text)
        for domain, words in table.items()
    }
    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] > 0 else None


def parse_label(raw: str) -> str | None:
    """从模型输出里解析出标准领域标签，解析失败返回 None。"""
    if not raw:
        return None

    text = raw.strip().lower()
    # 模型有时会输出 "legal。" 或 "**legal**" 这类带修饰的文本
    cleaned = re.sub(r"[^a-z一-鿿]", "", text)

    for domain in DOMAINS:
        if domain in cleaned:
            return domain
    for alias, domain in _LABEL_ALIASES.items():
        if alias in cleaned:
            return domain
    return None


def detect_intent(user_input: str, chat_history: list[dict] | None = None) -> str:
    """识别用户意图，返回 "legal" / "medical" / "english" / "chat"。

    策略：寒暄和强关键词走快速通道；其余交给大模型，
    模型不可用或输出无法解析时回落到关键词判断。
    """
    user_input = (user_input or "").strip()
    if not user_input:
        return "chat"

    # 快速通道：明显的寒暄不必调用模型
    quick = detect_by_keywords(user_input)
    if quick == "chat":
        return "chat"

    try:
        from llm_client import get_client

        # 上一轮已经确定了领域时，把上下文一并给模型，减少短问题被误判
        context_hint = ""
        if chat_history:
            last_user = next(
                (t.get("content", "") for t in reversed(chat_history) if t.get("role") == "user"),
                "",
            )
            if last_user:
                context_hint = f"（用户上一个问题是：{last_user[:60]}）"

        raw = get_client().chat(
            INTENT_SYSTEM_PROMPT,
            f"{user_input}{context_hint}",
            history=None,
            temperature=0.0,
            max_tokens=16,
            reasoning_effort=config.LLM_REASONING_EFFORT,
        )
        label = parse_label(raw)
        if label:
            return label
    except Exception:
        # 模型不可用不应该是致命错误，继续用关键词兜底
        pass

    return quick or "chat"


def detect_intent_detail(
    user_input: str, chat_history: list[dict] | None = None, query: str | None = None
) -> dict:
    """返回带调试信息的意图判断结果，供接口和日志使用。

    query 传入改写后的查询时可提高判断准确率（可选）。
    """
    domain = detect_intent(query or user_input, chat_history)
    return {
        "domain": domain,
        "role": _role_of(domain),
        "need_retrieval": domain != "chat",
    }


def _role_of(domain: str) -> str:
    from domains import role_for

    return role_for(domain)


def build_rewrite_prompt(query: str, history: list[dict] | None = None) -> str:
    """构造 Query 改写提示词：把省略、指代的话补全成独立可检索的查询。"""
    if history:
        recent = history[-4:]
        history_text = "\n".join(
            f"{'用户' if t.get('role') == 'user' else '助手'}：{t.get('content', '')}" for t in recent
        )
    else:
        history_text = "（无历史对话）"

    return f"""你是一个查询改写助手。请把用户的问题改写成一句"独立、完整、适合检索"的查询语句。

要求：
1. 补全指代和省略的信息，让这句话脱离上下文也能看懂。
2. 保留原问题的核心意图和关键实体，不要添加用户没提到的信息。
3. 只输出改写后的那一句话，不要任何解释、前缀或引号。

历史对话：
{history_text}

用户当前问题：{query}

改写后的查询："""


def rewrite_query(user_input: str, chat_history: list[dict] | None = None) -> str:
    """Query 改写：把省略和指代补全成独立可检索的查询。

    改写失败或结果明显异常时，原样返回用户输入，保证不影响主流程。
    """
    user_input = (user_input or "").strip()
    if not user_input:
        return ""

    # 没有历史上下文时改写没有意义，省一次模型调用
    if not chat_history:
        return user_input

    try:
        from llm_client import get_client

        rewritten = get_client().chat(
            "你是一个查询改写助手，只输出改写后的查询语句。",
            build_rewrite_prompt(user_input, chat_history),
            history=None,
            temperature=0.0,
            max_tokens=128,
            reasoning_effort=config.LLM_REASONING_EFFORT,
        ).strip()
    except Exception:
        return user_input

    rewritten = rewritten.strip().strip("\"'“”")
    # 模型偶尔会输出整段解释，长度异常或为空时放弃改写
    if not rewritten or len(rewritten) > 200 or len(rewritten) < 2:
        return user_input
    return rewritten


if __name__ == "__main__":
    cases = {
        "劳动合同到期不续签，公司需要赔偿吗？": "legal",
        "我最近总是失眠，需要吃药吗？": "medical",
        "虚拟语气怎么用，能举个例子吗？": "english",
        "你好": "chat",
        "邻居装修太吵我能报警吗": "legal",
    }
    for text, expected in cases.items():
        got = detect_by_keywords(text)
        print(f"  {text[:20]:22s} -> 关键词={got} 期望={expected}")

    assert detect_by_keywords("劳动合同到期不续签，公司需要赔偿吗？") == "legal"
    assert detect_by_keywords("我最近总是失眠，需要吃药吗？") == "medical"
    assert detect_by_keywords("你好") == "chat"

    assert parse_label("legal") == "legal"
    assert parse_label("**Medical**") == "medical"
    assert parse_label("这属于法律问题") == "legal"
    print("intent 自检通过。")
