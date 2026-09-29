"""
domains/base_domain.py — 领域基类与闲聊兜底

法律、医疗、英语各自封装成一个类，用哪一块就调哪一块：

    LegalDomain   法律顾问     kb_legal
    MedicalDomain 医疗咨询     kb_medical
    EnglishDomain 英语学习助手  kb_english
    ChatDomain    通用助手     不检索

BaseDomain 规定三个接口：

    route()     这条输入属于本领域吗（返回置信度 0~1）
    retrieve()  用本领域自己的集合与检索参数取资料
    generate()  用本领域的人设与提示词生成回答

子类只填人设、关键词、集合名这些数据，流程由基类共享，避免三个文件互相抄。

提示词是四段式：人设 +【历史对话参考】+【知识库检索】+【最近对话】。
"""

from __future__ import annotations

from typing import Any, Iterator

import config

# 命中多少个关键词算"完全匹配"，用于把命中数折算成 0~1 的置信度
_ROUTE_FULL_HITS = 2

# 检索为空时要求模型说的话。需求点名了这句原话，所以写成常量而不是散在模板里。
NO_KB_REPLY = "资料中未找到相关内容"

SYSTEM_PROMPT_TEMPLATE = """{identity}

# 你的性格特点
{personality}

# 你的回答风格
{speaking_style}

【历史对话参考】
{memory}

【知识库检索】
{knowledge}

【最近对话】
{history}

# 回答要求
1. 优先依据【知识库检索】的内容作答，不得编造资料中不存在的事实、法条、数据或页码。
2. 若【知识库检索】是「本轮未检索到」，必须以"{no_kb}"开头，再基于通用知识谨慎作答。
3. 回答末尾另起一行附引用，格式为 `参考来源：文件名 第N页`；没有用到资料时写 `参考来源：通用知识`。
4. 使用简体中文回答，直接以专业口吻作答。{extra_rule}
"""

# 闲聊用的精简模板：不检索，也就不渲染三段
CHAT_PROMPT_TEMPLATE = """{identity}

# 你的性格特点
{personality}

# 你的回答风格
{speaking_style}

# 最近对话
{history}

请用简体中文自然地回应，不要提及检索、知识库、资料这类系统术语。
"""


def cite_of(doc: dict) -> str:
    """拼引用串：文件名 + 第N页。

    page 为 0 说明是纯文本导入，没有页码概念，此时只给文件名。
    """
    source = doc.get("source") or "未知来源"
    page = int(doc.get("page") or 0)
    return f"{source} 第{page}页" if page > 0 else source


def format_docs(docs: list[dict] | None) -> str:
    """把检索到的分块格式化成【知识库检索】段，每条带文件名与页码。"""
    blocks: list[str] = []
    for idx, doc in enumerate(docs or [], start=1):
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        score = doc.get("rerank_score", doc.get("score"))
        header = f"[资料{idx}] 来源：{cite_of(doc)}"
        if isinstance(score, (int, float)):
            header += f"（相关度 {float(score):.2f}）"
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks) or "（本轮未检索到相关的知识库内容）"


def format_history(history: list[dict] | None) -> str:
    """把短期记忆格式化成【最近对话】段。"""
    lines = [
        f"{'用户' if turn.get('role') == 'user' else '助手'}：{(turn.get('content') or '').strip()}"
        for turn in (history or [])
        if (turn.get("content") or "").strip()
    ]
    return "\n".join(lines) or "（暂无）"


class BaseDomain:
    """领域基类：定义三个接口，子类只填数据。"""

    domain: str = ""
    role_name: str = ""
    collection: str = ""
    greeting: str = ""
    disclaimer: str = ""

    # 关键词表，route() 用它算置信度
    keywords: tuple[str, ...] = ()

    # 人设三件套 + 该领域专属的额外约束
    identity: str = ""
    personality: str = ""
    speaking_style: str = ""
    extra_rule: str = ""

    # ---------------------------------------------------------- 接口一：路由

    def route(self, user_input: str, history: list[dict] | None = None) -> float:
        """本领域对这条输入的匹配置信度，取值 0~1。

        只按关键词命中数折算，不调用大模型——真正的意图识别在
        intent.detect_intent 里做（那里会调模型），这里只是兜底打分。
        """
        text = (user_input or "").strip().lower()
        if not text or not self.keywords:
            return 0.0
        hits = sum(1 for word in self.keywords if word.lower() in text)
        return min(1.0, hits / _ROUTE_FULL_HITS)

    # ---------------------------------------------------------- 接口二：检索

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """用本领域的集合做混合检索 + 重排 + 余弦过滤。

        闲聊领域没有集合，直接返回空——由调用方据此跳过检索。
        """
        if not self.collection:
            return []

        from retrieval import hybrid_retrieve

        return hybrid_retrieve(query, self.collection, top_k=top_k or config.KB_TOP_K)

    # ---------------------------------------------------------- 接口三：生成

    def generate(
        self,
        system_prompt: str,
        question: str,
        history: list[dict] | None = None,
    ) -> Iterator[str]:
        """流式生成回答。"""
        from llm_client import get_client

        return get_client().chat_stream(system_prompt, question, history)

    def answer(
        self,
        system_prompt: str,
        question: str,
        history: list[dict] | None = None,
    ) -> str:
        """一次性生成完整回答，供评测与非流式接口使用。"""
        from llm_client import get_client

        return get_client().chat(system_prompt, question, history)

    # ---------------------------------------------------------- 提示词组装

    def build_system_prompt(
        self,
        docs: list[dict] | None = None,
        memory_text: str = "",
        history: list[dict] | None = None,
    ) -> str:
        """组装四段式系统提示词。

        【历史对话参考】无条件渲染：召回为空时写"暂无历史记忆"，
        而不是整段省略——需求要求长期记忆这一段必须出现在提示词里。
        """
        return SYSTEM_PROMPT_TEMPLATE.format(
            identity=self.identity,
            personality=self.personality,
            speaking_style=self.speaking_style,
            extra_rule=f" {self.extra_rule}" if self.extra_rule else "",
            no_kb=NO_KB_REPLY,
            memory=(memory_text or "").strip() or "（暂无历史记忆）",
            knowledge=format_docs(docs),
            history=format_history(history),
        )

    def sources(self, docs: list[dict] | None = None) -> list[dict]:
        """把检索结果整理成前端展示用的来源列表。"""
        return format_sources(docs)

    def __repr__(self) -> str:  # pragma: no cover - 仅用于排查
        return f"<{type(self).__name__} domain={self.domain} role={self.role_name}>"


def format_sources(docs: list[dict] | None = None) -> list[dict]:
    """来源列表：每条带文件名、页码、相关度与预览。

    全程没有检索结果时返回「通用知识」一条，前端据此提示用户
    本轮回答没有资料支撑。
    """
    items: list[dict] = []
    for index, doc in enumerate(docs or [], start=1):
        score = doc.get("rerank_score", doc.get("score", 0.0))
        items.append(
            {
                "index": index,
                "file": doc.get("source") or "未知来源",
                "page": int(doc.get("page") or 0),
                "score": round(float(score or 0.0), 4),
                "preview": (doc.get("text") or "")[:100],
            }
        )
    if not items:
        # 前端按 type == "general" 判断要不要显示"未引用知识库"的提示条
        return [{"index": 0, "type": "general", "file": "通用知识", "page": 0, "score": 0.0, "preview": ""}]
    return items


class ChatDomain(BaseDomain):
    """闲聊兜底：不检索知识库，也不注入长期记忆。"""

    domain = "chat"
    role_name = "通用助手"
    identity = "你是一个友好、博学的通用智能助手。"
    personality = "热情、自然、简洁，不啰嗦。"
    speaking_style = "像朋友聊天一样自然回应，保持轻松的语气。"
    greeting = "你好！有什么可以帮你的吗？"

    def build_system_prompt(
        self,
        docs: list[dict] | None = None,
        memory_text: str = "",
        history: list[dict] | None = None,
    ) -> str:
        """闲聊不渲染资料段，提示词短一些，省下的 token 留给正文。"""
        return CHAT_PROMPT_TEMPLATE.format(
            identity=self.identity,
            personality=self.personality,
            speaking_style=self.speaking_style,
            history=format_history(history),
        )


if __name__ == "__main__":
    chat = ChatDomain()
    assert chat.route("你好", None) == 0.0, "闲聊领域不该有领域置信度"
    assert "通用助手" == chat.role_name

    prompt = chat.build_system_prompt(history=[{"role": "user", "content": "你好"}])
    assert "【知识库检索】" not in prompt, "闲聊不该出现知识库段"

    print(f"闲聊提示词 {len(prompt)} 字")
    print("base_domain 自检通过。")
