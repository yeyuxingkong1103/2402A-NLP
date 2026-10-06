# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单01 - 基于PDF文档的问答系统
"""
生成器：提示词组装 + 流式输出。

【System Prompt 为什么要这样写】
逐条对应工单01 的验收项，且每条都是实测踩出来的：

1. 「只依据片段作答」—— 招股书是强事实场景，模型没见过的数字必须拒答。
   实测纯 LLM（无检索）在强制作答时把法定代表人**编造成「李军」**
   （真值是「程家明」）。加这条约束后，无据可依时应输出"资料中未提及"。
2. 「用与提问相同的语言作答」—— 工单01 功能验收第 4 条要求中英双语。
   实测不写这句时，英文提问会得到中文回答（qwen3 的中文先验太强）。
3. 「页码标注」—— 可溯源是 RAG 相对纯 LLM 的核心差异，也是演示时的说服力来源。
4. 固定的开头 —— 放在 system 里且**内容恒定**，可命中 Ollama 的前缀 KV cache，
   显著降低 prefill 耗时（这是 ≤3 秒指标的关键手段之一）。
"""

from __future__ import annotations

from typing import AsyncIterator, Sequence

from app.config import settings
from app.core.ollama_client import OllamaClient, get_client
from app.core.vectorstore import SearchHit

SYSTEM_PROMPT = """你是一个严谨的文档问答助手，基于用户提供的《武汉兴图新科电子股份有限公司招股意向书》片段回答问题。

必须遵守以下规则：
1. 【只依据片段】只使用下面「参考资料」中明确出现的信息作答，不要引入任何外部知识。
2. 【无据拒答】如果片段中没有足够信息，直接回答"资料中未提及相关信息"，绝不猜测或编造。
3. 【同语言作答】用户用中文提问就用中文回答，用英文提问就用英文回答（Answer in English if the question is in English）。
4. 【标注页码】回答末尾用「（见 1-1-XX 页）」的形式标注依据的页码，可标注多个。
5. 【简洁】回答控制在 200 字以内，直接给结论和关键数字，不要复述问题、不要解释推理过程。
6. 【数字原样】涉及金额、比例、日期等数字时，原样引用，不要换算或四舍五入。
7. 【脱敏名称原样】原文用「某」字脱敏的名称（如《某视频技术规范1.0》、
   「某情报、指挥、控制与通信网络一体化工程」）必须**原样照抄**，
   不得补全、改写或意译成别的名称。"""

NO_CONTEXT_REPLY_ZH = "资料中未提及相关信息。"
NO_CONTEXT_REPLY_EN = "The provided document does not contain this information."

# 语言指令放在**用户消息末尾**，而不是只写在 system 里。
# 【实测】只写 system 时 qwen3 会无视它 —— 英文提问照样答中文
# （qwen3 的中文先验太强）。把指令贴到问题后面、用显式祈使句，
# 才能压住这个先验。这是工单01「功能验收 4：多语言支持」能否过的关键。
_LANG_DIRECTIVE_EN = ("\n\nIMPORTANT: Answer in English only. "
                      "Do not use Chinese in your answer.")
_LANG_DIRECTIVE_ZH = "\n\n请用中文回答。"


def build_messages(question: str, context: str,
                   history: Sequence[dict[str, str]] | None = None
                   ) -> list[dict[str, str]]:
    """组装对话消息。history 为工单05 多轮对话预留。"""
    msgs: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        msgs.extend(history)

    directive = _LANG_DIRECTIVE_EN if _is_english(question) else _LANG_DIRECTIVE_ZH
    if context:
        user = f"参考资料：\n{context}\n\n问题：{question}{directive}"
    else:
        user = f"{question}{directive}"
    msgs.append({"role": "user", "content": user})
    return msgs


def _is_english(text: str) -> bool:
    """粗判提问语言：ASCII 字母占比超过三成即视为英文。"""
    letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    return letters > len(text) * 0.3


class Generator:
    def __init__(self, client: OllamaClient | None = None) -> None:
        self.client = client or get_client()

    # ------------------------------------------------------------------
    async def generate(self, question: str, hits: Sequence[SearchHit],
                       *, context: str | None = None,
                       history: Sequence[dict[str, str]] | None = None) -> str:
        """非流式生成（供评估脚本使用）。"""
        if context is None:
            from app.core.retriever import build_context
            context = build_context(hits, query=question)

        if not context.strip():
            return NO_CONTEXT_REPLY_EN if _is_english(question) else NO_CONTEXT_REPLY_ZH

        msgs = build_messages(question, context, history)
        return await self.client.chat(msgs)

    # ------------------------------------------------------------------
    async def generate_stream(self, question: str, hits: Sequence[SearchHit],
                              *, context: str | None = None,
                              history: Sequence[dict[str, str]] | None = None
                              ) -> AsyncIterator[str]:
        """流式生成，逐块 yield 文本。"""
        if context is None:
            from app.core.retriever import build_context
            context = build_context(hits, query=question)

        if not context.strip():
            yield NO_CONTEXT_REPLY_EN if _is_english(question) else NO_CONTEXT_REPLY_ZH
            return

        msgs = build_messages(question, context, history)
        async for piece in self.client.chat_stream(msgs):
            yield piece

    # ------------------------------------------------------------------
    async def baseline_no_rag(self, question: str) -> str:
        """
        纯 LLM 基线（不提供任何检索片段）—— 用于工单01「对比分析」验收项。
        预期表现：面对招股书细节问题应当拒答；若强行作答则会产生幻觉。
        """
        msgs = [
            {"role": "system", "content":
                "你是一个知识助手。如实回答用户问题；如果不确定或不知道，"
                "请直接说明你不确定，绝对不要编造。"},
            {"role": "user", "content": question},
        ]
        return await self.client.chat(msgs)


async def warmup() -> None:
    """
    预热模型：把 qwen3 拉进常驻，避免首个真实请求吃 7 秒冷启动（实测）。

    注意必须是 async 版本 —— FastAPI 的 lifespan 本身跑在事件循环里，
    在它内部调 asyncio.run() 会直接抛
    "asyncio.run() cannot be called from a running event loop"。
    """
    c = get_client()
    await c.chat([{"role": "user", "content": "你好"}], max_tokens=1)
