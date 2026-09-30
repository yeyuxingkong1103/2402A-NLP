from dataclasses import dataclass  # 导入 dataclass，用于定义数据类
from typing import AsyncIterator  # 导入 AsyncIterator，用于标注异步迭代器类型

from ..config import get_settings  # 从上层 config 模块导入配置获取函数
from .prompt_builder import build_prompt  # 导入 prompt 构建函数
from .rag_pipeline import RAGPipeline  # 导入 RAG 检索流水线类


@dataclass  # 装饰器：自动生成 __init__ 等方法
class PreparedTurn:  # 定义「已准备好的一轮对话」数据类
    messages: list[dict]  # 发给 LLM 的消息列表
    sources: list[dict]  # 本轮检索到的来源列表，供前端展示


class ChatService:  # 定义聊天服务类
    def __init__(self, llm, rag: RAGPipeline, redis):  # 构造函数，注入三个依赖
        self.llm = llm  # 保存 LLM 组件
        self.rag = rag  # 保存 RAG 检索流水线
        self.redis = redis  # 保存 redis 客户端（存短期上下文/摘要）
        self.rounds = get_settings().short_term_rounds  # 读取短期记忆保留轮数配置

    async def prepare(self, session_id, user_id, character, query) -> PreparedTurn:  # 异步准备一轮对话所需的消息和来源
        retrieved = await self.rag.retrieve(character["id"], user_id, query)  # 调用 RAG 检索，拿到设定/记忆/文档结果
        summary = await self.redis.get_summary(session_id)  # 从 redis 取会话摘要
        recent = await self.redis.get_recent(session_id, self.rounds)  # 从 redis 取最近若干轮对话
        memories_text = [s.text for s in retrieved.settings] + [s.text for s in retrieved.memories]  # 合并设定与记忆文本
        documents_text = [s.text for s in retrieved.documents]  # 提取文档文本
        msgs = build_prompt(  # 调用 prompt 构建函数，组装最终消息列表
            character,  # 角色信息
            memories_text,  # 设定+记忆文本
            summary,  # 会话摘要
            recent + [{"role": "user", "content": query}],  # 最近对话 + 本轮用户输入
            documents=documents_text,  # 文档文本
        )
        sources = [s.to_dict() for s in retrieved.settings + retrieved.memories + retrieved.documents]  # 把三类来源合并并转成 dict
        return PreparedTurn(messages=msgs, sources=sources)  # 返回准备好的本轮数据

    async def chat(self, session_id, user_id, character: dict, query: str) -> str:  # 非流式聊天入口
        turn = await self.prepare(session_id, user_id, character, query)  # 准备本轮消息
        return await self.llm.chat(turn.messages)  # 调用 LLM 一次性返回完整回复

    async def chat_stream(self, session_id, user_id, character: dict, query: str) -> AsyncIterator[str]:  # 流式聊天入口
        turn = await self.prepare(session_id, user_id, character, query)  # 准备本轮消息
        async for token in self.llm.chat_stream(turn.messages):  # 异步迭代 LLM 返回的 token
            yield token  # 逐个 token 向下游抛出