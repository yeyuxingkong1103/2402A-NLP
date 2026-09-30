# -*- coding: utf-8 -*-
"""对话编排：取历史 → 拼提示词 → 调大模型 → 写入短期记忆。"""
from collections.abc import AsyncIterator
# 解析：AsyncIterator 用于流式方法返回类型标注

from app.core.prompts import build_messages
# 解析：提示词拼装函数（人设+历史+知识+记忆）


# 对话编排：取历史 → 拼提示词 → 调大模型 → 写短期记忆
class ChatService:
    # 解析：对话编排服务——把"记忆管理+提示词拼装+LLM 调用"串成完整对话流程
    def __init__(self, llm, memory):
        # 解析：构造——依赖注入（测试可传假实现）
        self.llm = llm
        # 解析：保存大模型客户端
        self.memory = memory
        # 解析：保存短期记忆对象（Redis 实现）

    async def chat(
        # 解析：非流式对话入口
        self, role, user_input: str, user_id: int, role_id: int, knowledge: str = "",
        # 解析：角色对象、用户输入、用户ID、角色ID、知识块文本（RAG 注入用）
        memories: list[str] = None,
        # 解析：长期记忆列表（可选）
    ) -> str:
        """非流式对话，返回完整回复。"""
        messages = self._build(role, user_input, user_id, role_id, knowledge, memories)
        # 解析：拼装提示词消息（内部会取 Redis 短期记忆历史）
        reply = await self.llm.chat(messages)
        # 解析：调用大模型得到完整回复
        self.memory.append(user_id, role_id, "user", user_input)
        # 解析：把用户消息写入短期记忆
        self.memory.append(user_id, role_id, "assistant", reply)
        # 解析：把角色回复写入短期记忆（多轮对话的"记住"）
        return reply
        # 解析：返回回复

    async def chat_stream(
        # 解析：流式对话入口
        self, role, user_input: str, user_id: int, role_id: int, knowledge: str = "",
        # 解析：参数与非流式一致
        memories: list[str] = None,
        # 解析：长期记忆列表
    ) -> AsyncIterator[str]:
        """流式对话：逐块产出；结束后把完整回复写入记忆，中途失败不写半截。"""
        messages = self._build(role, user_input, user_id, role_id, knowledge, memories)
        # 解析：拼装提示词消息
        self.memory.append(user_id, role_id, "user", user_input)
        # 解析：先把用户消息写入记忆（回复还没生成）

        parts = []
        # 解析：收集流式分块，用于最后拼完整回复
        async for chunk in self.llm.chat_stream(messages):
            # 解析：逐块消费大模型输出
            parts.append(chunk)
            # 解析：收集本块
            yield chunk
            # 解析：把本块交给上层（SSE 逐字推送）

        self.memory.append(user_id, role_id, "assistant", "".join(parts))
        # 解析：全部产出后才把完整回复写入记忆——中途断流不会留下半截回复（业务规则 BR-11）

    def _build(
        # 解析：内部方法——拼装提示词
        self, role, user_input: str, user_id: int, role_id: int, knowledge: str,
        # 解析：角色、输入、双方ID、知识
        memories: list[str] = None,
        # 解析：长期记忆
    ) -> list[dict]:
        history = self.memory.get_history(user_id, role_id)
        # 解析：取当前会话最近 10 轮短期记忆
        return build_messages(
            # 解析：调用提示词模块拼装标准消息列表
            role_name=role.name,
            # 解析：角色名
            persona=role.persona,
            # 解析：角色人设
            prompt_template=role.prompt_template,
            # 解析：该角色的提示词模板（可自定义）
            history=history,
            # 解析：短期记忆历史
            user_input=user_input,
            # 解析：本轮输入
            knowledge=knowledge,
            # 解析：知识块（RAG 检索结果）
            memories=memories,
            # 解析：长期记忆
        )
