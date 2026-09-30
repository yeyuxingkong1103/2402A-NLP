# -*- coding: utf-8 -*-
"""大模型兼容层：OpenAI 兼容协议，本地 vLLM/SGLang 与在线 API 通用。"""
from collections.abc import AsyncIterator
# 解析：AsyncIterator 用于标注流式方法的返回类型

from openai import AsyncOpenAI
# 解析：openai 官方 SDK 的异步客户端——OpenAI 兼容协议（DeepSeek/豆包/千问/本地 vLLM 通用）


class MockLLM:
    """压测/开发用：返回预设回复，不发任何外部请求。"""

    reply = "这是模拟回复：压测模式下的预设回答，用于测试 API 骨架吞吐。"
    # 解析：类属性——预设回复文本（chat 直接返回、stream 按它切块）

    async def chat(self, messages: list[dict]) -> str:
        # 解析：非流式对话（压测模式）
        return self.reply
        # 解析：不调用任何外部服务，直接返回预设回复（压测 S2/S4 场景用它测 API 骨架吞吐）

    async def chat_stream(self, messages: list[dict]):
        # 解析：流式对话（压测模式）
        for i in range(0, len(self.reply), 4):
            # 解析：按 4 字步长切分预设文本
            yield self.reply[i : i + 4]
            # 解析：逐块产出（模拟真实流式分块行为）


class QueryRewriter:
    """查询改写扩写：把用户问题改写成 1-2 个更利于检索的查询变体（LLM）。"""

    def __init__(self, llm):
        # 解析：构造——注入底层 LLM（依赖注入，测试可传假实现）
        self.llm = llm
        # 解析：保存 LLM 引用

    async def rewrite(self, query: str) -> list[str]:
        # 解析：把问题改写为检索变体列表
        messages = [
            # 解析：组装改写指令消息
            {
                "role": "system",
                # 解析：系统角色——给模型的改写指令
                "content": (
                    # 解析：指令内容
                    "你是检索查询改写助手。把用户问题改写为 1-2 个更利于知识库检索的查询变体，"
                    "每行一个，只输出查询本身，不要编号、引号或解释。"
                    # 解析：约束输出格式（每行一个变体、无编号无解释）
                ),
            },
            {"role": "user", "content": query},
            # 解析：用户角色——待改写的问题
        ]
        reply = await self.llm.chat(messages)
        # 解析：调用大模型生成改写结果
        return [line.strip() for line in reply.splitlines() if line.strip()][:2]
        # 解析：按行切分→去空白→过滤空行→最多取 2 条变体


class OpenAILLM:
    """封装 chat / chat_stream 两个方法；base_url、api_key、model 均来自配置。"""

    def __init__(
        # 解析：构造参数
        self,
        base_url: str,
        # 解析：API 地址（DeepSeek/豆包/千问或本地 vLLM/SGLang）
        api_key: str,
        # 解析：API 密钥
        model: str,
        # 解析：模型名
        timeout: float = 60.0,
        # 解析：请求超时（秒）
        http_client=None,
        # 解析：可选的自定义 HTTP 客户端（测试注入 MockTransport，生产为 None）
    ):
        self.model = model
        # 解析：保存模型名
        self.client = AsyncOpenAI(
            # 解析：创建异步 OpenAI 兼容客户端
            base_url=base_url,
            # 解析：指定 API 地址
            api_key=api_key,
            # 解析：指定密钥
            timeout=timeout,
            # 解析：指定超时
            http_client=http_client,  # 测试时注入 MockTransport；生产为 None
            # 解析：注入自定义传输层（单测用 MockTransport 模拟服务端响应，不发真实网络请求）
        )

    async def chat(self, messages: list[dict]) -> str:
        # 解析：非流式对话
        response = await self.client.chat.completions.create(
            # 解析：调用 OpenAI 兼容的 chat/completions 接口
            model=self.model, messages=messages, stream=False
            # 解析：指定模型、消息列表、非流式
        )
        content = response.choices[0].message.content
        # 解析：取第一条候选的回复内容
        return content or ""
        # 解析：返回内容（空内容防御性返回空串）

    async def chat_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        # 解析：流式对话（异步生成器）
        stream = await self.client.chat.completions.create(
            # 解析：调用流式接口
            model=self.model, messages=messages, stream=True
            # 解析：stream=True 开启流式
        )
        async for chunk in stream:
            # 解析：逐块消费服务端推送
            if not chunk.choices:
                # 解析：无候选的空块（部分服务商首尾包）跳过
                continue
            delta = chunk.choices[0].delta.content
            # 解析：取本块的增量文本
            if delta:
                # 解析：增量非空才产出
                yield delta
                # 解析：逐块产出给上层（SSE 逐字推送）
