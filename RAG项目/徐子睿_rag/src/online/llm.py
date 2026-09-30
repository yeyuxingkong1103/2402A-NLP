"""src/online/llm.py —— 大模型调用适配层（三个后端，统一流式接口）。

在链路中的位置：
    src/online/chain.py → 【本文件】 → Ollama / OpenAI 兼容服务 / mock

为什么只实现流式、非流式靠拼接：
    complete() 就是 "".join(stream(...))。只维护一条代码路径，
    就不会出现"流式和非流式行为不一致"这类难查的问题。
    非流式的代价是要等全部生成完才返回 —— 对短答案无影响。

三个后端由 LLM_BACKEND 配置切换（这是本项目的降级设计之一）：
    mock   不发任何请求，直接返回一段固定说明。
           **让整条链路在没有大模型的机器上也能端到端跑通** ——
           注册、建角色、建会话、检索、SSE 推流全都真实执行，只有"生成"这一步是假的。
           这是 README 里"资源不足时用降级配置"的第二个开关
    ollama 本地 Ollama（qwen2:7b 等），走 /api/chat 的原生协议
    api    OpenAI 兼容的 /chat/completions（vLLM、One-API、各类云服务都兼容这个协议）

两条流式协议格式不同，解析方式也不同 —— 这是本文件存在的主要理由：
    Ollama  每行一个 JSON 对象，取 message.content，靠 done 字段判断结束
    OpenAI  每行形如 "data: {json}"，取 choices[0].delta.content，靠 "data: [DONE]" 结束
"""
from __future__ import annotations

import json
from typing import Iterable

import requests

from configs.settings import get_settings


class LLMClient:
    """按配置把生成请求路由到对应后端。"""

    def __init__(self) -> None:
        self.settings = get_settings()

    def stream(self, messages: list[dict[str, str]], temperature: float = 0.4) -> Iterable[str]:
        """流式生成，逐段产出文本增量。

        参数：
            messages: OpenAI 格式的消息列表 [{"role": ..., "content": ...}]
            temperature: 采样温度。默认 0.4 —— 角色扮演要有一点表达变化，
                         但又明显低于自由创作的水平
        返回（生成器）：
            文本增量片段；mock 后端直接产出一整段。

        异常：
            网络或 HTTP 错误直接抛出 —— 由 src/online/chain.py 捕获，
            流式接口转成 error 事件、非流式接口由上层决定如何处理。

        三个分支的注意点：
            1. mock 用 yield + return：
               它必须是生成器（因为调用方按生成器遍历），
               但只产出一次就结束，所以 yield 之后立刻 return
            2. ollama 的 stream=True + iter_lines：
               用 with 语句确保连接在生成器中途被放弃时也能正常关闭 ——
               客户端断连时生成器被垃圾回收，with 会触发 connection 释放
            3. api 分支的 line[5:] 是在切掉 "data:" 前缀（5 个字符）；
               切之前先判断 startswith 是必需的 ——
               有些服务会在流中插入空行或注释行，不判断就会解析出乱码
        """
        if self.settings.llm_backend == "mock":
            # 降级后端：不发请求。措辞刻意与真实回答的风格一致，便于前端联调时看不出差别
            yield "当前为 mock 模式：我会基于检索上下文谨慎回答，并在信息不足时说明边界。"
            return
        if self.settings.llm_backend == "ollama":
            payload = {"model": self.settings.llm_model, "messages": messages, "stream": True, "options": {"temperature": temperature}}
            with requests.post(self.settings.ollama_chat_url, json=payload, stream=True, timeout=180) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue  # Ollama 会发空行保活，跳过
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if data.get("done"):
                        return  # 收到 done 立刻结束，不依赖连接关闭
            return
        # 默认走 OpenAI 兼容协议
        url = self.settings.llm_base_url.rstrip("/") + "/chat/completions"
        # 没配 key 就不带 Authorization 头：本地部署的兼容服务通常不需要鉴权
        headers = {"Authorization": f"Bearer {self.settings.llm_api_key}"} if self.settings.llm_api_key else {}
        payload = {"model": self.settings.llm_model, "messages": messages, "stream": True, "temperature": temperature}
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=180) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                # 只处理 SSE 的 data 行，跳过空行和其他类型的行
                if not line or not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    return  # OpenAI 协议的流结束标记
                data = json.loads(raw)
                # choices 可能为空列表（如仅含用量统计的收尾帧），用 [{}] 兜底避免下标越界
                delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if delta:
                    yield delta

    def complete(self, messages: list[dict[str, str]], temperature: float = 0.4) -> str:
        """非流式生成：把流式增量拼成完整答案。

        参数：
            messages: 消息列表
            temperature: 采样温度
        返回：
            完整答案文本。

        刻意不做单独的 HTTP 调用：
            单条代码路径意味着"流式能用，非流式就一定也能用"，
            不必维护两套解析逻辑，也不会出现两者行为不一致的问题。
        """
        return "".join(self.stream(messages, temperature))


# 模块级单例：持有配置，供全项目统一调用
llm_client = LLMClient()
