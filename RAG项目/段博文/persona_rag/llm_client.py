# -*- coding: utf-8 -*-
"""
大模型客户端：支持在线 API（DeepSeek/千问/Claude/GPT/Gemini/豆包/硅基流动）
和本地部署（vLLM/SGLang/xInference），统一走 OpenAI 兼容接口

在系统中的位置：
    上游是对话接口——它把 prompt_templates 拼好的系统提示词、Redis 里的短期记忆历史、
    以及带检索上下文的当前提问传进来；本模块只负责"取客户端 + 发请求"，不关心业务语义。

职责：
    1. get_llm：按 (线路, 模型, 温度) 缓存并复用 ChatOpenAI 客户端；
    2. chat：阻塞式一次性返回，适合离线评估、批处理；
    3. stream_chat：流式逐 chunk 返回，给前端 SSE 用，用户能立刻看到字一个个蹦出来。

关键设计取舍：
    在线 DeepSeek 与本地部署（vLLM/SGLang 等）都实现了 OpenAI 协议，所以统一用 ChatOpenAI 一个类，
    只换 base_url/model 就能切换线路，不需要维护两套客户端代码。
    本地部署没有鉴权，api_key 随便填占位即可（但必须传，SDK 会做非空校验）。
"""

import os  # 环境变量（读取 deepseek_api_key1 等配置）
from typing import Optional  # 类型标注

from langchain_openai import ChatOpenAI  # 所有 OpenAI 兼容接口都走这个类
from langchain_core.messages import SystemMessage, HumanMessage  # 对话消息

from config import (  # 配置
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    LOCAL_LLM_BASE_URL, LOCAL_LLM_MODEL,
)
from logger import get_logger  # 日志

logger = get_logger(__name__)  # 本模块 logger

# LLM 单例：按 (base_url, model) 缓存，切换模型时不复用旧实例
# 键里带上 temperature 的原因：温度是构造参数而不是每次请求的参数，
# 不同角色温度不同，若不带温度做键，后创建的角色会复用前一个角色的温度、导致人格漂移
_llm_instances: dict = {}


def get_llm(temperature: float = None, use_local: bool = False) -> ChatOpenAI:
    """
    获取 LLM 客户端单例

    Args:
        temperature: 温度（不传用配置默认值；角色专属温度可通过 role 表覆盖）
                     取值范围一般 0~1：越小越稳定、可复现，越大越发散、越有创造力
        use_local: True=用本地部署模型；False=用在线 API（默认）
                   另外：只要配置里配了 LOCAL_LLM_BASE_URL，即使 use_local=False 也会走本地，
                   即"配了本地地址就默认用本地"，方便一键切换内网部署
    Returns:
        可直接调用的 ChatOpenAI 实例（同一组 线路/模型/温度 会命中缓存、返回同一个对象）
    异常:
        走在线线路但环境变量里没有 API Key（deepseek_api_key1）时抛 RuntimeError，
        属于启动期配置错误，应尽早暴露而不是等到调用时才 401
    """
    if use_local or LOCAL_LLM_BASE_URL:  # 用本地部署（vLLM/SGLang 等，均实现 OpenAI 兼容协议）
        key = ("local", LOCAL_LLM_MODEL, temperature or LLM_TEMPERATURE)  # 用 "local" 前缀区分线路，避免和在线同名模型撞缓存
        if key not in _llm_instances:
            _llm_instances[key] = ChatOpenAI(
                model=LOCAL_LLM_MODEL,
                api_key="not-needed",  # 本地部署不需要 key，占位即可（SDK 要求非空，传 None 会报错）
                base_url=LOCAL_LLM_BASE_URL,
                temperature=temperature or LLM_TEMPERATURE,  # 注意用 or：temperature=0 也会走默认值（0 是合法温度，这里属于已知小坑）
                max_tokens=LLM_MAX_TOKENS,  # 限制单次输出长度，防止跑飞或耗光显存
            )
            logger.info(f"本地 LLM 已加载：{LOCAL_LLM_MODEL} @ {LOCAL_LLM_BASE_URL}")
        return _llm_instances[key]
    else:  # 在线 API（DeepSeek，key 取自环境变量 deepseek_api_key1）
        key = ("online", LLM_MODEL, temperature or LLM_TEMPERATURE)
        if key not in _llm_instances:
            if not LLM_API_KEY:
                raise RuntimeError("未配置 LLM_API_KEY（环境变量 deepseek_api_key1）")
            _llm_instances[key] = ChatOpenAI(
                model=LLM_MODEL,
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL,
                temperature=temperature or LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            logger.info(f"在线 LLM 已加载：{LLM_MODEL} @ {LLM_BASE_URL}")
        return _llm_instances[key]


def chat(system_prompt: str, history: list, user_message: str,
         temperature: float = None, use_local: bool = False) -> str:
    """
    多轮对话：系统提示词 + 历史消息 + 当前用户消息 → LLM 回答

    Args:
        system_prompt: 角色系统提示词（定义人格、工作原则；来自 prompt_templates）
        history:      历史消息列表 [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
                      通常由 Redis 里的短期记忆读出（已截断过的最近若干轮），role 只认 user/assistant
        user_message: 当前用户消息（含检索上下文，已由 build_context 拼好：参考资料 + 原始提问）
        temperature:  温度（None 表示用配置默认值；角色专属温度由调用方按角色表传入）
        use_local:    是否用本地部署模型（False 时若配了 LOCAL_LLM_BASE_URL 仍会走本地）
    Returns:
        LLM 回答文本（str）；注意这里返回的是完整答案，不做后处理，清洗交给 post_processor
    """
    llm = get_llm(temperature=temperature, use_local=use_local)  # 获取 LLM（命中缓存时几乎零开销）
    # 1. 先放系统提示词：SystemMessage 必须且只能是最前面那条，放错位置模型可能不遵守人格设定
    messages = [SystemMessage(content=system_prompt)]  # 系统提示词（第一条）
    # 把历史消息加上
    # 2. 按顺序追加历史消息，保持和实际对话相同的时间顺序（乱序会让模型以为对话跳跃）
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))  # 用户消息
        elif msg["role"] == "assistant":
            from langchain_core.messages import AIMessage  # 函数内导入：只有多轮对话才需要，省一次导入开销
            messages.append(AIMessage(content=msg["content"]))  # 角色回答
    # 3. 追加当前提问（带参考资料），历史里不放参考资料是为了省 token，旧资料往往已经没用了
    messages.append(HumanMessage(content=user_message))  # 当前消息
    # 4. 同步调用：invoke 会阻塞到整段回答生成完，长回答时请求会挂较久（流式场景请用 stream_chat）
    response = llm.invoke(messages)  # 调用 LLM（同步阻塞）
    return response.content  # 返回回答文本（AIMessage.content 就是纯文本，不含元信息）


def stream_chat(system_prompt: str, history: list, user_message: str,
                temperature: float = None, use_local: bool = False):
    """
    流式对话：逐 token 返回，前端用 SSE 接收

    Args:
        system_prompt: 角色系统提示词（同 chat）
        history:      历史消息列表，格式与 chat 一致
        user_message: 当前用户消息（含检索上下文）
        temperature:  温度（None 用默认值）
        use_local:    是否用本地部署模型
    Yields:
        每个 chunk 的文本片段（str）；正常情况下把 yield 出来的内容按顺序拼接，就等于 chat 的返回值
    说明:
        调用方（FastAPI 路由）会把这些片段包成 SSE 事件推给前端，实现"打字机"效果；
        本函数是生成器 —— 不调用不执行，只有在 for 循环迭代时才会真正发起请求
    异常与降级:
        不在这里 try/except：一旦网络中断或对端报错，异常会从生成器里抛出，
        由路由层捕获并转成 SSE 的错误事件；这样做的原因是"流已经发了一半"时没法再改 HTTP 状态码，
        只能把错误也当成流内容来传，避免前端一直傻等
    """
    llm = get_llm(temperature=temperature, use_local=use_local)  # 获取 LLM（与 chat 共用同一套缓存，不会重复建连接）
    # 1. 消息顺序与 chat 完全一致：系统提示词 -> 历史 -> 当前提问
    messages = [SystemMessage(content=system_prompt)]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            from langchain_core.messages import AIMessage
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_message))
    # 2. stream 返回迭代器：每拿到一个 chunk 就立刻 yield，调用方才能边收边展示
    for chunk in llm.stream(messages):  # 流式生成
        if chunk.content:  # 有些 chunk 只带元信息（如 usage/结束标记）内容为空串，过滤掉避免前端多出空事件
            yield chunk.content  # 逐 chunk 返回
