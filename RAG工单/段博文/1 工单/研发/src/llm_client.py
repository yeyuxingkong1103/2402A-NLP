# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""
大模型客户端：DeepSeek API（OpenAI 兼容协议），统一走 ChatOpenAI。

在系统中的位置：
    上游是问答接口——它把检索到的上下文文档拼成 prompt，调本模块生成回答；
    本模块只负责「取客户端 + 发请求」，不关心业务语义。

职责：
    1. get_llm：单例缓存 ChatOpenAI 客户端；
    2. generate_answer：阻塞式一次性返回，适合离线评估、批处理；
    3. stream_answer：流式逐 chunk 返回，给前端 SSE 用，用户能立刻看到
       字一个个蹦出来。

关键设计取舍：
    DeepSeek 实现了 OpenAI 协议，所以统一用 ChatOpenAI 一个类，只换
    base_url/model 就能切换厂商，不需要维护两套客户端代码。鉴权用环境
    变量里的 deepseek_api_key1，取不到时早失败（属于启动期配置错误）。
"""

from typing import List, Optional  # 类型标注

from langchain_openai import ChatOpenAI  # OpenAI 兼容接口都走这个类
from langchain_core.messages import SystemMessage, HumanMessage  # 对话消息
from langchain_core.documents import Document  # 统一文档结构

from config import (  # 配置
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
)
from logger import get_logger  # 日志

logger = get_logger(__name__)  # 本模块 logger

# LLM 单例（进程内只创建一次）
# 之所以放在模块级：FastAPI 的每次请求可能落在不同线程上，模块级变量
# 天然是「整个进程共享一份」，避免重复建客户端；温度是构造参数而不是
# 每次请求的参数，所以缓存键里不必带温度（本项目只用一个温度）
_llm: Optional[ChatOpenAI] = None

# 系统提示词：定义 LLM 的角色与回答风格
# 这段话会作为 SystemMessage 放在消息列表的第一条，约束模型始终基于
# 检索到的资料回答，避免「不知道」或者凭空发挥；找不到相关资料时也要
# 诚实承认，不能编造（防止幻觉）
SYSTEM_PROMPT = (
    "你是一个基于知识库的 PDF 问答助手。请严格依据下方【参考资料】回答"
    "用户的问题；如果参考资料不足以回答，请直接说明「知识库中未找到"
    "相关内容」，不要凭空编造。回答时尽量引用资料中的关键句，并在末尾"
    "标注出处（来源文件 + 页码）。"
)


def get_llm() -> ChatOpenAI:
    """
    获取 LLM 客户端单例

    返回：
        可直接调用的 ChatOpenAI 实例（首次调用时创建，之后复用同一个对象）。
    异常：
        环境变量里没有 API Key（deepseek_api_key1）时抛 RuntimeError，
        属于启动期配置错误，应尽早暴露而不是等到调用时才 401。
    """
    global _llm
    if _llm is None:  # 双检
        if not LLM_API_KEY:
            raise RuntimeError("未配置 LLM_API_KEY（环境变量 deepseek_api_key1）")
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        logger.info(f"LLM 已加载：{LLM_MODEL} @ {LLM_BASE_URL}")
    return _llm


def _build_context(query: str, context_docs: List[Document]) -> str:
    """
    内部工具：把检索到的文档列表拼成带编号的参考资料块，再拼上用户提问

    参数：
        query：用户原始提问。
        context_docs：检索到的 Document 列表（来自 retriever.hybrid_search）。
    返回：
        拼好的字符串，形如：
            【参考资料】
            [1] (来源: xxx.pdf, 第 3 页) ...
            [2] (来源: xxx.pdf, 第 5 页) ...

            【用户问题】
            ...
    说明：
        没有检索结果时（context_docs 为空）也会拼一个「无相关资料」的
        提示给 LLM，让它诚实回答而不是胡编；出处信息从 metadata.source
        和 metadata.page_number 取，缺失时给空值兜底。
    """
    if not context_docs:
        return "【参考资料】\n（无相关资料）\n\n【用户问题】\n" + query

    blocks = []  # 每条资料的文本块
    for i, doc in enumerate(context_docs, start=1):  # 编号从 1 开始（人类习惯）
        source = doc.metadata.get("source", "未知来源")  # 来源文件名
        page = doc.metadata.get("page_number", 0)  # 页码（0 表示未知）
        page_tag = f", 第 {page} 页" if page else ""  # 有页码才显示
        blocks.append(f"[{i}] (来源: {source}{page_tag}) {doc.page_content}")
    context_text = "\n\n".join(blocks)  # 用空行分隔每条资料，便于 LLM 区分

    return (
        f"【参考资料】\n{context_text}\n\n"
        f"【用户问题】\n{query}"
    )


def generate_answer(query: str, context_docs: List[Document]) -> str:
    """
    基于检索上下文生成回答（阻塞式）

    参数：
        query：用户原始提问。
        context_docs：检索到的 Document 列表（已按相关度排序，会被拼进
                      提示词作为参考资料）。
    返回：
        LLM 回答文本（str）；context_docs 为空时也会让 LLM 诚实回答
        「不知道」，而不是抛异常。
    说明：
        SystemMessage 必须且只能是最前面那条，放错位置模型可能不遵守
        设定；invoke 会阻塞到整段回答生成完，长回答时会等较久（流式
        场景请用 stream_answer）。
    """
    llm = get_llm()  # 获取 LLM（命中缓存时几乎零开销）
    user_message = _build_context(query, context_docs)  # 拼上下文 + 提问
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),  # 系统提示词（第一条）
        HumanMessage(content=user_message),  # 用户消息（含参考资料）
    ]
    response = llm.invoke(messages)  # 同步调用 LLM
    logger.info(f"生成回答：query='{query[:30]}'，回答长度={len(response.content)}")
    return response.content  # 返回回答文本


def stream_answer(query: str, context_docs: List[Document]):
    """
    流式生成回答：逐 chunk 返回，前端用 SSE 接收

    参数：
        query：用户原始提问。
        context_docs：检索到的 Document 列表。
    Yields：
        每个 chunk 的文本片段（str）；正常情况下把 yield 出来的内容按
        顺序拼接，就等于 generate_answer 的返回值。
    说明：
        调用方（FastAPI 路由）会把这些片段包成 SSE 事件推给前端，实现
        「打字机」效果；本函数是生成器 —— 不调用不执行，只有在 for 循环
        迭代时才会真正发起请求。
    异常与降级：
        不在这里 try/except：一旦网络中断或对端报错，异常会从生成器里
        抛出，由路由层捕获并转成 SSE 的错误事件；这样做的原因是「流已经
        发了一半」时没法再改 HTTP 状态码，只能把错误也当成流内容来传。
    """
    llm = get_llm()
    user_message = _build_context(query, context_docs)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]
    for chunk in llm.stream(messages):  # 流式生成
        if chunk.content:  # 过滤掉只带元信息的空 chunk，避免前端多出空事件
            yield chunk.content  # 逐 chunk 返回
