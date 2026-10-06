# -*- coding: utf-8 -*-
"""
LLM 客户端模块
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统

功能:
    1. 调用 OpenAI 兼容接口 (可对接 OpenAI / 本地 Ollama / 其他)
    2. 提供 RAG 与 纯 LLM 两种调用模式
    3. 未配置 API Key 时降级为本地规则生成 (保证可演示)
"""
import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)


def _call_openai_api(messages: list, temperature: float = 0.0,
                     max_tokens: int = 800) -> Optional[str]:
    """调用 OpenAI 兼容接口"""
    if not config.LLM_API_KEY:
        logger.warning("未配置 LLM_API_KEY, 跳过远程调用")
        return None
    try:
        import requests
    except ImportError:
        logger.error("未安装 requests")
        return None

    url = f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(url, headers=headers,
                             json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"LLM API 调用失败: {e}")
        return None


def generate_with_rag(question: str, contexts: list) -> str:
    """
    RAG 模式: 基于检索到的上下文生成答案

    Args:
        question: 用户问题
        contexts: 检索到的文本块 (字符串列表)
    """
    context_text = "\n\n".join([f"[片段{i+1}] {c}" for i, c in enumerate(contexts)])
    prompt = (
        "你是一个专业的金融文档问答助手。请根据下面提供的招股说明书片段, "
        "针对用户问题给出准确、简洁的中文回答。\n"
        "如果片段中没有相关信息, 请回答: '文档中未找到相关信息'。\n"
        "回答时请引用相关数据, 不要编造内容。\n\n"
        f"【招股说明书片段】\n{context_text}\n\n"
        f"【用户问题】\n{question}\n\n"
        "【回答】"
    )
    messages = [
        {"role": "system", "content": "你是金融文档问答助手, 仅依据提供的内容回答。"},
        {"role": "user", "content": prompt},
    ]
    answer = _call_openai_api(messages)
    if answer:
        return answer
    # 降级: 本地规则生成
    return _fallback_answer(question, contexts)


def generate_without_rag(question: str) -> str:
    """纯 LLM 模式: 不提供上下文, 用于对比基线"""
    messages = [
        {"role": "system", "content": "你是一个通用问答助手。"},
        {"role": "user", "content": f"请回答: {question}"},
    ]
    answer = _call_openai_api(messages)
    if answer:
        return answer
    return "[纯LLM模式未启用] 未配置 LLM_API_KEY, 无法调用远程 LLM; " \
           "如需对比, 请设置环境变量 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL"


def _fallback_answer(question: str, contexts: list) -> str:
    """无 LLM 时的降级回答: 直接返回最相关片段"""
    if not contexts:
        return "未检索到相关内容, 且未启用 LLM, 无法生成回答。"
    top = contexts[0]
    return (
        "[本地模式-降级回答] 检索到的最相关招股说明书片段如下:\n"
        f"{top[:300]}\n\n"
        "(注: 未配置 LLM_API_KEY, 不能生成 LLM 总结; 配置后可启用完整 RAG)"
    )


if __name__ == "__main__":
    q = "武汉兴图新科电子科技有限公司的注册资本是多少?"
    print("=== RAG 模式 ===")
    print(generate_with_rag(q, ["公司注册资本为 7,360.00 万元"]))
    print("\n=== 纯 LLM 模式 ===")
    print(generate_without_rag(q))
