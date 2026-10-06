# -*- coding: utf-8 -*-
"""
LLM 客户端 V2 - 优化 Prompt 模板
工单编号: 人工智能 NLP-RAG-基于 PDF 文档的问答系统优化

核心改进 (对比 V1):
    1. 区分表格块与文本块的格式化方式
    2. 明确要求 LLM 先给答案, 再引用来源
    3. 强制"无答案"时诚实返回而非编造
"""
import os
import logging
from typing import List, Optional

import config_v2 as config

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
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"LLM API 调用失败: {e}")
        return None


def _format_contexts(contexts: List[str], types: List[str] = None) -> str:
    """将检索结果格式化为 LLM 友好的格式"""
    formatted_parts = []
    for i, c in enumerate(contexts):
        label = "表格数据" if types and types[i] == "table" else "文本片段"
        formatted_parts.append(f"【{label} {i+1}】\n{c}")
    return "\n\n".join(formatted_parts)


def generate_with_rag_v2(question: str, contexts: List[str],
                         context_types: List[str] = None) -> str:
    """
    V2 RAG 生成: 更好的 Prompt 模板

    Args:
        question: 用户问题
        contexts: 检索到的文本块
        context_types: 每个块的类型 ("text" / "table")
    """
    context_text = _format_contexts(contexts, context_types)

    system_prompt = (
        "你是一个专业的金融文档问答助手, 擅长从招股说明书中提取关键信息。\n"
        "规则:\n"
        "1. 仅根据提供的招股说明书片段回答, 不要编造内容。\n"
        "2. 如果所有片段都没有相关信息, 请准确回答: '文档中未找到相关信息'。\n"
        "3. 回答财务数据时请保留原始数字和单位 (万元/亿元/%)。\n"
        "4. 回答结构: 先给结论 (1-2句), 再引用数据或事实。"
    )

    user_prompt = (
        "请根据以下招股说明书片段回答用户问题。\n\n"
        f"=== 招股说明书片段 ===\n{context_text}\n\n"
        f"=== 用户问题 ===\n{question}\n\n"
        "=== 你的回答 ==="
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    answer = _call_openai_api(messages)
    if answer:
        return answer
    return _fallback_v2(question, contexts, context_types)


def generate_without_rag(question: str) -> str:
    """纯 LLM 模式, 用于对比基线"""
    messages = [
        {"role": "system", "content": "你是通用问答助手。"},
        {"role": "user", "content": f"请回答: {question}"},
    ]
    answer = _call_openai_api(messages)
    if answer:
        return answer
    return "[纯LLM模式未启用] 未配置 LLM_API_KEY"


def _fallback_v2(question: str, contexts: List[str],
                 context_types: List[str] = None) -> str:
    """无 LLM 时的降级回答 (比 V1 更结构化)"""
    if not contexts:
        return "未检索到相关内容, 且未启用 LLM。"
    # 按相关性排好序, 返回前 3 个片段
    top = contexts[:3]
    types = context_types[:3] if context_types else ["text"] * 3
    parts = []
    for i, (t, ty) in enumerate(zip(top, types)):
        label = "表格数据" if ty == "table" else "文本片段"
        parts.append(f"--- {label} {i+1} ---\n{t[:400]}")
    return (
        "[本地模式-降级回答] 未配置 LLM_API_KEY, 无法生成 LLM 总结。\n"
        "以下为系统检索到的最相关内容:\n\n"
        + "\n\n".join(parts)
        + "\n\n(配置 LLM_API_KEY 后可启用完整 RAG 生成)"
    )


if __name__ == "__main__":
    q = "武汉兴图新科电子股份有限公司的注册资本是多少?"
    ctx = ["本次发行后总股本 7,360.00 万股, 每股面值 1.00 元"]
    print("=== V2 RAG ===")
    print(generate_with_rag_v2(q, ctx, ["text"]))
    print("\n=== 纯 LLM ===")
    print(generate_without_rag(q))
