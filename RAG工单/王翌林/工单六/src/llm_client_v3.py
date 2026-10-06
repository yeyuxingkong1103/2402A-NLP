# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
src/llm_client_v3.py —— 工单三 LLM 客户端（DeepSeek-v4-flash）

工单二 src.llm_client 的轻量扩展：
  - 复用同一 DeepSeek 客户端（api_key/base_url/model 一致）
  - 新增 build_table_aware_prompt：表格感知 prompt 组装
  - 新增 chat_with_tables：传入文本+表格上下文，输出结构化 answer
"""
import os
import time
from typing import Any, Dict, List, Optional

from loguru import logger

# 工单三：复用工单二客户端单例
_client = None


def get_client():
    """工单三：DeepSeek 客户端单例（与工单二共享）"""
    global _client
    if _client is None:
        from openai import OpenAI
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key or api_key.startswith("your_key"):
            raise RuntimeError("请在 .env 中设置 DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        _client = OpenAI(api_key=api_key, base_url=base_url, timeout=30)
        logger.info(f"[llm_client_v3] DeepSeek client 就绪: {base_url}")
    return _client


def chat(messages, model=None, temperature=0.3, max_tokens=2048,
         disable_thinking=True, **kwargs):
    """工单三：与工单二一致的 chat 接口（默认禁用思维链降延迟）"""
    model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    client = get_client()
    # 工单四（人工智能NLP-RAG-图像内容解析及检索优化）：单调时钟计时
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"thinking": {"type": "disabled"}} if disable_thinking else None,
            **kwargs,
        )
    except Exception as e:
        if disable_thinking and ("thinking" in str(e) or "400" in str(e)
                                  or "extra_body" in str(e)):
            logger.warning(f"[llm_client_v3] thinking 不支持，回退: {str(e)[:80]}")
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                max_tokens=max_tokens, **kwargs,
            )
        else:
            raise
    latency_ms = (time.perf_counter() - t0) * 1000
    usage = resp.usage
    result = {
        "content": resp.choices[0].message.content or "",
        "latency_ms": round(latency_ms, 1),
        "token_usage": {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        },
        "model": model,
    }
    logger.info(
        f"[llm_client_v3] LLM 返回: model={model}, "
        f"latency={result['latency_ms']:.0f}ms, "
        f"tokens={result['token_usage']['total_tokens']}"
    )
    return result


# ================= 工单三：表格感知 prompt =================
RAG_V3_SYSTEM = """你是一名专业的投资分析师，擅长研读招股说明书并回答问题。
请严格遵循：
1. 仅基于提供的【参考资料】回答，不要编造资料中没有的信息。
2. 如果参考资料不足以回答，请说明"参考资料中未提及"。
3. 引用资料时标注资料编号，例如 [资料1] 或 [表1]。
4. 表格类问题优先引用表格数据，给出具体数值、单位、页码。
5. 回答要条理清晰、准确、完整。"""


def build_table_aware_prompt(
    query: str,
    text_chunks: List[Dict[str, Any]],
    table_chunks: List[Dict[str, Any]],
    max_chars: int = 8000,
) -> str:
    """工单三：组装表格感知 prompt

    上下文结构：
      【文本资料】
        [资料1] (第N页, doc_id) ...
      【表格资料】
        [表1] (table_id=tbl_003, 第N页, doc_id)
        表标题: ...
        列: ...
        行1: ...
      【问题】
        ...
    """
    parts: List[str] = []
    total = 0

    # 文本资料
    if text_chunks:
        parts.append("【文本资料】")
        for i, c in enumerate(text_chunks, 1):
            page = c.get("page", "?")
            doc = c.get("doc_id", "")
            content = (c.get("content") or c.get("text") or "").strip()
            block = f"[资料{i}] (第{page}页, {doc})\n{content}"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)

    # 表格资料（独立编号 [表1] [表2]...）
    if table_chunks:
        parts.append("")
        parts.append("【表格资料】")
        for i, t in enumerate(table_chunks, 1):
            page = t.get("page", "?")
            doc = t.get("doc_id", "")
            tid = t.get("table_id", "")
            table_text = (t.get("table_text") or t.get("content") or "").strip()
            block = (f"[表{i}] (table_id={tid}, 第{page}页, {doc})\n{table_text}")
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)

    context = "\n\n".join(parts)
    prompt = f"""{context}

【问题】
{query}

请基于以上参考资料回答问题，并在答案中引用对应的资料编号（如 [资料1] 或 [表1]）。
表格类问题请给出具体数值、单位和页码。"""
    return prompt


def simple_generate(prompt: str, system: str = RAG_V3_SYSTEM, **kwargs):
    """工单三：便捷单轮生成"""
    return chat(
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        **kwargs,
    )
