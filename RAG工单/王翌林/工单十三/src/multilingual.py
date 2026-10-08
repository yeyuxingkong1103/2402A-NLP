# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
src/multilingual.py —— 工单二中英文问答支持

策略（bge-m3 原生多语言，翻译路由为主、跨语言嵌入兜底）：
  1. 语言检测：中文字符占比 ≥ 20% 判 zh，否则 en
  2. 检索路由：
     - zh 问题 → 直接用 bge-m3 中文向量 + BM25 检索
     - en 问题 → LLM 翻译为中文再检索（提升 BM25 词命中与上下文相关性）；
                翻译失败/超时 → 回退 bge-m3 跨语言向量检索（不中断）
  3. 回答语言一致性：LLM system prompt 按提问语言拼接（zh 用中文 / en 用英文）
  4. 可选 lang_override：前端语言切换按钮强制指定回答语言

用法：
  from src.multilingual import bilingual_answer
  ans = bilingual_answer("What is the registered capital?", retriever)
"""
import re
from typing import Any, Dict, Optional

from loguru import logger

ZH_RATIO_THRESHOLD = 0.20  # 中文字符占比阈值（人工智能NLP-RAG-基于PDF文档的问答系统优化）

_SYSTEM_ZH = ("你是一名专业的投资分析师。请仅依据给定的参考信息回答问题，"
              "回答使用中文，并标注引用的页码。若参考信息不足以回答，请明确说明。")
_SYSTEM_EN = ("You are a professional investment analyst. Answer strictly based on the "
              "given reference context, in English, citing the page numbers. If the context "
              "is insufficient, say so explicitly.")


def detect_language(text: str) -> str:
    """工单二语言检测（人工智能NLP-RAG-基于PDF文档的问答系统优化）：
    中文字符（CJK 统一表意文字）占比 ≥ 20% → zh，否则 en"""
    if not text:
        return "zh"
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    letters = len(re.findall(r"[A-Za-z]", text))
    if zh_chars == 0:
        return "en" if letters > 0 else "zh"
    total = zh_chars + letters
    return "zh" if zh_chars / total >= ZH_RATIO_THRESHOLD else "en"


def _gen_text(prompt: str, system: str, temperature: float = 0.2, max_tokens: int = 500) -> str:
    """统一取 LLM 返回文本（工单一 simple_generate 返回 dict，人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
    from src.llm_client import simple_generate
    resp = simple_generate(prompt, system=system, temperature=temperature, max_tokens=max_tokens)
    return resp["content"] if isinstance(resp, dict) else (resp or "")


def translate_to_chinese(text: str) -> str:
    """工单二英文→中文翻译（LLM，人工智能NLP-RAG-基于PDF文档的问答系统优化）：
    失败时返回原文（由 bge-m3 跨语言向量兜底）"""
    try:
        out = _gen_text(
            f"Translate the following question into Simplified Chinese. "
            f"Output ONLY the translated question, no explanation:\n{text}",
            system="You are a translator for financial prospectus QA.",
            temperature=0.1, max_tokens=200).strip()
        # 翻译结果必须含中文才采用
        return out if re.search(r"[\u4e00-\u9fff]", out) else text
    except Exception as e:
        logger.warning(f"英文翻译失败（回退跨语言检索）: {e}")
        return text


def build_system_prompt(lang: str, lang_override: Optional[str] = None) -> str:
    """按回答语言构建 system prompt（人工智能NLP-RAG-基于PDF文档的问答系统优化）：
    lang_override 优先（前端语言切换按钮），否则与提问语言一致"""
    eff = lang_override or lang
    return _SYSTEM_ZH if eff == "zh" else _SYSTEM_EN


def bilingual_retrieve(retriever, query: str, top_k: int = 5,
                       lang_override: Optional[str] = None) -> Dict[str, Any]:
    """工单二双语检索路由（人工智能NLP-RAG-基于PDF文档的问答系统优化）：
    zh 直接检索；en 先翻译（失败回退 bge-m3 跨语言检索），返回语言与检索结果"""
    lang = lang_override if lang_override in ("zh", "en") else detect_language(query)
    search_query, translated = query, False
    if lang == "en":
        try:
            zh_query = translate_to_chinese(query)
            if zh_query != query:
                search_query, translated = zh_query, True
        except Exception as e:  # 翻译链路任何异常都不中断检索（bge-m3 跨语言兜底）
            logger.warning(f"翻译路由失败，回退跨语言检索: {e}")
    out = retriever.retrieve(search_query, top_k=top_k)
    return {"lang": lang, "original_query": query, "search_query": search_query,
            "translated": translated, "results": out["results"],
            "rewritten": out.get("rewritten", {})}


def _format_context(results: list) -> str:
    """把检索结果拼装为带页码的参考上下文（人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] (page={r.get('page')}) {r.get('content', '')[:800]}")
    return "\n\n".join(parts)


def bilingual_answer(query: str, retriever, top_k: int = 5,
                     lang_override: Optional[str] = None) -> Dict[str, Any]:
    """工单二主入口：双语检索 + 与提问语言一致的 LLM 回答（人工智能NLP-RAG-基于PDF文档的问答系统优化）"""
    r = bilingual_retrieve(retriever, query, top_k, lang_override)
    lang = r["lang"]
    system = build_system_prompt(lang, lang_override)
    ctx = _format_context(r["results"])
    if lang == "en":
        prompt = (f"Answer the question in English based on the context below.\n\n"
                  f"Question: {query}\n\nContext:\n{ctx}")
    else:
        prompt = f"请根据下面的参考信息用中文回答问题。\n\n问题：{query}\n\n参考信息：\n{ctx}"
    answer = _gen_text(prompt, system=system, temperature=0.2, max_tokens=800)
    return {"lang": lang, "original_query": query, "search_query": r["search_query"],
            "translated": r["translated"], "answer": answer, "references": r["results"]}
