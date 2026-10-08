# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
src/multilingual_v3.py —— 工单三中英文问答支持

策略（bge-m3 原生多语言 + 翻译路由）：
  1. 语言检测：中文字符占比 ≥ 20% → zh，否则 en
  2. 检索路由：
     - zh 问题 → 直接用 bge-m3 检索（表格+文本）
     - en 问题 → LLM 翻译为中文再检索（提升 BM25 词命中）；
                翻译失败 → 回退 bge-m3 跨语言向量检索
  3. 回答语言一致性：LLM system prompt 按提问语言拼接
  4. 表格感知：中英文都能命中表格 table_text

与工单二 multilingual.py 的区别：
  - 使用 RAGEngineV3（表格+文本融合检索）
  - 支持 doc_id 过滤（多文档）
  - 表格类问题中英文均可回答
"""
import re
from typing import Any, Dict, Optional

from loguru import logger

ZH_RATIO_THRESHOLD = 0.20

_SYSTEM_ZH = (
    "你是一名专业的投资分析师。请仅依据给定的参考信息回答问题，"
    "回答使用中文，并标注引用的页码和表格编号。"
    "若参考信息不足以回答，请明确说明。"
)
_SYSTEM_EN = (
    "You are a professional investment analyst. Answer strictly based on the "
    "given reference context, in English, citing page numbers and table IDs. "
    "If the context is insufficient, say so explicitly."
)


def detect_language(text: str) -> str:
    """工单三语言检测：中文字符占比 ≥ 20% → zh，否则 en"""
    if not text:
        return "zh"
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    letters = len(re.findall(r"[A-Za-z]", text))
    if zh_chars == 0:
        return "en" if letters > 0 else "zh"
    total = zh_chars + letters
    return "zh" if zh_chars / total >= ZH_RATIO_THRESHOLD else "en"


def _gen_text(prompt: str, system: str, temperature: float = 0.2,
              max_tokens: int = 500) -> str:
    """工单三：统一取 LLM 返回文本"""
    from src.llm_client_v3 import simple_generate
    resp = simple_generate(prompt, system=system,
                           temperature=temperature, max_tokens=max_tokens)
    return resp["content"] if isinstance(resp, dict) else (resp or "")


def translate_to_chinese(text: str) -> str:
    """工单三：英文→中文翻译（LLM），失败返回原文"""
    try:
        out = _gen_text(
            f"Translate the following question into Simplified Chinese. "
            f"Output ONLY the translated question, no explanation:\n{text}",
            system="You are a translator for financial prospectus QA.",
            temperature=0.1, max_tokens=200,
        ).strip()
        return out if re.search(r"[\u4e00-\u9fff]", out) else text
    except Exception as e:
        logger.warning(f"[multilingual_v3] 翻译失败: {e}")
        return text


def build_system_prompt(lang: str,
                       lang_override: Optional[str] = None) -> str:
    """工单三：按回答语言构建 system prompt"""
    eff = lang_override or lang
    return _SYSTEM_ZH if eff == "zh" else _SYSTEM_EN


def bilingual_ask_rag(
    engine,
    query: str,
    doc_id: Optional[str] = None,
    top_k: Optional[int] = None,
    lang_override: Optional[str] = None,
) -> Dict[str, Any]:
    """工单三：中英文 RAG 问答主入口

    流程：
      1. 语言检测
      2. 英文→翻译为中文（提升检索）
      3. 调用 RAGEngineV3.ask_rag（路由+表格+文本+rerank+LLM）
      4. 英文问题→LLM 用英文回答

    Args:
        engine: RAGEngineV3 实例
        query: 用户问题（中/英）
        doc_id: 文档过滤（招股说明书1/招股说明书2）
        top_k: 检索条数
        lang_override: 强制回答语言（zh/en）

    Returns:
        {lang, original_query, search_query, translated, answer, references, ...}
    """
    lang = lang_override if lang_override in ("zh", "en") else detect_language(query)
    search_query = query
    translated = False

    # 英文→中文翻译路由（提升 BM25 命中）
    if lang == "en":
        zh_query = translate_to_chinese(query)
        if zh_query != query:
            search_query = zh_query
            translated = True
            logger.info(f"[multilingual_v3] en→zh: {query[:50]} → {zh_query[:50]}")

    # 调用工单三 RAGEngineV3（表格+文本融合检索 + LLM）
    result = engine.ask_rag(
        search_query, doc_id=doc_id, top_k=top_k,
    )

    # 英文问题→用英文重新回答（检索用中文，但回答用英文）
    if lang == "en" and translated:
        en_prompt = (
            f"Answer the question in English based on the context below.\n\n"
            f"Original Question: {query}\n"
            f"(Search was done with Chinese translation: {search_query})\n\n"
            f"Context:\n{_extract_context(result)}"
        )
        en_answer = _gen_text(
            en_prompt, system=_SYSTEM_EN,
            temperature=0.2, max_tokens=600,
        )
        result["answer"] = en_answer
        result["answer_lang"] = "en"
    else:
        result["answer_lang"] = "zh"

    result["lang"] = lang
    result["original_query"] = query
    result["search_query"] = search_query
    result["translated"] = translated
    return result


def _extract_context(rag_result: Dict[str, Any]) -> str:
    """工单三：从 RAGEngineV3 结果提取上下文文本"""
    parts = []
    for i, h in enumerate(rag_result.get("retrieved_text_chunks", []), 1):
        parts.append(f"[Text {i}] (page={h.get('page')}) "
                    f"{(h.get('content') or '')[:600]}")
    for i, t in enumerate(rag_result.get("retrieved_tables", []), 1):
        parts.append(f"[Table {i}] (page={t.get('page')}, "
                     f"table_id={t.get('table_id')}) "
                     f"{(t.get('content') or '')[:600]}")
    return "\n\n".join(parts) if parts else "(no context)"
