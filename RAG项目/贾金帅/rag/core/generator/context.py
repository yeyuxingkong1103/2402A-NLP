"""把统一检索结果转换为模型可消费的证据上下文。"""

from __future__ import annotations

import json

from src.core.retrieval.base.retriever_base import RetrievalResult


#  关键业务字段优先（MR002 剂量 / MR003 相互作用 / 禁忌 / 不良反应），
#  保证上下文截断时不优先丢弃这些证据。
CONTEXT_PRIORITY_FIELDS = (
    "dosage",
    "interaction",
    "contraindication",
    "adverse",
    "usage",
    "notice",
    "specification",
    "indication",
    "ingredient",
    "related_diseases",
    "category",
    "otc",
    "drug_class",
    "dosage_form",
    "insurance_class",
    "manufacturer",
)


def format_retrieval_context(
    results: list[RetrievalResult], max_chars: int = 2000
) -> str:
    """同时格式化药品实体和文档块，不丢失正文。

    ``max_chars`` 默认 2000：与生成器的 ``truncate_context(max_tokens=2000)``
    对齐（中文约 1 字 ≈ 1 token），避免两层截断上限不一致导致上下文被二次压缩。 
    """
    max_chars = max(1, int(max_chars))
    lines: list[str] = []
    used_chars = 0

    for index, result in enumerate(results, start=1):
        content = result.content
        title = content.get("title") or content.get("name") or "未知"
        parts = [f"[{index}] title={title}", f"source_type={result.source}"]

        # 向量证据的可追溯字段。
        for key in ("chunk_id", "parent_id", "document_id", "chunk_type",
                    "section_title", "section_type"):
            value = content.get(key)
            if value:
                parts.append(f"{key}={value}")

        section_path = content.get("section_path")
        if section_path:
            section = " > ".join(str(item) for item in section_path) \
                if isinstance(section_path, list) else str(section_path)
            parts.append(f"section={section}")

        document = content.get("document") or content.get("text")
        if document:
            parts.append(f"document={document}")
        if content.get("source_path"):
            parts.append(f"source_path={content['source_path']}")

        metadata = content.get("metadata")
        if metadata:
            metadata_text = json.dumps(metadata, ensure_ascii=False) \
                if isinstance(metadata, (dict, list)) else str(metadata)
            parts.append(f"metadata={metadata_text}")

        for key in CONTEXT_PRIORITY_FIELDS:
            value = content.get(key)
            if value:
                parts.append(f"{key}={value}")

        if result.reason:
            parts.append(f"(来源:{result.source} | {result.reason})")

        line = "；".join(parts)
        separator_chars = 1 if lines else 0
        remaining = max_chars - used_chars - separator_chars
        if remaining <= 0:
            break
        if len(line) > remaining:
            # 整行放不下：在最后一个「；」分隔符处截断以保持字段完整；
            # 没有安全分隔则跳过整行，避免半行截断丢关键字段（如 dosage）。
            cut = line.rfind("；", 0, remaining)
            if cut <= 0:
                continue
            line = line[:cut]
        lines.append(line)
        used_chars += separator_chars + len(line)
        if used_chars >= max_chars:
            break

    return "\n".join(lines)
