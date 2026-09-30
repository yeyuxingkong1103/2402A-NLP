# -*- coding: utf-8 -*-
"""图谱检索的类型契约与 plan 归一化。

自 ``graph_retrieval.py`` 拆出。这里只有数据结构与纯函数，
不连库、不调模型，便于单独阅读和测试。
"""
from __future__ import annotations

import json
import re
from typing import Any, Literal

from typing_extensions import NotRequired, TypedDict


MAX_RESULT_LIMIT = 20


class Entity(TypedDict):
    name: str
    type: Literal[
        "Treatment", "Disease", "RelatedCondition", "ClinicalScope",
        "TreatmentCategory", "EvidenceSource", "Any",
    ]


class QueryPlan(TypedDict):
    intent: Literal[
        "treatment_detail", "disease_to_treatment", "scope_to_treatment",
        "related_condition", "category_to_treatment", "evidence_search",
        "general_search",
    ]
    entities: list[Entity]
    focus: list[str]
    limit: int


class RetrievalState(TypedDict):
    question: str
    requested_limit: NotRequired[int]
    plan: NotRequired[QueryPlan]
    records: NotRequired[list[dict[str, Any]]]
    context: NotRequired[str]
    answer: NotRequired[str]
    error: NotRequired[str]


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence = chr(96) * 3
    if cleaned.startswith(fence):
        cleaned = re.sub(
            "^" + re.escape(fence) + r"(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
    if cleaned.endswith(fence):
        cleaned = cleaned[: -len(fence)].rstrip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError(f"LLM did not return valid JSON: {text}") from None
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Query plan must be a JSON object")
    return value


def _fallback_entity(question: str) -> str:
    value = question.strip()
    patterns = (
        r"(有哪些|有什么|哪些|什么|怎么|如何|可以|是否|能否)",
        r"(治疗药品|治疗药物|药品|药物|食药物质|非药物治疗|治疗措施)",
        r"(适用范围|适用证型|适用|用法用量|用法|怎么用|如何使用)",
        r"(禁忌症|禁忌|不能用|不宜使用|注意事项)",
        r"(相关疾病|并发症|合并症|相关状况|证据来源|来源)",
        r"(查询|查找|列出|介绍|说明|详情|信息)",
        r"[？?。！!，,：:]",
    )
    for pattern in patterns:
        value = re.sub(pattern, "", value)
    return value.strip() or question.strip()


def _normalize_plan(
    raw: dict[str, Any], fallback: str, requested_limit: int
) -> QueryPlan:
    allowed_intents = {
        "treatment_detail", "disease_to_treatment", "scope_to_treatment",
        "related_condition", "category_to_treatment", "evidence_search",
        "general_search",
    }
    allowed_types = {
        "Treatment", "Disease", "RelatedCondition", "ClinicalScope",
        "TreatmentCategory", "EvidenceSource", "Any",
    }
    allowed_focus = {
        "description", "usage", "contraindications", "applicable_scope",
        "evidence_source", "category", "entity_type", "related_conditions",
        "overview",
    }
    intent = str(raw.get("intent", "general_search"))
    if intent not in allowed_intents:
        intent = "general_search"

    entities: list[Entity] = []
    for item in raw.get("entities", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        entity_type = str(item.get("type", "Any"))
        if name:
            entities.append({
                "name": name,
                "type": entity_type if entity_type in allowed_types else "Any",
            })
    if not entities:
        entities = [{"name": _fallback_entity(fallback), "type": "Any"}]

    focus = [
        str(item) for item in raw.get("focus", [])
        if str(item) in allowed_focus
    ] or ["overview"]
    try:
        model_limit = int(raw.get("limit", requested_limit))
    except (TypeError, ValueError):
        model_limit = requested_limit
    return {
        "intent": intent,
        "entities": entities,
        "focus": list(dict.fromkeys(focus)),
        "limit": max(1, min(model_limit, requested_limit, MAX_RESULT_LIMIT)),
    }  # type: ignore[return-value]
