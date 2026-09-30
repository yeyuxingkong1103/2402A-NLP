"""基于共享大模型的统一检索规划器。

本模块不维护实体词典、同义词表或关键词判断规则。模型只调用一次，输出供
Milvus 和 Neo4j 共同使用的结构化计划；模型不可用时，两路都直接使用原问题。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.model.llm import ModelClient, get_default_model


PLANNER_SYSTEM_PROMPT = """你是医疗 RAG 的语义检索规划器。请根据整句话的含义规划检索，
不要使用表面关键词机械分类。只输出合法 JSON，不要 Markdown、解释或思考过程。

业务 intent 只能取：
drug_info、disease_info、disease_treatment、symptom_assessment、
indicator_interpretation、treatment_advice、comparison、general_medical、non_medical。

图谱 graph_intent 只能取：
treatment_detail、disease_to_treatment、scope_to_treatment、related_condition、
category_to_treatment、evidence_search、general_search。

图谱实体 type 只能取：Treatment、Disease、RelatedCondition、ClinicalScope、
TreatmentCategory、EvidenceSource、Any。

focus 只能取：description、usage、contraindications、applicable_scope、
evidence_source、category、entity_type、related_conditions、overview。

输出格式：
{
  "intent": "disease_treatment",
  "entities": [{"name": "高血压", "type": "Disease"}],
  "vector_query": "高血压的健康风险、治疗方法、药物和非药物干预",
  "graph_query": "高血压相关治疗药物和非药物治疗",
  "graph_intent": "disease_to_treatment",
  "focus": ["overview", "usage", "contraindications"],
  "use_vector": true,
  "use_graph": true
}

要求：
1. entities 只提取用户明确提到或语义上明确指向的医疗实体，不凭空诊断。
2. vector_query 保留用户原意，可补充必要的中英文医学名称以增强语义召回。
3. graph_query 面向疾病—治疗关系图谱，简洁保留实体和关系目标。
4. 疾病、药物、治疗关系问题通常同时使用 vector 和 graph；纯文档解释可只用 vector。
5. 非医疗问题将 intent 设为 non_medical，并将两个 use 字段设为 false。
"""


@dataclass
class AIQueryPlan:
    """一次 AI 分析产生的共享检索计划。"""

    original_query: str
    intent: str = "general_medical"
    entities: list[dict[str, str]] = field(default_factory=list)
    vector_query: str = ""
    graph_query: str = ""
    graph_intent: str = "general_search"
    focus: list[str] = field(default_factory=lambda: ["overview"])
    use_vector: bool = True
    use_graph: bool = True
    fallback: bool = False
    error: str = ""

    def graph_plan(self, limit: int) -> dict[str, Any]:
        """转换成知识图谱检索器现有的计划结构。"""
        return {
            "intent": self.graph_intent,
            "entities": self.entities,
            "focus": self.focus,
            "limit": limit,
        }


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError("模型未返回 JSON") from None
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("检索计划必须是 JSON 对象")
    return value


class AIQueryPlanner:
    """调用共享模型完成语义意图识别和两路查询规划。"""

    _INTENTS = {
        "drug_info", "disease_info", "disease_treatment", "symptom_assessment",
        "indicator_interpretation", "treatment_advice", "comparison",
        "general_medical", "non_medical",
    }
    _GRAPH_INTENTS = {
        "treatment_detail", "disease_to_treatment", "scope_to_treatment",
        "related_condition", "category_to_treatment", "evidence_search",
        "general_search",
    }
    _ENTITY_TYPES = {
        "Treatment", "Disease", "RelatedCondition", "ClinicalScope",
        "TreatmentCategory", "EvidenceSource", "Any",
    }
    _FOCUS = {
        "description", "usage", "contraindications", "applicable_scope",
        "evidence_source", "category", "entity_type", "related_conditions",
        "overview",
    }

    def __init__(self, model_client: ModelClient | None = None) -> None:
        self.model_client = model_client or get_default_model()

    @staticmethod
    def fallback(query: str, error: str = "") -> AIQueryPlan:
        """模型异常时不猜实体，直接将原问题交给两路检索。"""
        q = (query or "").strip()
        return AIQueryPlan(
            original_query=q,
            vector_query=q,
            graph_query=q,
            entities=[{"name": q, "type": "Any"}] if q else [],
            fallback=True,
            error=error,
        )

    def plan(self, query: str) -> AIQueryPlan:
        q = (query or "").strip()
        if not q:
            return self.fallback(q, "问题为空")
        try:
            response = self.model_client.chat(
                system=PLANNER_SYSTEM_PROMPT,
                user=f"用户问题：{q}",
                use_thinking=False,
            )
            return self._normalize(_extract_json(response), q)
        except Exception as exc:
            return self.fallback(q, f"{type(exc).__name__}: {exc}")

    def _normalize(self, raw: dict[str, Any], query: str) -> AIQueryPlan:
        intent = str(raw.get("intent", "general_medical")).strip()
        if intent not in self._INTENTS:
            intent = "general_medical"
        graph_intent = str(raw.get("graph_intent", "general_search")).strip()
        if graph_intent not in self._GRAPH_INTENTS:
            graph_intent = "general_search"

        entities: list[dict[str, str]] = []
        for item in raw.get("entities", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            entity_type = str(item.get("type", "Any")).strip()
            if name:
                entities.append({
                    "name": name,
                    "type": entity_type if entity_type in self._ENTITY_TYPES else "Any",
                })

        focus = [
            str(value) for value in raw.get("focus", [])
            if str(value) in self._FOCUS
        ] or ["overview"]
        vector_query = str(raw.get("vector_query", "")).strip() or query
        graph_query = str(raw.get("graph_query", "")).strip() or query
        use_vector = raw.get("use_vector", True) is not False
        use_graph = raw.get("use_graph", True) is not False
        if intent == "non_medical":
            use_vector = False
            use_graph = False

        return AIQueryPlan(
            original_query=query,
            intent=intent,
            entities=entities,
            vector_query=vector_query[:1000],
            graph_query=graph_query[:1000],
            graph_intent=graph_intent,
            focus=list(dict.fromkeys(focus)),
            use_vector=use_vector,
            use_graph=use_graph,
        )


__all__ = ["AIQueryPlan", "AIQueryPlanner", "PLANNER_SYSTEM_PROMPT"]
