"""定义在线 RAG 支持的工作流路线，以及每条路线允许检索的公共集合。"""

# 延迟解析类型注解。
from __future__ import annotations

# 问题理解模型只能从这些路线中选择。
WORKFLOW_ROUTE_VALUES = [
    "direct",
    "law_only",
    "case_law",
    "evidence_strategy",
    "procedure_guide",
    "current_law_web",
    "document_review",
    "calculation",
    "full_retrieval",
]
# 集合用于快速校验路线是否合法。
VALID_WORKFLOW_ROUTES = set(WORKFLOW_ROUTE_VALUES)
# 连接成提示词中的可选值说明。
VALID_WORKFLOW_ROUTES_TEXT = "、".join(WORKFLOW_ROUTE_VALUES)

# 每条路线对应的公共知识集合：
# direct 不检索；其他路线按问题需要缩小范围；full_retrieval 使用全部主要集合。
ROUTE_COLLECTIONS = {
    "direct": [],
    "law_only": ["civil_code_articles", "civil_interpretations", "civil_elements", "civil_citations"],
    "case_law": ["civil_cases", "civil_code_articles", "civil_interpretations", "civil_elements"],
    "evidence_strategy": ["civil_evidence", "civil_cases", "civil_code_articles", "civil_processes", "civil_interpretations"],
    "procedure_guide": ["civil_processes", "civil_questions", "civil_code_articles", "civil_interpretations"],
    "current_law_web": ["civil_code_articles", "civil_interpretations", "civil_processes", "civil_questions"],
    "document_review": ["civil_code_articles", "civil_interpretations", "civil_evidence", "civil_cases", "civil_elements"],
    "calculation": ["civil_code_articles", "civil_elements", "civil_cases", "civil_evidence", "civil_processes", "civil_interpretations"],
    "full_retrieval": [
        "civil_code_articles",
        "civil_interpretations",
        "civil_cases",
        "civil_elements",
        "civil_evidence",
        "civil_processes",
        "civil_questions",
    ],
}


def normalize_workflow_route(value: object) -> str:
    """把任意模型输出规范为合法路线；非法值返回空字符串。"""

    # 统一转字符串并清理空白。
    route = str(value or "").strip()
    # 只允许白名单值通过。
    return route if route in VALID_WORKFLOW_ROUTES else ""


def select_workflow_route(model_route: object = "") -> str:
    """使用模型返回路线；缺失或非法时保守选择全量检索。"""

    # full_retrieval 会多查资料而不是错误跳过依据，适合作为法律问答降级路线。
    return normalize_workflow_route(model_route) or "full_retrieval"


def route_collections(route: str) -> list[str]:
    """返回路线对应集合的新列表，防止调用者修改全局配置。"""

    # 未知路线同样回退到 full_retrieval。
    return list(ROUTE_COLLECTIONS.get(normalize_workflow_route(route) or "full_retrieval", ROUTE_COLLECTIONS["full_retrieval"]))
