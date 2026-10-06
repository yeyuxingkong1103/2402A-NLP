"""Workflow route classification for the RAG pipeline."""

from __future__ import annotations

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
VALID_WORKFLOW_ROUTES = set(WORKFLOW_ROUTE_VALUES)
VALID_WORKFLOW_ROUTES_TEXT = "、".join(WORKFLOW_ROUTE_VALUES)

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
    route = str(value or "").strip()
    return route if route in VALID_WORKFLOW_ROUTES else ""


def select_workflow_route(model_route: object = "") -> str:
    """Use the route returned by the LLM; invalid or missing output stays neutral."""
    return normalize_workflow_route(model_route) or "full_retrieval"


def route_collections(route: str) -> list[str]:
    return list(ROUTE_COLLECTIONS.get(normalize_workflow_route(route) or "full_retrieval", ROUTE_COLLECTIONS["full_retrieval"]))
