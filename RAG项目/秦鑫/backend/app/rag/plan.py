from ..core import PUBLIC_COLLECTIONS
from pydantic import BaseModel, Field
from .route import route_collections, select_workflow_route


class CollectionConfig(BaseModel):
    top_k: int = 10
    enable_exact: bool = True
    enable_keyword: bool = True
    enable_vector: bool = True
    min_score: float = 0.0


class SearchPlan(BaseModel):
    query: str
    collections: list[str]
    route: str = "full_retrieval"
    include_web: bool = True
    search_private: bool = True
    search_public: bool = True
    top_k: int = 20
    original_query: str = ""
    rewritten_query: str = ""
    query_variants: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    priority_collections: list[str] = Field(default_factory=list)
    article_numbers: list[str] = Field(default_factory=list)
    legal_domains: list[str] = Field(default_factory=list)
    intent: str = "general"
    route_reason: str = ""
    wants_citation_relations: bool = False
    retrieval_notes: list[str] = Field(default_factory=list)
    collection_configs: dict[str, CollectionConfig] = Field(default_factory=dict)


CORE_COLLECTIONS = ["civil_code_articles", "civil_interpretations", "civil_elements"]
INTENT_COLLECTIONS = {
    "article_lookup": ["civil_code_articles", "civil_interpretations", "civil_elements", "civil_citations"],
    "case_reference": ["civil_cases", "civil_elements", "civil_interpretations", "civil_code_articles"],
    "evidence_strategy": ["civil_evidence", "civil_code_articles", "civil_processes", "civil_interpretations"],
    "procedure": ["civil_processes", "civil_questions", "civil_code_articles", "civil_interpretations"],
    "compensation": ["civil_code_articles", "civil_elements", "civil_cases", "civil_evidence", "civil_processes"],
    "risk_assessment": ["civil_cases", "civil_elements", "civil_evidence", "civil_code_articles"],
    "general": ["civil_code_articles", "civil_interpretations", "civil_elements", "civil_questions"],
}
DOMAIN_COLLECTIONS = {
    "marriage_family": ["civil_code_articles", "civil_interpretations", "civil_cases", "civil_evidence", "civil_elements"],
    "loan_debt": ["civil_questions", "civil_code_articles", "civil_cases", "civil_evidence", "civil_processes"],
    "labor": ["civil_processes", "civil_evidence", "civil_code_articles", "civil_interpretations", "civil_questions"],
    "contract": ["civil_code_articles", "civil_cases", "civil_evidence", "civil_processes"],
    "tort": ["civil_code_articles", "civil_cases", "civil_evidence", "civil_elements"],
    "inheritance": ["civil_code_articles", "civil_interpretations", "civil_cases", "civil_elements"],
    "property": ["civil_code_articles", "civil_cases", "civil_elements", "civil_processes"],
}

PRIORITY_TOP_K = {
    "civil_code_articles": 5,
    "civil_interpretations": 8,
    "civil_elements": 8,
    "civil_cases": 15,
    "civil_evidence": 10,
    "civil_processes": 8,
    "civil_questions": 8,
    "civil_citations": 5,
}


class SearchRouter:
    def __init__(self, settings=None):
        self.settings = settings

    def route(self, info: dict, include_web: bool = True) -> SearchPlan:
        intent = str(info.get("intent") or "general")
        domains = list(info.get("legal_domains") or ["general"])
        query = info.get("rewritten_query") or info["query"]
        workflow_route = select_workflow_route(info.get("route"))
        route_priority = route_collections(workflow_route)
        route_allowed = set(PUBLIC_COLLECTIONS) if workflow_route == "full_retrieval" else set(route_priority)
        priority = []
        notes = []

        def add(collections: list[str], reason: str) -> None:
            added = []
            for collection in collections:
                if collection in PUBLIC_COLLECTIONS and collection in route_allowed and collection not in priority:
                    priority.append(collection)
                    added.append(collection)
            if added:
                notes.append(f"{reason}: {', '.join(added)}")

        if route_priority:
            add(route_priority, f"按问答路线 {workflow_route} 优先检索")

        intent_collections = [
            collection
            for collection in INTENT_COLLECTIONS.get(intent, INTENT_COLLECTIONS["general"])
            if collection != "civil_citations"
        ]
        citation_relations = bool(info.get("wants_citation_relations"))
        if citation_relations:
            add(["civil_citations"], "按模型识别的引用关系查询补强")
        add(intent_collections, f"按问题意图 {intent} 优先检索")
        for domain in domains:
            add(DOMAIN_COLLECTIONS.get(domain, []), f"按法律领域 {domain} 补强")
        add(CORE_COLLECTIONS, "保留核心规范依据")
        top_k = int(getattr(self.settings, "retrieval_candidate_pool_max", 20) or 20)
        search_public = workflow_route != "direct"
        search_private = workflow_route not in {"direct", "law_only", "current_law_web"}
        planned_include_web = workflow_route == "current_law_web" or (workflow_route == "full_retrieval" and bool(include_web))
        if workflow_route == "direct":
            priority = []
            planned_include_web = False
            top_k = min(top_k, 4)
            notes.append("按 direct 路线跳过检索，直接生成回答")
        elif workflow_route == "current_law_web":
            notes.append("按 current_law_web 路线强制启用联网检索")
        planned_collections = list(PUBLIC_COLLECTIONS) if workflow_route == "full_retrieval" else list(route_priority)

        # ========== 新增：生成表级配置 ==========
        collection_configs = {}

        # 民法典条文：精确匹配为主，不需要向量，少取几条
        collection_configs["civil_code_articles"] = CollectionConfig(top_k=5, enable_vector=False)

        # 案例：需要多一些
        collection_configs["civil_cases"] = CollectionConfig(top_k=15)

        # 问答库：适量
        collection_configs["civil_questions"] = CollectionConfig(top_k=8)

        # 司法解释：适量
        collection_configs["civil_interpretations"] = CollectionConfig(top_k=8)

        # 民事要件：适量
        collection_configs["civil_elements"] = CollectionConfig(top_k=8)

        # 证据规则：适量
        collection_configs["civil_evidence"] = CollectionConfig(top_k=8)

        # 程序流程：适量
        collection_configs["civil_processes"] = CollectionConfig(top_k=8)

        # 引用关系：少量，精确为主
        collection_configs["civil_citations"] = CollectionConfig(top_k=5, enable_vector=False)

        # 优先集合按表类型保留取数策略，避免条文等精确表被统一抬高。
        for collection in priority:
            if collection not in collection_configs:
                collection_configs[collection] = CollectionConfig(top_k=PRIORITY_TOP_K.get(collection, 12))
            else:
                existing = collection_configs[collection]
                collection_configs[collection] = CollectionConfig(
                    top_k=max(existing.top_k, PRIORITY_TOP_K.get(collection, existing.top_k)),
                    enable_exact=existing.enable_exact,
                    enable_keyword=existing.enable_keyword,
                    enable_vector=existing.enable_vector,
                    min_score=existing.min_score,
                )

        # 确保所有 priority 中的表都在 collection_configs 中
        for collection in priority:
            if collection not in collection_configs:
                collection_configs[collection] = CollectionConfig(top_k=PRIORITY_TOP_K.get(collection, 12))
        # ========== 表级配置结束 ==========

        return SearchPlan(
            query=query,
            collections=planned_collections,
            route=workflow_route,
            include_web=planned_include_web,
            search_private=search_private,
            search_public=search_public,
            top_k=top_k,
            original_query=str(info.get("original_query") or info.get("query") or ""),
            rewritten_query=str(query),
            query_variants=list(info.get("query_variants") or [query]),
            keywords=list(info.get("keywords") or []),
            priority_collections=priority,
            article_numbers=list(info.get("article_numbers") or []),
            legal_domains=domains,
            intent=intent,
            route_reason=str(info.get("route_reason") or info.get("intent_reason") or "").strip()[:120],
            wants_citation_relations=citation_relations,
            retrieval_notes=notes,
            collection_configs=collection_configs,  # 新增
        )


# Backward-compatible lazy access for callers that imported this name from plan.
def __getattr__(name: str):
    if name == 'QueryUnderstanding':
        from .understand import QueryUnderstanding
        return QueryUnderstanding
    raise AttributeError(name)

