"""根据问题理解结果生成一份明确、可执行的检索计划。

这里不真正访问数据库，只决定：查哪些公共集合、是否查用户资料、是否联网、每个集合
取多少条、哪些集合优先。workflow.py 随后按照 SearchPlan 并行执行检索。
"""

# PUBLIC_COLLECTIONS 是系统真实支持的公共集合白名单。
from ..core import PUBLIC_COLLECTIONS
# BaseModel 校验计划字段；Field 为列表/字典创建独立默认值。
from pydantic import BaseModel, Field
# route 模块把理解阶段路线规范成有限值，并给出路线对应的集合。
from .route import route_collections, select_workflow_route


class CollectionConfig(BaseModel):
    """一个 Milvus 公共集合的具体检索参数。"""

    # 当前集合最多取多少条候选。
    top_k: int = 10
    # 是否启用精确条号/主键查询。
    enable_exact: bool = True
    # 是否启用关键词检索。
    enable_keyword: bool = True
    # 是否启用向量相似度检索。
    enable_vector: bool = True
    # 低于该分数的结果可以被过滤；0 表示暂不过滤。
    min_score: float = 0.0


class SearchPlan(BaseModel):
    """一轮问答中所有检索通道共同遵循的结构化计划。"""

    # query 是实际用于检索的改写问题。
    query: str
    # collections 是允许搜索的全部公共集合。
    collections: list[str]
    # route 是 direct、law_only、full_retrieval 等工作流路线。
    route: str = "full_retrieval"
    # include_web 是路由计算后的最终联网决定。
    include_web: bool = True
    # retrieval_mode 保存用户选择的 local/auto/force。
    retrieval_mode: str = "auto"
    # 两个布尔值分别控制用户私有资料和公共知识库。
    search_private: bool = True
    search_public: bool = True
    # 合并前总体候选池上限。
    top_k: int = 20
    # 原问题、改写问题和多种查询表达用于日志及多路召回。
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


# 无论哪类民事问题都值得保留的三类核心规范依据。
CORE_COLLECTIONS = ["civil_code_articles", "civil_interpretations", "civil_elements"]
# 不同用户意图最可能需要的集合，前面的优先级更高。
INTENT_COLLECTIONS = {
    "article_lookup": ["civil_code_articles", "civil_interpretations", "civil_elements", "civil_citations"],
    "case_reference": ["civil_cases", "civil_elements", "civil_interpretations", "civil_code_articles"],
    "evidence_strategy": ["civil_evidence", "civil_code_articles", "civil_processes", "civil_interpretations"],
    "procedure": ["civil_processes", "civil_questions", "civil_code_articles", "civil_interpretations"],
    "compensation": ["civil_code_articles", "civil_elements", "civil_cases", "civil_evidence", "civil_processes"],
    "risk_assessment": ["civil_cases", "civil_elements", "civil_evidence", "civil_code_articles"],
    "general": ["civil_code_articles", "civil_interpretations", "civil_elements", "civil_questions"],
}
# 不同民事法律领域需要补强的资料集合。
DOMAIN_COLLECTIONS = {
    "marriage_family": ["civil_code_articles", "civil_interpretations", "civil_cases", "civil_evidence", "civil_elements"],
    "loan_debt": ["civil_questions", "civil_code_articles", "civil_cases", "civil_evidence", "civil_processes"],
    "labor": ["civil_processes", "civil_evidence", "civil_code_articles", "civil_interpretations", "civil_questions"],
    "contract": ["civil_code_articles", "civil_cases", "civil_evidence", "civil_processes"],
    "tort": ["civil_code_articles", "civil_cases", "civil_evidence", "civil_elements"],
    "inheritance": ["civil_code_articles", "civil_interpretations", "civil_cases", "civil_elements"],
    "property": ["civil_code_articles", "civil_cases", "civil_elements", "civil_processes"],
}

# 各集合自身的候选数量；案例较多，法条精确度较高所以数量较少。
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
    """把 QueryUnderstanding 返回的字典转换成 SearchPlan。"""

    def __init__(self, settings=None):
        # settings 提供全局候选池上限；测试时允许为 None。
        self.settings = settings

    def route(self, info: dict, include_web: bool = True, retrieval_mode: str | None = None) -> SearchPlan:
        """综合意图、领域、路线和联网模式生成最终检索计划。"""

        # 没识别出具体意图时使用 general。
        intent = str(info.get("intent") or "general")
        # 法律领域可能有多个；完全为空时用 general。
        domains = list(info.get("legal_domains") or ["general"])
        # 优先使用指代消解和问题改写后的表达。
        query = info.get("rewritten_query") or info["query"]
        # 将模型可能返回的路线别名限制到系统支持的标准路线。
        workflow_route = select_workflow_route(info.get("route"))
        # 取得这条路线允许检索的集合顺序。
        route_priority = route_collections(workflow_route)
        # full_retrieval 可使用所有集合；其他路线只能在路线集合内调整优先级。
        route_allowed = set(PUBLIC_COLLECTIONS) if workflow_route == "full_retrieval" else set(route_priority)
        # priority 保存最终优先集合，notes 解释每次加入的原因。
        priority = []
        notes = []

        def add(collections: list[str], reason: str) -> None:
            """按顺序添加合法且尚未出现的优先集合，并记录原因。"""

            added = []
            for collection in collections:
                # 同时满足真实存在、路线允许、尚未添加三个条件才加入。
                if collection in PUBLIC_COLLECTIONS and collection in route_allowed and collection not in priority:
                    priority.append(collection)
                    added.append(collection)
            # 只有确实新增集合时才写说明。
            if added:
                notes.append(f"{reason}: {', '.join(added)}")

        # 路线本身的集合优先级最高。
        if route_priority:
            add(route_priority, f"按问答路线 {workflow_route} 优先检索")

        # 根据意图选择集合，但引用关系集合要由专门信号开启。
        intent_collections = [
            collection
            for collection in INTENT_COLLECTIONS.get(intent, INTENT_COLLECTIONS["general"])
            if collection != "civil_citations"
        ]
        # 用户问“哪些案例引用某条法条”等关系问题时才查引用集合。
        citation_relations = bool(info.get("wants_citation_relations"))
        if citation_relations:
            add(["civil_citations"], "按模型识别的引用关系查询补强")
        # 加入意图相关集合。
        add(intent_collections, f"按问题意图 {intent} 优先检索")
        # 一个问题可涉及多个领域，逐一补强。
        for domain in domains:
            add(DOMAIN_COLLECTIONS.get(domain, []), f"按法律领域 {domain} 补强")
        # 最后尝试加入核心规范依据，仍受路线白名单约束。
        add(CORE_COLLECTIONS, "保留核心规范依据")
        # 从配置读取总体候选池上限，旧配置缺失时默认为 20。
        top_k = int(getattr(self.settings, "retrieval_candidate_pool_max", 20) or 20)
        # direct 路线不查公共库；其他路线查公共库。
        search_public = workflow_route != "direct"
        # 纯法条、时效性法律网页和 direct 路线不查用户上传资料。
        search_private = workflow_route not in {"direct", "law_only", "current_law_web"}
        # 显式合法模式优先；旧请求没有模式时由 include_web 转成 force/auto。
        mode = retrieval_mode if retrieval_mode in {"local", "auto", "force"} else ("force" if include_web else "auto")
        # local 永不联网。
        if mode == "local":
            planned_include_web = False
        # force 除 direct 外都联网。
        elif mode == "force":
            planned_include_web = workflow_route != "direct"
        else:
            # auto 只在问题明确需要现行/时效信息的 current_law_web 路线联网。
            planned_include_web = workflow_route == "current_law_web"
        # direct 跳过所有检索并缩小候选上限。
        if workflow_route == "direct":
            priority = []
            planned_include_web = False
            top_k = min(top_k, 4)
            notes.append("按 direct 路线跳过检索，直接生成回答")
        elif planned_include_web:
            notes.append(f"按 {mode} 检索模式启用联网检索")
        else:
            notes.append(f"按 {mode} 检索模式仅使用本地知识与用户材料")
        # full_retrieval 搜全部集合；其他路线只使用路线定义集合。
        planned_collections = list(PUBLIC_COLLECTIONS) if workflow_route == "full_retrieval" else list(route_priority)

        # 为每个计划集合生成独立参数；引用关系集合没有适合做语义向量的长正文。
        collection_configs = {}
        for collection in planned_collections:
            collection_configs[collection] = CollectionConfig(
                top_k=PRIORITY_TOP_K.get(collection, 8),
                enable_vector=collection != "civil_citations",
            )

        # 把全部决定封装成经过 Pydantic 校验的 SearchPlan。
        return SearchPlan(
            query=query,
            collections=planned_collections,
            route=workflow_route,
            include_web=planned_include_web,
            retrieval_mode=mode,
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
            collection_configs=collection_configs,
        )


# 旧代码曾从 plan.py 导入 QueryUnderstanding；按需转发可避免一次性循环导入。
def __getattr__(name: str):
    """只为旧的 QueryUnderstanding 导入提供惰性兼容。"""

    if name == 'QueryUnderstanding':
        from .understand import QueryUnderstanding
        return QueryUnderstanding
    # 其他未知属性保持 Python 标准行为。
    raise AttributeError(name)

