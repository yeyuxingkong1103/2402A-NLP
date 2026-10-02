"""执行公共库、用户资料和网页检索，再融合、重排并筛选最终证据。

公共库支持明确条号查询、关键词检索和 BGE-M3 向量检索；多路候选先用 RRF 融合，
再用 BGE-reranker-v2-m3 重排。精确法条和用户材料有业务优先级，但分数不代表法院
结论或法律正确率，只用于本次候选排序。
"""

# logging 记录单路检索、嵌入或重排降级。
import logging
# math 检查向量和分数是否为有限数字。
import math
# re 清理标题、网页和提取条号相关文本。
import re
# ThreadPoolExecutor 并行查询多个 Milvus 集合。
from concurrent.futures import ThreadPoolExecutor

# httpx 调用 Tavily 网页检索接口。
import httpx

# Settings 提供候选数、接口地址和超时等配置。
from ..config import Settings
# ModelGateway 提供问题向量和重排模型。
from ..models import ModelGateway
# MilvusStore 执行公共向量/关键词/条件查询；milvus_string 转义过滤字符串。
from ..storage.vector import MilvusStore, milvus_string
# WorkspaceService 检索当前用户和会话上传的私有材料。
from ..workspace import WorkspaceService
# evidence_excerpt 裁剪非规范证据并明确标注节选。
from .context import evidence_excerpt
# CollectionConfig 和 SearchPlan 来自路由阶段。
from .plan import CollectionConfig, SearchPlan

# 独立日志器便于观察检索质量和降级。
logger = logging.getLogger("law_rag.search")
# 三类核心公共规范集合。
CORE_PUBLIC_COLLECTIONS = {"civil_code_articles", "civil_interpretations", "civil_elements"}
# 这两种来源都视为用户私有材料。
PRIVATE_SOURCE_TYPES = {"private", "user_upload"}
# 普通公共证据重排分低于 0.55 时不进入最终回答依据。
MIN_PUBLIC_RELEVANCE_SCORE = 0.55
# RRF 中不同通道的业务权重：精确条号和私有材料更高。
DEFAULT_CHANNEL_WEIGHTS = {"exact": 2.4, "private": 2.0, "keyword": 1.35, "vector": 1.0, "web": 0.7}
# 通道优先级用于其他排序或展示。
CHANNEL_PRIORITY = {"exact": 5, "private": 4, "keyword": 3, "vector": 2, "web": 1}


def source_key(row: dict) -> str:
    """为去重选择稳定来源键：ID 优先，其次标题，最后正文前 80 字。"""

    return str(row.get("source_id") or row.get("title") or row.get("content", "")[:80])


def retrieval_channel(row: dict) -> str:
    """取得候选来自精确、关键词、向量、私有或网页哪个通道。"""

    return str(row.get("retrieval_channel") or row.get("source_type") or "unknown")


def number(value) -> float:
    """异常分数按0处理，不让NaN或字符串破坏排序。"""

    try:
        result = float(value or 0)
        # NaN/无穷大都转为 0。
        return result if math.isfinite(result) else 0.0
    except (TypeError, ValueError):
        return 0.0


def relevance_score(row: dict) -> float:
    """优先使用重排分；没有重排时使用原召回分。"""

    # 一旦有重排分，不能再让较高向量分覆盖重排模型的不相关判断。
    return number(row.get("rerank_score", row.get("score", 0)))


def is_exact_match(row: dict) -> bool:
    """判断记录是否来自明确法条号/主键精确查询。"""

    return retrieval_channel(row) == "exact" or row.get("retrieval_reason") == "article_exact"


def source_priority(row: dict) -> int:
    """给精确法条、精确关系、私有材料和优先集合分配业务优先级。"""

    # 直接命中的民法典条文最高。
    if row.get("direct_article_match"):
        return 6
    # 其他精确命中次之。
    if is_exact_match(row):
        return 5
    # 用户本人的材料优先于普通公共相似内容。
    if row.get("source_type") in PRIVATE_SOURCE_TYPES:
        return 4
    # 路由器指定的重点集合为 2，其他为 1。
    return 2 if row.get("priority_collection") else 1


def setting_value(settings, name: str, default: int) -> int:
    """安全读取正整数配置，非法值回退默认值。"""

    try:
        return max(1, int(getattr(settings, name, default)))
    except (TypeError, ValueError):
        return default


def unique_texts(values: list) -> list[str]:
    """把值清理成非空字符串并按首次出现顺序去重。"""

    return list(dict.fromkeys(str(value).strip() for value in values if str(value or "").strip()))


def document_title(value: str) -> str:
    """删除书名号和所有空白，用于严格比较司法解释名称。"""

    return re.sub(r"[\s《》〈〉]", "", str(value))


class VectorRetriever:
    """正式公共知识统一从Milvus读取；本地JSON不参与在线抢答。"""

    def __init__(self, model: ModelGateway, milvus: MilvusStore, settings: Settings | None = None):
        # model 负责问题向量，milvus 负责数据库检索。
        self.model = model
        self.milvus = milvus
        self.settings = settings

    @staticmethod
    def article_filter(collection: str, article_number: str) -> str:
        """针对不同集合，把一个明确条号转换成 Milvus 条件表达式。"""

        value = str(article_number).strip()
        # 只接受纯数字，防止非法表达式或注入。
        if not re.fullmatch(r"[0-9]+", value):
            return ""
        # int 再转 str 去掉无意义前导零。
        number = str(int(value))
        # 每个集合保存条号的字段和类型不同，因此分别生成过滤条件。
        if collection == "civil_code_articles":
            return f'id == "civil_code_articles_article_{number}"'
        if collection == "civil_interpretations":
            return f'array_contains(related_article_numbers_text, "{number}")'
        if collection == "civil_elements":
            return f'serial_number == "{number}"'
        if collection in {"civil_cases", "civil_evidence", "civil_processes", "civil_questions"}:
            return f"array_contains(related_article_numbers, {number})"
        if collection == "civil_citations":
            return f"article_number == {number}"
        return ""

    def _query(self, collection: str, channel: str, function, *args, **kwargs) -> tuple[list[dict], bool]:
        """单路故障允许其他路继续；日志保留故障位置。"""

        try:
            # function 是 MilvusStore 的条件、关键词或向量查询方法。
            return function(collection, *args, **kwargs), True
        except Exception:
            # 返回空结果和 False，调用者可继续其他集合但知道这一通道失败。
            logger.warning("公共库%s检索失败：%s", channel, collection, exc_info=True)
            return [], False

    @staticmethod
    def _tag(rows: list[dict], collection: str, channel: str, priority: set, min_score: float) -> list[dict]:
        """过滤低分结果并补充检索通道、原因和优先集合标记。"""

        result = []
        for row in rows:
            # 原始分数低于当前集合阈值时跳过。
            if number(row.get("score")) < min_score:
                continue
            # 创建新字典，不修改 MilvusStore 返回的原对象。
            item = {**row, "retrieval_channel": channel, "retrieval_reason": channel,
                    "priority_collection": collection in priority, "matched_channels": [channel]}
            # 精确通道进一步区分直接法条和引用关系命中。
            if channel == "exact":
                item["retrieval_reason"] = "article_exact"
                item["direct_article_match"] = collection == "civil_code_articles"
                item["citation_query_match"] = collection == "civil_citations"
            result.append(item)
        return result

    @staticmethod
    def merge_results(result_sets: list[list[dict]]) -> list[dict]:
        """按来源键合并重复命中，保留更强记录并汇总 matched_channels。"""

        # merged 的键是稳定来源 ID。
        merged = {}
        for rows in result_sets:
            for row in rows:
                key = source_key(row)
                # 首次出现直接复制。
                if key not in merged:
                    merged[key] = dict(row)
                    continue
                current = merged[key]
                # 合并该来源从哪些通道被找到。
                channels = unique_texts([*current.get("matched_channels", []), *row.get("matched_channels", [])])
                # 精确命中优先；否则保留相关分更高版本。
                if is_exact_match(row) or (not is_exact_match(current) and relevance_score(row) > relevance_score(current)):
                    merged[key] = {**current, **row}
                merged[key]["matched_channels"] = channels
        # 先按业务优先级、再按相关分降序。
        return sorted(merged.values(), key=lambda row: (source_priority(row), relevance_score(row)), reverse=True)

    def _vectors(self, variants: list[str]) -> dict[str, list[float]]:
        """批量生成查询向量，并严格检查数量、维度和有限数值。"""

        vectors = self.model.embed(variants)
        # BGE-M3 正常为 1024 维，允许配置覆盖。
        dimension = setting_value(self.settings, "embedding_dim", 1024)
        # 每个查询变体必须对应一个向量。
        if len(vectors) != len(variants):
            raise ValueError("问题向量数量与输入数量不一致")
        for vector in vectors:
            if len(vector) != dimension or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in vector
            ):
                raise ValueError(f"问题向量必须是{dimension}维有限数字")
        # 将查询文字映射到对应向量，后续按变体取用。
        return dict(zip(variants, vectors, strict=True))

    def search_public(
        self, query: str, collections: list[str], limit: int | None = None,
        query_variants: list[str] | None = None, priority_collections: list[str] | None = None,
        article_numbers: list[str] | None = None, keywords: list[str] | None = None,
        wants_citation_relations: bool = False, collection_configs: dict[str, CollectionConfig] | None = None,
    ) -> list[dict]:
        """先处理明确条号；普通问题用关键词和向量查找，最多并发查4个集合。"""

        # priority 用集合实现快速判断当前结果是否来自重点集合。
        priority = set(priority_collections or [])
        # 先放重点集合，再放全部计划集合，并保持顺序去重。
        ordered = unique_texts([*(priority_collections or []), *collections])
        # 防止 priority 中意外出现计划外集合。
        ordered = [name for name in ordered if name in collections]
        # 用户没有询问引用关系时不查 civil_citations，减少无关候选。
        if not wants_citation_relations:
            ordered = [name for name in ordered if name != "civil_citations"]
        # 没有合法集合时直接返回。
        if not ordered:
            return []
        # 每集合配置可能控制 exact/keyword/vector 和 top_k。
        configs = collection_configs or {}
        # candidate_limit 是整个公共检索最终候选池上限。
        candidate_limit = setting_value(self.settings, "retrieval_candidate_pool_max", 32)
        # results 保存不同集合和通道的批次列表。
        results = []
        # succeeded 用于区分“确实没结果”和“所有数据库查询都失败”。
        succeeded = False

        # ---------- 路线一：明确条号精确检索 ----------
        # 条号查询无需生成问题向量；没有匹配时也不能用同号其他法律冒充。
        if article_numbers:
            # 先提取书名号中的法律/司法解释名称。
            titles = re.findall(r"《([^《》]+)》", query)
            # “民法典”不算其他文件名，其余标题用于查询具体司法解释。
            named_titles = [title for title in titles if title not in {"民法典", "中华人民共和国民法典"}]
            document = named_titles[0] if named_titles else ""
            # 检查“第×条”之前是否出现司法解释/条例等名称特征。
            prefix = re.split(r"第\s*[一二三四五六七八九十百千万零〇两0-9]+\s*条", query, maxsplit=1)[0].rstrip("》 ")
            # 用户说“某规定第5条”但名称没解析清时，宁可无结果也不猜成民法典第5条。
            if not document and ("民法典" not in query
                                 or re.search(r"(?:司法解释|(?:民法典|编|关于).*(?:解释|规定|条例|办法))", prefix)):
                return []
            # 有具体文件名就查司法解释自身条号，否则默认查民法典条文。
            exact_names = ["civil_interpretations"] if document else ["civil_code_articles"]
            # 明确要求引用关系时，可在全部路线集合中按条号精确查。
            if wants_citation_relations and not document:
                exact_names = ordered
            # 仍限制在计划允许集合内。
            exact_names = [name for name in exact_names if name in ordered]
            if not exact_names:
                return []
            # 逐集合、逐条号执行条件查询。
            for name in exact_names:
                config = configs.get(name, CollectionConfig())
                # 某集合配置关闭精确查询时跳过。
                if not config.enable_exact:
                    continue
                for article in unique_texts(article_numbers):
                    # 生成当前集合字段对应的 Milvus 表达式。
                    expression = self.article_filter(name, article)
                    if not expression:
                        continue
                    # 指定司法解释名称时，同时限制自身条号和 law_name。
                    if document:
                        own_number = str(int(article))
                        expression = (f'id like "civil_interpretations_%_{own_number}_%" and '
                                      f'law_name like "%{milvus_string(document)}%"')
                    # 精确查询 top_k 较小，并给直接法条固定高分。
                    rows, ok = self._query(name, "exact", self.milvus.query_public, expression,
                                           min(config.top_k, setting_value(self.settings, "retrieval_exact_top_k", 4)),
                                           score=1.0 if name == "civil_code_articles" else 0.95)
                    succeeded = succeeded or ok
                    if document:
                        # 再严格核对名称和 ID 中自身条号，避免 like 模糊查询串到其他文件。
                        rows = [row for row in rows if str(row.get("id", "")).split("_")[-2:-1] == [own_number]
                                and document_title(document) == document_title(row.get("law_name", ""))]
                        # 展示时补上用户查的司法解释自身条号。
                        rows = [{**row, "article_number": own_number} for row in rows]
                    # 给结果补充 exact 通道标记。
                    results.append(self._tag(rows, name, "exact", priority, config.min_score))
            # 所有条件查询都异常时明确报服务失败；查询成功但没命中则正常返回空。
            if not succeeded:
                raise RuntimeError("公共法律库检索失败，暂时无法核对指定条文")
            # 合并重复命中并限制总候选数量。
            return self.merge_results(results)[:candidate_limit]

        # ---------- 路线二：普通问题关键词 + 向量检索 ----------
        # 最多使用原问题和一种改写，避免同一问题反复查库。
        variants = unique_texts([query, *(query_variants or [])])
        variants = variants[:min(2, setting_value(self.settings, "retrieval_max_query_variants", 2))]
        # 关键词最多取 8 个。
        terms = unique_texts(keywords or [])[:8]
        # 没有查询，或所有查询过短且没有关键词时，不做容易产生噪声的检索。
        if not variants or (all(len(value) <= 2 for value in variants) and not terms):
            return []
        # 限制一次问题最多查询多少集合。
        names = ordered[:setting_value(self.settings, "retrieval_max_collections", 6)]
        # vectors 只在至少一个集合启用向量检索时生成。
        vectors = {}
        if any(configs.get(name, CollectionConfig()).enable_vector for name in names):
            try:
                vectors = self._vectors(variants)
            except ValueError:
                # 错误维度或 NaN 属于数据契约问题，不能发送给 Milvus。
                raise
            except Exception:
                # 模型服务临时失败时仍保留关键词检索路线。
                logger.warning("问题向量生成失败，尝试保留关键词检索", exc_info=True)

        # 每个集合执行相同的关键词/向量组合；这个内层函数会被线程池并发调用。
        def search_collection(name: str):
            """检索一个公共集合，并返回各通道批次和是否至少一路成功。"""

            config = configs.get(name, CollectionConfig())
            # batches 分开保留关键词和各查询变体向量结果。
            batches = []
            ok_any = False
            # 有关键词且该集合开启关键词检索时执行。
            if terms and config.enable_keyword:
                top_k = min(4, config.top_k, setting_value(self.settings, "retrieval_keyword_top_k", 4))
                rows, ok = self._query(name, "keyword", self.milvus.search_public_keyword, terms, top_k)
                ok_any = ok_any or ok
                batches.append(self._tag(rows, name, "keyword", priority, config.min_score))
            # 该集合允许向量检索时执行语义召回。
            if config.enable_vector:
                # 优先集合和普通集合使用不同配置 top_k。
                key = "retrieval_priority_top_k" if name in priority else "retrieval_non_priority_top_k"
                top_k = min(config.top_k, limit or setting_value(self.settings, key, 4 if name in priority else 1))
                # 优先集合可以使用两个查询变体，普通集合只用第一个控制成本。
                selected = variants if name in priority else variants[:1]
                for variant in selected:
                    # 嵌入失败时这个变体没有向量，跳过但保留关键词结果。
                    if variant not in vectors:
                        continue
                    rows, ok = self._query(name, "vector", self.milvus.search_public, vectors[variant], top_k)
                    ok_any = ok_any or ok
                    # 向量结果至少需要 0.25 分或集合更高阈值。
                    batches.append(self._tag(rows, name, "vector", priority, max(0.25, config.min_score)))
            return batches, ok_any

        # 最多四个集合同时检索，避免数据库瞬时请求过多。
        with ThreadPoolExecutor(max_workers=min(4, len(names))) as executor:
            # executor.map 保持 names 顺序返回每个集合结果。
            for batches, ok in executor.map(search_collection, names):
                results.extend(batches)
                succeeded = succeeded or ok
        # 没有任何数据库通道正常完成时报告服务失败，而不是伪装成无依据。
        if not succeeded:
            raise RuntimeError("公共法律库检索失败，请稍后重试")
        # 跨集合/通道去重并限制总候选池。
        return self.merge_results(results)[:candidate_limit]


def csv_values(value: str) -> list[str]:
    """把逗号分隔的域名配置转成去空列表。"""

    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def clean_web_text(text: str, limit: int = 3000) -> str:
    """删除网页 Markdown 图片、链接和 URL，并限制送给模型的长度。"""

    # 第一条正则删除 Markdown 图片和裸 URL。
    value = re.sub(r"!\[[^\]]*\]\([^)]*\)|https?://\S+", " ", str(text or ""))
    # 第二条删除普通 Markdown 链接标记。
    value = re.sub(r"\[[^\]]{0,30}\]\([^)]*\)", " ", value)
    # 合并空白并最多保留 limit 字符。
    return re.sub(r"\s+", " ", value).strip()[:limit]


class WebRetriever:
    """保留已有可选联网入口；本轮不扩展联网功能。"""

    def __init__(self, settings: Settings):
        # settings 包含 Tavily 密钥、域名白名单、数量和超时。
        self.settings = settings
        # last_error 保存最近联网失败原因，供状态或日志查看。
        self.last_error = ""

    def search(self, query: str, limit: int | None = None) -> list[dict]:
        """调用 Tavily 搜索网页并转换成统一证据候选。"""

        # 未配置密钥代表未启用联网，正常返回空列表。
        if not self.settings.tavily_api_key:
            return []
        # 构造基础请求，要求返回 raw_content 供模型引用。
        payload = {"api_key": self.settings.tavily_api_key, "query": query,
                   "max_results": limit or self.settings.tavily_max_results,
                   "search_depth": self.settings.tavily_search_depth, "include_raw_content": True}
        # 可选添加允许域名和禁止域名列表。
        for field, value in (("include_domains", self.settings.tavily_allowed_domains),
                             ("exclude_domains", self.settings.tavily_blocked_domains)):
            if value:
                payload[field] = csv_values(value)
        # 网络、状态码和响应结构都可能失败，因此捕获后降级为空网页结果。
        try:
            # 发送 JSON POST 请求。
            response = httpx.post(f"{self.settings.tavily_base_url.rstrip('/')}/search",
                                  json=payload, timeout=self.settings.tavily_timeout)
            # 4xx/5xx 转成异常。
            response.raise_for_status()
            # 成功后清空旧错误。
            self.last_error = ""
            rows = []
            # 逐条把 Tavily 结果转换成检索统一字段。
            for item in response.json().get("results", []):
                content = clean_web_text(item.get("raw_content") or item.get("content", ""))
                # 没有可用正文的网页不加入候选。
                if content:
                    rows.append({"source_id": item.get("url", ""), "source_type": "web",
                                 "retrieval_channel": "web", "title": clean_web_text(item.get("title", ""), 180),
                                 "content": content, "score": number(item.get("score"))})
            return rows
        except Exception as exc:
            # 保存错误和日志，但不让网页故障阻断本地法律库回答。
            self.last_error = str(exc)
            logger.warning("联网检索失败", exc_info=True)
            return []


class Retriever:
    """问答流程的检索入口；保持现有调用，不增加新的模块。"""

    def __init__(self, settings: Settings, model: ModelGateway, milvus: MilvusStore, workspace: WorkspaceService):
        # 保存配置并分别组装公共、私有和网页三个检索器。
        self.settings = settings
        self.public = VectorRetriever(model, milvus, settings)
        self.workspace = workspace
        self.web = WebRetriever(settings)

    def search_public_knowledge(self, plan: SearchPlan) -> list[dict]:
        """按 SearchPlan 查询公共 Milvus 知识库。"""

        # direct 等路线可能明确关闭公共检索。
        if not plan.search_public:
            return []
        # 把计划中的查询、集合、条号、关键词和逐集合配置完整传入。
        return self.public.search_public(
            plan.query, plan.collections, query_variants=plan.query_variants,
            priority_collections=plan.priority_collections, article_numbers=plan.article_numbers,
            keywords=plan.keywords, wants_citation_relations=plan.wants_citation_relations,
            collection_configs=plan.collection_configs,
        )

    def search_user_materials(self, plan: SearchPlan, user: dict | None, session_id: str | None) -> list[dict]:
        """只在当前用户和会话范围内检索其上传材料。"""

        # 未登录或路线关闭私有检索时绝不查询用户库。
        if not user or not plan.search_private:
            return []
        # 合并改写问题、原问题、变体和关键词，增强上传材料的全文/向量召回。
        query = "\n".join(unique_texts([plan.query, plan.original_query, plan.rewritten_query,
                                       *plan.query_variants, *plan.keywords]))
        # WorkspaceService 内部同时按 user_id 和 session_id 隔离。
        rows = self.workspace.search(user, query, session_id,
                                     limit=setting_value(self.settings, "retrieval_private_top_k", 4))
        # 旧记录缺少通道时补为 private。
        for row in rows:
            row.setdefault("retrieval_channel", "private")
        # 用户资料自身也按来源去重。
        return VectorRetriever.merge_results([rows])

    def search_web_pages(self, plan: SearchPlan) -> list[dict]:
        """仅在计划最终允许联网时查询网页。"""

        if not plan.include_web:
            return []
        return self.web.search(plan.query, setting_value(self.settings, "retrieval_web_top_k", 3))

    def search(self, plan: SearchPlan, user: dict | None, session_id: str | None = None) -> list[dict]:
        """兼容串行旧入口：依次合并公共、私有和网页结果。"""

        return (self.search_public_knowledge(plan) + self.search_user_materials(plan, user, session_id)
                + self.search_web_pages(plan))


def rrf(channels: list[list[dict]], limit: int = 12, k: int = 60,
        channel_weights: dict[str, float] | None = None, private_limit: int = 4, exact_limit: int = 4) -> list[dict]:
    """融合分数仅用于排序，不把它当成相关性或法律正确率。"""

    # 调用者可以覆盖部分通道权重，未覆盖项保留默认值。
    weights = {**DEFAULT_CHANNEL_WEIGHTS, **(channel_weights or {})}
    # merged 按来源 ID 合并多通道命中。
    merged = {}
    # 每个 channels 元素是一条检索结果列表。
    for rows in channels:
        # rank 从 1 开始，排名越靠前 RRF 加分越多。
        for rank, row in enumerate(rows, start=1):
            key = source_key(row)
            # 首次出现时复制记录并初始化融合分和命中通道。
            current = merged.setdefault(key, {**row, "source_id": key, "rrf_score": 0.0, "matched_channels": []})
            channel = retrieval_channel(row)
            # 某些记录已经在集合内合并过多个通道。
            matched = row.get("matched_channels") or [channel]
            # RRF 公式为权重之和/(k+名次)，避免不同原始分数尺度不可比。
            current["rrf_score"] += sum(weights.get(value, 1.0) for value in matched) / (k + rank)
            # 汇总命中通道并去重。
            current["matched_channels"] = unique_texts([*current["matched_channels"], *matched])
            # 更高业务优先级版本覆盖展示字段，但保留融合分和通道。
            if source_priority(row) > source_priority(current):
                current.update({name: value for name, value in row.items() if name not in {"rrf_score", "matched_channels"}})
    # 先业务优先级、再 RRF 分排序。
    ranked = sorted(merged.values(), key=lambda row: (source_priority(row), row["rrf_score"]), reverse=True)
    # 单独提取最多 exact_limit 条精确命中和 private_limit 条私有材料。
    exact = [row for row in ranked if is_exact_match(row)][:exact_limit]
    private = [row for row in ranked if row.get("source_type") in PRIVATE_SOURCE_TYPES][:private_limit]
    # selected 按“精确 -> 私有 -> 全体排名”选择，used 防重复。
    selected = []
    used = set()
    for row in [*exact, *private, *ranked]:
        if row["source_id"] not in used:
            selected.append(row)
            used.add(row["source_id"])
        if len(selected) >= limit:
            break
    return selected


class Reranker:
    """用重排模型重新判断候选与当前问题的相关性。"""

    def __init__(self, model: ModelGateway, input_limit: int = 10, doc_max_chars: int = 800):
        # model.rerank 最终调用配置的 BGE-reranker-v2-m3。
        self.model = model
        # 最多向重排接口发送多少条非精确候选。
        self.input_limit = max(1, int(input_limit))
        # 每条候选最多取多少字符，至少 200。
        self.doc_max_chars = max(200, int(doc_max_chars))

    def rank(self, query: str, rows: list[dict]) -> list[dict]:
        """兼容 workflow 的简短方法名。"""

        return self.rerank(query, rows)

    def rerank(self, query: str, rows: list[dict]) -> list[dict]:
        """保留精确命中，并对其余前 N 条候选调用模型重排。"""

        # 精确条号已经由数据库条件确定，单独保留；其他候选仍需重排。
        exact = [row for row in rows if is_exact_match(row)]
        candidates = [row for row in rows if not is_exact_match(row)][:self.input_limit]
        # 没有普通候选时直接返回精确依据。
        if not candidates:
            return exact
        # 只发送每条正文前 doc_max_chars 字，控制延迟和模型上限。
        docs = [str(row.get("content") or "")[:self.doc_max_chars] for row in candidates]
        # 接口失败或返回异常结构时走可观察降级。
        try:
            # pairs 应为 (原候选下标, 重排分数) 列表。
            pairs = self.model.rerank(query, docs)
            # 模型网关可能内部降级返回旧排序，不能伪装成真实重排成功。
            if getattr(self.model, "last_rerank_error", ""):
                raise RuntimeError("重排接口已降级")
            ranked = []
            # used 防止模型重复返回同一下标。
            used = set()
            for index, score in pairs:
                # 下标必须是真正 int、在范围内且未重复。
                if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(candidates) and index not in used:
                    # NaN/无穷大不能参与排序。
                    if not math.isfinite(float(score)):
                        raise ValueError("重排分数不是有限数字")
                    ranked.append({**candidates[index], "rerank_score": float(score)})
                    used.add(index)
            # 完全没有有效 pair 视为接口失败。
            if not ranked:
                raise ValueError("重排没有返回有效结果")
            # 按重排分降序。
            ranked.sort(key=relevance_score, reverse=True)
        except Exception:
            # 降级时保留原候选顺序，并打标供日志和风险提示识别。
            logger.warning("重排失败，保留原候选并标记降级", exc_info=True)
            ranked = [{**row, "rerank_degraded": True} for row in candidates]
        # 记录已经被重排返回的来源键。
        keys = {source_key(row) for row in ranked}
        # 私有材料不能因重排模型 top_n 截断而完全消失。
        private = [row for row in rows if row.get("source_type") in PRIVATE_SOURCE_TYPES and source_key(row) not in keys]
        # 最终顺序为精确依据、补回私有材料、重排候选。
        return exact + private + ranked


class EvidenceBuilder:
    """从重排结果中去重、裁剪并按配额选择最终回答依据。"""

    def __init__(self, limit: int = 8, max_content_chars: int = 1200, private_limit: int = 4, exact_limit: int = 4):
        # 最终证据总数至少 1。
        self.limit = max(1, int(limit))
        # 单条非规范证据裁剪长度至少 200 字。
        self.max_content_chars = max(200, int(max_content_chars))
        # 私有和精确证据各自也有上限，防止单一来源挤占全部名额。
        self.private_limit = max(1, int(private_limit))
        self.exact_limit = max(1, int(exact_limit))

    def clean_rows(self, rows: list[dict]) -> list[dict]:
        """删除重复或空正文，并对非规范材料生成明确节选。"""

        cleaned = []
        used = set()
        for row in rows:
            key = source_key(row)
            # 同来源只保留一次，空正文不能作为答案依据。
            if key in used or not str(row.get("content") or "").strip():
                continue
            used.add(key)
            # 补上统一 source_id，并按规则裁剪非规范材料。
            cleaned.append(evidence_excerpt({**row, "source_id": key}, self.max_content_chars))
        return cleaned

    def build(self, rows: list[dict]) -> list[dict]:
        """按优先级、分数和分组配额选择最终证据。"""

        evidence = []
        # counts 跟踪私有和精确组已经选择多少条。
        counts = {"private": 0, "exact": 0}
        # 先清理，再按业务优先级和相关分排序。
        ranked = sorted(self.clean_rows(rows), key=lambda row: (source_priority(row), relevance_score(row)), reverse=True)
        for row in ranked:
            # 用户没有明确查引用关系时，引用记录不能单独作为回答证据。
            if row.get("collection") == "civil_citations" and not row.get("citation_query_match"):
                continue
            # 精确、私有和其他公共证据使用不同配额。
            if is_exact_match(row):
                group, quota = "exact", self.exact_limit
            elif row.get("source_type") in PRIVATE_SOURCE_TYPES:
                group, quota = "private", self.private_limit
            else:
                # 普通公共/网页资料低于相关阈值时过滤。
                if relevance_score(row) < MIN_PUBLIC_RELEVANCE_SCORE:
                    continue
                group, quota = "other", self.limit
            # 当前组已达配额时跳过。
            if counts.get(group, 0) >= quota:
                continue
            evidence.append(row)
            counts[group] = counts.get(group, 0) + 1
            # 达到最终总数上限后停止。
            if len(evidence) >= self.limit:
                break
        return evidence


class ProcessingPipeline:
    """workflow仍调用这三个步骤，其他问答代码不必跟着重写。"""

    def __init__(self, reranker: Reranker, evidence: EvidenceBuilder, rrf_limit: int = 12,
                 rrf_k: int = 60, channel_weights: dict[str, float] | None = None):
        # 保存重排器和证据筛选器。
        self.ranker = reranker
        self.evidence = evidence
        # 融合候选数量和 RRF 常数至少为 1。
        self.rrf_limit = max(1, int(rrf_limit))
        self.rrf_k = max(1, int(rrf_k))
        # 可选通道权重覆盖。
        self.channel_weights = channel_weights

    def merge_results(self, channels: list[list[dict]]) -> list[dict]:
        """第一步：用 RRF 融合多路候选。"""

        return rrf(channels, limit=self.rrf_limit, k=self.rrf_k, channel_weights=self.channel_weights)

    def rank(self, query: str, rows: list[dict]) -> list[dict]:
        """第二步：用重排模型重新打分。"""

        return self.ranker.rerank(query, rows)

    def build_evidence(self, rows: list[dict]) -> list[dict]:
        """第三步：按质量和配额筛选最终证据。"""

        return self.evidence.build(rows)

    def process(self, query: str, channels: list[list[dict]]) -> dict:
        """一次执行融合、重排和证据构建，并返回各阶段统计。"""

        fused = self.merge_results(channels)
        ranked = self.rank(query, fused)
        evidence = self.build_evidence(ranked)
        # 同时返回中间结果，方便测试和问题排查。
        return {"fused": fused, "ranked": ranked, "evidence": evidence,
                "counts": {"fused": len(fused), "ranked": len(ranked), "evidence": len(evidence)}}

