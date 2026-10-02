"""调用大模型理解用户问题，并定义前端提交给 RAG 的请求数据结构。

这一阶段不做向量检索。它先利用会话记忆消解“这个、他、刚才”等指代，再让大模型
只返回 JSON，识别意图、法律领域、条号、关键词、查询改写和工作流路线。
"""

# json 用于把记忆放入提示词，以及解析大模型返回的 JSON。
import json
# logging 记录问题理解模型失败，方便排查但不中断问答。
import logging

# BaseModel 定义接口请求；Field 创建安全默认容器；两个 validator 校验字段和整体验证。
from pydantic import BaseModel, Field, field_validator, model_validator

# 重新导出这两个计划模型，兼容旧导入路径。
from .plan import CollectionConfig, SearchPlan
# 路由模块提供允许值文字和标准化函数。
from .route import VALID_WORKFLOW_ROUTES_TEXT, select_workflow_route


# 独立日志器便于只筛选理解阶段异常。
logger = logging.getLogger("law_rag.rag.understand")

# 大模型只能从这些意图中选择，避免自由文本导致路由无法处理。
INTENT_VALUES = [
    "article_lookup",
    "case_reference",
    "evidence_strategy",
    "procedure",
    "compensation",
    "risk_assessment",
    "general",
]
# 民事系统当前支持的法律领域分类。
LEGAL_DOMAIN_VALUES = [
    "marriage_family",
    "loan_debt",
    "labor",
    "contract",
    "tort",
    "inheritance",
    "property",
    "general",
]
# 集合用于 O(1) 校验模型结果。
VALID_INTENTS = set(INTENT_VALUES)
VALID_LEGAL_DOMAINS = set(LEGAL_DOMAIN_VALUES)
# 中文顿号连接的文字直接嵌入提示词，让模型看到完整允许范围。
VALID_INTENTS_TEXT = "、".join(INTENT_VALUES)
VALID_LEGAL_DOMAINS_TEXT = "、".join(LEGAL_DOMAIN_VALUES)

# 中文数字字符和数位，用于把“第五百八十五条”规范成 "585"。
CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
CHINESE_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def normalize_article_number(value: str) -> str:
    """把模型返回的中文或阿拉伯法条号统一成阿拉伯数字字符串。"""

    # 统一输入并清理空白。
    raw = str(value or "").strip()
    if not raw:
        return ""
    # 已经是数字时保持原字符串。
    if raw.isdigit():
        return raw
    # total 保存万级累计，section 保存当前万以内部分，number 保存当前数字。
    total = 0
    section = 0
    number = 0
    # matched 用来区分完全没有中文数字的原始字符串。
    matched = False
    # 从左到右解析中文数值。
    for char in raw:
        if char in CHINESE_DIGITS:
            number = CHINESE_DIGITS[char]
            matched = True
            continue
        # 当前字符不是数字时尝试当作十/百/千/万单位。
        unit = CHINESE_UNITS.get(char)
        # 出现未知字符时不强行转换，原样返回供后续校验。
        if not unit:
            return raw
        matched = True
        # 万会结束并放大当前 section。
        if unit == 10000:
            section = (section + (number or 1)) * unit
            total += section
            section = 0
        else:
            # “十”前省略一时按 1 计算。
            section += (number or 1) * unit
        number = 0
    # 没匹配任何中文数字时保持原值。
    if not matched:
        return raw
    # 返回统一字符串，便于精确法条检索。
    return str(total + section + number)


# 这些词常出现在依赖上文的短追问中。
FOLLOWUP_REFERENCE_MARKERS = (
    "他",
    "她",
    "它",
    "这个",
    "那个",
    "这些",
    "那些",
    "上述",
    "上面",
    "前面",
    "刚才",
    "刚刚",
    "这份",
    "这个证据",
    "这段视频",
    "这段录音",
    "这张图",
    "能不能",
    "可以吗",
    "怎么办",
    "怎么赔",
    "多少钱",
)


def looks_followup_query(query: str) -> bool:
    """用可解释规则判断本轮是否可能是依赖上下文的追问。"""

    text = str(query or "").strip()
    # 空问题不是追问。
    if not text:
        return False
    # 去空白后用于长度判断。
    compact = "".join(text.split())
    # 短问题或包含指代/继续追问词时，路由应重点参考记忆。
    return len(compact) <= 40 or any(marker in text for marker in FOLLOWUP_REFERENCE_MARKERS)


def _route_memory_context(memory_context=None) -> dict:
    """从完整记忆对象中提取少量、与检索路线判断有关的内容。"""

    # 没有记忆时返回空字典。
    if memory_context is None:
        return {}
    # 读取当前主题/目标、摘要和旧版案件记忆。
    short_memory = getattr(memory_context, "short_memory", {}) or {}
    summary = getattr(memory_context, "summary", {}) or {}
    legacy = getattr(memory_context, "legacy_case_memory", getattr(memory_context, "case_memory", {})) or {}
    # recent 最多保留最近 8 条消息，每条压到 600 字。
    recent = []
    for item in (getattr(memory_context, "short_term", []) or [])[-8:]:
        # 合并多余空白，减少提示词体积。
        content = " ".join(str(item.get("content") or "").split())
        if content:
            recent.append({"role": str(item.get("role") or ""), "content": content[:600]})
    # original_hits 是记忆检索回查的原始消息，最多 4 条。
    original_hits = []
    for item in (getattr(memory_context, "original_hits", []) or [])[:4]:
        content = " ".join(str(item.get("content") or "").split())
        if content:
            original_hits.append({"role": str(item.get("role") or ""), "content": content[:600]})
    # 形成专门给问题理解模型使用的紧凑记忆。
    result = {
        "current_topic": str(short_memory.get("current_topic") or "").strip(),
        "current_goal": str(short_memory.get("current_goal") or "").strip(),
        "summary": str(summary.get("summary") if isinstance(summary, dict) else summary or "").strip()[-1800:],
        "legacy_case_memory": legacy,
        "recent_messages": recent,
        "original_hits": original_hits,
    }
    # 删除所有空字段，减少无效 token。
    return {key: value for key, value in result.items() if value not in (None, "", [], {})}


def _route_memory_text(route_memory: dict) -> str:
    """把路由记忆压成最多 5000 字的紧凑 JSON。"""

    if not route_memory:
        return ""
    # separators 去掉 JSON 多余空格，降低模型输入长度。
    return json.dumps(route_memory, ensure_ascii=False, separators=(",", ":"))[:5000]


class QueryUnderstanding:
    """负责指代消解、问题改写和大模型结构化意图识别。"""

    def __init__(self, model=None):
        # model 通常是 ModelGateway；测试或降级环境可不传。
        self.model = model

    def understand(self, query: str, memory_context=None) -> dict:
        """理解一条问题；模型失败时返回保守的 full_retrieval 结果。"""

        # 保存用户原始问题。
        original = str(query or "").strip()

        # 第一步取得记忆服务已经识别出的“他/这个”等指代关系。
        resolved_references = self.resolve_reference(original, memory_context)

        # 第二步优先使用记忆服务生成的完整问题。
        text = self.rewrite_query(original, memory_context, resolved_references)

        # 提取专用于路由判断的紧凑记忆及其 JSON 文本。
        route_memory = _route_memory_context(memory_context)
        route_memory_text = _route_memory_text(route_memory)
        # base 是无论模型成功失败都必须保留的基础结果。
        base = {
            "query": text,
            "original_query": original,
            "rewritten_query": text,
            "resolved_references": resolved_references,
            "route_memory": route_memory,
            "route_memory_text": route_memory_text,
            "route_memory_used": bool(route_memory_text and looks_followup_query(original)),
        }
        # 没有模型时不使用固定劳动规则猜测，直接返回保守全量检索。
        if not self.model:
            return self.empty_result(base, "LLM 模型未注入")
        # 模型接口、JSON 解析和字段规范都可能失败，因此整体保护。
        try:
            # 从模型网关读取意图识别专用配置。
            settings = getattr(self.model, "settings", None)
            # 关闭深度思考并使用低温度，要求稳定、快速地返回 JSON。
            response = self.model.chat(
                self.messages(text, original, resolved_references, memory_context),
                max_tokens=int(getattr(settings, "intent_max_tokens", 900) or 900),
                thinking_enabled=False,
                temperature=float(getattr(settings, "intent_llm_temperature", 0.0) or 0.0),
            )
            # 清理代码围栏并解析模型 JSON。
            parsed = self.safe_json(response)
            if not parsed:
                return self.empty_result(base, "LLM 未返回有效 JSON")

            # 校验所有枚举、列表、置信度和条号后返回。
            return self.normalize_result(parsed, base)
        except Exception as exc:
            # 理解失败不阻断问答；记录完整日志并采用保守检索路线。
            logger.warning(
                "LLM 问题理解失败，使用保守空理解结果",
                extra={"event": "llm_query_understanding_failed", "fields": {"error": str(exc)}},
                exc_info=True,
            )
            return self.empty_result(base, str(exc))

    @staticmethod
    def messages(text: str, original: str, resolved_references: dict, memory_context=None) -> list[dict]:
        """构造问题理解提示词，要求模型只返回固定 JSON 而不回答法律问题。"""

        # 兼容新旧案件记忆字段。
        legacy_case_memory = getattr(memory_context, "legacy_case_memory", getattr(memory_context, "case_memory", {})) or {}
        # 摘要帮助模型判断短追问真正指向什么。
        summary = getattr(memory_context, "summary", {}) or {}
        # route_memory 是经过长度控制的最近消息和主题信息。
        route_memory = _route_memory_context(memory_context)
        # 提示词明确提供原问题、消解问题、记忆、输出 schema、分类解释和约束。
        # 使用 f-string 把动态允许值直接写入，减少模型返回系统不支持的类别。
        prompt = f"""
你是法律 RAG 系统的问题理解器。请只做检索前的结构化理解，不要回答法律问题。

用户原始问题：
{original}

上下文消解后的问题：
{text}

已消解指代：
{json.dumps(resolved_references, ensure_ascii=False)}

当前对话案件记忆（旧数据，可能为空）：
{json.dumps(legacy_case_memory, ensure_ascii=False)}

当前对话摘要（可能为空）：
{json.dumps(summary, ensure_ascii=False)}

路由记忆上下文（用于二次追问判断，可能为空）：
{json.dumps(route_memory, ensure_ascii=False)}

请只输出一个 JSON object，不要输出 Markdown、解释、代码块或多余文字。
字段必须是：
{{
  "intent": "article_lookup | case_reference | evidence_strategy | procedure | compensation | risk_assessment | general 之一",
  "route": "direct | law_only | case_law | evidence_strategy | procedure_guide | current_law_web | document_review | calculation | full_retrieval 之一",
  "confidence": 0.0到1.0之间的数字,
  "reason": "不超过40个中文字符的分类理由",
  "article_numbers": ["阿拉伯数字条号；没有则空数组"],
  "legal_domains": ["marriage_family | loan_debt | labor | contract | tort | inheritance | property | general 中的一项或多项"],
  "keywords": ["用于检索的中文关键词或短语"],
  "query_variants": ["用于向量和关键词检索的改写查询"],
  "rewritten_query": "适合检索的完整问题",
  "wants_citation_relations": false
}}

意图说明：
- article_lookup：用户主要在问具体法条、条文原文、法律规定或某条如何适用。
- case_reference：用户主要在问案例、判例、法院怎么判、类案裁判规则。
- evidence_strategy：用户主要在问需要什么证据、材料清单、证明对象、证据能不能用。
- procedure：用户主要在问怎么办、流程、起诉、仲裁、投诉、调解等处理步骤。
- compensation：用户主要在问赔偿、补偿、违约金、损失金额或计算标准。
- risk_assessment：用户主要在问风险、胜诉可能、能不能、是否可行。
- general：问题只有法律关系或事实描述，尚未形成明确诉求。

路线说明：
- direct：简单解释、闲聊或不需要检索材料即可回答的问题。
- law_only：以法条、法规、司法解释为核心的问题。
- case_law：需要类案、裁判规则或法院观点支撑的问题。
- evidence_strategy：证据证明力、证据缺口、材料清单、举证建议。
- procedure_guide：起诉、仲裁、投诉、立案、执行、保全等流程问题。
- current_law_web：明确询问最新、现行、新规、政策或官方网页信息。
- document_review：用户要求审查、解读、核对上传的合同、判决书、证据材料或其他文件。
- calculation：用户要求计算赔偿金、补偿金、利息、违约金、工伤赔偿或其他具体金额。
- full_retrieval：复杂综合案件，需要法条、案例、证据、流程等共同支撑。

约束：
- 可选 intent 只能是：{VALID_INTENTS_TEXT}。
- 可选 route 只能是：{VALID_WORKFLOW_ROUTES_TEXT}。
- 可选 legal_domains 只能是：{VALID_LEGAL_DOMAINS_TEXT}。
- route 要根据 intent、问题复杂度和是否需要最新信息判断；不要为了省检索把法律咨询误判成 direct。
- 如果用户本轮是“这个/它/刚才/上面/能不能/怎么办”等二次追问，必须结合路由记忆上下文判断真实诉求、法律领域和是否涉及上传材料。
- 如果路由记忆显示上一轮围绕上传证据、视频、录音、合同、判决书或聊天记录，本轮又在问“这个证据/能不能证明/讲了什么/有没有用”，优先选择 document_review 或 evidence_strategy，不要选 direct。
- 不要根据单个词机械分类，要判断用户真正想解决什么。
- rewritten_query 优先使用上下文消解后的问题；只有原问题明显缺主语、对象或案情时才补全。
- query_variants 至少包含 rewritten_query，最多 8 条。
- keywords 最多 16 个，避免无意义停用词。
- article_numbers 只提取用户问题中明确出现的条号；不要推测或补充条号。
- 只有用户明确要查条文引用关系、引用链、关联条文、配套条文或被引用情况时，wants_citation_relations 才为 true。
- 如果无法判断领域，legal_domains 返回 ["general"]。
""".strip()
        # 这里只需要一个 user 消息；模型收到的任务是分类而不是法律作答。
        return [{"role": "user", "content": prompt}]

    @staticmethod
    def safe_json(text: str) -> dict | None:
        """容忍 Markdown 代码围栏或前后说明，从模型文本中提取 JSON 对象。"""

        # 统一输入并清理空白。
        value = str(text or "").strip()
        # 依次移除 ```json、普通 ``` 开头和结尾围栏。
        if value.startswith("```json"):
            value = value.removeprefix("```json").strip()
        if value.startswith("```"):
            value = value.removeprefix("```").strip()
        if value.endswith("```"):
            value = value[:-3].strip()
        # 第一种情况：模型严格只返回 JSON。
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            # 第二种情况：前后混入少量说明，寻找最外层花括号。
            start = value.find("{")
            end = value.rfind("}")
            if start < 0 or end <= start:
                return None
            try:
                parsed = json.loads(value[start:end + 1])
            except json.JSONDecodeError:
                # 仍无法解析就让上层使用降级结果。
                return None
        # 只接受 JSON object；数组或基础值不符合约定 schema。
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def clean_list(value, allowed: set[str] | None = None, limit: int = 16) -> list[str]:
        """把模型列表清理为去空、去重、可选白名单限制的字符串列表。"""

        # 非列表不能按预期处理，返回空列表。
        if not isinstance(value, list):
            return []
        # items 保持模型原顺序。
        items = []
        for item in value:
            text = str(item or "").strip()
            # 空值跳过。
            if not text:
                continue
            # 提供 allowed 时只保留系统支持的枚举。
            if allowed and text not in allowed:
                continue
            # 去重但不打乱顺序。
            if text not in items:
                items.append(text)
        # 最终截到最大数量，防止模型返回异常长列表。
        return items[:limit]

    @staticmethod
    def normalize_confidence(value) -> float:
        """把模型置信度转换并夹在 0.0—1.0 范围。"""

        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        # max 防负数，min 防超过 1。
        return min(1.0, max(0.0, number))

    @classmethod
    def normalize_result(cls, parsed: dict, base: dict) -> dict:
        """严格清理模型 JSON，生成路由器可以安全使用的结果。"""

        # 未知意图回退 general。
        intent = str(parsed.get("intent") or "general").strip()
        if intent not in VALID_INTENTS:
            intent = "general"
        # 规范置信度和检索问题。
        confidence = cls.normalize_confidence(parsed.get("confidence"))
        rewritten_query = str(parsed.get("rewritten_query") or base["rewritten_query"]).strip() or base["rewritten_query"]
        # 条号先清理列表，再把中文数字转阿拉伯数字并去重。
        article_numbers = [normalize_article_number(item) for item in cls.clean_list(parsed.get("article_numbers"), limit=8)]
        article_numbers = list(dict.fromkeys(item for item in article_numbers if item))
        # 领域只允许白名单，全部无效时使用 general。
        legal_domains = cls.clean_list(parsed.get("legal_domains"), VALID_LEGAL_DOMAINS, limit=4) or ["general"]
        # general 和具体领域同时出现时删除 general。
        if "general" in legal_domains and len(legal_domains) > 1:
            legal_domains = [domain for domain in legal_domains if domain != "general"]
        # 清理关键词和查询变体。
        keywords = cls.clean_list(parsed.get("keywords"), limit=16)
        query_variants = cls.clean_list(parsed.get("query_variants"), limit=8)
        # 改写问题必须是第一个可用查询变体。
        if rewritten_query and rewritten_query not in query_variants:
            query_variants.insert(0, rewritten_query)
        # 路线同样通过白名单标准化。
        route = select_workflow_route(parsed.get("route"))
        # 合并 base，并用规范化字段覆盖原模型值。
        return {
            **base,
            "query": rewritten_query,
            "rewritten_query": rewritten_query,
            "route": route,
            "article_numbers": article_numbers[:8],
            "keywords": keywords,
            "legal_domains": legal_domains,
            "intent": intent,
            "query_variants": query_variants[:8] or [rewritten_query],
            "confidence": confidence,
            "intent_reason": str(parsed.get("reason") or "").strip()[:80],
            "understanding_source": "llm",
            "wants_citation_relations": bool(parsed.get("wants_citation_relations")),
            "route_memory_used": bool(base.get("route_memory_used")),
        }

    @staticmethod
    def empty_result(base: dict, error: str) -> dict:
        """模型不可用时返回不猜测意图、但仍能继续问答的保守结果。"""

        # 仍保留已完成指代消解的问题。
        rewritten_query = str(base.get("rewritten_query") or base.get("query") or "").strip()
        # full_retrieval 防止错误分类漏掉本应检索的法律依据。
        return {
            **base,
            "query": rewritten_query,
            "rewritten_query": rewritten_query,
            "route": "full_retrieval",
            "article_numbers": [],
            "keywords": [],
            "legal_domains": ["general"],
            "intent": "general",
            "query_variants": [rewritten_query] if rewritten_query else [],
            "confidence": 0.0,
            "intent_reason": "LLM 未完成问题理解。",
            "understanding_source": "llm_unavailable",
            "understanding_error": str(error or "未知错误")[:200],
            "wants_citation_relations": False,
            "route_memory_used": bool(base.get("route_memory_used")),
        }

    @staticmethod
    def resolve_reference(text: str, memory_context=None) -> dict:
        """读取记忆层已经计算的指代映射，不在这里重复猜测。"""

        return dict(getattr(memory_context, "resolved_references", {}) or {})

    @staticmethod
    def rewrite_query(text: str, memory_context=None, resolved_references: dict | None = None) -> str:
        """优先使用记忆层的 resolved_query，否则保持用户原问题。"""

        value = str(text or "").strip()
        resolved_query = str(getattr(memory_context, "resolved_query", "") or "").strip()
        return resolved_query or value

class AskRequest(BaseModel):
    """前端普通/流式提问接口的请求模型。"""

    # question 是新字段，query 兼容旧前端。
    question: str = ""
    query: str = ""
    # 登录用户可传会话 ID；API 层会对游客强制清空。
    session_id: str | None = None
    # include_web 兼容旧开关；新流程主要看 retrieval_mode。
    include_web: bool = False
    # local=仅本地，auto=按需要，force=强制联网。
    retrieval_mode: str = "auto"
    # concise/standard/detailed 控制回答展开程度。
    answer_detail: str = "standard"
    # None 表示沿用服务端默认思考开关。
    thinking_enabled: bool | None = None

    @field_validator("retrieval_mode")
    @classmethod
    def validate_retrieval_mode(cls, value: str) -> str:
        """拒绝系统不支持的检索模式。"""

        if value not in {"local", "auto", "force"}:
            raise ValueError("retrieval_mode 必须是 local、auto 或 force")
        return value

    @field_validator("answer_detail")
    @classmethod
    def validate_answer_detail(cls, value: str) -> str:
        """拒绝系统不支持的回答详细度。"""

        if value not in {"concise", "standard", "detailed"}:
            raise ValueError("answer_detail 必须是 concise、standard 或 detailed")
        return value

    @model_validator(mode="after")
    def validate_text(self):
        """question 和兼容字段 query 至少一个必须有非空文字。"""

        if not self.text():
            raise ValueError("问题不能为空")
        return self

    def text(self) -> str:
        """统一取得前端问题，优先使用 question。"""

        return (self.question or self.query).strip()


class SolutionGenerateRequest(BaseModel):
    """根据一轮问答生成详细法律解决方案的请求模型。"""

    question: str = ""
    # answer 兼容完整结构化回答或仅正文字符串。
    answer: dict | str = Field(default_factory=dict)
    # sources 是本轮引用依据。
    sources: list[dict] = Field(default_factory=list)
    session_id: str | None = None
    # True 时 API 跳过 Redis 缓存重新生成。
    force_refresh: bool = False

    @model_validator(mode="after")
    def validate_solution_input(self):
        """方案生成必须有问题、有效回答，且来源数量不能过多。"""

        if not self.question.strip():
            raise ValueError("问题不能为空")
        # 字符串直接检查；字典从 answer 字段取正文。
        if isinstance(self.answer, str):
            valid = bool(self.answer.strip())
        else:
            valid = bool(str(self.answer.get("answer", "")).strip())
        if not valid:
            raise ValueError("回答不能为空")
        # 限制来源数量可以控制方案提示词长度和模型费用。
        if len(self.sources) > 20:
            raise ValueError("参考来源过多")
        return self


class SolutionPdfRequest(BaseModel):
    """下载 PDF 时提交的 Markdown 内容。"""

    markdown: str

    @field_validator("markdown")
    @classmethod
    def validate_markdown(cls, value: str) -> str:
        """清理 Markdown，并限制空内容和异常超长内容。"""

        value = value.strip()
        if not value:
            raise ValueError("解决方案内容不能为空")
        if len(value) > 100_000:
            raise ValueError("解决方案内容过长")
        return value


# 本模块明确提供给 API 和旧代码的公共名称。
__all__ = [
    "AskRequest", "CollectionConfig", "QueryUnderstanding", "SearchPlan",
    "SolutionGenerateRequest", "SolutionPdfRequest",
]

