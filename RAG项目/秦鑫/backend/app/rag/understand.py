"""语义识别和 RAG 请求模型的统一入口。"""

import json
import logging

from pydantic import BaseModel, Field, field_validator, model_validator

from .plan import CollectionConfig, SearchPlan
from .route import VALID_WORKFLOW_ROUTES_TEXT, select_workflow_route


logger = logging.getLogger("law_rag.rag.understand")

INTENT_VALUES = [
    "article_lookup",
    "case_reference",
    "evidence_strategy",
    "procedure",
    "compensation",
    "risk_assessment",
    "general",
]
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
VALID_INTENTS = set(INTENT_VALUES)
VALID_LEGAL_DOMAINS = set(LEGAL_DOMAIN_VALUES)
VALID_INTENTS_TEXT = "、".join(INTENT_VALUES)
VALID_LEGAL_DOMAINS_TEXT = "、".join(LEGAL_DOMAIN_VALUES)

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
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.isdigit():
        return raw
    total = 0
    section = 0
    number = 0
    matched = False
    for char in raw:
        if char in CHINESE_DIGITS:
            number = CHINESE_DIGITS[char]
            matched = True
            continue
        unit = CHINESE_UNITS.get(char)
        if not unit:
            return raw
        matched = True
        if unit == 10000:
            section = (section + (number or 1)) * unit
            total += section
            section = 0
        else:
            section += (number or 1) * unit
        number = 0
    if not matched:
        return raw
    return str(total + section + number)


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
    text = str(query or "").strip()
    if not text:
        return False
    compact = "".join(text.split())
    return len(compact) <= 40 or any(marker in text for marker in FOLLOWUP_REFERENCE_MARKERS)


def _route_memory_context(memory_context=None) -> dict:
    if memory_context is None:
        return {}
    short_memory = getattr(memory_context, "short_memory", {}) or {}
    summary = getattr(memory_context, "summary", {}) or {}
    legacy = getattr(memory_context, "legacy_case_memory", getattr(memory_context, "case_memory", {})) or {}
    recent = []
    for item in (getattr(memory_context, "short_term", []) or [])[-8:]:
        content = " ".join(str(item.get("content") or "").split())
        if content:
            recent.append({"role": str(item.get("role") or ""), "content": content[:600]})
    original_hits = []
    for item in (getattr(memory_context, "original_hits", []) or [])[:4]:
        content = " ".join(str(item.get("content") or "").split())
        if content:
            original_hits.append({"role": str(item.get("role") or ""), "content": content[:600]})
    result = {
        "current_topic": str(short_memory.get("current_topic") or "").strip(),
        "current_goal": str(short_memory.get("current_goal") or "").strip(),
        "summary": str(summary.get("summary") if isinstance(summary, dict) else summary or "").strip()[-1800:],
        "legacy_case_memory": legacy,
        "recent_messages": recent,
        "original_hits": original_hits,
    }
    return {key: value for key, value in result.items() if value not in (None, "", [], {})}


def _route_memory_text(route_memory: dict) -> str:
    if not route_memory:
        return ""
    return json.dumps(route_memory, ensure_ascii=False, separators=(",", ":"))[:5000]


class QueryUnderstanding:
    def __init__(self, model=None):
        self.model = model

    def understand(self, query: str, memory_context=None) -> dict:
        original = str(query or "").strip()
        resolved_references = self.resolve_reference(original, memory_context)
        text = self.rewrite_query(original, memory_context, resolved_references)
        route_memory = _route_memory_context(memory_context)
        route_memory_text = _route_memory_text(route_memory)
        base = {
            "query": text,
            "original_query": original,
            "rewritten_query": text,
            "resolved_references": resolved_references,
            "route_memory": route_memory,
            "route_memory_text": route_memory_text,
            "route_memory_used": bool(route_memory_text and looks_followup_query(original)),
        }
        if not self.model:
            return self.empty_result(base, "LLM 模型未注入")
        try:
            settings = getattr(self.model, "settings", None)
            response = self.model.chat(
                self.messages(text, original, resolved_references, memory_context),
                max_tokens=int(getattr(settings, "intent_max_tokens", 900) or 900),
                thinking_enabled=False,
                temperature=float(getattr(settings, "intent_llm_temperature", 0.0) or 0.0),
            )
            parsed = self.safe_json(response)
            if not parsed:
                return self.empty_result(base, "LLM 未返回有效 JSON")
            return self.normalize_result(parsed, base)
        except Exception as exc:
            logger.warning(
                "LLM 问题理解失败，使用保守空理解结果",
                extra={"event": "llm_query_understanding_failed", "fields": {"error": str(exc)}},
                exc_info=True,
            )
            return self.empty_result(base, str(exc))

    @staticmethod
    def messages(text: str, original: str, resolved_references: dict, memory_context=None) -> list[dict]:
        legacy_case_memory = getattr(memory_context, "legacy_case_memory", getattr(memory_context, "case_memory", {})) or {}
        summary = getattr(memory_context, "summary", {}) or {}
        route_memory = _route_memory_context(memory_context)
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
        return [{"role": "user", "content": prompt}]

    @staticmethod
    def safe_json(text: str) -> dict | None:
        value = str(text or "").strip()
        if value.startswith("```json"):
            value = value.removeprefix("```json").strip()
        if value.startswith("```"):
            value = value.removeprefix("```").strip()
        if value.endswith("```"):
            value = value[:-3].strip()
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            start = value.find("{")
            end = value.rfind("}")
            if start < 0 or end <= start:
                return None
            try:
                parsed = json.loads(value[start:end + 1])
            except json.JSONDecodeError:
                return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def clean_list(value, allowed: set[str] | None = None, limit: int = 16) -> list[str]:
        if not isinstance(value, list):
            return []
        items = []
        for item in value:
            text = str(item or "").strip()
            if not text:
                continue
            if allowed and text not in allowed:
                continue
            if text not in items:
                items.append(text)
        return items[:limit]

    @staticmethod
    def normalize_confidence(value) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return min(1.0, max(0.0, number))

    @classmethod
    def normalize_result(cls, parsed: dict, base: dict) -> dict:
        intent = str(parsed.get("intent") or "general").strip()
        if intent not in VALID_INTENTS:
            intent = "general"
        confidence = cls.normalize_confidence(parsed.get("confidence"))
        rewritten_query = str(parsed.get("rewritten_query") or base["rewritten_query"]).strip() or base["rewritten_query"]
        article_numbers = [normalize_article_number(item) for item in cls.clean_list(parsed.get("article_numbers"), limit=8)]
        article_numbers = list(dict.fromkeys(item for item in article_numbers if item))
        legal_domains = cls.clean_list(parsed.get("legal_domains"), VALID_LEGAL_DOMAINS, limit=4) or ["general"]
        if "general" in legal_domains and len(legal_domains) > 1:
            legal_domains = [domain for domain in legal_domains if domain != "general"]
        keywords = cls.clean_list(parsed.get("keywords"), limit=16)
        query_variants = cls.clean_list(parsed.get("query_variants"), limit=8)
        if rewritten_query and rewritten_query not in query_variants:
            query_variants.insert(0, rewritten_query)
        route = select_workflow_route(parsed.get("route"))
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
        rewritten_query = str(base.get("rewritten_query") or base.get("query") or "").strip()
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
        legacy_case_memory = getattr(memory_context, "legacy_case_memory", getattr(memory_context, "case_memory", {})) or {}
        parties = legacy_case_memory.get("parties") or {}
        resolved = {}
        employee = parties.get("employee")
        employer = parties.get("employer")
        if employee and any(word in text for word in ["他", "她", "其", "劳动者", "员工"]):
            resolved["他"] = employee
        if employer and any(word in text for word in ["公司", "单位", "老板", "对方"]):
            resolved["公司"] = employer
        return resolved

    @staticmethod
    def rewrite_query(text: str, memory_context=None, resolved_references: dict | None = None) -> str:
        value = str(text or "").strip()
        legacy_case_memory = getattr(memory_context, "legacy_case_memory", getattr(memory_context, "case_memory", {})) or {}
        if not value or not legacy_case_memory:
            return value
        vague = any(word in value for word in ["他", "她", "这个", "那个", "那", "刚才", "前面", "赔多少", "怎么算"])
        if not vague:
            return value
        parties = legacy_case_memory.get("parties") or {}
        facts = legacy_case_memory.get("facts") or {}
        case_type = str(legacy_case_memory.get("case_type") or "")
        context_text = json.dumps(legacy_case_memory, ensure_ascii=False)
        labor_context = (
            case_type == "劳动争议"
            or bool(parties.get("employee") or parties.get("employer"))
            or any(word in context_text for word in ("工资", "劳动", "加班", "辞退", "离职", "工伤", "社保", "劳动合同", "用人单位"))
        )
        if not labor_context:
            return value
        employee = (resolved_references or {}).get("他") or parties.get("employee")
        employer = parties.get("employer")
        details = []
        if employee:
            details.append(str(employee))
        if employer:
            details.append(f"在{employer}")
        if facts.get("employment_years") is not None:
            details.append(f"工作{QueryUnderstanding._format_number(facts['employment_years'])}年")
        if facts.get("salary") is not None:
            details.append(f"月工资{QueryUnderstanding._format_number(facts['salary'])}元")
        event = str(facts.get("event") or "")
        prefix = "，".join(details)
        if any(word in value for word in ["赔", "补偿", "赔偿金"]):
            action = "违法解除劳动合同" if "解除" in event or "劳动" in event or case_type == "劳动争议" else event or "发生争议"
            subject = employer or "用人单位"
            worker = employee or "劳动者"
            return f"{prefix}，若{subject}{action}，{worker}依法可以获得多少赔偿金？" if prefix else value
        return f"{prefix}。用户追问：{value}" if prefix else value

    @staticmethod
    def _format_number(value) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        return str(int(number)) if number.is_integer() else str(number)




class AskRequest(BaseModel):
    question: str = ""
    query: str = ""
    session_id: str | None = None
    include_web: bool = True
    thinking_enabled: bool | None = None

    @model_validator(mode="after")
    def validate_text(self):
        if not self.text():
            raise ValueError("问题不能为空")
        return self

    def text(self) -> str:
        return (self.question or self.query).strip()


class SolutionGenerateRequest(BaseModel):
    question: str = ""
    answer: dict | str = Field(default_factory=dict)
    sources: list[dict] = Field(default_factory=list)
    session_id: str | None = None
    force_refresh: bool = False

    @model_validator(mode="after")
    def validate_solution_input(self):
        if not self.question.strip():
            raise ValueError("问题不能为空")
        if isinstance(self.answer, str):
            valid = bool(self.answer.strip())
        else:
            valid = bool(str(self.answer.get("answer", "")).strip())
        if not valid:
            raise ValueError("回答不能为空")
        if len(self.sources) > 20:
            raise ValueError("参考来源过多")
        return self


class SolutionPdfRequest(BaseModel):
    markdown: str

    @field_validator("markdown")
    @classmethod
    def validate_markdown(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("解决方案内容不能为空")
        if len(value) > 100_000:
            raise ValueError("解决方案内容过长")
        return value


__all__ = [
    "AskRequest", "CollectionConfig", "QueryUnderstanding", "SearchPlan",
    "SolutionGenerateRequest", "SolutionPdfRequest",
]

