"""低分检索结果的统一三档处置。"""

import re
from copy import copy
from dataclasses import dataclass
from typing import Any

from app.chat.guard import should_refuse


@dataclass
class LowScoreDecision:
    """返回低分处置后的检索结果和动作。"""

    retrieval_result: Any
    action: str
    event: str | None = None


_GREETING_PHRASES = frozenset({
    "你好",
    "您好",
    "嗨",
    "哈喽",
    "在吗",
    "谢谢",
    "感谢",
    "你能做什么",
    "你能帮我吗",
    "能帮我吗",
})
# 领域词表宁可多收，不放过领域内问题；低分时只把明确领域外问题拒答。
LABOR_LAW_DOMAIN_TERMS = frozenset({
    "劳动",
    "劳动合同",
    "合同",
    "工资",
    "薪",
    "加班",
    "社保",
    "公积金",
    "辞退",
    "开除",
    "解雇",
    "裁员",
    "离职",
    "解除",
    "仲裁",
    "补偿",
    "赔偿",
    "工伤",
    "试用期",
    "竞业",
    "年假",
    "产假",
    "病假",
    "调岗",
    "欠薪",
    "法条",
    "法规",
    "违法",
    "法院",
    "诉讼",
    "用人单位",
    "劳动者",
    "公司",
    "员工",
    "入职",
    "用工",
    "工作年限",
    "工作时间",
    "通知",
    "签字",
    "工龄",
    "退休",
    "劳务",
    "五险一金",
    "最低工资",
    "经济性裁员",
    "劳动关系",
})
_FOLLOW_UP_MARKERS = frozenset({
    "那",
    "这个",
    "这种",
    "这样",
    "还",
    "再",
    "呢",
    "吗",
    "么",
})
_FOLLOW_UP_PREFIXES = (
    "能",
    "可以",
    "是否",
    "有没有",
    "怎么",
    "怎么办",
    "多少",
    "多久",
    "几",
    "哪",
    "什么",
    "为什么",
    "如何",
    "需要",
    "要不要",
)


def _normalize_question(question: str) -> str:
    """去除首尾空白和常见标点，保留正文用于严格白名单匹配。"""
    return re.sub(r"[\s，。！？、；：,.!?;:]+$", "", question.strip())


def _contains_labor_law_term(question: str) -> bool:
    """按领域词表判断问题是否涉及劳动法，宁可多收不放过领域内问题。"""
    normalized = _normalize_question(question)
    return any(term in normalized for term in LABOR_LAW_DOMAIN_TERMS)


def _looks_like_follow_up(question: str) -> bool:
    """识别依赖前文的省略追问，避免把上下文中的领域词扩散到新话题。"""
    normalized = _normalize_question(question)
    if not normalized:
        return False
    return normalized.startswith(tuple(_FOLLOW_UP_MARKERS | set(_FOLLOW_UP_PREFIXES)))


def is_labor_law_domain_question(
    question: str,
    previous_question: str | None = None,
) -> bool:
    """判断低分问题是否仍处于劳动法对话，支持带领域上下文的事实补充和追问。"""
    if _contains_labor_law_term(question):
        return True
    if previous_question and _looks_like_follow_up(question):
        return _contains_labor_law_term(previous_question)
    return False


def is_conservative_greeting(question: str) -> bool:
    """仅识别严格短句白名单，避免把真实法律问题放过。"""
    normalized = _normalize_question(question)
    if not normalized or len(normalized) > 12:
        return False
    if _contains_labor_law_term(normalized):
        return False
    return normalized in _GREETING_PHRASES


def _clear_retrieval_sources(retrieval_result: Any) -> Any:
    """清空低分领域引导的法源，防止无依据回答携带 citation。"""
    guidance_result = copy(retrieval_result)
    guidance_result.articles = []
    guidance_result.context_block = ""
    return guidance_result


def _previous_user_question(
    retrieval_service: Any,
    user_id: str | None,
    session_id: str | None,
) -> str | None:
    """从当前用户会话读取最近一条用户问题，失败时安全返回空。"""
    if not user_id or not session_id:
        return None
    memory = getattr(retrieval_service, "short_term_memory", None)
    if memory is None:
        return None
    try:
        messages = memory.read_messages(user_id, session_id)
    except Exception:
        return None
    for message in reversed(messages or []):
        if message.get("role") == "user":
            content = str(message.get("content") or "").strip()
            if content and not is_conservative_greeting(content):
                return content
    return None


def _merge_result_articles(original_result: Any, retry_result: Any) -> Any:
    """为轻量测试替身合并候选，生产服务使用检索层统一重排实现。"""
    merged_by_key = {}
    for article in [*original_result.articles, *retry_result.articles]:
        merged_by_key.setdefault(article.chunk_key, article)
    merged_result = copy(retry_result)
    merged_result.articles = list(merged_by_key.values())
    merged_result.context_block = "\n\n".join(
        article.content or "" for article in merged_result.articles
    )
    return merged_result


def resolve_low_score(
    question: str,
    retrieval_result: Any,
    *,
    retrieval_service: Any,
    min_vector_score: float,
    retrieve_kwargs: dict[str, Any],
) -> LowScoreDecision:
    """统一执行低分候选的回退重检索、寒暄引导和拒答决策。"""
    if not retrieval_result.articles or not should_refuse(
        retrieval_result.articles, min_vector_score=min_vector_score
    ):
        return LowScoreDecision(retrieval_result, "answer")

    previous_question = _previous_user_question(
        retrieval_service,
        retrieve_kwargs.get("user_id"),
        retrieve_kwargs.get("session_id"),
    )
    if previous_question and previous_question != question:
        retry_question = f"{previous_question}。{question}"
        retry_result = retrieval_service.retrieve(retry_question, **retrieve_kwargs)
        if retry_result.articles and not should_refuse(
            retry_result.articles, min_vector_score=min_vector_score
        ):
            merge_results = getattr(retrieval_service, "merge_retry_results", None)
            if callable(merge_results):
                retry_result = merge_results(
                    retrieval_result,
                    retry_result,
                    retry_question,
                    top_n=retrieve_kwargs.get("rerank_top_n", 5),
                )
            else:
                retry_result = _merge_result_articles(retrieval_result, retry_result)
            return LowScoreDecision(retry_result, "answer", "low_score_retry_hit")

    if is_conservative_greeting(question):
        return LowScoreDecision(retrieval_result, "greeting", "greeting_low_score")
    if is_labor_law_domain_question(question, previous_question):
        return LowScoreDecision(
            _clear_retrieval_sources(retrieval_result),
            "domain_guidance",
            "low_score_domain_guidance",
        )
    return LowScoreDecision(retrieval_result, "refuse", "refuse_low_score")
