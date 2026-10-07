"""
Query 理解模块
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

对应工单功能需求「Query理解：意图识别 / 消歧 / 分解与抽象」。

实现方式是**一次轻量 LLM 调用**（qwen-flash 实测 ~0.4s，max_tokens 锁到 320），
输出结构化 JSON。之所以不拆成三次调用：本工单硬性要求端到端 < 3 秒，
每多一次往返就多 0.4~1 秒预算。

降级策略（很重要）：
  LLM 超时/报错/返回非法 JSON 时，**不能**让整个问答失败。
  退回「规则归一化结果」继续检索 —— 用户拿到的答案质量略降，但链路不断。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .config import QUERY_UNDERSTANDING_ENABLED, WORK_ORDER_NO  # noqa: F401
from .llm import chat_json
from .query_norm import normalize_query

logger = logging.getLogger(__name__)

# 招股说明书问答里常见的意图枚举
INTENTS = [
    "事实查询",      # 注册资本、法定代表人这类点状事实
    "财务数据查询",  # 收入、占比、金额
    "业务与行业",    # 上下游、供应商、技术标准
    "风险与诉讼",
    "其他",
]

_SYSTEM = """你是招股说明书问答系统的查询理解模块。请把用户问题改写成适合检索的查询，并输出 JSON。

要求：
1. intent：从 ["事实查询","财务数据查询","业务与行业","风险与诉讼","其他"] 中选一个。
2. rewritten：把问题改写成**陈述式的检索式**，补全指代（如"该公司"→"武汉兴图新科电子股份有限公司"），
   去掉"请问""麻烦"等口语词。保留全部关键实体与时间范围。不要添加原文中不存在的信息。
3. sub_questions：**默认只返回原问题（单元素数组）**。只有当问题同时询问**多件不同的事**
   （例如同时问"上游涉及哪些企业"和"下游包括哪些行业"）时才拆成 2 个子问题。
   硬性要求：每个子问题都必须是**能在招股说明书里原样搜到**的表述，
   禁止造词（如把"报告期内"拆成"报告期第一年/第二年"——文档里根本没有这种说法）。
   拆出来的子问题如果召回不到东西，会污染检索结果。
4. keywords：3~8 个用于关键词检索的核心词（公司名、指标名、年份等）。
5. needs_retrieval：布尔值。纯寒暄/与文档无关的闲聊为 false，其余为 true。

只输出 JSON，不要解释。示例：
{"intent":"财务数据查询","rewritten":"武汉兴图新科电子股份有限公司报告期内来自军用领域的收入","sub_questions":["武汉兴图新科电子股份有限公司报告期内来自军用领域的收入"],"keywords":["兴图新科","军用领域","收入","报告期"],"needs_retrieval":true}"""


@dataclass
class QueryUnderstanding:
    """Query 理解结果。"""

    original: str
    normalized: str
    intent: str = "其他"
    rewritten: str = ""
    sub_questions: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    needs_retrieval: bool = True
    degraded: bool = False   # True 表示走了规则降级分支
    ms: float = 0.0

    @property
    def retrieval_query(self) -> str:
        """真正送进检索器的查询串。"""
        return self.rewritten or self.normalized or self.original

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "normalized": self.normalized,
            "intent": self.intent,
            "rewritten": self.rewritten,
            "sub_questions": self.sub_questions,
            "keywords": self.keywords,
            "needs_retrieval": self.needs_retrieval,
            "degraded": self.degraded,
            "ms": round(self.ms, 2),
        }


# 与文档无关的纯闲聊。**只有命中这条才跳过检索**，见 understand() 里的说明。
_CHITCHAT_RE = re.compile(
    r"^\s*(你好|您好|hi|hello|在吗|谢谢|感谢|再见|拜拜|"
    r"你是谁|你叫什么|介绍一下你自己|你能做什么)\s*[!！。.~?？]*\s*$",
    re.I,
)


def _rule_based(query: str, normalized: str) -> QueryUnderstanding:
    """不依赖 LLM 的兜底理解。"""
    qu = QueryUnderstanding(
        original=query,
        normalized=normalized,
        rewritten=normalized,
        sub_questions=[normalized] if normalized else [],
        degraded=True,
    )
    if _CHITCHAT_RE.match(query):
        qu.intent = "其他"
        qu.needs_retrieval = False
    elif re.search(r"收入|金额|占比|比重|注册资本|股本|利润|毛利", query):
        qu.intent = "财务数据查询"
    elif re.search(r"供应商|上游|下游|客户|技术标准|行业", query):
        qu.intent = "业务与行业"
    else:
        qu.intent = "事实查询"
    return qu


def understand(query: str, enabled: bool | None = None) -> QueryUnderstanding:
    """理解用户查询。任何异常都降级到规则分支。"""
    import time

    t0 = time.perf_counter()
    normalized = normalize_query(query)
    use_llm = QUERY_UNDERSTANDING_ENABLED if enabled is None else enabled

    if not use_llm or not normalized:
        qu = _rule_based(query, normalized)
        qu.ms = (time.perf_counter() - t0) * 1000
        return qu

    data = chat_json(
        [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": query},
        ],
        max_tokens=320,
    )

    if not isinstance(data, dict):
        qu = _rule_based(query, normalized)
        qu.ms = (time.perf_counter() - t0) * 1000
        return qu

    # ⚠️ needs_retrieval **不接受模型自由裁量**。
    # 实测模型会把「如何用 Python 实现快速排序？」判成闲聊 → 直接跳过检索、
    # 返回寒暄语；而正确处理应该是「照常检索 → 被阈值闸门拦空 → 如实回答无依据」。
    # 判据必须是确定性的：只有命中纯闲聊正则才跳过检索。
    qu = QueryUnderstanding(
        original=query,
        normalized=normalized,
        intent=str(data.get("intent") or "其他").strip(),
        rewritten=str(data.get("rewritten") or "").strip(),
        sub_questions=[s for s in (data.get("sub_questions") or []) if isinstance(s, str) and s.strip()],
        keywords=[k for k in (data.get("keywords") or []) if isinstance(k, str) and k.strip()],
        needs_retrieval=not _CHITCHAT_RE.match(query),
        degraded=False,
    )
    if not qu.rewritten:
        qu.rewritten = normalized
    if not qu.sub_questions:
        qu.sub_questions = [qu.rewritten]
    if qu.intent not in INTENTS:
        qu.intent = "其他"
    qu.ms = (time.perf_counter() - t0) * 1000
    return qu
