"""
Query 理解模块
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化

对应工单功能需求「Query理解：意图识别 / 消歧 / 分解与抽象」。

实现方式是**一次轻量 LLM 调用**（qwen-flash 实测 ~0.4s，max_tokens 锁到 320），
输出结构化 JSON。之所以不拆成三次调用：本工单硬性要求端到端 < 3 秒，
每多一次往返就多 0.4~1 秒预算。

--------------------------------------------------------------------------
工单2 优化：「高置信直通」（QU bypass）
--------------------------------------------------------------------------
实测（10 题端到端拆解）：Query 理解平均 **962ms**，占总耗时 **48%**，
而检索本身只要 **26ms**。也就是说，为了「改写一下检索式」，我们付掉了整个
延迟预算的一半。但改写对结果的影响远没有这么值钱——对「主语明确、无指代、
无多跳」的直陈式问题，规则归一化后的原问题已经能稳定召回正确答案。

所以改成**先花 26ms 试一次检索，再决定要不要花 900ms 调 LLM**：

    规则归一化 → 试检索(top1) → 依据分 ≥ 0.88 ？
                                  是 → 直通，跳过 LLM（省 ~900ms）
                                  否 → 老实调 LLM 做 Query 理解

三个保守设计，避免为了快而牺牲质量：
  1) 旁路阈值 0.88 **远高于** 闸门阈值（.env 里是 0.80），留足余量；
     只是「差不多」，一定要走 LLM 理解。
  2) 命中指代/多跳/比较类词（它、该、分别、对比、哪些、为什么……）一律不旁路 ——
     这类问题恰恰是 LLM 改写最能帮上忙的地方。
  3) 直通只影响「检索式从哪来」，不影响检索与生成，链路其余部分完全一致。

--------------------------------------------------------------------------
降级策略（很重要）
--------------------------------------------------------------------------
  LLM 超时/报错/返回非法 JSON 时，**不能**让整个问答失败。
  退回「规则归一化结果」继续检索 —— 用户拿到的答案质量略降，但链路不断。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .config import (
    QU_BYPASS_BLOCK_RE,
    QU_BYPASS_ENABLED,
    QU_BYPASS_MIN_EVIDENCE,
    QUERY_UNDERSTANDING_ENABLED,
    WORK_ORDER_NO,  # noqa: F401
    WORK_ORDER_NO_OPT,  # noqa: F401
)
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
    degraded: bool = False       # True 表示走了规则降级分支（LLM 没帮上忙）
    bypassed: bool = False       # True 表示走了「高置信直通」，有意跳过 LLM（工单2 优化）
    bypass_evidence: float = 0.0  # 直通时的依据分，用于回看这个决策是否合理
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
            "bypassed": self.bypassed,
            "bypass_evidence": round(self.bypass_evidence, 4),
            "ms": round(self.ms, 2),
        }


# 与文档无关的纯闲聊。**只有命中这条才跳过检索**，见 understand() 里的说明。
_CHITCHAT_RE = re.compile(
    r"^\s*(你好|您好|hi|hello|在吗|谢谢|感谢|再见|拜拜|"
    r"你是谁|你叫什么|介绍一下你自己|你能做什么)\s*[!！。.~?？]*\s*$",
    re.I,
)


# ---------------------------------------------------------------- 高置信直通（工单2 优化）

_BYPASS_BLOCK = re.compile(QU_BYPASS_BLOCK_RE)
# 太长的问题往往包含多个子句/条件，交给 LLM 理解更稳
_BYPASS_MAX_CHARS = 60


def _can_bypass(query: str) -> bool:
    """
    判断这个问题**是否允许**走直通。这里只做「不满足就一定要 LLM」的排除，
    真正的放行还要看试检索的依据分（见 _probe_evidence）。
    """
    q = (query or "").strip()
    if not q or len(q) > _BYPASS_MAX_CHARS:
        return False
    if _BYPASS_BLOCK.search(q):
        return False
    return True


def _probe_evidence(normalized: str) -> float | None:
    """
    用规则归一化后的式子试检索一次，返回 top1 依据分。

    成本 ~26ms（本地向量 + BM25），换来的是「是否需要付 900ms 调 LLM」的决策依据。
    任何异常都返回 None（= 不旁路，老实走 LLM），不允许因为探测失败而改变主链路行为。
    """
    try:
        from .retriever import retrieve

        r = retrieve(normalized, top_k=1, override_query=normalized, apply_gate=False)
        return r.items[0].evidence if r.items else 0.0
    except Exception as exc:  # noqa: BLE001
        logger.warning("直通探测失败（%s），改走 LLM 理解", exc)
        return None


def _rule_based(query: str, normalized: str, degraded: bool = True) -> QueryUnderstanding:
    """不依赖 LLM 的兜底理解。degraded=False 用于「有意直通」这种非降级场景。"""
    qu = QueryUnderstanding(
        original=query,
        normalized=normalized,
        rewritten=normalized,
        sub_questions=[normalized] if normalized else [],
        degraded=degraded,
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

    # ---- 工单2 优化：高置信直通。先花 26ms 试一次，够准就不调 LLM 了。
    if QU_BYPASS_ENABLED and _can_bypass(query):
        ev = _probe_evidence(normalized)
        if ev is not None and ev >= QU_BYPASS_MIN_EVIDENCE:
            qu = _rule_based(query, normalized, degraded=False)
            qu.bypassed = True
            qu.bypass_evidence = ev
            qu.ms = (time.perf_counter() - t0) * 1000
            logger.info("QU 直通：依据分 %.4f ≥ %.2f，跳过 LLM（%.1fms）",
                        ev, QU_BYPASS_MIN_EVIDENCE, qu.ms)
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
