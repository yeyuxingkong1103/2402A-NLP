"""
多轮对话上下文消解（工单5 核心）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
工单编号：人工智能NLP-RAG-Query理解优化任务

工单5 要解决的问题：**单轮检索问答接不住「依赖上文」的追问**。

工单给出的 5 轮示例对话里，有 3 轮是「自己看不懂自己」的：

  Q2 他参与的哪个工程荣获了国家科技进步一等奖？      ←「他」是谁？
  Q3 这个公司的法定代表人是谁？                      ←「这个公司」是哪家？
  Q4 那武汉力源信息技术股份有限公司呢？              ← 到底问力源的什么？

前两轮是**指代消解**，第三轮是**省略补全**（只有主体、没有谓词，谓词要从上一轮继承）。
不处理的话，这三轮的检索式分别是「他参与的哪个工程…」「这个公司的法定代表人…」
「那武汉力源…呢」—— 指代词与「那…呢」在语料里根本不存在，检索必然召回无关文档。

--------------------------------------------------------------------------
两类现象，两条处理路径
--------------------------------------------------------------------------
| 现象 | 例子 | 判定 | 处理 |
|---|---|---|---|
| 指代 | 他 / 这个公司 / 该企业 / 其 | 命中 `MULTITURN_COREF_RE` | 用**最近一个带主体的轮次**替换掉指代词 |
| 省略 | 那X呢？ / X呢？ | 命中 `MULTITURN_ELLIPSIS_RE` | 用**上一轮的主题（topic）**补出谓词，套到本轮主体上 |

「向上回溯」而不是「只看上一轮」很重要：用户会连问省略句
（「那力源呢？」→「注册资本呢？」），中间那轮没有新主体，主体应继续沿用更早那轮。

--------------------------------------------------------------------------
四种运行模式（消融实验的单变量开关，`MULTITURN_MODE`）
--------------------------------------------------------------------------
| 模式 | 行为 | 对应实验臂 |
|---|---|---|
| `off` | 完全不做多轮，每轮都是独立问题 | T0 原系统（优化前） |
| `concat` | 把最近 N 轮**原文**直接拼在问题前面丢给检索 | T1 朴素拼接 |
| `rule` | 只走规则消解（零额外 LLM 调用） | T2 规则消解 |
| `rule+llm` | 规则优先；**规则结果不可信时**才调 LLM 兜底 | T3 交付配置 |

--------------------------------------------------------------------------
「规则不可信」的判据：还是用检索依据分（与工单2 的直通优化同一套思路）
--------------------------------------------------------------------------
规则消解出来的式子，**先花 ~26ms 试检索一次**：
  * 首条依据分 ≥ `MULTITURN_RULE_MIN_EVIDENCE` → 规则已经够了，**不调 LLM**（守住 ≤3s）；
  * 否则说明规则拼出来的式子召不回东西（可能是规则没覆盖的句型），
    才把「历史 + 规则候选」交给 LLM 重写一次。

这样设计的好处：**绝大多数轮次零额外 LLM 开销**，只有真正棘手的句型才付一次往返。
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from .config import (
    DOCS_BY_KEY,
    MULTITURN_COREF_RE,
    MULTITURN_ELLIPSIS_RE,
    MULTITURN_HISTORY_CHARS,
    MULTITURN_LLM_MAX_TOKENS,
    MULTITURN_MODE,
    WORK_ORDER_NO_QU,  # noqa: F401  工单编号：人工智能NLP-RAG-Query理解优化任务
)
from .session import Session, detect_subject, detect_topic

logger = logging.getLogger(__name__)

# 规则消解结果的「可信线」。与检索闸门（0.75）一致：
# 规则拼出来的式子如果连闸门都过不去，就说明这个式子是错的，值得再花一次 LLM。
MULTITURN_RULE_MIN_EVIDENCE = 0.75

_COREF = re.compile(MULTITURN_COREF_RE)
_ELLIPSIS = re.compile(MULTITURN_ELLIPSIS_RE)

# 省略补全时用的谓词模板：把「事」还原成一句能检索的话。
# 之所以不用「{subject}的{topic}是什么」这种统一句式 —— 检索是关键词匹配，
# 句式越短、关键词越集中，召回越稳。直接用「主体 + 事的短标签」。
_ELLIPSIS_TEMPLATE = "{subject}{topic}"


@dataclass
class Resolved:
    """本轮问题的上下文消解结果。"""

    question: str                 # 原始问题
    resolved: str                 # 消解后的自包含问题（真正送检索的）
    mode: str = "off"             # 实际生效的模式
    coref: bool = False           # 是否检测到指代
    ellipsis: bool = False        # 是否检测到省略式追问
    subject: str = ""             # 本轮主体（公司全称）
    doc_key: str = ""             # 本轮主体的文档键
    topic: str = ""               # 本轮「事」的短标签
    inherited_subject: str = ""   # 从历史继承来的主体（区别于本轮显式写的）
    inherited_topic: str = ""     # 从历史继承来的主题
    degraded: bool = False        # True = 上下文没能消解成功，退回原文
    llm_used: bool = False        # 是否真的调了 LLM 兜底
    rule_evidence: float = 0.0    # 规则候选式子的试检索依据分（判可信度用）
    ms: float = 0.0
    reason: str = ""              # 人类可读的判定说明（前端「检索链路」要展示）

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "resolved": self.resolved,
            "mode": self.mode,
            "coref": self.coref,
            "ellipsis": self.ellipsis,
            "subject": self.subject,
            "doc_key": self.doc_key,
            "topic": self.topic,
            "inherited_subject": self.inherited_subject,
            "inherited_topic": self.inherited_topic,
            "degraded": self.degraded,
            "llm_used": self.llm_used,
            "rule_evidence": round(self.rule_evidence, 4),
            "ms": round(self.ms, 2),
            "reason": self.reason,
        }


# ---------------------------------------------------------------- 规则层

def _replace_coref(question: str, subject: str) -> str:
    """
    把问题里的指代词换成主体全称。

    只替换第一个指代词：招股书问句里出现两个不同所指的概率极低，
    而「都换」在同指重复时反而会把句子改得别扭（如「他在他参与的…」）。
    """
    return _COREF.sub(subject, question, count=1)


def _probe(expr: str) -> float:
    """
    试检索取首条依据分（~26ms）。任何异常返回 0.0（= 视为不可信），
    不允许因为探测失败而改变主链路行为。
    """
    if not expr:
        return 0.0
    try:
        from .retriever import retrieve

        r = retrieve(expr, top_k=1, override_query=expr, apply_gate=False)
        return r.items[0].evidence if r.items else 0.0
    except Exception as exc:  # noqa: BLE001
        logger.warning("多轮试检索失败（%s），按不可信处理", exc)
        return 0.0


# ---------------------------------------------------------------- LLM 层

_MULTITURN_SYSTEM = """你是招股说明书问答系统的多轮对话理解模块。
用户在一个连续会话里提问，当前这一轮可能省略了主语或谓语，需要你结合上文补全。

你要输出一个**自包含的检索式**：把它单独拿出来看，也能明确知道问的是哪家公司、问的什么事。
要求：
1. 补全指代：如「他」「这个公司」「该企业」要替换成上文提到的公司全称。
2. 补全省略：如「那武汉力源信息技术股份有限公司呢？」，要继承上一轮问的「事」
   （上一轮问法定代表人，这一轮就是问力源的法定代表人）。
3. 只做补全，**不要回答问题**，不要添加上文没有的信息。
4. 如果上文不足以补全，resolved 就原样返回当前问题，并把 sufficient 设为 false。

只输出 JSON，不要解释：
{"resolved":"武汉力源信息技术股份有限公司的法定代表人是谁","subject":"武汉力源信息技术股份有限公司","topic":"法定代表人","sufficient":true}"""


def _format_history(session: Session, n: int = 3) -> str:
    """把最近几轮渲染成紧凑文本。总长受 `MULTITURN_HISTORY_CHARS` 限制。"""
    lines: list[str] = []
    for i, t in enumerate(session.recent(n), 1):
        lines.append(f"[第{i}轮] 用户：{t.q}")
        if t.resolved:
            lines.append(f"        系统理解为：{t.resolved}")
    text = "\n".join(lines)
    if len(text) > MULTITURN_HISTORY_CHARS:
        # 从**前面**截断：最近的一轮最重要，必须保住
        text = "…" + text[-MULTITURN_HISTORY_CHARS:]
    return text


def _llm_resolve(question: str, session: Session, rule_candidate: str) -> dict | None:
    """调 LLM 做一次上下文补全。失败返回 None（调用方降级到规则结果）。"""
    from .llm import chat_json

    hist = _format_history(session)
    if not hist:
        return None
    user = (
        f"【上文】\n{hist}\n\n"
        f"【当前问题】{question}\n"
        + (f"【规则模块的候选补全】{rule_candidate}\n" if rule_candidate else "")
        + "\n请输出 JSON。"
    )
    data = chat_json(
        [
            {"role": "system", "content": _MULTITURN_SYSTEM},
            {"role": "user", "content": user},
        ],
        max_tokens=MULTITURN_LLM_MAX_TOKENS,
    )
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------- 主入口

def resolve(question: str, session: Session | None, mode: str | None = None) -> Resolved:
    """
    把「当前问题 + 历史」消解成**自包含的检索式**。

    这一层**不碰检索、不碰生成**，只回答一个问题：
    「把用户这句话单独拿出来，系统应该拿什么式子去搜？」

    任何异常都不向外抛 —— 消解失败就退回原文，让主链路照常工作。
    """
    t0 = time.perf_counter()
    q = (question or "").strip()
    mode = (mode or MULTITURN_MODE or "off").strip().lower()

    res = Resolved(question=q, resolved=q, mode=mode)

    def _done(reason: str) -> Resolved:
        res.reason = reason
        res.ms = (time.perf_counter() - t0) * 1000
        return res

    if not q:
        return _done("空问题")

    if mode == "off":
        # T0：完全不看历史。但**仍然记录**本轮主体/主题，
        # 否则这个会话的历史永远是空的，后面的臂就无从对比。
        doc_key, subject = detect_subject(q)
        res.subject, res.doc_key = subject, doc_key
        res.topic = detect_topic(q)
        return _done("多轮关闭（单轮模式）")

    # ---- 1. 本轮自己认出的主体（显式 > 继承）
    doc_key_now, subject_now = detect_subject(q)
    coref = bool(_COREF.search(q))
    ellipsis = bool(_ELLIPSIS.match(q))

    prev_doc_key, prev_subject = session.last_subject() if session else ("", "")
    prev_topic = session.last_topic() if session else ""

    res.coref, res.ellipsis = coref, ellipsis

    # ---- 2. 决定本轮主体
    if subject_now:
        res.subject, res.doc_key = subject_now, doc_key_now
    elif coref and prev_subject:
        res.subject, res.doc_key = prev_subject, prev_doc_key
        res.inherited_subject = prev_subject

    # ---- 3. 决定本轮主题
    topic_now = detect_topic(q)
    if topic_now:
        res.topic = topic_now
    elif (coref or ellipsis) and prev_topic:
        res.topic = prev_topic
        res.inherited_topic = prev_topic
    elif coref or ellipsis:
        # 指代/省略，但历史里没有「事」→ 试试从上一轮的 resolved 里捞
        if session:
            res.topic = detect_topic(session.last_resolved())

    # ---- 4. concat 模式：不做消解，把历史原文拼上去（T1 臂）
    if mode == "concat":
        if session and session.turns:
            hist = " ".join(t.q for t in session.recent(2))
            res.resolved = f"{hist} {q}".strip()
            return _done("朴素拼接历史原文（不对指代/省略做处理）")
        return _done("无历史，按原文检索")

    # ---- 5. 规则消解
    candidate = q
    if coref and res.inherited_subject:
        candidate = _replace_coref(q, res.inherited_subject)
    elif ellipsis and res.subject and res.topic:
        candidate = _ELLIPSIS_TEMPLATE.format(subject=res.subject, topic=res.topic)

    need_llm = False
    if candidate == q and (coref or ellipsis):
        # 检测到指代/省略，但规则没能拼出新式子 → 规则不足以处理
        need_llm = True
        res.degraded = True
    elif candidate != q and mode == "rule+llm":
        # 规则拼出来了，但**先验证它靠不靠谱**：试检索一次看依据分
        ev = _probe(candidate)
        res.rule_evidence = ev
        if ev < MULTITURN_RULE_MIN_EVIDENCE:
            need_llm = True

    if need_llm and mode == "rule+llm" and session and session.turns:
        data = _llm_resolve(q, session, candidate)
        if data:
            llm_resolved = str(data.get("resolved") or "").strip()
            if llm_resolved and bool(data.get("sufficient", True)):
                res.resolved = llm_resolved
                res.llm_used = True
                res.degraded = False
                sj = str(data.get("subject") or "").strip()
                if sj:
                    res.subject = sj
                    dk, _ = detect_subject(sj)
                    if dk:
                        res.doc_key = dk
                tp = str(data.get("topic") or "").strip()
                if tp:
                    res.topic = tp
                return _done("规则结果依据分不足，由大模型结合上文补全")

    res.resolved = candidate
    if coref and res.inherited_subject and candidate != q:
        return _done(f"指代消解：用上文主体「{res.inherited_subject}」替换指代词")
    if ellipsis and res.topic:
        return _done(f"省略补全：继承上文主题「{res.topic}」，套到主体「{res.subject}」上")
    if coref or ellipsis:
        return _done("检测到指代/省略，但上文不足以补全，按原文检索（降级）")
    return _done("无指代/省略，按原文检索")


def resolve_with_history(question: str, history: list[str] | None,
                         mode: str | None = None) -> Resolved:
    """
    便捷入口：用**纯文本历史**（不带答案）跑一次消解。

    评测脚本与离线消融用得上（它们只有问题列表，没有真实会话对象）。
    """
    from .session import Session, Turn

    s = Session(session_id="__eval__")
    for h in (history or []):
        h = (h or "").strip()
        if not h:
            continue
        dk, subj = detect_subject(h)
        s.add(Turn(q=h, resolved=h, subject=subj, doc_key=dk, topic=detect_topic(h)))
    return resolve(question, s, mode=mode)
