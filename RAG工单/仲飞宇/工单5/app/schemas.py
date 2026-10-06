# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 工单编号：人工智能NLP-RAG-Query 理解优化任务
# 工单01 - 基于PDF文档的问答系统
# 工单02 - 基于PDF文档的问答系统的优化
# 工单05 - Query 理解优化（多轮对话）
"""API 请求/响应模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ----------------------------------------------------------------------
class HealthOut(BaseModel):
    ok: bool
    ollama: str = ""
    milvus: str = ""
    collection_rows: int = 0
    llm_model: str = ""
    embed_model: str = ""


# ----------------------------------------------------------------------
class CitationOut(BaseModel):
    page_label: str
    page_no: int
    chunk_type: str
    section_path: str = ""
    snippet: str = ""
    score: float = 0.0
    # 工单02：true = 这是邻块扩展补进来的上下文，不是答案的直接依据。
    # 前端渲染成灰色的「邻接补充」；评测的精确率分母也排除它。
    is_neighbor: bool = False


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    history: list[dict[str, str]] | None = None
    # 工单01「纯 LLM vs RAG 对比」：true 时不走检索，用于对照
    no_rag: bool = False
    stream: bool = True
    # 工单02：用哪套检索策略（baseline / delivered / optimized）。
    # 不传则用 settings.retrieval_profile 的默认值。
    profile: str | None = None
    # ---------- 工单05：多轮对话 ----------
    # 会话 id。不传 = 单轮（行为与工单01-04 完全一致，也继续吃响应缓存）。
    # 传了 = 多轮：走服务端会话（指代消解/省略补全），并**绕过响应缓存**
    # （同一句话在不同上下文下答案不同，缓存语义不成立）。
    session_id: str | None = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationOut] = []
    mode: Literal["rag", "no_rag"] = "rag"
    ttft_ms: int = 0          # 首 token 延迟 —— 工单01 ≤3s 的验收口径
    total_ms: int = 0         # 完整答案耗时（主动披露）
    retrieval_ms: int = 0
    n_candidates: int = 0
    n_filtered: int = 0
    # ---------- 工单02 ----------
    profile: str = ""          # 这次用的是哪套检索策略
    cached: bool = False       # 是否命中响应缓存（TTFT 会因此失真，必须暴露）
    context_chars: int = 0
    n_merged: int = 0
    n_added_neighbors: int = 0
    # ---------- 工单03 ----------
    # 实体路由：这题被限定在哪一份文档上检索。空 = 没过滤（没点名 / 跨文档）。
    # 必须暴露 —— 两份招股书章节结构雷同，出问题时第一件要确认的就是"路由到哪了"。
    routed_doc: str = ""
    route_matched: str = ""    # 命中的公司专名（排查用）
    route_fallback: bool = False  # 过滤后零召回、已回退全库
    # ---------- 工单05：多轮对话（Query 理解） ----------
    # 【为什么整块都要显式声明】pydantic v2 `extra="ignore"` —— 漏加字段不报错，
    # 只是**静默丢弃**，接口照常 200 而前端拿不到（见下方 EvalItemOut 的同类警告）。
    session_id: str = ""
    turn_index: int = 0            # 该会话内的第几轮（1 起）
    rewritten_question: str = ""   # 改写后的独立问题（= 真正拿去检索的问题）
    rewrite_method: str = ""       # none/rule-coref/rule-switch/llm/llm-failed
    rewrite_evidence: str = ""     # 人话依据（演示用）
    focus_entity: str = ""         # 本轮之后的会话焦点实体（规范全称）
    carried_intent: str = ""       # 继承的问点（话题切换时有值）
    session_expired: bool = False  # 上一轮的会话已过 TTL，本轮已按单轮重建
    history_chars: int = 0         # 送进 prompt 的 history 字数（预算闸门可观测）
    history_dropped: bool = False  # 因预算不足丢掉了最旧的轮次
    rewrite_ms: int = 0            # 改写耗时（规则层为 0；LLM 兜底才非 0）


# ----------------------------------------------------------------------
class IngestRequest(BaseModel):
    pdf_path: str | None = None
    rebuild: bool = False
    # 只解析前 N 页，供快速验证；None = 全量
    limit_pages: int | None = None


class IngestStatus(BaseModel):
    running: bool = False
    stage: str = "idle"
    done: int = 0
    total: int = 0
    message: str = ""
    error: str = ""
    stats: dict[str, Any] = {}


# ----------------------------------------------------------------------
class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: Literal["up", "down"]
    comment: str = ""
    citations: list[CitationOut] = []
    # 工单05：多轮界面里每轮各自反馈 —— 带上会话与轮次，便于按轮定位。
    session_id: str = ""
    turn_index: int = 0


class FeedbackOut(BaseModel):
    ok: bool
    received: int


# ----------------------------------------------------------------------
class EvalItemOut(BaseModel):
    id: int
    question: str
    category: str = ""
    # 【工单05 补】原先漏声明这三个字段，而 app.js 的「答案对照（逐题展开）」在渲染
    # it.reference_answer / it.evidence_page —— pydantic v2 `extra="ignore"` 让它们
    # **静默丢弃**，页面显示的是「人工参考答案（undefined）」+ 空白答案，接口却照常 200。
    # 这正是本文件 :121-126 那条警告说的坑，工单05 演示要复用评估页，顺手补齐。
    reference_answer: str = ""
    evidence_page: str = ""
    error: str = ""
    rag_answer: str = ""
    no_rag_answer: str = ""
    rule_hit: bool | None = None
    # 【工单05 补】同样漏声明：逐题表的「纯LLM命中」列与「路由」列就靠它。
    # 缺了它，页面上一律显示"未命中"和"—"（看着像模型全错、路由没生效）。
    no_rag_rule_hit: bool | None = None
    rule_detail: str = ""
    context_relevance: float | None = None
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_recall: float | None = None
    answer_correctness: float | None = None
    citations: list[str] = []
    ttft_ms: int = 0
    # ---------- 工单02：检索侧指标 ----------
    # 这几个字段与 evaluator.EvalItem 一一对应。
    # 注意：pydantic v2 默认 `extra="ignore"`，所以**漏加不会报错**，
    # 而是静默丢弃 —— `/api/evaluate/report` 会照常 200，只是前端拿不到这些字段。
    # 实测确认过：EvalSummary(n=10, 未知键=1) 不抛异常。
    # 这种"静默缺字段"比报错更难查，所以新增指标时两边必须同步改。
    profile: str = ""
    retrieved_pages: list[str] = []
    evidence_pages: list[str] = []
    page_precision: float | None = None
    page_recall: float | None = None
    context_keyword_coverage: float | None = None
    context_chars: int = 0
    n_merged: int = 0
    n_added_neighbors: int = 0
    # 【工单05 补】工单03 的路由三件套也漏声明了，前端逐题表的「路由」列读不到。
    routed_doc: str = ""
    route_matched: str = ""
    route_fallback: bool = False


class EvalSummary(BaseModel):
    n: int = 0
    profile: str = ""
    rule_hit_rate: float = 0.0
    rag_rule_hits: int = 0
    no_rag_rule_hits: int = 0
    avg_page_precision: float = 0.0
    avg_page_recall: float = 0.0
    avg_context_keyword_coverage: float = 0.0
    avg_context_chars: int = 0
    n_merged_total: int = 0
    n_added_neighbors_total: int = 0
    avg_context_relevance: float = 0.0
    avg_faithfulness: float = 0.0
    avg_answer_relevance: float = 0.0
    avg_context_recall: float = 0.0
    avg_answer_correctness: float = 0.0
    ttft_p50_ms: int = 0
    ttft_p95_ms: int = 0
    # n=10 时 P95 就是最大值，单独看会被离群点绑架；这三个是配套的口径披露
    ttft_under_3s: int = 0
    ttft_under_3s_rate: float = 0.0
    ttft_max_ms: int = 0
    warmup: bool = True
    total_p50_ms: int = 0
    total_p95_ms: int = 0
    generated_at: str = ""


class EvalReport(BaseModel):
    summary: EvalSummary
    items: list[EvalItemOut] = []


# ----------------------------------------------------------------------
class KBCollectionOut(BaseModel):
    name: str
    rows: int
    description: str = ""


class KBListOut(BaseModel):
    collections: list[KBCollectionOut] = []


class DeleteDocRequest(BaseModel):
    doc_id: str
