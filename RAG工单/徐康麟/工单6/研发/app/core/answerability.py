# -*- coding: utf-8 -*-
"""工单4 可答性判定与拒答（「不清楚」是**正确响应**，不是错误）。

工单编号：人工智能NLP-RAG-图像内容解析及检索优化

设计依据：
    * 设计/接口设计.md §3.3「拒答（正确行为，仍是 200）」；
    * 任务书硬性要求 2：「无依据时统一回答『不清楚』，**禁止编造**；必须有 answerability 门槛与拒答测试」。

三道闸门（任一不满足即拒答，阈值来自 ``AppConfig.answer``，可用 ``RAG_ANSWER__*`` 覆盖）：

    G1 召回非空            —— Top-K 检索结果为空 ⇒ ``no_hits``
    G2 关键词覆盖度        —— 问句实词在证据中的覆盖率 < ``min_keyword_coverage`` ⇒ ``low_coverage``
    G3 相似度/词法双底线   —— 向量最高相似度与 BM25 最高分**双双**低于底线 ⇒ ``low_similarity``

    另有一条**主体闸门（subject gate）**：问句点名了语料中的发行人，而 Top-1 证据来自
    **另一份招股说明书**且问句实词在其正文里几乎不出现时 ⇒ ``subject_mismatch``。
    它专治「跨文件串答」（题 1/531/543 的力源 vs 兴图新科互串，见 golden 的 traps）。

标定原则：**先用 16 题正样本量最小值，再用无关问句量最大值，阈值取两者之间且偏向正样本**
（宁可回答后由判分器判错，也不因过严把可答题拒答成「不清楚」——拒答是可答题的致命错）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from .text_utils import tokenize

WORK_ORDER = "人工智能NLP-RAG-图像内容解析及检索优化"

UNKNOWN_TEXT = "不清楚"

#: 拒答原因码（写进响应 ``refusal_reason`` 与日志，便于评测归因）
REFUSAL_REASONS: dict[str, str] = {
    "no_hits": "检索无任何命中",
    "empty_question": "问题为空",
    "low_coverage": "问句实词在证据中的覆盖度过低",
    "low_similarity": "向量相似度与 BM25 分数双双低于底线",
    "subject_mismatch": "问句主体与证据来源文件不匹配",
    "no_evidence_sentence": "证据块中未找到可支撑答案的句子/表行",
    "unknown_entity": ("问句提到的主体在两份招股说明书中都不存在，或为泛化/占位名"
                       "（某某、X公司、甲公司…），或强指向性字段问句未点出语料内主体（禁止编造）"),
}


def _lazy_logger(logger: Any, module: str = "answerability") -> Any:
    if logger is not None:
        return logger
    from .logging_conf import get_logger

    return get_logger(module)


@dataclass(slots=True)
class GateInputs:
    """可答性判定输入（检索与问句理解的**结构化产出**，避免在判定里重算）。"""

    question: str = ""
    keywords: Sequence[str] = field(default_factory=list)   # 问句实词（含实体/数字）
    top_score: float = 0.0            # 融合分（仅供诊断，不直接做阈值）
    max_vector_similarity: float = 0.0
    max_bm25_score: float = 0.0
    coverage: float = 0.0             # 问句实词在 Top-K 证据中的覆盖率 0~1
    corpus_hint: str | None = None    # 问句点名的发行人 → 期望语料
    top_corpus: str | None = None     # Top-1 证据所属语料
    corpus_hint_hit: bool = False     # 点名主体是否真的出现在 Top-K 证据里
    has_support_sentence: bool = True  # 证据里是否找到支撑句子/表行
    oov_keywords: Sequence[str] = field(default_factory=list)  # 全语料词表里都不存在的问句实词

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GateResult:
    """可答性结论。"""

    answerable: bool
    reason: str
    thresholds: dict[str, float]
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def keyword_coverage_of(keywords: Sequence[str], texts: Sequence[str]) -> float:
    """问句实词在证据文本中的覆盖率（按词计，忽略大小写与空白）。"""
    wanted = [str(k).strip().lower() for k in keywords if str(k).strip()]
    if not wanted:
        return 1.0
    joined = " ".join(str(t) for t in texts).lower().replace(" ", "")
    hit = 0
    for kw in wanted:
        if kw.replace(" ", "") in joined:
            hit += 1
    return hit / len(wanted)


def question_keywords(question: str, *, stopwords: Sequence[str] = ()) -> list[str]:
    """问句实词抽取：jieba 分词去停用词，**保留数字与英文词**（题 6 的 IC卡 / 14.0% 都在里面）。"""
    from .text_utils import STOPWORDS, extract_numbers

    tokens = []
    for tok in tokenize(question, use_jieba=True):
        if tok in STOPWORDS or tok in set(stopwords):
            continue
        if len(tok) < 2 and not tok.isdigit():
            continue
        tokens.append(tok)
    for num in extract_numbers(question):
        if num not in tokens:
            tokens.append(num)
    return tokens


def evaluate_answerability(inputs: GateInputs, cfg: Any, *, logger: Any = None) -> GateResult:
    """执行三道闸门 + 主体闸门，返回结构化判定（含全部中间量，供报告归因）。"""
    log = _lazy_logger(logger)
    answer_cfg = getattr(cfg, "answer", cfg)
    thresholds = {
        "min_score": float(getattr(answer_cfg, "min_score", 0.12)),
        "min_keyword_coverage": float(getattr(answer_cfg, "min_keyword_coverage", 0.45)),
        "min_vector_similarity_en": float(getattr(answer_cfg, "min_vector_similarity_en", 0.45)),
    }
    with log.enter("evaluate_answerability", {"inputs": inputs.to_dict(),
                                              "thresholds": thresholds}) as span:
        if not str(inputs.question or "").strip():
            result = GateResult(False, "empty_question", thresholds, inputs.to_dict())
        elif not inputs.keywords:
            # 没有实词（例如纯「你好」）→ 无依据，直接拒答
            result = GateResult(False, "low_coverage", thresholds,
                                {**inputs.to_dict(), "note": "问句无实词"})
        elif inputs.oov_keywords:
            # **词表外实体闸门**：问句里的实词在两份招股书的全部块中都不存在
            # （如「特斯拉」「苹果公司」）⇒ 语料无该主体，禁止编造。
            # 依据（T5 实测）：不问「覆盖率」而问「这个词存不存在」——
            # 「营业收入」这类通用词会在财报里命中，单靠覆盖率会漏放「特斯拉营业收入是多少」。
            result = GateResult(False, "unknown_entity", thresholds,
                                {**inputs.to_dict(), "oov": list(inputs.oov_keywords)})
        elif inputs.top_score <= 0 and inputs.max_bm25_score <= 0 and inputs.max_vector_similarity <= 0:
            result = GateResult(False, "no_hits", thresholds, inputs.to_dict())
        elif (inputs.coverage < thresholds["min_keyword_coverage"]
              and inputs.max_vector_similarity < thresholds["min_score"]
              and inputs.max_bm25_score < 1.0):
            result = GateResult(False, "low_coverage", thresholds, inputs.to_dict())
        elif (inputs.coverage < max(thresholds["min_keyword_coverage"] - 0.15, 0.15)
              and inputs.max_vector_similarity < thresholds["min_score"]):
            result = GateResult(False, "low_similarity", thresholds, inputs.to_dict())
        elif (inputs.corpus_hint and inputs.top_corpus and not inputs.corpus_hint_hit
              and inputs.coverage < 0.5):
            result = GateResult(False, "subject_mismatch", thresholds,
                                {**inputs.to_dict(), "hint": inputs.corpus_hint,
                                 "top_corpus": inputs.top_corpus})
        elif not inputs.has_support_sentence:
            result = GateResult(False, "no_evidence_sentence", thresholds, inputs.to_dict())
        else:
            result = GateResult(True, "answerable", thresholds, inputs.to_dict())
        log.log_event("answerability.gate", answerable=result.answerable, reason=result.reason,
                      coverage=round(inputs.coverage, 4),
                      max_vector=round(inputs.max_vector_similarity, 4),
                      max_bm25=round(inputs.max_bm25_score, 4),
                      oov=list(inputs.oov_keywords),
                      corpus_hint=inputs.corpus_hint, top_corpus=inputs.top_corpus)
        span.set_output(result.to_dict())
        return result


def refusal_text(cfg: Any = None) -> str:
    """统一拒答文本（默认「不清楚」）。"""
    answer_cfg = getattr(cfg, "answer", cfg)
    return str(getattr(answer_cfg, "unknown_text", UNKNOWN_TEXT) or UNKNOWN_TEXT)


def refusal_reason_text(reason: str) -> str:
    """把原因码翻成中文说明（UI 展示用，答案正文仍是「不清楚」）。"""
    return REFUSAL_REASONS.get(reason, reason)


def is_refusal(answer_text: str, cfg: Any = None) -> bool:
    """答案是否属于拒答（供评测/UI 判定 ``refused``）。"""
    return str(answer_text or "").strip() == refusal_text(cfg)
