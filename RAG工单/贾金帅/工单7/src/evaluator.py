"""
RAG 评估模块
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

对应工单：「选择 RAG 评估体系进行评估，返回评估结果」+「对比基于 pdf 的返回结果
和只使用 LLM 返回的答案的对比分析」。

评估体系：RAGAS 四项核心指标，用 LLM-as-judge 实现（自带实现，不依赖 ragas 包）：

    faithfulness       忠实度   —— 答案里的每个事实是否都能被检索到的上下文支撑
    answer_relevancy   答案相关性 —— 答案是否切题（不含无关内容、不答非所问）
    context_precision  上下文精确率 —— 召回的片段里有多少是与问题相关的
    context_recall     上下文召回率 —— 参考答案里的要点有多少能从召回片段中找到
                                   （需要参考答案；工单 10 题已人工核对补入）

为什么自带实现而不用 ragas 包：
  1) ragas 会拉进 langchain-community 一整套依赖（百 MB 级），而本机内存常态只剩
     几百 MB，装完还要和 torch 抢内存；
  2) 上面四项的定义本身是清晰的、可自查的，用同样的 prompt 逻辑实现，
     结果可复现、可人工复核；
  3) 不引入新依赖 = 少一处版本冲突。

**串行执行**：本机 LLM 侧有账号级限流，并发只会把请求全打成 429，
而重试退避会让「跑几分钟、一条分数都没有」。宁可慢，不可假死。
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from . import llm, rag
from .config import EVAL_DIR, LLM_JUDGE_MODEL, WORK_ORDER_NO  # noqa: F401

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = (
    "你是一名严格的 RAG 系统评测员。你只依据给定的材料打分，不引入外部知识。"
    "必须输出 JSON，不要输出任何解释性文字。"
)


def _judge(prompt: str, max_tokens: int = 300) -> dict | None:
    """调用评审模型，返回 JSON。失败返回 None（调用方按缺失处理，不中断整轮评估）。"""
    return llm.chat_json(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        model=LLM_JUDGE_MODEL,
        max_tokens=max_tokens,
    )


# ------------------------------------------------------------------ 四个指标


def score_faithfulness(answer: str, contexts: list[str]) -> dict:
    """忠实度：答案的每个断言是否被上下文支撑。"""
    if not answer.strip():
        return {"score": 0.0, "reason": "答案为空"}
    ctx = "\n\n".join(contexts)[:6000]
    prompt = f"""【上下文】
{ctx}

【答案】
{answer}

请判断：答案中的每一个事实性断言，是否都能在上下文里找到直接依据？
把答案拆成若干条独立断言，逐条判定 supported（有依据）或 unsupported（无依据/与上下文矛盾）。
打分规则：score = supported 条数 / 总条数（0~1）。
上下文为空且答案仍在陈述事实 → score 直接为 0。

只输出 JSON：{{"claims":[{{"text":"...","supported":true}}],"score":0.0,"reason":"一句话说明"}}"""
    out = _judge(prompt, max_tokens=800)
    if not isinstance(out, dict):
        return {"score": None, "reason": "评审调用失败"}
    try:
        score = float(out.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    return {"score": round(max(0.0, min(1.0, score)), 4), "reason": str(out.get("reason", ""))[:200],
            "claims": out.get("claims", [])}


def score_answer_relevancy(question: str, answer: str) -> dict:
    """答案相关性：答案是否切题、有没有答非所问或回避。"""
    if not answer.strip():
        return {"score": 0.0, "reason": "答案为空"}
    prompt = f"""【问题】
{question}

【答案】
{answer}

请判断该答案对问题的相关性：
- 1.0：直接、完整地回答了问题；数据与口径齐全
- 0.7~0.9：基本回答了，但遗漏部分要素
- 0.3~0.6：只沾边，或答非所问，或大量无关内容
- 0.0~0.2：完全无关，或只是拒绝回答
注意：「资料中没有相关内容」若问题确实超出材料范围则不算错，但若材料里明明有则判低分。

只输出 JSON：{{"score":0.0,"reason":"一句话说明"}}"""
    out = _judge(prompt, max_tokens=200)
    if not isinstance(out, dict):
        return {"score": None, "reason": "评审调用失败"}
    try:
        score = float(out.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    return {"score": round(max(0.0, min(1.0, score)), 4), "reason": str(out.get("reason", ""))[:200]}


def score_context_precision(question: str, contexts: list[str]) -> dict:
    """上下文精确率：召回的片段有多少真的与问题相关。"""
    if not contexts:
        return {"score": 0.0, "reason": "无召回片段"}
    numbered = "\n\n".join(f"[片段{i}] {c[:900]}" for i, c in enumerate(contexts, 1))
    prompt = f"""【问题】
{question}

【召回片段】
{numbered}

逐条判断每个片段是否与问题相关（能提供回答问题所需信息）。
打分规则：score = 相关片段数 / 总片段数（0~1）。

只输出 JSON：{{"relevant":[true,false],"score":0.0,"reason":"一句话说明"}}"""
    out = _judge(prompt, max_tokens=400)
    if not isinstance(out, dict):
        return {"score": None, "reason": "评审调用失败"}
    try:
        score = float(out.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    return {"score": round(max(0.0, min(1.0, score)), 4), "reason": str(out.get("reason", ""))[:200]}


def score_context_recall(question: str, contexts: list[str], reference: str) -> dict:
    """上下文召回率：参考答案的要点有多少能在召回片段里找到。需要人工参考答案。"""
    if not reference.strip():
        return {"score": None, "reason": "无参考答案，跳过"}
    if not contexts:
        return {"score": 0.0, "reason": "无召回片段"}
    ctx = "\n\n".join(contexts)[:6000]
    prompt = f"""【问题】
{question}

【参考答案】
{reference}

【召回片段】
{ctx}

请把参考答案拆成若干条关键要点，逐条判断能否在召回片段中找到依据。
打分规则：score = 有依据的要点数 / 总要点数（0~1）。

只输出 JSON：{{"points":[{{"text":"...","covered":true}}],"score":0.0,"reason":"一句话说明"}}"""
    out = _judge(prompt, max_tokens=800)
    if not isinstance(out, dict):
        return {"score": None, "reason": "评审调用失败"}
    try:
        score = float(out.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    return {"score": round(max(0.0, min(1.0, score)), 4), "reason": str(out.get("reason", ""))[:200]}


def score_answer_correctness(question: str, answer: str, reference: str) -> dict:
    """答案正确性：与参考答案比对（事实是否一致、有无编造）。"""
    if not reference.strip():
        return {"score": None, "reason": "无参考答案，跳过"}
    prompt = f"""【问题】
{question}

【参考答案】
{reference}

【待评答案】
{answer}

请比对：待评答案中的关键事实（数字、名称、年份）与参考答案是否一致？
- 一致且完整：1.0
- 基本一致但缺部分要素：0.6~0.9
- 有明显错误数字/名称：0.1~0.4
- 与参考答案矛盾（看起来是编造的）：0.0

只输出 JSON：{{"score":0.0,"reason":"一句话说明"}}"""
    out = _judge(prompt, max_tokens=300)
    if not isinstance(out, dict):
        return {"score": None, "reason": "评审调用失败"}
    try:
        score = float(out.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    return {"score": round(max(0.0, min(1.0, score)), 4), "reason": str(out.get("reason", ""))[:200]}


# ------------------------------------------------------------------ 跑一轮评估


@dataclass
class CaseResult:
    qid: int
    question: str
    reference: str = ""
    rag_answer: str = ""
    llm_answer: str = ""
    citations: list = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)         # 逐块，供上下文精确率使用
    judge_contexts: list[str] = field(default_factory=list)   # 含页码表头，供忠实度/召回率使用
    timing: dict = field(default_factory=dict)
    gated: bool = False
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.qid,
            "question": self.question,
            "reference": self.reference,
            "gated": self.gated,
            "rag_answer": self.rag_answer,
            "llm_only_answer": self.llm_answer,
            "citations": self.citations,
            "timing": self.timing,
            "metrics": self.metrics,
        }


def load_questions(dataset: str = "ticket_questions.json") -> list[dict]:
    path = EVAL_DIR / dataset
    if not path.exists():
        raise FileNotFoundError(f"评测集不存在：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(
    questions: list[dict],
    with_metrics: bool = True,
    top_k: int | None = None,
    progress=print,
) -> dict:
    """逐题跑 RAG + 纯 LLM，并（可选）计算指标。全程串行。"""
    cases: list[CaseResult] = []
    t_all = time.perf_counter()

    for i, item in enumerate(questions, 1):
        qid = int(item.get("id", i))
        question = item["question"]
        reference = (item.get("reference") or "").strip()
        progress(f"[{i}/{len(questions)}] id={qid} {question[:40]}…")

        case = CaseResult(qid=qid, question=question, reference=reference)

        # --- RAG
        try:
            a = rag.answer(question, top_k=top_k) if top_k else rag.answer(question)
            case.rag_answer = a.answer
            case.citations = a.citations
            case.contexts = [c["text"] for c in a.citations]
            # 忠实度/召回率要的是**真正喂给生成模型的那份上下文**（含「来源：…第X页」表头）。
            # 只喂引用片段正文，评审模型看不到页码，会把正确的 [第X页] 引用判成
            # unsupported —— 那是评测口径问题，不是生成质量问题。
            # 注意：上下文精确率仍必须用**逐块**的 contexts，不能塞成一整块，
            # 否则「相关片段/总片段」会退化成单样本，指标失去意义。
            case.judge_contexts = [a.context_text] if a.context_text else case.contexts
            case.timing = a.timing
            case.gated = a.mode == "no_evidence"
        except Exception as exc:  # noqa: BLE001
            logger.exception("RAG 分支失败 id=%s", qid)
            case.rag_answer = f"（RAG 失败：{exc}）"

        # --- 纯 LLM 对照组
        try:
            b = rag.answer_llm_only(question)
            case.llm_answer = b.answer
        except Exception as exc:  # noqa: BLE001
            case.llm_answer = f"（LLM 失败：{exc}）"

        # --- 指标
        if with_metrics:
            m: dict = {}
            jc = case.judge_contexts or case.contexts
            m["faithfulness"] = score_faithfulness(case.rag_answer, jc)
            m["answer_relevancy"] = score_answer_relevancy(question, case.rag_answer)
            m["context_precision"] = score_context_precision(question, case.contexts)
            if reference:
                m["context_recall"] = score_context_recall(question, jc, reference)
                m["answer_correctness"] = score_answer_correctness(question, case.rag_answer, reference)
                m["llm_only_correctness"] = score_answer_correctness(question, case.llm_answer, reference)
                m["llm_only_relevancy"] = score_answer_relevancy(question, case.llm_answer)
                m["llm_only_faithfulness"] = score_faithfulness(case.llm_answer, jc)
            case.metrics = m
            progress("      RAG: " + _fmt_metrics(m))

        cases.append(case)

    report = {
        "work_order_no": WORK_ORDER_NO,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_cases": len(cases),
        "elapsed_seconds": round(time.perf_counter() - t_all, 1),
        "summary": summarize(cases),
        "cases": [c.to_dict() for c in cases],
    }
    return report


def _fmt_metrics(m: dict) -> str:
    parts = []
    for k in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        v = (m.get(k) or {}).get("score")
        parts.append(f"{k}={v if v is not None else '-'}")
    return " ".join(parts)


def _avg(values: list) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    return round(sum(nums) / len(nums), 4) if nums else None


def summarize(cases: list[CaseResult]) -> dict:
    keys = ["faithfulness", "answer_relevancy", "context_precision",
            "context_recall", "answer_correctness", "llm_only_correctness",
            "llm_only_relevancy", "llm_only_faithfulness"]
    out: dict = {}
    for k in keys:
        out[k] = _avg([(c.metrics.get(k) or {}).get("score") for c in cases])
    totals = [c.timing.get("total_ms") for c in cases if c.timing.get("total_ms")]
    out["avg_total_ms"] = _avg(totals)
    out["max_total_ms"] = max(totals) if totals else None
    out["within_3s_ratio"] = (
        round(sum(1 for t in totals if t <= 3000) / len(totals), 4) if totals else None
    )
    out["gated_cases"] = sum(1 for c in cases if c.gated)
    return out
