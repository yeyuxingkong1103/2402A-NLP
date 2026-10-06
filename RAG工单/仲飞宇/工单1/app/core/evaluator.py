# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单01 - 基于PDF文档的问答系统
"""
评估器：三轨指标，自实现（不依赖 ragas）。

【为什么不用 ragas】
ragas 走 Ollama 有三个必炸点：
  1. 它依赖结构化输出，而 Ollama 的 OpenAI 兼容端点 /v1 对
     response_format 的支持不合规（ollama#7978）；
  2. `think:false` 在 /v1 上不保证透传 —— 实测 /v1 返回的是空串，
     qwen3 的思考链会污染 judge 的 JSON；
  3. 依赖地狱：ragas 会拉进 langchain 全家桶，与 pymilvus 的
     `setuptools<81` 约束冲突风险高。
自己实现只要一个强约束 JSON 的调用 + 重试，可控得多。

【三轨设计】
轨1 规则化数值命中 —— **最可信**。5 题的答案是数字/人名，用正则硬匹配，
    没有 LLM 打分的主观性。演示时「命中 4/4 个年度数字」比
    「LLM 打了 0.87 分」有说服力得多，而且可复现、可争辩。
轨2 LLM judge —— 无参考答案也能算：Context Relevance / Faithfulness /
    Answer Relevance。用于衡量检索质量和幻觉程度。
轨3 对比人工参考答案 —— Context Recall / Answer Correctness。
    10 条参考答案写好了，工单02 的「90% 准确率」才有分子分母。

【数字匹配的归一化】
模型可能把 `5,520.00` 写成 `5520` 或 `5,520`，所以匹配前把候选答案与
关键词都去掉逗号、空格再比对 —— 否则会把正确回答误判为未命中。
"""

from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from app.config import PROJECT_ROOT, settings
from app.core.generator import Generator
from app.core.retriever import Retriever, build_context
from app.core.vectorstore import SearchHit

QUESTIONS_FILE = PROJECT_ROOT / "eval" / "questions.json"

# ----------------------------------------------------------------------
# 参考答案的「脱敏原文」提示：招股书里大量使用「某」代替真实名称
# （如《某视频技术规范1.0》），评估时必须原样接受，不能因"名字不完整"扣分。
DESENSITIZED_NOTE = (
    "注意：原文中大量使用「某」字对敏感信息脱敏（例如《某视频技术规范1.0》、"
    "「某情报、指挥、控制与通信网络一体化工程」）。这些是原文原貌，"
    "回答中出现带「某」的名称属于正确，不得因此判为错误或不完整。"
)


@dataclass
class EvalItem:
    id: int
    question: str
    category: str = ""
    reference_answer: str = ""
    evidence_page: str = ""
    rag_answer: str = ""
    no_rag_answer: str = ""
    citations: list[str] = field(default_factory=list)
    rule_hit: bool | None = None
    rule_detail: str = ""
    no_rag_rule_hit: bool | None = None
    context_relevance: float | None = None
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_recall: float | None = None
    answer_correctness: float | None = None
    ttft_ms: int = 0
    total_ms: int = 0
    error: str = ""


# ----------------------------------------------------------------------
# 轨1：规则化数值命中
# ----------------------------------------------------------------------
def _norm_digits(s: str) -> str:
    """归一化数字写法差异：去逗号、去空格、全角转半角。"""
    s = s.replace("，", ",").replace("．", ".").replace("％", "%")
    s = s.replace(" ", "").replace(",", "")
    return s


_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _numbers_in(s: str) -> set[float]:
    """
    抽取文本中所有数字的数值。

    【为什么需要】同一事实有多种合法写法，字符串匹配太脆：
      真值 `15,000.00万元`，模型可能答 `15000万元`、`1.5亿元` 的不同形式里
      前两种应算命中。按**数值**比较即可统一。
      注意 `5,520.00` 与 `5,520万元` 数值都是 5520.0，都能对上。
    """
    out: set[float] = set()
    for m in _NUM_RE.finditer(s.replace(",", "").replace("，", "")):
        try:
            out.add(round(float(m.group()), 2))
        except ValueError:
            continue
    return out


def _keyword_is_numeric(kw: str) -> bool:
    """关键词是否本质上是数值（去掉数字、百分号、千分位、小数点后为空）。"""
    stripped = re.sub(r"[\d,.\s%％]", "", kw)
    return stripped == "" and bool(re.search(r"\d", kw))


def rule_hit(answer: str, keywords: Sequence[str], rule_type: str = "any"
              ) -> tuple[bool, str]:
    """
    规则化命中判定。
    rule_type='all' → 全部关键词都要出现（多值答案，如四个年度的收入）
    rule_type='any' → 任一出现即可（同一事实存在多种合法表述）
    """
    if not keywords:
        return False, "无关键词"
    norm = _norm_digits(answer)
    answer_nums = _numbers_in(answer)

    hits: list[str] = []
    for k in keywords:
        # ① 先做字符串包含（精确、快）
        if _norm_digits(k) in norm:
            hits.append(k)
            continue
        # ② 数值型关键词退化为数值比较（容忍 15,000.00 / 15000 / 5,520万 写法差异）
        if _keyword_is_numeric(k):
            want = _numbers_in(k)
            if want and want <= answer_nums:
                hits.append(k)
                continue

    missed = [k for k in keywords if k not in hits]
    ok = (len(hits) == len(keywords)) if rule_type == "all" else bool(hits)
    detail = f"命中 {len(hits)}/{len(keywords)}：{hits}"
    if missed:
        detail += f"；缺失：{missed}"
    return ok, detail


# ----------------------------------------------------------------------
# 轨2/轨3：LLM judge
# ----------------------------------------------------------------------
_JUDGE_SYS = (
    "你是一个严格的评估员。只输出 JSON，不要输出任何解释文字、不要用 markdown 代码块。"
)

_JUDGE_SCHEMA_HINT = (
    '你必须只输出如下 JSON 结构：\n'
    '{"context_relevance": <0到1的小数>, "faithfulness": <0到1的小数>, '
    '"answer_relevance": <0到1的小数>, "context_recall": <0到1的小数>, '
    '"answer_correctness": <0到1的小数>, "reason": "<一句话中文理由>"}'
)


def _parse_json(text: str) -> dict[str, Any] | None:
    """从模型输出里抠出 JSON —— 容忍它偶尔加代码块围栏。"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def _clamp01(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, f))


class LLMJudge:
    """单指标单次调用 + 重试，避免一次失败全盘皆输。"""

    def __init__(self, generator: Generator | None = None, retries: int = 2) -> None:
        self.gen = generator or Generator()
        self.retries = retries

    async def judge(self, question: str, context: str, answer: str,
                    reference: str) -> dict[str, Any]:
        prompt = f"""请评估下面这次检索问答的质量，按 0~1 打分（1 最好）。

{DESENSITIZED_NOTE}

【用户问题】
{question}

【检索到的资料片段】
{context[:2500] if context else "（无检索资料）"}

【系统回答】
{answer}

【人工参考答案】
{reference}

评分项：
- context_relevance：检索到的资料与问题是否相关（无资料则为 0）
- faithfulness：系统回答是否完全有资料支撑、无编造（无资料且回答"未提及"应给 1）
- answer_relevance：回答是否切题、是否直接回应了问题
- context_recall：资料是否覆盖了参考答案所需的全部要点
- answer_correctness：回答与参考答案的事实一致程度

{_JUDGE_SCHEMA_HINT}"""

        msgs = [{"role": "system", "content": _JUDGE_SYS},
                {"role": "user", "content": prompt}]

        last_err = ""
        for _ in range(self.retries + 1):
            try:
                raw = await self.gen.client.chat(
                    msgs, fmt="json", temperature=0.0, max_tokens=300
                )
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
                continue
            obj = _parse_json(raw)
            if obj is None:
                last_err = f"JSON 解析失败：{raw[:120]!r}"
                continue
            return {
                "context_relevance": _clamp01(obj.get("context_relevance")),
                "faithfulness": _clamp01(obj.get("faithfulness")),
                "answer_relevance": _clamp01(obj.get("answer_relevance")),
                "context_recall": _clamp01(obj.get("context_recall")),
                "answer_correctness": _clamp01(obj.get("answer_correctness")),
                "reason": str(obj.get("reason", ""))[:200],
            }
        return {"error": last_err}


# ----------------------------------------------------------------------
class Evaluator:
    def __init__(self) -> None:
        self.retriever = Retriever()
        self.generator = Generator()
        self.judge = LLMJudge(self.generator)

    # ------------------------------------------------------------------
    @staticmethod
    def load_questions(path=None) -> list[dict[str, Any]]:
        p = path or QUESTIONS_FILE
        data = json.loads(open(p, encoding="utf-8").read())
        return data["questions"]

    # ------------------------------------------------------------------
    async def evaluate_one(self, q: dict[str, Any], *,
                           with_no_rag: bool = True,
                           with_judge: bool = True) -> EvalItem:
        item = EvalItem(
            id=q["id"], question=q["question"],
            category=q.get("category", ""),
            reference_answer=q.get("reference_answer", ""),
            evidence_page=q.get("evidence_page", ""),
        )
        t0 = time.perf_counter()
        try:
            # ---------- RAG ----------
            rres = await self.retriever.retrieve(item.question)
            hits: list[SearchHit] = rres.hits
            ctx = build_context(hits, query=item.question)
            item.citations = [h.page_label for h in hits]

            t_gen = time.perf_counter()
            item.rag_answer = await self.generator.generate(
                item.question, hits, context=ctx
            )
            item.ttft_ms = int((time.perf_counter() - t_gen) * 1000)  # 非流式，近似
            item.total_ms = int((time.perf_counter() - t0) * 1000)

            # ---------- 轨1 ----------
            ok, detail = rule_hit(item.rag_answer, q.get("rule_keywords", []),
                                  q.get("rule_type", "any"))
            item.rule_hit, item.rule_detail = ok, detail

            # ---------- 轨1 对照：纯 LLM ----------
            if with_no_rag:
                item.no_rag_answer = await self.generator.baseline_no_rag(item.question)
                ok2, _ = rule_hit(item.no_rag_answer, q.get("rule_keywords", []),
                                  q.get("rule_type", "any"))
                item.no_rag_rule_hit = ok2

            # ---------- 轨2/轨3 ----------
            if with_judge:
                j = await self.judge.judge(item.question, ctx, item.rag_answer,
                                           item.reference_answer)
                if "error" in j:
                    item.error = j["error"]
                else:
                    item.context_relevance = j["context_relevance"]
                    item.faithfulness = j["faithfulness"]
                    item.answer_relevance = j["answer_relevance"]
                    item.context_recall = j["context_recall"]
                    item.answer_correctness = j["answer_correctness"]

        except Exception as e:  # noqa: BLE001
            item.error = f"{type(e).__name__}: {e}"
        return item

    # ------------------------------------------------------------------
    async def evaluate_all(self, *, limit: int | None = None,
                           with_no_rag: bool = True,
                           with_judge: bool = True,
                           progress=None) -> dict[str, Any]:
        qs = self.load_questions()
        if limit:
            qs = qs[:limit]

        items: list[EvalItem] = []
        for i, q in enumerate(qs, 1):
            items.append(await self.evaluate_one(
                q, with_no_rag=with_no_rag, with_judge=with_judge))
            if progress:
                progress(i, len(qs), items[-1])

        return self.build_report(items)

    # ------------------------------------------------------------------
    @staticmethod
    def _avg(vals: Sequence[float | None]) -> float:
        xs = [v for v in vals if v is not None]
        return round(statistics.mean(xs), 4) if xs else 0.0

    @staticmethod
    def _pct(vals: Sequence[int], p: float) -> int:
        xs = sorted(v for v in vals if v)
        if not xs:
            return 0
        idx = min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1))))
        return xs[idx]

    def build_report(self, items: Sequence[EvalItem]) -> dict[str, Any]:
        n = len(items)
        rag_hits = sum(1 for it in items if it.rule_hit)
        no_rag_hits = sum(1 for it in items if it.no_rag_rule_hit)

        summary = {
            "n": n,
            "rule_hit_rate": round(rag_hits / n, 4) if n else 0.0,
            "rag_rule_hits": rag_hits,
            "no_rag_rule_hits": no_rag_hits,
            "avg_context_relevance": self._avg([it.context_relevance for it in items]),
            "avg_faithfulness": self._avg([it.faithfulness for it in items]),
            "avg_answer_relevance": self._avg([it.answer_relevance for it in items]),
            "avg_context_recall": self._avg([it.context_recall for it in items]),
            "avg_answer_correctness": self._avg([it.answer_correctness for it in items]),
            "ttft_p50_ms": self._pct([it.ttft_ms for it in items], 50),
            "ttft_p95_ms": self._pct([it.ttft_ms for it in items], 95),
            "total_p50_ms": self._pct([it.total_ms for it in items], 50),
            "total_p95_ms": self._pct([it.total_ms for it in items], 95),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        return {
            "summary": summary,
            "items": [it.__dict__ for it in items],
        }

    # ------------------------------------------------------------------
    @staticmethod
    def save_report(report: dict[str, Any]) -> dict[str, str]:
        """同时落 JSON 与 CSV —— 前端表格用 JSON，交作业用 CSV。"""
        import csv
        out = settings.data_path / "eval"
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        jp = out / f"eval-{stamp}.json"
        jp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "eval-latest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        cp = out / f"eval-{stamp}.csv"
        cols = ["id", "question", "category", "rule_hit", "no_rag_rule_hit",
                "context_relevance", "faithfulness", "answer_relevance",
                "context_recall", "answer_correctness", "ttft_ms", "total_ms",
                "citations", "rag_answer", "no_rag_answer", "rule_detail"]
        with cp.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for it in report["items"]:
                row = dict(it)
                row["citations"] = " / ".join(it.get("citations") or [])
                w.writerow(row)
        return {"json": str(jp), "csv": str(cp)}
