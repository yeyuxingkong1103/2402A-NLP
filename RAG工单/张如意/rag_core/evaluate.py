# -*- coding: utf-8 -*-
"""
RAG 评估模块
工单编号：人工智能NLP-RAG-功能测试及评估
          人工智能NLP-RAG-Graph RAG 优化任务

按 RAGAS 论文（Es et al., 2023）的方法论自实现四大核心指标，
同时提供对官方 `ragas` 包的适配层（装了就用官方实现，没装就用自实现）：

  · Faithfulness       忠实度     —— 答案中的论断有多少能被上下文支撑（抗幻觉）
  · Answer Relevancy   答案相关性 —— 答案是否切题（反向生成问题算相似度）
  · Context Precision  上下文精度 —— 检索到的片段里，有用的是否排在前面
  · Context Recall     上下文召回 —— 标准答案的信息有多少能从检索上下文里找到

另附检索层指标：Hit Rate / MRR / Recall@k / 关键词命中率。
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config, llm


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------
@dataclass
class EvalRecord:
    """单个问题的评估明细。"""
    qid: int | str
    question: str
    answer: str
    ground_truth: str = ""
    contexts: list[str] = field(default_factory=list)
    reference_doc: str = ""              # 该问题应命中的文档
    retrieved_docs: list[str] = field(default_factory=list)
    retrieved_pages: list[int] = field(default_factory=list)
    latency: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.qid, "question": self.question,
            "answer": self.answer, "ground_truth": self.ground_truth,
            "reference_doc": self.reference_doc,
            "retrieved_docs": self.retrieved_docs,
            "retrieved_pages": self.retrieved_pages,
            "latency": round(self.latency, 3),
            "contexts": [c[:300] for c in self.contexts],
            **{k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in self.metrics.items()},
        }


# ---------------------------------------------------------------------------
# 论断分解（RAGAS 各指标的基础步骤）
# ---------------------------------------------------------------------------
_CLAIM_SYS = """把给定的文本拆解为若干条「原子论断」。
原子论断 = 一个不可再分的、可独立判断真伪的事实陈述。
- 每条论断必须自包含（带上主语，不要用「它」「该公司」这类指代）
- 保留所有数字、单位、年份
- 不要输出评价性/过渡性语句（如「综上所述」「值得注意的是」）
- 如果文本本身只有一个事实，就输出一条

只输出 JSON：{"claims": ["论断1", "论断2"]}"""


def split_claims(text: str) -> list[str]:
    """把一段文本拆成原子论断列表。"""
    text = (text or "").strip()
    if not text:
        return []
    try:
        resp = llm.chat_json(
            [{"role": "system", "content": _CLAIM_SYS},
             {"role": "user", "content": text[:3000]}],
            temperature=0.0, max_tokens=1200, tag="claims",
        )
        claims = [c.strip() for c in resp.get("claims", []) if c and c.strip()]
        if claims:
            return claims
    except Exception as e:
        print(f"  [warn] 论断分解失败，回退分句：{e}")
    # 回退：按句号分句
    return [s.strip() for s in re.split(r"[。；;\n]", text) if len(s.strip()) > 5]


# ---------------------------------------------------------------------------
# 指标 1：忠实度
# ---------------------------------------------------------------------------
def faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Faithfulness = 被上下文支撑的论断数 / 答案总论断数

    这是抗幻觉的核心指标：1.0 表示答案里每一句话都能在文档中找到依据。
    """
    claims = split_claims(answer)
    if not claims:
        return 0.0
    ctx = "\n\n".join(contexts)[:6000]
    if not ctx.strip():
        return 0.0

    supported = _judge_support(claims, ctx)
    return supported / len(claims)


_JUDGE_SYS = """你要判断若干条【论断】是否能由【参考资料】支撑。

判定标准（严格）：
- supported：参考资料中直接包含该论断的信息，或可由参考资料严格推出
- unsupported：参考资料中没有相关信息，或论断与参考资料矛盾，
  或论断包含参考资料中没有的具体数字/名称

对每条论断独立判断，不要因为其他论断正确就放宽标准。

只输出 JSON：{"verdicts": [{"id": 0, "supported": true, "reason": "<15字内理由>"}]}"""


def _judge_support(claims: list[str], context: str) -> int:
    """返回被支撑的论断数量。"""
    listing = "\n".join(f"[{i}] {c}" for i, c in enumerate(claims))
    try:
        resp = llm.chat_json(
            [{"role": "system", "content": _JUDGE_SYS},
             {"role": "user", "content": f"【参考资料】\n{context}\n\n【论断】\n{listing}"}],
            temperature=0.0, max_tokens=1500, tag="faithfulness",
        )
        verdicts = resp.get("verdicts", [])
        if verdicts:
            return sum(1 for v in verdicts if v.get("supported"))
    except Exception as e:
        print(f"  [warn] 支撑判断失败，回退关键词匹配：{e}")

    # 回退：粗粒度关键词覆盖判断
    return sum(1 for c in claims if _rough_supported(c, context))


def _rough_supported(claim: str, context: str) -> bool:
    """
    关键词重叠度回退判断（仅在 LLM 不可用时使用）。

    判据：
      1. 论断中的数字必须**全部**出现在上下文里——数字是最强的证据，
         对不上就直接判不支持（这是抗幻觉的底线）
      2. 中文实词的重合率 >= 0.5。这里用 0.5 而非更高阈值，是因为
         短论断往往只有 2 个实词，而措辞差异（「年营业收入」vs
         「年实现营业收入」）会造成合理的漏配
    """
    nums = re.findall(r"\d[\d,.]*%?", claim)
    if nums and not all(n in context for n in nums):
        return False

    toks = [t for t in re.findall(r"[一-鿿]{2,}", claim) if len(t) >= 2]
    if not toks:
        return bool(nums)          # 纯数字论断：数字全中即视为有支撑
    hit = sum(1 for t in toks if t in context)
    return hit / len(toks) >= 0.5


# ---------------------------------------------------------------------------
# 指标 2：答案相关性
# ---------------------------------------------------------------------------
_GENQ_SYS = """根据给定的答案，反向生成它可能回答的问题。
要求：生成 3 个不同表述但语义相同的问题，这些问题应该正好能被该答案回答。

只输出 JSON：{"questions": ["问题1", "问题2", "问题3"]}"""


def answer_relevancy(question: str, answer: str, n: int = 3) -> float:
    """
    Answer Relevancy = 反向生成问题与原问题的平均余弦相似度

    原理：如果答案切题，那么从答案反推出来的问题应该和原问题高度相似；
    如果答案答非所问（比如全是套话或跑题），反推的问题就会偏离。
    """
    from . import embed

    if not answer.strip() or "未能找到" in answer:
        return 0.0
    try:
        resp = llm.chat_json(
            [{"role": "system", "content": _GENQ_SYS},
             {"role": "user", "content": f"答案：{answer[:1500]}"}],
            temperature=0.3, max_tokens=600, tag="relevancy",
        )
        gen_qs = [q.strip() for q in resp.get("questions", []) if q.strip()][:n]
    except Exception:
        gen_qs = []

    if not gen_qs:
        # 回退：直接用词面相似度
        return _lexical_similarity(question, answer)

    vecs = embed.encode([question] + gen_qs, is_query=False)
    sims = vecs[1:] @ vecs[0]
    return float(np.mean(sims))


def _lexical_similarity(a: str, b: str) -> float:
    """字符级 Jaccard 相似度回退方案。"""
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1)


# ---------------------------------------------------------------------------
# 指标 3：上下文精度
# ---------------------------------------------------------------------------
def context_precision(question: str, contexts: list[str],
                      ground_truth: str = "") -> float:
    """
    Context Precision@K = Σ (precision@k × relevance_k) / 相关片段总数

    衡量「有用的片段是否排在前面」——排序质量比单纯召回更贴近用户体验。
    片段是否相关由 LLM 对照标准答案判定。
    """
    if not contexts:
        return 0.0

    relevance = _judge_relevance(question, contexts, ground_truth)
    if not any(relevance):
        return 0.0

    total_rel = sum(relevance)
    score, cum_rel = 0.0, 0
    for k, rel in enumerate(relevance, 1):
        if rel:
            cum_rel += 1
            score += cum_rel / k
    return score / total_rel


_REL_SYS = """判断每个【候选片段】对回答【问题】是否有用。

判定标准（严格）：
- 相关：片段包含回答该问题所需的关键信息（具体数字、名称、事实），
  或提供了必要的背景定义
- 不相关：片段只是主题相近，但不含回答所需的关键信息

若提供了【参考答案】，以「能否支撑参考答案中的信息」为准绳。

只输出 JSON：{"relevance": [true, false, ...]}  —— 顺序与候选片段一致"""


def _judge_relevance(question: str, contexts: list[str],
                     ground_truth: str = "") -> list[bool]:
    listing = "\n\n".join(
        f"[{i}] {c[:400]}" for i, c in enumerate(contexts)
    )
    gt = f"\n\n【参考答案】\n{ground_truth}" if ground_truth else ""
    try:
        resp = llm.chat_json(
            [{"role": "system", "content": _REL_SYS},
             {"role": "user",
              "content": f"【问题】{question}{gt}\n\n【候选片段】\n{listing}"}],
            temperature=0.0, max_tokens=800, tag="ctx_precision",
        )
        rel = resp.get("relevance", [])
        if isinstance(rel, list) and len(rel) == len(contexts):
            return [bool(x) for x in rel]
    except Exception as e:
        print(f"  [warn] 相关性判定失败：{e}")
    return [_rough_relevant(question, c) for c in contexts]


def _rough_relevant(question: str, ctx: str) -> bool:
    toks = [t for t in re.findall(r"[一-鿿]{2,}", question) if len(t) >= 2]
    if not toks:
        return False
    return sum(1 for t in toks if t in ctx) / len(toks) >= 0.5


# ---------------------------------------------------------------------------
# 指标 4：上下文召回
# ---------------------------------------------------------------------------
def context_recall(ground_truth: str, contexts: list[str]) -> float:
    """
    Context Recall = 标准答案的论断中，能被检索上下文支撑的比例

    衡量「该找的信息有没有找全」，是 RAG 检索环节最重要的指标。
    """
    if not ground_truth.strip():
        return float("nan")            # 无参考答案时该指标不适用
    claims = split_claims(ground_truth)
    if not claims:
        return 0.0
    ctx = "\n\n".join(contexts)[:8000]
    return _judge_support(claims, ctx) / len(claims)


# ---------------------------------------------------------------------------
# 指标 5：答案正确性
# ---------------------------------------------------------------------------
def answer_correctness(answer: str, ground_truth: str) -> float:
    """
    Answer Correctness = 0.7 × 事实重叠 + 0.3 × 语义相似
    （与 RAGAS 官方实现的权重一致）
    """
    if not ground_truth.strip():
        return float("nan")
    if not answer.strip():
        return 0.0

    from . import embed

    # 语义相似
    v = embed.encode([answer, ground_truth])
    sem = float(v[0] @ v[1])

    # 事实重叠：用标准答案的论断有多少被答案覆盖
    gt_claims = split_claims(ground_truth)
    if gt_claims:
        fact = _judge_support(gt_claims, answer) / len(gt_claims)
    else:
        fact = sem

    return 0.7 * fact + 0.3 * sem


# ---------------------------------------------------------------------------
# 检索层指标（不依赖 LLM，纯计算）
# ---------------------------------------------------------------------------
def retrieval_metrics(records: list[EvalRecord]) -> dict[str, float]:
    """
    计算检索侧指标：
      Hit Rate @k —— 前 k 条里命中了正确文档的比例
      MRR         —— 正确文档首次出现排名的倒数均值
      Recall@k    —— 正确文档被召回的比例
    """
    hit, mrr, recall, n = 0, 0.0, 0, 0
    for r in records:
        if not r.reference_doc:
            continue
        n += 1
        hit_docs = [d for d in r.retrieved_docs if r.reference_doc in d]
        if hit_docs:
            hit += 1
            recall += len(set(hit_docs)) / max(len(set(r.retrieved_docs)), 1)
            first = next(i for i, d in enumerate(r.retrieved_docs, 1)
                         if r.reference_doc in d)
            mrr += 1.0 / first
    if n == 0:
        return {"hit_rate": 0.0, "mrr": 0.0, "recall_at_k": 0.0, "n": 0}
    return {
        "hit_rate": hit / n, "mrr": mrr / n,
        "recall_at_k": recall / n, "n": n,
    }


def keyword_accuracy(records: list[EvalRecord], answers_expected: dict) -> dict:
    """
    关键词命中式准确率（自动化「准确率」判定）：
    对每个问题预设若干「答案必须包含的关键信息点」，命中全部即判对。
    工单01/02 要求的「准确率达到 90% 以上」用此口径统计。
    """
    correct, details = 0, []
    for r in records:
        kws = answers_expected.get(str(r.qid)) or answers_expected.get(r.qid)
        if not kws:
            continue
        ans = r.answer
        hits = [k for k in kws if k in ans]
        ok = len(hits) == len(kws)
        correct += int(ok)
        details.append({
            "id": r.qid, "正确": ok, "命中": hits,
            "漏答": [k for k in kws if k not in ans],
        })
    total = len(details)
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct, "total": total, "details": details,
    }


# ---------------------------------------------------------------------------
# 批量评估
# ---------------------------------------------------------------------------
def evaluate_records(
    records: list[EvalRecord],
    metrics: list[str] | None = None,
    use_ragas: bool = False,
    verbose: bool = True,
) -> dict:
    """
    批量评估并汇总。

    Args:
        metrics: 需要计算的指标子集，默认全部
        use_ragas: True 且已安装 ragas 时走官方实现
    """
    metrics = metrics or ["faithfulness", "answer_relevancy",
                          "context_precision", "context_recall",
                          "answer_correctness"]

    if use_ragas:
        ragas_result = _try_official_ragas(records, metrics)
        if ragas_result is not None:
            return ragas_result

    for i, r in enumerate(records, 1):
        if verbose:
            print(f"  [{i}/{len(records)}] 评估 id={r.qid} …")
        if "faithfulness" in metrics:
            r.metrics["faithfulness"] = faithfulness(r.answer, r.contexts)
        if "answer_relevancy" in metrics:
            r.metrics["answer_relevancy"] = answer_relevancy(r.question, r.answer)
        if "context_precision" in metrics:
            r.metrics["context_precision"] = context_precision(
                r.question, r.contexts, r.ground_truth)
        if "context_recall" in metrics:
            r.metrics["context_recall"] = context_recall(r.ground_truth, r.contexts)
        if "answer_correctness" in metrics:
            r.metrics["answer_correctness"] = answer_correctness(
                r.answer, r.ground_truth)

    return aggregate(records, metrics)


def _try_official_ragas(records: list[EvalRecord], metrics: list[str]) -> dict | None:
    """调用官方 ragas 包（若已安装）。"""
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (answer_correctness, answer_relevancy,
                                   context_precision, context_recall,
                                   faithfulness)
    except ImportError:
        print("  [info] 未安装 ragas，使用内置评估实现")
        return None

    metric_map = {
        "faithfulness": faithfulness, "answer_relevancy": answer_relevancy,
        "context_precision": context_precision, "context_recall": context_recall,
        "answer_correctness": answer_correctness,
    }
    ds = Dataset.from_dict({
        "question": [r.question for r in records],
        "answer": [r.answer for r in records],
        "contexts": [r.contexts for r in records],
        "ground_truth": [r.ground_truth for r in records],
    })
    try:
        result = ragas_evaluate(ds, metrics=[metric_map[m] for m in metrics])
        df = result.to_pandas()
        for i, r in enumerate(records):
            for m in metrics:
                if m in df.columns:
                    r.metrics[m] = float(df.iloc[i][m])
        out = aggregate(records, metrics)
        out["evaluator"] = "ragas(official)"
        return out
    except Exception as e:
        print(f"  [warn] 官方 ragas 评估失败，回退内置实现：{e}")
        return None


def aggregate(records: list[EvalRecord], metrics: list[str]) -> dict:
    """把逐条指标汇总成均值，并附上检索层指标。"""
    summary: dict = {"n": len(records), "evaluator": "builtin"}
    for m in metrics:
        vals = [r.metrics.get(m) for r in records]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        summary[m] = round(float(np.mean(vals)), 4) if vals else None

    lat = [r.latency for r in records if r.latency > 0]
    if lat:
        summary["latency_avg"] = round(float(np.mean(lat)), 3)
        summary["latency_p95"] = round(float(np.percentile(lat, 95)), 3)
        summary["latency_max"] = round(float(np.max(lat)), 3)

    summary.update(retrieval_metrics(records))
    return summary


def save_report(records: list[EvalRecord], summary: dict, path: Path) -> Path:
    """把评估结果落盘为 JSON（逐条 + 汇总）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "summary": summary,
        "records": [r.to_dict() for r in records],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _display_width(s: str) -> int:
    """计算字符串的终端显示宽度（中日韩全角字符占 2 列）。"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s: str, width: int) -> str:
    """按显示宽度右侧补空格，保证中文表格对齐。"""
    return s + " " * max(width - _display_width(s), 0)


def format_summary(summary: dict) -> str:
    """把汇总指标格式化成人读的表格文本（按显示宽度对齐）。"""
    names = {
        "faithfulness": "忠实度 Faithfulness",
        "answer_relevancy": "答案相关性 Answer Relevancy",
        "context_precision": "上下文精度 Context Precision",
        "context_recall": "上下文召回 Context Recall",
        "answer_correctness": "答案正确性 Answer Correctness",
        "hit_rate": "命中率 Hit Rate",
        "mrr": "MRR",
        "recall_at_k": "召回率 Recall@k",
        "latency_avg": "平均耗时(s)",
        "latency_p95": "P95 耗时(s)",
        "latency_max": "最大耗时(s)",
    }
    width = 34
    lines = [_pad("指标", width) + "数值",
             "-" * (width + 12)]
    for k, v in summary.items():
        if k in ("details", "n", "evaluator"):
            continue
        label = names.get(k, k)
        if isinstance(v, float):
            lines.append(_pad(label, width) + f"{v:.4f}")
        elif v is not None:
            lines.append(_pad(label, width) + str(v))
    lines.append(_pad("样本数 n", width) + str(summary.get("n", 0)))
    lines.append(_pad("评估器", width) + str(summary.get("evaluator", "builtin")))
    return "\n".join(lines)
