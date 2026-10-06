"""
evaluate.py — RAGAS 评测脚本

评测四个指标（对应需求文档 4.12）：
    faithfulness        忠实度：回答是否基于检索到的内容
    answer_relevancy    相关性：回答是否切题
    context_precision   上下文精确率：检索结果是否精准
    context_recall      上下文召回率：是否覆盖了标准答案的要点

    python evaluate.py                  # 全量评测（15 题）
    python evaluate.py --limit 3        # 只跑前 3 题，省时间省费用
    python evaluate.py --domain legal   # 只评测某个领域

优先使用 ragas，未安装时自动降级为 LLM 打分（仍使用 config 中配置的模型）。
结果输出到 logs/eval_report.json 与 logs/eval_report.png，评测集见 eval_set.py。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from eval_set import EVAL_SET, METRICS


def run_pipeline(samples: list[dict], user_id: str = "eval_user") -> list[dict]:
    """对评测集跑一遍 RAG 流程，收集回答与检索到的上下文。"""
    import chat
    from llm_client import get_client

    client = get_client()
    results: list[dict] = []

    for index, item in enumerate(samples, start=1):
        question = item["question"]
        print(f"  [{index}/{len(samples)}] {question[:32]}", flush=True)
        try:
            ctx = chat.prepare(user_id, question)
            # 评测不写记忆，避免污染、也避免长期记忆影响后续题目
            answer = client.chat(ctx["system_prompt"], question, ctx["history"]).strip()
            if not answer:
                raise RuntimeError("模型返回空回答（通常是思维链占满了 max_tokens）")
        except Exception as exc:
            print(f"      失败：{exc}")
            results.append({**item, "answer": "", "contexts": [], "error": str(exc)})
            continue

        results.append(
            {
                **item,
                "answer": answer,
                "contexts": [doc.get("text", "") for doc in ctx["docs"]],
                "detected_domain": ctx["domain"],
                "rewritten_query": ctx["rewritten_query"],
                "sources": [s.get("source", "") for s in ctx["sources"]],
            }
        )
    return results


def _as_ragas_dataset(samples: list[dict]):
    """把样本转成 ragas 需要的 Dataset。"""
    from datasets import Dataset

    return Dataset.from_dict({
        "question": [s["question"] for s in samples],
        "answer": [s["answer"] for s in samples],
        "contexts": [s["contexts"] for s in samples],
        "ground_truth": [s["ground_truth"] for s in samples],
    })


def evaluate_ragas(samples: list[dict]) -> dict | None:
    """用 ragas 计算四项指标，环境不具备时返回 None。"""
    try:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError:
        return None

    try:
        dataset = _as_ragas_dataset(samples)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        scores = result.to_pandas()
        summary = {
            metric: round(float(scores[metric].dropna().mean()), 4)
            for metric in METRICS
            if metric in scores.columns
        }
        return {"engine": "ragas", "summary": summary, "per_sample": scores.to_dict("records")}
    except Exception as exc:
        print(f"  ragas 评测失败（{exc}），改用 LLM 打分")
        return None


JUDGE_PROMPT = """你是一个严格的评审员，请为下面这组问答打分。

评分维度（每项 0 到 1 之间的小数）：
1. faithfulness：回答是否完全基于【参考资料】中的信息，有无编造。资料没提到的内容即使正确也要扣分。
2. answer_relevancy：回答是否切中【问题】，有无答非所问。
3. context_precision：【参考资料】是否都与问题相关，有无大量无关内容。
4. context_recall：【参考答案】的要点是否都能在【参考资料】中找到。

只输出一行 JSON，格式：
{{"faithfulness": 0.9, "answer_relevancy": 0.9, "context_precision": 0.8, "context_recall": 0.7}}

【问题】
{question}

【参考资料】
{contexts}

【模型回答】
{answer}

【参考答案】
{ground_truth}

请输出 JSON："""


def evaluate_llm_judge(samples: list[dict]) -> dict:
    """降级方案：让配置的模型自己按四个维度打分。"""
    from llm_client import get_client

    client = get_client()
    per_sample: list[dict] = []

    for item in samples:
        contexts = "\n---\n".join(item.get("contexts") or [])[:6000] or "（无检索结果）"
        prompt = JUDGE_PROMPT.format(
            question=item["question"],
            contexts=contexts,
            answer=item.get("answer", "")[:2000],
            ground_truth=item["ground_truth"],
        )

        row = {"question": item["question"], "domain": item["domain"]}
        try:
            # 判分必须关掉思维链：推理模型的思考会占满 max_tokens 导致正文为空，
            # 解析不出指标时若当成 0 分，整份报告会显示成"全部 0.0000"
            raw = client.chat(
                "你是一个评审员，只输出 JSON。", prompt, temperature=0.0, max_tokens=200,
                reasoning_effort=config.LLM_REASONING_EFFORT,
            )
            start, end = raw.find("{"), raw.rfind("}")
            parsed = json.loads(raw[start : end + 1]) if start >= 0 and end > start else {}
            if [m for m in METRICS if m not in parsed]:
                raise ValueError(f"评审模型未返回全部指标；原始输出={raw[:80]!r}")
            for metric in METRICS:
                row[metric] = float(parsed[metric])
        except Exception as exc:
            row["error"] = str(exc)
            for metric in METRICS:
                row[metric] = 0.0
        per_sample.append(row)

    # 打分失败的题目不参与平均，避免把"没打成"当成"答得差"
    scored = [row for row in per_sample if not row.get("error")]
    summary = {
        metric: round(statistics.mean([row[metric] for row in scored]), 4) if scored else 0.0
        for metric in METRICS
    }
    return {"engine": "llm_judge", "summary": summary, "scored_count": len(scored),
            "failed_count": len(per_sample) - len(scored), "per_sample": per_sample}


def plot_report(summary: dict, output: Path) -> bool:
    """把指标画成柱状图。matplotlib 缺失时跳过。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    names, values = list(summary), list(summary.values())

    figure, axes = plt.subplots(figsize=(8, 4.5))
    bars = axes.bar(names, values, color=["#4a90d9", "#5aa469", "#e0a458", "#c0504d"][: len(names)])
    axes.set_ylim(0, 1.05)
    axes.set_ylabel("score")
    axes.set_title("RAG Evaluation Metrics")
    axes.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, value in zip(bars, values):
        axes.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)

    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)
    return True


def build_report(samples: list[dict], evaluation: dict) -> dict:
    """汇总成最终报告。"""
    domain_stats: dict[str, dict] = {}
    for row in evaluation.get("per_sample", []):
        if row.get("error"):
            continue  # 打分失败的题目不参与分领域平均
        domain = row.get("domain", "unknown")
        bucket = domain_stats.setdefault(domain, {metric: [] for metric in METRICS})
        for metric in METRICS:
            if isinstance(row.get(metric), (int, float)):
                bucket[metric].append(float(row[metric]))

    by_domain = {
        domain: {metric: round(statistics.mean(values), 4) for metric, values in metrics.items() if values}
        for domain, metrics in domain_stats.items()
    }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "provider": config.LLM_PROVIDER,
        "model": config.get_llm_config()["model"],
        "local_mode": config.LOCAL_MODE,
        "embedding_backend": config.EMBEDDING_BACKEND,
        "sample_count": len(samples),
        "engine": evaluation.get("engine"),
        "summary": evaluation.get("summary", {}),
        "by_domain": by_domain,
        "failed_questions": [s["question"] for s in samples if s.get("error")],
        "judge_scored": evaluation.get("scored_count", 0),
        "judge_failed": evaluation.get("failed_count", 0),
        "per_sample": evaluation.get("per_sample", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS 评测")
    parser.add_argument("--limit", type=int, default=0, help="只评测前 N 题（0 表示全部）")
    parser.add_argument("--domain", default="", help="只评测指定领域：legal / medical / english")
    args = parser.parse_args()

    config.ensure_dirs()
    samples = EVAL_SET
    if args.domain:
        samples = [s for s in samples if s["domain"] == args.domain]
    if args.limit > 0:
        samples = samples[: args.limit]

    if not samples:
        print("没有匹配的评测题目，请检查 --domain 参数")
        return 1

    print(f"评测题目：{len(samples)} 道")
    print(f"模型：{config.LLM_PROVIDER} / {config.get_llm_config()['model']}")
    print("\n[1/3] 运行 RAG 流程收集回答与上下文")
    results = run_pipeline(samples)

    usable = [s for s in results if s.get("answer")]
    if not usable:
        print("所有题目都执行失败，请检查模型配置与知识库是否已导入")
        return 1

    print("\n[2/3] 计算指标")
    evaluation = evaluate_ragas(usable)
    if evaluation is None:
        print("  使用 LLM 打分（未安装 ragas 或 ragas 调用失败）")
        evaluation = evaluate_llm_judge(usable)

    report = build_report(results, evaluation)

    report_path = config.LOG_DIR / "eval_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[3/3] 生成报告")
    print(f"  JSON：{report_path}")
    if plot_report(report["summary"], config.LOG_DIR / "eval_report.png"):
        print(f"  图表：{config.LOG_DIR / 'eval_report.png'}")

    print("\n评测结果：")
    for metric, value in report["summary"].items():
        print(f"  {metric:20s} {value:.4f}")
    for domain, metrics in report["by_domain"].items():
        joined = "  ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        print(f"  [{domain}] {joined}")

    if report["failed_questions"] or report["judge_failed"]:
        print(f"  注意：{len(report['failed_questions'])} 道题未生成回答，"
              f"{report['judge_failed']} 道题打分失败，均未计入平均")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
