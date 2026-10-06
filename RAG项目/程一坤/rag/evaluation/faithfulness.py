"""Faithfulness 轻量打分器（批次 10 任务 5，不装 ragas）。

思路：用现有 DeepSeek 客户端，对每条非拒答回答做一次判定——
回答里的每个结论是否都能被「注入提示词的法条原文」支持？
输出 0~1 分 + 一句话理由；汇总均值与最差 5 条。

费用说明：每条回答恰好 1 次 LLM 调用（输入约 1~2k token、输出约 50 token），
调用次数与估算费用写在报告里；估算单价可在命令行传入，不硬编码汇率。

用法：
    python evaluation/faithfulness.py                       # 评 reports/latest_eval.json
    python evaluation/faithfulness.py --eval-json <path>    # 评指定报告
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from run_eval import prepare_env, with_retry  # noqa: E402

prepare_env()

SYSTEM_PROMPT = """你是法律问答系统的忠实度（Faithfulness）评审员。
你的任务：判断回答中的每个结论是否都能被「注入的法条原文」支持。
判定标准：
- 回答结论与法条原文一致（含合理的同义转述）→ 支持；
- 回答出现法条原文之外的断言、数字、期限、比例，且原文查不到 → 不支持；
- 回答明确承认"资料中未提及/无法确定"的表述不算不忠实；
- 一般性的程序性提示（建议咨询律师等）不算结论，不计入。
只输出 JSON，格式：{"score": 0.0到1.0之间的小数, "reason": "一句话理由（中文）"}
score = 被支持的结论占全部结论的比例；没有结论（纯拒答/纯提示）给 1.0。"""

USER_TEMPLATE = """【用户问题】
{question}

【回答】
{answer}

【注入的法条原文】
{context}

请按系统指令输出 JSON。"""


def _extract_json(text: str) -> dict | None:
    """从模型输出里抠出 JSON（容忍 markdown 代码块包裹）。"""
    match = re.search(r"\{.*\}", text or "", flags=re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    score = data.get("score")
    if not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 1.0:
        return None
    return {"score": float(score), "reason": str(data.get("reason", ""))[:200]}


def build_context_text(excerpts: list[dict]) -> str:
    lines = []
    for item in excerpts:
        content = (item.get("content") or "").strip()
        lines.append(
            f"[{item.get('index')}] {item.get('law_name')} {item.get('article_number') or ''}\n{content}"
        )
    return "\n\n".join(lines)


def score_item(llm_client, detail: dict) -> dict:
    """对单条评测明细做 Faithfulness 打分；返回打分记录。"""
    record = {
        "id": detail["id"],
        "type": detail["type"],
        "score": None,
        "reason": None,
        "skipped": True,
        "skip_reason": "",
    }
    if detail.get("refused_flag") or detail.get("expect_refusal"):
        record["skip_reason"] = "拒答题不打分"
        return record
    excerpts = detail.get("context_excerpts") or []
    answer = detail.get("answer") or ""
    if not excerpts:
        record["skip_reason"] = "无法条原文（旧报告缺 context_excerpts）"
        return record
    if not answer.strip():
        record["skip_reason"] = "回答为空"
        return record
    prompt = USER_TEMPLATE.format(
        question=detail.get("question") or "",
        answer=answer,
        context=build_context_text(excerpts),
    )
    try:
        raw = with_retry(
            lambda: llm_client.chat(SYSTEM_PROMPT, prompt), attempts=4
        )
    except Exception as error:  # noqa: BLE001
        record["skip_reason"] = f"打分调用失败: {type(error).__name__}"
        return record
    record["skipped"] = False
    record["prompt_chars"] = len(SYSTEM_PROMPT) + len(prompt)
    parsed = _extract_json(raw)
    if parsed is None:
        record["score"] = None
        record["reason"] = f"输出不可解析: {raw[:80]}"
        record["parse_failed"] = True
    else:
        record.update(parsed)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", type=Path, default=PROJECT_ROOT / "reports" / "latest_eval.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--input-price-per-m-tokens", type=float, default=1.0,
                        help="输入单价估算：元/百万token（DeepSeek 官网为准）")
    parser.add_argument("--output-price-per-m-tokens", type=float, default=2.0,
                        help="输出单价估算：元/百万token")
    args = parser.parse_args()

    payload = json.loads(args.eval_json.read_text(encoding="utf-8"))
    details = payload.get("details") or []
    if args.limit:
        details = details[: args.limit]

    from app.models.llm import build_chat_client_from_settings

    llm_client = build_chat_client_from_settings()

    records = []
    start = time.time()
    for index, detail in enumerate(details, start=1):
        print(f"[{index}/{len(details)}] {detail.get('id')} …", flush=True)
        records.append(score_item(llm_client, detail))
    elapsed = time.time() - start

    scored = [r for r in records if not r["skipped"]]
    valid = [r for r in scored if r["score"] is not None]
    mean = sum(r["score"] for r in valid) / len(valid) if valid else 0.0

    # token/费用估算：输入按平均 prompt 字符数折算（中文约 1 字 ≈ 0.6 token），
    # 输出固定按 60 token/条（JSON + 一句话理由）；单价由命令行传入，只是量级估算
    total_prompt_chars = sum(r.get("prompt_chars", 0) for r in scored)
    input_tokens = int(total_prompt_chars * 0.6)
    output_tokens = len(scored) * 60
    cost = (
        input_tokens / 1e6 * args.input_price_per_m_tokens
        + output_tokens / 1e6 * args.output_price_per_m_tokens
    )

    worst = sorted([r for r in valid if r["score"] is not None], key=lambda r: r["score"])[:5]

    summary = {
        "eval_json": str(args.eval_json),
        "total_details": len(details),
        "scored": len(scored),
        "valid_scores": len(valid),
        "parse_failed": sum(1 for r in scored if r.get("parse_failed")),
        "skipped": len(records) - len(scored),
        "faithfulness_mean": round(mean, 4),
        "llm_calls": len(scored),
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_cost_cny": round(cost, 4),
        "price_assumption": f"输入 {args.input_price_per_m_tokens} 元/百万token，输出 {args.output_price_per_m_tokens} 元/百万token（估算用，以供应商账单为准）",
        "elapsed_seconds": round(elapsed, 1),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {"summary": summary, "worst_samples": worst, "records": records}
    (args.output_dir / f"faithfulness_{stamp}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "latest_faithfulness.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Faithfulness 评测报告（轻量实现，未安装 ragas）",
        "",
        f"- 评测来源：`{summary['eval_json']}`",
        f"- 打分口径：回答的每个结论是否被注入的法条原文支持（DeepSeek 单次判定，0~1 分）",
        f"- **Faithfulness 均值：{summary['faithfulness_mean']:.4f}**（{summary['valid_scores']} 条有效 / {summary['scored']} 条打分 / {summary['skipped']} 条跳过）",
        f"- 解析失败：{summary['parse_failed']} 条",
        "",
        "## 调用次数与费用估算",
        "",
        f"- LLM 调用次数：{summary['llm_calls']} 次（每条回答 1 次）",
        f"- 估算输入 token：{input_tokens:,}；估算输出 token：{output_tokens:,}",
        f"- 估算费用：约 {summary['estimated_cost_cny']:.4f} 元（{summary['price_assumption']}）",
        f"- 耗时：{summary['elapsed_seconds']} 秒",
        "",
        "## 最差 5 条",
        "",
    ]
    for record in worst:
        lines.append(f"### {record['id']}（{record['type']}）— {record['score']:.2f}")
        lines.append(f"- 理由：{record['reason']}")
        lines.append("")
    (args.output_dir / "latest_faithfulness.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n==== Faithfulness 完成 ====")
    print(f"均值={summary['faithfulness_mean']:.4f}  打分={summary['valid_scores']} 条  "
          f"跳过={summary['skipped']}  解析失败={summary['parse_failed']}")
    print(f"LLM 调用 {summary['llm_calls']} 次，估算费用 {summary['estimated_cost_cny']:.4f} 元")
    print(f"报告：{args.output_dir / 'latest_faithfulness.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
