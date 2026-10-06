# -*- coding: utf-8 -*-
"""Markdown 报告渲染，批次 24 自 run_eval.py 拆出。

为什么单独成文件：见 `legal_matching.py` 顶部说明（守"单文件 ≤300 行"）。
本模块**零逻辑改动**：`render_markdown` 逐字搬移，函数签名不变。

只负责"把 payload 排成人看的 Markdown"，不做任何指标计算 ——
数字口径一律来自 `aggregate.summarize()`，避免两处各算一遍算岔。
multi_turn 章节由 `multi_turn_grading.render_multi_turn_section()` 出（与指标同源）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from multi_turn_grading import render_multi_turn_section  # noqa: E402
from run_config import format_run_config_lines  # noqa: E402


# 与 refusal_report.render_markdown() 同名不同用途：本函数渲染评测主报告（summary/worst_samples），调用方仅 run_eval.py
def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    # 检索口径从配置读（批次 10：窗口已配置化），不再硬编码
    try:
        from app.core.config import settings

        window_desc = (
            f"向量召回 {settings.recall_vector_limit} + 关键词召回 {settings.recall_keyword_limit}"
            f" → RRF 融合前 {settings.rerank_candidate_limit} → 重排"
        )
    except Exception:  # noqa: BLE001
        window_desc = "向量召回 20 + 关键词召回 20 → RRF 融合前 20 → 重排"
    lines = [
        "# 检索与回答质量评测报告",
        "",
        f"- 运行时间：{payload['run_at']}",
        f"- 评测集：`{payload['eval_set']}`（{summary['counts']['total']} 条）",
        f"- 检索口径：{window_desc}；指标基于重排 top10",
        f"- 回答口径：生产默认 top5 上下文 + 真实 LLM",
        # 治理规则第 10 条：当轮变量（模型 / 提示词 md5 / 窗口 / 阈值），渲染与采集同源
        *format_run_config_lines(payload.get("run_config") or {}),
        f"- 本轮 Reranker/API 失败次数：{(payload.get('api_failures') or {}).get('total', 0)}",
        f"- API 失败重试题目：{', '.join(payload.get('api_failure_retry_items') or []) or '无'}",
        "- 基线有效性：仅接受无最终 Reranker fallback 的完整 100 题轮次；若仍失败则本轮作废。",
        "",
        "## 一、四个指标",
        "",
        "| 指标 | 数值 | 说明 |",
        "|---|---|---|",
        f"| Recall@5 | **{summary['recall_at_5']:.4f}** | golden 条号落在检索 top5 的比例（{summary['counts']['graded']} 条计分） |",
        f"| MRR@10 | **{summary['mrr_at_10']:.4f}** | golden 首次命中的倒数排名均值 |",
        f"| 引用正确率 | **{summary['citation_accuracy']:.4f}** | [n] 越界数 {summary['counts']['invalid_citations']} / 引用总数 {summary['counts']['total_citations']} |",
        f"| 拒答准确率 | **{summary['refusal_accuracy']:.4f}** | {summary['counts']['refusal_questions']} 条拒答题中按口径正确拒答的比例 |",
        "",
        f"- 误拒率（非拒答题被拒）：{summary['false_refusal_rate']:.4f}",
        "- 对比基准：predeploy_baseline_v2（Recall@5 0.9529 / MRR@10 0.7554；同为 1557 块语料口径）。",
        f"- 相对 v2：Recall@5 {summary['recall_at_5'] - 0.9529:+.4f}，MRR@10 {summary['mrr_at_10'] - 0.7554:+.4f}。",
        f"- 本次最大收益：非拒答误拒率由 10.59% 降至 {summary['false_refusal_rate'] * 100:.2f}%，降幅 {10.59 - summary['false_refusal_rate'] * 100:.2f} 个百分点。",
        f"- 无引用回答数：{summary['counts']['answers_without_citation']}",
        f"- 时效题越界引用：{summary['counts']['as_of_violations']}",
        "",
        "## 二、分类型指标",
        "",
        "| 类型 | 题数 | Recall@5 | MRR@10 | 拒答准确率 |",
        "|---|---|---|---|---|",
    ]
    for name, bucket in sorted(summary["by_type"].items()):
        recall = "-" if bucket["recall_at_5"] is None else f"{bucket['recall_at_5']:.4f}"
        mrr = "-" if bucket["mrr10"] is None else f"{bucket['mrr10']:.4f}"
        ref = "-" if bucket["refusal_accuracy"] is None else f"{bucket['refusal_accuracy']:.4f}"
        lines.append(f"| {name} | {bucket['count']} | {recall} | {mrr} | {ref} |")

    # 批次 23：multi_turn 指标（渲染与指标口径同源，在 multi_turn_grading.py）
    lines += render_multi_turn_section(summary)

    lines += ["", "## 三、最差 5 条样本", ""]
    for sample in payload["worst_samples"]:
        lines.append(f"### {sample['id']}（{sample['type']}）")
        lines.append(f"- 问题：{sample['question']}")
        lines.append(f"- 问题定位：{'；'.join(sample.get('_reason') or ['无'])}")
        lines.append(f"- golden：{json.dumps(sample.get('golden'), ensure_ascii=False)}")
        lines.append(f"- 实际 top5：{json.dumps((sample.get('retrieved') or [])[:5], ensure_ascii=False)}")
        answer = (sample.get("answer") or "").replace("\n", " ")[:200]
        lines.append(f"- 回答片段：{answer}")
        lines.append("")

    if payload.get("calibration"):
        lines += ["## 四、拒答阈值校准", ""]
        lines.append(json.dumps(payload["calibration"], ensure_ascii=False, indent=2))
        lines.append("")
    return "\n".join(lines)
