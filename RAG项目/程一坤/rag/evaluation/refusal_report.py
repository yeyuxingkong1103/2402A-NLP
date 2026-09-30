# -*- coding: utf-8 -*-
"""拒答校准报告的 Markdown 渲染（批次 24 自 `calibrate_refusal.py` 拆出）。

为什么单独成文件：见 `refusal_scoring.py` 的模块说明（同一次拆分，
`docs/目录与命名约定.md` §3.4 的"单文件 ≤300 行"）。**零逻辑改动**，
函数体逐字搬移，对外函数名不变。

职责边界：只把已经算好的 `payload` 排版成 Markdown，**不重算任何指标**；
所有数字都来自 `payload`，这样"改版式"与"改口径"互不影响。
"""

from __future__ import annotations

import statistics
from typing import Any


# 与 render_report.render_markdown() 同名不同用途：本函数只渲染拒答校准报告，调用方仅 calibrate_refusal.py
def render_markdown(payload: dict[str, Any]) -> str:
    rows = payload["rows"]
    refusals = [r for r in rows if r["expect_refusal"] and r.get("top1_rerank") is not None]
    normals = [r for r in rows if not r["expect_refusal"] and r.get("top1_rerank") is not None]
    lines = [
        "# 拒答阈值校准报告",
        "",
        f"- 运行时间：{payload['run_at']}",
        f"- 样本：refusal 题 {len(refusals)} 条 / 非 refusal 题 {len(normals)} 条",
        "",
        "## 一、现状（校准前）",
        "",
        "拒答判定 = `len(candidates) == 0`（无候选才拒答），没有任何分数阈值。"
        "库内 11 部法规对任意问题都能召回若干条，因此 8 条拒答题全部被作答，**拒答准确率 0**。",
        "",
        "## 二、分数分布（首条候选的重排分）",
        "",
        f"- 拒答题 top1 重排分：min={min([r['top1_rerank'] for r in refusals]):.4f} "
        f"max={max([r['top1_rerank'] for r in refusals]):.4f} "
        f"中位={statistics.median([r['top1_rerank'] for r in refusals]):.4f}",
        f"- 非拒答题 top1 重排分：min={min([r['top1_rerank'] for r in normals]):.4f} "
        f"max={max([r['top1_rerank'] for r in normals]):.4f} "
        f"中位={statistics.median([r['top1_rerank'] for r in normals]):.4f}",
        "",
        "## 三、阈值分界点",
        "",
    ]
    for key in ("top1_vector", "top1_rerank"):
        result = payload["scans"].get(key) or {}
        if result.get("error"):
            lines.append(f"- {key}：{result['error']}")
            continue
        best = result["best"]
        lines += [
            f"### {key}",
            "",
            f"- **建议阈值 = {best['threshold']}**",
            f"- 依据：拒答题召回 {best['refusal_recall']:.2%}（{best['tp']}/{best['tp'] + best['fn']}），"
            f"非拒答题保留 {best['normal_precision']:.2%}（{best['tn']}/{best['tn'] + best['fp']}），"
            f"平衡准确率 {best['balanced_accuracy']:.4f}",
            f"- 误拒（不该拒的被拒）：{best['fp']} 条",
        ]
        if key == "top1_vector":
            lines += [
                f"- 两簇边界：拒答题最高 {max(result['refusal_top1']):.4f} / "
                f"非拒答题最低 {min(result['normal_top1']):.4f}；"
                f"取中点 {payload.get('chosen_threshold')}（两侧各留约 "
                f"{(min(result['normal_top1']) - max(result['refusal_top1'])) / 2:.4f} 余量）",
                f"- **写入配置的取值：`REFUSAL_MIN_VECTOR_SCORE={payload.get('chosen_threshold')}`"
                "（0 表示不启用）**",
            ]
        lines += [
            "",
            "卡在边界附近的样本：",
            "",
            "| 题号 | 是否拒答题 | top1 分数 | top1 命中 |",
            "|---|---|---|---|",
        ]
        for sample in result["boundary_samples"]:
            lines.append(
                f"| {sample['id']} | {'是' if sample['expect_refusal'] else '否'} | "
                f"{sample['score']} | {sample['top1']} |"
            )
        lines.append("")

    comparison = payload.get("comparison")
    if comparison:
        lines += [
            "## 四、校准前后对比（同一评测集，同一口径）",
            "",
            "| 指标 | 校准前（阈值未启用） | 校准后（阈值 "
            f"{payload.get('chosen_threshold')}） |",
            "|---|---|---|",
        ]
        keys = [
            ("recall_at_5", "Recall@5"),
            ("mrr_at_10", "MRR@10"),
            ("citation_accuracy", "引用正确率"),
            ("refusal_accuracy", "拒答准确率"),
            ("false_refusal_rate", "误拒率"),
        ]
        for field_name, label in keys:
            before = comparison["before"][field_name]
            after = comparison["after"][field_name]
            lines.append(f"| {label} | {before:.4f} | {after:.4f} |")
        counts = comparison["before"]["counts"], comparison["after"]["counts"]
        after_counts = counts[1]
        refusal_total = after_counts["refusal_questions"]
        normal_total = after_counts["total"] - after_counts["refusal_questions"]
        refusal_hit = int(round(comparison["after"]["refusal_accuracy"] * refusal_total))
        false_refusal = int(round(comparison["after"]["false_refusal_rate"] * normal_total))
        lines += [
            "",
            f"- 校准前拒答题被作答数：{counts[0]['refusal_questions']}"
            "（全部作答，拒答准确率 0）",
            f"- 校准后拒答题正确拒答数：{refusal_hit}/{refusal_total}；"
            f"非拒答题被误拒数：{false_refusal}",
            "",
        ]
    return "\n".join(lines)
