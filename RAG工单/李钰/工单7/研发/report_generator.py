# -*- coding: utf-8 -*-
"""
报告生成器 - Markdown + JSON
工单编号: 人工智能 NLP-RAG-功能测试及评估
"""
import json
import os
from typing import List, Dict


def generate_markdown_report(results: List[Dict], 
                             retrieval_metrics: Dict,
                             qa_metrics: Dict,
                             perf_metrics: Dict,
                             problem_analysis: List[Dict]) -> str:
    """生成完整 Markdown 评估报告"""
    lines = []
    lines.append("# RAG 系统功能测试及评估报告\n")
    lines.append("> 工单编号: 人工智能 NLP-RAG-功能测试及评估")
    lines.append("> 测试日期: 2025-01-21")
    lines.append("> 测试套件: 10 个 CCF 风格问题\n")

    # ============ 一、检索评估 ============
    lines.append("## 一、检索评估\n")
    lines.append("| 指标 | 分数 | 目标 | 达标 |")
    lines.append("|------|------|------|------|")
    target_map = {"precision@k": 0.7, "recall@k": 0.95, "hit_rate@k": 0.95,
                  "mrr": 0.8, "ndcg@k": 0.85}
    for key, target in target_map.items():
        val = retrieval_metrics.get(key, 0)
        ok = "✅" if val >= target else "❌"
        lines.append(f"| {key} | {val} | ≥ {target} | {ok} |")
    lines.append("")

    # ============ 二、问答评估 ============
    lines.append("## 二、问答评估\n")
    lines.append("| 指标 | 分数 | 目标 | 达标 |")
    lines.append("|------|------|------|------|")
    qa_targets = {"keyword_coverage": 0.9, "bleu_1": 0.3,
                  "rouge_l": 0.4, "hallucination_ratio": 0.05}
    for key, target in qa_targets.items():
        val = qa_metrics.get(key, 0)
        ok = "✅" if key == "hallucination_ratio" and val <= target or \
                     key != "hallucination_ratio" and val >= target else "❌"
        lines.append(f"| {key} | {val} | {'≤' if key == 'hallucination_ratio' else '≥'} {target} | {ok} |")
    lines.append("")

    # ============ 三、性能评估 ============
    lines.append("## 三、性能评估\n")
    lines.append(f"| 指标 | 数值 |")
    lines.append(f"|------|------|")
    for key in ["avg_time", "p50_time", "p90_time", "p95_time",
                "p99_time", "max_time", "within_3s_ratio"]:
        lines.append(f"| {key} | {perf_metrics.get(key, 'N/A')} |")
    lines.append("")

    # ============ 四、逐题结果 ============
    lines.append("## 四、逐题测试结果\n")
    lines.append("| # | 问题(截断) | 类型 | 检索P@5 | MRR | 关键词覆盖 | RAG答案(前50) |")
    lines.append("|---|-----------|------|---------|-----|-----------|--------------|")
    for r in results:
        q = r.get("question", "")[:30] + "..."
        rtype = r.get("type", "")
        p = r.get("retrieval", {}).get("precision@k", "?")
        mrr = r.get("retrieval", {}).get("mrr", "?")
        cov = r.get("qa", {}).get("keyword_coverage", "?")
        ans = r.get("rag_answer", "")[:50].replace("\n", " ")
        lines.append(f"| Q{r.get('id', '?')} | {q} | {rtype} | {p} | {mrr} | {cov} | {ans} |")
    lines.append("")

    # ============ 五、问题分析 ============
    lines.append("## 五、问题分析\n")
    if problem_analysis:
        for pa in problem_analysis:
            lines.append(f"### Q{pa['id']}: {pa['question'][:40]}...\n")
            lines.append(f"**问题类型**: {pa.get('issue_type', '未知')}\n")
            lines.append(f"**原因分析**: {pa.get('reason', 'N/A')}\n")
            lines.append(f"**改进建议**: {pa.get('suggestion', 'N/A')}\n")
            lines.append("")
    else:
        lines.append("所有问题检索效果良好, 无明显问题。\n")

    # ============ 六、总结 ============
    lines.append("## 六、总结\n")
    total = len(results)
    pass_count = sum(1 for r in results
                     if r.get("qa", {}).get("keyword_coverage", 0) >= 0.7)
    lines.append(f"- 测试问题总数: {total}")
    lines.append(f"- 关键词覆盖率 ≥ 70% 的问题: {pass_count}/{total}")
    lines.append(f"- 整体达标率: {pass_count/total*100:.0f}%")
    lines.append(f"- 平均响应时间: {perf_metrics.get('avg_time', 'N/A')}s")
    lines.append(f"- 3秒内响应比例: {perf_metrics.get('within_3s_ratio', 'N/A')}")
    lines.append("")

    return "\n".join(lines)


def save_json_results(results: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"JSON 结果: {path}")


if __name__ == "__main__":
    print("报告生成器已就绪")
