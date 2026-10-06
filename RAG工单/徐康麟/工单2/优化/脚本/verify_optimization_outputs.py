# -*- coding: utf-8 -*-
"""T7 验收自检：校验优化前后对比的四件产出是否满足任务书要求，并做「非抄写」一致性检查。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：优化 / 前后对比（自检）

检查项
    ① optimization_compare.csv：表头含 before/after 两列；含 accuracy / retrieval_hit_rate /
       first_token_avg_ms / first_token_p95_ms / citation_accuracy 五项汇总指标与 10 题逐题明细。
    ② optimization_compare.md：含任务书要求的六节 + RAGAS 未运行 + 数据质量声明。
    ③ accuracy_report.json：meta/环境/before/after/逐题齐全；after 数值内部自洽（correct/count==accuracy）。
    ④ ragas_report.md：显式含「RAGAS 未运行」；且**未被覆盖**（仍保留 T2 原报告特征文本）。
    ⑤ 非抄写检查：优化后答案与基线答案**不相同**的题数应为 10/10；且 accuracy 与 before 不一致。

复现（工作目录 = E:\\gao6gongdan\\工单2）：
    pwsh -NoProfile -File run_py.ps1 -B 优化/脚本/verify_optimization_outputs.py
退出码：0 通过；1 不通过。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "优化" / "评估结果"
CSV_PATH = OUT / "optimization_compare.csv"
MD_PATH = OUT / "optimization_compare.md"
JSON_PATH = OUT / "accuracy_report.json"
RAGAS_PATH = OUT / "ragas_report.md"

REQUIRED_SUMMARY_METRICS = [
    "accuracy",
    "retrieval_hit_rate",
    "first_token_avg_ms",
    "first_token_p95_ms",
    "citation_accuracy",
]
REQUIRED_MD_SECTIONS = [
    "## 1. 被测对象与环境",
    "## 2. 优化前后指标对比",
    "## 3. 逐条优化措施与量化收益",
    "## 4. 逐题对照",
    "## 5. 结论与遗留问题",
    "## 6. 数据质量与指标口径诚实声明",
]

problems: list[str] = []


def check(cond: bool, ok_msg: str, bad_msg: str) -> None:
    if cond:
        print(f"  ✅ {ok_msg}")
    else:
        print(f"  ❌ {bad_msg}")
        problems.append(bad_msg)


print("=== ① optimization_compare.csv ===")
if not CSV_PATH.exists():
    problems.append("optimization_compare.csv 不存在")
    print("  ❌ 文件不存在")
else:
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    header = list(rows[0].keys()) if rows else []
    check("before" in header and "after" in header,
          f"表头含 before/after 两列：{header}", f"表头缺少 before/after：{header}")
    summary = {r["metric"]: r for r in rows if r["section"] == "summary"}
    miss = [m for m in REQUIRED_SUMMARY_METRICS if m not in summary]
    check(not miss, f"含五项必需汇总指标（共 {len(summary)} 项）", f"缺少汇总指标 {miss}")
    qrows = [r for r in rows if r["section"] == "question"]
    qids = sorted({r["key"] for r in qrows})
    check(len(qids) == 10, f"逐题明细覆盖 10 题（{len(qrows)} 行）", f"逐题明细仅 {len(qids)} 题")
    for metric in ("is_correct", "retrieval_rank", "first_token_ms", "answer"):
        check(any(r["metric"] == metric for r in qrows), f"逐题含 {metric}", f"逐题缺少 {metric}")

print("\n=== ② optimization_compare.md ===")
if not MD_PATH.exists():
    problems.append("optimization_compare.md 不存在")
    print("  ❌ 文件不存在")
else:
    md = MD_PATH.read_text(encoding="utf-8")
    for section in REQUIRED_MD_SECTIONS:
        check(section in md, f"含章节「{section}」", f"缺少章节「{section}」")
    check("RAGAS" in md and "未运行" in md, "含 RAGAS 未运行声明", "缺少 RAGAS 未运行声明")
    check("evidence_pages" in md, "含 evidence_pages 错标声明", "缺少 evidence_pages 错标声明")
    check("样本量" in md and "10" in md, "含样本量 10 条声明", "缺少样本量声明")
    check(len(md) > 6000, f"文档体量 {len(md)} 字符", "文档过短")

print("\n=== ③ accuracy_report.json ===")
if not JSON_PATH.exists():
    problems.append("accuracy_report.json 不存在")
    print("  ❌ 文件不存在")
else:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    for key in ("meta", "environment", "before", "after", "per_question"):
        check(key in data, f"含顶层键 {key}", f"缺少顶层键 {key}")
    after, before = data["after"], data["before"]
    for key in ("accuracy", "correct", "count") + tuple(REQUIRED_SUMMARY_METRICS):
        check(key in after, f"after 含 {key}", f"after 缺少 {key}")
    self_consistent = abs(after["accuracy"] - after["correct"] / after["count"]) < 1e-6
    check(self_consistent, f"after 自洽：{after['correct']}/{after['count']} = {after['accuracy']}",
          "after 的 accuracy 与 correct/count 不一致")
    check(len(data["per_question"]) == 10, f"逐题 {len(data['per_question'])} 条", "逐题不足 10 条")
    print(f"    优化前 accuracy={before['accuracy']}；优化后 accuracy={after['accuracy']}；"
          f"首字均值={after['first_token_avg_ms']}ms 最大={after['first_token_max_ms']}ms")

print("\n=== ④ ragas_report.md ===")
if not RAGAS_PATH.exists():
    problems.append("ragas_report.md 不存在")
    print("  ❌ 文件不存在")
else:
    ragas = RAGAS_PATH.read_text(encoding="utf-8")
    check("RAGAS" in ragas and "未运行" in ragas, "含「RAGAS 未运行」声明", "缺少「RAGAS 未运行」声明")
    check("T7-RAGAS-DECLARATION" in ragas, "含 T7 追加段标记", "缺少 T7 追加段标记")
    check("逐题结果" in ragas or "RAG vs 纯 LLM" in ragas,
          "T2 原报告内容仍在（未被覆盖）", "T2 原报告内容疑似被覆盖")
    check("optimization_compare.md" in ragas, "已指向对比报告", "未指向对比报告")

print("\n=== ⑤ 非抄写检查（禁止把基线值抄成优化后值）===")
if JSON_PATH.exists():
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    items = data["per_question"]
    differing = sum(1 for it in items if (it["before"]["answer"] or "") != (it["after"]["answer"] or ""))
    check(differing == 10, f"优化后答案与基线答案全部不同（{differing}/10）",
          f"仅 {differing}/10 题答案不同，疑似抄写")
    b_acc, a_acc = data["before"]["accuracy"], data["after"]["accuracy"]
    check(abs(a_acc - b_acc) > 1e-9, f"优化后准确率 {a_acc} 与基线 {b_acc} 不同（未被抄写）",
          f"优化后准确率与基线相同（{a_acc}），疑似抄写")
    modes = sorted({it["after"].get("mode") for it in items})
    print(f"    优化后回答模式：{modes}")
    print(f"    末次运行时间：{data['meta']['生成时间']}")

print("\n=== 结论 ===")
if problems:
    for p in problems:
        print(f"  ❌ {p}")
    print(f"  总体: ❌ 不通过（{len(problems)} 项）")
    raise SystemExit(1)
print("  ✅ T7 四件产出符合任务书要求，且非抄写检查通过")
print("  总体: ✅ 通过")
raise SystemExit(0)
