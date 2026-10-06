# -*- coding: utf-8 -*-
"""T6 验收自检：校验三件基线产出是否满足任务书要求的字段与数值。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：优化 / 基线采集（自检）

检查项
    ① baseline_results.json：来源/采集时间/判分口径齐全；逐题含 question_id/question/golden/
       baseline_answer/is_correct/citation_pages/first_token_ms/total_ms；汇总含
       accuracy/unknown_accuracy/citation_accuracy/first_token_avg_ms/first_token_p95_ms/
       first_token_max_ms/total_avg_ms；且 accuracy=0.50、逐题判定与 eval_records 一致。
    ② baseline_retrieval.json：含任务口径（证据前 80 字落块内）的 top-5/top-20 逐题命中与命中率。
    ③ 基线说明.md：含来源、采集方法、复现命令、已知数据缺陷三要素。

复现（工作目录 = E:\\gao6gongdan\\工单2）：
    pwsh -NoProfile -File run_py.ps1 -B 优化/脚本/verify_baseline_outputs.py
退出码：0 全部通过；1 存在不合规项。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parents[2] / "优化" / "基线"
RESULTS = BASE / "baseline_results.json"
RETRIEVAL = BASE / "baseline_retrieval.json"
DOC = BASE / "基线说明.md"

REQUIRED_ROW = [
    "question_id", "question", "golden", "baseline_answer", "is_correct",
    "citation_pages", "first_token_ms", "total_ms",
]
REQUIRED_SUMMARY = [
    "accuracy", "unknown_accuracy", "citation_accuracy", "first_token_avg_ms",
    "first_token_p95_ms", "first_token_max_ms", "total_avg_ms",
]

problems: list[str] = []


def check(cond: bool, ok_msg: str, bad_msg: str) -> None:
    if cond:
        print(f"  ✅ {ok_msg}")
    else:
        print(f"  ❌ {bad_msg}")
        problems.append(bad_msg)


print("=== ① baseline_results.json ===")
if not RESULTS.exists():
    problems.append("baseline_results.json 不存在")
    print("  ❌ 文件不存在")
else:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    meta, summary = data.get("meta", {}), data.get("summary", {})
    src = meta.get("来源文件") or []
    check(bool(src) and all("path" in s and "sha1" in s for s in src),
          f"来源文件齐全（{len(src)} 个，均含 path+sha1）", "来源文件缺失或缺少 sha1")
    check(bool(meta.get("采集时间")), f"采集时间 = {meta.get('采集时间')}", "缺少采集时间")
    check(bool(meta.get("判分口径")), "判分口径说明存在", "缺少判分口径说明")
    rows = data.get("per_question_rag", [])
    check(len(rows) == 10, f"逐题记录 {len(rows)} 条", f"逐题记录应为 10 条，实为 {len(rows)}")
    missing = sorted({k for r in rows for k in REQUIRED_ROW if k not in r})
    check(not missing, "逐题必需字段齐全", f"逐题缺少字段 {missing}")
    miss_s = [k for k in REQUIRED_SUMMARY if k not in summary]
    check(not miss_s, "汇总指标齐全（7 项）", f"汇总缺少 {miss_s}")
    check(abs(float(summary.get("accuracy", -1)) - 0.5) < 1e-9,
          f"accuracy = {summary.get('accuracy')}（与工单1 记录一致）",
          f"accuracy 异常：{summary.get('accuracy')}（应为 0.5）")
    check(summary.get("accuracy_count") == "5/10",
          f"对/错 = {summary.get('accuracy_count')}", f"对/错计数异常：{summary.get('accuracy_count')}")
    v = data.get("verification", {})
    check(bool(v.get("per_question_judgements_all_reproduced")),
          "10/10 逐题判定已用原始判分函数复现一致", "逐题判定复现不一致")
    check("未运行" in str(v.get("ragas", "")), "RAGAS 已显式标注「未运行」", "未标注 RAGAS 状态")
    print(f"    核对值: accuracy={summary.get('accuracy')} 首字avg={summary.get('first_token_avg_ms')}ms "
          f"p95={summary.get('first_token_p95_ms')}ms max={summary.get('first_token_max_ms')}ms "
          f"端到端avg={summary.get('total_avg_ms')}ms 引用正确率={summary.get('citation_accuracy')}")

print("\n=== ② baseline_retrieval.json（任务口径）===")
if not RETRIEVAL.exists():
    problems.append("baseline_retrieval.json 不存在")
    print("  ❌ 文件不存在")
else:
    ret = json.loads(RETRIEVAL.read_text(encoding="utf-8"))
    spec = ret.get("检索命中率_任务口径", {})
    check(bool(spec), "含「检索命中率_任务口径」块", "缺少任务口径块")
    for key, label in (("归一化A_仅去空白", "A(仅去空白)"), ("归一化B_去空白与标点", "B(判分同款)")):
        block = spec.get(key) or {}
        t5, t20 = block.get("top5", {}), block.get("top20", {})
        rows = block.get("逐题", [])
        ok = bool(t5) and bool(t20) and len(rows) == 10 and all("top5_hit" in r and "top20_hit" in r for r in rows)
        check(ok, f"{label}: top-5 {t5.get('hit_count')}/{t5.get('total')}、"
                  f"top-20 {t20.get('hit_count')}/{t20.get('total')}、逐题 {len(rows)} 条",
              f"{label}: 任务口径字段不完整")
    pq = ret.get("per_question", [])
    check(len(pq) == 10, f"逐题明细 {len(pq)} 条", f"逐题明细应为 10 条，实为 {len(pq)}")

print("\n=== ③ 基线说明.md ===")
if not DOC.exists():
    problems.append("基线说明.md 不存在")
    print("  ❌ 文件不存在")
else:
    text = DOC.read_text(encoding="utf-8")
    for key, label in (
        ("## 8. 基线来源与可追溯性", "基线来源章节"),
        ("## 8.2 采集方法", "采集方法章节"),
        ("## 7. 复现命令与产物清单", "复现命令章节"),
        ("## 9. 已知数据缺陷", "已知数据缺陷章节"),
        ("run_py.ps1", "复现命令使用 run_py.ps1"),
        ("evidence_pages", "缺陷①evidence_pages 错标"),
        ("样本量", "缺陷②样本量小"),
        ("释义页", "缺陷③释义页噪声"),
    ):
        check(key in text, f"含 {label}", f"缺少 {label}（未找到 `{key}`）")
    check(len(text) > 12000, f"文档体量 {len(text)} 字符", "文档过短，疑似不完整")

print("\n=== 结论 ===")
if problems:
    for p in problems:
        print(f"  ❌ {p}")
    print(f"  总体: ❌ 不通过（{len(problems)} 项）")
    raise SystemExit(1)
print("  ✅ T6 三件产出字段与数值全部符合任务要求")
print("  总体: ✅ 通过")
raise SystemExit(0)
