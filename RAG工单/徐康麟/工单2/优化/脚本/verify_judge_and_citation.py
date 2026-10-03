# -*- coding: utf-8 -*-
"""核验脚本：判分口径未放宽（负例仍判错）+ 引用页码可回查真实 chunk。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：验证（独立核验，只读不改）

命名说明（2026-10-03，tester 按 captain 指令转正）：
原名 ``_t9_verify_judge_and_citation.py``（归档队伍 t9 遗留）。下划线前缀属**临时件命名**，
F7 清理与 `测试/离线/test_deliverable_integrity.py` 的守卫已改为「交付目录不得有 `_` 前缀文件」，
故转正为现名。

## ⚠️ N 守卫（2026-10-03 加，captain 授权）

**存在理由**：本脚本读 ``优化/评估结果/accuracy_report.json`` 判「准确率是否达标」。
若该产物被 **1 题口径的冒烟运行覆盖**（实测发生过：`optimization_compare.py --limit 1`），
`accuracy = 1.0 (1/1)` 会被打印成 ``✅ ≥0.90`` —— 这是**在残缺输入上给假绿**，
与报告生成器的 L467/L471 属同类缺陷，会误导验收 1 的判定。
→ 因此先从 ``测试/测试数据/golden_qa.jsonl`` 读**期望题数 N**，当
``accuracy_report.json`` 的 ``after.count`` 或 ``per_question`` 长度 ≠ N 时：
**不打印 ✅**，改打印「⚠️ 输入不完整」并以**退出码 2**（不可判定）结束。

退出码：``0`` = 全部核验通过；``1`` = 有核验项不通过；``2`` = 输入不完整（N 守卫触发，不可判定）。

用法（工作目录 = 工单2）：
    pwsh -NoProfile -File run_py.ps1 优化/脚本/verify_judge_and_citation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "研发"
sys.path.insert(0, str(SOURCE))

from app.core.chunker import Chunker  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.evaluator import FUZZY_THRESHOLD, get_evaluator  # noqa: E402

settings = get_settings()
evaluator = get_evaluator()

print("=" * 90)
print("核验 A：判分口径未放宽（环境事实 §4.2 负例必须仍判错）")
print(f"FUZZY_THRESHOLD = {FUZZY_THRESHOLD}（工单1 基线亦为 0.62）")
print("=" * 90)

GOLDEN_33 = "报告期内，公司来自军用领域的收入占主营业务收入的比重分别为82.10%、97.31%、94.84%和94.34%。"
GOLDEN_531 = "法定代表人是程家明。"
GOLDEN_543 = "注册资本为5,520万元。"
GOLDEN_207 = "公司计划使用本次发行募集资金15,000.00万元用于补充流动资金。"

cases = [
    ("N-1 多值漏答（环境事实 §4.2 指定负例：4 个比重，回答只给 1 个）", GOLDEN_33, "[页码: 129] 82.10%", False),
    ("N-1b 多值漏答（带引用标签，给 3 个缺 1 个）", GOLDEN_33, "[页码: 129] 82.10%、97.31%、94.84%", False),
    ("N-2 单位错（5,520 万元写成 5,520 元）", GOLDEN_543, "注册资本为5,520元。", False),
    ("N-3 答非所问", GOLDEN_531, "公司的主要产品是视频指挥系统。", False),
    ("N-4 只给合计（漏分项）", GOLDEN_207, "募集资金合计30,000万元。", False),
    ("P-1 金额等价（跨单位）", "公司计划使用本次发行募集资金1.5亿元用于补充流动资金。", "1.5亿元", True),
    ("P-2 金额等价（小数位）", GOLDEN_543, "注册资本为5,520.00万元。", True),
    ("P-3 完整句（参考答案为答案子串）", GOLDEN_33,
     "报告期内，公司来自军用领域的收入占主营业务收入的比重分别为82.10%、97.31%、94.84%和94.34%。", True),
    ("X-1 口径固有容忍（故意错别字仍判对，不得当放宽证据）", GOLDEN_531, "法定代表人是程家的明。", True),
]

# 口径属性（非失败项）：无引用标签时，参考答案的**连续前缀**会被第 1 条「去标点子串」判据判对。
# 该属性在 工单1 与 工单2 的 evaluator 中完全一致（已用字节级 diff 证明），不是 工单2 引入的放宽。
PROPERTY_CASES = [
    ("口径属性：无标签的连续前缀（3/4 个比重）被判对", GOLDEN_33, "82.10%、97.31%、94.84%"),
]

ok_all = True
for name, golden_text, answer, expect in cases:
    got, reason = evaluator.check_answer(answer, golden_text)
    good = bool(got) == expect
    ok_all = ok_all and good
    print(f"{'✅' if good else '❌'} {name}")
    print(f"      期望={expect} 实测={bool(got)} 理由={reason}")

print()
print("口径属性（非失败项，工单1/工单2 的 evaluator 一致）：")
for name, golden_text, answer in PROPERTY_CASES:
    got, reason = evaluator.check_answer(answer, golden_text)
    print(f"  ℹ️  {name}: 实测={bool(got)} 理由={reason}")

print()
print("=" * 90)
print("核验 B：引用页码可回查真实 chunk（页 ∈ [1,548] 且该页存在 chunk）")
print("=" * 90)

chunks = Chunker.load(settings.paths.data_processed / "chunks.jsonl")
pages_with_chunk = {c.page for c in chunks}
page_max = max(pages_with_chunk)
print(f"索引块数={len(chunks)}；出现页码范围={min(pages_with_chunk)}..{page_max}")

results_path = REPO_ROOT / "优化" / "评估结果" / "accuracy_report.json"
data = json.loads(results_path.read_text(encoding="utf-8"))
bad: list[str] = []
total_cites = 0
for item in data["per_question"]:
    qid = item["question_id"]
    pages = item["after"]["citation_pages"] or []
    total_cites += len(pages)
    for page in pages:
        if not (isinstance(page, int) and 1 <= page <= page_max and page in pages_with_chunk):
            bad.append(f"Q{qid}: 引用页 {page} 无法回查到 chunk")
    print(f"  Q{qid}: 引用页={pages} → {'✅ 全部可回查' if not any(f'Q{qid}:' in b for b in bad) else '❌ 存在不可回查页'}")

print()
print(f"引用总数={total_cites}；不可回查={len(bad)}")
if bad:
    for b in bad:
        print(f"  ❌ {b}")

print()
print("=" * 90)
print("核验 C：优化后核心指标（读重跑后的 accuracy_report.json）")
print("=" * 90)
after = data["after"]
# --- N 守卫（防止在残缺产物上给假绿，见文件头说明）------------------------
golden_path = REPO_ROOT / "测试" / "测试数据" / "golden_qa.jsonl"
EXPECTED_N = sum(1 for line in golden_path.read_text(encoding="utf-8").splitlines() if line.strip())
after_count = int(after.get("count") or 0)
per_question_n = len(data.get("per_question") or [])
n_ok = after_count == EXPECTED_N and per_question_n == EXPECTED_N
print(f"  N 守卫             = after.count={after_count}、per_question={per_question_n}，期望 N={EXPECTED_N} → "
      + ("✅ 输入完整" if n_ok else "⚠️ 输入不完整：**本结果不可作为验收 1 证据**"))
print(f"  accuracy           = {after['accuracy']} ({after['correct']}/{after['count']})  "
      + ("✅ ≥0.90" if (after["accuracy"] >= 0.9 and n_ok)
         else ("⚠️ 不可判定（输入不完整）" if not n_ok else "❌ <0.90")))
print(f"  first_token 最大    = {after['first_token_max_ms']} ms  "
      f"{'✅ ≤3000' if after['first_token_max_ms'] <= 3000 else '❌ >3000'}")
print(f"  first_token 均值/P95= {after['first_token_avg_ms']} / {after['first_token_p95_ms']} ms")
print(f"  citation_accuracy  = {after['citation_accuracy']} ({after['citation_valid']}/{after['citation_total']})")
print(f"  生成时间           = {data['meta']['生成时间']}")

print()
print("核验 D：cold_start 字段含义核对（captain 指出命名误导）")
cold = data.get("cold_start_first_token_ms")
print(f"  cold_start_first_token_ms = {cold}")
print(f"  本次运行 LLM 答案进入终果题数 = "
      f"{sum(1 for i in data['per_question'] if i['after'].get('mode') == 'llm')}/{EXPECTED_N}（其余为抽取式）")
print("  → 该字段实为「预热后首题的首字耗时」，**不含 LLM 首次加载**，命名有误导风险（见矩阵 finding）")

verdict = ok_all and not bad and n_ok and after["accuracy"] >= 0.9 and after["first_token_max_ms"] <= 3000
print()
if not n_ok:
    print(f"⚠️ N 守卫触发：accuracy_report.json 为 {after_count} 题口径（期望 {EXPECTED_N} 题）——"
          f"输入不完整，**本次核验不可判定**，不得作为验收 1 证据。")
    print("   修复：全量重跑（不得带 --limit）→ pwsh -NoProfile -File run_py.ps1 优化/脚本/compare_optimization.py")
print(f"综合：判分口径{'保持' if ok_all else '异常'}；引用{'全部可回查' if not bad else '存在不可回查'}；"
      f"输入{'完整' if n_ok else '不完整'}；"
      f"准确率{'达标' if (after['accuracy'] >= 0.9 and n_ok) else '未达标/不可判定'}；"
      f"首字{'达标' if after['first_token_max_ms'] <= 3000 else '未达标'}")
raise SystemExit(0 if verdict else (2 if not n_ok else 1))
