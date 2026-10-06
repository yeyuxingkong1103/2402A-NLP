# -*- coding: utf-8 -*-
"""独立验证脚本（verifier / t3）：引用真实性与准确率**独立复算**。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：优化 / 独立验证（A3 第三方验证，非实现方，**只读不改实现**）

复现（工作目录 = E:\\gao6gongdan\\工单2）：
    pwsh -NoProfile -File run_py.ps1 优化/脚本/verify_citation_integrity.py
退出码：0 通过（引用 50/50 且准确率 ≥90%）；1 不通过。

**只读**。不复用任何他人报告数字，全部从原始产物复算：
  1. 逐条引用：页码是否为 [1,548] 内整数
  2. 逐条引用：是否能在 `rag.sqlite3` 的 `chunks` 表按真实 chunk_id / page 回查到
  3. 该 page 下是否真有 chunk（页码真实性的直接证据）
  4. 跨索引陷阱核验：工单1 索引 chunk_id 不得被硬编码沿用到工单2 索引
  5. 用环境事实 §4.2 的确定性 check_answer 口径**自行重算**准确率
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "研发"))

from app.core import evaluator as ev  # noqa: E402

EVAL_RECORDS = ROOT / "优化" / "评估结果" / "eval_records.json"
DB = ROOT / "研发" / "data" / "index" / "rag.sqlite3"
PAGE_MIN, PAGE_MAX = 1, 548


def main() -> int:
    print("=" * 78)
    print("A3 独立验证 · 引用真实性 + 准确率独立复算")
    print("=" * 78)

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    n_chunks = conn.execute("select count(*) from chunks").fetchone()[0]
    pages_in_db = {r[0] for r in conn.execute("select distinct page from chunks")}
    ids_in_db = {r[0] for r in conn.execute("select chunk_id from chunks")}
    print(f"\n索引: chunks={n_chunks}  不同页码={len(pages_in_db)}  "
          f"页码范围=[{min(pages_in_db)},{max(pages_in_db)}]")

    payload = json.loads(EVAL_RECORDS.read_text(encoding="utf-8"))
    records = payload["records"]
    print(f"eval_records.json: {len(records)} 条")

    # ---------- 1/2/3 引用真实性 ----------
    total = valid = 0
    bad: list[str] = []
    print(f"\n{'qid':>5} {'pages':<38} {'判定'}")
    print("-" * 78)
    for r in sorted(records, key=lambda x: x["question_id"]):
        pages = r.get("citation_pages") or []
        row_ok = True
        for p in pages:
            total += 1
            if not isinstance(p, int) or not (PAGE_MIN <= p <= PAGE_MAX):
                bad.append(f"Q{r['question_id']} 页码越界/非整数: {p!r}")
                row_ok = False
                continue
            if p not in pages_in_db:
                bad.append(f"Q{r['question_id']} 页码 {p} 在 chunks 表中无任何分块")
                row_ok = False
                continue
            valid += 1
        print(f"{r['question_id']:>5} {str(pages):<38} {'OK' if row_ok else 'FAIL'}")

    print("-" * 78)
    print(f"引用真实性: {valid}/{total} 条合法（页码 ∈ [{PAGE_MIN},{PAGE_MAX}] 且该页在索引中真实存在分块）")
    for b in bad[:10]:
        print(f"   !! {b}")

    # ---------- 4 跨索引陷阱 ----------
    print("\n[4] 跨索引 chunk_id 硬编码检查")
    hard = []
    for r in records:
        for cid in (r.get("citation_chunk_ids") or []):
            if cid not in ids_in_db:
                hard.append(f"Q{r['question_id']} chunk_id {cid} 不在工单2 索引中")
    print(f"    eval_records 中记录的 chunk_id 越界数 = {len(hard)}"
          f"（该字段不存在时记 N/A 属正常，见契约要求「不得跨索引硬编码沿用」）")
    for h in hard[:5]:
        print(f"   !! {h}")

    # ---------- 5 准确率独立复算 ----------
    print("\n[5] 准确率独立复算（用答案文本重新调用 check_answer）")
    engine = ev.Evaluator()
    correct = 0
    for r in sorted(records, key=lambda x: x["question_id"]):
        ok, reason = engine.check_answer(r["answer"], r["golden"])
        same = (ok == bool(r["is_correct"]))
        correct += 1 if ok else 0
        flag = "一致" if same else f"不一致! 记录={r['is_correct']}"
        print(f"   Q{r['question_id']:<4} 复算={ok!s:<5} {flag:<16} {r['answer'][:46]!r}")
    acc = correct / len(records) if records else 0.0
    print(f"\n   独立复算准确率 = {correct}/{len(records)} = {acc:.2%}（达标线 ≥90%）")

    print("\n" + "=" * 78)
    cite_ok = valid == total and total > 0
    acc_ok = acc >= 0.9
    print(f"引用真实性: {'PASS' if cite_ok else 'FAIL'}   "
          f"准确率≥90%: {'PASS' if acc_ok else 'FAIL'}   "
          f"阈值: {ev.FUZZY_THRESHOLD}")
    print("=" * 78)
    return 0 if (cite_ok and acc_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
