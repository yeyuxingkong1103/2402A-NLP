# -*- coding: utf-8 -*-
"""独立验证脚本（verifier / t3 复验）：对比报告四路数值一致性 + 反抄写核验。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：优化 / 独立验证（A3 第三方验证，非实现方，**只读不改实现**）

复现（工作目录 = E:\\gao6gongdan\\工单2）：
    pwsh -NoProfile -File run_py.ps1 优化/脚本/verify_compare_consistency.py
退出码：0 全一致；1 有不一致。

核验四处数值：`optimization_compare.md` / `.csv` / `accuracy_report.json` / `eval_records.json`
并完成契约第 9(b) 项「before.answer 与 after.answer 逐题不同」的 10/10 反抄写。
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "优化" / "评估结果"
MD = OUT / "optimization_compare.md"
CSV = OUT / "optimization_compare.csv"
JSON = OUT / "accuracy_report.json"
EVAL = OUT / "eval_records.json"

EXPECTED_Q = 10
problems: list[str] = []


def chk(cond: bool, ok: str, bad: str) -> None:
    print(f"  {'✅' if cond else '❌'} {ok if cond else bad}")
    if not cond:
        problems.append(bad)


def md_accuracy(text: str) -> tuple[float | None, int | None, str]:
    """从对比报告 §2 表解析「答案准确率（N 题）」行的优化后值。"""
    m = re.search(r"答案准确率[^|\n]*\|\s*([\d.]+)\s*\|\s*([\d.]+)", text)
    if not m:
        return None, None, ""
    label = text[m.start(): m.end()]
    total_m = re.search(r"(\d+)\s*题", label)
    return float(m.group(2)), (int(total_m.group(1)) if total_m else None), label.strip()


def main() -> int:
    print("=" * 84)
    print("A3 复验 · 对比报告四路数值一致性 + 10/10 反抄写")
    print("=" * 84)

    md = MD.read_text(encoding="utf-8")
    payload = json.loads(JSON.read_text(encoding="utf-8"))
    evalp = json.loads(EVAL.read_text(encoding="utf-8"))
    with CSV.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    summary = {r["metric"]: r for r in rows if r["section"] == "summary"}

    # ---------- 1. 四处 accuracy ----------
    print("\n【1】准确率：md / csv / json / eval_records 四处")
    md_acc, md_total, md_label = md_accuracy(md)
    csv_acc = float(summary["accuracy"]["after"]) if "accuracy" in summary else None
    json_acc = payload["after"]["accuracy"]
    json_cnt = payload["after"]["count"]
    ev_ok = sum(1 for r in evalp["records"] if r["is_correct"])
    ev_n = len(evalp["records"])
    ev_acc = ev_ok / ev_n if ev_n else 0.0
    print(f"     md          = {md_acc}（表头声明 {md_total} 题）")
    print(f"     csv         = {csv_acc}")
    print(f"     json        = {json_acc}（count={json_cnt}, correct={payload['after']['correct']}）")
    print(f"     eval_records= {ev_acc:.4f}（{ev_ok}/{ev_n}）")
    chk(md_acc is not None and abs(md_acc - 0.9) < 1e-9, f"md 准确率 = 0.9", f"md 准确率 = {md_acc}，应为 0.9")
    chk(abs(csv_acc - 0.9) < 1e-9, "csv 准确率 = 0.9", f"csv 准确率 = {csv_acc}，应为 0.9")
    chk(abs(json_acc - 0.9) < 1e-9, "json 准确率 = 0.9", f"json 准确率 = {json_acc}，应为 0.9")
    chk(abs(ev_acc - 0.9) < 1e-9, "eval_records 逐题重算 = 0.9", f"eval_records 重算 = {ev_acc}")
    chk(md_total == EXPECTED_Q, f"md 表头声明 {EXPECTED_Q} 题", f"md 表头声明 {md_total} 题")
    chk(json_cnt == EXPECTED_Q, f"json after.count = {EXPECTED_Q}", f"json after.count = {json_cnt}")
    chk(len(payload["per_question"]) == EXPECTED_Q,
        f"json per_question = {EXPECTED_Q} 条", f"json per_question = {len(payload['per_question'])} 条")

    # ---------- 2. 三口径并存的矛盾是否消除 ----------
    print("\n【2】报告自相矛盾是否消除（原「10 题 / 1-1 / 9-10」三口径并存）")
    has_one_of_one = re.search(r"\b1\s*/\s*1\b", md) is not None
    has_ten = re.search(rf"{EXPECTED_Q}\s*题", md) is not None
    chk(not has_one_of_one, "报告中不再出现裸「1/1」", "报告仍出现「1/1」")
    chk(has_ten, "报告显式声明「10 题」口径", "报告未声明 10 题口径")
    chk(not (has_one_of_one and has_ten), "不再同时出现两种互相矛盾的口径", "仍同时出现矛盾口径")

    # ---------- 3. 关键汇总指标 ----------
    print("\n【3】关键汇总指标（csv 与 json 一致）")
    for metric, key in (("first_token_max_ms", "first_token_max_ms"),
                        ("first_token_avg_ms", "first_token_avg_ms"),
                        ("citation_accuracy", "citation_accuracy"),
                        ("retrieval_hit_rate", "retrieval_hit_rate")):
        c = float(summary[metric]["after"]) if metric in summary else None
        j = payload["after"].get(key)
        same = c is not None and j is not None and abs(float(c) - float(j)) < 1e-6
        chk(same, f"{metric}: csv={c} == json={j}", f"{metric}: csv={c} != json={j}")
    ft_max = float(summary["first_token_max_ms"]["after"])
    chk(ft_max <= 3000.0, f"首字最大 {ft_max} ms ≤ 3000 ms 预算", f"首字最大 {ft_max} ms 超预算")
    print(f"     余量 = {3000.0 - ft_max:.2f} ms（{100 * (3000.0 - ft_max) / 3000.0:.2f}%）")

    # ---------- 4. 10/10 反抄写（契约第 9(b) 项） ----------
    print("\n【4】反抄写：before.answer 与 after.answer 逐题不同（契约第 9(b) 项）")
    items = payload["per_question"]
    differing = 0
    for it in sorted(items, key=lambda x: x["question_id"]):
        b = (it["before"].get("answer") or "").strip()
        a = (it["after"].get("answer") or "").strip()
        same = (b == a)
        differing += 0 if same else 1
        print(f"     Q{it['question_id']:<4} {'相同 ❌' if same else '不同 ✅'}  "
              f"before={b[:28]!r}… after={a[:28]!r}…")
    chk(differing == EXPECTED_Q,
        f"{differing}/{EXPECTED_Q} 题 before/after 答案不同 → 反抄写通过",
        f"仅 {differing}/{EXPECTED_Q} 题不同，疑似抄写")

    # ---------- 5. 逐题判定一致 ----------
    print("\n【5】逐题 is_correct：json 与 eval_records 一致")
    ev_map = {r["question_id"]: r["is_correct"] for r in evalp["records"]}
    for it in sorted(items, key=lambda x: x["question_id"]):
        j = bool(it["after"].get("is_correct"))
        e = bool(ev_map.get(it["question_id"]))
        if j != e:
            problems.append(f"Q{it['question_id']} json={j} 与 eval_records={e} 不一致")
    chk(not any("不一致" in p for p in problems), "逐题判定 json 与 eval_records 完全一致", "存在逐题判定不一致")
    wrong = [q for q, v in ev_map.items() if not v]
    print(f"     唯一判错题 = {wrong}（应为 [95]）")
    chk(wrong == [95], "判错题恰为 Q95（已知遗留）", f"判错题为 {wrong}，与「仅 Q95」不符")

    # ---------- 汇总 ----------
    print("\n" + "=" * 84)
    print(f"复验结论：{'PASS（四路一致 + 10/10 反抄写）' if not problems else 'FAIL'}"
          f"（失败 {len(problems)} 项）")
    for p in problems:
        print(f"   ❌ {p}")
    print("=" * 84)
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
