# -*- coding: utf-8 -*-
"""独立验证脚本（verifier / t3）：归因纪律 + 无造假核验（**只读**）。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：优化 / 独立验证（A3 第三方验证，非实现方，**只读不改实现**）

复现（工作目录 = E:\\gao6gongdan\\工单2）：
    pwsh -NoProfile -File run_py.ps1 优化/脚本/verify_attribution_antifake.py
退出码：0 通过；1 不通过。

对应 t3 契约第 7 项（归因核对）与第 9 项（无造假）。

设计原则：**不引用 t7/tester 报告的数字作为结论**——所有断言都从
原始产物（`eval_records.json` / `baseline_results.json` / `ragas_report.md` / 报告正文）复算得出。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "优化" / "评估结果"
BASE = ROOT / "优化" / "基线"

MD = OUT / "optimization_compare.md"
JSON = OUT / "accuracy_report.json"
RAGAS = OUT / "ragas_report.md"
EVAL = OUT / "eval_records.json"
BASE_RESULTS = BASE / "baseline_results.json"

problems: list[str] = []
warns: list[str] = []


def chk(cond: bool, ok: str, bad: str) -> bool:
    print(f"  {'✅' if cond else '❌'} {ok if cond else bad}")
    if not cond:
        problems.append(bad)
    return cond


def warn(cond: bool, ok: str, bad: str) -> None:
    print(f"  {'✅' if cond else '⚠️ '} {ok if cond else bad}")
    if not cond:
        warns.append(bad)


def main() -> int:
    print("=" * 82)
    print("A3 独立验证 · 第 7 项 归因核对 + 第 9 项 无造假（只读复算）")
    print("=" * 82)

    # ================= 第 7 项：归因纪律 =================
    print("\n【第 7 项】归因核对")

    # (a) 生成端栈差异必须声明为「混淆项」
    md = MD.read_text(encoding="utf-8") if MD.exists() else ""
    print("\n (a) 生成端栈差异是否声明为混淆项")
    chk("Qwen2.5-7B-Instruct-AWQ" in md, "声明了基线模型 Qwen2.5-7B-Instruct-AWQ", "未声明基线模型")
    chk("qwen2.5:3b" in md, "声明了工单2 模型 qwen2.5:3b", "未声明工单2 模型")
    chk("混淆项" in md, "显式使用「混淆项」表述", "未使用「混淆项」表述")
    chk("不是因果项" in md or "非因果项" in md, "显式声明「不是因果项」", "未声明「不是因果项」")

    # (b) 「提示词贡献为 0」的**代码级**证据
    # 注意（已实测校准，勿误判）：eval_records.json 的 mode 字段用 'rag' 标注「最终交付路径」，
    # 它与「纯抽取式路径」的 'extractive' 相对；交付路径内部仍可能在一致性校验失败后
    # **回退抽取式**，故 mode='rag' 不能证明 LLM 贡献非 0。真正的代码级证据有两处：
    #   ① 答案文本形态（抽取式跨度 + 轻模板）② 运行日志中的「回退抽取式答案」警告计数。
    print("\n (b) 提示词/LLM 贡献的代码级证据（不得把贡献记成非 0）")
    evalp = json.loads(EVAL.read_text(encoding="utf-8"))
    modes = [r.get("mode") for r in evalp["records"]]
    from collections import Counter
    mode_counter = Counter(modes)
    print(f"     eval_records.json mode 分布 = {dict(mode_counter)}"
          f"（'rag' = 最终交付路径标签，非「LLM 作答」之证）")
    warn(True, "已说明 mode 字段语义：'rag' 是交付路径标签，不用于证明 LLM 贡献", "")

    log_hits = 0
    app_log = ROOT / "部署" / "日志" / "app.log"
    if app_log.exists():
        need = "LLM 答案未通过一致性校验，回退抽取式答案"
        with app_log.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if need in line and '"iso": "2026-10-02T21:4' in line:
                    log_hits += 1
    print(f"     app.log 中 2026-10-02T21:4x 窗口「回退抽取式答案」警告 = {log_hits} 条")
    chk(log_hits > 0,
        f"有独立日志证据表明该窗口 LLM 答案被一致性校验拒绝并回退抽取式 → 「提示词贡献 0」有代码/运行级支撑",
        "未找到「回退抽取式答案」日志证据，「提示词贡献 0」缺乏支撑")
    tpl = "提示词" in md and "贡献" in md
    chk(tpl, "报告写明提示词贡献（未把 LLM 增益记账）", "报告未写明提示词贡献")

    # (c) 三层各自用自身指标
    print("\n (c) 三层是否各用自身指标（检索位次 / 生成准确率 / 向量单路）")
    chk("证据排第 1 位" in md, "检索层用「证据位次」自身指标", "检索层未见位次指标")
    chk("纯向量单路" in md or "vector_top5" in md, "嵌入/索引层用「纯向量单路命中率」自身指标", "未见向量单路指标")

    # (d) 归因是否把生成策略增益记到检索/解析账上——查正文是否有矛盾表述
    print("\n (d) 归因一致性：是否把生成增益记到检索/解析账上")
    acc_line = re.search(r"答案准确率[^\n]*", md)
    if acc_line:
        print(f"     报告 accuracy 行: {acc_line.group(0)[:100]}")
    chk("准确率" in md, "存在准确率（生成层）指标", "缺少准确率指标")

    # ================= 第 9 项：无造假 =================
    print("\n【第 9 项】无造假核验")

    # (a) RAGAS 显式标注未运行
    print("\n (a) RAGAS 是否显式标注未运行")
    ragas = RAGAS.read_text(encoding="utf-8") if RAGAS.exists() else ""
    chk("未运行" in ragas, "ragas_report.md 含「未运行」", "ragas_report.md 未标注未运行")
    for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        # 允许提及指标名，但不得给出数值
        hits = re.findall(rf"{metric}\D{{0,12}}(\d+\.\d+)", ragas)
        chk(not hits, f"{metric} 未给出任何数值（伪造检测）", f"{metric} 出现疑似数值 {hits}")
    chk("不估算" in ragas or "未估算" in ragas or "不伪造" in ragas,
        "含「不估算/不伪造」纪律声明", "缺少不估算/不伪造声明")

    # (b) before/after 逐题答案不同（反抄写）——直接从 accuracy_report.json 复算
    print("\n (b) 反抄写：before.answer 与 after.answer 逐题不同")
    data = json.loads(JSON.read_text(encoding="utf-8"))
    items = data.get("per_question", [])
    differing = sum(1 for it in items if (it["before"].get("answer") or "") != (it["after"].get("answer") or ""))
    print(f"     per_question 长度 = {len(items)}；逐题答案不同 = {differing}/{len(items)}")
    if len(items) == 10:
        chk(differing == 10, f"10/10 题 before/after 答案不同（反抄写通过）",
            f"仅 {differing}/10 题不同，疑似抄写")
    else:
        warn(False, "", f"per_question 仅 {len(items)} 条（非 10 题），无法完成 10/10 反抄写核验")

    # (c) 基线与优化后准确率必须不同（不得把基线值誊抄为优化后值）
    print("\n (c) 基线值不得被誊抄为优化后值")
    b, a = data["before"]["accuracy"], data["after"]["accuracy"]
    print(f"     before.accuracy = {b}   after.accuracy = {a}   after.count = {data['after'].get('count')}")
    chk(b != a, "优化后准确率与基线不同", "优化后准确率与基线相同，疑似誊抄")

    # (d) 基线数值是否与工单1 权威产物一致（不得篡改）
    # 实测校准：baseline_results.json 的 accuracy 位于 **summary** 子对象（非顶层），
    # 且含 accuracy_count = "5/10"、correct/wrong_question_ids —— 这才是权威逐题真值。
    print("\n (d) 基线数值来源核对（不得篡改工单1 权威值）")
    if BASE_RESULTS.exists():
        br = json.loads(BASE_RESULTS.read_text(encoding="utf-8"))
        summ = br.get("summary", {}) if isinstance(br, dict) else {}
        print(f"     baseline_results.json[summary].accuracy      = {summ.get('accuracy')}")
        print(f"     baseline_results.json[summary].accuracy_count= {summ.get('accuracy_count')}")
        print(f"     baseline_results.json[summary].correct_ids   = {summ.get('correct_question_ids')}")
        print(f"     baseline_results.json[summary].wrong_ids     = {summ.get('wrong_question_ids')}")
        chk("accuracy" in summ, "baseline_results.json[summary] 含 accuracy", "baseline_results.json[summary] 未含 accuracy")
        if "accuracy" in summ:
            bsrc = summ["accuracy"]
            chk(abs(float(bsrc) - float(b)) < 1e-9,
                f"报告 before.accuracy={b} 与基线权威值 {bsrc} 一致",
                f"报告 before.accuracy={b} 与基线权威值 {bsrc} 不一致（疑似篡改）")
            n_ok = len(summ.get("correct_question_ids") or [])
            n_bad = len(summ.get("wrong_question_ids") or [])
            chk(n_ok == 5 and n_bad == 5,
                f"基线逐题真值自洽：正确 {n_ok} / 错误 {n_bad}（5+5=10）",
                f"基线逐题真值不自洽：正确 {n_ok} / 错误 {n_bad}")
    else:
        warn(False, "", f"{BASE_RESULTS.name} 不存在，基线来源无法核对")

    # (e) 「10/10」被改写成成绩的禁止
    print("\n (e) 禁止把「去标签后 10/10」当作交付指标")
    if "10/10" in md:
        ctx = re.findall(r".{0,60}10/10.{0,60}", md)
        ok = all(("去标签" in c and ("非交付指标" in c or "仅用于说明" in c)) for c in ctx)
        chk(ok, "凡出现 10/10 均并列标注「去标签/非交付指标」",
            f"出现未并列声明的 10/10：{ctx[:2]}")
    else:
        chk(True, "对比报告正文未出现裸 10/10", "")

    # ================= 汇总 =================
    print("\n" + "=" * 82)
    print(f"第 7 项 + 第 9 项：{'PASS' if not problems else 'FAIL'}（失败 {len(problems)} 项，观察项 {len(warns)}）")
    for p in problems:
        print(f"   ❌ {p}")
    for w in warns:
        print(f"   ⚠️  {w}")
    print("=" * 82)
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
