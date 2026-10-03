# -*- coding: utf-8 -*-
"""独立验证脚本（verifier / t3）：判分口径未放宽核验。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：优化 / 独立验证（A3 第三方验证，非实现方，**只读不改实现**）

复现（工作目录 = E:\\gao6gongdan\\工单2）：
    pwsh -NoProfile -File run_py.ps1 优化/脚本/verify_evaluator_gauge.py
退出码：0 通过（口径未放宽）；1 不通过（阈值被改或负例被判对）。

本脚本**只读**调用 `研发/app/core/evaluator.py` 的判分逻辑，不改动任何实现文件。
核验内容（对应 t3 契约第 8 项）：
  1. FUZZY_THRESHOLD 是否为 0.62（未被下调）
  2. 环境事实 §4.2.1 的判错负例（N-1 多值漏答）是否仍判错
  3. 判对正例（P-1/P-2 金额等价）是否仍判对，证明阈值可达到而非被人为抬高
  4. PUNCT_TO_STRIP 是否含 `[` `]` 但不含「页」「码」（解释 Q95 口径交互）
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "研发"))

from app.core import evaluator as ev  # noqa: E402


def main() -> int:
    print("=" * 78)
    print("A3 独立验证 · 判分口径核验（第三方，只读）")
    print("=" * 78)

    # ---------- 1. 阈值 ----------
    print(f"\n[1] FUZZY_THRESHOLD = {ev.FUZZY_THRESHOLD!r}")
    thr_ok = ev.FUZZY_THRESHOLD == 0.62
    print(f"    期望 0.62 -> {'PASS' if thr_ok else 'FAIL'}")

    # ---------- 2/3. 负例与正例 ----------
    # N-1 多值漏答：参考答案 4 个比重（Q33 的真实形态），回答只给 1 个
    golden_33 = (
        "报告期内，公司来自军用领域的收入占主营业务收入的比重分别为"
        "82.10%、97.31%、94.84%和94.34%。"
    )
    neg_cases = [
        ("N-1 多值漏答（4 个比重只答 1 个，即基线 Q33 形态）",
         "报告期内来自军用领域的收入占主营业务收入比重为82.10%。", golden_33),
        ("N-3 答非所问", "今天天气不错，适合出门散步。", golden_33),
        ("N-2 单位错（把 5,520 万元写成 5,520 元）",
         "武汉兴图新科电子股份有限公司注册资本为5,520元。",
         "武汉兴图新科电子股份有限公司注册资本为5,520万元。"),
    ]
    pos_cases = [
        ("P-1 金额等价跨单位（1.5 亿元 == 15,000 万元）",
         "募集资金15,000万元用于补充流动资金。",
         "本次发行募集资金1.5亿元用于补充流动资金。"),
        ("P-2 金额等价小数位（5,520 == 5,520.00 万元）",
         "武汉兴图新科电子股份有限公司注册资本：5,520.00 万元。",
         "武汉兴图新科电子股份有限公司注册资本为5,520万元。"),
        ("P-3 完整句（参考答案为答案子串，忽略标点）",
         "[页码: 52] 武汉兴图新科电子股份有限公司注册资本：5,520 万元。",
         "武汉兴图新科电子股份有限公司注册资本为5,520万元。"),
    ]

    engine = ev.Evaluator() if hasattr(ev, "Evaluator") else None
    print(f"\n[2] 判分器实例: {type(engine).__name__ if engine else '无 Evaluator 类'}")

    all_ok = thr_ok
    if engine is not None and hasattr(engine, "check_answer"):
        print("\n--- 负例（期望 is_correct=False） ---")
        for name, ans, gold in neg_cases:
            ok, reason = engine.check_answer(ans, gold)
            verdict = "PASS" if ok is False else "FAIL（被判对=口径放宽）"
            all_ok = all_ok and (ok is False)
            print(f"  [{verdict}] {name}\n      is_correct={ok}  reason={reason}")

        print("\n--- 正例（期望 is_correct=True） ---")
        for name, ans, gold in pos_cases:
            ok, reason = engine.check_answer(ans, gold)
            verdict = "PASS" if ok is True else "FAIL"
            all_ok = all_ok and (ok is True)
            print(f"  [{verdict}] {name}\n      is_correct={ok}  reason={reason}")
    else:
        print("  !! 未找到 Evaluator.check_answer，无法核验判分用例")

    # ---------- 4. 标点表 ----------
    print("\n[3] PUNCT_TO_STRIP 构成")
    has_brackets = "[" in ev.PUNCT_TO_STRIP and "]" in ev.PUNCT_TO_STRIP
    has_ye = "页" in ev.PUNCT_TO_STRIP
    has_ma = "码" in ev.PUNCT_TO_STRIP
    print(f"    含 '[' ']' = {has_brackets}")
    print(f"    含 '页' = {has_ye}，含 '码' = {has_ma}")
    print(f"    -> `[页码: 160]` 去标点后残留 `页码160`：{'符合预期' if (has_brackets and not has_ye and not has_ma) else '与预期不符'}")

    print("\n" + "=" * 78)
    print(f"判分口径核验总体: {'PASS（未放宽，阈值 0.62）' if all_ok else 'FAIL'}")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
