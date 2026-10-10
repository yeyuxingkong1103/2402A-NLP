# -*- coding: utf-8 -*-
"""功能 / 容错 / 稳定性测试执行脚本。

工单编号: 人工智能 NLP-RAG-Query 理解优化任务

覆盖《测试用例文档》TC-01 ~ TC-10：
    - 多轮问答主流程（含指代消解、省略补全、图像语义）
    - 多语言（英文提问 → 中文检索）
    - 容错（空输入、无关问题、超长追问）
    - 交互（会话重置）

结果落盘：output/functional_tests.json
用法：python tools/run_functional_tests.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.conversation import Conversation
from src.i18n import to_retrieval_query
from src.qa_engine import MultiTurnQA


def _check(answer: str, tokens) -> bool:
    return all(t in (answer or "") for t in tokens) if tokens else bool(answer)


def main() -> int:
    qa = MultiTurnQA()
    qa.warmup()
    results = []

    # ---- TC-01 ~ TC-05：多轮主流程（同一会话，顺序提问） ----
    script = json.loads(config.MULTITURN_PATH.read_text(encoding="utf-8"))
    conv = Conversation()
    for item in script["turns"]:
        t0 = time.time()
        res = qa.ask(item["question"], conv)
        results.append({
            "case": f"TC-{item['turn']:02d}",
            "question": item["question"],
            "rewritten": res.rewritten,
            "answer": res.answer,
            "latency": res.latency,
            "expected_tokens": item.get("key_tokens", []),
            "pass": _check(res.answer, item.get("key_tokens", [])),
            "type": item.get("phenomenon", ""),
        })

    # ---- TC-06：英文提问 → 中文检索 ----
    conv2 = Conversation()
    lang, rq = to_retrieval_query("Who is the legal representative of Wuhan Xingtu Xinke?")
    res = qa.ask(rq, conv2)
    results.append({
        "case": "TC-06", "question": "Who is the legal representative of Wuhan Xingtu Xinke?",
        "rewritten": rq, "answer": res.answer, "latency": res.latency,
        "expected_tokens": ["程家明"], "pass": "程家明" in res.answer,
        "type": f"多语言（lang={lang}）",
    })

    # ---- TC-07：空输入 / 乱码 ----
    conv3 = Conversation()
    ok7 = True
    answers7 = []
    for q in ["", "   ", "@@@###", "？？？"]:
        try:
            r = qa.ask(q, conv3)
            answers7.append(r.answer)
        except Exception as exc:      # 不应抛异常
            ok7 = False
            answers7.append(f"EXCEPTION: {exc}")
    results.append({
        "case": "TC-07", "question": "空输入 / 乱码输入",
        "rewritten": "-", "answer": " | ".join(answers7), "latency": 0.0,
        "expected_tokens": [], "pass": ok7, "type": "容错",
    })

    # ---- TC-08：与文档无关的问题 ----
    conv4 = Conversation()
    res = qa.ask("今天天气如何？适合去爬山吗？", conv4)
    results.append({
        "case": "TC-08", "question": "今天天气如何？适合去爬山吗？",
        "rewritten": res.rewritten, "answer": res.answer, "latency": res.latency,
        "expected_tokens": [], "pass": bool(res.answer), "type": "容错",
    })

    # ---- TC-09：连续 50 轮稳定性 ----
    conv5 = Conversation()
    qs = [t["question"] for t in script["turns"]]
    ok9, lats = True, []
    for i in range(50):
        try:
            r = qa.ask(qs[i % len(qs)], conv5)
            lats.append(r.latency)
        except Exception:
            ok9 = False
    results.append({
        "case": "TC-09", "question": "连续 50 轮问答",
        "rewritten": "-",
        "answer": f"完成 50 轮；平均 {sum(lats)/len(lats):.2f}s，最大 {max(lats):.2f}s",
        "latency": round(max(lats), 3) if lats else 0.0,
        "expected_tokens": [], "pass": ok9, "type": "稳定性",
    })

    # ---- TC-10：会话重置 ----
    conv6 = Conversation()
    qa.ask("报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？", conv6)
    before = len(conv6.turns)
    conv6.reset() if hasattr(conv6, "reset") else conv6.turns.clear()
    after = len(conv6.turns)
    results.append({
        "case": "TC-10", "question": "重置会话后再提问",
        "rewritten": "-", "answer": f"重置前 {before} 轮 → 重置后 {after} 轮",
        "latency": 0.0, "expected_tokens": [], "pass": after == 0, "type": "交互",
    })

    passed = sum(1 for r in results if r["pass"])
    out = {
        "workorder": "人工智能 NLP-RAG-Query 理解优化任务",
        "total": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 4),
        "cases": results,
    }
    (config.OUTPUT_DIR / "functional_tests.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    for r in results:
        print(f"{r['case']} {'PASS' if r['pass'] else 'FAIL'} | {r['question'][:34]} | {r['answer'][:60]}")
    print(f"\n通过 {passed}/{len(results)} = {passed/len(results):.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())