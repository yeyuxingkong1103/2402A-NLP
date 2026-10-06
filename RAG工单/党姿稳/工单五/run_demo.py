# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Query理解优化任务
演示脚本：跑一遍 5 轮多轮对话（含指代/省略），
并对比【关闭Query改写】与【开启Query改写】的检索页码，验证 Query 理解优化的价值。
"""
import os
import sys
import json
import time
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from kb import load_all_chunks
from retriever import Retriever
from dialog import run_dialog
from config import MULTI_TURN_DEMO

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    allc, textc, imgc = load_all_chunks()
    print(f"知识库：文本/表格 {len(textc)} 块，图像 {len(imgc)} 块")
    retriever = Retriever(allc)

    t0 = time.time()
    print("\n" + "=" * 70)
    print("【对比组】关闭 Query 改写：直接用原问句检索")
    print("=" * 70)
    log_off = run_dialog(MULTI_TURN_DEMO, retriever, use_rewrite=False)

    print("\n" + "=" * 70)
    print("【优化组】开启 Query 改写：指代消解 + 省略补全")
    print("=" * 70)
    log_on = run_dialog(MULTI_TURN_DEMO, retriever, use_rewrite=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {"timestamp": ts, "elapsed": round(time.time() - t0, 1),
              "demo_questions": MULTI_TURN_DEMO,
              "without_rewrite": log_off, "with_rewrite": log_on}
    json_path = os.path.join(OUTPUT_DIR, f"dialog_demo_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md_path = os.path.join(OUTPUT_DIR, f"dialog_demo_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 工单五 Query理解优化（多轮对话）演示报告\n\n")
        f.write("**工单编号：** 人工智能NLP-RAG-Query理解优化任务\n\n")
        f.write(f"**时间：** {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        f.write("## 多轮对话过程（开启Query改写）\n\n")
        for r in log_on:
            f.write(f"### 第{r['turn']}轮\n\n")
            f.write(f"- **用户：** {r['question']}\n")
            if r["rewritten"] != r["question"]:
                f.write(f"- **Query改写：** {r['rewritten']}\n")
            f.write(f"- **答案：** {r['answer']}\n")
            f.write(f"- **检索页码：** {r['pages']}（{ '、'.join(r['docs']) }）\n\n")
        f.write("## 改写前后检索对比\n\n| 轮次 | 用户问题 | 关闭改写·检索页码 | 开启改写·检索页码 | 改写后Query |\n|---|---|---|---|---|\n")
        for a, b in zip(log_off, log_on):
            f.write(f"| {a['turn']} | {a['question']} | {a['pages']} | {b['pages']} | {b['rewritten']} |\n")
    print(f"\n✅ 演示报告已保存:\n{json_path}\n{md_path}")


if __name__ == "__main__":
    main()
