# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
评估脚本：对10个验收问题，对比【优化前（仅向量检索）】与【优化后（混合检索+重排）】
的检索准确率与响应时间，输出 JSON + Markdown 报告。
"""
import os
import sys
import json
import time
from datetime import datetime

# Windows 控制台默认 GBK，统一改为 UTF-8，避免中文/符号打印报错
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
from pdf_parser import build_chunks, load_chunks
from retriever import Retriever
from rag_chain import judge_contexts, retrieve_contexts
from config import EVAL_QUESTIONS, CHUNKS_FILE

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate_one(question, retriever, optimize):
    """单次评估：检索 -> 计时 -> 裁判是否命中"""
    t0 = time.time()
    contexts = retrieve_contexts(question, retriever, optimize=optimize)
    retrieve_time = time.time() - t0
    hit = judge_contexts(question, contexts)
    return hit, retrieve_time, contexts


def evaluate_all():
    chunks = load_chunks() if os.path.exists(CHUNKS_FILE) else build_chunks()
    retriever = Retriever(chunks)

    results = []
    for i, item in enumerate(EVAL_QUESTIONS, 1):
        q, qid = item["question"], item["id"]
        print(f"\n[{i}/{len(EVAL_QUESTIONS)}] Q{qid}: {q}")

        base_hit, base_t, base_ctx = evaluate_one(q, retriever, optimize=False)
        opt_hit, opt_t, opt_ctx = evaluate_one(q, retriever, optimize=True)
        print(f"  优化前命中={base_hit} ({base_t:.2f}s) | 优化后命中={opt_hit} ({opt_t:.2f}s)")

        results.append({
            "id": qid, "question": q,
            "baseline": {"hit": base_hit, "time": round(base_t, 3),
                         "pages": [c["page"] for c, _ in base_ctx]},
            "optimized": {"hit": opt_hit, "time": round(opt_t, 3),
                          "pages": [c["page"] for c, _ in opt_ctx]},
        })
    return results


def save_report(results):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    n = len(results)
    base_acc = sum(r["baseline"]["hit"] for r in results) / n
    opt_acc = sum(r["optimized"]["hit"] for r in results) / n
    base_t = sum(r["baseline"]["time"] for r in results) / n
    opt_t = sum(r["optimized"]["time"] for r in results) / n

    summary = {
        "timestamp": ts, "n_questions": n,
        "baseline_accuracy": round(base_acc, 4),
        "optimized_accuracy": round(opt_acc, 4),
        "accuracy_improve": round(opt_acc - base_acc, 4),
        "baseline_avg_retrieve_time": round(base_t, 3),
        "optimized_avg_retrieve_time": round(opt_t, 3),
    }
    json_path = os.path.join(OUTPUT_DIR, f"eval_report_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)

    md_path = os.path.join(OUTPUT_DIR, f"eval_report_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 工单二 检索优化评估报告\n\n")
        f.write("**工单编号：** 人工智能NLP-RAG-基于PDF文档的问答系统优化\n\n")
        f.write(f"**评估时间：** {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        f.write("## 一、优化前后对比\n\n")
        f.write("| 指标 | 优化前（仅向量） | 优化后（混合检索） | 变化 |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| 检索准确率 | {base_acc:.1%} | {opt_acc:.1%} | {opt_acc-base_acc:+.1%} |\n")
        f.write(f"| 平均检索耗时 | {base_t:.3f}s | {opt_t:.3f}s | {opt_t-base_t:+.3f}s |\n\n")
        f.write("## 二、逐题结果\n\n")
        f.write("| 问题ID | 优化前命中 | 优化后命中 | 优化前页 | 优化后页 |\n|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['id']} | {'✅' if r['baseline']['hit'] else '❌'} | "
                    f"{'✅' if r['optimized']['hit'] else '❌'} | "
                    f"{r['baseline']['pages']} | {r['optimized']['pages']} |\n")
    print(f"\n✅ 报告已保存:\n  {json_path}\n  {md_path}")
    print(f"\n准确率: 优化前 {base_acc:.1%} -> 优化后 {opt_acc:.1%}")
    return json_path, md_path


if __name__ == "__main__":
    save_report(evaluate_all())
