# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
评估脚本：对比【仅文本块检索（无表格解析）】与【文本块+表格块检索（表格解析优化）】
在 14 个验收问题上的检索准确率，验证表格解析带来的提升。
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

from pdf_parser import build_chunks, load_chunks
from retriever import Retriever
from rag_chain import judge_contexts, retrieve_contexts
from config import EVAL_QUESTIONS, CHUNKS_FILE

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate_one(question, retriever, top_k=5):
    t0 = time.time()
    contexts = retrieve_contexts(question, retriever, top_k=top_k, optimize=True)
    t = time.time() - t0
    return judge_contexts(question, contexts), t, contexts


def evaluate_all():
    chunks = load_chunks() if os.path.exists(CHUNKS_FILE) else build_chunks()
    text_only = [c for c in chunks if c["type"] == "text"]
    print(f"知识库：全部 {len(chunks)} 块；纯文本 {len(text_only)} 块；"
          f"表格 {len(chunks) - len(text_only)} 块")

    print("\n[构建] 无表格解析检索器（仅文本块）...")
    r_base = Retriever(text_only, use_cache=False)
    print("\n[构建] 表格解析检索器（文本+表格）...")
    r_full = Retriever(chunks, use_cache=True)

    results = []
    for i, item in enumerate(EVAL_QUESTIONS, 1):
        q, qid = item["question"], item["id"]
        print(f"\n[{i}/{len(EVAL_QUESTIONS)}] Q{qid}: {q}")
        b_hit, b_t, b_ctx = evaluate_one(q, r_base)
        o_hit, o_t, o_ctx = evaluate_one(q, r_full)
        print(f"  无表格={b_hit} ({b_t:.2f}s) | 表格解析={o_hit} ({o_t:.2f}s)")
        results.append({
            "id": qid, "question": q,
            "baseline": {"hit": b_hit, "time": round(b_t, 3),
                         "pages": [c["page"] for c, _ in b_ctx],
                         "has_table": any(c["type"] == "table" for c, _ in b_ctx)},
            "optimized": {"hit": o_hit, "time": round(o_t, 3),
                          "pages": [c["page"] for c, _ in o_ctx],
                          "has_table": any(c["type"] == "table" for c, _ in o_ctx)},
        })
    return results


def save_report(results):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    n = len(results)
    b_acc = sum(r["baseline"]["hit"] for r in results) / n
    o_acc = sum(r["optimized"]["hit"] for r in results) / n

    summary = {"timestamp": ts, "n_questions": n,
               "baseline_accuracy": round(b_acc, 4),
               "optimized_accuracy": round(o_acc, 4),
               "accuracy_improve": round(o_acc - b_acc, 4)}
    json_path = os.path.join(OUTPUT_DIR, f"eval_report_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)

    md_path = os.path.join(OUTPUT_DIR, f"eval_report_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 工单三 表格解析及检索优化 评估报告\n\n")
        f.write("**工单编号：** 人工智能NLP-RAG-PDF文档的表格解析及检索优化\n\n")
        f.write(f"**评估时间：** {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        f.write("## 一、对比结论\n\n")
        f.write("| 指标 | 无表格解析（仅文本） | 表格解析优化 | 变化 |\n|---|---|---|---|\n")
        f.write(f"| 检索准确率 | {b_acc:.1%} | {o_acc:.1%} | {o_acc-b_acc:+.1%} |\n\n")
        f.write("## 二、逐题结果\n\n")
        f.write("| 问题ID | 无表格命中 | 表格解析命中 | 命中表格块 |\n|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['id']} | {'✅' if r['baseline']['hit'] else '❌'} | "
                    f"{'✅' if r['optimized']['hit'] else '❌'} | "
                    f"{'是' if r['optimized']['has_table'] else '否'} |\n")
    print(f"\n✅ 报告已保存:\n  {json_path}\n  {md_path}")
    print(f"\n准确率: 无表格 {b_acc:.1%} -> 表格解析 {o_acc:.1%}")
    return json_path, md_path


if __name__ == "__main__":
    save_report(evaluate_all())
