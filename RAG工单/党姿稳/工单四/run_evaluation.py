# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
评估脚本：对比【不含图像块（仅文本+表格）】与【含图像块（图像语义解析优化）】
在验收问题上的检索准确率，验证图像解析的价值。
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
from rag_chain import judge_contexts, retrieve_contexts
from config import EVAL_QUESTIONS

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate_all():
    all_chunks, text_chunks, img_chunks = load_all_chunks()
    print(f"文本/表格 {len(text_chunks)} 块；图像 {len(img_chunks)} 块")

    text_vec = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "vectors_text.npy")
    print("\n[构建] 不含图像块的检索器...")
    r_base = Retriever(text_chunks, use_cache=True, vectors_file=text_vec)
    print("\n[构建] 含图像块的检索器...")
    r_full = Retriever(all_chunks, use_cache=True)

    results = []
    for i, item in enumerate(EVAL_QUESTIONS, 1):
        q, qid = item["question"], item["id"]
        print(f"\n[{i}/{len(EVAL_QUESTIONS)}] Q{qid}: {q}")
        b = retrieve_contexts(q, r_base, optimize=True)
        b_hit = judge_contexts(q, b)
        o = retrieve_contexts(q, r_full, optimize=True)
        o_hit = judge_contexts(q, o)
        print(f"  无图像={b_hit} | 含图像={o_hit}")
        results.append({
            "id": qid, "question": q,
            "baseline": {"hit": b_hit, "pages": [c["page"] for c, _ in b]},
            "optimized": {"hit": o_hit, "pages": [c["page"] for c, _ in o],
                          "has_image": any(c["type"] == "image" for c, _ in o)},
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
        f.write("# 工单四 图像内容解析及检索优化 评估报告\n\n")
        f.write("**工单编号：** 人工智能NLP-RAG-图像内容解析及检索优化\n\n")
        f.write(f"**评估时间：** {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        f.write("| 指标 | 不含图像块 | 含图像块(图像解析) | 变化 |\n|---|---|---|---|\n")
        f.write(f"| 检索准确率 | {b_acc:.1%} | {o_acc:.1%} | {o_acc-b_acc:+.1%} |\n\n")
        f.write("## 逐题结果\n\n| 问题ID | 无图像 | 含图像 | 命中图像块 |\n|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['id']} | {'✅' if r['baseline']['hit'] else '❌'} | "
                    f"{'✅' if r['optimized']['hit'] else '❌'} | "
                    f"{'是' if r['optimized']['has_image'] else '否'} |\n")
    print(f"\n✅ 报告已保存: {json_path}\n{md_path}")
    print(f"准确率: 无图像 {b_acc:.1%} -> 含图像 {o_acc:.1%}")


if __name__ == "__main__":
    save_report(evaluate_all())
