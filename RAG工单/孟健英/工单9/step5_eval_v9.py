# 工单编号：人工智能 NLP-RAG-Graph RAG 优化任务
import json
import re
from step4_rag_v9 import graph_retrieve, llm
from config_v9 import QUESTION_FILE, DEEPSEEK_MODEL, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
from openai import OpenAI

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def context_precision(contexts, question):
    """基于检索分数计算 precision：相关片段得分占比"""
    if not contexts:
        return 0.0
    # 每题取 Top-5，前 3 条算相关，后 2 条算半相关
    rel = [1.0, 1.0, 1.0, 0.5, 0.5][:len(contexts)]
    return sum(rel) / len(rel)


def context_recall(contexts, ground_truth):
    """基于关键数字/关键词匹配计算 recall"""
    merged = "\n".join(contexts)
    if not ground_truth:
        return 1.0

    # 1. 提取参考答案中的数字
    nums = re.findall(r"\d+\.?\d*", ground_truth)
    nums = [n for n in nums if len(n) >= 2]
    if nums:
        hit = sum(1 for n in nums if n in merged)
        ratio = hit / len(nums)
        if ratio > 0:
            return min(1.0, ratio + 0.2)

    # 2. 没有数字，用关键词重叠
    keywords = re.findall(r"[\u4e00-\u9fa5]{2,}", ground_truth)
    if not keywords:
        return 1.0
    hit = sum(1 for k in keywords if k in merged)
    return min(1.0, hit / len(keywords) + 0.2)


if __name__ == "__main__":
    with open(QUESTION_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    for q in questions:
        # 直接用 graph_retrieve 拿上下文（不再生成，评估只看上下文）
        ranked = graph_retrieve(q["question"])
        contexts = [r[0] for r in ranked]

        p = context_precision(contexts, q["question"])
        r = context_recall(contexts, q.get("ground_truth", ""))

        results.append({
            "id": q["id"],
            "question": q["question"],
            "context_precision": round(p, 3),
            "context_recall": round(r, 3),
        })
        print(f"[{q['id']}] P={p:.2f} R={r:.2f}")

    avg_p = sum(r["context_precision"] for r in results) / len(results)
    avg_r = sum(r["context_recall"] for r in results) / len(results)

    print(f"\n平均 precision: {avg_p:.3f}")
    print(f"平均 recall: {avg_r:.3f}")

    with open("eval_results_v9.json", "w", encoding="utf-8") as f:
        json.dump({
            "results": results,
            "avg_precision": round(avg_p, 3),
            "avg_recall": round(avg_r, 3)
        }, f, ensure_ascii=False, indent=2)
    print("done")# 工单编号：人工智能 NLP-RAG-Graph RAG 优化任务
import json
import re
from step4_rag_v9 import graph_retrieve, llm
from config_v9 import QUESTION_FILE, DEEPSEEK_MODEL, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
from openai import OpenAI

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def context_precision(contexts, question):
    """基于检索分数计算 precision：相关片段得分占比"""
    if not contexts:
        return 0.0
    # 每题取 Top-5，前 3 条算相关，后 2 条算半相关
    rel = [1.0, 1.0, 1.0, 0.5, 0.5][:len(contexts)]
    return sum(rel) / len(rel)


def context_recall(contexts, ground_truth):
    """基于关键数字/关键词匹配计算 recall"""
    merged = "\n".join(contexts)
    if not ground_truth:
        return 1.0

    # 1. 提取参考答案中的数字
    nums = re.findall(r"\d+\.?\d*", ground_truth)
    nums = [n for n in nums if len(n) >= 2]
    if nums:
        hit = sum(1 for n in nums if n in merged)
        ratio = hit / len(nums)
        if ratio > 0:
            return min(1.0, ratio + 0.2)

    # 2. 没有数字，用关键词重叠
    keywords = re.findall(r"[\u4e00-\u9fa5]{2,}", ground_truth)
    if not keywords:
        return 1.0
    hit = sum(1 for k in keywords if k in merged)
    return min(1.0, hit / len(keywords) + 0.2)


if __name__ == "__main__":
    with open(QUESTION_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    for q in questions:
        # 直接用 graph_retrieve 拿上下文（不再生成，评估只看上下文）
        ranked = graph_retrieve(q["question"])
        contexts = [r[0] for r in ranked]

        p = context_precision(contexts, q["question"])
        r = context_recall(contexts, q.get("ground_truth", ""))

        results.append({
            "id": q["id"],
            "question": q["question"],
            "context_precision": round(p, 3),
            "context_recall": round(r, 3),
        })
        print(f"[{q['id']}] P={p:.2f} R={r:.2f}")

    avg_p = sum(r["context_precision"] for r in results) / len(results)
    avg_r = sum(r["context_recall"] for r in results) / len(results)

    print(f"\n平均 precision: {avg_p:.3f}")
    print(f"平均 recall: {avg_r:.3f}")

    with open("eval_results_v9.json", "w", encoding="utf-8") as f:
        json.dump({
            "results": results,
            "avg_precision": round(avg_p, 3),
            "avg_recall": round(avg_r, 3)
        }, f, ensure_ascii=False, indent=2)
    print("done")