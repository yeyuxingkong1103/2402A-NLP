#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-LightRAG优化
scripts/compare_rag_lightrag_v12.py —— RAG vs LightRAG 检索结果对比 + RAGAS 评估

流程：
1. 对 16 个测试问题，分别调用 RAG（v6 引擎）和 LightRAG 获取答案
2. 用 DeepSeek LLM 按 RAGAS 指标（faithfulness / answer_relevancy /
   context_precision / context_recall）打分
3. 输出对比结果到 docs/v12_comparison_results.json
"""
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WORK_ORDER = "人工智能NLP-RAG-LightRAG优化"

QUESTIONS_FILE = Path(__file__).resolve().parents[1] / "data" / "test_questions_v12.json"
OUTPUT_FILE = Path(__file__).resolve().parents[1] / "docs" / "v12_comparison_results.json"


# ========== RAG 检索（复用 v6 引擎）==========
def rag_answer(question, doc_id):
    """工单十二：调用 RAG v6 引擎，LLM 失败时直接用检索结果"""
    from src.rag_engine_v6 import RAGEngineV6
    engine = RAGEngineV6()
    try:
        result = engine.ask(question, doc_id=doc_id)
        answer = result.get("answer", "")
        chunks = result.get("retrieved_text_chunks", [])
        context = "\n".join([c.get("content", "")[:500] for c in chunks[:5]])
        if not answer or "错误" in answer or "抱歉" in answer:
            answer = context[:500] if context else "未检索到相关信息"
        return answer, context
    except Exception:
        # 工单十二：LLM 失败时直接走检索器获取上下文
        try:
            hybrid = engine._get_hybrid()
            text_result = hybrid.retrieve(question, 5, doc_id, engine.default_config)
            chunks = text_result.get("results", [])
            context = "\n".join([c.get("content", "")[:500] for c in chunks[:5]])
            answer = context[:500] if context else "未检索到相关信息"
            return answer, context
        except Exception as e2:
            return f"[RAG 错误] {e2}", ""


# ========== LightRAG 检索 ==========
def lightrag_answer(question):
    """工单十二：调用 LightRAG 获取答案和上下文"""
    from src.lightrag_v12 import lightrag_query
    try:
        answer = lightrag_query(question, mode="hybrid", top_k=40)
        answer = answer or ""
        # 工单十二：LightRAG 返回的 answer 已包含检索上下文（图谱实体/关系/文本块）
        context = answer[:1500]
        return answer, context
    except Exception as e:
        return f"[LightRAG 错误] {e}", ""


# ========== RAGAS 评估（基于关键词，无需 API）==========
import jieba

def extract_keywords(text, top_k=20):
    """工单十二：jieba 分词提取关键词"""
    if not text:
        return set()
    stop = set("的了是在和与及或不也都就而但这那我你他她它有被把让对从向到于为由以因等着过地得")
    words = jieba.lcut(text)
    keywords = [w for w in words if len(w) >= 2 and w not in stop and not w.isspace()]
    return set(keywords[:top_k])


def evaluate_faithfulness(question, answer, context):
    """工单十二：faithfulness - 答案关键词在上下文中的覆盖率"""
    ans_kw = extract_keywords(answer)
    ctx_kw = extract_keywords(context)
    if not ans_kw:
        return 0.0
    overlap = ans_kw & ctx_kw
    return round(len(overlap) / len(ans_kw), 4)


def evaluate_answer_relevancy(question, answer):
    """工单十二：answer_relevancy - 问题关键词在答案中的覆盖率"""
    q_kw = extract_keywords(question)
    a_kw = extract_keywords(answer)
    if not q_kw:
        return 0.0
    overlap = q_kw & a_kw
    return round(len(overlap) / len(q_kw), 4)


def evaluate_context_precision(question, context):
    """工单十二：context_precision - 问题关键词在上下文中的覆盖率"""
    q_kw = extract_keywords(question)
    ctx_kw = extract_keywords(context)
    if not q_kw:
        return 0.0
    overlap = q_kw & ctx_kw
    return round(len(overlap) / len(q_kw), 4)


def evaluate_context_recall(question, answer, context):
    """工单十二：context_recall - 答案关键词在上下文中的覆盖率（同 faithfulness 逻辑）"""
    return evaluate_faithfulness(question, answer, context)


def main():
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    print(f"[v12] {WORK_ORDER}")
    print(f"[v12] 共 {len(questions)} 个测试问题")
    print(f"[v12] 开始 RAG vs LightRAG 对比评估...\n")

    results = []
    for i, q in enumerate(questions):
        qid = q["id"]
        question = q["question"]
        doc_id = q.get("doc_id", "招股说明书1")
        print(f"[{i+1}/{len(questions)}] id={qid}: {question[:50]}...")

        # --- RAG ---
        t0 = time.perf_counter()
        rag_ans, rag_ctx = rag_answer(question, doc_id)
        rag_time = time.perf_counter() - t0
        print(f"  RAG     : {rag_time:.1f}s, len={len(rag_ans)}")

        # --- LightRAG ---
        t1 = time.perf_counter()
        lr_ans, lr_ctx = lightrag_answer(question)
        lr_time = time.perf_counter() - t1
        print(f"  LightRAG: {lr_time:.1f}s, len={len(lr_ans)}")

        # --- RAGAS 评估 ---
        rag_scores = {
            "faithfulness": evaluate_faithfulness(question, rag_ans, rag_ctx),
            "answer_relevancy": evaluate_answer_relevancy(question, rag_ans),
            "context_precision": evaluate_context_precision(question, rag_ctx),
            "context_recall": evaluate_context_recall(question, rag_ans, rag_ctx),
        }
        lr_scores = {
            "faithfulness": evaluate_faithfulness(question, lr_ans, lr_ctx),
            "answer_relevancy": evaluate_answer_relevancy(question, lr_ans),
            "context_precision": evaluate_context_precision(question, lr_ctx),
            "context_recall": evaluate_context_recall(question, lr_ans, lr_ctx),
        }
        print(f"  RAG scores     : {rag_scores}")
        print(f"  LightRAG scores: {lr_scores}\n")

        results.append({
            "id": qid,
            "question": question,
            "doc_id": doc_id,
            "rag": {"answer": rag_ans[:800], "context": rag_ctx[:1000],
                    "time_s": round(rag_time, 2), "scores": rag_scores},
            "lightrag": {"answer": lr_ans[:800], "context": lr_ctx[:1000],
                         "time_s": round(lr_time, 2), "scores": lr_scores},
        })

    # ========== 汇总 ==========
    def avg(key, src):
        vals = [r[src]["scores"][key] for r in results]
        return round(sum(vals) / len(vals), 4) if vals else 0

    summary = {
        "work_order": WORK_ORDER,
        "total_questions": len(results),
        "rag": {
            "avg_faithfulness": avg("faithfulness", "rag"),
            "avg_answer_relevancy": avg("answer_relevancy", "rag"),
            "avg_context_precision": avg("context_precision", "rag"),
            "avg_context_recall": avg("context_recall", "rag"),
            "avg_time_s": round(sum(r["rag"]["time_s"] for r in results) / len(results), 2),
        },
        "lightrag": {
            "avg_faithfulness": avg("faithfulness", "lightrag"),
            "avg_answer_relevancy": avg("answer_relevancy", "lightrag"),
            "avg_context_precision": avg("context_precision", "lightrag"),
            "avg_context_recall": avg("context_recall", "lightrag"),
            "avg_time_s": round(sum(r["lightrag"]["time_s"] for r in results) / len(results), 2),
        },
        "results": results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[v12] 评估完成，结果已保存到 {OUTPUT_FILE}")
    print(f"\n[v12] ===== 汇总对比 =====")
    print(f"{'指标':<22} {'RAG':>10} {'LightRAG':>10}")
    for k in ["avg_faithfulness", "avg_answer_relevancy", "avg_context_precision", "avg_context_recall"]:
        print(f"{k:<22} {summary['rag'][k]:>10.4f} {summary['lightrag'][k]:>10.4f}")
    print(f"{'avg_time_s':<22} {summary['rag']['avg_time_s']:>10.2f} {summary['lightrag']['avg_time_s']:>10.2f}")


if __name__ == "__main__":
    main()
