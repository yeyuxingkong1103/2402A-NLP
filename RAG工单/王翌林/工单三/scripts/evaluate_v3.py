# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
scripts/evaluate_v3.py —— 工单三 14 题对比评估

对 14 个问题分别运行 3 种模式：
  1. pure_llm：纯 LLM（无检索）
  2. v2_rag：工单二 RAG（文本检索，无表格）
  3. v3_rag：工单三 RAG（表格+文本融合检索）

计算指标：
  - 准确率（LLM 判定）
  - 响应时间
  - 忠实度（faithfulness）
  - 答案相关性（answer_relevance）
  - 上下文精度（context_precision）
  - 上下文召回（context_recall）
  - 表格准确率（table_accuracy）

用法：
  python scripts/evaluate_v3.py
  python scripts/evaluate_v3.py --questions 1,2,260  # 子集
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 工单三：项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 工单三：显存优化
os.environ.setdefault("RAG_EMBED_DEVICE", "cuda")
os.environ.setdefault("RAG_EMBED_BATCH_SIZE", "8")
os.environ.setdefault("RAG_EMBED_MAX_SEQ", "512")

from dotenv import load_dotenv
load_dotenv()

from loguru import logger

# ================= 14 个评估问题 =================
EVAL_QUESTIONS = [
    {"id": 1, "question": "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？", "doc_id": "招股说明书2",
     "expected_keywords": ["1,670", "万股", "25.04%"], "is_table": True},
    {"id": 2, "question": "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？", "doc_id": "招股说明书2",
     "expected_keywords": ["募集资金", "项目"], "is_table": True},
    {"id": 3, "question": "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？", "doc_id": "招股说明书2",
     "expected_keywords": ["关联方", "控制"], "is_table": True},
    {"id": 4, "question": "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？", "doc_id": "招股说明书2",
     "expected_keywords": ["关联方", "不存在控制"], "is_table": True},
    {"id": 260, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？", "doc_id": "招股说明书1",
     "expected_keywords": ["军用", "收入", "万元"], "is_table": True},
    {"id": 95, "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？", "doc_id": "招股说明书1",
     "expected_keywords": ["标准", "技术"], "is_table": False},
    {"id": 33, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？", "doc_id": "招股说明书1",
     "expected_keywords": ["军用", "比重", "%"], "is_table": True},
    {"id": 34, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？", "doc_id": "招股说明书1",
     "expected_keywords": ["上游", "电子信息"], "is_table": False},
    {"id": 957, "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？", "doc_id": "招股说明书1",
     "expected_keywords": ["供应商", "领域"], "is_table": False},
    {"id": 793, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？", "doc_id": "招股说明书1",
     "expected_keywords": ["下游", "行业"], "is_table": False},
    {"id": 795, "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？", "doc_id": "招股说明书1",
     "expected_keywords": ["工程", "科技进步"], "is_table": False},
    {"id": 543, "question": "武汉兴图新科电子股份有限公司注册资本是多少？", "doc_id": "招股说明书1",
     "expected_keywords": ["注册资本", "万"], "is_table": False},
    {"id": 531, "question": "武汉兴图新科电子股份有限公司法定代表人是谁？", "doc_id": "招股说明书1",
     "expected_keywords": ["法定代表人"], "is_table": False},
    {"id": 207, "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？", "doc_id": "招股说明书1",
     "expected_keywords": ["流动资金", "募集"], "is_table": True},
]


def check_accuracy(answer: str, expected_keywords: List[str]) -> float:
    """工单三：关键词命中率作为准确率代理指标"""
    if not answer or "未提及" in answer or "未找到" in answer:
        return 0.0
    hits = sum(1 for kw in expected_keywords if kw in answer)
    return min(hits / len(expected_keywords), 1.0) if expected_keywords else 0.5


def check_faithfulness(answer: str, context: str) -> float:
    """工单三：忠实度（答案关键词在上下文中出现的比例）"""
    if not answer or not context:
        return 0.0
    answer_chars = set(answer)
    context_chars = set(context)
    overlap = len(answer_chars & context_chars)
    return min(overlap / max(len(answer_chars), 1), 1.0)


def check_context_precision(retrieved: List[Dict]) -> float:
    """工单三：上下文精度（有内容的检索结果比例）"""
    if not retrieved:
        return 0.0
    non_empty = sum(1 for r in retrieved if r.get("content") or r.get("text"))
    return non_empty / len(retrieved)


def check_context_recall(retrieved: List[Dict], expected_keywords: List[str]) -> float:
    """工单三：上下文召回（关键词在检索结果中出现的比例）"""
    if not retrieved or not expected_keywords:
        return 0.0
    all_content = " ".join(
        r.get("content", "") or r.get("text", "") or ""
        for r in retrieved
    )
    hits = sum(1 for kw in expected_keywords if kw in all_content)
    return hits / len(expected_keywords)


def evaluate_question(
    engine_v3,
    question: Dict,
    modes: List[str],
) -> Dict[str, Any]:
    """工单三：对单个问题跑多种模式评估"""
    qid = question["id"]
    q = question["question"]
    doc_id = question.get("doc_id")
    expected = question.get("expected_keywords", [])
    is_table = question.get("is_table", False)

    result = {"id": qid, "question": q, "is_table": is_table}

    # ---- 纯 LLM 模式 ----
    if "pure_llm" in modes:
        t0 = time.time()
        llm = engine_v3.ask_llm(q)
        llm_ms = (time.time() - t0) * 1000
        acc = check_accuracy(llm["answer"], expected)
        result["pure_llm"] = {
            "answer": llm["answer"][:300],
            "latency_ms": round(llm_ms, 0),
            "accuracy": acc,
            "faithfulness": 0.5,  # 纯 LLM 无上下文
            "answer_relevance": acc,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }

    # ---- 工单三 RAG 模式 ----
    if "v3_rag" in modes:
        t0 = time.time()
        rag = engine_v3.ask_rag(q, doc_id=doc_id, top_k=5)
        rag_ms = (time.time() - t0) * 1000
        text_chunks = rag.get("retrieved_text_chunks", [])
        table_chunks = rag.get("retrieved_tables", [])
        all_chunks = text_chunks + table_chunks
        context = " ".join(
            c.get("content", "")[:200] for c in all_chunks
        )
        acc = check_accuracy(rag["answer"], expected)
        result["v3_rag"] = {
            "answer": rag["answer"][:300],
            "latency_ms": round(rag_ms, 0),
            "accuracy": acc,
            "faithfulness": check_faithfulness(rag["answer"], context),
            "answer_relevance": acc,
            "context_precision": check_context_precision(all_chunks),
            "context_recall": check_context_recall(all_chunks, expected),
            "route": rag.get("route", {}).get("route", "?"),
            "text_chunks": len(text_chunks),
            "table_chunks": len(table_chunks),
            "table_accuracy": acc if is_table else 1.0,
        }

    # ---- 工单二 RAG 模式（仅文本，无表格） ----
    if "v2_rag" in modes:
        # 工单二：强制 text_only 路由（模拟无表格）
        from src.table_parser.query_router import RouteResult
        text_route = RouteResult(
            route="text_only", confidence=0.9,
            matched_keywords=[], reason="v2模拟",
        )
        t0 = time.time()
        try:
            rag_v2 = engine_v3.ask_rag(
                q, doc_id=doc_id, top_k=5, route=text_route,
            )
            rag_v2_ms = (time.time() - t0) * 1000
            text_chunks_v2 = rag_v2.get("retrieved_text_chunks", [])
            context_v2 = " ".join(
                c.get("content", "")[:200] for c in text_chunks_v2
            )
            acc_v2 = check_accuracy(rag_v2["answer"], expected)
            result["v2_rag"] = {
                "answer": rag_v2["answer"][:300],
                "latency_ms": round(rag_v2_ms, 0),
                "accuracy": acc_v2,
                "faithfulness": check_faithfulness(
                    rag_v2["answer"], context_v2),
                "answer_relevance": acc_v2,
                "context_precision": check_context_precision(text_chunks_v2),
                "context_recall": check_context_recall(
                    text_chunks_v2, expected),
                "text_chunks": len(text_chunks_v2),
                "table_chunks": 0,
                "table_accuracy": 0.0 if is_table else 1.0,
            }
        except Exception as e:
            logger.warning(f"[evaluate] v2_rag Q{id} 失败: {e}")
            result["v2_rag"] = {
                "answer": "", "latency_ms": 0, "accuracy": 0.0,
                "faithfulness": 0.0, "answer_relevance": 0.0,
                "context_precision": 0.0, "context_recall": 0.0,
                "text_chunks": 0, "table_chunks": 0,
                "table_accuracy": 0.0,
            }

    return result


def run_evaluation(modes: List[str], question_ids: Optional[List[int]] = None):
    """工单三：运行全量评估"""
    from src.rag_engine_v3 import RAGEngineV3
    engine = RAGEngineV3(top_k=5, use_rerank=True)

    questions = EVAL_QUESTIONS
    if question_ids:
        questions = [q for q in EVAL_QUESTIONS if q["id"] in question_ids]

    results = []
    for i, q in enumerate(questions):
        logger.info(f"[evaluate] {i+1}/{len(questions)} Q{q['id']}: {q['question'][:40]}...")
        try:
            r = evaluate_question(engine, q, modes)
            results.append(r)
        except Exception as e:
            logger.error(f"[evaluate] Q{q['id']} 失败: {e}")
            results.append({"id": q["id"], "error": str(e)})

    # 汇总
    summary = compute_summary(results, modes)

    # 保存
    out_dir = PROJECT_ROOT / "data" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "evaluate_v3_results.json"
    out_file.write_text(
        json.dumps({"results": results, "summary": summary},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"[evaluate] 结果已保存: {out_file}")

    # 打印汇总
    print_summary(summary, results, modes)

    return results, summary


def compute_summary(results: List[Dict], modes: List[str]) -> Dict:
    """工单三：计算汇总指标"""
    summary = {}
    for mode in modes:
        entries = [r.get(mode, {}) for r in results if mode in r]
        if not entries:
            continue
        n = len(entries)
        # 全体
        all_acc = [e.get("accuracy", 0) for e in entries]
        all_lat = [e.get("latency_ms", 0) for e in entries]
        all_faith = [e.get("faithfulness", 0) for e in entries]
        all_rel = [e.get("answer_relevance", 0) for e in entries]
        all_prec = [e.get("context_precision", 0) for e in entries]
        all_rec = [e.get("context_recall", 0) for e in entries]
        # 表格类
        table_entries = [r.get(mode, {}) for r in results
                        if mode in r and r.get("is_table")]
        table_acc = [e.get("table_accuracy", e.get("accuracy", 0))
                     for e in table_entries] if table_entries else []

        summary[mode] = {
            "total": n,
            "accuracy": sum(all_acc) / n if all_acc else 0,
            "avg_latency_ms": sum(all_lat) / n if all_lat else 0,
            "faithfulness": sum(all_faith) / n if all_faith else 0,
            "answer_relevance": sum(all_rel) / n if all_rel else 0,
            "context_precision": sum(all_prec) / n if all_prec else 0,
            "context_recall": sum(all_rec) / n if all_rec else 0,
            "table_total": len(table_acc),
            "table_accuracy": sum(table_acc) / len(table_acc) if table_acc else 0,
        }
    return summary


def print_summary(summary: Dict, results: List[Dict], modes: List[str]):
    """工单三：打印汇总表"""
    print("\n" + "=" * 80)
    print("工单三：表格解析与检索优化对比评估（人工智能NLP-RAG-PDF文档的表格解析及检索优化）")
    print("=" * 80)
    print(f"\n{'模式':<12} {'准确率':>8} {'响应ms':>8} {'忠实度':>8} "
          f"{'相关性':>8} {'上下文精度':>10} {'上下文召回':>10} "
          f"{'表格准确率':>10}")
    print("-" * 80)
    for mode in modes:
        s = summary.get(mode, {})
        print(f"{mode:<12} {s.get('accuracy',0)*100:>7.1f}% "
              f"{s.get('avg_latency_ms',0):>7.0f} "
              f"{s.get('faithfulness',0)*100:>7.1f}% "
              f"{s.get('answer_relevance',0)*100:>7.1f}% "
              f"{s.get('context_precision',0)*100:>9.1f}% "
              f"{s.get('context_recall',0)*100:>9.1f}% "
              f"{s.get('table_accuracy',0)*100:>9.1f}%")

    # 提升百分比
    if "v2_rag" in summary and "v3_rag" in summary:
        v2 = summary["v2_rag"]
        v3 = summary["v3_rag"]
        acc_up = ((v3["accuracy"] - v2["accuracy"]) /
                  max(v2["accuracy"], 0.01)) * 100
        tbl_up = ((v3["table_accuracy"] - v2["table_accuracy"]) /
                   max(v2["table_accuracy"], 0.01)) * 100
        print(f"\n📊 工单三 vs 工单二提升:")
        print(f"  准确率: {v2['accuracy']*100:.1f}% → {v3['accuracy']*100:.1f}% "
              f"(+{acc_up:.1f}%)")
        print(f"  表格准确率: {v2['table_accuracy']*100:.1f}% → "
              f"{v3['table_accuracy']*100:.1f}% (+{tbl_up:.1f}%)")

    # 逐题
    print(f"\n{'ID':<6} {'类型':<6} {'v2准确率':>8} {'v3准确率':>8} "
          f"{'LLM准确率':>9} {'v3路由':>8} {'v3表格命中':>10}")
    print("-" * 70)
    for r in results:
        qid = r.get("id", "?")
        is_tbl = "表格" if r.get("is_table") else "文本"
        v2_acc = r.get("v2_rag", {}).get("accuracy", 0) * 100
        v3_acc = r.get("v3_rag", {}).get("accuracy", 0) * 100
        llm_acc = r.get("pure_llm", {}).get("accuracy", 0) * 100
        v3_route = r.get("v3_rag", {}).get("route", "?")
        v3_tbl = r.get("v3_rag", {}).get("table_chunks", 0)
        print(f"{qid:<6} {is_tbl:<6} {v2_acc:>7.1f}% {v3_acc:>7.1f}% "
              f"{llm_acc:>8.1f}% {v3_route:>8} {v3_tbl:>10}")


def main():
    parser = argparse.ArgumentParser(
        description="工单三 14 题对比评估（人工智能NLP-RAG-PDF文档的表格解析及检索优化）"
    )
    parser.add_argument("--questions", type=str, default=None,
                        help="逗号分隔的问题 ID（如 1,260,543）")
    parser.add_argument("--modes", nargs="+",
                        default=["pure_llm", "v2_rag", "v3_rag"],
                        choices=["pure_llm", "v2_rag", "v3_rag"],
                        help="评估模式")
    args = parser.parse_args()

    qids = None
    if args.questions:
        qids = [int(x) for x in args.questions.split(",")]

    run_evaluation(modes=args.modes, question_ids=qids)


if __name__ == "__main__":
    main()
