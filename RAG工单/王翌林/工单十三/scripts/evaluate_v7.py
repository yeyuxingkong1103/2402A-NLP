# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-功能测试及评估
scripts/evaluate_v7.py —— 工单七 10 题 RAG 功能测试与检索评估（新增）

流程：
  1. 加载 01-06 工单实现的 RAG 系统（RAGEngineV6，默认 hybrid 混合检索）
  2. 读取 data/ccf_reports/test_questions_v7.json 的 10 个测试题（金标已核实）
  3. 逐题执行 RAG：记录检索结果（top5 chunk 的 doc/page/score/通道/内容）+ 答案 + 耗时
  4. 用 src.evaluation_v7 评估框架计算：
     doc_recall@5 / MRR / 上下文关键词召回 / 答案准确率 / 问题自动归类
  5. 输出 docs/eval_v7_results.json（每题完整检索结果+评估结果+问题标签）

用法：python scripts/evaluate_v7.py [--top-k 5] [--strategy hybrid]
"""
import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv()

from loguru import logger  # noqa: E402

from src.evaluation_v7 import evaluate_case, summarize  # noqa: E402

WORK_ORDER = "人工智能NLP-RAG-功能测试及评估"
QUESTIONS_PATH = PROJECT_ROOT / "data" / "ccf_reports" / "test_questions_v7.json"
OUT_JSON = PROJECT_ROOT / "docs" / "eval_v7_results.json"


def run(top_k: int = 5, strategy: str = "hybrid") -> dict:
    from src.rag_engine_v6 import RAGEngineV6

    cases = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    logger.info(f"[v7] 加载 RAGEngineV6（01-06 工单 RAG 系统，strategy={strategy}）")
    engine = RAGEngineV6(top_k=top_k)
    print("--- 工单七：预热（bge-m3/reranker/全文索引/一条真实查询）---")
    engine.warmup()
    engine.ask("平安银行2019年经营情况", doc_id="平安银行2019年报",
               retrieval_config={"use_table": False, "use_image": False})
    print("--- 预热完成，开始 10 题测试 ---\n")

    rows = []
    for q in cases:
        cfg = {"use_table": False, "use_image": False}
        if strategy != "hybrid":
            cfg["mode"] = strategy
        t0 = time.perf_counter()
        try:
            result = engine.ask(q["question"], doc_id=q.get("doc_id"),
                                retrieval_config=cfg)
            latency = (time.perf_counter() - t0) * 1000
            retrieved = result.get("retrieved_text_chunks", [])
            metrics = evaluate_case(
                retrieved, result.get("answer", ""),
                q["gold_docs"], q["keywords"], latency, k=top_k)
            # 工单七：保存完整检索结果（每题 top-k chunk 快照）
            results_snapshot = [{
                "rank": i + 1,
                "doc_id": c.get("doc_id"), "chunk_id": c.get("chunk_id"),
                "page": c.get("page"),
                "search_path": c.get("search_path", ""),
                "score": round(float(c.get("rerank_score",
                                           c.get("score", 0))), 4),
                "content": (c.get("content") or "")[:300],
            } for i, c in enumerate(retrieved[:top_k])]
            row = {"id": q["id"], "question": q["question"],
                   "doc_id_filter": q.get("doc_id"),
                   "gold_docs": q["gold_docs"],
                   "answer_points": q["answer_points"], "source": q["source"],
                   "answer": result.get("answer", ""),
                   "retrieval_meta": result.get("retrieval", {}),
                   "results": results_snapshot, "metrics": metrics,
                   "error": None}
        except Exception as e:
            logger.exception(f"[v7] 题 {q['id']} 执行失败")
            row = {"id": q["id"], "question": q["question"],
                   "doc_id_filter": q.get("doc_id"),
                   "gold_docs": q["gold_docs"],
                   "answer_points": q["answer_points"], "source": q["source"],
                   "answer": "", "retrieval_meta": {}, "results": [],
                   "metrics": None, "error": str(e)}
        rows.append(row)
        if metrics := row["metrics"]:
            print(f"Q{q['id']:>2} doc_recall@5={metrics['doc_recall@5']} "
                  f"mrr={metrics['mrr']} ctx_recall={metrics['context_recall']} "
                  f"ans_acc={metrics['answer_acc']} "
                  f"lat={metrics['latency_ms']}ms "
                  f"issues={'；'.join(metrics['issues']) or '无'}")

    valid = [r for r in rows if r["metrics"]]
    summary = summarize(valid)
    output = {"work_order": WORK_ORDER,
              "ts": time.strftime("%F %T"),
              "corpus": "ccf_competition（9 份 A 股年报 2019-2021）",
              "rag_system": "RAGEngineV6（01-06 工单成果，hybrid 混合检索+LLM重排）",
              "strategy": strategy, "top_k": top_k,
              "questions_file": str(QUESTIONS_PATH.relative_to(PROJECT_ROOT)),
              "rows": rows, "summary": summary}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    print("\n| 指标 | 值 | 验收参考 |")
    print("|---|---|---|")
    print(f"| 金标文档召回 doc_recall@5 | {summary['doc_recall@5_avg']} | 越高越好 |")
    print(f"| MRR | {summary['mrr_avg']} | 越高越好 |")
    print(f"| 上下文关键词召回 | {summary['context_recall_avg']} | ≥0.95 |")
    print(f"| 答案准确率 | {summary['answer_accuracy']} | ≥0.90 |")
    print(f"| 平均响应 ms | {summary['latency_avg_ms']} | ≤3000 |")
    print(f"| ≤3s 占比 | {summary['latency_ok_rate']} | — |")
    print("\n问题分布：")
    for issue, cnt in summary["issue_distribution"].items():
        print(f"  - {issue}: {cnt} 题")
    print(f"\n完整结果已写入 {OUT_JSON}")
    return output


def main() -> int:
    p = argparse.ArgumentParser(description=WORK_ORDER)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--strategy", default="hybrid",
                   choices=["hybrid", "vector", "fulltext"])
    args = p.parse_args()
    run(top_k=args.top_k, strategy=args.strategy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
