# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块说明：评估脚本。针对工单给定的问题列表，分别运行
          （1）RAG 问答链路 与 （2）仅大模型（无检索）回答，
          进行对比分析，并通过 RAG 评估体系（RAGAS / 内置评审）输出量化结果。

使用示例：
    python evaluate.py                       # 完整评估并导出结果
    python evaluate.py --limit 3             # 只评估前 3 个问题
    python evaluate.py --no-ragas            # 仅使用内置评审
    python evaluate.py --ground-truth data/ground_truth.json
"""
import argparse
import json
import statistics
import time
from datetime import datetime

import config
from src.evaluation import EvalRecord, evaluate_records_builtin
from src.utils import logger


def load_questions(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("questions", data if isinstance(data, list) else [])


def load_ground_truth(path: str) -> dict:
    """加载参考答案（可选），格式：{"260": "参考答案文本", ...}"""
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {str(k): v for k, v in json.load(f).items()}
    except Exception as exc:
        logger.warning("参考答案加载失败：%s", exc)
        return {}


def run_evaluation(questions: list, ground_truth: dict = None, limit: int = None,
                   use_ragas: bool = True, skip_pure: bool = False) -> dict:
    """执行评估主流程。"""
    from src.rag_chain import RAGPipeline

    ground_truth = ground_truth or {}
    pipeline = RAGPipeline()
    subset = questions[:limit] if limit else questions

    records = []
    for idx, item in enumerate(subset, 1):
        qid = item.get("id")
        question = item.get("question", "")
        logger.info("[%d/%d] 评估问题 id=%s", idx, len(subset), qid)

        record = EvalRecord(question_id=qid, question=question)
        record.ground_truth = ground_truth.get(str(qid), "")

        # RAG 链路
        start = time.perf_counter()
        try:
            result = pipeline.ask(question)
            record.rag_answer = result.answer
            record.contexts = result.context_texts
            record.pages = result.pages
        except Exception as exc:
            record.rag_answer = f"[RAG 失败] {exc}"
            logger.error("RAG 问答失败：%s", exc)
        record.response_time = round(time.perf_counter() - start, 3)

        # 纯大模型对照
        if not skip_pure:
            try:
                record.pure_answer = pipeline.ask_pure_llm(question)
            except Exception as exc:
                record.pure_answer = f"[纯 LLM 失败] {exc}"

        records.append(record)

    # 量化评估
    eval_result = evaluate_records_builtin(records, use_ragas=use_ragas)

    times = [r.response_time for r in records]
    summary = {
        "work_order_no": config.WORK_ORDER_NO,
        "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "questions_total": len(records),
        "retrieval_hit_rate": round(
            sum(1 for r in records if r.contexts) / len(records), 4) if records else 0.0,
        "response_time_avg": round(statistics.mean(times), 3) if times else 0.0,
        "response_time_p95": round(
            statistics.quantiles(times, n=20)[-1] if len(times) > 1 else (times[0] if times else 0.0), 3),
        "response_time_max": round(max(times), 3) if times else 0.0,
        "within_3s_ratio": round(
            sum(1 for t in times if t <= config.RESPONSE_TIME_TARGET) / len(times), 4) if times else 0.0,
        "eval_backend": eval_result["backend"],
        "ragas_scores": eval_result["scores"],
    }
    return {"summary": summary, "records": records, "details": eval_result["details"]}


def export_results(result: dict, output_dir: str) -> tuple:
    """导出 JSON 与 CSV 结果，返回文件路径。"""
    import pandas as pd

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"{output_dir}/eval_result_{stamp}.json"
    csv_path = f"{output_dir}/eval_result_{stamp}.csv"

    rows = []
    for rec in result["records"]:
        rows.append(
            {
                "id": rec.question_id,
                "question": rec.question,
                "rag_answer": rec.rag_answer,
                "pure_llm_answer": rec.pure_answer,
                "retrieved_pages": "、".join(str(p) for p in rec.pages),
                "contexts_count": len(rec.contexts),
                "response_time_s": rec.response_time,
                **rec.metrics,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    payload = {
        "summary": result["summary"],
        "details": result["details"],
        "records": [
            {
                "id": r.question_id,
                "question": r.question,
                "rag_answer": r.rag_answer,
                "pure_llm_answer": r.pure_answer,
                "contexts": r.contexts,
                "pages": r.pages,
                "response_time": r.response_time,
                "metrics": r.metrics,
            }
            for r in result["records"]
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return json_path, csv_path


def print_report(result: dict):
    """控制台打印对比报告。"""
    summary = result["summary"]
    print("\n" + "=" * 78)
    print(f"评估报告｜工单编号：{summary['work_order_no']}")
    print("=" * 78)
    print(f"评估问题数        : {summary['questions_total']}")
    print(f"检索命中率        : {summary['retrieval_hit_rate']}")
    print(f"平均响应时间      : {summary['response_time_avg']} s")
    print(f"P95 响应时间      : {summary['response_time_p95']} s")
    print(f"最大响应时间      : {summary['response_time_max']} s")
    print(f"满足 ≤3s 的比例   : {summary['within_3s_ratio']}")
    print(f"评估后端          : {summary['eval_backend']}")
    print("-" * 78)
    print("RAG 评估指标（RAGAS 体系）：")
    for key, value in (summary["ragas_scores"] or {}).items():
        print(f"  {key:<20}: {value}")
    print("-" * 78)
    print("RAG 回答 与 纯 LLM 回答 对比：")
    for rec in result["records"]:
        print(f"\n【问题 id={rec.question_id}】{rec.question}")
        print(f"  RAG 回答   ：{(rec.rag_answer or '')[:200].replace(chr(10), ' ')}")
        print(f"  纯 LLM 回答：{(rec.pure_answer or '')[:200].replace(chr(10), ' ')}")
        print(f"  检索页码   ：{rec.pages or '无'}｜耗时 {rec.response_time}s")
    print("=" * 78)


def build_parser():
    parser = argparse.ArgumentParser(description=f"{config.WORK_ORDER_NO} —— 评估脚本")
    parser.add_argument("--questions", default=config.QUESTIONS_FILE, help="问题列表 JSON 路径")
    parser.add_argument("--ground-truth", default="", help="参考答案 JSON 路径（可选）")
    parser.add_argument("--output", default=config.EVAL_OUTPUT_DIR, help="结果输出目录")
    parser.add_argument("--limit", type=int, default=None, help="只评估前 N 个问题")
    parser.add_argument("--no-ragas", action="store_true", help="不使用 RAGAS，仅使用内置评审")
    parser.add_argument("--skip-pure", action="store_true", help="跳过纯 LLM 对照回答")
    return parser


def main():
    args = build_parser().parse_args()
    questions = load_questions(args.questions)
    if not questions:
        logger.error("未读取到任何问题，请检查 %s", args.questions)
        return

    result = run_evaluation(
        questions,
        ground_truth=load_ground_truth(args.ground_truth),
        limit=args.limit,
        use_ragas=not args.no_ragas,
        skip_pure=args.skip_pure,
    )
    print_report(result)
    json_path, csv_path = export_results(result, str(args.output))
    print(f"结果已导出：\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
