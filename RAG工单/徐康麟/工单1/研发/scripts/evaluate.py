"""评估脚本：RAG vs 纯 LLM 对比 + RAGAS 指标。

用法::

    # 完整评估（RAG 系统 + 纯 LLM 对照组）
    python scripts/evaluate.py

    # 只跑 RAG 系统（不调用任何 LLM）
    python scripts/evaluate.py --no-llm

    # 额外运行 RAGAS 四项指标（需要 vLLM/SGLang 已启动）
    python 研发/scripts/evaluate.py --ragas

    # 只评估前 3 题（快速验证）
    python scripts/evaluate.py --limit 3

产出（均在 优化/评估结果/eval_results/）：
- rag_vs_llm.csv      逐题对比
- ragas_report.md     评估报告
- eval_records.json   完整明细（含每题的答案、引用、耗时）

说明：
“纯 LLM 回答”指**不给任何检索资料**，只让模型凭自身参数知识回答，
用于证明招股书中的具体数字必须依赖文档检索。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 允许以 `python 研发/scripts/evaluate.py` 方式直接运行：
# 本文件位于 <root>/研发/scripts/，需要把「研发」加入 sys.path 才能 import app.*
SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.evaluator import Evaluator  # noqa: E402
from app.core.qa_engine import QAEngine  # noqa: E402
from app.core.logging_conf import log_stage, logger, setup_logging  # noqa: E402
from app.models.schemas import Answer, EvalRecord, GoldenQA  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 系统评估")
    parser.add_argument("--golden", type=str, default=None, help="标准问答文件（默认 data/eval/golden_qa.jsonl）")
    parser.add_argument("--limit", type=int, default=0, help="只评估前 N 题，0 表示全部")
    parser.add_argument("--no-llm", action="store_true", help="跳过纯 LLM 对照组")
    parser.add_argument("--no-rag", action="store_true", help="跳过 RAG 组（只跑纯 LLM）")
    parser.add_argument("--ragas", action="store_true", help="运行 RAGAS 指标（需要裁判 LLM）")
    parser.add_argument(
        "--english",
        action="store_true",
        help="额外运行英文问答对照（工单追加要求：支持英文）",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出汇总")
    return parser.parse_args()


def load_golden(path: Path) -> list[GoldenQA]:
    """读取标准问答 JSONL。"""
    items: list[GoldenQA] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(GoldenQA(**json.loads(line)))
    return items


# 英文对照题（工单追加要求：支持英文问答）。
# 用同一批事实、英文提问，验证跨语言链路是否给出等价答案。
ENGLISH_CASES: list[tuple[int, str, str]] = [
    (543, "What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?", "RMB 55.20 million"),
    (531, "Who is the legal representative of the company?", "Cheng Jiaming"),
    (95, "Which technical standard did the company participate in formulating?", "military video command system technical standard"),
    (33, "What percentage of main business revenue came from the military sector during the reporting period?", "82.10%"),
    (207, "How much of the raised funds will be used to supplement working capital?", "RMB 150.00 million"),
]


def pure_llm_answer(question: str, settings) -> Answer:
    """纯 LLM 回答：**不提供任何检索资料**。

    这组结果用来与 RAG 对比，说明模型自身知识无法回答招股书细节问题。
    """
    started = time.perf_counter()
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
            timeout=settings.llm.read_timeout,
        )
        response = client.chat.completions.create(
            model=settings.llm.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个知识助手。请仅凭你已有的知识回答问题。"
                        "如果你不知道确切答案，只回复四个字：不清楚。"
                        "不要编造、不要猜测、不要给出大概范围。"
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=settings.llm.max_tokens,
        )
        text = (response.choices[0].message.content or "").strip()
        elapsed = (time.perf_counter() - started) * 1000
        is_unknown = text.strip("。.！!") in {settings.app.unknown_answer, "不知道", "无法回答"} or not text
        return Answer(
            answer=text or settings.app.unknown_answer,
            citations=[],
            is_unknown=is_unknown,
            unknown_reason="纯 LLM 回答（无资料）" if is_unknown else "",
            first_token_ms=round(elapsed, 2),
            total_ms=round(elapsed, 2),
            mode="llm",
        )
    except Exception as exc:
        logger.warning(
            "scripts.evaluate", "纯 LLM 回答失败（服务可能未启动）", error=f"{type(exc).__name__}: {exc}"
        )
        elapsed = (time.perf_counter() - started) * 1000
        return Answer(
            answer=f"[纯 LLM 调用失败: {type(exc).__name__}]",
            is_unknown=True,
            unknown_reason=f"LLM 不可用: {exc}",
            first_token_ms=round(elapsed, 2),
            total_ms=round(elapsed, 2),
            mode="llm",
        )


def main() -> int:
    args = parse_args()
    setup_logging()
    settings = get_settings()
    started = time.perf_counter()

    golden_path = Path(args.golden) if args.golden else settings.paths.data_eval / "golden_qa.jsonl"
    if not golden_path.exists():
        logger.error("scripts.evaluate", "标准问答文件不存在", path=str(golden_path))
        print(f"[错误] 标准问答文件不存在: {golden_path}", file=sys.stderr)
        return 2

    golden = load_golden(golden_path)
    if args.limit:
        golden = golden[: args.limit]
    log_stage("阶段4", "开始评估", questions=len(golden), golden=str(golden_path))

    evaluator = Evaluator()
    records = []
    ragas_samples: list[dict[str, object]] = []

    # ---------------- RAG 组 ----------------
    engine: QAEngine | None = None
    if not args.no_rag:
        engine = QAEngine()
        if not engine.stats()["index_ready"]:
            logger.error("scripts.evaluate", "索引未就绪，请先运行 scripts/build_index.py")
            print("[错误] 索引未就绪，请先执行: python scripts/build_index.py", file=sys.stderr)
            return 3

        # 先预热，把模型加载/分词器构建等一次性开销排除在计时之外，
        # 否则第一题的“首字延迟”会被冷启动污染，得到失真的慢值。
        warmup = engine.warmup()
        log_stage("阶段4", "预热完成", **warmup)

        llm_ok = engine.generator.check_llm_available()
        mode = "rag" if llm_ok else "extractive"
        log_stage(
            "阶段4",
            "RAG 组开始",
            mode=mode,
            note="LLM 可用" if llm_ok else "LLM 不可用，使用抽取式回答（无需 LLM）",
        )
        for item in golden:
            answer = engine.ask(item.question, conversation_id=None, allow_llm=llm_ok, save=False)
            record = evaluator.evaluate_answer(item, answer, mode)
            records.append(record)
            ragas_samples.append(
                {
                    "user_input": item.question,
                    "response": answer.answer,
                    "retrieved_contexts": [ctx.chunk.content for ctx in answer.retrieved],
                    "reference": item.answer,
                }
            )
            print(
                f"  [{'✓' if record.is_correct else '✗'}] 题{item.id} "
                f"首字 {record.first_token_ms:.0f}ms 引用 {record.citation_pages} :: {answer.answer[:50]}"
            )

    # ---------------- 纯 LLM 对照组 ----------------
    if not args.no_llm:
        if not engine:
            engine = QAEngine()
        if engine.generator.check_llm_available():
            log_stage("阶段4", "纯 LLM 对照组开始（不提供任何资料）")
            for item in golden:
                answer = pure_llm_answer(item.question, settings)
                record = evaluator.evaluate_answer(item, answer, "llm")
                records.append(record)
                print(f"  [{'✓' if record.is_correct else '✗'}] 题{item.id} LLM :: {answer.answer[:50]}")
        else:
            log_stage("阶段4", "跳过纯 LLM 对照组（LLM 服务不可用）")

    if not records:
        print("[错误] 没有任何评估记录产生", file=sys.stderr)
        return 4

    # ---------------- 英文对照（可选，用 --english 开启）----------------
    if args.english and engine is not None:
        log_stage("阶段4", "英文问答对照开始", cases=len(ENGLISH_CASES))
        for question_id, question, expectation in ENGLISH_CASES:
            answer = engine.ask(question, allow_llm=False, save=False)
            hit = expectation.lower() in answer.answer.lower()
            print(f"  [{'✓' if hit else '✗'}] EN q{question_id} :: {answer.answer[:64]}")
            records.append(
                EvalRecord(
                    question_id=question_id,
                    question=question,
                    mode="rag_en",
                    answer=answer.answer,
                    golden=expectation,
                    is_correct=hit,
                    is_unknown=answer.is_unknown,
                    citation_pages=[c.page for c in answer.citations],
                    citation_valid=bool(answer.citations) and all(c.chunk_id for c in answer.citations),
                    citation_valid_count=sum(1 for c in answer.citations if c.chunk_id),
                    first_token_ms=answer.first_token_ms,
                    total_ms=answer.total_ms,
                )
            )

    # ---------------- 入库 ----------------
    try:
        evaluator.store.clear_eval_records()
        for record in records:
            evaluator.store.add_eval_record(record)
    except Exception as exc:
        logger.warning("scripts.evaluate", "评估结果入库失败（不影响报告生成）", error=str(exc))

    # ---------------- RAGAS ----------------
    ragas_result = None
    if args.ragas:
        log_stage("阶段4", "运行 RAGAS 指标")
        ragas_result = evaluator.run_ragas(ragas_samples)

    # ---------------- 报告 ----------------
    summaries = evaluator.summarize(records)
    extra = {
        "文档": settings.paths.default_pdf.name,
        "RAG 组模式": "LLM 生成" if summaries.get("rag") else "抽取式（无 LLM）",
        "嵌入模型": engine.embedder.name if engine else "—",
        "向量库后端": engine.vector_store.name if engine else "—",
        "分块数": engine.stats()["chunks"] if engine else 0,
        "LLM 服务": settings.llm.base_url,
        "LLM 模型": settings.llm.model,
        "生成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    csv_path = evaluator.write_comparison_csv(records)
    report_path = evaluator.write_report(records, ragas_result=ragas_result, extra=extra)
    json_path = evaluator.write_json(records)

    summary_payload = {
        "summaries": {mode: summary.as_dict() for mode, summary in summaries.items()},
        "artifacts": {"csv": str(csv_path), "report": str(report_path), "json": str(json_path)},
        "ragas": ragas_result,
        "elapsed_s": round(time.perf_counter() - started, 2),
    }

    log_stage("阶段4", "评估结束", elapsed_s=summary_payload["elapsed_s"])
    if args.json:
        print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    else:
        print("\n===== 评估汇总 =====")
        for mode, summary in summaries.items():
            print(
                f"[{mode}] 题数={summary.count} 准确率={summary.accuracy:.1%} "
                f"引用正确率={summary.citation_accuracy:.1%} "
                f"首字均值={summary.first_token_avg_ms}ms 最大={summary.first_token_max_ms}ms"
            )
        print(f"\n对比 CSV : {csv_path}")
        print(f"评估报告 : {report_path}")
        print(f"完整明细 : {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
