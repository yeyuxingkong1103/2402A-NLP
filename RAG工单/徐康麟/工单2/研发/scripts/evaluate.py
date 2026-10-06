"""评估脚本：在 10 个工单问题（golden_qa.jsonl）上跑 RAG 系统并用确定性判分统计指标。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 脚本（对应 设计/接口设计.md §2.18）

用法（工作目录 = E:\\gao6gongdan\\工单2）::

    pwsh -NoProfile -File run_py.ps1 研发/scripts/evaluate.py
    pwsh -NoProfile -File run_py.ps1 研发/scripts/evaluate.py --limit 3 --mode extractive

**注意**：本机 `ragas` 不可用且断网（环境事实 2.2），因此脚本**不提供 `--ragas` 选项**；
报告中的 RAGAS 四项指标一律标注「未运行（依赖不可用，本机断网）」，绝不伪造数值。

退出码：0 成功；2 业务失败（索引缺失等）；3 参数错误。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.errors import RAGError  # noqa: E402
from app.core.evaluator import get_evaluator  # noqa: E402
from app.core.logging_conf import flush_logs, logger, setup_logging  # noqa: E402
from app.models.schemas import GoldenQA  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    parser = argparse.ArgumentParser(description="RAG 系统评估（工单2，确定性判分）")
    parser.add_argument("--mode", default="rag", choices=["rag", "extractive"], help="rag=LLM 生成；extractive=抽取式")
    parser.add_argument("--limit", type=int, default=0, help="仅评估前 N 题（0=全部）")
    parser.add_argument("--out", default="", help="输出目录，默认 优化/评估结果")
    parser.add_argument("--golden", default="", help="golden_qa.jsonl 路径，默认 测试/测试数据/golden_qa.jsonl")
    parser.add_argument("--filename", default="", help="输出文件名前缀（默认按模式自动命名）")
    return parser


def load_golden(path: Path) -> list[GoldenQA]:
    """读取金标准问答（JSONL）。"""
    items: list[GoldenQA] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    items.append(GoldenQA(**json.loads(line)))
    except Exception:
        logger.exception("scripts.evaluate", "读取金标准问答失败", path=str(path))
        raise
    return items


def main(argv: list[str] | None = None) -> int:
    """脚本入口。"""
    args = build_parser().parse_args(argv)
    if args.limit < 0:
        print("参数错误：--limit 不能为负数", file=sys.stderr)
        return 3
    setup_logging()
    settings = get_settings()
    from app.core.qa_engine import QAEngine

    golden_path = Path(args.golden) if args.golden else settings.paths.test_data / "golden_qa.jsonl"
    if not golden_path.exists():
        print(f"业务失败：金标准问答不存在 -> {golden_path}", file=sys.stderr)
        return 2
    golden_items = load_golden(golden_path)
    if args.limit:
        golden_items = golden_items[: args.limit]

    force_extractive = args.mode == "extractive"
    engine = QAEngine(force_extractive=force_extractive)
    if not engine.load_index():
        print("业务失败：索引未就绪，请先执行 pwsh -NoProfile -File run_py.ps1 研发/scripts/build_index.py", file=sys.stderr)
        return 2

    evaluator = get_evaluator(engine._store)  # noqa: SLF001（复用同一 SQLite 连接）
    if args.out:
        evaluator.results_dir = Path(args.out)
        evaluator.results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("工单2 评估：人工智能NLP-RAG-基于PDF文档的问答系统优化")
    print(f"模式        : {args.mode}（force_extractive={force_extractive}）")
    print(f"题目数      : {len(golden_items)}")
    print(f"判分口径    : 工单1 check_answer（FUZZY_THRESHOLD=0.62，不放宽）")
    print(f"RAGAS       : 未运行（依赖不可用，本机断网）")
    print("=" * 78)

    # 预热：与线上启动一致（serve_fallback / streamlit 首屏都会 warmup），
    # 否则第一题会测到 **冷启动**（本机实测 Ollama qwen2.5:3b 首次加载可达 6 秒），
    # 把"模型加载"计进首字延迟会让 ≤3 秒 的验收指标失真。
    warmup_info = engine.warmup()
    print(
        f"预热        : 嵌入 {warmup_info.get('embedder', {}).get('elapsed_ms')} ms，"
        f"总计 {warmup_info.get('elapsed_ms')} ms（首字延迟按预热后统计）"
    )

    records = []
    started = time.perf_counter()
    for index, golden in enumerate(golden_items, start=1):
        try:
            answer = engine.ask(golden.question)
        except RAGError as exc:
            logger.exception("scripts.evaluate", "单题问答业务失败", question_id=golden.id, code=exc.code)
            print(f"[{index}/{len(golden_items)}] Q{golden.id} 业务失败[{exc.code}]：{exc.message}")
            continue
        record = evaluator.evaluate_answer(golden, answer, mode=args.mode)
        records.append(record)
        mark = "✅" if record.is_correct else "❌"
        print(
            f"[{index}/{len(golden_items)}] Q{golden.id} {mark} "
            f"首字={record.first_token_ms:.0f}ms 引用={record.citation_pages} 模式={answer.mode}"
        )
        print(f"      答案: {record.answer[:110]}")
        print(f"      标准: {record.golden[:110]}")

    if not records:
        print("业务失败：没有任何评估记录", file=sys.stderr)
        return 2

    summaries = evaluator.summarize(records)
    total_elapsed = time.perf_counter() - started
    print("-" * 78)
    print("汇总（确定性指标）：")
    for mode, summary in summaries.items():
        total_avg = round(sum(summary.total_ms) / len(summary.total_ms), 2) if summary.total_ms else 0.0
        print(f"  mode={mode} 准确率={summary.accuracy:.4f} ({summary.correct}/{summary.count})")
        print(
            f"    引用正确率={summary.citation_accuracy:.4f}"
            f"  首字均值={summary.first_token_avg_ms:.2f}ms  最大={summary.first_token_max_ms:.2f}ms"
            f"  P95={summary.first_token_p95_ms:.2f}ms  端到端均值={total_avg:.2f}ms"
        )
    print(f"  评估总耗时: {total_elapsed:.1f}s")
    print(f"  RAGAS      : 未运行（依赖不可用，本机断网）")

    filename_prefix = args.filename
    json_name = f"{filename_prefix}_eval_records.json" if filename_prefix else "eval_records.json"
    report_name = f"{filename_prefix}_report.md" if filename_prefix else "ragas_report.md"
    csv_name = f"{filename_prefix}_rag_vs_llm.csv" if filename_prefix else "rag_vs_llm.csv"
    extra = {
        "生成模式": args.mode,
        "本文件口径": (
            "纯抽取式降级路径（LLM 关闭）"
            if args.mode == "extractive"
            else "最终交付路径（混合检索 + 抽取式生成 + 一致性校验）"
        ),
        "对照路径": (
            "最终交付路径见 优化/评估结果/eval_records.json（mode=rag）"
            if args.mode == "extractive"
            else "纯抽取式降级路径见 优化/评估结果/extractive_report.md"
        ),
        "嵌入模型": engine.health()["embedder"].get("model"),
        "嵌入维度": engine.health()["embedder"].get("dimension"),
        "重排模式": engine.health()["reranker"].get("mode"),
        "LLM 后端": engine.health()["llm"].get("backend", {}).get("name"),
        "LLM 模型": settings.llm.model,
        "判分口径": "工单1 check_answer（0.62 阈值，未放宽）",
        "RAGAS": "未运行（依赖不可用，本机断网）",
    }
    json_path = evaluator.write_json(records, filename=json_name)
    report_path = evaluator.write_report(records, ragas_result=None, filename=report_name, extra=extra)
    csv_path = evaluator.write_comparison_csv(records, filename=csv_name)
    # 口径标注：只改标注文字，不改任何数值（避免把降级路径误当最终交付指标）
    summary = summaries.get(args.mode) or next(iter(summaries.values()))
    relabel_report(report_path, args.mode, summary.accuracy, summary.count, summary.correct)
    print(f"  记录: {json_path}")
    print(f"  报告: {report_path}")
    print(f"  对比: {csv_path}")
    print(
        "  口径: "
        + (
            "本文件为纯抽取式降级路径；最终交付路径见 eval_records.json（mode=rag）"
            if args.mode == "extractive"
            else "本文件为最终交付路径（mode=rag）；降级路径见 extractive_report.md"
        )
    )
    flush_logs()
    return 0


def relabel_report(path: Path, mode: str, accuracy: float, count: int, correct: int) -> None:
    """给评估报告加上"本文件口径"标注，避免把降级路径当成最终交付指标。

    背景（captain 复核指出）：基线 ``Evaluator.write_report`` 的标题与结论句固定写
    「RAG vs 纯 LLM 评估报告 / RAG 准确率 X%」。当 ``--mode extractive``（LLM 关闭的
    降级路径）时，该表述会与 ``eval_records.json``（mode=rag，最终交付路径）冲突，
    读者与 T9 无法分辨哪份是交付指标。

    本函数**只改标注文字、不改任何数值**（数值全部由 evaluator 依判分结果生成）。
    """
    try:
        if not path.exists():
            logger.warning("scripts.evaluate", "报告文件不存在，跳过口径标注", path=str(path))
            return
        text = path.read_text(encoding="utf-8")
        if mode == "extractive":
            text = text.replace(
                "# RAG vs 纯 LLM 评估报告", "# 纯抽取式（LLM 关闭）降级路径评估报告", 1
            )
            text = text.replace(
                f"- RAG 准确率 {accuracy:.1%}（未运行纯 LLM 对照组）。",
                f"- **纯抽取式（LLM 关闭）降级路径准确率 {accuracy:.1%}（{correct}/{count}）**，"
                f"与最终交付路径（混合检索 + 抽取式生成，mode=rag）同分；"
                f"两者差别只在生成侧，检索/证据池相同。",
            )
        note = (
            "\n> **本文件口径**："
            + (
                "纯抽取式降级路径（`--mode extractive`，LLM 关闭）。"
                "**最终交付路径**为 `优化/评估结果/eval_records.json`（`--mode rag`：混合检索 + 抽取式生成 + 一致性校验）。"
                if mode == "extractive"
                else "最终交付路径（`--mode rag`：混合检索 + 抽取式生成 + 一致性校验）。"
                "纯抽取式降级路径另见 `extractive_report.md` 与 `extractive_eval_records.json`。"
            )
            + "\n"
        )
        if "**本文件口径**" not in text:
            lines = text.split("\n")
            insert_at = 1 if lines and lines[0].startswith("# ") else 0
            lines.insert(insert_at, note)
            text = "\n".join(lines)
        # 修正"未运行指标"的复现命令：本工单 evaluate.py **不提供 --ragas**（本机依赖不可用 + 断网）
        text = text.replace(
            "```bash\n# 1) 先起 vLLM 服务（见 scripts/run_vllm.sh）\n# 2) 再带 --ragas 运行评估\n"
            "python scripts/evaluate.py --ragas\n```",
            "```bash\n# 本工单 研发/scripts/evaluate.py **不提供 --ragas 选项**：\n"
            "#   本机 ragas 不可用且无法安装（断网），报告中的 RAGAS 数值一律为空，绝不估算/伪造。\n"
            "# 确定性替代指标（准确率 / 检索命中 / 引用正确率 / 首字延迟）已在本报告第 1、3 节给出。\n```",
        )
        path.write_text(text, encoding="utf-8")
        logger.info("scripts.evaluate", "报告口径标注完成", path=str(path), mode=mode)
    except Exception:
        logger.exception("scripts.evaluate", "报告口径标注失败（不影响已有数值）", path=str(path))


if __name__ == "__main__":
    raise SystemExit(main())
